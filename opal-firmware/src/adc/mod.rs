//! The EMG acquisition front end: one TI ADS1298 8-channel ADC.
//!
//! The model still expects 16 channels; [`preprocess`] zero-pads the upper eight
//! until it is retrained (see the TODO there).
//!
//! Layout. [`registers`] is the typed register map and [`ads1298`] the register-level
//! driver; [`decode`]/[`status`]/[`convert`] are the hardware-free frame maths,
//! [`preprocess`] turns ADC codes into the model's int8 input, and [`acquisition`] runs
//! the sampling thread. [`bring_up`] does the wiring and hands back a configured,
//! streaming pair.
//!
//! Threading. Sampling cannot live on the main loop: at 2 kSPS a frame arrives every
//! 500 µs, while the main loop runs one 244 ms inference window per iteration and
//! feeds the task watchdog. [`acquisition::start`] moves the driver onto its own
//! thread, so everything here owns its peripherals at `'static` instead of borrowing
//! them.

pub(crate) mod acquisition;
pub(crate) mod ads1298;
mod channel;
mod convert;
mod decode;
mod preprocess;
mod registers;
mod spi_commands;
mod status;

pub(crate) use channel::Channel;

use anyhow::{Context, Result};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, PinDriver, Pull};
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::info;
use std::sync::Arc;

use ads1298::{Ads1298Device, Ads1298FrontEnd};

/// The ADS1298 reports this in its ID register. Bring-up checks it so a dead bus fails
/// loudly at boot instead of producing plausible-looking zeroes forever.
const EXPECTED_DEVICE_ID: u8 = 0x92;

/// Every pin the front end needs, type-erased so the wiring lives in one place.
pub(crate) struct AdcPins {
    pub(crate) clock: AnyOutputPin<'static>,
    pub(crate) data_in: AnyOutputPin<'static>,
    pub(crate) data_out: AnyInputPin<'static>,
    pub(crate) chip_select: AnyOutputPin<'static>,
    pub(crate) data_ready: AnyInputPin<'static>,
    pub(crate) reset: AnyOutputPin<'static>,
    pub(crate) power_down: AnyOutputPin<'static>,
    pub(crate) start: AnyOutputPin<'static>,
}

/// Brings the chip up and leaves it streaming in RDATAC mode.
///
/// `baud_rate_hz` is the SPI clock. One 27-byte frame has to clear well inside the
/// 500 µs sample period at 2 kSPS; the bench validated 2 MHz (burst framing makes the
/// register path legal at any rate the sweep passed).
/// `test_signal_channel` is `Some(channel)` to drive the ADS1298's internal square
/// wave into that channel instead of the electrodes. See
/// [`ads1298::Ads1298Device::enable_test_signal`].
pub(crate) fn bring_up<SPI: SpiAnyPins + 'static>(
    spi: SPI,
    pins: AdcPins,
    baud_rate_hz: u32,
    test_signal_channel: Option<Channel>,
) -> Result<Ads1298FrontEnd> {
    // One bus driver shared by both chips through an Arc, so the devices are 'static
    // and can move onto the acquisition thread.
    let bus = Arc::new(
        SpiDriver::new(
            spi,
            pins.clock,
            pins.data_in,
            Some(pins.data_out),
            &SpiDriverConfig::new(),
        )
        .context("SPI bus init")?,
    );

    // The ADS1298 samples DIN on the falling edge and shifts DOUT on the rising edge:
    // SPI mode 1 (CPOL=0, CPHA=1).
    let spi_config = SpiConfig::new()
        .baudrate(baud_rate_hz.Hz())
        .data_mode(MODE_1);

    // No hardware CS: the driver toggles CS as a GPIO, held low across each whole
    // command — the mechanism the bring-up harness validated.
    let device = Ads1298Device::new(
        SpiDeviceDriver::new(bus, Option::<AnyOutputPin>::None, &spi_config)
            .context("SPI device")?,
        PinDriver::output(pins.chip_select)?,
        // DRDY is actively driven by the ADS1298, so no internal pull is needed.
        PinDriver::input(pins.data_ready, Pull::Floating)?,
        PinDriver::output(pins.reset)?,
        PinDriver::output(pins.power_down)?,
    )
    .context("ADC pin init")?;

    let mut front_end = Ads1298FrontEnd::new(device, PinDriver::output(pins.start)?);

    // Roughly 2.5 s of mandated settling. It must stay ahead of the task watchdog
    // registration in `main`.
    info!("ADS1298: powering up");
    front_end
        .device
        .power_up(EXPECTED_DEVICE_ID)
        .context("ADS1298 power-up")?;

    // Still in SDATAC, so registers are readable. Once RDATAC starts below there is no
    // way to check them again without tearing the stream down.
    front_end.device.log_configuration_readback();

    if let Some(channel) = test_signal_channel {
        // Registers can only be written outside RDATAC, which power_up leaves us in.
        front_end.device.enable_test_signal(channel)?;
        info!(
            "ADS1298: internal test signal on channel {channel}; electrodes are NOT \
             being read"
        );
    }

    front_end.device.read_data_continuous()?;
    front_end.start_conversion()?;
    info!("ADS1298: streaming at {baud_rate_hz} Hz SPI");

    Ok(front_end)
}
