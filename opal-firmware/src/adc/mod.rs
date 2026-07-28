//! The EMG acquisition front end: two TI ADS1298 8-channel ADCs on a shared SPI bus,
//! giving the 16 channels the model expects.
//!
//! Layout. [`ads1298`] is the register-level driver, [`decode`]/[`loff`]/[`convert`]
//! are the hardware-free frame maths, [`preprocess`] turns ADC codes into the model's
//! int8 input, and [`acquisition`] runs the sampling thread. [`bring_up`] does the
//! wiring and hands back a configured, streaming pair.
//!
//! Threading. Sampling cannot live on the main loop: at 2 kSPS a frame arrives every
//! 500 µs, while the main loop runs one 244 ms inference window per iteration and
//! feeds the task watchdog. [`acquisition::start`] moves the driver onto its own
//! thread, so everything here owns its peripherals at `'static` instead of borrowing
//! them.

pub(crate) mod acquisition;
pub(crate) mod ads1298;
mod convert;
mod decode;
mod loff;
mod preprocess;
mod registers;
mod spi_commands;

pub(crate) use decode::CHANNELS_PER_DEVICE;

use anyhow::{Context, Result};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, PinDriver, Pull};
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::info;
use std::sync::Arc;

use ads1298::{Ads1298Device, Ads1298Pair, ChipRole};

/// The ADS1298 reports this in its ID register. Bring-up checks it so a dead bus fails
/// loudly at boot instead of producing plausible-looking zeroes forever.
const EXPECTED_DEVICE_ID: u8 = 0x92;

/// Every pin the front end needs, type-erased so the wiring lives in one place.
///
/// SCLK, DIN and DOUT are shared by both chips. CS, DRDY, RESET and PWDN are per-chip.
/// START is tied together, so both chips convert on the same edge.
pub(crate) struct AdcPins {
    pub(crate) clock: AnyOutputPin<'static>,
    pub(crate) data_in: AnyOutputPin<'static>,
    pub(crate) data_out: AnyInputPin<'static>,
    pub(crate) chip_select_a: AnyOutputPin<'static>,
    pub(crate) chip_select_b: AnyOutputPin<'static>,
    pub(crate) data_ready_a: AnyInputPin<'static>,
    pub(crate) data_ready_b: AnyInputPin<'static>,
    pub(crate) reset_a: AnyOutputPin<'static>,
    pub(crate) reset_b: AnyOutputPin<'static>,
    pub(crate) power_down_a: AnyOutputPin<'static>,
    pub(crate) power_down_b: AnyOutputPin<'static>,
    pub(crate) start: AnyOutputPin<'static>,
}

/// Brings both chips up and leaves them streaming in RDATAC mode.
///
/// `baud_rate_hz` is the SPI clock. It has to be fast enough that two 27-byte frames
/// clear inside one sample period: at 2 kSPS that period is 500 µs, and 54 bytes is
/// 432 bits, so 1 MHz leaves under 70 µs for CS toggling and driver overhead. Raise it
/// until the drop counter in [`acquisition`] stays at zero.
/// `test_signal_channel` is `Some(0..=7)` to drive the ADS1298's internal square wave
/// into that channel on both chips instead of the electrodes. See
/// [`ads1298::Ads1298Device::enable_test_signal`].
pub(crate) fn bring_up<SPI: SpiAnyPins + 'static>(
    spi: SPI,
    pins: AdcPins,
    baud_rate_hz: u32,
    test_signal_channel: Option<usize>,
) -> Result<Ads1298Pair> {
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

    let device_a = Ads1298Device::new(
        SpiDeviceDriver::new(bus.clone(), Some(pins.chip_select_a), &spi_config)
            .context("SPI device A")?,
        // DRDY is actively driven by the ADS1298, so no internal pull is needed.
        PinDriver::input(pins.data_ready_a, Pull::Floating)?,
        PinDriver::output(pins.reset_a)?,
        PinDriver::output(pins.power_down_a)?,
    );
    let device_b = Ads1298Device::new(
        SpiDeviceDriver::new(bus, Some(pins.chip_select_b), &spi_config).context("SPI device B")?,
        PinDriver::input(pins.data_ready_b, Pull::Floating)?,
        PinDriver::output(pins.reset_b)?,
        PinDriver::output(pins.power_down_b)?,
    );

    let mut pair = Ads1298Pair::new(device_a, device_b, PinDriver::output(pins.start)?);

    // Roughly 2.2 s of mandated settling per chip, done in sequence, so this blocks for
    // about 4.4 s. It must stay ahead of the task watchdog registration in `main`.
    info!("ADS1298: powering up chip A");
    pair.adc1
        .power_up(ChipRole::A, EXPECTED_DEVICE_ID)
        .context("chip A power-up")?;
    info!("ADS1298: powering up chip B");
    pair.adc2
        .power_up(ChipRole::B, EXPECTED_DEVICE_ID)
        .context("chip B power-up")?;

    if let Some(channel) = test_signal_channel {
        // Registers can only be written outside RDATAC, which power_up leaves us in.
        pair.adc1.enable_test_signal(channel)?;
        pair.adc2.enable_test_signal(channel)?;
        info!(
            "ADS1298: internal test signal on channel {channel}; electrodes are NOT \
             being read"
        );
    }

    pair.adc1.read_data_continuous()?;
    pair.adc2.read_data_continuous()?;
    pair.start_conversion()?;
    info!("ADS1298: both chips streaming at {baud_rate_hz} Hz SPI");

    Ok(pair)
}
