//! The EMG acquisition front end: two TI ADS1298 8-channel ADCs.
//!
//! Each chip has its own complete SPI bus (SPI2 and SPI3 hosts), its own PWDN,
//! chip select, START, RESET, and DRDY — the chips share nothing but ground.
//! Separate buses are not a luxury: SBAS459K §9.4.1.2 says a rising SCLK edge
//! pulls DRDY high *regardless of CS*, so a shared SCLK would toggle the idle
//! chip's DRDY on every read of its peer, and the datasheet's own prescription
//! for multi-device buses is to gate SCLK per chip. Independent buses also let
//! the stochastic conversion deaths the bring-up campaign characterised be
//! detected and warm-recovered per chip without restarting the healthy one.
//!
//! Both chips self-clock from their internal 2.048 MHz oscillators (CLKSEL
//! strapped to 3V3). They are therefore *not* phase-locked; [`preprocess`] and
//! [`acquisition`] document what that does to the sixteen-channel pairing.
//!
//! Layout. [`registers`] is the typed register map and [`ads1298`] the register-level
//! driver; [`decode`]/[`status`]/[`convert`]/[`conditioning`] are the hardware-free
//! frame and signal maths, and [`preprocess`] turns ADC codes into the model's int8
//! input. [`bring_up`] does the wiring and hands back a configured, streaming pair.
//!
//! Threading. Sampling cannot live on the main loop: at ~2 kSPS a frame arrives from
//! each chip every ~487 µs, while the main loop runs one inference window per
//! iteration and feeds the task watchdog. [`acquisition::start`] spawns one
//! [`chip_pipeline`] thread per chip — each owning its chip's SPI bus, DRDY
//! interrupt, and health, so the two chips' reads overlap instead of serialising —
//! plus a combiner thread that places both streams onto one time grid
//! ([`emg_runtime::alignment`]) and builds the model and wire windows. Everything
//! here owns its peripherals at `'static` instead of borrowing them.

pub(crate) mod acquisition;
pub(crate) mod ads1298;
mod channel;
mod chip_pipeline;
mod conditioning;
mod convert;
mod decode;
mod preprocess;
mod registers;
mod spi_commands;
mod status;

pub(crate) use channel::Channel;
use channel::DEVICE_COUNT;
pub(crate) use convert::MICROVOLTS_PER_WIRE_COUNT;

use anyhow::{Context, Result};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, Output, PinDriver, Pull};
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::{info, warn};
use std::sync::Arc;

use ads1298::{Ads1298Device, Ads1298FrontEnd};

/// The ADS1298 reports this in its ID register. Bring-up checks it so a dead bus fails
/// loudly at boot instead of producing plausible-looking zeroes forever.
const EXPECTED_DEVICE_ID: u8 = 0x92;

/// The conversion clock each chip generates for itself: the internal oscillator's
/// nominal 2.048 MHz (CLKSEL strapped to 3V3 on both boards). Accuracy is ±0.5%
/// at 25 °C and ±2% over 0-70 °C (SBAS459K §7.5), and the two chips' oscillators
/// are independent — nothing enforces a common rate or phase.
pub(crate) const INTERNAL_OSCILLATOR_HZ: u32 = 2_048_000;

/// One chip's complete wiring: a full private SPI bus plus every control line.
/// Nothing here is shared with the other chip.
pub(crate) struct AdcChipWiring {
    pub(crate) sclk: AnyOutputPin<'static>,
    pub(crate) data_in: AnyOutputPin<'static>,
    pub(crate) data_out: AnyInputPin<'static>,
    /// This chip's own PWDN. It only participates in cold-boot sequencing — warm
    /// recovery is RESET-only — but per-chip lines keep the two boards fully
    /// independent.
    pub(crate) power_down: AnyOutputPin<'static>,
    pub(crate) chip_select: AnyOutputPin<'static>,
    pub(crate) data_ready: AnyInputPin<'static>,
    pub(crate) reset: AnyOutputPin<'static>,
    pub(crate) start: AnyOutputPin<'static>,
}

/// The configured, streaming pair, plus both PWDN lines kept alive (and high)
/// for as long as the front end exists.
pub(crate) struct FrontEnds {
    pub(super) chips: [Ads1298FrontEnd; DEVICE_COUNT],
    pub(super) _power_down: [PinDriver<'static, Output>; DEVICE_COUNT],
}

/// Builds one chip's front end on its own private SPI host.
///
/// No hardware CS: the driver toggles CS as a GPIO, held low across each whole
/// command — the mechanism the bring-up harness validated. The `Arc` keeps the
/// device `'static` so it can move onto the acquisition thread.
fn build_front_end<SPI: SpiAnyPins + 'static>(
    spi: SPI,
    wiring: AdcChipWiring,
    spi_config: &SpiConfig,
) -> Result<(Ads1298FrontEnd, PinDriver<'static, Output>)> {
    let bus = Arc::new(
        SpiDriver::new(
            spi,
            wiring.sclk,
            wiring.data_in,
            Some(wiring.data_out),
            &SpiDriverConfig::new(),
        )
        .context("SPI bus init")?,
    );
    // Held low from construction: the chip stays in power-down until the shared
    // sequencing in `bring_up` raises both boards together.
    let mut power_down = PinDriver::output(wiring.power_down)?;
    power_down.set_low()?;
    let device = Ads1298Device::new(
        SpiDeviceDriver::new(bus, Option::<AnyOutputPin>::None, spi_config)
            .context("SPI device")?,
        PinDriver::output(wiring.chip_select)?,
        // DRDY is actively driven by the ADS1298, so no internal pull is needed.
        PinDriver::input(wiring.data_ready, Pull::Floating)?,
        PinDriver::output(wiring.reset)?,
    )
    .context("ADC pin init")?;
    Ok((
        Ads1298FrontEnd::new(device, PinDriver::output(wiring.start)?),
        power_down,
    ))
}

/// Brings both chips up and leaves them streaming in RDATAC mode.
///
/// The chips self-clock (CLKSEL at 3V3), so no external clock has to be running
/// first: the post-RESET lockout counts cycles of each chip's own oscillator.
///
/// `baud_rate_hz` is the SPI clock. One 27-byte frame per chip has to clear well
/// inside the ~500 µs sample period; the bench validated 2 MHz (burst framing makes
/// the register path legal at any rate the sweep passed).
/// `test_signal_channel` is `Some(channel)` to drive each chip's internal square
/// wave into that channel instead of the electrodes. See
/// [`ads1298::Ads1298Device::enable_test_signal`].
pub(crate) fn bring_up<SpiA: SpiAnyPins + 'static, SpiB: SpiAnyPins + 'static>(
    spi_a: SpiA,
    spi_b: SpiB,
    wiring: [AdcChipWiring; DEVICE_COUNT],
    baud_rate_hz: u32,
    test_signal_channel: Option<Channel>,
) -> Result<FrontEnds> {
    // The ADS1298 samples DIN on the falling edge and shifts DOUT on the rising edge:
    // SPI mode 1 (CPOL=0, CPHA=1).
    //
    // Interrupt-driven transactions, not the default polling: a polled transfer
    // busy-spins the CPU for its whole duration, and two chips at 2000 SPS spin
    // ~4000 times a second — enough, with the rest of the acquisition path, to
    // starve the idle task and trip the task watchdog. Blocking on the transfer
    // instead costs some per-transaction latency, which the ~500 µs sample period
    // absorbs easily.
    let spi_config = SpiConfig::new()
        .baudrate(baud_rate_hz.Hz())
        .data_mode(MODE_1)
        .polling(false);

    let [wiring_a, wiring_b] = wiring;
    let (front_end_a, power_down_a) = build_front_end(spi_a, wiring_a, &spi_config)?;
    let (front_end_b, power_down_b) = build_front_end(spi_b, wiring_b, &spi_config)?;
    let mut chips = [front_end_a, front_end_b];
    let mut power_down = [power_down_a, power_down_b];

    // Shared power sequencing: minimum PWDN assertion, release, then the supply
    // settling both chips wait out together. Roughly 2.5 s all told, which must stay
    // ahead of the task watchdog registration in `main`.
    info!("ADS1298 pair: powering up");
    FreeRtos::delay_ms(5);
    for pin in &mut power_down {
        pin.set_high()?;
    }
    for chip in &mut chips {
        chip.device.release_reset()?;
    }
    FreeRtos::delay_ms(2000);

    // Every chip is probed and reported before any mismatch is fatal. With two
    // boards on independent buses the comparison between them is the diagnosis:
    // both chips failing the same way points at something genuinely common (power,
    // ground, firmware), one chip disagreeing points at that board's own wiring.
    let mut identities = [0u8; DEVICE_COUNT];
    for (index, chip) in chips.iter_mut().enumerate() {
        let id = chip
            .device
            .probe_identity()
            .with_context(|| format!("ADS1298 chip {index} identity probe"))?;
        identities[index] = id;
        if id == EXPECTED_DEVICE_ID {
            info!("ADS1298 chip {index}: ID {id:#04x} as expected");
        } else {
            warn!(
                "ADS1298 chip {index}: ID {id:#04x}, expected {EXPECTED_DEVICE_ID:#04x} -- {}",
                match id {
                    0x00 | 0xFF =>
                        "bus reads all-zero or all-one, so this is wiring, \
                                    power, or chip select rather than a wrong part",
                    _ =>
                        "the bus responds but the byte is wrong: suspect SPI mode, \
                          a shifted or reversed ribbon, or a chip whose CLKSEL strap \
                          is not actually at 3V3 (an unclocked chip cannot decode \
                          commands)",
                }
            );
        }
    }
    if identities.iter().any(|&id| id != EXPECTED_DEVICE_ID) {
        anyhow::bail!(
            "ADS1298 ID mismatch: chip 0 read {:#04x}, chip 1 read {:#04x}, both should \
             be {EXPECTED_DEVICE_ID:#04x}",
            identities[0],
            identities[1],
        );
    }

    for (index, chip) in chips.iter_mut().enumerate() {
        chip.device
            .initialize()
            .with_context(|| format!("ADS1298 chip {index} initialise"))?;
    }

    // Internal reference settling: the datasheet's 150 ms start-up time with margin,
    // after `configure` powers each reference buffer and before START. One wait
    // covers both chips — their references started within milliseconds of each other.
    FreeRtos::delay_ms(300);

    for (index, chip) in chips.iter_mut().enumerate() {
        // Still in SDATAC, so registers are readable. Once RDATAC starts below there
        // is no way to check them again without tearing the stream down.
        info!("ADS1298 chip {index} configuration readback:");
        chip.device.log_configuration_readback();

        if let Some(channel) = test_signal_channel {
            // Registers can only be written outside RDATAC.
            chip.device.enable_test_signal(channel)?;
            info!(
                "ADS1298 chip {index}: internal test signal on channel {channel}; \
                 electrodes are NOT being read"
            );
        }

        chip.device.read_data_continuous()?;
        chip.start_conversion()?;
    }
    info!("ADS1298 pair: streaming at {baud_rate_hz} Hz SPI");

    Ok(FrontEnds {
        chips,
        _power_down: power_down,
    })
}
