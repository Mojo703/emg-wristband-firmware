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
//! Threading. The frame read itself is not on a thread at all: [`frame_reader`]
//! clocks each frame out inside the chip's DRDY interrupt, because a conversion
//! not read within one ~487 µs period is gone and no scheduling policy can
//! promise that. What the threads do is everything after the bytes are safe.
//! [`acquisition::start`] spawns one [`chip_pipeline`] thread per chip — draining
//! that chip's frame ring, validating it, and owning its health — plus a combiner
//! thread that places both streams onto one time grid
//! ([`emg_runtime::alignment`]) and builds the model and wire windows. Everything
//! here owns its peripherals at `'static` instead of borrowing them.

pub(crate) mod acquisition;
pub(crate) mod ads1298;
pub(crate) mod channel;
mod chip_pipeline;
mod conditioning;
mod convert;
mod decode;
mod frame_reader;
mod preprocess;
mod registers;
mod spi_commands;
mod status;

pub(crate) use channel::Channel;
pub(crate) use channel::DEVICE_COUNT;
pub(crate) use convert::MICROVOLTS_PER_WIRE_COUNT;

use registers::Register;

/// Registers in one chip's map, which is how long a provenance snapshot is.
pub(crate) const REGISTER_COUNT: usize = Register::ALL.len();

/// The name and address of the register at `index` in a snapshot, so
/// [`crate::provenance`] can label the bytes without owning the register map.
pub(crate) fn register_identity(index: usize) -> (&'static str, u8) {
    let register = Register::ALL[index];
    (register.name(), register.addr())
}

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
    /// Each chip's interrupt-side frame path, claimed and proven against the
    /// driver path in [`bring_up`] but not yet enabled: the pipeline thread that
    /// drains a ring is the thing that opens its interrupt window.
    pub(super) readers: [frame_reader::FrameReader; DEVICE_COUNT],
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
    right_leg_drive: ads1298::RightLegDriveMode,
    command_config: &SpiConfig,
    frame_config: &SpiConfig,
) -> Result<(
    Ads1298FrontEnd,
    PinDriver<'static, Output>,
    esp_idf_svc::sys::spi_host_device_t,
)> {
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
    // Two device handles on the one bus: the command path at its clock and the
    // frame-read hot path at its own — esp-idf serialises them on the bus lock,
    // and CS is a GPIO either way.
    let device = Ads1298Device::new(
        SpiDeviceDriver::new(bus.clone(), Option::<AnyOutputPin>::None, command_config)
            .context("SPI command device")?,
        SpiDeviceDriver::new(bus, Option::<AnyOutputPin>::None, frame_config)
            .context("SPI frame device")?,
        PinDriver::output(wiring.chip_select)?,
        // DRDY is actively driven by the ADS1298, so no internal pull is needed.
        PinDriver::input(wiring.data_ready, Pull::Floating)?,
        PinDriver::output(wiring.reset)?,
        right_leg_drive,
    )
    .context("ADC pin init")?;
    Ok((
        Ads1298FrontEnd::new(device, PinDriver::output(wiring.start)?),
        power_down,
        // Carried out with the front end rather than looked up later: the host
        // is fixed by the peripheral this call consumed, and the interrupt-side
        // reader addresses it by register base, which must not be able to name
        // the wrong one.
        <SPI as esp_idf_svc::hal::spi::Spi>::device(),
    ))
}

/// Brings both chips up and leaves them streaming in RDATAC mode.
///
/// The chips self-clock (CLKSEL at 3V3), so no external clock has to be running
/// first: the post-RESET lockout counts cycles of each chip's own oscillator.
///
/// `command_baud_rate_hz` clocks registers and opcodes — the bench-validated rate,
/// nothing here is latency-sensitive. `frame_baud_rate_hz` clocks the RDATAC frame
/// reads: one 27-byte frame per chip has to clear well inside the ~500 µs sample
/// period, and every microsecond of transfer is edge-service budget, so this one
/// wants to be as fast as the wiring proves clean.
/// `test_signal_channel` is `Some(channel)` to drive each chip's internal square
/// wave into that channel instead of the electrodes. See
/// [`ads1298::Ads1298Device::enable_test_signal`].
pub(crate) fn bring_up<SpiA: SpiAnyPins + 'static, SpiB: SpiAnyPins + 'static>(
    spi_a: SpiA,
    spi_b: SpiB,
    wiring: [AdcChipWiring; DEVICE_COUNT],
    command_baud_rate_hz: u32,
    frame_baud_rate_hz: u32,
    test_signal_channel: Option<Channel>,
) -> Result<FrontEnds> {
    // The ADS1298 samples DIN on the falling edge and shifts DOUT on the rising edge:
    // SPI mode 1 (CPOL=0, CPHA=1).
    //
    // Interrupt-driven transactions, not polling. A polled transfer spins at
    // priority without yielding, starving whatever shares its core — measured on
    // the bench as the per-chip miss rate doubling and bad-status reads rising
    // tenfold, against ~15 µs saved on an uncontended read. Blocking transfers
    // let reads interleave with everything else, and the ~100 µs ISR turnaround
    // is absorbed by the sample period.
    let command_config = SpiConfig::new()
        .baudrate(command_baud_rate_hz.Hz())
        .data_mode(MODE_1)
        .polling(false);
    let frame_config = SpiConfig::new()
        .baudrate(frame_baud_rate_hz.Hz())
        .data_mode(MODE_1)
        .polling(false);

    let [wiring_a, wiring_b] = wiring;
    let [drive_a, drive_b] = ads1298::RIGHT_LEG_DRIVE_MODE;
    let (front_end_a, power_down_a, spi_host_a) =
        build_front_end(spi_a, wiring_a, drive_a, &command_config, &frame_config)?;
    // The GPIO interrupt dispatcher is installed from a thread pinned to core 1
    // because esp-idf allocates an interrupt on whichever core runs the
    // allocating call, and that dispatcher is now the frame-read path for both
    // chips: every DRDY edge is serviced there. Chip B's bus initialises in the
    // same thread, which puts its (command-path only) completion interrupt on
    // the same quiet core. See `crate::cores` for the whole plan.
    let command_config_b = command_config.clone();
    let frame_config_b = frame_config.clone();
    let built_b =
        crate::cores::spawn_pinned(crate::cores::GPIO_INTERRUPT_DISPATCHER_CORE, || {
            std::thread::Builder::new()
                .name("adc-init-b".into())
                .stack_size(8192)
                .spawn(move || {
                    frame_reader::install_interrupt_dispatcher()?;
                    build_front_end(spi_b, wiring_b, drive_b, &command_config_b, &frame_config_b)
                })
        })??
        .join();
    let (front_end_b, power_down_b, spi_host_b) =
        built_b.map_err(|_| anyhow::anyhow!("chip B front-end init thread panicked"))??;
    let mut chips = [front_end_a, front_end_b];
    let mut power_down = [power_down_a, power_down_b];
    let spi_hosts = [spi_host_a, spi_host_b];

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

        // The whole map, off the chip, last thing before the stream starts: this
        // is the acquisition setup a recorded session is stored against, and it
        // has to be taken after the test-signal muxing above or it would describe
        // a chip that is not the one converting.
        crate::provenance::record_front_end(index, chip.device.read_all_registers());

        chip.device.read_data_continuous()?;
        chip.start_conversion()?;
    }
    info!(
        "ADS1298 pair: streaming, frame reads at {frame_baud_rate_hz} Hz SPI, commands at {command_baud_rate_hz} Hz"
    );

    let mut claimed = Vec::with_capacity(DEVICE_COUNT);
    for (index, chip) in chips.iter_mut().enumerate() {
        claimed.push(claim_frame_reader(index, chip, spi_hosts[index])?);
    }
    let readers: [frame_reader::FrameReader; DEVICE_COUNT] = claimed
        .try_into()
        .map_err(|_| anyhow::anyhow!("one frame reader per chip"))?;

    Ok(FrontEnds {
        chips,
        readers,
        _power_down: power_down,
    })
}

/// Hands one streaming chip's SPI host over to its interrupt-side reader, and
/// proves the handover before anything depends on it.
///
/// The driver reads one frame — which is also what leaves the host's registers
/// holding the frame configuration the recipe is snapshotted from — and then the
/// reader's own code path reads the next one, still in task context. Two
/// consecutive frames off the same streaming chip: a recipe that clocks the
/// wrong length, mode, or bit order cannot produce a second valid status marker
/// by accident. The interrupt is attached but left disabled; the chip's pipeline
/// thread opens the window once it is ready to drain.
fn claim_frame_reader(
    index: usize,
    chip: &mut Ads1298FrontEnd,
    spi_host: esp_idf_svc::sys::spi_host_device_t,
) -> Result<frame_reader::FrameReader> {
    // A conversion has to have finished, or the read clocks out a frame the chip
    // is still producing and both sides of the comparison are garbage.
    FreeRtos::delay_ms(2);
    let driver_frame = chip
        .read_frame()
        .with_context(|| format!("chip {index} driver frame read"))?;
    let reader = frame_reader::FrameReader::claim(
        index,
        spi_host,
        chip.device.chip_select_pin(),
        chip.device.data_ready_pin(),
    )?;
    FreeRtos::delay_ms(2);
    match reader.validate().map(|bytes| decode::parse_sample(&bytes)) {
        Some(frame) if frame.status_word().is_some() => info!(
            "chip {index} interrupt-side frame path verified: driver status {:#08x}, \
             interrupt-path status {:#08x}",
            driver_frame.status, frame.status
        ),
        Some(frame) => warn!(
            "chip {index} interrupt-side frame path read {:#08x} with no status marker (the \
             driver read {:#08x}): the register recipe is wrong and this chip's data cannot \
             be trusted",
            frame.status, driver_frame.status
        ),
        None => warn!(
            "chip {index} interrupt-side frame path produced no frame: the SPI host did not \
             complete the transfer"
        ),
    }
    reader.attach()?;
    Ok(reader)
}
