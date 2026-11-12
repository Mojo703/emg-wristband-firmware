//! The EMG acquisition front end: two TI ADS1298 8-channel ADCs on the XIAO
//! motherboard.
//!
//! The chips share one SPI bus and the RESET, START, and PWDN lines; each has its
//! own chip select and DRDY. IC2 (CLKSEL high) runs its internal 2.048 MHz
//! oscillator and drives it out of CLK, and IC1 (CLKSEL low) converts on it, so
//! the two convert in lockstep from one START edge. Every RESET clears IC2's
//! clock output and stops IC1, which is why [`ChipPair`] resets and recovers the
//! two together and in a fixed order.
//!
//! Layout. [`registers`] is the typed register map and [`ads1298`] the register-level
//! driver; [`decode`]/[`status`]/[`convert`] are the hardware-free frame and signal
//! maths, and [`preprocess`] turns ADC codes into the raw-count stream consumed by
//! the calibrated feature pipeline. [`bring_up`] does the wiring and hands back a
//! configured, streaming pair.
//!
//! Threading. The frame read itself is not on a thread at all: [`frame_reader`]
//! clocks both chips' frames out inside IC1's DRDY interrupt, because a conversion
//! not read within one ~500 µs period is gone and no scheduling policy can promise
//! that. [`acquisition::start`] spawns one [`acquisition::pipeline`] thread that
//! drains both frame rings, validates them, and owns the pair's health, plus a
//! combiner thread that places both streams onto one time grid
//! ([`emg_runtime::alignment`]) and builds raw-count windows. Everything here owns
//! its peripherals at `'static` instead of borrowing them.

pub(crate) mod acquisition;
pub(crate) mod ads1298;
pub(crate) mod channel;
#[cfg(test)]
mod conditioning;
mod convert;
mod decode;
mod frame_reader;
mod preprocess;
mod registers;
mod status;

pub(crate) use channel::Channel;
pub(crate) use channel::DEVICE_COUNT;
pub(crate) use convert::MICROVOLTS_PER_WIRE_COUNT;

use registers::Register;

use anyhow::{Context, Result};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, Output, PinDriver, Pull};
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::{info, warn};
use std::sync::Arc;

use ads1298::Ads1298Device;

/// The ADS1298 reports this in its ID register. Bring-up checks it so a dead bus fails
/// loudly at boot instead of producing plausible-looking zeroes forever.
const EXPECTED_DEVICE_ID: u8 = 0x92;

/// The conversion clock: the clock source's internal oscillator, nominally
/// 2.048 MHz. Accuracy is ±0.5% at 25 °C and ±2% over 0-70 °C (SBAS459K §7.5);
/// both chips share whatever error it has.
pub(crate) const INTERNAL_OSCILLATOR_HZ: u32 = 2_048_000;

/// The front end's wiring: one SPI bus, the shared control lines, and each
/// chip's chip select and DRDY in device order (IC1, IC2).
pub(crate) struct FrontEndWiring {
    pub(crate) sclk: AnyOutputPin<'static>,
    pub(crate) data_in: AnyOutputPin<'static>,
    pub(crate) data_out: AnyInputPin<'static>,
    pub(crate) power_down: AnyOutputPin<'static>,
    pub(crate) reset: AnyOutputPin<'static>,
    pub(crate) start: AnyOutputPin<'static>,
    pub(crate) chip_select: [AnyOutputPin<'static>; DEVICE_COUNT],
    pub(crate) data_ready: [AnyInputPin<'static>; DEVICE_COUNT],
}

/// The two chips and every line they share.
pub(crate) struct ChipPair {
    chips: [Ads1298Device; DEVICE_COUNT],
    power_down: PinDriver<'static, Output>,
    reset: PinDriver<'static, Output>,
    start: PinDriver<'static, Output>,
}

/// After a power-up the follower counts tPOR (2^18 tCLK, 128 ms) from the first
/// clock edge it sees, which is the source's `CLK_EN` write.
const COLD_CLOCK_SETTLE_MILLISECONDS: u32 = 250;
/// After a warm RESET only the clock's restart needs covering.
const WARM_CLOCK_SETTLE_MILLISECONDS: u32 = 5;
/// How long bring-up waits for the first DRDY after START: several tSETTLE.
const FIRST_CONVERSION_TIMEOUT_MICROSECONDS: u64 = 20_000;

impl ChipPair {
    fn index_of(&self, role: ads1298::ClockRole) -> usize {
        self.chips
            .iter()
            .position(|chip| chip.clock_role() == role)
            .expect("ads1298::CLOCK_ROLE names one source and one follower")
    }

    pub(super) fn start_conversion(&mut self) -> Result<()> {
        self.start.set_high()?;
        Ok(())
    }

    pub(super) fn stop_conversion(&mut self) -> Result<()> {
        self.start.set_low()?;
        Ok(())
    }

    /// Resets both chips, restores the clock, checks both identities, and writes
    /// both register sets, leaving both in SDATAC.
    ///
    /// The shared RESET pulse clears the source's `CLK_EN`, so the source comes
    /// back first and drives CLK again before the follower is reset by opcode —
    /// the pulse itself reached an unclocked follower and cannot be trusted to
    /// have done anything there. Both identities are read and reported before a
    /// mismatch is fatal: the source failing points at the shared bus, the
    /// follower alone failing points at the clock.
    fn reset_and_configure(&mut self, clock_settle_milliseconds: u32) -> Result<()> {
        self.reset.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset.set_high()?;
        // 18 tCLK after RESET rises before any command (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);

        let source = self.index_of(ads1298::ClockRole::Source);
        let follower = self.index_of(ads1298::ClockRole::Follower);
        let mut identities = [0u8; DEVICE_COUNT];
        identities[source] = self.chips[source]
            .probe_identity()
            .with_context(|| format!("ADS1298 chip {source} identity probe"))?;
        self.chips[source].write_config1()?;
        FreeRtos::delay_ms(clock_settle_milliseconds);
        self.chips[follower].reset_by_command()?;
        identities[follower] = self.chips[follower]
            .probe_identity()
            .with_context(|| format!("ADS1298 chip {follower} identity probe"))?;

        for (index, &id) in identities.iter().enumerate() {
            if id == EXPECTED_DEVICE_ID {
                info!("ADS1298 chip {index}: ID {id:#04x} as expected");
            } else {
                warn!(
                    "ADS1298 chip {index}: ID {id:#04x}, expected {EXPECTED_DEVICE_ID:#04x} -- {}",
                    match (id, index == follower) {
                        (_, true) if identities[source] == EXPECTED_DEVICE_ID =>
                            "the clock source answers, so suspect the CLK net or this \
                             chip's CLKSEL strap",
                        (0x00 | 0xFF, _) =>
                            "bus reads all-zero or all-one, so this is wiring, \
                             power, or chip select rather than a wrong part",
                        _ =>
                            "the bus responds but the byte is wrong: suspect SPI mode \
                             or signal integrity on the shared bus",
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

        // The source first: its recipe keeps CLK_EN set, so the follower stays
        // clocked through its own writes.
        for index in [source, follower] {
            self.chips[index]
                .configure()
                .with_context(|| format!("ADS1298 chip {index} configure"))?;
        }
        Ok(())
    }

    /// Turns lead-off detection on or off on both chips. Conversions stop for the
    /// register writes and restart together on one START edge.
    pub(super) fn set_lead_off(&mut self, on: bool) -> Result<()> {
        self.stop_conversion()?;
        for chip in &mut self.chips {
            chip.stop_read_data_continuous()?;
        }
        FreeRtos::delay_ms(1);
        for chip in &mut self.chips {
            chip.set_lead_off(on)?;
            chip.read_data_continuous()?;
        }
        self.start_conversion()
    }

    /// Warm-recovers both chips after either dies mid-session: conversions
    /// stopped, the shared reset sequence, RDATAC, START again. The register work
    /// takes ~10 ms; the true cost is the reference-settling discard window
    /// acquisition applies afterwards, on both chips.
    pub(super) fn warm_recover(&mut self) -> Result<()> {
        self.stop_conversion()?;
        self.reset_and_configure(WARM_CLOCK_SETTLE_MILLISECONDS)?;
        for chip in &mut self.chips {
            chip.read_data_continuous()?;
        }
        self.start_conversion()
    }
}

/// The configured, streaming pair and its interrupt-side frame path, claimed and
/// proven against the driver path in [`bring_up`] but not yet enabled: the
/// pipeline thread that drains the rings opens the interrupt window.
pub(crate) struct StreamingPair {
    pub(super) pair: ChipPair,
    pub(super) reader: frame_reader::FrameReader,
}

/// Brings both chips up and leaves them streaming in RDATAC mode.
///
/// `command_baud_rate_hz` clocks registers and opcodes — the bench-validated rate,
/// nothing here is latency-sensitive. `frame_baud_rate_hz` clocks the RDATAC frame
/// reads: both chips' frames have to clear well inside the ~500 µs sample period,
/// and every microsecond of transfer is edge-service budget, so this one wants to be
/// as fast as the wiring proves clean. `test_signal_channel` is `Some(channel)` to
/// drive each chip's internal square wave into that channel instead of the
/// electrodes. See [`ads1298::Ads1298Device::enable_test_signal`].
pub(crate) fn bring_up<SPI: SpiAnyPins + 'static>(
    spi: SPI,
    wiring: FrontEndWiring,
    command_baud_rate_hz: u32,
    frame_baud_rate_hz: u32,
    test_signal_channel: Option<Channel>,
) -> Result<StreamingPair> {
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

    let FrontEndWiring {
        sclk,
        data_in,
        data_out,
        power_down,
        reset,
        start,
        chip_select,
        data_ready,
    } = wiring;
    // Low from construction: the chips stay powered down and in reset until the
    // power sequence below.
    let mut power_down = PinDriver::output(power_down)?;
    power_down.set_low()?;
    let mut reset = PinDriver::output(reset)?;
    reset.set_low()?;
    let mut start = PinDriver::output(start)?;
    start.set_low()?;

    let bus = Arc::new(
        SpiDriver::new(spi, sclk, data_in, Some(data_out), &SpiDriverConfig::new())
            .context("SPI bus init")?,
    );
    let spi_host = <SPI as esp_idf_svc::hal::spi::Spi>::device();
    let [chip_select_0, chip_select_1] = chip_select;
    let [data_ready_0, data_ready_1] = data_ready;
    let build_chip = |index: usize,
                      chip_select: AnyOutputPin<'static>,
                      data_ready: AnyInputPin<'static>|
     -> Result<Ads1298Device> {
        // Two device handles per chip on the one bus: the command path at its
        // clock and the frame-read hot path at its own. esp-idf serialises them
        // on the bus lock, and CS is a GPIO either way, held low across each
        // whole command — the mechanism the bring-up harness validated.
        Ads1298Device::new(
            SpiDeviceDriver::new(bus.clone(), Option::<AnyOutputPin>::None, &command_config)
                .context("SPI command device")?,
            SpiDeviceDriver::new(bus.clone(), Option::<AnyOutputPin>::None, &frame_config)
                .context("SPI frame device")?,
            PinDriver::output(chip_select)?,
            // DRDY is actively driven by the ADS1298, so no internal pull is needed.
            PinDriver::input(data_ready, Pull::Floating)?,
            ads1298::RIGHT_LEG_DRIVE_MODE[index],
            ads1298::CLOCK_ROLE[index],
        )
        .with_context(|| format!("ADS1298 chip {index} pin init"))
    };
    let chips = [
        build_chip(0, chip_select_0, data_ready_0)?,
        build_chip(1, chip_select_1, data_ready_1)?,
    ];
    let mut pair = ChipPair {
        chips,
        power_down,
        reset,
        start,
    };

    // The GPIO interrupt dispatcher is installed from a thread pinned to core 1
    // because esp-idf allocates an interrupt on whichever core runs the
    // allocating call, and that dispatcher is the frame-read path for both chips.
    // See `crate::cores` for the whole plan.
    crate::cores::spawn_pinned(crate::cores::GPIO_INTERRUPT_DISPATCHER_CORE, || {
        std::thread::Builder::new()
            .name("adc-dispatch".into())
            .stack_size(4096)
            .spawn(frame_reader::install_interrupt_dispatcher)
    })??
    .join()
    .map_err(|_| anyhow::anyhow!("DRDY dispatcher install thread panicked"))??;

    // Minimum PWDN assertion, release, then the supply settling. Roughly 2.5 s all
    // told, which must stay ahead of the task watchdog registration in `main`.
    info!("ADS1298 pair: powering up");
    FreeRtos::delay_ms(5);
    pair.power_down.set_high()?;
    pair.reset.set_high()?;
    FreeRtos::delay_ms(2000);

    pair.reset_and_configure(COLD_CLOCK_SETTLE_MILLISECONDS)?;

    // Internal reference settling: the datasheet's 150 ms start-up time with margin,
    // after `configure` powers each reference buffer and before START.
    FreeRtos::delay_ms(300);

    for (index, chip) in pair.chips.iter_mut().enumerate() {
        if let Some(channel) = test_signal_channel {
            // Registers can only be written outside RDATAC.
            chip.enable_test_signal(channel)?;
            info!(
                "ADS1298 chip {index}: internal test signal on channel {channel}; \
                 electrodes are NOT being read"
            );
        }

        // The whole map, off the chip, after the test-signal muxing above, so the
        // log describes the chip that is converting.
        let snapshot = chip.read_all_registers();
        let registers: Vec<String> = Register::ALL
            .iter()
            .zip(snapshot)
            .map(|(register, value)| match value {
                Some(value) => format!("{}={value:02x}", register.name()),
                None => format!("{}=??", register.name()),
            })
            .collect();
        info!("ADS1298 chip {index} registers: {}", registers.join(" "));

        chip.read_data_continuous()?;
    }
    pair.start_conversion()?;
    info!(
        "ADS1298 pair: streaming, frame reads at {frame_baud_rate_hz} Hz SPI, commands at {command_baud_rate_hz} Hz"
    );

    let reader = claim_frame_reader(&mut pair, spi_host)?;
    Ok(StreamingPair { pair, reader })
}

/// Hands the streaming pair's SPI host over to the interrupt-side reader, and
/// proves the handover before anything depends on it.
///
/// The driver reads one frame from each chip — which also leaves the host's
/// registers holding the frame configuration the recipe is snapshotted from — and
/// then the reader's own code path reads the next pair, still in task context.
/// Consecutive frames off the same streaming chips: a recipe that clocks the wrong
/// length, mode, or bit order cannot produce valid status markers by accident. The
/// interrupts are attached but left disabled; the pipeline thread opens the window
/// once it is ready to drain.
fn claim_frame_reader(
    pair: &mut ChipPair,
    spi_host: esp_idf_svc::sys::spi_host_device_t,
) -> Result<frame_reader::FrameReader> {
    // A conversion has to have finished, or the read clocks out a frame the chip
    // is still producing and both sides of the comparison are garbage. The first
    // one lands tSETTLE (4616 tCLK, ~2.3 ms) after START.
    let deadline = crate::device_now_us() + FIRST_CONVERSION_TIMEOUT_MICROSECONDS;
    while !pair.chips[0].data_is_ready() {
        if crate::device_now_us() > deadline {
            anyhow::bail!("chip 0 DRDY never fell after START");
        }
        FreeRtos::delay_ms(1);
    }
    let mut driver_status = [0u32; DEVICE_COUNT];
    for (index, chip) in pair.chips.iter_mut().enumerate() {
        let frame = chip
            .read_frame()
            .with_context(|| format!("chip {index} driver frame read"))?;
        if frame.status_word().is_none() {
            anyhow::bail!(
                "chip {index} driver frame has no status marker ({:#08x})",
                frame.status
            );
        }
        driver_status[index] = frame.status;
    }
    let [first, second] = &pair.chips;
    let reader = frame_reader::FrameReader::claim(
        spi_host,
        second.frame_spi_handle(),
        [first.chip_select_pin(), second.chip_select_pin()],
        [first.data_ready_pin(), second.data_ready_pin()],
    )?;
    FreeRtos::delay_ms(2);
    let frames = reader.validate().ok_or_else(|| {
        anyhow::anyhow!(
            "interrupt-side frame path produced no frames: the SPI host did not complete \
             the transfer"
        )
    })?;
    for (index, bytes) in frames.iter().enumerate() {
        let frame = decode::parse_sample(bytes);
        if frame.status_word().is_none() {
            anyhow::bail!(
                "chip {index} interrupt-side frame has no status marker ({:#08x}); driver \
                 status was {:#08x}",
                frame.status,
                driver_status[index]
            );
        }
        info!(
            "chip {index} interrupt-side frame path verified: driver status {:#08x}, \
             interrupt-path status {:#08x}",
            driver_status[index], frame.status
        );
    }
    reader.attach()?;
    Ok(reader)
}
