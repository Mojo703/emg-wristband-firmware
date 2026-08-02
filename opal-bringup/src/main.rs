//! ADS1298 bench harness: one chip, one thread, one text console.
//!
//! This exists because `opal-firmware` cannot answer a hardware question. The fault
//! under investigation: with two or more channel amplifiers powered and converting,
//! conversions stop 9 ms to 1.7 s after START — stochastically, on both AFE boards,
//! under every register configuration, input condition, and clock source tested. The
//! full campaign is written up in `documentation/ads1298-bringup-2026-07-31/`
//! (TEST-LOG.md for the summary, RESULTS-single-chip.md for the narrative).
//!
//! The harness has four modes, chosen by the knobs below:
//!
//! - **Staged stream** (`RUN_SURVEY = false`): configure, START, stream and audit
//!   CONFIG1-3 between frames, recovering and reporting on every revert.
//! - **Operating-point survey** (`RUN_SURVEY = true`): walk configuration cells,
//!   each from a fresh hardware reset, and print a survival matrix.
//! - **Fast-recovery acquisition** (`VALIDATE_RECOVERY_ONLY = true`): the working
//!   partial operating mode — stream 8 channels, detect a death in 10 ms, warm-reset
//!   and rewrite in ~3 ms, and report the verified yield. Measured 97% at
//!   2000 SPS x 8 channels (run 29).
//! - **Readout candidates** (`RUN_READOUT_CANDIDATES = true`): three candidate ways
//!   to get samples off a chip whose readout kills its own conversions — shorten the
//!   burst in time (SCLK ladder), shorten it in bits (partial-frame ladder), or move
//!   it out of the conversion entirely (sample-and-stop). Takes precedence over the
//!   two survey knobs.
//!
//! Frames are clocked out with RDATA rather than RDATAC so registers stay readable
//! mid-stream; DRDY is edge-waited so a floating wire cannot fake data; and logs go
//! straight out the USB cable.

mod ads1298;

use ads1298::{
    Ads1298, Frame, CHANNELS, REGISTER_NAMES, REG_CH1SET, REG_CONFIG1, REG_CONFIG2, REG_CONFIG3,
    REG_CONFIG4, REG_COUNT, REG_GPIO, REG_ID, REG_RLD_SENSP,
};
use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, PinDriver, Pull};
use esp_idf_svc::hal::ledc::{config::TimerConfig, LedcDriver, LedcTimerDriver, Resolution};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::{info, warn};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Knobs. Everything the bench changes between runs is in this block.
// ---------------------------------------------------------------------------

/// Which board is on the bench, for the banner. The control pins it names are chosen
/// in the pin map in `main`; keep the two in step.
const BOARD: &str = "A";

/// SPI clock for the streaming loop, picked from what the sweep below reports.
const SPI_BAUD_RATE_HZ: u32 = 2_000_000;

/// Clocks the boot-time sweep measures an ID-read error rate at. All five read clean
/// once multi-byte commands were burst-framed; kept as a per-boot health check.
const SWEEP_BAUD_RATES_HZ: [u32; 5] = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000];

/// ID reads per sweep step. Enough that a bus failing occasionally still shows up.
const SWEEP_ATTEMPTS: u32 = 200;

/// Drive the chip's internal square wave into this channel and short the rest. `None`
/// reads the electrodes. Recorded as making the reversion strictly worse, so it stays
/// off until the chip is stable without it.
const TEST_SIGNAL_CHANNEL: Option<usize> = None;

/// CONFIG1.DR. With HR set, `0b110` converts at 500 SPS — slow enough that a polled
/// loop never has to hurry, and exactly twice the 250 SPS a reset chip converts at, so
/// a revert doubles the measured period rather than nudging it.
const DATA_RATE_BITS: u8 = 0b100; // 2000 SPS in HR mode -- the ML target rate

/// CONFIG1: HR=1, DAISY_EN=1 (per-chip readback), CLK_EN=0, DR as above.
///
/// CLK_EN is 0 because the boards are separately clocked here: the wire between the two
/// CLK pins is gone and this board's CLKSEL is tied high, so nothing is listening. Set
/// it to 1 only to give a scope a trigger — the CLK output stopping is the cleanest
/// edge the reversion produces.
const CONFIG1: u8 = 0b1100_0000 | DATA_RATE_BITS; // CLK_EN=0: the CLK pin is an input now -- the ESP32 drives it

/// CONFIG2: internal test-signal generator off, or bit 4 set to switch it on.
const CONFIG2: u8 = if TEST_SIGNAL_CHANNEL.is_some() {
    0x10
} else {
    0x00
};

/// CONFIG2 with the internal test-signal generator off, and with it on at the default
/// amplitude and ~1 Hz (TEST_FREQ = 00, fCLK/2^21). The sample-and-stop cells switch
/// between the two per rung rather than following [`TEST_SIGNAL_CHANNEL`], because they
/// need the same rung measured both ways in one run.
const CONFIG2_TEST_SIGNAL_OFF: u8 = 0x00;
const CONFIG2_TEST_SIGNAL_ON: u8 = 0x10;

/// CONFIG3: internal reference buffer on (bit 7), bit 6 reserved-one, right-leg drive
/// and the lead-off sense that rides on it entirely off.
const CONFIG3: u8 = 0xC0;

/// CHnSET: powered up, gain 6, reading its electrode pair.
const CHANNEL_NORMAL: u8 = 0x00;
/// CHnSET muxed to the internal test signal.
const CHANNEL_TEST_SIGNAL: u8 = 0x05;
/// CHnSET with the inputs shorted, so the channel reads its own noise floor.
const CHANNEL_SHORTED: u8 = 0x01;
/// CHnSET with the channel amplifier powered down and the inputs shorted, so a cell
/// or stage can convert with no analog channel load at all.
const CHANNEL_POWERED_DOWN: u8 = 0x81;

/// The chip's four GPIO pins driven as outputs, low. The GPIO register resets to
/// 0x0F — four floating CMOS inputs on a bench that ties none of them — and floating
/// inputs are a classic source of on-die noise and excess current.
const GPIO_OUTPUTS_LOW: u8 = 0x00;

/// Wait after CONFIG3 powers the internal reference buffer before START. The datasheet
/// specifies a 150 ms reference start-up time and its initialization flow settles it
/// before conversions begin (SBAS459K figure 93); doubled for margin.
const REFERENCE_SETTLE_MS: u32 = 300;

/// How long the idle stage (configured, START low) runs before conversions begin.
/// Zero skips it. The reversion has arrived inside two seconds on every run that
/// produced it, so fifteen covers it several times over.
///
/// The staged run (15 s idle, 15 s powered-down) streamed 120 s with zero reverts
/// where the everything-at-once startup reverted inside a second — but it moved two
/// variables at once. These two knobs exist to separate them: START asserted later
/// versus channel amplifiers powered after conversions are already running.
const IDLE_STAGE_MS: u64 = 0;

/// How long the converting-with-channels-powered-down stage runs. Zero skips it, and
/// START is then asserted with the channels already configured normal — the
/// everything-at-once startup the reversion was first caught on.
const POWERED_DOWN_STAGE_MS: u64 = 0;

/// Gap between enabling one channel amplifier and the next after the powered-down
/// stage. Zero enables all eight in one burst of writes instead. Only meaningful when
/// the powered-down stage runs.
const CHANNEL_STAGGER_MS: u64 = 500;

/// Duty-cycle soft start for each channel amplifier: the PD bit is toggled at a ~2 ms
/// period with linearly rising on-time over this window, so the analog supply sees a
/// ramping average load instead of a step. The instantaneous on-burst is shallow
/// enough for the rail's output capacitance to carry. Zero switches the amplifier on
/// in one write, the way every run before run 14 did.
const CHANNEL_SOFT_START_MS: u64 = 0;

/// How often the idle stage re-reads the configuration.
const IDLE_AUDIT_PERIOD_MS: u32 = 100;

/// How long to wait for DRDY before calling the front end stalled.
const FRAME_TIMEOUT_US: u64 = 500_000;

/// Frames between summary lines. At 500 SPS this is one line every two seconds.
const REPORT_INTERVAL: u32 = 1000;

/// Frames between register audits. The reversion has been seen as early as 200 ms in,
/// so this needs to be frequent enough to time it, not just notice it.
const AUDIT_INTERVAL: u32 = 50;

/// Frames printed in full at the start of each stage, before the summaries take over.
const RAW_FRAMES_LOGGED: u32 = 3;

/// Pause between the DRDY falling edge and the first SPI activity of the frame read.
/// SBAS459K §9.5.2.6 forbids reading inside the tUPDATE keep-out, 4 tCLK (~2 µs)
/// around the DRDY pulse — and an edge-triggered read starts exactly there. Isolated
/// random-phase reads survive 198/198 while edge-triggered streaming dies within
/// ~10 frames, which is this window's fingerprint. 20 µs clears it with margin and
/// costs 4% of the sample period.
const READ_DELAY_AFTER_DATA_READY_US: u32 = 20;

/// Skip the discriminator cells and run only the fast-recovery data-quality pass,
/// with the channels on their electrode inputs (MUX=000) so whatever is wired to the
/// analog connector is actually in circuit.
const VALIDATE_RECOVERY_ONLY: bool = true;

/// Run the operating-point survey instead of the staged stream: walk channel counts
/// and windowed-conversion patterns, with a full hardware reset between cells, and
/// report a survival matrix. This is the map of what the board can actually sustain,
/// for choosing a partial operating point for the product.
const RUN_SURVEY: bool = true;

/// How long a survey cell must stream to be called a survivor. Every collapse so far
/// has arrived inside a second; ten covers it an order of magnitude over.
const CELL_DURATION_MS: u64 = 10_000;

/// Run the readout-candidate families instead of either survey. Overrides
/// [`RUN_SURVEY`] and [`VALIDATE_RECOVERY_ONLY`]; flip this one knob and nothing else.
///
/// The fault is pinned to clocking conversion data out of a converting chip, with a
/// dose that scales with data bits shifted (runs 33-37). Runs 38a/38b then refuted the
/// high-SCLK candidate outright, put the partial-frame threshold between 24 and 72
/// data bits, and turned up a reproducible split by input mux that nothing in the
/// campaign predicts. This round follows those three leads.
const RUN_READOUT_CANDIDATES: bool = false;

/// The SCLK ladder is concluded — 16 cells across runs 38a/38b, zero config-held
/// survivals at 8/12/16/20 MHz with every signal-integrity gate clean. The code stays
/// for the record and for a rerun on a respun board; it is out of the run list.
const RUN_CLOCK_LADDER: bool = false;

/// SCLK rungs for the clock ladder. The ESP32-S3 SPI master divides 80 MHz, so 8, 16
/// and 20 MHz land exactly and 12 MHz is served as 80/7 = 11.43 MHz; the rung is named
/// for what was asked, and the gate below measures what was actually delivered.
const CANDIDATE_CLOCK_RATES_HZ: [u32; 4] = [8_000_000, 12_000_000, 16_000_000, 20_000_000];

/// Rounds of the clock-ladder signal-integrity gate: one ID read plus one scratch
/// write/readback each. A rung that fails any of them never gets to dose, because a
/// garbled bus and a dead chip are the same silence from the DRDY pin.
const SIGNAL_INTEGRITY_ROUNDS: u32 = 100;

/// Scratch patterns for that gate, cycled per round. Adjacent-bit and nibble
/// transitions are what a marginal clock loses first.
const SIGNAL_INTEGRITY_PATTERNS: [u8; 4] = [0x55, 0xAA, 0x0F, 0xF0];

/// The DRDY period at the configured rate: 500 µs at 2000 SPS. Dosed cells run at this
/// as well as at 2 ms, because per-DRDY pacing is what a real readout would do.
const DATA_READY_INTERVAL_US: u64 = 1_000_000 / samples_per_second() as u64;

/// Repeats per mux-discriminator combination within one flash. Two is what fits the
/// bench minutes and still distinguishes "always" from "sometimes" on a fault whose
/// per-cell outcome has been stochastic all campaign.
const MUX_DISCRIMINATOR_REPEATS: u32 = 2;

/// Partial-frame rungs as (data bytes after RDATA, dose interval, repeats).
///
/// Runs 38a/38b: 3 bytes survived 4/4 at both pacings, 9 bytes survived 1/4, and 15 and
/// 27 bytes died 0/8. The threshold is therefore between 24 and 72 data bits, so this
/// round adds the untested 6-byte rung (status plus one channel, 48 bits) at both
/// pacings and repeats the 9-byte rung at the tighter pacing to firm up its 1-in-4.
/// The 15- and 27-byte rungs are dropped: eight deaths out of eight is settled.
const PARTIAL_FRAME_RUNGS: [(usize, u64, u32); 3] = [
    (6, 2_000, 4),
    (6, DATA_READY_INTERVAL_US, 4),
    (9, DATA_READY_INTERVAL_US, 4),
];

/// Sample-and-stop rungs: DRDY edges counted inside one conversion window before START
/// drops. Kept from runs 38a/38b for their survival data — these cells are the clearest
/// statement of the mux split — but see [`run_sample_and_stop_cell`] for why their
/// per-channel numbers are not measurements of the input.
const SAMPLE_AND_STOP_RUNGS: [u32; 5] = [1, 2, 3, 4, 8];

/// How long each sample-and-stop or single-shot rung cycles for.
const SAMPLE_AND_STOP_DURATION_MS: u64 = 5_000;

/// How long one windowed conversion may take to produce its DRDY edge before the rung
/// is called dead. tSETTLE at DR = 100 in high-resolution mode is 4616 tCLK = 2.25 ms
/// (SBAS459K table 12), so this is twenty times the longest legitimate wait.
const SAMPLE_AND_STOP_EDGE_TIMEOUT_US: u64 = 50_000;

/// CONFIG4 with SINGLE_SHOT (bit 3) set, and the reset value it replaces.
const CONFIG4_SINGLE_SHOT: u8 = 0x08;

/// How long START is held low between single-shot conversions. The datasheet asks for
/// at least 2 tCLK, which is under a microsecond at 2.048 MHz (SBAS459K §9.4.1.4).
const START_PULSE_LOW_US: u32 = 5;

/// Repeats per single-shot input configuration.
const SINGLE_SHOT_REPEATS: u32 = 3;

// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== opal-bringup ({}) ===", reset_reason());
    info!("board {BOARD}, SPI {SPI_BAUD_RATE_HZ} Hz, CONFIG1 {CONFIG1:#04x} ({} SPS), test signal {TEST_SIGNAL_CHANNEL:?}",
        samples_per_second());

    let peripherals = Peripherals::take()?;
    let pins = peripherals.pins;

    // The external master clock: 2.048 MHz from the LEDC peripheral into the chip's
    // CLK pin, with CLKSEL strapped to GND so the chip uses it instead of its
    // internal RC oscillator. The oscillator is the one shared conversion-domain
    // block never yet exonerated: DRDY dying while SPI (clocked by SCLK) stays alive
    // is exactly what a stopping master clock looks like. The clock must run before
    // the chip comes out of reset, so this precedes power-up. The driver stays bound
    // for the whole session; dropping it would silence the clock.
    let clock_timer = LedcTimerDriver::new(
        peripherals.ledc.timer0,
        &TimerConfig::new()
            .frequency(2_048_000.Hz())
            .resolution(Resolution::Bits4),
    )?;
    let mut master_clock = LedcDriver::new(peripherals.ledc.channel0, &clock_timer, pins.gpio10)?;
    master_clock.set_duty(master_clock.get_max_duty() / 2)?;
    info!("external master clock: 2.048 MHz on GPIO10, CLKSEL expected at GND");

    // The pin map, rewired 2026-07-31 so the jumpers run in header order on the
    // breakout: SCLK 2, CS 3, START 4, RST 5, MOSI 6, MISO 7, PWDN 8, DRDY 9.
    // GPIO 19 and 20 belong to USB-Serial-JTAG.
    let clock: AnyOutputPin = pins.gpio2.into();
    let data_in: AnyOutputPin = pins.gpio6.into();
    let data_out: AnyInputPin = pins.gpio7.into();
    let start: AnyOutputPin = pins.gpio4.into();

    let chip_select: AnyOutputPin = pins.gpio3.into();
    let data_ready: AnyInputPin = pins.gpio9.into();
    let reset_n: AnyOutputPin = pins.gpio5.into();
    let power_down: AnyOutputPin = pins.gpio8.into();

    let bus = Arc::new(SpiDriver::new(
        peripherals.spi2,
        clock,
        data_in,
        Some(data_out),
        &SpiDriverConfig::new(),
    )?);

    // No hardware chip select: `Ads1298` drives CS as a GPIO, low around each
    // transaction, so the rising edge resets the chip's command decoder and the SPI
    // clock stays free to be swapped for the sweep.
    let mut chip = Ads1298::new(
        spi_device(&bus, SWEEP_BAUD_RATES_HZ[0])?,
        PinDriver::input(data_ready, Pull::Floating)?,
        PinDriver::output(reset_n)?,
        PinDriver::output(power_down)?,
        PinDriver::output(start)?,
        PinDriver::output(chip_select)?,
    )?;

    info!("powering up (2 s settling)");
    chip.power_up()?;

    // A wrong ID does not stop the run. It used to be the reason to abort, but the
    // clock sweep below is precisely what tells a marginal bus from a dead one, and
    // aborting throws that away at the moment it is most wanted.
    match chip.read_register(REG_ID) {
        Ok(ads1298::DEVICE_ID) => info!("ID reads {:#04x} as expected", ads1298::DEVICE_ID),
        Ok(other) => warn!(
            "ID reads {other:#04x}, expected {:#04x}{}",
            ads1298::DEVICE_ID,
            match other {
                0x00 | 0xFF => " -- the bus is stuck at one level, so suspect wiring or power",
                _ => " -- the bus answers, so suspect SPI mode or the part number",
            }
        ),
        Err(error) => warn!("ID read failed: {error}"),
    }

    sweep_spi_clock(&mut chip, &bus)?;
    chip.set_spi(spi_device(&bus, SPI_BAUD_RATE_HZ)?);

    if RUN_READOUT_CANDIDATES {
        readout_candidates(&mut chip, &bus)?;
        info!("readout candidates complete; idling");
        loop {
            FreeRtos::delay_ms(10_000);
        }
    }

    if RUN_SURVEY {
        // An output the ADS1298 cannot see, toggled as a stand-in aggressor: if
        // edge activity on an unconnected ESP pin kills conversions too, the fault
        // is ESP-side supply/ground noise and the SPI port itself is innocent.
        let mut unrelated_output = PinDriver::output(pins.gpio11)?;
        survey(&mut chip, &bus, &mut unrelated_output)?;
        info!("survey complete; idling");
        loop {
            FreeRtos::delay_ms(10_000);
        }
    }

    log_registers(&mut chip, "after reset")?;

    let normal_writes = planned_writes(normal_channel_value);
    apply(&mut chip, &normal_writes)?;
    verify(&mut chip, &normal_writes)?;
    log_registers(&mut chip, "after configure")?;

    info!("waiting {REFERENCE_SETTLE_MS} ms for the internal reference to settle");
    FreeRtos::delay_ms(REFERENCE_SETTLE_MS);

    if IDLE_STAGE_MS > 0 {
        run_stage(
            &mut chip,
            "idle, START low",
            &normal_writes,
            false,
            Some(IDLE_STAGE_MS),
        )?;
    }

    if POWERED_DOWN_STAGE_MS > 0 {
        let powered_down_writes = planned_writes(|_| CHANNEL_POWERED_DOWN);
        apply(&mut chip, &powered_down_writes)?;
        verify(&mut chip, &powered_down_writes)?;
        chip.start_conversion()?;
        info!("START asserted");
        run_stage(
            &mut chip,
            "converting, channels powered down",
            &powered_down_writes,
            true,
            Some(POWERED_DOWN_STAGE_MS),
        )?;
        if CHANNEL_STAGGER_MS > 0 {
            // One channel amplifier at a time, each watched as its own stage — so a
            // load-driven wedge names the channel count it arrived at.
            let mut cumulative_writes = powered_down_writes.clone();
            for channel in 0..CHANNELS {
                let slot = 3 + channel;
                let (address, _) = cumulative_writes[slot];
                let value = normal_channel_value(channel);
                soft_start_channel(&mut chip, address, value)?;
                cumulative_writes[slot] = (address, value);
                run_stage(
                    &mut chip,
                    &format!("channels 1..={} enabled", channel + 1),
                    &cumulative_writes,
                    true,
                    Some(CHANNEL_STAGGER_MS),
                )?;
            }
        } else {
            apply(&mut chip, &normal_writes)?;
            verify(&mut chip, &normal_writes)?;
        }
    } else {
        chip.start_conversion()?;
        info!("START asserted with channels already normal");
    }
    run_stage(
        &mut chip,
        "converting, channels normal",
        &normal_writes,
        true,
        None,
    )?;
    Ok(())
}

fn spi_device(bus: &Arc<SpiDriver<'static>>, baud_rate_hz: u32) -> Result<ads1298::SpiDevice> {
    // The ADS1298 samples DIN on the falling edge and shifts DOUT on the rising edge:
    // SPI mode 1 (CPOL=0, CPHA=1).
    Ok(SpiDeviceDriver::new(
        bus.clone(),
        Option::<AnyOutputPin>::None,
        &SpiConfig::new()
            .baudrate(baud_rate_hz.Hz())
            .data_mode(MODE_1),
    )?)
}

/// The rate `CONFIG1` selects, so the banner cannot disagree with the register.
/// fMOD is fCLK/4 in high-resolution mode, and DR divides it by `16 << DR`.
const fn samples_per_second() -> u32 {
    (2_048_000 / 4) / (16u32 << DATA_RATE_BITS)
}

/// Measures the ID-read error rate at each clock in [`SWEEP_BAUD_RATES_HZ`].
fn sweep_spi_clock(chip: &mut Ads1298, bus: &Arc<SpiDriver<'static>>) -> Result<()> {
    for baud_rate_hz in SWEEP_BAUD_RATES_HZ {
        chip.set_spi(spi_device(bus, baud_rate_hz)?);
        let failures = chip.probe_identity(SWEEP_ATTEMPTS);
        if failures == 0 {
            info!("SPI {baud_rate_hz:>8} Hz: {SWEEP_ATTEMPTS}/{SWEEP_ATTEMPTS} ID reads correct");
        } else {
            warn!("SPI {baud_rate_hz:>8} Hz: {failures}/{SWEEP_ATTEMPTS} ID reads WRONG");
        }
    }
    Ok(())
}

/// One row of the survival matrix. `config_held` distinguishes a genuine survivor
/// from a chip that silently reverted to power-on defaults and kept converting at
/// 250 SPS — which the frame count alone confused for survival in the first surveys.
struct CellReport {
    label: String,
    frames: u32,
    died_at_ms: Option<u64>,
    config_held: bool,
    /// Set only on a clock-ladder gate row: `Some(0)` for a rung whose bus checked out,
    /// `Some(n)` for n wrong ID reads or scratch readbacks. That is a failure of the
    /// wiring at that SCLK, not of the chip, and the matrix has to say so — a rung
    /// that never got to dose must never read as a rung that dosed and survived.
    signal_integrity_failures: Option<u32>,
}

/// Prints the survival matrix. The format is diffed across bench runs, so the three
/// existing verdict lines are fixed; gate rows are new and carry their own keywords.
fn print_survival_matrix(reports: &[CellReport]) {
    info!("=== survival matrix ===");
    for report in reports {
        if let Some(failures) = report.signal_integrity_failures {
            if failures == 0 {
                info!(
                    "GATE-OK   {}: {} of {} bus checks clean",
                    report.label, report.frames, report.frames
                );
            } else {
                info!(
                    "SI-FAIL   {}: {} of {} bus checks wrong, dosed cells skipped",
                    report.label, failures, report.frames
                );
            }
            continue;
        }
        let config = if report.config_held {
            "config HELD"
        } else {
            "config LOST"
        };
        match report.died_at_ms {
            None => info!(
                "SURVIVED  {}: {} frames, {config}",
                report.label, report.frames
            ),
            Some(at) => info!(
                "DIED      {}: at {} ms, {} frames, {config}",
                report.label, at, report.frames
            ),
        }
    }
}

/// Walks the operating-point cells, each from a fresh hardware reset, and prints the
/// survival matrix at the end. Inputs are internally shorted (MUX=001) throughout, so
/// the cells measure the board, not the electrodes.
fn survey(
    chip: &mut Ads1298,
    bus: &Arc<SpiDriver<'static>>,
    unrelated_output: &mut PinDriver<'static, esp_idf_svc::hal::gpio::Output>,
) -> Result<()> {
    if VALIDATE_RECOVERY_ONLY {
        // Three-way input-path discriminator, 20 s each. Internal short measures the
        // chip-side noise floor with the externals out of circuit; internal test
        // signal proves mux/PGA/ADC per channel; electrode mux puts the external
        // path (connector, ESD network, series resistors) in circuit. A channel
        // healthy on the first two but dead on the third is broken between the
        // package pin and the connector.
        run_recovery_cell(chip, 20_000, CHANNEL_SHORTED, CONFIG2_TEST_SIGNAL_OFF)?;
        run_recovery_cell(chip, 20_000, CHANNEL_TEST_SIGNAL, CONFIG2_TEST_SIGNAL_ON)?;
        return run_recovery_cell(chip, 20_000, CHANNEL_NORMAL, CONFIG2_TEST_SIGNAL_OFF);
    }
    let mut reports = Vec::new();

    // The murder weapon is the frame read specifically (register reads at 1 ms
    // spacing are harmless; frame reads revert at 100 ms and kill at 2 ms). These
    // cells split it: the unbroken 216-clock data burst (chunked vs continuous)
    // versus the RDATA command path (opcode with no data clocks).
    reports.push(run_dosed_cell(chip, 2_000, Dose::FrameContinuous, 1)?);
    reports.push(run_dosed_cell(chip, 2_000, Dose::FrameChunked, 1)?);
    reports.push(run_dosed_cell(chip, 2_000, Dose::FrameChunked, 2)?);
    reports.push(run_dosed_cell(chip, 2_000, Dose::OpcodeOnly, 1)?);
    reports.push(run_dosed_cell(chip, 2_000, Dose::OpcodeOnly, 2)?);
    reports.push(run_dosed_cell(chip, 2_000, Dose::Register, 1)?);

    print_survival_matrix(&reports);
    Ok(())
}

/// Resets the chip and configures `powered_channels` amplifiers on (inputs shorted),
/// the rest powered down.
fn reset_and_configure(
    chip: &mut Ads1298,
    powered_channels: usize,
    config1: u8,
    config3: u8,
    gpio: Option<u8>,
) -> Result<Vec<(u8, u8)>> {
    chip.stop_conversion()?;
    chip.power_up()?;
    let mut writes = planned_writes(|channel| {
        if channel < powered_channels {
            CHANNEL_SHORTED
        } else {
            CHANNEL_POWERED_DOWN
        }
    });
    for entry in writes.iter_mut() {
        if entry.0 == REG_CONFIG1 {
            entry.1 = config1;
        }
        if entry.0 == REG_CONFIG3 {
            entry.1 = config3;
        }
    }
    if let Some(value) = gpio {
        writes.push((REG_GPIO, value));
    }
    apply(chip, &writes)?;
    FreeRtos::delay_ms(REFERENCE_SETTLE_MS);
    Ok(writes)
}

/// What a cell's powered channels are looking at: the CHnSET mux value and whether the
/// internal test generator runs behind it.
///
/// This is a cell parameter rather than a constant because of a pattern nobody was
/// looking for: every dosed-cell death in runs 33-38 ran the shorted mux, and the
/// sample-and-stop family then split cleanly by mux — all ten test-signal rungs lived,
/// nine of ten shorted rungs died, same code and same pins. Either the shorted input
/// state is part of the kill condition or the split is a coincidence across twenty
/// cells, and one dosed ladder settles it.
#[derive(Clone, Copy)]
struct InputConfiguration {
    label: &'static str,
    /// How the sample-and-stop cells spelled this configuration in runs 38a/38b. Those
    /// labels are in the bench logs and get diffed run to run, so they are carried
    /// verbatim rather than folded into the shorter `label` the dosed cells use.
    window_label: &'static str,
    channel_value: u8,
    config2: u8,
}

/// Inputs shorted internally (MUX=001). The condition every recorded death ran under,
/// and the tie back to the run 33-38 baseline.
const INPUT_SHORTED: InputConfiguration = InputConfiguration {
    label: "shorted",
    window_label: "inputs shorted",
    channel_value: CHANNEL_SHORTED,
    config2: CONFIG2_TEST_SIGNAL_OFF,
};

/// The internal square wave on every channel (MUX=101, generator on).
const INPUT_TEST_SIGNAL: InputConfiguration = InputConfiguration {
    label: "test signal",
    window_label: "internal test signal",
    channel_value: CHANNEL_TEST_SIGNAL,
    config2: CONFIG2_TEST_SIGNAL_ON,
};

/// The electrode mux (MUX=000): connector, ESD network, series resistors and PGA all in
/// circuit, with the P-N pairs externally tied on this bench. This is the production
/// path, and no dosed cell has ever run it.
const INPUT_ELECTRODE: InputConfiguration = InputConfiguration {
    label: "electrode",
    window_label: "electrode inputs",
    channel_value: CHANNEL_NORMAL,
    config2: CONFIG2_TEST_SIGNAL_OFF,
};

/// [`reset_and_configure`] with an input configuration laid over it: CONFIG2 for the
/// generator, and every powered channel's CHnSET for the mux. Powered-down channels
/// keep their powered-down value, so the amplifier count is untouched.
fn reset_and_configure_inputs(
    chip: &mut Ads1298,
    inputs: InputConfiguration,
) -> Result<Vec<(u8, u8)>> {
    let mut writes = reset_and_configure(chip, CHANNELS, CONFIG1, CONFIG3, Some(GPIO_OUTPUTS_LOW))?;
    for entry in writes.iter_mut() {
        if entry.0 == REG_CONFIG2 {
            entry.1 = inputs.config2;
        }
        let is_channel = entry.0 >= REG_CH1SET && entry.0 < REG_CH1SET + CHANNELS as u8;
        if is_channel && entry.1 == CHANNEL_SHORTED {
            entry.1 = inputs.channel_value;
        }
    }
    apply(chip, &writes)?;
    Ok(writes)
}

/// Streams continuously for [`CELL_DURATION_MS`] or until DRDY dies.
fn run_continuous_cell(
    chip: &mut Ads1298,
    powered_channels: usize,
    config1: u8,
    config3: u8,
    gpio: Option<u8>,
    start_by_opcode: bool,
    tag: &str,
) -> Result<CellReport> {
    let label = format!("{powered_channels} channels continuous, {tag}");
    info!("cell '{label}'");
    reset_and_configure(chip, powered_channels, config1, config3, gpio)?;
    if start_by_opcode {
        chip.start_conversion_by_command()?;
    } else {
        chip.start_conversion()?;
    }
    let started_us = now_us();
    let mut frames = 0u32;
    let mut died_at_ms = None;
    loop {
        let elapsed_ms = (now_us() - started_us) / 1000;
        if elapsed_ms >= CELL_DURATION_MS {
            break;
        }
        match wait_data_ready_falling_edge(chip) {
            Ok(()) => {
                Ets::delay_us(READ_DELAY_AFTER_DATA_READY_US);
                if chip.read_frame().is_ok() {
                    frames += 1;
                }
            }
            Err(_) => {
                died_at_ms = Some(elapsed_ms);
                break;
            }
        }
    }
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == config1)
        .unwrap_or(false);
    if start_by_opcode {
        chip.stop_conversion_by_command()?;
    } else {
        chip.stop_conversion()?;
    }
    Ok(CellReport {
        label,
        frames,
        died_at_ms,
        config_held,
        signal_integrity_failures: None,
    })
}

/// Conversions with the SPI bus untouched for the whole cell: DRDY falling edges are
/// counted by polling the pin, and the first register access happens only after the
/// cell ends. If the chip still dies here, no ESP-side SPI behavior can be the
/// trigger. `frames` in the report is the edge count (nominal: duration x rate).
fn run_silent_bus_cell(chip: &mut Ads1298) -> Result<CellReport> {
    let label = "8 channels converting, bus silent".to_string();
    info!("cell '{label}'");
    reset_and_configure(chip, CHANNELS, CONFIG1, CONFIG3, Some(GPIO_OUTPUTS_LOW))?;
    chip.start_conversion()?;
    let started_us = now_us();
    let mut edges: u32 = 0;
    let mut last_edge_us = started_us;
    let mut was_low = chip.data_ready();
    while (now_us() - started_us) / 1000 < CELL_DURATION_MS {
        let low = chip.data_ready();
        if low && !was_low {
            edges += 1;
            last_edge_us = now_us();
        }
        was_low = low;
    }
    chip.stop_conversion()?;
    let last_edge_ms = (last_edge_us - started_us) / 1000;
    // Dead if the edges stopped well before the cell did.
    let died_at_ms = (last_edge_ms + 1_000 < CELL_DURATION_MS).then_some(last_edge_ms);
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == CONFIG1)
        .unwrap_or(false);
    Ok(CellReport {
        label,
        frames: edges,
        died_at_ms,
        config_held,
        signal_integrity_failures: None,
    })
}

/// One register read per `interval_ms` on an otherwise silent bus, while counting
/// DRDY edges by GPIO. Reports how many doses were issued before the edges stopped,
/// which is the per-transaction death probability measured directly.
#[derive(Clone, Copy, Debug)]
enum Dose {
    Register,
    FrameContinuous,
    FrameChunked,
    OpcodeOnly,
    /// A frame truncated to this many data bytes, for walking the burst-length axis.
    FramePartial(usize),
}

fn run_dosed_cell(
    chip: &mut Ads1298,
    interval_us: u64,
    dose: Dose,
    repeat: u32,
) -> Result<CellReport> {
    let label = format!("dosed: {dose:?} per {interval_us} us, run {repeat}");
    run_dosed_cell_labeled(chip, interval_us, dose, INPUT_SHORTED, label)
}

/// The dosed cell proper, with the input configuration and label supplied. The
/// readout-candidate families name their cells by the axis they walk rather than by the
/// `Dose` variant, and those labels are what the matrix is grepped on.
fn run_dosed_cell_labeled(
    chip: &mut Ads1298,
    interval_us: u64,
    dose: Dose,
    inputs: InputConfiguration,
    label: String,
) -> Result<CellReport> {
    info!("cell '{label}'");
    reset_and_configure_inputs(chip, inputs)?;
    chip.start_conversion()?;
    let started_us = now_us();
    let mut edges: u32 = 0;
    let mut doses: u32 = 0;
    let mut last_edge_us = started_us;
    let mut next_dose_us = started_us + interval_us;
    let mut was_low = chip.data_ready();
    while (now_us() - started_us) / 1000 < CELL_DURATION_MS {
        let low = chip.data_ready();
        if low && !was_low {
            edges += 1;
            last_edge_us = now_us();
        }
        was_low = low;
        if now_us() >= next_dose_us {
            match dose {
                Dose::Register => drop(chip.read_register(REG_ID)),
                Dose::FrameContinuous => drop(chip.read_frame()),
                Dose::FrameChunked => drop(chip.read_frame_chunked()),
                Dose::OpcodeOnly => drop(chip.read_frame_opcode_only()),
                Dose::FramePartial(bytes) => drop(chip.read_frame_partial(bytes)),
            }
            doses += 1;
            next_dose_us += interval_us;
        }
        // Stop dosing a corpse: once edges have been silent for a second the cell's
        // number is decided, and further doses only blur it.
        if (now_us() - last_edge_us) / 1000 > 1_000 {
            break;
        }
    }
    chip.stop_conversion()?;
    let last_edge_ms = (last_edge_us - started_us) / 1000;
    let died_at_ms = (last_edge_ms + 1_000 < CELL_DURATION_MS).then_some(last_edge_ms);
    info!("  dosed: {doses} transactions issued, edges stopped at {last_edge_ms} ms");
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == CONFIG1)
        .unwrap_or(false);
    Ok(CellReport {
        label,
        frames: edges,
        died_at_ms,
        config_held,
        signal_integrity_failures: None,
    })
}

/// A transaction's worth of edges on a pin the chip cannot see, once per 100 ms,
/// bus silent. Kills here mean ESP-side supply/ground noise; survival exonerates
/// the ESP's mere switching and points back at the SPI lines themselves.
fn run_gpio_aggressor_cell(
    chip: &mut Ads1298,
    output: &mut PinDriver<'static, esp_idf_svc::hal::gpio::Output>,
    repeat: u32,
) -> Result<CellReport> {
    let label = format!("unrelated GPIO aggressor, run {repeat}");
    info!("cell '{label}'");
    reset_and_configure(chip, CHANNELS, CONFIG1, CONFIG3, Some(GPIO_OUTPUTS_LOW))?;
    chip.start_conversion()?;
    let started_us = now_us();
    let mut edges: u32 = 0;
    let mut last_edge_us = started_us;
    let mut next_burst_us = started_us + 100_000;
    let mut was_low = chip.data_ready();
    while (now_us() - started_us) / 1000 < CELL_DURATION_MS {
        let low = chip.data_ready();
        if low && !was_low {
            edges += 1;
            last_edge_us = now_us();
        }
        was_low = low;
        if now_us() >= next_burst_us {
            // ~a register read's worth of transitions at roughly SCLK pace.
            for _ in 0..108 {
                output.set_high()?;
                Ets::delay_us(1);
                output.set_low()?;
                Ets::delay_us(1);
            }
            next_burst_us += 100_000;
        }
    }
    chip.stop_conversion()?;
    let last_edge_ms = (last_edge_us - started_us) / 1000;
    let died_at_ms = (last_edge_ms + 1_000 < CELL_DURATION_MS).then_some(last_edge_ms);
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == CONFIG1)
        .unwrap_or(false);
    Ok(CellReport {
        label,
        frames: edges,
        died_at_ms,
        config_held,
        signal_integrity_failures: None,
    })
}

// ---------------------------------------------------------------------------
// Readout candidates: ways to survive a chip whose own readout kills its conversions,
// plus the discriminator that runs 38a/38b turned up by accident. Each family stands
// alone; the mode runs them all and prints one matrix, because the interesting
// comparison is across families.
// ---------------------------------------------------------------------------

/// Runs the families in order, then prints the combined matrix and the settling table.
/// Every cell inside resets and reconfigures from scratch, so a death in one cannot
/// carry into the next.
fn readout_candidates(chip: &mut Ads1298, bus: &Arc<SpiDriver<'static>>) -> Result<()> {
    let mut reports = Vec::new();

    if RUN_CLOCK_LADDER {
        info!("=== SCLK ladder ===");
        reports.extend(run_clock_ladder_family(chip, bus)?);
    }

    info!("=== mux discriminator ===");
    reports.extend(run_mux_discriminator_family(chip)?);

    info!("=== partial-frame dose ladder ===");
    reports.extend(run_partial_frame_family(chip)?);

    info!("=== sample-and-stop ===");
    let sample_and_stop = run_sample_and_stop_family(chip)?;
    let single_shot = run_single_shot_family(chip)?;
    reports.extend(
        sample_and_stop
            .iter()
            .map(SampleAndStopReport::as_cell_report),
    );
    reports.extend(single_shot.iter().map(SampleAndStopReport::as_cell_report));

    print_survival_matrix(&reports);
    let mut groups: Vec<&[SampleAndStopReport]> = sample_and_stop
        .chunks(SAMPLE_AND_STOP_RUNGS.len())
        .collect();
    groups.extend(single_shot.chunks(SINGLE_SHOT_REPEATS as usize));
    print_sample_and_stop_summary(&groups);
    Ok(())
}

/// The mux discriminator: does the kill condition depend on what the inputs are
/// connected to?
///
/// A dosed full-frame cell is the murder weapon in its most reliable form, and every
/// one ever recorded ran the shorted mux. This walks the same cell across all three
/// input configurations at both pacings. The shorted rungs are the control and are
/// expected to die, which is what ties this run to the run 33-38 baseline; if the
/// test-signal or electrode rungs live, the fault is not readout alone and the
/// production input path is unexplored territory.
fn run_mux_discriminator_family(chip: &mut Ads1298) -> Result<Vec<CellReport>> {
    let mut reports = Vec::new();
    for inputs in [INPUT_SHORTED, INPUT_TEST_SIGNAL, INPUT_ELECTRODE] {
        for interval_us in [2_000, DATA_READY_INTERVAL_US] {
            for repeat in 1..=MUX_DISCRIMINATOR_REPEATS {
                reports.push(run_dosed_cell_labeled(
                    chip,
                    interval_us,
                    Dose::FrameContinuous,
                    inputs,
                    format!(
                        "mux discriminator: {}, full frame per {interval_us} us, run {repeat}",
                        inputs.label
                    ),
                )?);
            }
        }
    }
    Ok(reports)
}

/// Family 1: does a faster SCLK buy survival by shortening the burst in time?
///
/// Runs 32-36 measured the opposite end of this axis — a slower SCLK dies sooner,
/// because the transaction is longer — so the extrapolation says fast enough might
/// clear it. Each rung is gated first, since the ESP32-S3 drives these clocks down
/// jumper wires with no ground return in the SPI header, and a bus that has stopped
/// working looks exactly like a chip that has stopped converting.
fn run_clock_ladder_family(
    chip: &mut Ads1298,
    bus: &Arc<SpiDriver<'static>>,
) -> Result<Vec<CellReport>> {
    let mut reports = Vec::new();
    for rate_hz in CANDIDATE_CLOCK_RATES_HZ {
        let megahertz = rate_hz / 1_000_000;
        // Reclocking is a fresh `SpiDeviceDriver` on the shared bus: a device driver
        // fixes its baud rate at construction, which is why `Ads1298` holds the bus
        // behind an `Arc` and takes a new device through `set_spi`. CS is a plain GPIO
        // inside the driver, so nothing about the swap disturbs the chip.
        chip.set_spi(spi_device(bus, rate_hz)?);

        let gate = run_signal_integrity_gate(chip)?;
        let failures = gate.identity_failures + gate.scratch_failures;
        let label = format!("clock ladder: SCLK {megahertz} MHz, gate");
        if failures > 0 {
            warn!(
                "SCLK {megahertz} MHz: {} of {SIGNAL_INTEGRITY_ROUNDS} ID reads and {} of {SIGNAL_INTEGRITY_ROUNDS} scratch readbacks wrong -- SI-FAIL, skipping this rung's dosed cells",
                gate.identity_failures, gate.scratch_failures
            );
        } else {
            info!("SCLK {megahertz} MHz: gate clean, {SIGNAL_INTEGRITY_ROUNDS} ID reads and {SIGNAL_INTEGRITY_ROUNDS} scratch readbacks");
        }
        reports.push(CellReport {
            label,
            frames: SIGNAL_INTEGRITY_ROUNDS * 2,
            died_at_ms: None,
            config_held: failures == 0,
            signal_integrity_failures: Some(failures),
        });
        if failures > 0 {
            continue;
        }

        reports.push(run_dosed_cell_labeled(
            chip,
            2_000,
            Dose::FrameContinuous,
            INPUT_SHORTED,
            format!("clock ladder: SCLK {megahertz} MHz, full frame per 2000 us"),
        )?);
        reports.push(run_dosed_cell_labeled(
            chip,
            DATA_READY_INTERVAL_US,
            Dose::FrameContinuous,
            INPUT_SHORTED,
            format!(
                "clock ladder: SCLK {megahertz} MHz, full frame per {DATA_READY_INTERVAL_US} us (per DRDY)"
            ),
        )?);
    }
    // Families 2 and 3 belong at the bench's chosen clock, not at whatever the ladder
    // left behind.
    chip.set_spi(spi_device(bus, SPI_BAUD_RATE_HZ)?);
    Ok(reports)
}

/// What the signal-integrity gate found at one SCLK.
struct SignalIntegrityGate {
    identity_failures: u32,
    scratch_failures: u32,
}

/// Reads ID and writes-then-reads a scratch register [`SIGNAL_INTEGRITY_ROUNDS`] times
/// against an idle chip. Idle is the point: SPI against a chip that is not converting
/// has been harmless on every run of this campaign, so anything wrong here is the bus.
/// The write half matters as much as the read — MOSI and MISO fail at different
/// clocks, and an ID read only exercises one of them.
fn run_signal_integrity_gate(chip: &mut Ads1298) -> Result<SignalIntegrityGate> {
    chip.stop_conversion()?;
    chip.power_up()?;
    let mut identity_failures = 0;
    let mut scratch_failures = 0;
    for round in 0..SIGNAL_INTEGRITY_ROUNDS {
        if !matches!(chip.read_register(REG_ID), Ok(ads1298::DEVICE_ID)) {
            identity_failures += 1;
        }
        let pattern = SIGNAL_INTEGRITY_PATTERNS[round as usize % SIGNAL_INTEGRITY_PATTERNS.len()];
        let wrote = chip.write_register(REG_RLD_SENSP, pattern).is_ok();
        if !wrote || chip.read_register(REG_RLD_SENSP).ok() != Some(pattern) {
            scratch_failures += 1;
        }
    }
    // Back to the reset value, so the dosed cell that follows starts where every other
    // cell in this harness starts.
    drop(chip.write_register(REG_RLD_SENSP, 0x00));
    Ok(SignalIntegrityGate {
        identity_failures,
        scratch_failures,
    })
}

/// The partial-frame ladder: how short does the burst have to be?
///
/// Runs 38a/38b put the threshold between 24 and 72 data bits, with the 72-bit rung
/// surviving one time in four — a boundary, not a cliff. This round fills the untested
/// 48-bit rung and repeats the boundary, both at the pacing a real readout would use.
/// A survivor here is a usable product: status plus one channel per DRDY is a
/// single-channel front end, which is exactly what already works deterministically.
fn run_partial_frame_family(chip: &mut Ads1298) -> Result<Vec<CellReport>> {
    let mut reports = Vec::new();
    for (bytes, interval_us, repeats) in PARTIAL_FRAME_RUNGS {
        let bits = bytes * 8;
        for repeat in 1..=repeats {
            reports.push(run_dosed_cell_labeled(
                chip,
                interval_us,
                Dose::FramePartial(bytes),
                INPUT_SHORTED,
                format!(
                    "partial frame: {bytes} bytes, {bits} data bits, per {interval_us} us, run {repeat}"
                ),
            )?);
        }
    }
    Ok(reports)
}

/// One sample-and-stop rung.
struct SampleAndStopReport {
    label: String,
    cycles: u32,
    cycle_rate_hz: u32,
    died_at_ms: Option<u64>,
    config_held: bool,
    channel_mean: [f64; CHANNELS],
    channel_sigma: [f64; CHANNELS],
    channel_peak_to_peak: [i64; CHANNELS],
}

impl SampleAndStopReport {
    /// The survival half of the rung, for the combined matrix. `frames` is cycles
    /// completed; the data quality lives in the settling table instead.
    fn as_cell_report(&self) -> CellReport {
        CellReport {
            label: self.label.clone(),
            frames: self.cycles,
            died_at_ms: self.died_at_ms,
            config_held: self.config_held,
            signal_integrity_failures: None,
        }
    }
}

/// Stop overlapping readout with conversion at all — kept, with its read path known to
/// be invalid.
///
/// Assert START from the pin, count `window_edges` DRDY falling edges with the bus
/// completely silent, drop START, wait for the conversion in flight to finish, and only
/// then clock out one frame. The survival half of that works and is worth keeping: runs
/// 38a/38b split cleanly by input mux here, which is what the discriminator family now
/// chases.
///
/// The data half does not work, and the datasheet says why. SBAS459K §9.5.2.8: "To
/// retrieve data from the device after RDATA command is issued, make sure that either
/// the START pin is high or the START command is issued." This cell reads with START
/// already low, so retrieval is not enabled and the frame it clocks out is not the k-th
/// conversion — which is exactly what the bench saw, a 15,000-count square wave
/// reduced to a constant 20 counts. The per-channel statistics below are therefore
/// readback artefacts, not measurements of the input; [`run_single_shot_cell`] is the
/// same idea done in a way the chip supports. They stay printed so the two are
/// comparable side by side.
fn run_sample_and_stop_cell(
    chip: &mut Ads1298,
    window_edges: u32,
    inputs: InputConfiguration,
) -> Result<SampleAndStopReport> {
    let label = format!(
        "sample-and-stop: k={window_edges} edges, {}",
        inputs.window_label
    );
    info!("cell '{label}'");
    reset_and_configure_inputs(chip, inputs)?;

    let started_us = now_us();
    let mut cycles = 0u32;
    let mut samples = 0u32;
    let mut died_at_ms = None;
    let mut accumulator = ChannelStatistics::default();

    while (now_us() - started_us) / 1000 < SAMPLE_AND_STOP_DURATION_MS {
        chip.start_conversion()?;
        let mut edges_seen = 0;
        while edges_seen < window_edges {
            if wait_data_ready_falling_edge_within(chip, SAMPLE_AND_STOP_EDGE_TIMEOUT_US).is_err() {
                break;
            }
            edges_seen += 1;
        }
        chip.stop_conversion()?;
        if edges_seen < window_edges {
            // Edges stopped arriving: the chip died mid-sweep, which is the outcome
            // this candidate is supposed to make impossible.
            died_at_ms = Some((now_us() - started_us) / 1000);
            break;
        }
        wait_for_conversions_to_stop(chip);
        if let Ok(frame) = chip.read_frame() {
            if frame.marker_ok() {
                samples += 1;
                accumulator.record(&frame);
            }
        }
        cycles += 1;
    }
    chip.stop_conversion()?;

    let elapsed_us = (now_us() - started_us).max(1);
    let cycle_rate_hz = (u64::from(cycles) * 1_000_000 / elapsed_us) as u32;
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == CONFIG1)
        .unwrap_or(false);
    info!("  sample-and-stop: {cycles} cycles, {samples} frames kept, {cycle_rate_hz} Hz");
    Ok(accumulator.into_report(
        label,
        cycles,
        cycle_rate_hz,
        died_at_ms,
        config_held,
        samples,
    ))
}

/// Per-channel running sums for a windowed cell.
struct ChannelStatistics {
    sum: [i64; CHANNELS],
    sum_of_squares: [i128; CHANNELS],
    minimum: [i32; CHANNELS],
    maximum: [i32; CHANNELS],
}

impl Default for ChannelStatistics {
    fn default() -> Self {
        Self {
            sum: [0; CHANNELS],
            sum_of_squares: [0; CHANNELS],
            minimum: [i32::MAX; CHANNELS],
            maximum: [i32::MIN; CHANNELS],
        }
    }
}

impl ChannelStatistics {
    fn record(&mut self, frame: &Frame) {
        for (index, &code) in frame.channels.iter().enumerate() {
            self.sum[index] += i64::from(code);
            self.sum_of_squares[index] += i128::from(code) * i128::from(code);
            self.minimum[index] = self.minimum[index].min(code);
            self.maximum[index] = self.maximum[index].max(code);
        }
    }

    fn into_report(
        self,
        label: String,
        cycles: u32,
        cycle_rate_hz: u32,
        died_at_ms: Option<u64>,
        config_held: bool,
        samples: u32,
    ) -> SampleAndStopReport {
        let mut channel_mean = [0f64; CHANNELS];
        let mut channel_sigma = [0f64; CHANNELS];
        let mut channel_peak_to_peak = [0i64; CHANNELS];
        let count = f64::from(samples.max(1));
        for index in 0..CHANNELS {
            let mean = self.sum[index] as f64 / count;
            let mean_of_squares = self.sum_of_squares[index] as f64 / count;
            channel_mean[index] = mean;
            // Variance as E[x^2] - E[x]^2. The codes are 24-bit and the sums stay exact
            // in i64/i128, so the subtraction only loses precision once, at the end.
            channel_sigma[index] = (mean_of_squares - mean * mean).max(0.0).sqrt();
            channel_peak_to_peak[index] = if samples == 0 {
                0
            } else {
                i64::from(self.maximum[index]) - i64::from(self.minimum[index])
            };
        }
        SampleAndStopReport {
            label,
            cycles,
            cycle_rate_hz,
            died_at_ms,
            config_held,
            channel_mean,
            channel_sigma,
            channel_peak_to_peak,
        }
    }
}

/// Waits out the conversion still in flight when START drops. The chip finishes it and
/// pulses DRDY once more, and a frame read that lands inside that pulse is a read
/// during conversion — the one thing this cell exists to avoid. Returns once DRDY has
/// held still for two conversion periods, or after a bounded wait if it never settles.
fn wait_for_conversions_to_stop(chip: &Ads1298) {
    let quiet_us = DATA_READY_INTERVAL_US * 2;
    let deadline_us = now_us() + quiet_us * 4;
    let mut last_change_us = now_us();
    let mut was_low = chip.data_ready();
    while now_us() - last_change_us < quiet_us {
        let low = chip.data_ready();
        if low != was_low {
            was_low = low;
            last_change_us = now_us();
        }
        if now_us() > deadline_us {
            return;
        }
    }
}

/// Every sample-and-stop rung, shorted first and then on the internal test signal. Two
/// passes rather than interleaved so a log reads as two curves against k.
fn run_sample_and_stop_family(chip: &mut Ads1298) -> Result<Vec<SampleAndStopReport>> {
    warn!(
        "sample-and-stop cells read with START already low, which SBAS459K 9.5.2.8 does not \
         enable for RDATA -- their survival is data, their per-channel numbers are not"
    );
    let mut reports = Vec::new();
    for inputs in [INPUT_SHORTED, INPUT_TEST_SIGNAL] {
        for window_edges in SAMPLE_AND_STOP_RUNGS {
            reports.push(run_sample_and_stop_cell(chip, window_edges, inputs)?);
        }
    }
    Ok(reports)
}

/// The windowed-conversion candidate, done the way the chip supports it.
///
/// Single-shot mode (CONFIG4 bit 3, SBAS459K §9.4.1.4): taking the START pin high runs
/// exactly one conversion, after which "DRDY goes low and further conversions are
/// stopped" on the chip's own initiative. START then stays high while the frame is
/// clocked out, which is what §9.5.2.8 requires for RDATA to retrieve anything, and the
/// chip is not converting during that read — the safe half of the lethal pair. Pulsing
/// START low and back high starts the next conversion.
///
/// This is what the sample-and-stop cells were reaching for. They deasserted START
/// before reading, which stops conversions and disables retrieval in the same stroke;
/// single-shot mode gets the first without the second, because the chip stops itself.
///
/// Settling is free here rather than swept. tSETTLE at DR = 100 in high-resolution mode
/// is 4616 tCLK, 2.25 ms at 2.048 MHz (table 12), and the first DRDY after START does
/// not arrive until it has elapsed — so every single-shot sample is fully settled by
/// construction, and the ceiling on rate is 1/tSETTLE, about 440 Hz before read
/// overhead. The proof that the read path works is the test-signal rung: it has to show
/// the square wave at its full amplitude, thousands of counts rather than twenty.
fn run_single_shot_cell(
    chip: &mut Ads1298,
    inputs: InputConfiguration,
    repeat: u32,
) -> Result<SampleAndStopReport> {
    let label = format!("single-shot: {}, run {repeat}", inputs.window_label);
    info!("cell '{label}'");
    let mut writes = reset_and_configure_inputs(chip, inputs)?;
    writes.push((REG_CONFIG4, CONFIG4_SINGLE_SHOT));
    apply(chip, &writes)?;
    // Entering single-shot mode wants the START signal pulsed (SBAS459K §9.4.1.4).
    // START has been low since the reset, so the first assertion in the loop is it.

    let started_us = now_us();
    let mut cycles = 0u32;
    let mut samples = 0u32;
    let mut died_at_ms = None;
    let mut accumulator = ChannelStatistics::default();

    while (now_us() - started_us) / 1000 < SAMPLE_AND_STOP_DURATION_MS {
        chip.start_conversion()?;
        if wait_data_ready_falling_edge_within(chip, SAMPLE_AND_STOP_EDGE_TIMEOUT_US).is_err() {
            died_at_ms = Some((now_us() - started_us) / 1000);
            chip.stop_conversion()?;
            break;
        }
        // DRDY is low, the conversion is over and the chip has stopped itself. START is
        // still high, so this read is both legal and outside any conversion.
        let frame = chip.read_frame();
        chip.stop_conversion()?;
        Ets::delay_us(START_PULSE_LOW_US);
        if let Ok(frame) = frame {
            if frame.marker_ok() {
                samples += 1;
                accumulator.record(&frame);
            }
        }
        cycles += 1;
    }
    chip.stop_conversion()?;

    let elapsed_us = (now_us() - started_us).max(1);
    let config_held = chip
        .read_register(REG_CONFIG1)
        .map(|read| read == CONFIG1)
        .unwrap_or(false);
    info!("  single-shot: {cycles} cycles, {samples} frames kept");
    Ok(accumulator.into_report(
        label,
        cycles,
        (u64::from(cycles) * 1_000_000 / elapsed_us) as u32,
        died_at_ms,
        config_held,
        samples,
    ))
}

/// Both single-shot input configurations, repeated. The shorted rung is the noise floor
/// and the survival control; the test-signal rung is the proof that the read path
/// returns the conversion it is supposed to.
fn run_single_shot_family(chip: &mut Ads1298) -> Result<Vec<SampleAndStopReport>> {
    let mut reports = Vec::new();
    for inputs in [INPUT_SHORTED, INPUT_TEST_SIGNAL] {
        for repeat in 1..=SINGLE_SHOT_REPEATS {
            reports.push(run_single_shot_cell(chip, inputs, repeat)?);
        }
    }
    Ok(reports)
}

/// The settling table. Each rung is scored against the last rung of its own group — the
/// largest k for a sample-and-stop group, the last repeat for a single-shot group — so
/// the ratios read as "how far from the most settled reading available". On the test
/// signal the figure of merit is peak-to-peak amplitude, which an unsettled filter
/// attenuates; on shorted inputs it is sigma, which an unsettled filter inflates.
///
/// For the single-shot groups the ratios are a reproducibility check rather than a
/// settling measurement, because every single-shot conversion is settled by
/// construction. There the number that matters is the absolute test-signal amplitude.
fn print_sample_and_stop_summary(groups: &[&[SampleAndStopReport]]) {
    info!("=== sample-and-stop settling ===");
    for group in groups {
        let Some(baseline) = group.last() else {
            continue;
        };
        let baseline_sigma = mean_of(&baseline.channel_sigma);
        let baseline_amplitude = mean_of_integers(&baseline.channel_peak_to_peak);
        for report in group.iter() {
            info!(
                "SAMPLE-AND-STOP  {}: {} cycles, {} Hz, {}, config {}",
                report.label,
                report.cycles,
                report.cycle_rate_hz,
                match report.died_at_ms {
                    None => "SURVIVED".to_string(),
                    Some(at) => format!("DIED at {at} ms"),
                },
                if report.config_held { "HELD" } else { "LOST" },
            );
            info!(
                "SAMPLE-AND-STOP  {}: mean {}",
                report.label,
                format_codes(&report.channel_mean)
            );
            info!(
                "SAMPLE-AND-STOP  {}: sigma {}",
                report.label,
                format_codes(&report.channel_sigma)
            );
            info!(
                "SAMPLE-AND-STOP  {}: peak-to-peak {:?}",
                report.label, report.channel_peak_to_peak
            );
            let sigma = mean_of(&report.channel_sigma);
            let amplitude = mean_of_integers(&report.channel_peak_to_peak);
            info!(
                "SETTLING  {}: sigma {:.0} ({:.2}x baseline), amplitude {:.0} ({:.2}x baseline)",
                report.label,
                sigma,
                ratio(sigma, baseline_sigma),
                amplitude,
                ratio(amplitude, baseline_amplitude),
            );
        }
    }
}

fn mean_of(values: &[f64; CHANNELS]) -> f64 {
    values.iter().sum::<f64>() / CHANNELS as f64
}

fn mean_of_integers(values: &[i64; CHANNELS]) -> f64 {
    values.iter().sum::<i64>() as f64 / CHANNELS as f64
}

/// Zero baselines happen when a rung produced no frames at all, and a printed `inf`
/// would be read as a measurement rather than as missing data.
fn ratio(value: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        0.0
    } else {
        value / baseline
    }
}

fn format_codes(values: &[f64; CHANNELS]) -> String {
    let rendered: Vec<String> = values.iter().map(|value| format!("{value:.0}")).collect();
    format!("[{}]", rendered.join(", "))
}

/// The partial operating mode candidate: accept that a conversion run dies, and
/// measure how much coverage fast recovery buys. On each death: a bare RESET pulse,
/// SDATAC, rewrite, START — no cold-start settling — and back to streaming. The
/// yield number (frames achieved / frames possible) is what the ML side needs to know
/// to decide whether gap-aware training on this hardware is viable.
fn run_recovery_cell(
    chip: &mut Ads1298,
    duration_ms: u64,
    channel_value: u8,
    config2: u8,
) -> Result<()> {
    info!("cell 'fast-recovery acquisition, 8 channels, CHnSET {channel_value:#04x}, CONFIG2 {config2:#04x}, {duration_ms} ms'");
    let mut writes = reset_and_configure(chip, CHANNELS, CONFIG1, CONFIG3, Some(GPIO_OUTPUTS_LOW))?;
    for entry in writes.iter_mut() {
        if entry.0 >= REG_CH1SET && entry.0 < REG_CH1SET + CHANNELS as u8 {
            entry.1 = channel_value;
        }
        if entry.0 == REG_CONFIG2 {
            entry.1 = config2;
        }
    }
    apply(chip, &writes)?;
    chip.start_conversion()?;
    let started_us = now_us();
    let mut frames = 0u32;
    let mut deaths = 0u32;
    let mut recovery_time_us = 0u64;
    let mut frames_since_audit = 0u32;
    let mut channel_minimum = [i32::MAX; CHANNELS];
    let mut channel_maximum = [i32::MIN; CHANNELS];
    // Post-start settling: frames right after START (or a warm recovery) carry the
    // reference and filter transient, which would dominate min/max and mask the
    // per-channel operating point the pass exists to measure.
    const SETTLE_DISCARD_FRAMES: u32 = 8;
    let mut discard_remaining = SETTLE_DISCARD_FRAMES;
    let mut channel_sum = [0i64; CHANNELS];
    let mut channel_squares = [0i128; CHANNELS];
    let mut settled_frames = 0u64;
    while (now_us() - started_us) / 1000 < duration_ms {
        // A frame only counts when its marker is intact, and every 64 frames the
        // configuration is re-read — a silent revert produces DRDY and markers at the
        // wrong rate, so without this gate the yield number counts garbage.
        let mut dead = false;
        match wait_data_ready_falling_edge_within(chip, 10_000) {
            Ok(()) => match chip.read_frame() {
                Ok(frame) if frame.marker_ok() => {
                    frames += 1;
                    if discard_remaining > 0 {
                        discard_remaining -= 1;
                    } else {
                        settled_frames += 1;
                        for (index, &code) in frame.channels.iter().enumerate() {
                            channel_minimum[index] = channel_minimum[index].min(code);
                            channel_maximum[index] = channel_maximum[index].max(code);
                            channel_sum[index] += i64::from(code);
                            channel_squares[index] += i128::from(code) * i128::from(code);
                        }
                    }
                    frames_since_audit += 1;
                    if frames_since_audit >= 64 {
                        frames_since_audit = 0;
                        if chip.read_register(REG_CONFIG1).ok() != Some(CONFIG1) {
                            frames = frames.saturating_sub(64);
                            dead = true;
                        }
                    }
                }
                _ => {}
            },
            Err(_) => dead = true,
        }
        if dead {
            deaths += 1;
            let recovery_started_us = now_us();
            chip.stop_conversion()?;
            chip.reset_pulse()?;
            apply(chip, &writes)?;
            chip.start_conversion()?;
            recovery_time_us += now_us() - recovery_started_us;
            frames_since_audit = 0;
            discard_remaining = SETTLE_DISCARD_FRAMES;
        }
    }
    chip.stop_conversion()?;
    let possible = duration_ms * u64::from(samples_per_second()) / 1000;
    info!(
        "RECOVERY  8 channels {duration_ms} ms: {frames} frames of {possible} possible ({}%), {deaths} deaths, {} ms recovering",
        frames as u64 * 100 / possible.max(1),
        recovery_time_us / 1000,
    );
    let spreads: Vec<i64> = (0..CHANNELS)
        .map(|index| i64::from(channel_maximum[index]) - i64::from(channel_minimum[index]))
        .collect();
    info!("RECOVERY  per-channel peak-to-peak codes: {spreads:?}");
    info!("RECOVERY  settled frames: {settled_frames}");
    for index in 0..CHANNELS {
        if settled_frames == 0 {
            break;
        }
        let n = settled_frames as f64;
        let mean = channel_sum[index] as f64 / n;
        let variance = (channel_squares[index] as f64 / n) - mean * mean;
        let sigma = variance.max(0.0).sqrt();
        info!(
            "RECOVERY  channel {index}: mean {mean:.0}, sigma {sigma:.0}, min {}, max {}",
            channel_minimum[index], channel_maximum[index],
        );
    }
    Ok(())
}

/// The eleven registers a stage writes: CONFIG1-3 and the eight CHnSET, with the
/// channel value chosen per channel. Everything omitted — LOFF, RLD_SENSP/N,
/// LOFF_SENSP/N, LOFF_FLIP, GPIO, PACE, RESP, CONFIG4, WCT1/2 — already holds the
/// value this harness wants after reset.
fn planned_writes(channel_value: impl Fn(usize) -> u8) -> Vec<(u8, u8)> {
    let mut writes = vec![
        (REG_CONFIG1, CONFIG1),
        (REG_CONFIG2, CONFIG2),
        (REG_CONFIG3, CONFIG3),
    ];
    for channel in 0..CHANNELS {
        writes.push((REG_CH1SET + channel as u8, channel_value(channel)));
    }
    writes
}

fn normal_channel_value(channel: usize) -> u8 {
    match TEST_SIGNAL_CHANNEL {
        Some(driven) if driven == channel => CHANNEL_TEST_SIGNAL,
        Some(_) => CHANNEL_SHORTED,
        None => CHANNEL_NORMAL,
    }
}

/// Brings one channel amplifier up by duty-cycling its PD bit with linearly rising
/// on-time, approximating a current ramp the analog supply can follow. With
/// [`CHANNEL_SOFT_START_MS`] zero this is a single write, the pre-run-14 behaviour.
fn soft_start_channel(chip: &mut Ads1298, address: u8, powered_value: u8) -> Result<()> {
    if CHANNEL_SOFT_START_MS == 0 {
        return chip.write_register(address, powered_value);
    }
    const STEPS: u64 = 50;
    const PERIOD_US: u64 = 2_000;
    let cycles_per_step = (CHANNEL_SOFT_START_MS * 1000 / PERIOD_US / STEPS).max(1);
    for step in 1..STEPS {
        let on_us = (PERIOD_US * step / STEPS) as u32;
        for _ in 0..cycles_per_step {
            chip.write_register(address, powered_value)?;
            Ets::delay_us(on_us);
            chip.write_register(address, CHANNEL_POWERED_DOWN)?;
            Ets::delay_us((PERIOD_US as u32).saturating_sub(on_us));
        }
    }
    chip.write_register(address, powered_value)
}

fn apply(chip: &mut Ads1298, writes: &[(u8, u8)]) -> Result<()> {
    for &(address, value) in writes {
        chip.write_register(address, value)?;
    }
    Ok(())
}

/// Reads back every register in `writes`.
fn verify(chip: &mut Ads1298, writes: &[(u8, u8)]) -> Result<()> {
    let mut mismatches = 0;
    for &(address, written) in writes {
        let read = chip.read_register(address)?;
        if read != written {
            mismatches += 1;
            warn!(
                "{}: wrote {written:#04x}, reads {read:#04x} -- the write did not take",
                REGISTER_NAMES[address as usize]
            );
        }
    }
    if mismatches == 0 {
        info!("all {} registers read back as written", writes.len());
    }
    Ok(())
}

fn log_registers(chip: &mut Ads1298, label: &str) -> Result<()> {
    let values = chip.read_all_registers()?;
    for row in 0..(REG_COUNT as usize).div_ceil(8) {
        let names: Vec<String> = (row * 8..((row + 1) * 8).min(REG_COUNT as usize))
            .map(|address| format!("{}={:02x}", REGISTER_NAMES[address], values[address]))
            .collect();
        info!("registers {label}: {}", names.join(" "));
    }
    Ok(())
}

/// Running frame statistics between summary lines.
#[derive(Default)]
struct Stats {
    frames: u32,
    last_frame_us: u64,
    period_min_us: u64,
    period_max_us: u64,
    period_sum_us: u64,
    bad_markers: u32,
}

impl Stats {
    fn record(&mut self, now_us: u64, frame: &Frame) {
        if self.last_frame_us != 0 {
            let period = now_us - self.last_frame_us;
            self.period_min_us = if self.frames == 0 {
                period
            } else {
                self.period_min_us.min(period)
            };
            self.period_max_us = self.period_max_us.max(period);
            self.period_sum_us += period;
            self.frames += 1;
        }
        self.last_frame_us = now_us;
        if !frame.marker_ok() {
            self.bad_markers += 1;
        }
    }

    fn summary(&mut self) -> String {
        let mean = self.period_sum_us / self.frames.max(1) as u64;
        let line = format!(
            "period min/mean/max {}/{}/{} us, bad markers {}",
            self.period_min_us, mean, self.period_max_us, self.bad_markers
        );
        *self = Stats {
            last_frame_us: self.last_frame_us,
            ..Stats::default()
        };
        line
    }
}

/// Runs one stage: streams frames if `converting`, audits the configuration
/// throughout, and recovers from every revert so the stage also measures whether the
/// fault repeats and at what interval. `None` for `duration_ms` runs until the plug is
/// pulled.
fn run_stage(
    chip: &mut Ads1298,
    name: &str,
    writes: &[(u8, u8)],
    converting: bool,
    duration_ms: Option<u64>,
) -> Result<()> {
    info!(
        "stage '{name}': {}",
        match duration_ms {
            Some(limit) => format!("{} s", limit / 1000),
            None => "until reset".to_string(),
        }
    );
    let started_us = now_us();
    let mut monitor = RevertMonitor::default();
    let mut stats = Stats::default();
    let mut frames: u32 = 0;
    let mut stalled = false;

    loop {
        if let Some(limit) = duration_ms {
            if (now_us() - started_us) / 1000 >= limit {
                break;
            }
        }

        if !converting {
            FreeRtos::delay_ms(IDLE_AUDIT_PERIOD_MS);
            monitor.check(chip, writes, name, started_us);
            continue;
        }

        match wait_data_ready_falling_edge(chip) {
            Ok(()) => {}
            Err(level) => {
                if !stalled {
                    warn!(
                        "DRDY stuck {level} for {} ms at frame {frames} ({} ms in); {}",
                        FRAME_TIMEOUT_US / 1000,
                        (now_us() - started_us) / 1000,
                        match level {
                            "low" => "a wire off a floating input reads this way, or a frame was never collected",
                            _ => "the chip is not converting; is START asserted and the DRDY wire on?",
                        }
                    );
                    // A stalled chip is exactly the one worth interrogating: ID correct
                    // with the configuration reverted is a chip that reset; ID wrong is
                    // a chip that is off the bus entirely.
                    match chip.read_register(REG_ID) {
                        Ok(ads1298::DEVICE_ID) => info!("while stalled: ID still reads correctly"),
                        Ok(other) => warn!("while stalled: ID reads {other:#04x}"),
                        Err(error) => warn!("while stalled: ID read failed: {error}"),
                    }
                    stalled = true;
                }
                monitor.check(chip, writes, name, started_us);
                continue;
            }
        }
        stalled = false;

        let frame = match chip.read_frame() {
            Ok(frame) => frame,
            Err(error) => {
                warn!("frame {frames} read failed: {error}");
                continue;
            }
        };
        frames += 1;
        stats.record(now_us(), &frame);

        if frames <= RAW_FRAMES_LOGGED {
            // With no lead-off sensing configured, bits 23:4 are exactly 0xC0000: the
            // fixed 1100 marker and sixteen zeroed lead-off flags. Bits 3:0 mirror the
            // live level on the chip's own GPIO pins (inputs after reset), so on a
            // board that leaves them floating the low nibble is meaningless.
            info!(
                "frame {frames}: status {:#08x}{} channels {:?}",
                frame.status,
                if frame.marker_ok() { "" } else { " MARKER BAD" },
                frame.channels
            );
        }

        if frames % AUDIT_INTERVAL == 0 {
            monitor.check(chip, writes, name, started_us);
        }

        if frames % REPORT_INTERVAL == 0 {
            info!("frame {frames}: {}", stats.summary());
        }
    }

    info!(
        "stage '{name}' ends: {frames} frames, {} reverts",
        monitor.reverts
    );
    Ok(())
}

/// Watches CONFIG1-3 against what the stage wrote, and rewrites the configuration
/// whenever they no longer match.
///
/// This is the measurement the whole harness exists for. RDATAC would make it
/// impossible: registers are unreadable while a chip streams, so the reversion could
/// only ever be inferred after the fact from the DRDY period. Recovering rather than
/// halting turns one capture into a measurement of the fault's period, not just its
/// onset — and if recovery holds, the harness is also the workaround.
#[derive(Default)]
struct RevertMonitor {
    reverts: u32,
    last_revert_us: u64,
}

impl RevertMonitor {
    fn check(
        &mut self,
        chip: &mut Ads1298,
        writes: &[(u8, u8)],
        stage: &str,
        stage_started_us: u64,
    ) {
        let audited = [REG_CONFIG1, REG_CONFIG2, REG_CONFIG3];
        let mut read_values = [0u8; 3];
        for (slot, address) in audited.into_iter().enumerate() {
            match chip.read_register(address) {
                Ok(value) => read_values[slot] = configuration_bits(address, value),
                Err(error) => {
                    warn!(
                        "audit read of {} failed: {error}",
                        REGISTER_NAMES[address as usize]
                    );
                    return;
                }
            }
        }
        let expected =
            audited.map(|address| configuration_bits(address, written_value(writes, address)));
        if read_values == expected {
            return;
        }

        self.reverts += 1;
        let elapsed_ms = (now_us() - stage_started_us) / 1000;
        let since_ms = if self.last_revert_us == 0 {
            elapsed_ms
        } else {
            (now_us() - self.last_revert_us) / 1000
        };
        self.last_revert_us = now_us();
        warn!(
            "REVERT {} in stage '{stage}' at {elapsed_ms} ms (+{since_ms} ms since last): \
             CONFIG1/2/3 read {:#04x}/{:#04x}/{:#04x}, expected {:#04x}/{:#04x}/{:#04x}; rewriting",
            self.reverts,
            read_values[0],
            read_values[1],
            read_values[2],
            expected[0],
            expected[1],
            expected[2],
        );
        if let Err(error) = apply(chip, writes) {
            warn!("rewrite after revert failed: {error}");
        }
    }
}

fn written_value(writes: &[(u8, u8)], address: u8) -> u8 {
    writes
        .iter()
        .find(|&&(candidate, _)| candidate == address)
        .map_or(0, |&(_, value)| value)
}

/// Drops the bits of a register the chip writes on its own, so the audit compares only
/// configuration. CONFIG3 bit 0 is RLD_STAT, a read-only lead-off status that follows
/// the electrode, not the register file; it read 0xC1 on this bench with the RLD
/// amplifier powered down.
fn configuration_bits(address: u8, value: u8) -> u8 {
    if address == REG_CONFIG3 {
        value & !0x01
    } else {
        value
    }
}

/// Spins until DRDY produces a falling edge: high first, then low. On timeout the
/// error names the level the pin was stuck at.
///
/// An edge rather than a level, because a level can lie. DRDY read constantly low on
/// one boot of this bench — a floating input with the wire off — and a level-triggered
/// loop obligingly free-ran at the SPI's own speed, reading unsynchronised garbage
/// that looked like an analog fault. Only the high-then-low sequence proves a
/// conversion actually completed. The chip holds DRDY high from the first SCLK of the
/// previous read until the next conversion, so waiting out the high phase costs
/// nothing.
///
/// A spin rather than an interrupt: at 500 SPS the loop has 2 ms of slack, nothing else
/// runs on this chip, and the task watchdog is off (see sdkconfig.defaults). An ISR,
/// a notification and a second thread are three things that can be wrong about a
/// measurement whose whole purpose is to be trusted.
fn wait_data_ready_falling_edge(chip: &Ads1298) -> Result<(), &'static str> {
    wait_data_ready_falling_edge_within(chip, FRAME_TIMEOUT_US)
}

/// As above with an explicit timeout, for the recovery loop where detection latency
/// is lost signal: DRDY arrives every 500 µs at the 2000 SPS the harness now runs,
/// so 10 ms is already twenty missed periods.
fn wait_data_ready_falling_edge_within(
    chip: &Ads1298,
    timeout_us: u64,
) -> Result<(), &'static str> {
    let deadline_us = now_us() + timeout_us;
    while chip.data_ready() {
        if now_us() > deadline_us {
            return Err("low");
        }
    }
    while !chip.data_ready() {
        if now_us() > deadline_us {
            return Err("high");
        }
    }
    Ok(())
}

fn now_us() -> u64 {
    unsafe { esp_idf_svc::sys::esp_timer_get_time() as u64 }
}

fn reset_reason() -> &'static str {
    match unsafe { esp_idf_svc::sys::esp_reset_reason() } {
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_POWERON => "power-on",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_SW => "software reset",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_PANIC => "panic",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_BROWNOUT => "brownout",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_USB => "usb reset",
        _ => "other",
    }
}
