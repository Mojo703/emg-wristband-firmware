//! ADS1298 bench harness: one chip, one thread, one text console.
//!
//! This exists because `opal-firmware` cannot answer a hardware question. That binary
//! brings up two chips on a shared bus, in two different register configurations, one
//! clocking the other, behind wifi, a model, an interrupt-driven sampling thread and a
//! CBOR link that has to come up before any log line is readable. Every one of those is
//! a variable in a fault that is currently unexplained.
//!
//! So: one board wired on its own and self-clocked, the eleven registers needed to
//! convert and nothing else, DRDY polled on the main task, frames clocked out with
//! RDATA so registers stay readable mid-stream, and logs straight out the USB cable.
//!
//! What it is looking for is a chip that reverts to its power-on defaults shortly after
//! conversions start. A run moves through three stages, each holding the chip in a
//! different amount of activity, and watches CONFIG1-3 throughout:
//!
//! 1. **idle, START low** — registers written, nothing converting. A revert here means
//!    conversion activity is not the trigger.
//! 2. **converting, channels powered down** — START high with every channel amplifier
//!    off, so the digital core and reference carry the only load.
//! 3. **converting, channels normal** — the full front end, until the cable is pulled.
//!
//! On every revert the harness logs the elapsed time and the interval since the last
//! one, rewrites the configuration, and keeps going — so one capture shows whether the
//! fault is one-shot or periodic, and which stage wakes it.

mod ads1298;

use ads1298::{
    Ads1298, Frame, CHANNELS, REGISTER_NAMES, REG_CH1SET, REG_CONFIG1, REG_CONFIG2, REG_CONFIG3,
    REG_COUNT, REG_ID,
};
use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, PinDriver, Pull};
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
const SPI_BAUD_RATE_HZ: u32 = 1_000_000;

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
const DATA_RATE_BITS: u8 = 0b110;

/// CONFIG1: HR=1, DAISY_EN=1 (per-chip readback), CLK_EN=0, DR as above.
///
/// CLK_EN is 0 because the boards are separately clocked here: the wire between the two
/// CLK pins is gone and this board's CLKSEL is tied high, so nothing is listening. Set
/// it to 1 only to give a scope a trigger — the CLK output stopping is the cleanest
/// edge the reversion produces.
const CONFIG1: u8 = 0b1110_0000 | DATA_RATE_BITS; // CLK_EN=1: oscillator out the CLK pin as a scope probe

/// CONFIG2: internal test-signal generator off, or bit 4 set to switch it on.
const CONFIG2: u8 = if TEST_SIGNAL_CHANNEL.is_some() {
    0x10
} else {
    0x00
};

/// CONFIG3: internal reference buffer on (bit 7), bit 6 reserved-one, right-leg drive
/// and the lead-off sense that rides on it entirely off.
const CONFIG3: u8 = 0xC0;

/// CHnSET: powered up, gain 6, reading its electrode pair.
const CHANNEL_NORMAL: u8 = 0x00;
/// CHnSET muxed to the internal test signal.
const CHANNEL_TEST_SIGNAL: u8 = 0x05;
/// CHnSET with the inputs shorted, so the channel reads its own noise floor.
const CHANNEL_SHORTED: u8 = 0x01;
/// CHnSET with the channel amplifier powered down and the inputs shorted, for the
/// stage that converts with no analog channel load at all.
const CHANNEL_POWERED_DOWN: u8 = 0x81;

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

// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== opal-bringup ({}) ===", reset_reason());
    info!("board {BOARD}, SPI {SPI_BAUD_RATE_HZ} Hz, CONFIG1 {CONFIG1:#04x} ({} SPS), test signal {TEST_SIGNAL_CHANNEL:?}",
        samples_per_second());

    let peripherals = Peripherals::take()?;
    let pins = peripherals.pins;

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
    let deadline_us = now_us() + FRAME_TIMEOUT_US;
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
