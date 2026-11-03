//! Bring-up check for the XIAO ESP32-S3 motherboard, run with no electrodes on.
//!
//! Each stage proves one layer of the board before the next relies on it: the
//! shared bus answers, both register sets hold, IC2's clock converts both chips
//! in lockstep at the product rate, shorted inputs are quiet, the internal test
//! signal reaches all sixteen channels in phase, the supplies and die
//! temperatures read sane, and the product configuration streams without losing
//! a frame. Electrode-mode and bias-drive readings are printed for reference;
//! with nothing on the inputs they have no pass condition.
//!
//! The pin map is the battery test's (`examples/battery-test/main.rs`).

#![allow(dead_code)]

#[path = "../src/adc/ads1298.rs"]
mod ads1298;
#[path = "../src/adc/channel.rs"]
mod channel;
#[path = "../src/adc/decode.rs"]
mod decode;
#[path = "../src/adc/registers.rs"]
mod registers;
#[path = "../src/adc/status.rs"]
mod status;

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, Input, Output, PinDriver, Pull};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{Operation, SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use esp_idf_svc::sys::{self, esp};
use log::info;

use ads1298::{device_config1, RIGHT_LEG_DRIVE_MODE, SAMPLE_RATE_HZ};
use decode::{parse_sample, FRAME_BYTES};
use registers::{Config1, Register, RegisterValue};

/// Read by `registers` for the data-rate arithmetic.
const INTERNAL_OSCILLATOR_HZ: u32 = 2_048_000;

const EXPECTED_DEVICE_ID: u8 = 0x92;
const COMMAND_RESET: u8 = 0x06;
const COMMAND_READ_DATA_CONTINUOUS: u8 = 0x10;
const COMMAND_STOP_READ_DATA_CONTINUOUS: u8 = 0x11;
const COMMAND_READ_REGISTER: u8 = 0x20;
const COMMAND_WRITE_REGISTER: u8 = 0x40;

const COMMAND_SPI_BAUD_RATE_HZ: u32 = 2_000_000;
const FRAME_SPI_BAUD_RATE_HZ: u32 = 8_000_000;
const COMMAND_DECODE_GAP_NANOSECONDS: u32 = 5_000;
const CHIP_SELECT_GAP_MICROSECONDS: u32 = 5;

const SUPPLY_SETTLE_MILLISECONDS: u32 = 2000;
const CLOCKED_CHIP_POWER_ON_MILLISECONDS: u32 = 250;
const REFERENCE_SETTLE_MILLISECONDS: u32 = 300;
const DATA_READY_TIMEOUT_MICROSECONDS: i64 = 20_000;
/// The digital filter's settling plus margin, dropped after every mode change.
const SETTLING_FRAMES: u32 = 200;

/// CHnSET fields (SBAS459K table 17).
const GAIN_SIX: u8 = 0b000 << 4;
const GAIN_ONE: u8 = 0b001 << 4;
const INPUT_ELECTRODE: u8 = 0b000;
const INPUT_SHORTED: u8 = 0b001;
const INPUT_RIGHT_LEG_DRIVE: u8 = 0b010;
const INPUT_SUPPLY: u8 = 0b011;
const INPUT_TEMPERATURE: u8 = 0b100;
const INPUT_TEST_SIGNAL: u8 = 0b101;
const CONFIG2_INTERNAL_TEST_SIGNAL: u8 = 1 << 4;
const CONFIG3_FOUR_VOLT_REFERENCE: u8 = 1 << 5;
const CONFIG3_RIGHT_LEG_DRIVE_MEASUREMENT: u8 = 1 << 4;

const FULL_SCALE_CODE: f64 = 8_388_607.0;
const RAIL_CODE: i32 = 8_300_000;
const CHANNELS: usize = 16;

/// DRDY falling edges per chip, counted in the interrupt.
static DATA_READY_EDGES: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chip {
    Ic1,
    Ic2,
}

impl Chip {
    const ALL: [Chip; 2] = [Chip::Ic1, Chip::Ic2];

    const fn index(self) -> usize {
        match self {
            Chip::Ic1 => 0,
            Chip::Ic2 => 1,
        }
    }

    /// The product recipe with this chip's clock role and `mode`'s overrides.
    fn configuration(self, mode: Mode) -> Vec<(Register, u8)> {
        let config1 = Config1 {
            output_clock_enabled: self == Chip::Ic2,
            ..device_config1()
        };
        RIGHT_LEG_DRIVE_MODE[self.index()]
            .expected_configuration()
            .into_iter()
            .map(|(register, value)| {
                let value = match register {
                    Register::Config1 => config1.to_byte(),
                    Register::Config2 => mode.config2(value),
                    Register::Config3 => mode.config3(value),
                    _ => match channel_settings_index(register) {
                        Some(channel) => mode.channel_settings(channel),
                        None => value,
                    },
                };
                (register, value)
            })
            .collect()
    }
}

fn channel_settings_index(register: Register) -> Option<usize> {
    match register {
        Register::Ch1Set => Some(0),
        Register::Ch2Set => Some(1),
        Register::Ch3Set => Some(2),
        Register::Ch4Set => Some(3),
        Register::Ch5Set => Some(4),
        Register::Ch6Set => Some(5),
        Register::Ch7Set => Some(6),
        Register::Ch8Set => Some(7),
        _ => None,
    }
}

fn channel_name(channel: usize) -> String {
    format!("IC{} CH{}", channel / 8 + 1, channel % 8 + 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Product,
    Shorted,
    TestSignal,
    Supply,
    Temperature,
    RightLegDriveMeasurement,
}

impl Mode {
    fn channel_settings(self, channel: usize) -> u8 {
        match self {
            Mode::Product => GAIN_SIX | INPUT_ELECTRODE,
            Mode::Shorted => GAIN_SIX | INPUT_SHORTED,
            Mode::TestSignal => GAIN_SIX | INPUT_TEST_SIGNAL,
            Mode::Supply => GAIN_ONE | INPUT_SUPPLY,
            Mode::Temperature => GAIN_ONE | INPUT_TEMPERATURE,
            Mode::RightLegDriveMeasurement if channel == 0 => GAIN_ONE | INPUT_RIGHT_LEG_DRIVE,
            Mode::RightLegDriveMeasurement => GAIN_SIX | INPUT_SHORTED,
        }
    }

    fn config2(self, product: u8) -> u8 {
        match self {
            Mode::TestSignal => product | CONFIG2_INTERNAL_TEST_SIGNAL,
            _ => product,
        }
    }

    /// The supply reading on channels 1, 2 and 5 to 8 sits near 2.5 V, past
    /// the 2.4 V reference, so that mode runs on the 4 V one.
    fn config3(self, product: u8) -> u8 {
        match self {
            Mode::Supply => product | CONFIG3_FOUR_VOLT_REFERENCE,
            Mode::RightLegDriveMeasurement => product | CONFIG3_RIGHT_LEG_DRIVE_MEASUREMENT,
            _ => product,
        }
    }

    fn microvolts_per_code(self, channel: usize) -> f64 {
        let reference_volts = match self {
            Mode::Supply => 4.0,
            _ => 2.4,
        };
        let gain = match self.channel_settings(channel % 8) & (0b111 << 4) {
            GAIN_ONE => 1.0,
            _ => 6.0,
        };
        reference_volts / gain / FULL_SCALE_CODE * 1e6
    }
}

type SpiDevice = SpiDeviceDriver<'static, Arc<SpiDriver<'static>>>;

struct Bus {
    command: SpiDevice,
    frame: SpiDevice,
    chip_select: [PinDriver<'static, Output>; 2],
}

impl Bus {
    fn selected<T>(
        &mut self,
        chip: Chip,
        transfer: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.chip_select[chip.index()].set_low()?;
        let result = transfer(self);
        Ets::delay_us(CHIP_SELECT_GAP_MICROSECONDS);
        self.chip_select[chip.index()].set_high()?;
        Ets::delay_us(CHIP_SELECT_GAP_MICROSECONDS);
        result
    }

    fn send_command(&mut self, chip: Chip, command: u8) -> Result<()> {
        self.selected(chip, |bus| Ok(bus.command.write(&[command])?))
    }

    fn write_register(&mut self, chip: Chip, register: Register, value: u8) -> Result<()> {
        self.selected(chip, |bus| {
            bus.command.transaction(&mut [
                Operation::Write(&[COMMAND_WRITE_REGISTER | register.addr()]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[0x00]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[value]),
            ])?;
            Ok(())
        })
    }

    fn read_register(&mut self, chip: Chip, register: Register) -> Result<u8> {
        self.selected(chip, |bus| {
            let mut received = [0u8; 1];
            bus.command.transaction(&mut [
                Operation::Write(&[COMMAND_READ_REGISTER | register.addr()]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[0x00]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Transfer(&mut received, &[0x00]),
            ])?;
            Ok(received[0])
        })
    }

    fn read_frame(&mut self, chip: Chip) -> Result<[u8; FRAME_BYTES]> {
        const DATA_IN_LOW: [u8; FRAME_BYTES] = [0u8; FRAME_BYTES];
        self.selected(chip, |bus| {
            let mut raw = [0u8; FRAME_BYTES];
            bus.frame.transfer(&mut raw, &DATA_IN_LOW)?;
            Ok(raw)
        })
    }

    /// Writes `mode`'s recipe and returns every register that reads back wrong.
    fn configure(&mut self, chip: Chip, mode: Mode) -> Result<Vec<String>> {
        let configuration = chip.configuration(mode);
        for &(register, value) in &configuration {
            self.write_register(chip, register, value)?;
        }
        let mut mismatches = Vec::new();
        for &(register, expected) in &configuration {
            let read = self.read_register(chip, register)?;
            // CONFIG3 bit 0 is the read-only RLD_STAT.
            let read = match register {
                Register::Config3 => read & !1,
                _ => read,
            };
            if read != expected {
                mismatches.push(format!(
                    "{chip:?} {} wrote {expected:#04x} reads {read:#04x}",
                    register.name()
                ));
            }
        }
        Ok(mismatches)
    }
}

pub struct Wiring {
    sclk: AnyOutputPin<'static>,
    data_in: AnyOutputPin<'static>,
    data_out: AnyInputPin<'static>,
    power_down: AnyOutputPin<'static>,
    reset: AnyOutputPin<'static>,
    start: AnyOutputPin<'static>,
    chip_select: [AnyOutputPin<'static>; 2],
    data_ready: [AnyInputPin<'static>; 2],
}

struct Board {
    bus: Bus,
    power_down: PinDriver<'static, Output>,
    reset: PinDriver<'static, Output>,
    start: PinDriver<'static, Output>,
    data_ready: [PinDriver<'static, Input>; 2],
}

#[derive(Default)]
struct Capture {
    missed: u32,
    bad_status: [u32; 2],
    read_errors: u32,
    ic1_edges: u32,
    ic2_edges: u32,
    elapsed_microseconds: i64,
}

impl Capture {
    fn clean(&self) -> bool {
        self.missed == 0 && self.bad_status == [0, 0] && self.read_errors == 0
    }

    fn summary(&self) -> String {
        format!(
            "{} IC1 / {} IC2 DRDY edges, {} missed, bad status {}/{}, {} read errors",
            self.ic1_edges,
            self.ic2_edges,
            self.missed,
            self.bad_status[0],
            self.bad_status[1],
            self.read_errors
        )
    }
}

fn now_microseconds() -> i64 {
    unsafe { sys::esp_timer_get_time() }
}

impl Board {
    fn new<SPI: SpiAnyPins + 'static>(spi: SPI, wiring: Wiring) -> Result<Self> {
        let [chip_select_ic1, chip_select_ic2] = wiring.chip_select;
        let mut chip_select = [
            PinDriver::output(chip_select_ic1)?,
            PinDriver::output(chip_select_ic2)?,
        ];
        for pin in &mut chip_select {
            pin.set_high()?;
        }
        let mut power_down = PinDriver::output(wiring.power_down)?;
        power_down.set_low()?;
        let mut reset = PinDriver::output(wiring.reset)?;
        reset.set_low()?;
        let mut start = PinDriver::output(wiring.start)?;
        start.set_low()?;
        let [data_ready_ic1, data_ready_ic2] = wiring.data_ready;
        let data_ready = [
            PinDriver::input(data_ready_ic1, Pull::Floating)?,
            PinDriver::input(data_ready_ic2, Pull::Floating)?,
        ];
        let bus = Arc::new(
            SpiDriver::new(
                spi,
                wiring.sclk,
                wiring.data_in,
                Some(wiring.data_out),
                &SpiDriverConfig::new(),
            )
            .context("SPI bus")?,
        );
        let config = |baud_rate_hz: u32| {
            SpiConfig::new()
                .baudrate(baud_rate_hz.Hz())
                .data_mode(MODE_1)
                .polling(true)
        };
        let none = Option::<AnyOutputPin>::None;
        let command = SpiDeviceDriver::new(bus.clone(), none, &config(COMMAND_SPI_BAUD_RATE_HZ))?;
        let none = Option::<AnyOutputPin>::None;
        let frame = SpiDeviceDriver::new(bus, none, &config(FRAME_SPI_BAUD_RATE_HZ))?;
        Ok(Self {
            bus: Bus {
                command,
                frame,
                chip_select,
            },
            power_down,
            reset,
            start,
            data_ready,
        })
    }

    fn power_up(&mut self) -> Result<()> {
        FreeRtos::delay_ms(5);
        self.power_down.set_high()?;
        self.reset.set_high()?;
        FreeRtos::delay_ms(SUPPLY_SETTLE_MILLISECONDS);
        Ok(())
    }

    /// The battery test's sequence: IC2 first, since the shared RESET clears
    /// its CLK_EN and so stops IC1's clock, then IC1 by opcode once clocked.
    fn identify(&mut self, report: &mut Report) -> Result<bool> {
        self.start.set_low()?;
        self.reset.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset.set_high()?;
        FreeRtos::delay_ms(1);

        self.bus
            .send_command(Chip::Ic2, COMMAND_STOP_READ_DATA_CONTINUOUS)?;
        FreeRtos::delay_ms(1);
        let ic2 = self.bus.read_register(Chip::Ic2, Register::Id)?;
        let ic1_unclocked = self.bus.read_register(Chip::Ic1, Register::Id)?;
        report.note(
            "IC1 before IC2 drives CLK",
            format!("ID reads {ic1_unclocked:#04x}"),
        );
        let config1 = Chip::Ic2.configuration(Mode::Product)[0].1;
        self.bus
            .write_register(Chip::Ic2, Register::Config1, config1)?;
        FreeRtos::delay_ms(CLOCKED_CHIP_POWER_ON_MILLISECONDS);

        self.bus.send_command(Chip::Ic1, COMMAND_RESET)?;
        FreeRtos::delay_ms(1);
        self.bus
            .send_command(Chip::Ic1, COMMAND_STOP_READ_DATA_CONTINUOUS)?;
        FreeRtos::delay_ms(1);
        let ic1 = self.bus.read_register(Chip::Ic1, Register::Id)?;

        let pass = ic1 == EXPECTED_DEVICE_ID && ic2 == EXPECTED_DEVICE_ID;
        report.record(
            "identity",
            pass,
            format!("IC1 {ic1:#04x}, IC2 {ic2:#04x}, expected {EXPECTED_DEVICE_ID:#04x}"),
        );
        Ok(pass)
    }

    /// Stops both chips, writes `mode` and restarts them on one START edge.
    fn enter(&mut self, mode: Mode) -> Result<Vec<String>> {
        self.start.set_low()?;
        for chip in Chip::ALL {
            self.bus
                .send_command(chip, COMMAND_STOP_READ_DATA_CONTINUOUS)?;
        }
        FreeRtos::delay_ms(1);
        let mut mismatches = self.bus.configure(Chip::Ic2, mode)?;
        mismatches.extend(self.bus.configure(Chip::Ic1, mode)?);
        if mode == Mode::TestSignal {
            // The generator's phase runs from the INT_TEST write, so the two
            // chips get that write back to back.
            let product = Chip::Ic1.configuration(Mode::Product);
            let config2 = |configuration: &[(Register, u8)]| {
                configuration
                    .iter()
                    .find(|(register, _)| *register == Register::Config2)
                    .map(|&(_, value)| value)
                    .unwrap_or_default()
            };
            let off = config2(&product);
            let on = config2(&Chip::Ic1.configuration(Mode::TestSignal));
            for value in [off, on] {
                for chip in [Chip::Ic2, Chip::Ic1] {
                    self.bus.write_register(chip, Register::Config2, value)?;
                }
            }
        }
        FreeRtos::delay_ms(REFERENCE_SETTLE_MILLISECONDS);
        for chip in Chip::ALL {
            self.bus.send_command(chip, COMMAND_READ_DATA_CONTINUOUS)?;
        }
        self.start.set_high()?;
        self.capture(SETTLING_FRAMES, |_, _| {})?;
        Ok(mismatches)
    }

    /// Reads one frame from each chip per IC1 DRDY edge. `each` gets the
    /// frame's edge index, so a missed frame leaves a gap rather than
    /// shifting time.
    fn capture(
        &mut self,
        frames: u32,
        mut each: impl FnMut(u32, &[i32; CHANNELS]),
    ) -> Result<Capture> {
        let mut capture = Capture::default();
        let ic1_start = DATA_READY_EDGES[0].load(Ordering::Relaxed);
        let ic2_start = DATA_READY_EDGES[1].load(Ordering::Relaxed);
        let mut handled = ic1_start;
        let started = now_microseconds();
        for _ in 0..frames {
            let waited_from = now_microseconds();
            let edges = loop {
                let edges = DATA_READY_EDGES[0].load(Ordering::Relaxed);
                if edges != handled {
                    break edges;
                }
                if now_microseconds() - waited_from > DATA_READY_TIMEOUT_MICROSECONDS {
                    bail!("IC1 DRDY stopped ({})", capture.summary());
                }
            };
            capture.missed += edges.wrapping_sub(handled) - 1;
            handled = edges;
            let mut values = [0i32; CHANNELS];
            let mut good = true;
            for chip in Chip::ALL {
                match self.bus.read_frame(chip) {
                    Ok(raw) => {
                        let sample = parse_sample(&raw);
                        if sample.status_word().is_none() {
                            capture.bad_status[chip.index()] += 1;
                            good = false;
                        }
                        values[chip.index() * 8..][..8].copy_from_slice(&sample.channels);
                    }
                    Err(_) => {
                        capture.read_errors += 1;
                        good = false;
                    }
                }
            }
            if good {
                each(handled.wrapping_sub(ic1_start), &values);
            }
        }
        capture.elapsed_microseconds = now_microseconds() - started;
        capture.ic1_edges = handled.wrapping_sub(ic1_start);
        capture.ic2_edges = DATA_READY_EDGES[1]
            .load(Ordering::Relaxed)
            .wrapping_sub(ic2_start);
        Ok(capture)
    }
}

/// Integer running statistics; the offset keeps the squares inside i64.
#[derive(Clone, Copy)]
struct Statistics {
    count: i64,
    offset: i64,
    sum: i64,
    sum_of_squares: i64,
    minimum: i32,
    maximum: i32,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            count: 0,
            offset: 0,
            sum: 0,
            sum_of_squares: 0,
            minimum: i32::MAX,
            maximum: i32::MIN,
        }
    }
}

impl Statistics {
    fn push(&mut self, value: i32) {
        if self.count == 0 {
            self.offset = value as i64;
        }
        let deviation = value as i64 - self.offset;
        self.count += 1;
        self.sum += deviation;
        self.sum_of_squares += deviation * deviation;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
    }

    fn mean(&self) -> f64 {
        self.offset as f64 + self.sum as f64 / self.count.max(1) as f64
    }

    fn standard_deviation(&self) -> f64 {
        let count = self.count.max(1) as f64;
        let mean = self.sum as f64 / count;
        (self.sum_of_squares as f64 / count - mean * mean)
            .max(0.0)
            .sqrt()
    }

    fn peak_to_peak(&self) -> i32 {
        self.maximum - self.minimum
    }

    fn railed(&self) -> bool {
        self.maximum >= RAIL_CODE || self.minimum <= -RAIL_CODE
    }
}

fn collect(board: &mut Board, frames: u32) -> Result<([Statistics; CHANNELS], Capture)> {
    let mut statistics = [Statistics::default(); CHANNELS];
    let capture = board.capture(frames, |_, values| {
        for (channel, &value) in values.iter().enumerate() {
            statistics[channel].push(value);
        }
    })?;
    Ok((statistics, capture))
}

struct Report {
    results: Vec<(&'static str, bool)>,
}

impl Report {
    fn record(&mut self, name: &'static str, pass: bool, detail: String) {
        let verdict = match pass {
            true => "PASS",
            false => "FAIL",
        };
        info!("{verdict} {name}: {detail}");
        self.results.push((name, pass));
    }

    fn note(&self, name: &str, detail: String) {
        info!("INFO {name}: {detail}");
    }

    fn configuration(&mut self, name: &'static str, mismatches: Vec<String>) {
        let detail = match mismatches.is_empty() {
            true => "every register reads back as written".to_string(),
            false => mismatches.join("; "),
        };
        self.record(name, mismatches.is_empty(), detail);
    }

    fn print_summary(&self) {
        let failed: Vec<_> = self
            .results
            .iter()
            .filter(|(_, pass)| !pass)
            .map(|(name, _)| *name)
            .collect();
        info!(
            "==== {} checks, {} failed ====",
            self.results.len(),
            failed.len()
        );
        if !failed.is_empty() {
            info!("failed: {}", failed.join(", "));
        }
    }
}

fn check_shorted(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::Shorted)?;
    report.configuration("shorted-input registers", mismatches);
    let (statistics, capture) = collect(board, 2 * SAMPLE_RATE_HZ)?;
    let mut worst_noise = 0f64;
    let mut worst_offset = 0f64;
    for (channel, statistic) in statistics.iter().enumerate() {
        let scale = Mode::Shorted.microvolts_per_code(channel);
        let offset = statistic.mean() * scale;
        let noise = statistic.standard_deviation() * scale;
        worst_noise = worst_noise.max(noise);
        worst_offset = worst_offset.max(offset.abs());
        info!(
            "  {}: offset {offset:+8.1} uV, noise {noise:5.2} uVrms, {:6.2} uVpp",
            channel_name(channel),
            statistic.peak_to_peak() as f64 * scale
        );
    }
    report.record("shorted-input capture", capture.clean(), capture.summary());
    report.record(
        "shorted-input noise",
        worst_noise < 3.0,
        format!("worst channel {worst_noise:.2} uVrms referred to input at gain 6 (limit 3, datasheet typical 1.1)"),
    );
    report.record(
        "shorted-input offset",
        worst_offset < 500.0,
        format!("worst channel {worst_offset:.1} uV (limit 500, the datasheet offset error)"),
    );
    Ok(())
}

const TEST_SIGNAL_FRAMES: u32 = 5 * SAMPLE_RATE_HZ;
const TRANSITIONS_KEPT: usize = 8;

fn check_test_signal(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::TestSignal)?;
    report.configuration("test-signal registers", mismatches);
    let (range, first) = collect(board, TEST_SIGNAL_FRAMES)?;

    let midpoints: [i64; CHANNELS] = core::array::from_fn(|channel| {
        (range[channel].minimum as i64 + range[channel].maximum as i64) / 2
    });
    let mut high = [Statistics::default(); CHANNELS];
    let mut low = [Statistics::default(); CHANNELS];
    let mut above = [None::<bool>; CHANNELS];
    let mut transitions = [[0u32; TRANSITIONS_KEPT]; CHANNELS];
    let mut transition_counts = [0usize; CHANNELS];
    let second = board.capture(TEST_SIGNAL_FRAMES, |index, values| {
        for (channel, &value) in values.iter().enumerate() {
            let is_above = value as i64 > midpoints[channel];
            match is_above {
                true => high[channel].push(value),
                false => low[channel].push(value),
            }
            if above[channel].is_some_and(|was| was != is_above) {
                let count = &mut transition_counts[channel];
                if *count < TRANSITIONS_KEPT {
                    transitions[channel][*count] = index;
                }
                *count += 1;
            }
            above[channel] = Some(is_above);
        }
    })?;

    let mut amplitudes = [0f64; CHANNELS];
    for channel in 0..CHANNELS {
        let scale = Mode::TestSignal.microvolts_per_code(channel);
        amplitudes[channel] = (high[channel].mean() - low[channel].mean()) * scale;
        let kept = transition_counts[channel].min(TRANSITIONS_KEPT);
        let half_period = match kept {
            0 | 1 => 0.0,
            _ => {
                (transitions[channel][kept - 1] - transitions[channel][0]) as f64
                    / (kept - 1) as f64
            }
        };
        info!(
            "  {}: step {:7.1} uV, {} transitions, half period {half_period:6.1} frames, first at {}",
            channel_name(channel),
            amplitudes[channel],
            transition_counts[channel],
            transitions[channel][0]
        );
    }
    report.record(
        "test-signal capture",
        first.clean() && second.clean(),
        format!(
            "range pass {}; level pass {}",
            first.summary(),
            second.summary()
        ),
    );

    let mut sorted = amplitudes;
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[CHANNELS / 2 - 1] + sorted[CHANNELS / 2]) / 2.0;
    let spread = amplitudes
        .iter()
        .map(|amplitude| (amplitude - median).abs() / median.abs().max(1.0))
        .fold(0f64, f64::max);
    report.record(
        "test-signal amplitude",
        median > 100.0 && spread < 0.05,
        format!(
            "median step {median:.1} uV, worst channel {:.1}% from median (limit 5%)",
            spread * 100.0
        ),
    );

    let reference = transitions[0];
    let reference_count = transition_counts[0].min(TRANSITIONS_KEPT);
    let misaligned: Vec<String> = (1..CHANNELS)
        .filter(|&channel| {
            let count = transition_counts[channel].min(TRANSITIONS_KEPT);
            count != reference_count
                || (0..count).any(|n| transitions[channel][n].abs_diff(reference[n]) > 1)
        })
        .map(channel_name)
        .collect();
    let expected_half_period =
        SAMPLE_RATE_HZ as f64 * (1u32 << 21) as f64 / INTERNAL_OSCILLATOR_HZ as f64 / 2.0;
    let observed_half_period = match reference_count {
        0 | 1 => 0.0,
        _ => (reference[reference_count - 1] - reference[0]) as f64 / (reference_count - 1) as f64,
    };
    report.record(
        "test-signal phase",
        misaligned.is_empty() && reference_count >= 4,
        match misaligned.is_empty() {
            true => format!(
                "all 16 channels switch on the same frame (within 1); half period {observed_half_period:.1} frames, expected {expected_half_period:.1}"
            ),
            false => format!("out of step with IC1 CH1: {}", misaligned.join(", ")),
        },
    );
    Ok(())
}

fn check_supplies(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::Supply)?;
    report.configuration("supply-measurement registers", mismatches);
    let (statistics, capture) = collect(board, SAMPLE_RATE_HZ / 2)?;
    let volts: [f64; CHANNELS] = core::array::from_fn(|channel| {
        statistics[channel].mean() * Mode::Supply.microvolts_per_code(channel) / 1e6
    });
    for (channel, value) in volts.iter().enumerate() {
        info!("  {}: {value:+.4} V", channel_name(channel));
    }
    report.record("supply capture", capture.clean(), capture.summary());
    for chip in Chip::ALL {
        let base = chip.index() * 8;
        let digital = (volts[base + 2] + volts[base + 3]) / 2.0 * 4.0;
        let analog: Vec<f64> = [0, 1, 4, 5, 6, 7]
            .iter()
            .map(|&channel| volts[base + channel])
            .collect();
        // SBAS459K §9.3.1.3.5: CH1, 2 and 5 to 8 read 0.5 × (AVDD − AVSS).
        let analog = analog.iter().sum::<f64>() / analog.len() as f64 * 2.0;
        report.record(
            match chip {
                Chip::Ic1 => "IC1 DVDD",
                Chip::Ic2 => "IC2 DVDD",
            },
            (3.0..3.6).contains(&digital),
            format!("{digital:.3} V from CH3/CH4 x4 (limit 3.0 to 3.6)"),
        );
        report.record(
            match chip {
                Chip::Ic1 => "IC1 AVDD-AVSS",
                Chip::Ic2 => "IC2 AVDD-AVSS",
            },
            (4.75..5.25).contains(&analog),
            format!("{analog:.3} V from CH1/2/5-8 x2 (limit 4.75 to 5.25)"),
        );
    }
    Ok(())
}

fn check_temperature(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::Temperature)?;
    report.configuration("temperature registers", mismatches);
    let (statistics, capture) = collect(board, SAMPLE_RATE_HZ / 2)?;
    report.record("temperature capture", capture.clean(), capture.summary());
    let mut degrees = [0f64; 2];
    for chip in Chip::ALL {
        let base = chip.index() * 8;
        let microvolts = (0..8)
            .map(|channel| {
                statistics[base + channel].mean()
                    * Mode::Temperature.microvolts_per_code(base + channel)
            })
            .sum::<f64>()
            / 8.0;
        degrees[chip.index()] = (microvolts - 145_300.0) / 490.0 + 25.0;
        info!(
            "  {chip:?}: {microvolts:.0} uV, {:.1} C",
            degrees[chip.index()]
        );
    }
    report.record(
        "die temperature",
        degrees.iter().all(|degree| (10.0..60.0).contains(degree))
            && (degrees[0] - degrees[1]).abs() < 5.0,
        format!(
            "IC1 {:.1} C, IC2 {:.1} C (limit 10 to 60, within 5 of each other)",
            degrees[0], degrees[1]
        ),
    );
    Ok(())
}

fn note_right_leg_drive(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::RightLegDriveMeasurement)?;
    report.configuration("bias-measurement registers", mismatches);
    let (statistics, _) = collect(board, SAMPLE_RATE_HZ / 2)?;
    for chip in Chip::ALL {
        let channel = chip.index() * 8;
        let millivolts = statistics[channel].mean()
            * Mode::RightLegDriveMeasurement.microvolts_per_code(channel)
            / 1e3;
        report.note(
            match chip {
                Chip::Ic1 => "IC1 bias drive (RLDIN on CH1)",
                Chip::Ic2 => "IC2 bias drive (RLDIN on CH1)",
            },
            format!(
                "{millivolts:+.1} mV, {:?} in the product recipe",
                RIGHT_LEG_DRIVE_MODE[chip.index()]
            ),
        );
    }
    Ok(())
}

const STREAM_SECONDS: u32 = 30;

fn check_product_stream(board: &mut Board, report: &mut Report) -> Result<()> {
    let mismatches = board.enter(Mode::Product)?;
    report.configuration("product registers", mismatches);
    DATA_READY_OFFSET_MINIMUM.store(i32::MAX, Ordering::Relaxed);
    DATA_READY_OFFSET_MAXIMUM.store(i32::MIN, Ordering::Relaxed);
    let (statistics, capture) = collect(board, STREAM_SECONDS * SAMPLE_RATE_HZ)?;
    let offsets = [
        DATA_READY_OFFSET_MINIMUM.load(Ordering::Relaxed),
        DATA_READY_OFFSET_MAXIMUM.load(Ordering::Relaxed),
    ];
    for (channel, statistic) in statistics.iter().enumerate() {
        let scale = Mode::Product.microvolts_per_code(channel) / 1e3;
        info!(
            "  {}: mean {:+9.2} mV, range {:+9.2} to {:+9.2} mV{}",
            channel_name(channel),
            statistic.mean() * scale,
            statistic.minimum as f64 * scale,
            statistic.maximum as f64 * scale,
            match statistic.railed() {
                true => ", railed",
                false => "",
            }
        );
    }
    let seconds = capture.elapsed_microseconds as f64 / 1e6;
    let rates = [
        capture.ic1_edges as f64 / seconds,
        capture.ic2_edges as f64 / seconds,
    ];
    let expected = SAMPLE_RATE_HZ as f64;
    report.record(
        "product stream",
        capture.clean(),
        format!("{STREAM_SECONDS} s: {}", capture.summary()),
    );
    report.record(
        "conversion rate",
        rates
            .iter()
            .all(|rate| (rate / expected - 1.0).abs() < 0.02),
        format!(
            "IC1 {:.1} SPS, IC2 {:.1} SPS, expected {expected:.0} within 2%",
            rates[0], rates[1]
        ),
    );
    report.record(
        "lockstep",
        capture.ic1_edges.abs_diff(capture.ic2_edges) <= 1,
        format!(
            "DRDY edge counts differ by {} over {STREAM_SECONDS} s",
            capture.ic1_edges.abs_diff(capture.ic2_edges)
        ),
    );
    report.record(
        "DRDY alignment",
        offsets[0].unsigned_abs().max(offsets[1].unsigned_abs()) <= DATA_READY_OFFSET_LIMIT_MICROSECONDS,
        format!(
            "IC2 DRDY falls {} to {} us after IC1's, interrupt timestamps (limit {DATA_READY_OFFSET_LIMIT_MICROSECONDS})",
            offsets[0], offsets[1]
        ),
    );
    Ok(())
}

const PERIOD_MICROSECONDS: i32 = (1_000_000 / SAMPLE_RATE_HZ) as i32;
/// Both edges are dispatched by one GPIO interrupt, so a few microseconds of
/// handler order is the floor; an unshared START would show hundreds.
const DATA_READY_OFFSET_LIMIT_MICROSECONDS: u32 = 20;
static LAST_EDGE_MICROSECONDS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
static DATA_READY_OFFSET_MINIMUM: AtomicI32 = AtomicI32::new(i32::MAX);
static DATA_READY_OFFSET_MAXIMUM: AtomicI32 = AtomicI32::new(i32::MIN);

unsafe extern "C" fn data_ready_edge(argument: *mut core::ffi::c_void) {
    let chip = argument as usize;
    let now = unsafe { sys::esp_timer_get_time() } as u32;
    DATA_READY_EDGES[chip].fetch_add(1, Ordering::Relaxed);
    LAST_EDGE_MICROSECONDS[chip].store(now, Ordering::Relaxed);
    if chip == Chip::Ic2.index() {
        let since_ic1 = now
            .wrapping_sub(LAST_EDGE_MICROSECONDS[Chip::Ic1.index()].load(Ordering::Relaxed))
            as i32;
        // IC2's handler can run just before IC1's within one dispatch.
        let offset = match since_ic1 > PERIOD_MICROSECONDS / 2 {
            true => since_ic1 - PERIOD_MICROSECONDS,
            false => since_ic1,
        };
        DATA_READY_OFFSET_MINIMUM.fetch_min(offset, Ordering::Relaxed);
        DATA_READY_OFFSET_MAXIMUM.fetch_max(offset, Ordering::Relaxed);
    }
}

fn attach_data_ready(pin: i32, chip: Chip) -> Result<()> {
    esp!(unsafe { sys::gpio_set_intr_type(pin, sys::gpio_int_type_t_GPIO_INTR_NEGEDGE) })?;
    esp!(unsafe {
        sys::gpio_isr_handler_add(
            pin,
            Some(data_ready_edge),
            chip.index() as *mut core::ffi::c_void,
        )
    })?;
    esp!(unsafe { sys::gpio_intr_enable(pin) })?;
    Ok(())
}

fn run_checks(board: &mut Board, report: &mut Report) -> Result<()> {
    board.power_up()?;
    if !board.identify(report)? {
        bail!("no later stage can run without both chips answering");
    }
    check_shorted(board, report)?;
    check_test_signal(board, report)?;
    check_supplies(board, report)?;
    check_temperature(board, report)?;
    note_right_leg_drive(board, report)?;
    check_start_control(board, report)?;
    check_product_stream(board, report)?;
    Ok(())
}

/// tSETTLE in high-resolution mode at fMOD/256 (SBAS459K table 12).
const SETTLING_CLOCKS: f64 = 4616.0;
const START_TRIALS: usize = 5;

/// Holds START low and confirms both chips stop, then times each chip's
/// first DRDY after START rises, which the datasheet fixes at tSETTLE.
fn check_start_control(board: &mut Board, report: &mut Report) -> Result<()> {
    board.enter(Mode::Product)?;
    let mut edges_while_low = [0u32; 2];
    let mut latencies = [[0i64; START_TRIALS]; 2];
    for trial in 0..START_TRIALS {
        board.start.set_low()?;
        FreeRtos::delay_ms(10);
        let before: [u32; 2] =
            core::array::from_fn(|chip| DATA_READY_EDGES[chip].load(Ordering::Relaxed));
        FreeRtos::delay_ms(50);
        for chip in 0..2 {
            edges_while_low[chip] += DATA_READY_EDGES[chip]
                .load(Ordering::Relaxed)
                .wrapping_sub(before[chip]);
        }
        let armed: [u32; 2] =
            core::array::from_fn(|chip| DATA_READY_EDGES[chip].load(Ordering::Relaxed));
        let raised = now_microseconds();
        board.start.set_high()?;
        let mut pending = [true, true];
        while pending.iter().any(|&waiting| waiting) {
            for chip in 0..2 {
                if pending[chip] && DATA_READY_EDGES[chip].load(Ordering::Relaxed) != armed[chip] {
                    let edge = LAST_EDGE_MICROSECONDS[chip].load(Ordering::Relaxed);
                    latencies[chip][trial] = edge.wrapping_sub(raised as u32) as i32 as i64;
                    pending[chip] = false;
                }
            }
            if now_microseconds() - raised > DATA_READY_TIMEOUT_MICROSECONDS {
                bail!("no DRDY after START on {:?}", pending);
            }
        }
        // Drain the frames so RDATAC keeps DRDY pulsing normally.
        board.capture(10, |_, _| {})?;
    }
    let expected = SETTLING_CLOCKS / INTERNAL_OSCILLATOR_HZ as f64 * 1e6;
    for chip in Chip::ALL {
        report.note(
            match chip {
                Chip::Ic1 => "IC1 START to first DRDY",
                Chip::Ic2 => "IC2 START to first DRDY",
            },
            format!(
                "{:?} us over {START_TRIALS} trials (tSETTLE {expected:.0} us)",
                latencies[chip.index()]
            ),
        );
    }
    report.record(
        "START halts both chips",
        edges_while_low == [0, 0],
        format!(
            "DRDY edges during {START_TRIALS} x 50 ms with START low: IC1 {}, IC2 {}",
            edges_while_low[0], edges_while_low[1]
        ),
    );
    let within = |latency: &i64| (*latency as f64 - expected).abs() < 50.0;
    report.record(
        "START to DRDY latency",
        latencies.iter().flatten().all(within),
        format!("every trial within 50 us of tSETTLE {expected:.0} us on both chips"),
    );
    Ok(())
}

fn main() -> Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    // Captures spin on core 0 for up to 30 s; the idle task must not be watched.
    let watchdog = sys::esp_task_wdt_config_t {
        timeout_ms: 5000,
        idle_core_mask: 0,
        trigger_panic: false,
    };
    esp!(unsafe { sys::esp_task_wdt_reconfigure(&watchdog) })?;
    // The monitor attaches after boot; give it time before the first line.
    FreeRtos::delay_ms(3000);
    info!("==== motherboard check ====");

    let peripherals = Peripherals::take()?;
    let pins = peripherals.pins;
    // Pads 39-42 come out of reset owned by JTAG and 43-44 by UART0.
    // PinDriver leaves IO_MUX alone, so gpio_config hands these to the GPIO
    // matrix; the SPI driver claims SCLK (43) itself.
    for (pin, mode) in [
        (40, sys::gpio_mode_t_GPIO_MODE_INPUT),
        (42, sys::gpio_mode_t_GPIO_MODE_OUTPUT),
        (44, sys::gpio_mode_t_GPIO_MODE_OUTPUT),
    ] {
        let config = sys::gpio_config_t {
            pin_bit_mask: 1u64 << pin,
            mode,
            pull_up_en: sys::gpio_pullup_t_GPIO_PULLUP_DISABLE,
            pull_down_en: sys::gpio_pulldown_t_GPIO_PULLDOWN_DISABLE,
            intr_type: sys::gpio_int_type_t_GPIO_INTR_DISABLE,
        };
        esp!(unsafe { sys::gpio_config(&config) })?;
    }
    let mut board = Board::new(
        peripherals.spi2,
        Wiring {
            sclk: pins.gpio43.into(),
            data_in: pins.gpio9.into(),
            data_out: pins.gpio2.into(),
            power_down: pins.gpio42.into(),
            reset: pins.gpio8.into(),
            start: pins.gpio44.into(),
            chip_select: [pins.gpio3.into(), pins.gpio7.into()],
            data_ready: [pins.gpio40.into(), pins.gpio1.into()],
        },
    )?;
    esp!(unsafe { sys::gpio_install_isr_service(0) })?;
    for chip in Chip::ALL {
        attach_data_ready(board.data_ready[chip.index()].pin() as i32, chip)?;
    }

    let mut report = Report {
        results: Vec::new(),
    };
    if let Err(error) = run_checks(&mut board, &mut report) {
        report.record("run", false, format!("stopped: {error:#}"));
    }
    report.print_summary();
    loop {
        FreeRtos::delay_ms(1000);
    }
}
