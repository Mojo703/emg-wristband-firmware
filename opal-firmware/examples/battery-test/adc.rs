//! The motherboard's two ADS1298s, streaming at the product rate.
//!
//! This board shares SCLK, DIN, DOUT, PWDN, RESET and START between the chips;
//! only CS and DRDY are per chip. IC2 (CLKSEL high) runs its internal oscillator
//! and drives it out of CLK, and IC1 (CLKSEL low) converts on that clock. Both
//! chips start on the same START edge and convert in lockstep, so one IC1 DRDY
//! edge means both frames are ready. IC2's DRDY is only counted, to catch IC2
//! dying while its oscillator keeps IC1 alive.
//!
//! The register recipe is the product driver's (`src/adc/ads1298.rs`), with
//! CLK_EN set on IC2. The frames are read and checked, then dropped: this
//! build exists to draw the product's current, not to classify.

#![allow(dead_code, unused_imports)]

#[path = "../../src/adc/ads1298.rs"]
mod ads1298;
#[path = "../../src/adc/channel.rs"]
mod channel;
#[path = "../../src/adc/decode.rs"]
mod decode;
#[path = "../../src/adc/registers.rs"]
mod registers;
#[path = "../../src/adc/status.rs"]
mod status;

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use esp_idf_svc::hal::cpu::Core;
use esp_idf_svc::hal::delay::{Ets, FreeRtos, TickType};
use esp_idf_svc::hal::gpio::{AnyInputPin, AnyOutputPin, Input, Output, PinDriver, Pull};
use esp_idf_svc::hal::spi::config::{Config as SpiConfig, MODE_1};
use esp_idf_svc::hal::spi::{Operation, SpiAnyPins, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::task::thread::ThreadSpawnConfiguration;
use esp_idf_svc::hal::units::FromValueType;
use esp_idf_svc::sys::{self, esp};
use log::info;

use ads1298::{device_config1, RightLegDriveMode, RIGHT_LEG_DRIVE_MODE};
use decode::{parse_sample, FRAME_BYTES};
use registers::{Config1, Register, RegisterValue};

use crate::record;

/// Read by `registers` for the data-rate arithmetic.
const INTERNAL_OSCILLATOR_HZ: u32 = 2_048_000;

const EXPECTED_DEVICE_ID: u8 = 0x92;
const COMMAND_RESET: u8 = 0x06;
const COMMAND_READ_DATA_CONTINUOUS: u8 = 0x10;
const COMMAND_STOP_READ_DATA_CONTINUOUS: u8 = 0x11;
const COMMAND_READ_REGISTER: u8 = 0x20;
const COMMAND_WRITE_REGISTER: u8 = 0x40;

/// The product's bench-validated clocks.
const COMMAND_SPI_BAUD_RATE_HZ: u32 = 2_000_000;
const FRAME_SPI_BAUD_RATE_HZ: u32 = 8_000_000;
/// tSDECODE between command bytes, with margin (SBAS459K §9.5.1.2.1).
const COMMAND_DECODE_GAP_NANOSECONDS: u32 = 5_000;
const CHIP_SELECT_GAP_MICROSECONDS: u32 = 5;

/// tPOR is 2^18 tCLK, 128 ms at 2.048 MHz. IC1 only starts counting once IC2
/// drives its clock.
const CLOCKED_CHIP_POWER_ON_MILLISECONDS: u32 = 250;
const SUPPLY_SETTLE_MILLISECONDS: u32 = 2000;
const REFERENCE_SETTLE_MILLISECONDS: u32 = 300;
const BRING_UP_RETRY_MILLISECONDS: u32 = 5000;

/// A wait this long for DRDY is forty missed conversions.
const DATA_READY_TIMEOUT_MILLISECONDS: u32 = 20;
const STALLS_BEFORE_RECOVERY: u32 = 3;
const BAD_FRAMES_BEFORE_RECOVERY: u32 = 200;
/// About a second of conversions between checks that IC2 still converts.
const FRAMES_PER_HEALTH_CHECK: u32 = ads1298::SAMPLE_RATE_HZ;

const THREAD_STACK_BYTES: usize = 8192;
const THREAD_PRIORITY: u8 = 10;

/// What the acquisition thread has seen since boot. Read by the run log.
pub struct Counters {
    pub frames: AtomicU32,
    pub missed_frames: AtomicU32,
    pub bad_frames: AtomicU32,
    pub stalls: AtomicU32,
    pub recoveries: AtomicU32,
    pub bring_up_failures: AtomicU32,
}

pub static COUNTERS: Counters = Counters {
    frames: AtomicU32::new(0),
    missed_frames: AtomicU32::new(0),
    bad_frames: AtomicU32::new(0),
    stalls: AtomicU32::new(0),
    recoveries: AtomicU32::new(0),
    bring_up_failures: AtomicU32::new(0),
};

/// DRDY falling edges per chip, counted in the interrupt.
static DATA_READY_EDGES: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
static ACQUISITION_TASK: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(std::ptr::null_mut());

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

    fn right_leg_drive(self) -> RightLegDriveMode {
        RIGHT_LEG_DRIVE_MODE[self.index()]
    }

    /// IC2 is the clock source, so it alone drives CLK.
    fn config1(self) -> Config1 {
        Config1 {
            output_clock_enabled: self == Chip::Ic2,
            ..device_config1()
        }
    }

    fn configuration(self) -> impl Iterator<Item = (Register, u8)> {
        let config1 = self.config1().to_byte();
        self.right_leg_drive()
            .expected_configuration()
            .into_iter()
            .map(move |(register, value)| match register {
                Register::Config1 => (register, config1),
                _ => (register, value),
            })
    }
}

pub struct Wiring {
    pub sclk: AnyOutputPin<'static>,
    pub data_in: AnyOutputPin<'static>,
    pub data_out: AnyInputPin<'static>,
    pub power_down: AnyOutputPin<'static>,
    pub reset: AnyOutputPin<'static>,
    pub start: AnyOutputPin<'static>,
    /// IC1, IC2.
    pub chip_select: [AnyOutputPin<'static>; 2],
    /// IC1, IC2.
    pub data_ready: [AnyInputPin<'static>; 2],
}

type SpiDevice = SpiDeviceDriver<'static, Arc<SpiDriver<'static>>>;

/// The shared bus. CS is a GPIO held low across each whole command, the
/// mechanism the product driver validated.
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

    /// DIN is held low for the whole read: in RDATAC every bit on it is a
    /// potential opcode.
    fn read_frame(&mut self, chip: Chip) -> Result<[u8; FRAME_BYTES]> {
        const DATA_IN_LOW: [u8; FRAME_BYTES] = [0u8; FRAME_BYTES];
        self.selected(chip, |bus| {
            let mut raw = [0u8; FRAME_BYTES];
            bus.frame.transfer(&mut raw, &DATA_IN_LOW)?;
            Ok(raw)
        })
    }

    fn check_identity(&mut self, chip: Chip) -> Result<()> {
        let id = self.read_register(chip, Register::Id)?;
        if id != EXPECTED_DEVICE_ID {
            bail!("{chip:?} ID reads {id:#04x}, expected {EXPECTED_DEVICE_ID:#04x}");
        }
        info!("{chip:?}: ID {id:#04x}");
        Ok(())
    }

    fn configure(&mut self, chip: Chip) -> Result<()> {
        for (register, value) in chip.configuration() {
            self.write_register(chip, register, value)?;
        }
        for (register, expected) in chip.configuration() {
            let read = self.read_register(chip, register)?;
            if read != expected {
                bail!(
                    "{chip:?} {} wrote {expected:#04x}, reads {read:#04x}",
                    register.name()
                );
            }
        }
        Ok(())
    }
}

struct FrontEnd {
    bus: Bus,
    power_down: PinDriver<'static, Output>,
    reset: PinDriver<'static, Output>,
    start: PinDriver<'static, Output>,
    data_ready: [PinDriver<'static, Input>; 2],
}

impl FrontEnd {
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
        // DRDY is driven by the chip, so no pull.
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
        // Polled transfers: this thread has core 1 to itself, and a polled
        // frame read finishes well inside one conversion period.
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
        info!("ADS1298 pair: powering up");
        FreeRtos::delay_ms(5);
        self.power_down.set_high()?;
        self.reset.set_high()?;
        FreeRtos::delay_ms(SUPPLY_SETTLE_MILLISECONDS);
        Ok(())
    }

    /// Reset, configure and start both chips. The shared RESET clears IC2's
    /// CLK_EN, which stops IC1's clock, so IC2 comes up first and IC1 is then
    /// reset on its own by opcode once its clock has run for tPOR.
    fn bring_up(&mut self) -> Result<()> {
        self.start.set_low()?;
        self.reset.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset.set_high()?;
        // 18 tCLK after RESET before any command (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);

        let bus = &mut self.bus;
        bus.send_command(Chip::Ic2, COMMAND_STOP_READ_DATA_CONTINUOUS)?;
        FreeRtos::delay_ms(1);
        bus.check_identity(Chip::Ic2)?;
        bus.write_register(Chip::Ic2, Register::Config1, Chip::Ic2.config1().to_byte())?;
        FreeRtos::delay_ms(CLOCKED_CHIP_POWER_ON_MILLISECONDS);

        bus.send_command(Chip::Ic1, COMMAND_RESET)?;
        FreeRtos::delay_ms(1);
        bus.send_command(Chip::Ic1, COMMAND_STOP_READ_DATA_CONTINUOUS)?;
        FreeRtos::delay_ms(1);
        bus.check_identity(Chip::Ic1)?;

        // IC2 first: its recipe keeps CLK_EN set, so IC1 stays clocked.
        bus.configure(Chip::Ic2)?;
        bus.configure(Chip::Ic1)?;
        FreeRtos::delay_ms(REFERENCE_SETTLE_MILLISECONDS);

        for chip in Chip::ALL {
            bus.send_command(chip, COMMAND_READ_DATA_CONTINUOUS)?;
        }
        self.start.set_high()?;
        info!("ADS1298 pair: streaming at {} SPS", ads1298::SAMPLE_RATE_HZ);
        Ok(())
    }

    /// Retries until both chips come up.
    fn bring_up_until_streaming(&mut self) {
        loop {
            match self.bring_up() {
                Ok(()) => return,
                Err(error) => {
                    COUNTERS.bring_up_failures.fetch_add(1, Ordering::Relaxed);
                    record::error(format!("ADC bring-up failed: {error:#}"));
                    let _ = self.start.set_low();
                    FreeRtos::delay_ms(BRING_UP_RETRY_MILLISECONDS);
                }
            }
        }
    }

    fn recover(&mut self, why: &str) {
        COUNTERS.recoveries.fetch_add(1, Ordering::Relaxed);
        record::error(format!("ADC recovery: {why}"));
        self.bring_up_until_streaming();
    }
}

/// Counts every edge, and wakes the acquisition thread on IC1's.
unsafe extern "C" fn data_ready_edge(argument: *mut core::ffi::c_void) {
    let chip = argument as usize;
    DATA_READY_EDGES[chip].fetch_add(1, Ordering::Relaxed);
    if chip == Chip::Ic1.index() {
        let task = ACQUISITION_TASK.load(Ordering::Relaxed);
        if !task.is_null() {
            esp_idf_svc::hal::task::notify_and_yield(task as sys::TaskHandle_t, NonZeroU32::MIN);
        }
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

/// Starts acquisition on its own thread on core 1. Everything the thread
/// allocates, including its interrupts, lands there. Failures after the thread
/// starts are recorded and retried, never returned.
pub fn start<SPI: SpiAnyPins + Send + 'static>(spi: SPI, wiring: Wiring) -> Result<()> {
    ThreadSpawnConfiguration {
        name: Some(c"adc"),
        stack_size: THREAD_STACK_BYTES,
        priority: THREAD_PRIORITY,
        pin_to_core: Some(Core::Core1),
        ..Default::default()
    }
    .set()?;
    let spawned = std::thread::Builder::new()
        .stack_size(THREAD_STACK_BYTES)
        .spawn(move || {
            if let Err(error) = run(spi, wiring) {
                record::error(format!("ADC thread stopped: {error:#}"));
            }
        });
    ThreadSpawnConfiguration::default().set()?;
    spawned?;
    Ok(())
}

fn run<SPI: SpiAnyPins + 'static>(spi: SPI, wiring: Wiring) -> Result<()> {
    let mut front_end = FrontEnd::new(spi, wiring)?;
    front_end.power_up()?;
    front_end.bring_up_until_streaming();

    ACQUISITION_TASK.store(
        unsafe { sys::xTaskGetCurrentTaskHandle() } as *mut core::ffi::c_void,
        Ordering::Relaxed,
    );
    esp!(unsafe { sys::gpio_install_isr_service(0) })?;
    for chip in Chip::ALL {
        attach_data_ready(front_end.data_ready[chip.index()].pin() as i32, chip)?;
    }

    let timeout = TickType::new_millis(DATA_READY_TIMEOUT_MILLISECONDS as u64).ticks();
    let mut handled_edges = DATA_READY_EDGES[Chip::Ic1.index()].load(Ordering::Relaxed);
    let mut checked_ic2_edges = DATA_READY_EDGES[Chip::Ic2.index()].load(Ordering::Relaxed);
    let mut stalls_in_a_row = 0;
    let mut bad_frames_in_a_row = 0;
    let mut frames_since_check = 0;
    let mut reading_failed = false;
    loop {
        if esp_idf_svc::hal::task::wait_notification(timeout).is_none() {
            COUNTERS.stalls.fetch_add(1, Ordering::Relaxed);
            stalls_in_a_row += 1;
            if stalls_in_a_row >= STALLS_BEFORE_RECOVERY {
                front_end.recover("IC1 DRDY stopped");
                stalls_in_a_row = 0;
                handled_edges = DATA_READY_EDGES[Chip::Ic1.index()].load(Ordering::Relaxed);
            }
            continue;
        }
        stalls_in_a_row = 0;

        let edges = DATA_READY_EDGES[Chip::Ic1.index()].load(Ordering::Relaxed);
        let unread = edges.wrapping_sub(handled_edges);
        handled_edges = edges;
        if unread > 1 {
            COUNTERS
                .missed_frames
                .fetch_add(unread - 1, Ordering::Relaxed);
        }

        let mut pair_good = true;
        for chip in Chip::ALL {
            match front_end.bus.read_frame(chip) {
                Ok(raw) => {
                    reading_failed = false;
                    if parse_sample(&raw).status_word().is_none() {
                        COUNTERS.bad_frames.fetch_add(1, Ordering::Relaxed);
                        pair_good = false;
                    }
                }
                Err(error) => {
                    if !reading_failed {
                        record::error(format!("{chip:?} frame read failed: {error:#}"));
                    }
                    reading_failed = true;
                    pair_good = false;
                }
            }
        }
        if pair_good {
            COUNTERS.frames.fetch_add(1, Ordering::Relaxed);
            bad_frames_in_a_row = 0;
        } else {
            bad_frames_in_a_row += 1;
            if bad_frames_in_a_row >= BAD_FRAMES_BEFORE_RECOVERY {
                front_end.recover("frames without a status marker");
                bad_frames_in_a_row = 0;
                handled_edges = DATA_READY_EDGES[Chip::Ic1.index()].load(Ordering::Relaxed);
            }
        }

        frames_since_check += 1;
        if frames_since_check >= FRAMES_PER_HEALTH_CHECK {
            frames_since_check = 0;
            let ic2_edges = DATA_READY_EDGES[Chip::Ic2.index()].load(Ordering::Relaxed);
            if ic2_edges == checked_ic2_edges {
                front_end.recover("IC2 DRDY stopped while IC1 converts");
                handled_edges = DATA_READY_EDGES[Chip::Ic1.index()].load(Ordering::Relaxed);
            }
            checked_ic2_edges = DATA_READY_EDGES[Chip::Ic2.index()].load(Ordering::Relaxed);
        }
    }
}
