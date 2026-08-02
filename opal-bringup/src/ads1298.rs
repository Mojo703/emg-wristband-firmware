//! One ADS1298, driven with plain bytes.
//!
//! Datasheet: TI SBAS459K.
//!
//! Registers are `u8` constants here rather than the typed map `opal-firmware` uses.
//! That map earns its keep across twenty-three writes spread over two chips and two
//! roles; this crate writes eleven, all in one function, all visible at once, so the
//! types would only stand between the bench and the datasheet.

use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{Input, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};
use std::sync::Arc;

/// The bus is behind an `Arc` so the SPI clock can be changed mid-session: a
/// `SpiDeviceDriver` fixes its baud rate at construction and never gives the bus back.
pub type SpiDevice = SpiDeviceDriver<'static, Arc<SpiDriver<'static>>>;

pub const DEVICE_ID: u8 = 0x92;
pub const CHANNELS: usize = 8;
pub const FRAME_BYTES: usize = 3 + CHANNELS * 3;

/// Gap between the bytes of a multi-byte command, and between the last SCLK of a
/// transaction and CS rising. The command decoder needs tSDECODE = 4 tCLK (~2 us at
/// 2.048 MHz) per byte; clocking a command as one continuous stream is only legal when
/// the SCLK is slow enough to provide that inside the byte itself (SBAS459K
/// §9.5.1.2.1). This is the "burst method" the datasheet prescribes for faster clocks,
/// and it is the experiment for the standing puzzle of ID reading 0x00 at 2 MHz and
/// 4 MHz: if spacing the RREG bytes fixes those rates, the failure was decode timing,
/// not signal integrity. 5 us rather than 2 for margin; register traffic is not on the
/// frame budget.
const COMMAND_DECODE_GAP_US: u32 = 5;

const SDATAC: u8 = 0x11;
const START_COMMAND: u8 = 0x08;
const STOP_COMMAND: u8 = 0x0A;
const RDATA: u8 = 0x12;
const RREG: u8 = 0x20;
const WREG: u8 = 0x40;

pub const REG_ID: u8 = 0x00;
pub const REG_CONFIG1: u8 = 0x01;
pub const REG_CONFIG2: u8 = 0x02;
pub const REG_CONFIG3: u8 = 0x03;
pub const REG_CH1SET: u8 = 0x05;
/// Eight fully writable bits with no effect while the right-leg amplifier is powered
/// down, which makes it the harmless scratch register for a write/readback bus check.
/// CHnSET cannot serve: its bit 3 is reserved and reads back zero, so half of any
/// walking pattern would look like a bus failure.
pub const REG_RLD_SENSP: u8 = 0x0D;
pub const REG_GPIO: u8 = 0x14;
/// Bit 3 is SINGLE_SHOT. Nothing else in this harness writes CONFIG4, so it holds its
/// reset value of 0x00 except in the single-shot cells.
pub const REG_CONFIG4: u8 = 0x17;
pub const REG_COUNT: u8 = 0x1A;

/// Register names in address order, so a dump reads as names rather than offsets.
pub const REGISTER_NAMES: [&str; REG_COUNT as usize] = [
    "ID",
    "CONFIG1",
    "CONFIG2",
    "CONFIG3",
    "LOFF",
    "CH1SET",
    "CH2SET",
    "CH3SET",
    "CH4SET",
    "CH5SET",
    "CH6SET",
    "CH7SET",
    "CH8SET",
    "RLD_SENSP",
    "RLD_SENSN",
    "LOFF_SENSP",
    "LOFF_SENSN",
    "LOFF_FLIP",
    "LOFF_STATP",
    "LOFF_STATN",
    "GPIO",
    "PACE",
    "RESP",
    "CONFIG4",
    "WCT1",
    "WCT2",
];

/// One frame: the status word and eight sign-extended 24-bit codes.
pub struct Frame {
    pub status: u32,
    pub channels: [i32; CHANNELS],
}

impl Frame {
    /// Bits 23:20 are a fixed `1100` on every frame. A read that lost bit alignment
    /// usually breaks this even though the SPI transaction itself succeeded.
    pub fn marker_ok(&self) -> bool {
        self.status >> 20 == 0b1100
    }
}

fn decode_frame(raw: &[u8; FRAME_BYTES]) -> Frame {
    let status = ((raw[0] as u32) << 16) | ((raw[1] as u32) << 8) | raw[2] as u32;
    let mut channels = [0i32; CHANNELS];
    for (index, channel) in channels.iter_mut().enumerate() {
        let offset = 3 + index * 3;
        *channel = decode_i24(&raw[offset..offset + 3]);
    }
    Frame { status, channels }
}

fn decode_i24(bytes: &[u8]) -> i32 {
    let raw = ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | bytes[2] as u32;
    if raw & 0x80_0000 != 0 {
        (raw | 0xFF00_0000) as i32
    } else {
        raw as i32
    }
}

pub struct Ads1298 {
    spi: SpiDevice,
    data_ready: PinDriver<'static, Input>,
    reset_n: PinDriver<'static, Output>,
    power_down: PinDriver<'static, Output>,
    start: PinDriver<'static, Output>,
    /// Raised between transactions. A CS rising edge is the only recovery the chip
    /// offers a command decoder that has lost count of its clocks (SBAS459K §9.5.1.1);
    /// held permanently low, one glitched SCLK edge would desynchronise every
    /// transaction that follows, indistinguishable from a chip fault. Driven as a GPIO
    /// rather than by the SPI peripheral so the clock can be swapped without
    /// surrendering the pin.
    chip_select: PinDriver<'static, Output>,
}

impl Ads1298 {
    pub fn new(
        spi: SpiDevice,
        data_ready: PinDriver<'static, Input>,
        mut reset_n: PinDriver<'static, Output>,
        mut power_down: PinDriver<'static, Output>,
        mut start: PinDriver<'static, Output>,
        mut chip_select: PinDriver<'static, Output>,
    ) -> Result<Self> {
        power_down.set_low()?;
        reset_n.set_low()?;
        start.set_low()?;
        chip_select.set_high()?;

        Ok(Self {
            spi,
            data_ready,
            reset_n,
            power_down,
            start,
            chip_select,
        })
    }

    /// Swaps in a bus handle at a different SPI clock.
    pub fn set_spi(&mut self, spi: SpiDevice) {
        self.spi = spi;
    }

    /// Releases power-down and reset, waits out the mandated settling, and leaves the
    /// chip in SDATAC so registers are accessible.
    pub fn power_up(&mut self) -> Result<()> {
        FreeRtos::delay_ms(5);
        self.power_down.set_high()?;
        self.reset_n.set_high()?;
        FreeRtos::delay_ms(2000);

        self.reset_n.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset_n.set_high()?;

        // 18 tCLK (~9 us at 2.048 MHz) after RESET rises, during which no command may
        // be sent (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);

        self.command(SDATAC)?;
        FreeRtos::delay_ms(1);
        Ok(())
    }

    /// A bare warm reset: RESET pulse, the post-reset lockout, SDATAC. For recovery
    /// mid-session, where the cold-start supply settling has long since happened.
    pub fn reset_pulse(&mut self) -> Result<()> {
        self.reset_n.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset_n.set_high()?;
        FreeRtos::delay_ms(1);
        self.command(SDATAC)?;
        Ok(())
    }

    pub fn start_conversion(&mut self) -> Result<()> {
        self.start.set_high()?;
        Ok(())
    }

    pub fn stop_conversion(&mut self) -> Result<()> {
        self.start.set_low()?;
        Ok(())
    }

    /// Begins conversions with the START opcode instead of the pin (SBAS459K
    /// §9.4.1.1 offers both). The pin stays wherever it is; a cell using this leaves
    /// it low, so the wire is out of the experiment entirely.
    pub fn start_conversion_by_command(&mut self) -> Result<()> {
        self.command(START_COMMAND)
    }

    pub fn stop_conversion_by_command(&mut self) -> Result<()> {
        self.command(STOP_COMMAND)
    }

    /// DRDY is active low.
    pub fn data_ready(&self) -> bool {
        self.data_ready.is_low()
    }

    /// Runs one transaction with CS low, and raises CS afterwards even on failure so
    /// the decoder-reset edge is never skipped. The gap before CS rises covers tSCCS
    /// (4 tCLK after the last SCLK); the gap after covers the minimum CS-high pulse.
    fn with_selection<T>(&mut self, transaction: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.chip_select.set_low()?;
        let result = transaction(self);
        Ets::delay_us(COMMAND_DECODE_GAP_US);
        self.chip_select.set_high()?;
        Ets::delay_us(COMMAND_DECODE_GAP_US);
        result
    }

    /// Clocks bytes out one at a time with [`COMMAND_DECODE_GAP_US`] between them --
    /// the datasheet's burst method for multi-byte commands.
    fn write_bytes_spaced(&mut self, bytes: &[u8]) -> Result<()> {
        for &byte in bytes {
            self.spi.write(&[byte])?;
            Ets::delay_us(COMMAND_DECODE_GAP_US);
        }
        Ok(())
    }

    fn command(&mut self, opcode: u8) -> Result<()> {
        self.with_selection(|chip| {
            chip.spi.write(&[opcode])?;
            Ok(())
        })
    }

    pub fn write_register(&mut self, address: u8, value: u8) -> Result<()> {
        // Second byte is "number of registers - 1".
        self.with_selection(|chip| chip.write_bytes_spaced(&[WREG | address, 0x00, value]))
    }

    pub fn read_register(&mut self, address: u8) -> Result<u8> {
        self.with_selection(|chip| {
            chip.write_bytes_spaced(&[RREG | address, 0x00])?;
            let mut value = [0u8; 1];
            chip.spi.read(&mut value)?;
            Ok(value[0])
        })
    }

    pub fn read_all_registers(&mut self) -> Result<[u8; REG_COUNT as usize]> {
        let mut values = [0u8; REG_COUNT as usize];
        for (address, value) in values.iter_mut().enumerate() {
            *value = self.read_register(address as u8)?;
        }
        Ok(values)
    }

    /// Clocks out one frame with an explicit RDATA.
    ///
    /// RDATAC would be one command cheaper per frame, but it locks the chip out of
    /// register reads for as long as it streams. On-demand reads are what let the poll
    /// loop audit CONFIG1 between frames and so catch a chip reverting at the frame it
    /// happens on, instead of inferring it afterwards from the DRDY period.
    pub fn read_frame(&mut self) -> Result<Frame> {
        let mut raw = [0u8; FRAME_BYTES];
        self.with_selection(|chip| {
            chip.spi
                .transaction(&mut [Operation::Write(&[RDATA]), Operation::Read(&mut raw)])?;
            Ok(())
        })?;
        Ok(decode_frame(&raw))
    }

    /// One frame clocked out as nine 3-byte chunks with decode-gap pauses between,
    /// CS held for the whole transaction. Same data as [`Self::read_frame`], but the
    /// 216-clock burst is broken into short runs — the discriminator for whether the
    /// unbroken burst is what disturbs a converting chip.
    pub fn read_frame_chunked(&mut self) -> Result<Frame> {
        let mut raw = [0u8; FRAME_BYTES];
        self.with_selection(|chip| {
            chip.spi.write(&[RDATA])?;
            Ets::delay_us(COMMAND_DECODE_GAP_US);
            for chunk in raw.chunks_mut(3) {
                chip.spi.read(chunk)?;
                Ets::delay_us(COMMAND_DECODE_GAP_US);
            }
            Ok(())
        })?;
        Ok(decode_frame(&raw))
    }

    /// Clocks out only the first `data_bytes` of a frame and then raises CS, abandoning
    /// the rest. Legal while converting: CS rising resets the command decoder
    /// (SBAS459K §9.5.1.1) and the next DRDY starts a fresh frame, so the truncation
    /// costs the remaining channels of that sample and nothing else. The dose the
    /// readout fault responds to scales with data bits shifted, so this is the knob
    /// that walks down the burst-length axis without leaving continuous conversion.
    pub fn read_frame_partial(&mut self, data_bytes: usize) -> Result<()> {
        let mut raw = [0u8; FRAME_BYTES];
        let wanted = data_bytes.min(FRAME_BYTES);
        self.with_selection(|chip| {
            chip.spi.transaction(&mut [
                Operation::Write(&[RDATA]),
                Operation::Read(&mut raw[..wanted]),
            ])?;
            Ok(())
        })
    }

    /// The RDATA opcode with no data clocks at all; the CS rising edge then resets
    /// the command decoder. Isolates the command path from the data burst.
    pub fn read_frame_opcode_only(&mut self) -> Result<()> {
        self.with_selection(|chip| {
            chip.spi.write(&[RDATA])?;
            Ok(())
        })
    }

    /// Reads ID `attempts` times and returns how many did not come back as
    /// [`DEVICE_ID`]. The bench needs an error *rate* rather than one read: 2 MHz and
    /// 4 MHz have been seen returning 0x00 where 1 MHz works, and a single sample
    /// cannot tell a marginal bus from a dead one.
    pub fn probe_identity(&mut self, attempts: u32) -> u32 {
        (0..attempts)
            .filter(|_| !matches!(self.read_register(REG_ID), Ok(DEVICE_ID)))
            .count() as u32
    }
}
