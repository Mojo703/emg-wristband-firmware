//! Driver for two TI ADS1298 8-channel ADCs in "Cascade Configuration"
//! 
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf>

use anyhow::Result;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{Input, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};

use crate::registers::Register;
use crate::spi_commands;

pub const CHANNELS_PER_DEVICE: usize = 8;
pub const DEVICE_COUNT: usize = 2;
const STATUS_BYTES: usize = 3;
const BYTES_PER_CHANNEL: usize = 3;
pub const FRAME_BYTES: usize = STATUS_BYTES + CHANNELS_PER_DEVICE * BYTES_PER_CHANNEL; // 27

/// Decoded status word + 8 sign-extended 24-bit channel codes for one device
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub status: u32,
    pub channels: [i32; CHANNELS_PER_DEVICE],
}

/// One frame is a sample from each of the two cascaded devices
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub devices: [Sample; DEVICE_COUNT],
}

/// One ADS1298 addressed via its own dedicated CS on the shared SPI bus
/// Each ADC also has its own DRDY pin, however START is tied together
/// so they DRDY pins should pull at same time 
pub struct Ads1298Device<'d> {
    spi: SpiDeviceDriver<'d, &'d SpiDriver<'d>>,
    drdy: PinDriver<'d, Input>,
}

impl<'d> Ads1298Device<'d> {
    pub fn new(spi: SpiDeviceDriver<'d, &'d SpiDriver<'d>>, drdy: PinDriver<'d, Input>) -> Self {
        Self { spi, drdy }
    }

    /// DRDY is active-low. It falls when a new frame is ready to be sampled.
    pub fn data_ready(&self) -> Result<bool> {
        Ok(self.drdy.is_low())
    }

    /// Helper for sending SPI commands
    fn send_command(&mut self, command: u8) -> Result<()> {
        self.spi.write(&[command])?;
        Ok(())
    }

    pub fn software_reset(&mut self) -> Result<()> {
        self.send_command(spi_commands::RESET)
    }

    pub fn wakeup(&mut self) -> Result<()> {
        self.send_command(spi_commands::WAKEUP)
    }

    pub fn standby(&mut self) -> Result<()> {
        self.send_command(spi_commands::STANDBY)
    }

    /// Disables read data continuous mode so that registers can be configured
    pub fn stop_read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::SDATAC)
    }

    /// Enables read data continuous mode. Once started, each DRDY
    /// falling edge means a frame is ready to clock out
    pub fn read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::RDATAC)
    }

    // Writes to a single register on the ADS1298
    pub fn write_register(&mut self, reg: Register, value: u8) -> Result<()> {
        // Second byte is "number of registers - 1" (0x00 = one register).
        self.spi.write(&[spi_commands::WREG_BASE | reg.addr(), 0x00, value])?;
        Ok(())
    }

    // Reads from a single register on the ADS1298
    pub fn read_register(&mut self, reg: Register) -> Result<u8> {
        let mut rx = [0u8; 1];
        self.spi.transaction(&mut [
            Operation::Write(&[spi_commands::RREG_BASE | reg.addr(), 0x00]),
            Operation::Read(&mut rx), 
        ])?;
        Ok(rx[0])
    }

    /// Clocks out one frame. Only valid while in RDATAC mode 
    // and after `data_ready()` reports true
    pub fn read_frame(&mut self) -> Result<Sample> {
        let mut raw = [0u8; FRAME_BYTES];
        self.spi.read(&mut raw)?;
        Ok(parse_sample(&raw))
    }

    /// First hardware bring-up check to confirm that SPI works. 
    //  Reads the Id register and compares against the
    //  datasheet's documented value for this device (0x92). 
    pub fn verify_id(&mut self, expected: u8) -> Result<bool> {
        Ok(self.read_register(Register::Id)? == expected)
    }
}

/// Owns both ADS1298 devices sharing one Cascaded SPI bus and two
/// control lines. Specifically in this format so that RLD registers 
// for each ADS1298 can be configured differently on startup.
pub struct Ads1298Pair<'d> {
    pub adc1: Ads1298Device<'d>,
    pub adc2: Ads1298Device<'d>,
    reset_n: PinDriver<'d, Output>,
    start: PinDriver<'d, Output>,
}

impl<'d> Ads1298Pair<'d> {
    pub fn new(
        adc1: Ads1298Device<'d>,
        adc2: Ads1298Device<'d>,
        reset_n: PinDriver<'d, Output>,
        start: PinDriver<'d, Output>,
    ) -> Self {
        Self {
            adc1,
            adc2,
            reset_n,
            start,
        }
    }

    /// Pulse RESET_N low then high. Datasheet mentions >= 2 tCLK low and a
    /// >= 18 tCLK wait before the first command can be sent. Both delays here are 
    /// temporarily 1ms since the actual CLK frequency isn't decided yet.
    pub fn hardware_reset(&mut self) -> Result<()> {
        self.reset_n.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset_n.set_high()?;
        FreeRtos::delay_ms(1);
        Ok(())
    }

    /// Pulls the shared START pin high 
    pub fn start_conversion(&mut self) -> Result<()> {
        self.start.set_high()?;
        Ok(())
    }

    /// Pulls the shared START pin low
    pub fn stop_conversion(&mut self) -> Result<()> {
        self.start.set_low()?;
        Ok(())
    }

    /// True only once both devices report their own DRDY low. Since each
    /// device has its own dedicated DRDY pin, this checks both independently
    /// rather than trusting just one
    pub fn data_ready(&self) -> Result<bool> {
        Ok(self.adc1.data_ready()? && self.adc2.data_ready()?)
    }

    /// Clocks out one frame from each device simultaneously. Only valid while
    /// both are in RDATAC mode and after `data_ready()` reports true
    pub fn read_frame(&mut self) -> Result<Frame> {
        let s1 = self.adc1.read_frame()?;
        let s2 = self.adc2.read_frame()?;
        Ok(Frame { devices: [s1, s2] })
    }
}

/// Sign-extends a 24-bit two's-complement sample into a full i32
fn decode_i24(b: [u8; 3]) -> i32 {
    let u = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
    if u & 0x00800000 != 0 {
        (u | 0xFF000000) as i32
    } else {
        u as i32
    }
}

/// Decodes one device's raw frame bytes (status word + 8 channels) into a `Sample`
fn parse_sample(raw: &[u8; FRAME_BYTES]) -> Sample {
    let status = ((raw[0] as u32) << 16) | ((raw[1] as u32) << 8) | (raw[2] as u32);
    let mut channels = [0i32; CHANNELS_PER_DEVICE];
    for (i, channel) in channels.iter_mut().enumerate() {
        let off = STATUS_BYTES + i * BYTES_PER_CHANNEL;
        *channel = decode_i24([raw[off], raw[off + 1], raw[off + 2]]);
    }
    Sample { status, channels }
}
