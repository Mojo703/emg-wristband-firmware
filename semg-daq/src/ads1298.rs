//! Driver for two TI ADS1298 8-channel ADCs in "Cascade Configuration"
//! 
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf>

use anyhow::Result;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{Input, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};

use sampling_utils::decode::parse_sample;
use sampling_utils::decode::{Sample, FRAME_BYTES};
pub(crate) use sampling_utils::decode::{Frame, CHANNELS_PER_DEVICE};

use crate::registers::Register;
use crate::spi_commands;

/// Which cascaded device this is — determines which registers differ
/// (CONFIG1.CLK_EN, CONFIG3.PD_RLD, RLD_SENSP/N). See CLAUDE_HANDOFF.md.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ChipRole {
    /// Clock master: drives CLK out to the other device (CONFIG1.CLK_EN=1)
    A,
    /// Clock slave: receives CLK from the master (CONFIG1.CLK_EN=0)
    B,
}

/// One ADS1298 addressed via its own dedicated CS on the shared SPI bus
/// Each ADC also has its own DRDY and RESET pin, however START is tied together
/// so they DRDY pins should pull at same time 
pub(crate) struct Ads1298Device<'d> {
    spi: SpiDeviceDriver<'d, &'d SpiDriver<'d>>,
    drdy: PinDriver<'d, Input>,
    reset_n: PinDriver<'d, Output>,
    pwdn: PinDriver<'d, Output>,
}

impl<'d> Ads1298Device<'d> {
    pub(crate) fn new(spi: SpiDeviceDriver<'d, &'d SpiDriver<'d>>, drdy: PinDriver<'d, Input>, reset_n: PinDriver<'d, Output>, pwdn: PinDriver<'d, Output>) -> Self {
        Self { spi, drdy, reset_n, pwdn }
    }

    /// DRDY is active-low. It falls when a new frame is ready to be sampled.
    fn data_ready(&self) -> Result<bool> {
        Ok(self.drdy.is_low())
    }

    fn reset_pulse(&mut self) -> Result<()> {
        self.reset_n.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset_n.set_high()?;
        Ok(())
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
    fn stop_read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::SDATAC)
    }

    /// Enables read data continuous mode. Once started, each DRDY
    /// falling edge means a frame is ready to clock out
    pub(crate) fn read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::RDATAC)
    }

    // Writes to a single register on the ADS1298
    fn write_register(&mut self, reg: Register, value: u8) -> Result<()> {
        // Second byte is "number of registers - 1" (0x00 = one register).
        self.spi.write(&[spi_commands::WREG_BASE | reg.addr(), 0x00, value])?;
        Ok(())
    }

    // Reads from a single register on the ADS1298
    fn read_register(&mut self, reg: Register) -> Result<u8> {
        let mut rx = [0u8; 1];
        self.spi.transaction(&mut [
            Operation::Write(&[spi_commands::RREG_BASE | reg.addr(), 0x00]),
            Operation::Read(&mut rx), 
        ])?;
        Ok(rx[0])
    }

    /// Clocks out one frame. Only valid while in RDATAC mode
    // and after `data_ready()` reports true
    fn read_frame(&mut self) -> Result<Sample> {
        let mut raw = [0u8; FRAME_BYTES];
        self.spi.read(&mut raw)?;
        Ok(parse_sample(&raw))
    }

    /// First hardware bring-up check to confirm that SPI works. 
    //  Reads the Id register and compares against the
    //  datasheet's documented value for this device (0x92). 
    fn verify_id(&mut self, expected: u8) -> Result<bool> {
        Ok(self.read_register(Register::Id)? == expected)
    }

    // Writes this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be written while streaming.
    fn configure(&mut self, role: ChipRole) -> Result<()> {
        // --- Role-specific registers ---
        let config1 = match role {
            ChipRole::A => 0xE4, 
            ChipRole::B => 0xC4, 
        };
        self.write_register(Register::Config1, config1)?;

        let config3 = match role {
            ChipRole::A => 0xC6,
            ChipRole::B => 0xC0, 
        };
        self.write_register(Register::Config3, config3)?;

        let rld_sens_p = match role {
            ChipRole::A => 0xFF, 
            ChipRole::B => 0x00, 
        };
        self.write_register(Register::RldSensP, rld_sens_p)?;

        let rld_sens_n = match role {
            ChipRole::A => 0xFF, 
            ChipRole::B => 0x00, 
        };
        self.write_register(Register::RldSensN, rld_sens_n)?;

        // Shared registers (identical on both chips)
        self.write_register(Register::Config2, 0x00)?; 
        self.write_register(Register::Loff, 0x13)?; 

        for reg in [
            Register::Ch1Set,
            Register::Ch2Set,
            Register::Ch3Set,
            Register::Ch4Set,
            Register::Ch5Set,
            Register::Ch6Set,
            Register::Ch7Set,
            Register::Ch8Set,
        ] {
            self.write_register(reg, 0x00)?;
        }

        self.write_register(Register::LoffSensP, 0xFF)?; 
        self.write_register(Register::LoffSensN, 0xFF)?;
        self.write_register(Register::LoffFlip, 0x00)?; 
        self.write_register(Register::Gpio, 0x00)?; 
        self.write_register(Register::Pace, 0x00)?; 
        self.write_register(Register::Resp, 0x20)?; 
        self.write_register(Register::Config4, 0x02)?; 
        self.write_register(Register::Wct1, 0x00)?; 
        self.write_register(Register::Wct2, 0x00)?; 

        Ok(())
    }

    // Power-up sequencing for each individual device
     pub(crate) fn power_up(&mut self, role: ChipRole) -> Result<()> {
        self.pwdn.set_low()?;
        self.reset_n.set_low()?;

        FreeRtos::delay_ms(5);

        self.pwdn.set_high()?;
        self.reset_n.set_high()?;

        FreeRtos::delay_ms(2000);

        self.reset_pulse()?;

        self.stop_read_data_continuous()?;

        FreeRtos::delay_ms(1);

        if !self.verify_id(0x92)? {
            anyhow::bail!("ADS1298 ({role:?}) ID mismatch");
        }

        self.configure(role)?;

        FreeRtos::delay_ms(200);

        Ok(())
    }

    pub fn enable_test_signal(&mut self, channel: usize) -> Result<()> {
        debug_assert!(channel < CHANNELS_PER_DEVICE, "channel must be 0..=7, got {channel}");

        self.write_register(Register::Config2, 0x10)?;

        let channels = [
            Register::Ch1Set,
            Register::Ch2Set,
            Register::Ch3Set,
            Register::Ch4Set,
            Register::Ch5Set,
            Register::Ch6Set,
            Register::Ch7Set,
            Register::Ch8Set,
        ];
        for (i, reg) in channels.into_iter().enumerate() {
            let value = if i == channel {
                0x05 
            } else {
                0x01 
            };
            self.write_register(reg, value)?;
        }

        self.write_register(Register::LoffSensP, 0x00)?; 
        self.write_register(Register::LoffSensN, 0x00)?;

        Ok(())
    }
}

/// Owns both ADS1298 devices sharing one Cascaded SPI bus and two
/// control lines. Specifically in this format so that RLD registers 
// for each ADS1298 can be configured differently on startup.
pub(crate) struct Ads1298Pair<'d> {
    pub(crate) adc1: Ads1298Device<'d>,
    pub(crate) adc2: Ads1298Device<'d>,
    start: PinDriver<'d, Output>,
}

impl<'d> Ads1298Pair<'d> {
    pub(crate) fn new(
        adc1: Ads1298Device<'d>,
        adc2: Ads1298Device<'d>,
        start: PinDriver<'d, Output>
    ) -> Self {
        Self {
            adc1,
            adc2,
            start,
        }
    }

    /// Pulls the shared START pin high
    pub(crate) fn start_conversion(&mut self) -> Result<()> {
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
    pub(crate) fn data_ready(&self) -> Result<bool> {
        Ok(self.adc1.data_ready()? && self.adc2.data_ready()?)
    }

    /// Clocks out one frame from each device simultaneously. Only valid while
    /// both are in RDATAC mode and after `data_ready()` reports true
    pub(crate) fn read_frame(&mut self) -> Result<Frame> {
        let s1 = self.adc1.read_frame()?;
        let s2 = self.adc2.read_frame()?;
        Ok(Frame { devices: [s1, s2] })
    }
}
