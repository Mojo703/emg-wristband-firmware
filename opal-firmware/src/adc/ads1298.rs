//! Driver for two TI ADS1298 8-channel ADCs in "Cascade Configuration"
//!
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf>

use anyhow::Result;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{Input, InterruptType, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};
use std::sync::Arc;

use super::decode::parse_sample;
use super::decode::{AdcFrame, Sample, CHANNELS_PER_DEVICE, FRAME_BYTES};

use super::registers::Register;
use super::spi_commands;

/// The SPI handle for one chip. Both chips share one `SpiDriver` through an `Arc`
/// rather than borrowing it, so the whole driver is `'static` and can move into the
/// acquisition thread. `SpiDriver` and `SpiDeviceDriver` are both `Send`, and the
/// bus is serialised internally by esp-idf's own device lock.
type SpiDevice = SpiDeviceDriver<'static, Arc<SpiDriver<'static>>>;

/// The output data rate the register set in [`Ads1298Device::configure`] selects.
///
/// Derivation, so this stays honest if the registers change: CONFIG1 is `0xE4` on chip
/// A and `0xC4` on chip B. Both set `HR = 1` (bit 7, high-resolution mode) and
/// `DR = 0b100` (bits 2:0). In high-resolution mode fMOD is fCLK/4 = 512 kHz, and
/// `DR = 0b100` selects fMOD/256, so 512000/256 = 2000 SPS.
///
/// This is 2000 Hz. The model trained at 2048 Hz, a 2.3% difference in the time base
/// that nobody has yet measured for its effect on accuracy. See
/// [`crate::adc::preprocess`].
pub(crate) const SAMPLE_RATE_HZ: u32 = 2000;

/// Which cascaded device this is — determines which registers differ
/// (CONFIG1.CLK_EN, CONFIG3.PD_RLD, RLD_SENSP/N).
#[derive(Clone, Copy, Debug)]
pub(super) enum ChipRole {
    /// Clock master: drives CLK out to the other device (CONFIG1.CLK_EN=1)
    A,
    /// Clock slave: receives CLK from the master (CONFIG1.CLK_EN=0)
    B,
}

/// One ADS1298 addressed via its own dedicated CS on the shared SPI bus
/// Each ADC also has its own DRDY and RESET pin, however START is tied together
/// so they DRDY pins should pull at same time
pub(super) struct Ads1298Device {
    spi: SpiDevice,
    drdy: PinDriver<'static, Input>,
    reset_n: PinDriver<'static, Output>,
    pwdn: PinDriver<'static, Output>,
}

impl Ads1298Device {
    pub(super) fn new(
        spi: SpiDevice,
        drdy: PinDriver<'static, Input>,
        mut reset_n: PinDriver<'static, Output>,
        mut pwdn: PinDriver<'static, Output>,
    ) -> Result<Self> {
        // Held low from construction, before either chip's `power_up()`/`configure()`
        // runs: both chips share DIN/DOUT/SCLK, so an un-reset chip could drive the
        // bus while its sibling is still being brought up.
        pwdn.set_low()?;
        reset_n.set_low()?;

        Ok(Self {
            spi,
            drdy,
            reset_n,
            pwdn,
        })
    }

    /// DRDY is active-low. It falls when a new frame is ready to be sampled.
    pub(super) fn data_ready(&self) -> Result<bool> {
        Ok(self.drdy.is_low())
    }

    /// Routes this chip's DRDY falling edge to `callback`, which runs in ISR context.
    ///
    /// # Safety
    ///
    /// `callback` runs in an interrupt. It must not call into std, libc or most of
    /// FreeRTOS. Notifying a task is one of the few things it may do.
    pub(super) unsafe fn subscribe_data_ready(
        &mut self,
        callback: impl FnMut() + Send + 'static,
    ) -> Result<()> {
        self.drdy.set_interrupt_type(InterruptType::NegEdge)?;
        unsafe { self.drdy.subscribe(callback)? };
        Ok(())
    }

    /// Arms the DRDY interrupt. esp-idf-hal disables it inside its own ISR to avoid
    /// re-entering, so this has to be called again after every notification, from
    /// outside interrupt context.
    pub(super) fn arm_data_ready_interrupt(&mut self) -> Result<()> {
        self.drdy.enable_interrupt()?;
        Ok(())
    }

    pub(super) fn reset_pulse(&mut self) -> Result<()> {
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

    #[allow(dead_code)] // Datasheet command set, kept whole for bring-up.
    pub(super) fn software_reset(&mut self) -> Result<()> {
        self.send_command(spi_commands::RESET)
    }

    #[allow(dead_code)] // Datasheet command set, kept whole for bring-up.
    pub(super) fn wakeup(&mut self) -> Result<()> {
        self.send_command(spi_commands::WAKEUP)
    }

    #[allow(dead_code)] // Datasheet command set, kept whole for bring-up.
    pub(super) fn standby(&mut self) -> Result<()> {
        self.send_command(spi_commands::STANDBY)
    }

    /// Disables read data continuous mode so that registers can be configured
    pub(super) fn stop_read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::SDATAC)
    }

    /// Enables read data continuous mode. Once started, each DRDY
    /// falling edge means a frame is ready to clock out
    pub(super) fn read_data_continuous(&mut self) -> Result<()> {
        self.send_command(spi_commands::RDATAC)
    }

    // Writes to a single register on the ADS1298
    pub(super) fn write_register(&mut self, reg: Register, value: u8) -> Result<()> {
        // Second byte is "number of registers - 1" (0x00 = one register).
        self.spi
            .write(&[spi_commands::WREG_BASE | reg.addr(), 0x00, value])?;
        Ok(())
    }

    // Reads from a single register on the ADS1298
    pub(super) fn read_register(&mut self, reg: Register) -> Result<u8> {
        let mut rx = [0u8; 1];
        self.spi.transaction(&mut [
            Operation::Write(&[spi_commands::RREG_BASE | reg.addr(), 0x00]),
            Operation::Read(&mut rx),
        ])?;
        Ok(rx[0])
    }

    /// Clocks out one frame. Only valid while in RDATAC mode
    // and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<Sample> {
        let mut raw = [0u8; FRAME_BYTES];
        self.spi.read(&mut raw)?;
        Ok(parse_sample(&raw))
    }

    // Writes this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be written while streaming.
    pub(super) fn configure(&mut self, role: ChipRole) -> Result<()> {
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
    pub(super) fn power_up(&mut self, role: ChipRole, expected_device_id: u8) -> Result<()> {
        // Held low since construction (see `new`); wait out the minimum power-down
        // assertion before bringing the chip up.
        FreeRtos::delay_ms(5);

        self.pwdn.set_high()?;
        self.reset_n.set_high()?;

        FreeRtos::delay_ms(2000);

        self.reset_pulse()?;

        self.stop_read_data_continuous()?;

        FreeRtos::delay_ms(1);

        let device_id = self.read_register(Register::Id)?;
        if device_id != expected_device_id {
            anyhow::bail!(
                "ADS1298 ({role:?}) ID mismatch: read {device_id:#04x}, expected \
                 {expected_device_id:#04x}. {}",
                match device_id {
                    0x00 | 0xFF =>
                        "Bus reads all-zero or all-one, so this is wiring, \
                                    power, or CS rather than a wrong part.",
                    _ => "The bus responds, so check SPI mode and the part number.",
                }
            );
        }

        self.configure(role)?;

        // TEMP bring-up experiment: bumped from 200, same reason as above.
        FreeRtos::delay_ms(1000);

        Ok(())
    }

    pub(super) fn enable_test_signal(&mut self, channel: usize) -> Result<()> {
        debug_assert!(
            channel < CHANNELS_PER_DEVICE,
            "channel must be 0..=7, got {channel}"
        );

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
            let value = if i == channel { 0x05 } else { 0x01 };
            self.write_register(reg, value)?;
        }

        self.write_register(Register::LoffSensP, 0x00)?;
        self.write_register(Register::LoffSensN, 0x00)?;

        // `configure()` includes every channel in the RLD derivation (RLD_SENSP/N =
        // 0xFF on chip A) for real electrode use. With the test signal active, that
        // sums the driven channel's square wave into the shared RLD reference and
        // bleeds an attenuated copy of it back onto every other channel, including
        // ones shorted above. There's no patient loop to cancel common-mode noise on
        // during a test-signal check, so drop every channel out of RLD entirely.
        self.write_register(Register::RldSensP, 0x00)?;
        self.write_register(Register::RldSensN, 0x00)?;

        Ok(())
    }
}

/// Owns both ADS1298 devices sharing one Cascaded SPI bus and two
/// control lines. Specifically in this format so that RLD registers
// for each ADS1298 can be configured differently on startup.
pub(crate) struct Ads1298Pair {
    pub(super) adc1: Ads1298Device,
    pub(super) adc2: Ads1298Device,
    start: PinDriver<'static, Output>,
}

impl Ads1298Pair {
    pub(super) fn new(
        adc1: Ads1298Device,
        adc2: Ads1298Device,
        start: PinDriver<'static, Output>,
    ) -> Self {
        Self { adc1, adc2, start }
    }

    /// Pulls the shared START pin high
    pub(super) fn start_conversion(&mut self) -> Result<()> {
        self.start.set_high()?;
        Ok(())
    }

    /// Pulls the shared START pin low. Nothing stops the chips today, since the
    /// firmware streams until it reboots, but halting conversion is the other half of
    /// `start_conversion` and bring-up will want it.
    #[allow(dead_code)]
    pub(super) fn stop_conversion(&mut self) -> Result<()> {
        self.start.set_low()?;
        Ok(())
    }

    /// True only once both devices report their own DRDY low. Since each
    /// device has its own dedicated DRDY pin, this checks both independently
    /// rather than trusting just one.
    ///
    /// Acquisition does not use this: it waits on chip A's DRDY interrupt and then
    /// checks chip B on its own. Kept because polling both is the obvious thing to
    /// reach for during bring-up.
    #[allow(dead_code)]
    pub(super) fn data_ready(&self) -> Result<bool> {
        Ok(self.adc1.data_ready()? && self.adc2.data_ready()?)
    }

    /// Clocks out one frame from each device simultaneously. Only valid while
    /// both are in RDATAC mode and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<AdcFrame> {
        let s1 = self.adc1.read_frame()?;
        let s2 = self.adc2.read_frame()?;
        Ok(AdcFrame { devices: [s1, s2] })
    }
}
