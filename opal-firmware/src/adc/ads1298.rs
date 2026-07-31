//! Driver for two TI ADS1298 8-channel ADCs in "Cascade Configuration"
//!
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf>

use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{Input, InterruptType, Output, PinDriver};
use esp_idf_svc::hal::spi::{SpiDeviceDriver, SpiDriver};
use log::{info, warn};
use std::sync::Arc;

use super::channel::Channel;
use super::decode::parse_sample;
use super::decode::{AdcFrame, Sample, FRAME_BYTES};

use super::registers::{
    ChannelInput, ChannelMask, ChannelSettings, Config1, Config2, Config3, Config4, DataRate, Gain,
    GeneralPurposeInputOutputOff, LeadOffComparatorThreshold, LeadOffControl, LeadOffCurrent,
    LeadOffDetection, LeadOffFlip, LeadOffSenseNegative, LeadOffSensePositive, PaceDetectOff,
    Register, RegisterValue, Respiration, RespirationControl, RespirationFrequency,
    RespirationPhase, RightLegDriveSenseNegative, RightLegDriveSensePositive, TestSignalAmplitude,
    TestSignalFrequency, WilsonCenterTerminalOneOff, WilsonCenterTerminalTwoOff,
};
use super::spi_commands;

/// The SPI handle for one chip. Both chips share one `SpiDriver` through an `Arc`
/// rather than borrowing it, so the whole driver is `'static` and can move into the
/// acquisition thread. `SpiDriver` and `SpiDeviceDriver` are both `Send`, and the
/// bus is serialised internally by esp-idf's own device lock.
type SpiDevice = SpiDeviceDriver<'static, Arc<SpiDriver<'static>>>;

/// Gap between the bytes of a multi-byte register command. The ADS1298's command
/// decoder needs tSDECODE = 4 tCLK (~2 µs at 2.048 MHz) per byte, and clocking a
/// multi-byte command as one continuous stream is only legal when the SCLK is slow
/// enough to provide that inside the byte itself (SBAS459K §9.5.1.2.1). The bench
/// proved this the hard way: without burst framing, 2 MHz and 4 MHz SPI read the ID
/// register as 0x00. 5 µs rather than 2 for margin; register traffic never sits on
/// the frame-read hot path.
const COMMAND_DECODE_GAP_US: u32 = 5;

/// The output data rate the register set in [`Ads1298Device::configure`] selects.
///
/// Derived from chip A's actual CONFIG1 rather than written down beside it, so it
/// cannot drift out of agreement with the register the driver writes: change
/// `data_rate` or `high_resolution` below and this number follows.
///
/// It comes out at 2000 Hz. The model trained at 2048 Hz, a 2.3% difference in the time
/// base that nobody has yet measured for its effect on accuracy. See
/// [`crate::adc::preprocess`].
pub(crate) const SAMPLE_RATE_HZ: u32 = {
    let config1 = config1_for(ChipRole::A);
    config1
        .data_rate
        .samples_per_second(config1.high_resolution)
};

/// CONFIG1 for a role. Chip A drives the clock out, chip B receives it. Both run in
/// high resolution at fMOD/256, which is where [`SAMPLE_RATE_HZ`] comes from. The reset
/// default is `0x06` — low power at fMOD/1024 — which converts at 250 SPS, so a chip
/// that has reverted is visible both in a readback and in the edge rate.
const fn config1_for(role: ChipRole) -> Config1 {
    Config1 {
        high_resolution: true,
        // Each chip has its own chip select, so they are read back individually rather
        // than daisy-chained through a single DOUT.
        multiple_readback: true,
        output_clock_enabled: match role {
            ChipRole::A => true,
            ChipRole::B => false,
        },
        data_rate: DataRate::ModulatorClockOver256,
    }
}

/// How chip A drives the subject reference. A TEMP bring-up experiment: chip A resets
/// itself to power-on defaults a variable 0.2-8 s after conversions start, and the
/// right-leg drive is the largest analog load that distinguishes chip A from chip B,
/// which never resets.
///
/// Each board makes its own +-2.5 V analog rails locally (TLV70025 and a TPS72325 fed
/// by a TPS60400 charge pump), so an analog supply that sags under load explains one
/// chip failing while the other does not. This is the knob that changes that load.
///
/// Sweep all three against a scope on AVSS. One run proves nothing here — the failure
/// time varies by more than an order of magnitude between boots, and an earlier
/// single-run comparison of `Disabled` against `ExternalReference` looked worse but was
/// well inside that noise.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RightLegDriveMode {
    /// Amplifier powered, reference taken from the RLDREF pin (`0xC6`). What the bench
    /// has been running. Suspect: a datasheet audit flagged that nothing is known to
    /// drive RLDREF, which would leave the amplifier's reference input floating and its
    /// output railed.
    ExternalReference,
    /// Amplifier powered, reference generated internally at mid-supply (`0xCE`). What
    /// the configuration probably should have been all along if the board has no
    /// RLDREF divider.
    InternalReference,
    /// Amplifier powered down entirely (`0xC0`), the same as chip B. Removes the load
    /// rather than fixing it — a diagnostic, not a destination, since the right-leg
    /// drive is what rejects common-mode noise once a subject is attached.
    Disabled,
}

/// The mode chip A is running right now. Chip B never powers the block whatever this
/// says.
const RIGHT_LEG_DRIVE_MODE: RightLegDriveMode = RightLegDriveMode::ExternalReference;

/// CONFIG3 for a role. Both chips run their internal reference buffer off the 2.4 V
/// reference; the difference is the right-leg drive, which chip A powers and chip B
/// does not.
///
/// One amplifier drives the subject's reference for the whole board, so exactly one
/// chip may own it. Chip A does, alongside the channels it sums into that drive
/// (`right_leg_drive_channels`) and the lead-off sense that rides on the same
/// amplifier. Chip B leaves the block powered down.
const fn config3_for(role: ChipRole) -> Config3 {
    config3_for_mode(role, RIGHT_LEG_DRIVE_MODE)
}

/// [`config3_for`] with the experiment's setting passed in, so the tests can pin every
/// mode's byte rather than only whichever one the bench happens to be running.
const fn config3_for_mode(role: ChipRole, mode: RightLegDriveMode) -> Config3 {
    let drive_enabled = match role {
        ChipRole::A => !matches!(mode, RightLegDriveMode::Disabled),
        ChipRole::B => false,
    };
    Config3 {
        internal_reference_enabled: true,
        // The 2.4 V reference, which is what `preprocess` converts codes against.
        four_volt_reference: false,
        right_leg_drive_measurement: false,
        right_leg_drive_reference_internal: match role {
            ChipRole::A => matches!(mode, RightLegDriveMode::InternalReference),
            ChipRole::B => false,
        },
        right_leg_drive_enabled: drive_enabled,
        // The lead-off sense rides on the same amplifier, so it goes wherever the
        // drive goes.
        right_leg_drive_lead_off_sense: drive_enabled,
    }
}

/// RLD_SENSP/N for a role: chip A sums every channel into the right-leg drive, chip B
/// none of them, so exactly one amplifier defines the reference for the whole board.
const fn right_leg_drive_channels(role: ChipRole) -> ChannelMask {
    match role {
        ChipRole::A => ChannelMask::ALL,
        ChipRole::B => ChannelMask::NONE,
    }
}

/// CONFIG2 with the internal test signal generator off, which is normal operation.
const NORMAL_CONFIG2: Config2 = Config2 {
    chopping_frequency_constant: false,
    test_signal_enabled: false,
    test_signal_amplitude: TestSignalAmplitude::Single,
    test_signal_frequency: TestSignalFrequency::PulsedSlow,
};

/// LOFF: DC lead-off detection, comparators at the widest thresholds, in resistor mode
/// so no excitation current is pushed into the electrode.
const LEAD_OFF_CONTROL: LeadOffControl = LeadOffControl {
    comparator_threshold: LeadOffComparatorThreshold::NinetyFivePercent,
    pull_resistor_mode: true,
    current: LeadOffCurrent::SixNanoamps,
    detection: LeadOffDetection::DirectCurrent,
};

/// CHnSET for normal operation: channel powered up, gain 6, reading its electrode pair.
const NORMAL_CHANNEL_SETTINGS: ChannelSettings = ChannelSettings {
    powered_down: false,
    gain: Gain::Six,
    input: ChannelInput::Electrode,
};

/// CONFIG4: continuous conversion, and the lead-off comparators powered up — without
/// them the status word's lead-off bits never set however LOFF_SENSP/N are configured,
/// and `preprocess` would pass a disconnected electrode's railed signal to the model.
const CONFIG4: Config4 = Config4 {
    respiration_frequency: RespirationFrequency::SixtyFourKilohertz,
    single_shot: false,
    wilson_center_terminal_to_right_leg_drive: false,
    lead_off_comparators_enabled: true,
};

/// RESP: the respiration block off. The byte is not zero only because bit 5 is reserved
/// and must be written 1.
const RESPIRATION_OFF: Respiration = Respiration {
    demodulation_enabled: false,
    modulation_enabled: false,
    phase: RespirationPhase::Zero,
    control: RespirationControl::None,
};

/// CONFIG2 for [`Ads1298Device::enable_test_signal`]: the internal generator switched
/// on, at its default amplitude and rate. Without `test_signal_enabled` a channel muxed
/// to the test signal reads an input pin this board does not drive.
const TEST_SIGNAL_CONFIG2: Config2 = Config2 {
    test_signal_enabled: true,
    ..NORMAL_CONFIG2
};

/// CHnSET during a test-signal check: the one channel under test is muxed to the
/// internal square wave, and every other channel is shorted so it reads its own noise
/// floor rather than a floating electrode.
const fn test_signal_channel_settings(driven: bool) -> ChannelSettings {
    ChannelSettings {
        input: if driven {
            ChannelInput::TestSignal
        } else {
            ChannelInput::Shorted
        },
        ..NORMAL_CHANNEL_SETTINGS
    }
}

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

    /// Warm recovery for a chip whose conversions have died mid-session: RESET pulse,
    /// the post-reset lockout, SDATAC, full reconfigure. The bench measured the whole
    /// pair recovering in ~3 ms; the cold-start supply settling in `power_up` is not
    /// repeated because the rails have long since settled.
    pub(super) fn warm_reset(&mut self, role: ChipRole) -> Result<()> {
        self.reset_pulse()?;
        // 18 tCLK after RESET rises before any command (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);
        self.stop_read_data_continuous()?;
        self.configure(role)
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

    /// Writes one typed register value. The address comes from the value's type, so
    /// there is no way to pair a byte with the wrong register here, and the read-only
    /// registers (ID, LOFF_STATP, LOFF_STATN) cannot be reached at all: no value type
    /// names them. See [`super::registers`].
    pub(super) fn write_register<V: RegisterValue>(&mut self, value: V) -> Result<()> {
        self.write_register_byte(V::ADDRESS, value.to_byte())
    }

    /// CH1SET..CH8SET are eight addresses sharing one value type, so the channel comes
    /// in alongside the value instead of being baked into it.
    pub(super) fn write_channel_settings(
        &mut self,
        channel: Channel,
        settings: ChannelSettings,
    ) -> Result<()> {
        self.write_register_byte(Register::channel_settings(channel), settings.to_byte())
    }

    // The one place a register address and a byte meet. Private, so every caller comes
    // through a typed path above.
    fn write_register_byte(&mut self, reg: Register, value: u8) -> Result<()> {
        // Second byte is "number of registers - 1" (0x00 = one register). Burst-framed:
        // one byte per transfer with a decode gap between, per SBAS459K §9.5.1.2.1.
        self.write_bytes_burst_framed(&[spi_commands::WREG_BASE | reg.addr(), 0x00, value])
    }

    // Reads from a single register on the ADS1298
    pub(super) fn read_register(&mut self, reg: Register) -> Result<u8> {
        self.write_bytes_burst_framed(&[spi_commands::RREG_BASE | reg.addr(), 0x00])?;
        let mut rx = [0u8; 1];
        self.spi.read(&mut rx)?;
        Ok(rx[0])
    }

    /// Clocks bytes out one at a time with [`COMMAND_DECODE_GAP_US`] between them —
    /// the datasheet's burst method for multi-byte commands.
    fn write_bytes_burst_framed(&mut self, bytes: &[u8]) -> Result<()> {
        for &byte in bytes {
            self.spi.write(&[byte])?;
            Ets::delay_us(COMMAND_DECODE_GAP_US);
        }
        Ok(())
    }

    /// Clocks out one frame. Only valid while in RDATAC mode
    // and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<Sample> {
        let mut raw = [0u8; FRAME_BYTES];
        self.spi.read(&mut raw)?;
        Ok(parse_sample(&raw))
    }

    /// Reads back the registers the sample rate and the status word depend on, and logs
    /// what actually landed next to what was written.
    ///
    /// `configure` writes 23 registers and checks none of them. A write that silently
    /// fails is invisible until it shows up as a wrong conversion rate, which is a long
    /// way downstream of the cause. Must be called in SDATAC, before RDATAC starts the
    /// stream — registers cannot be read while streaming.
    pub(super) fn log_configuration_readback(&mut self, role: ChipRole) {
        let expected = [
            (Register::Config1, config1_for(role).to_byte()),
            (Register::Config2, NORMAL_CONFIG2.to_byte()),
            (Register::Config3, config3_for(role).to_byte()),
            (
                Register::LoffSensP,
                LeadOffSensePositive(ChannelMask::ALL).to_byte(),
            ),
        ];
        for (register, written) in expected {
            match self.read_register(register) {
                Ok(read) if read == written => {
                    info!("chip {role:?}: {register:?} = {read:#04x} as written")
                }
                Ok(read) => warn!(
                    "chip {role:?}: {register:?} wrote {written:#04x}, reads {read:#04x} -- the write did not take"
                ),
                Err(error) => warn!("chip {role:?}: {register:?} readback failed: {error}"),
            }
        }
    }

    // Writes this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be written while streaming.
    pub(super) fn configure(&mut self, role: ChipRole) -> Result<()> {
        // --- Role-specific registers ---
        self.write_register(config1_for(role))?;

        self.write_register(config3_for(role))?;

        self.write_register(RightLegDriveSensePositive(right_leg_drive_channels(role)))?;
        self.write_register(RightLegDriveSenseNegative(right_leg_drive_channels(role)))?;

        // Shared registers (identical on both chips)
        self.write_register(NORMAL_CONFIG2)?;
        self.write_register(LEAD_OFF_CONTROL)?;

        for channel in Channel::ALL {
            self.write_channel_settings(channel, NORMAL_CHANNEL_SETTINGS)?;
        }

        // Watch both ends of every differential pair for lead-off, with no channel's
        // excitation direction flipped.
        self.write_register(LeadOffSensePositive(ChannelMask::ALL))?;
        self.write_register(LeadOffSenseNegative(ChannelMask::ALL))?;
        self.write_register(LeadOffFlip(ChannelMask::NONE))?;
        self.write_register(GeneralPurposeInputOutputOff)?;
        self.write_register(PaceDetectOff)?;
        self.write_register(RESPIRATION_OFF)?;
        self.write_register(CONFIG4)?;
        self.write_register(WilsonCenterTerminalOneOff)?;
        self.write_register(WilsonCenterTerminalTwoOff)?;

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

        // 18 tCLK (~9 µs at fCLK = 2.048 MHz) are needed after RESET returns high for
        // the device to finish initialising its registers, and no command may be sent
        // during that window (SBAS459K §9.3.2.3, §11.1). This delay used to sit *after*
        // the SDATAC below, which put that command inside the forbidden window.
        FreeRtos::delay_ms(1);

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

    pub(super) fn enable_test_signal(&mut self, channel: Channel) -> Result<()> {
        self.write_register(TEST_SIGNAL_CONFIG2)?;

        for candidate in Channel::ALL {
            self.write_channel_settings(
                candidate,
                test_signal_channel_settings(candidate == channel),
            )?;
        }

        self.write_register(LeadOffSensePositive(ChannelMask::NONE))?;
        self.write_register(LeadOffSenseNegative(ChannelMask::NONE))?;

        // `configure()` includes every channel in the RLD derivation (RLD_SENSP/N =
        // 0xFF on chip A) for real electrode use. With the test signal active, that
        // sums the driven channel's square wave into the shared RLD reference and
        // bleeds an attenuated copy of it back onto every other channel, including
        // ones shorted above. There's no patient loop to cancel common-mode noise on
        // during a test-signal check, so drop every channel out of RLD entirely.
        self.write_register(RightLegDriveSensePositive(ChannelMask::NONE))?;
        self.write_register(RightLegDriveSenseNegative(ChannelMask::NONE))?;

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

    /// Pulls the shared START pin low. Used by [`Self::warm_recover`].
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
    ///
    /// Careful where you call it. Both chips share SCLK, and the ADS1298 clears DRDY on
    /// the first SCLK falling edge whether or not that chip is selected (SBAS459K
    /// §9.4.1.2). So once either chip has been read this conversion cycle, both DRDY
    /// lines read high and this returns false for reasons that have nothing to do with
    /// the data.
    #[allow(dead_code)]
    pub(super) fn data_ready(&self) -> Result<bool> {
        Ok(self.adc1.data_ready()? && self.adc2.data_ready()?)
    }

    /// Warm-recovers both chips after a mid-session death: conversions stopped, then
    /// per-chip RESET/SDATAC/reconfigure, RDATAC, and START again. The bring-up
    /// campaign (documentation/ads1298-bringup-2026-07-31/) established that the front
    /// end dies stochastically under multi-channel conversion and that this recovery
    /// restores it within a few milliseconds, yielding 97% verified frames at
    /// 2 kSPS × 8 channels on the bench.
    pub(super) fn warm_recover(&mut self) -> Result<()> {
        self.stop_conversion()?;
        self.adc1.warm_reset(ChipRole::A)?;
        self.adc2.warm_reset(ChipRole::B)?;
        self.adc1.read_data_continuous()?;
        self.adc2.read_data_continuous()?;
        self.start_conversion()?;
        Ok(())
    }

    /// Clocks out one frame from each device simultaneously. Only valid while
    /// both are in RDATAC mode and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<AdcFrame> {
        let s1 = self.adc1.read_frame()?;
        let s2 = self.adc2.read_frame()?;
        Ok(AdcFrame { devices: [s1, s2] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte `configure` puts on the wire, asserted against the values the bench
    /// is running today.
    ///
    /// This test is the point of the typed register layer. The chips are mid-bring-up
    /// and the register set is an experiment in progress: a byte that shifts because
    /// someone renamed a field or forgot a reserved bit would corrupt the investigation
    /// rather than break the build, and it would do it silently, days before anyone
    /// noticed the data looked wrong. So the wire format is pinned here, by number,
    /// with the field names beside it -- change a field on purpose and this test tells
    /// you exactly which byte moved.
    #[test]
    fn configure_writes_the_bytes_the_bench_is_running() {
        // CONFIG1: high resolution, multiple readback, fMOD/256. Chip A also drives
        // the clock out to chip B, which is the only difference between the two.
        assert_eq!(config1_for(ChipRole::A).to_byte(), 0xE4);
        assert_eq!(config1_for(ChipRole::B).to_byte(), 0xC4);

        // CONFIG2: the internal test signal generator off.
        assert_eq!(NORMAL_CONFIG2.to_byte(), 0x00);

        // CONFIG3: chip A powers the right-leg drive and the lead-off sense that rides
        // on it, chip B does not. The two bytes differ only in those two bits.
        assert_eq!(config3_for(ChipRole::A).to_byte(), 0xC6);
        assert_eq!(config3_for(ChipRole::B).to_byte(), 0xC0);
    }

    /// Every setting of the right-leg-drive experiment, pinned by number. Whichever one
    /// the bench is running, the other two still have to emit the byte the datasheet
    /// says they do — otherwise switching modes mid-investigation silently changes
    /// something else as well.
    #[test]
    fn each_right_leg_drive_mode_emits_its_datasheet_byte() {
        use RightLegDriveMode::{Disabled, ExternalReference, InternalReference};

        assert_eq!(
            config3_for_mode(ChipRole::A, ExternalReference).to_byte(),
            0xC6
        );
        assert_eq!(
            config3_for_mode(ChipRole::A, InternalReference).to_byte(),
            0xCE
        );
        assert_eq!(config3_for_mode(ChipRole::A, Disabled).to_byte(), 0xC0);

        // Chip B never powers the block, whatever the experiment is set to.
        for mode in [ExternalReference, InternalReference, Disabled] {
            assert_eq!(config3_for_mode(ChipRole::B, mode).to_byte(), 0xC0);
        }

        // LOFF: DC lead-off detection in resistor mode.
        assert_eq!(LEAD_OFF_CONTROL.to_byte(), 0x13);

        // CH1SET..CH8SET: powered up, gain 6, on the electrode.
        assert_eq!(NORMAL_CHANNEL_SETTINGS.to_byte(), 0x00);

        // RLD_SENSP/N: every channel on chip A, none on chip B.
        assert_eq!(
            RightLegDriveSensePositive(right_leg_drive_channels(ChipRole::A)).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSenseNegative(right_leg_drive_channels(ChipRole::A)).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSensePositive(right_leg_drive_channels(ChipRole::B)).to_byte(),
            0x00
        );
        assert_eq!(
            RightLegDriveSenseNegative(right_leg_drive_channels(ChipRole::B)).to_byte(),
            0x00
        );

        // LOFF_SENSP/N and LOFF_FLIP.
        assert_eq!(LeadOffSensePositive(ChannelMask::ALL).to_byte(), 0xFF);
        assert_eq!(LeadOffSenseNegative(ChannelMask::ALL).to_byte(), 0xFF);
        assert_eq!(LeadOffFlip(ChannelMask::NONE).to_byte(), 0x00);

        // The blocks this board leaves off.
        assert_eq!(GeneralPurposeInputOutputOff.to_byte(), 0x00);
        assert_eq!(PaceDetectOff.to_byte(), 0x00);
        assert_eq!(WilsonCenterTerminalOneOff.to_byte(), 0x00);
        assert_eq!(WilsonCenterTerminalTwoOff.to_byte(), 0x00);

        // RESP is 0x20 rather than 0x00 purely because bit 5 is a reserved one.
        assert_eq!(RESPIRATION_OFF.to_byte(), 0x20);

        // CONFIG4: continuous conversion with the lead-off comparators powered.
        assert_eq!(CONFIG4.to_byte(), 0x02);
    }

    /// The same pinning for the test-signal path in `enable_test_signal`.
    #[test]
    fn the_test_signal_path_writes_the_bytes_it_used_to() {
        assert_eq!(TEST_SIGNAL_CONFIG2.to_byte(), 0x10);
        assert_eq!(test_signal_channel_settings(true).to_byte(), 0x05);
        assert_eq!(test_signal_channel_settings(false).to_byte(), 0x01);
        assert_eq!(LeadOffSensePositive(ChannelMask::NONE).to_byte(), 0x00);
        assert_eq!(LeadOffSenseNegative(ChannelMask::NONE).to_byte(), 0x00);
        assert_eq!(
            RightLegDriveSensePositive(ChannelMask::NONE).to_byte(),
            0x00
        );
        assert_eq!(
            RightLegDriveSenseNegative(ChannelMask::NONE).to_byte(),
            0x00
        );
    }

    /// Every register value the driver writes has to survive a round trip through the
    /// chip's own byte, or a readback cannot be compared against what was intended.
    #[test]
    fn written_bytes_decode_back_to_the_values_that_produced_them() {
        assert_eq!(
            Config1::from_byte(config1_for(ChipRole::A).to_byte()),
            Some(config1_for(ChipRole::A))
        );
        assert_eq!(
            Config3::from_byte(config3_for(ChipRole::B).to_byte()),
            Some(config3_for(ChipRole::B))
        );
        assert_eq!(
            Config2::from_byte(TEST_SIGNAL_CONFIG2.to_byte()),
            Some(TEST_SIGNAL_CONFIG2)
        );
        assert_eq!(Config4::from_byte(CONFIG4.to_byte()), Some(CONFIG4));
        assert_eq!(
            LeadOffControl::from_byte(LEAD_OFF_CONTROL.to_byte()),
            Some(LEAD_OFF_CONTROL)
        );
        assert_eq!(
            Respiration::from_byte(RESPIRATION_OFF.to_byte()),
            Some(RESPIRATION_OFF)
        );
        assert_eq!(
            ChannelSettings::from_byte(NORMAL_CHANNEL_SETTINGS.to_byte()),
            Some(NORMAL_CHANNEL_SETTINGS)
        );
    }

    /// `SAMPLE_RATE_HZ` is derived from chip A's CONFIG1 rather than written down, so
    /// this asserts the derivation lands where the hand arithmetic used to.
    #[test]
    fn sample_rate_is_derived_from_the_configured_data_rate() {
        assert_eq!(SAMPLE_RATE_HZ, 2000);
    }
}
