//! Driver for the single TI ADS1298 8-channel ADC front end.
//!
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf> (SBAS459K).
//!
//! One chip, self-clocked (CLKSEL strapped high), eight channels. The two-chip
//! cascade this driver grew up around is gone: the bring-up campaign
//! (documentation/ads1298-bringup-2026-07-31/) ran on one board, the product bench
//! has one board, and the second slot of the model's 16 input channels is
//! zero-padded until the model is retrained at 8.

use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{Input, InterruptType, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};
use log::{info, warn};
use std::sync::Arc;

use super::channel::Channel;
use super::decode::parse_sample;
use super::decode::{Sample, FRAME_BYTES};

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
///
/// The gaps must sit INSIDE one `transaction()`, which holds hardware CS asserted
/// across its operations. Splitting the bytes into separate `write()` calls raises CS
/// between them — and a CS rising edge resets the chip's command decoder (§9.5.1.1),
/// which kills every multi-byte command at its first byte boundary. That exact bug
/// shipped once and read every register as 0x00.
const COMMAND_DECODE_GAP_NANOSECONDS: u32 = 5_000;

/// Gap before CS rises after a transaction, and the minimum CS-high dwell after it.
const CHIP_SELECT_GAP_MICROSECONDS: u32 = 5;

/// The output data rate the register set in [`Ads1298Device::configure`] selects.
///
/// Derived from the actual CONFIG1 rather than written down beside it, so it cannot
/// drift out of agreement with the register the driver writes: change `data_rate` or
/// `high_resolution` below and this number follows.
///
/// It comes out at 2000 Hz. The model trained at 2048 Hz, a 2.3% difference in the time
/// base that nobody has yet measured for its effect on accuracy. See
/// [`crate::adc::preprocess`].
pub(crate) const SAMPLE_RATE_HZ: u32 = {
    let config1 = device_config1();
    config1
        .data_rate
        .samples_per_second(config1.high_resolution)
};

/// CONFIG1: high resolution at fMOD/256, which is where [`SAMPLE_RATE_HZ`] comes from.
/// The clock output stays off — nothing listens to the CLK pin with one self-clocked
/// chip. The reset default is `0x06` — low power at fMOD/1024 — which converts at
/// 250 SPS, so a chip that has silently reverted is visible both in a readback and in
/// the DRDY edge rate; acquisition's slow-period recovery trigger keys on exactly that.
const fn device_config1() -> Config1 {
    Config1 {
        high_resolution: true,
        multiple_readback: true,
        output_clock_enabled: false,
        data_rate: DataRate::ModulatorClockOver256,
    }
}

/// How the chip drives the subject bias reference.
///
/// TODO(bias drive): the subject bias / right-leg drive is OFF. On-skin sessions run
/// without common-mode rejection, so expect visibly more 50/60 Hz mains pickup in
/// collected data. Before enabling it: (1) fix the board — the compensation network
/// (1 MΩ ∥ 1 nF) must move from RLDIN to RLDINV per SBAS459K figure 94; (2) validate
/// the powered loop on the bench with the survival harness, since RLD load was never
/// exonerated in the conversion-death campaign; (3) pick Internal vs External
/// reference against the schematic (RLDREF is grounded, which only suits internal).
///
/// `Disabled` is the only bench-validated setting: the entire bring-up campaign ran
/// with the amplifier off, and the PCB review found the drive's compensation network
/// on the wrong pin (RLDIN, the monitor mux, instead of RLDINV, the feedback node —
/// SBAS459K figure 94), so the loop's stability when powered is unverified. Enabling
/// it is its own future experiment; expect worse mains rejection until then.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RightLegDriveMode {
    /// Amplifier powered, reference taken from the RLDREF pin (`0xC6`). RLDREF is
    /// grounded on this board, which is only correct with the internal reference
    /// selected — do not use this mode without re-reading the schematic.
    #[allow(dead_code)]
    ExternalReference,
    /// Amplifier powered, reference generated internally at mid-supply (`0xCE`).
    #[allow(dead_code)]
    InternalReference,
    /// Amplifier powered down entirely (`0xC0`). The validated configuration.
    Disabled,
}

/// The mode the chip is running right now.
const RIGHT_LEG_DRIVE_MODE: RightLegDriveMode = RightLegDriveMode::Disabled;

/// CONFIG3: the internal reference buffer on the 2.4 V reference, and the right-leg
/// drive wherever [`RIGHT_LEG_DRIVE_MODE`] puts it.
const fn device_config3() -> Config3 {
    config3_for_mode(RIGHT_LEG_DRIVE_MODE)
}

/// [`device_config3`] with the setting passed in, so the tests can pin every mode's
/// byte rather than only whichever one the bench happens to be running.
const fn config3_for_mode(mode: RightLegDriveMode) -> Config3 {
    let drive_enabled = !matches!(mode, RightLegDriveMode::Disabled);
    Config3 {
        internal_reference_enabled: true,
        // The 2.4 V reference, which is what `preprocess` converts codes against.
        four_volt_reference: false,
        right_leg_drive_measurement: false,
        right_leg_drive_reference_internal: matches!(mode, RightLegDriveMode::InternalReference),
        right_leg_drive_enabled: drive_enabled,
        // The lead-off sense rides on the same amplifier, so it goes wherever the
        // drive goes.
        right_leg_drive_lead_off_sense: drive_enabled,
    }
}

/// RLD_SENSP/N: which channels sum into the right-leg drive. Nothing while the drive
/// is off; every channel once a powered mode is validated.
const fn right_leg_drive_channels() -> ChannelMask {
    match RIGHT_LEG_DRIVE_MODE {
        RightLegDriveMode::Disabled => ChannelMask::NONE,
        _ => ChannelMask::ALL,
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

/// TODO(lead-off): detection is OFF. The bring-up campaign validated every stable
/// operating point with the lead-off block unpowered, and the first firmware run with
/// it on died ~35 times per second versus ~2 on the bench. Disconnected electrodes
/// therefore rail their channels instead of being zeroed by `preprocess` — re-enable
/// as its own bench experiment (flip LEAD_OFF_ENABLED) once the front end is stable
/// enough to isolate its effect.
const LEAD_OFF_ENABLED: bool = false;

/// CONFIG4: continuous conversion; lead-off comparators per [`LEAD_OFF_ENABLED`].
const CONFIG4: Config4 = Config4 {
    respiration_frequency: RespirationFrequency::SixtyFourKilohertz,
    single_shot: false,
    wilson_center_terminal_to_right_leg_drive: false,
    lead_off_comparators_enabled: LEAD_OFF_ENABLED,
};

/// LOFF_SENSP/N: which input halves the comparators watch.
const fn lead_off_channels() -> ChannelMask {
    if LEAD_OFF_ENABLED {
        ChannelMask::ALL
    } else {
        ChannelMask::NONE
    }
}

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

/// The one ADS1298, addressed via its dedicated CS.
///
/// CS is a GPIO the driver toggles itself, not the SPI peripheral's hardware CS.
/// This is the mechanism the bring-up harness validated end to end: held low across
/// a whole multi-byte command (whose bytes need tSDECODE gaps between them), raised
/// between transactions so the chip's command decoder gets its reset edge
/// (SBAS459K §9.5.1.1).
pub(super) struct Ads1298Device {
    spi: SpiDevice,
    chip_select: PinDriver<'static, Output>,
    drdy: PinDriver<'static, Input>,
    reset_n: PinDriver<'static, Output>,
    pwdn: PinDriver<'static, Output>,
}

impl Ads1298Device {
    pub(super) fn new(
        spi: SpiDevice,
        mut chip_select: PinDriver<'static, Output>,
        drdy: PinDriver<'static, Input>,
        mut reset_n: PinDriver<'static, Output>,
        mut pwdn: PinDriver<'static, Output>,
    ) -> Result<Self> {
        // Held in reset from construction until `power_up()` runs the sequence.
        pwdn.set_low()?;
        reset_n.set_low()?;
        chip_select.set_high()?;

        Ok(Self {
            spi,
            chip_select,
            drdy,
            reset_n,
            pwdn,
        })
    }

    /// Runs one transaction with CS low, raising CS afterwards even on failure so the
    /// decoder-reset edge is never skipped. The trailing gap covers tSCCS (4 tCLK
    /// after the last SCLK before CS may rise) and the minimum CS-high pulse.
    fn with_selection<T>(&mut self, transaction: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.chip_select.set_low()?;
        let result = transaction(self);
        Ets::delay_us(CHIP_SELECT_GAP_MICROSECONDS);
        self.chip_select.set_high()?;
        Ets::delay_us(CHIP_SELECT_GAP_MICROSECONDS);
        result
    }

    /// DRDY is active-low. It falls when a new frame is ready to be sampled.
    /// Acquisition waits on the interrupt instead; kept for bring-up polling.
    #[allow(dead_code)]
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
    pub(super) fn warm_reset(&mut self) -> Result<()> {
        self.reset_pulse()?;
        // 18 tCLK after RESET rises before any command (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);
        self.stop_read_data_continuous()?;
        self.configure()
    }

    /// Helper for sending SPI commands
    fn send_command(&mut self, command: u8) -> Result<()> {
        self.with_selection(|device| {
            device.spi.write(&[command])?;
            Ok(())
        })
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
        // Second byte is "number of registers - 1" (0x00 = one register). CS held for
        // the whole command, decode gaps between the bytes.
        self.with_selection(|device| {
            device.spi.transaction(&mut [
                Operation::Write(&[spi_commands::WREG_BASE | reg.addr()]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[0x00]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[value]),
            ])?;
            Ok(())
        })
    }

    // Reads from a single register on the ADS1298
    pub(super) fn read_register(&mut self, reg: Register) -> Result<u8> {
        self.with_selection(|device| {
            let mut rx = [0u8; 1];
            device.spi.transaction(&mut [
                Operation::Write(&[spi_commands::RREG_BASE | reg.addr()]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[0x00]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Read(&mut rx),
            ])?;
            Ok(rx[0])
        })
    }

    /// Clocks out one frame. Only valid while in RDATAC mode
    // and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<Sample> {
        self.with_selection(|device| {
            let mut raw = [0u8; FRAME_BYTES];
            device.spi.read(&mut raw)?;
            Ok(parse_sample(&raw))
        })
    }

    /// Reads back the registers the sample rate and the status word depend on, and logs
    /// what actually landed next to what was written.
    ///
    /// `configure` writes 23 registers and checks none of them. A write that silently
    /// fails is invisible until it shows up as a wrong conversion rate, which is a long
    /// way downstream of the cause. Must be called in SDATAC, before RDATAC starts the
    /// stream — registers cannot be read while streaming.
    pub(super) fn log_configuration_readback(&mut self) {
        let expected = [
            (Register::Config1, device_config1().to_byte()),
            (Register::Config2, NORMAL_CONFIG2.to_byte()),
            (Register::Config3, device_config3().to_byte()),
            (
                Register::LoffSensP,
                LeadOffSensePositive(lead_off_channels()).to_byte(),
            ),
        ];
        for (register, written) in expected {
            match self.read_register(register) {
                Ok(read) if read == written => {
                    info!("{register:?} = {read:#04x} as written")
                }
                Ok(read) => warn!(
                    "{register:?} wrote {written:#04x}, reads {read:#04x} -- the write did not take"
                ),
                Err(error) => warn!("{register:?} readback failed: {error}"),
            }
        }
    }

    // Writes this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be written while streaming.
    pub(super) fn configure(&mut self) -> Result<()> {
        self.write_register(device_config1())?;

        self.write_register(device_config3())?;

        self.write_register(RightLegDriveSensePositive(right_leg_drive_channels()))?;
        self.write_register(RightLegDriveSenseNegative(right_leg_drive_channels()))?;

        self.write_register(NORMAL_CONFIG2)?;
        self.write_register(LEAD_OFF_CONTROL)?;

        for channel in Channel::ALL {
            self.write_channel_settings(channel, NORMAL_CHANNEL_SETTINGS)?;
        }

        // Both ends of every differential pair when lead-off is enabled, nothing
        // while it is off (see LEAD_OFF_ENABLED).
        self.write_register(LeadOffSensePositive(lead_off_channels()))?;
        self.write_register(LeadOffSenseNegative(lead_off_channels()))?;
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
    pub(super) fn power_up(&mut self, expected_device_id: u8) -> Result<()> {
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
                "ADS1298 ID mismatch: read {device_id:#04x}, expected \
                 {expected_device_id:#04x}. {}",
                match device_id {
                    0x00 | 0xFF =>
                        "Bus reads all-zero or all-one, so this is wiring, \
                                    power, or CS rather than a wrong part.",
                    _ => "The bus responds, so check SPI mode and the part number.",
                }
            );
        }

        self.configure()?;

        // Internal reference settling: the datasheet's 150 ms start-up time with
        // margin, after CONFIG3 powers the reference buffer and before START.
        FreeRtos::delay_ms(300);

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

/// The front end: the one ADS1298 and the START line that gates its conversions.
pub(crate) struct Ads1298FrontEnd {
    pub(super) device: Ads1298Device,
    start: PinDriver<'static, Output>,
}

impl Ads1298FrontEnd {
    pub(super) fn new(device: Ads1298Device, start: PinDriver<'static, Output>) -> Self {
        Self { device, start }
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

    /// Warm-recovers the front end after a mid-session death: conversions stopped,
    /// RESET/SDATAC/reconfigure, RDATAC, START again. The bring-up campaign
    /// (documentation/ads1298-bringup-2026-07-31/) established that the front end dies
    /// stochastically under multi-channel conversion and that this recovery restores
    /// it within a few milliseconds, yielding 97% verified frames at 2 kSPS ×
    /// 8 channels on the bench.
    pub(super) fn warm_recover(&mut self) -> Result<()> {
        self.stop_conversion()?;
        self.device.warm_reset()?;
        self.device.read_data_continuous()?;
        self.start_conversion()?;
        Ok(())
    }

    /// Clocks out one frame. Only valid in RDATAC mode after DRDY falls.
    pub(super) fn read_frame(&mut self) -> Result<Sample> {
        self.device.read_frame()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte `configure` puts on the wire, pinned by number with the field names
    /// beside it — change a field on purpose and this test names the byte that moved.
    #[test]
    fn configure_writes_the_bytes_the_bench_validated() {
        // CONFIG1: high resolution, per-chip readback, fMOD/256 (2000 SPS), clock
        // output off — one self-clocked chip, nothing listening to CLK.
        assert_eq!(device_config1().to_byte(), 0xC4);

        // CONFIG2: the internal test signal generator off.
        assert_eq!(NORMAL_CONFIG2.to_byte(), 0x00);

        // CONFIG3: internal reference on, bias drive off — the only bench-validated
        // drive mode (see the TODO on RightLegDriveMode).
        assert_eq!(device_config3().to_byte(), 0xC0);

        // RLD_SENSP/N: no channels sum into a drive that is powered down.
        assert_eq!(
            RightLegDriveSensePositive(right_leg_drive_channels()).to_byte(),
            0x00
        );
        assert_eq!(
            RightLegDriveSenseNegative(right_leg_drive_channels()).to_byte(),
            0x00
        );

        // LOFF: DC lead-off detection in resistor mode.
        assert_eq!(LEAD_OFF_CONTROL.to_byte(), 0x13);

        // CH1SET..CH8SET: powered up, gain 6, on the electrode.
        assert_eq!(NORMAL_CHANNEL_SETTINGS.to_byte(), 0x00);

        // LOFF_SENSP/N and LOFF_FLIP: nothing sensed while lead-off is off.
        assert_eq!(LeadOffSensePositive(lead_off_channels()).to_byte(), 0x00);
        assert_eq!(LeadOffSenseNegative(lead_off_channels()).to_byte(), 0x00);
        assert_eq!(LeadOffFlip(ChannelMask::NONE).to_byte(), 0x00);

        // The blocks this board leaves off.
        assert_eq!(GeneralPurposeInputOutputOff.to_byte(), 0x00);
        assert_eq!(PaceDetectOff.to_byte(), 0x00);
        assert_eq!(WilsonCenterTerminalOneOff.to_byte(), 0x00);
        assert_eq!(WilsonCenterTerminalTwoOff.to_byte(), 0x00);

        // RESP is 0x20 rather than 0x00 purely because bit 5 is a reserved one.
        assert_eq!(RESPIRATION_OFF.to_byte(), 0x20);

        // CONFIG4: continuous conversion, lead-off comparators off.
        assert_eq!(CONFIG4.to_byte(), 0x00);
    }

    /// Every bias-drive mode's byte, pinned so switching the knob later cannot
    /// silently change anything else.
    #[test]
    fn each_right_leg_drive_mode_emits_its_datasheet_byte() {
        use RightLegDriveMode::{Disabled, ExternalReference, InternalReference};

        assert_eq!(config3_for_mode(ExternalReference).to_byte(), 0xC6);
        assert_eq!(config3_for_mode(InternalReference).to_byte(), 0xCE);
        assert_eq!(config3_for_mode(Disabled).to_byte(), 0xC0);
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
            Config1::from_byte(device_config1().to_byte()),
            Some(device_config1())
        );
        assert_eq!(
            Config3::from_byte(device_config3().to_byte()),
            Some(device_config3())
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

    /// `SAMPLE_RATE_HZ` is derived from CONFIG1 rather than written down, so this
    /// asserts the derivation lands where the hand arithmetic used to.
    #[test]
    fn sample_rate_is_derived_from_the_configured_data_rate() {
        assert_eq!(SAMPLE_RATE_HZ, 2000);
    }
}
