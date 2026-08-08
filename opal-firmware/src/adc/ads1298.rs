//! Driver for one TI ADS1298 8-channel ADC; the front end runs two of them.
//!
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf> (SBAS459K).
//!
//! Each chip runs on its own SPI bus and self-clocks from its internal 2.048 MHz
//! oscillator (CLKSEL strapped to 3V3), and each has its own chip select, START,
//! RESET, DRDY, and PWDN so one can be warm-recovered without touching the other.
//! The rail sequencing lives in [`super::bring_up`]; this type owns everything
//! per-chip.

use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{Input, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};
use log::{info, warn};
use std::sync::Arc;

use super::channel::{Channel, DEVICE_COUNT};
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

/// Gap between the bytes of a multi-byte register command, satisfying tSDECODE =
/// 4 tCLK (~2 µs at 2.048 MHz) per byte with margin (SBAS459K §9.5.1.2.1).
///
/// History, corrected by the spi-timing audit: the old 2/4 MHz ID-reads-0x00
/// failures were originally blamed on tSDECODE, but both rates satisfy it — the
/// real cure was CS framing (holding CS low across the whole command instead of
/// raising it between bytes, which resets the decoder). The gaps are kept anyway:
/// they are unconditionally legal, cost nothing off the frame-read hot path, and
/// remove tSDECODE as a variable entirely.
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
/// It comes out at 2000 SPS nominal from the internal 2.048 MHz oscillator, and
/// each chip misses it by its own oscillator error (±0.5% at 25 °C).
///
/// **Training windows must be cut at this rate.** The model reads its 500 samples
/// as a fixed span of time, so a window exported at any other rate arrives
/// stretched or compressed, and every temporal feature it learned lands at the
/// wrong scale. Nothing downstream can detect that: the tensor has the right
/// shape either way, the accuracy loss looks like a bad model, and the export is
/// in another repo. Check the exporter's rate against this constant before
/// believing any on-device accuracy number. See [`crate::adc::preprocess`].
pub(crate) const SAMPLE_RATE_HZ: u32 = {
    let config1 = device_config1();
    config1
        .data_rate
        .samples_per_second(config1.high_resolution)
};

/// CONFIG1: high resolution at fMOD/256, which is where [`SAMPLE_RATE_HZ`] comes from.
/// The clock output must stay off: with CLKSEL high the CLK pin is a disabled output
/// (3-state) and nothing connects to it — driving the oscillator out of an
/// unconnected pin would only add an aggressor to the board. The reset
/// default is `0x06` — low power at fMOD/1024 — which converts at 250 SPS, so a chip
/// that has silently reverted is visible both in a readback and in the DRDY edge
/// rate; acquisition's slow-period recovery trigger keys on exactly that.
const fn device_config1() -> Config1 {
    Config1 {
        high_resolution: true,
        multiple_readback: true,
        output_clock_enabled: false,
        data_rate: DataRate::ModulatorClockOver256,
    }
}

/// How a chip drives the subject bias reference.
///
/// The drive is per chip, and exactly one chip may hold it. Each board carries its
/// own amplifier, its own R1/C2 compensation network, and its own driven electrode
/// on J5 pins 3 and 4. Two powered amplifiers are two closed loops on one arm, each
/// summing a different eight channels and each forcing the body toward its own
/// RLDREF through its own electrode, so neither can see the other except as a
/// disturbance. The datasheet's multi-device arrangement, master RLDOUT into slave
/// RLDIN with the slave amplifier down, is not wired here: RLDOUT ties straight to
/// RLDIN on each board and reaches no connector.
///
/// One amplifier covers both chips because they share ground, so driving the arm
/// toward that reference lowers common mode everywhere. The driving chip's
/// RLD_SENSP/N still only sums its own eight channels, so the loop nulls what its
/// own electrodes see and the other chip benefits through the body. Expect the two
/// to improve by different amounts.
///
/// Either reference mode suits this board. AVDD is +2.5 V and AVSS is -2.5 V, so
/// mid-supply is 0 V, and the grounded RLDREF pin presents exactly the reference the
/// internal option would generate.
///
/// A powered amplifier needs its board's jumper from J4 pin 1 to the BIAS_DRV node,
/// which closes the compensation network into RLDINV instead of padding the output
/// (SBAS459K figure 94). Without it RLDINV floats, the loop is open, and the output
/// sits at a rail. Engineering log 0019 carries the trace and the one-wire fix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RightLegDriveMode {
    /// Amplifier powered, reference taken from the RLDREF pin (`0xC6`). The pin is
    /// grounded, which is mid-supply on this board's bipolar rails, so this mode is
    /// correct here.
    ExternalReference,
    /// Amplifier powered, reference generated internally at mid-supply (`0xCE`).
    #[allow(dead_code)]
    InternalReference,
    /// Amplifier powered down entirely (`0xC0`).
    Disabled,
}

/// Which chip drives, in chip order. Chip 0's board carries the jumper, so the
/// boards are no longer interchangeable between the two positions: the jumpered one
/// goes in the chip 0 socket or the drive runs open-loop into a rail.
///
/// TODO(bias drive): the powered loop is not bench-validated. Scope BIAS_DRV with the
/// survival harness before trusting a session, since RLD load was never exonerated in
/// the conversion-death campaign, and leave chip 1's bias electrode off the arm until
/// a powered-down RLDOUT is confirmed high-impedance.
pub(super) const RIGHT_LEG_DRIVE_MODE: [RightLegDriveMode; DEVICE_COUNT] = [
    RightLegDriveMode::ExternalReference,
    RightLegDriveMode::Disabled,
];

/// CONFIG3: the internal reference buffer on the 2.4 V reference, and the right-leg
/// drive wherever this chip's [`RIGHT_LEG_DRIVE_MODE`] entry puts it.
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

/// RLD_SENSP/N: which of this chip's channels sum into its right-leg drive. All
/// eight where the amplifier is powered, none where it is not.
const fn sense_channels_for_mode(mode: RightLegDriveMode) -> ChannelMask {
    match mode {
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
    /// Register and opcode traffic, at the bench-validated command clock.
    command_spi: SpiDevice,
    /// RDATAC frame reads only — the hot path — at its own, faster clock. Data
    /// clocking has no tSDECODE to honour (DIN stays low for the whole read, and
    /// 0x00 is not a live opcode), so its rate is bounded by signal integrity
    /// alone, not the command decoder.
    frame_spi: SpiDevice,
    chip_select: PinDriver<'static, Output>,
    drdy: PinDriver<'static, Input>,
    reset_n: PinDriver<'static, Output>,
    /// This chip's entry in [`RIGHT_LEG_DRIVE_MODE`]. Only one chip may hold a
    /// powered mode; the reasoning is on that constant.
    right_leg_drive: RightLegDriveMode,
}

impl Ads1298Device {
    pub(super) fn new(
        command_spi: SpiDevice,
        frame_spi: SpiDevice,
        mut chip_select: PinDriver<'static, Output>,
        drdy: PinDriver<'static, Input>,
        mut reset_n: PinDriver<'static, Output>,
        right_leg_drive: RightLegDriveMode,
    ) -> Result<Self> {
        // Held in reset from construction until `bring_up` runs the shared power
        // sequence and calls `initialize()`. The shared PWDN line lives there too.
        reset_n.set_low()?;
        chip_select.set_high()?;

        Ok(Self {
            command_spi,
            frame_spi,
            chip_select,
            drdy,
            reset_n,
            right_leg_drive,
        })
    }

    /// Releases the RESET line after the shared PWDN has been raised, ahead of the
    /// supply-settling wait `bring_up` owns. [`Self::initialize`] then runs the
    /// per-chip sequence.
    pub(super) fn release_reset(&mut self) -> Result<()> {
        self.reset_n.set_high()?;
        Ok(())
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

    /// This chip's DRDY pin number, for the interrupt-side frame reader, which
    /// works in GPIO numbers rather than through `PinDriver`.
    pub(super) fn data_ready_pin(&self) -> u8 {
        self.drdy.pin()
    }

    /// This chip's CS pin number. The interrupt-side reader drives CS straight
    /// from the GPIO output registers; the driver holds this `PinDriver` so the
    /// pad stays configured as an output and nothing else can claim it.
    pub(super) fn chip_select_pin(&self) -> u8 {
        self.chip_select.pin()
    }

    pub(super) fn reset_pulse(&mut self) -> Result<()> {
        self.reset_n.set_low()?;
        FreeRtos::delay_ms(1);
        self.reset_n.set_high()?;
        Ok(())
    }

    /// Warm recovery for a chip whose conversions have died mid-session: RESET pulse,
    /// the post-reset lockout, SDATAC, full reconfigure. The rails have long since
    /// settled so the cold-start supply wait is not repeated — but the RESET does
    /// power-cycle the internal reference buffer, whose 150 ms start-up cannot be
    /// skipped. Acquisition owns that: it discards the frames of the settling window
    /// rather than stalling here (see `REFERENCE_SETTLE_AFTER_RECOVERY_US`).
    pub(super) fn warm_reset(&mut self) -> Result<()> {
        self.reset_pulse()?;
        // 18 tCLK after RESET rises before any command (SBAS459K §9.3.2.3).
        FreeRtos::delay_ms(1);
        self.stop_read_data_continuous()?;
        // Same post-SDATAC breather the proven cold-boot path takes before its first
        // register access.
        FreeRtos::delay_ms(1);
        self.configure()
    }

    /// Helper for sending SPI commands
    fn send_command(&mut self, command: u8) -> Result<()> {
        self.with_selection(|device| {
            device.command_spi.write(&[command])?;
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
            device.command_spi.transaction(&mut [
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
            device.command_spi.transaction(&mut [
                Operation::Write(&[spi_commands::RREG_BASE | reg.addr()]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                Operation::Write(&[0x00]),
                Operation::DelayNs(COMMAND_DECODE_GAP_NANOSECONDS),
                // Transfer, not Read: drives DIN low during the byte (§9.4.1.3)
                // instead of leaving MOSI undriven.
                Operation::Transfer(&mut rx, &[0x00]),
            ])?;
            Ok(rx[0])
        })
    }

    /// Clocks out one frame. Only valid while in RDATAC mode
    // and after `data_ready()` reports true
    pub(super) fn read_frame(&mut self) -> Result<Sample> {
        // DIN must stay low for the entire read (SBAS459K §9.4.1.3): the command
        // decoder listens during RDATAC, so every bit on DIN is a potential opcode.
        // A bare `read` gives esp-idf a null TX buffer, which disables the MOSI
        // phase and leaves the pin at whatever level it last held — low only by
        // accident of the previous write's final bit. Transferring against
        // explicit zeros drives DIN low for all 216 clocks.
        const DIN_LOW: [u8; FRAME_BYTES] = [0u8; FRAME_BYTES];
        self.with_selection(|device| {
            let mut raw = [0u8; FRAME_BYTES];
            device.frame_spi.transfer(&mut raw, &DIN_LOW)?;
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
            (
                Register::Config3,
                config3_for_mode(self.right_leg_drive).to_byte(),
            ),
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

    /// Every register as the chip holds it, address order, for the provenance a
    /// session records. Same constraint as [`Self::log_configuration_readback`]:
    /// SDATAC only, before RDATAC starts the stream. A register whose read fails
    /// stays `None` — a snapshot that quietly reported the intended byte would
    /// defeat the point of reading the chip at all.
    pub(super) fn read_all_registers(&mut self) -> [Option<u8>; Register::ALL.len()] {
        let mut values = [None; Register::ALL.len()];
        for (slot, register) in values.iter_mut().zip(Register::ALL) {
            match self.read_register(register) {
                Ok(value) => *slot = Some(value),
                Err(error) => warn!("{register:?} snapshot read failed: {error}"),
            }
        }
        values
    }

    // Writes this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be written while streaming.
    pub(super) fn configure(&mut self) -> Result<()> {
        self.write_register(device_config1())?;

        self.write_register(config3_for_mode(self.right_leg_drive))?;

        let sense = sense_channels_for_mode(self.right_leg_drive);
        self.write_register(RightLegDriveSensePositive(sense))?;
        self.write_register(RightLegDriveSenseNegative(sense))?;

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

    /// Resets this chip, leaves it in SDATAC, and returns whatever its ID register
    /// says. Separate from [`Self::initialize`] so bring-up can probe every chip and
    /// report all of them, rather than stopping at the first one that disagrees —
    /// with two boards, whether the other chip reads the same wrong byte is what
    /// separates a harness-wide fault (clock, bus) from one bad board.
    pub(super) fn probe_identity(&mut self) -> Result<u8> {
        self.reset_pulse()?;

        // 18 tCLK (~9 µs at fCLK = 2.048 MHz) are needed after RESET returns high for
        // the device to finish initialising its registers, and no command may be sent
        // during that window (SBAS459K §9.3.2.3, §11.1). This delay used to sit *after*
        // the SDATAC below, which put that command inside the forbidden window.
        FreeRtos::delay_ms(1);

        self.stop_read_data_continuous()?;

        FreeRtos::delay_ms(1);

        self.read_register(Register::Id)
    }

    /// Per-chip initialisation, after `bring_up` has raised the shared PWDN, released
    /// both RESET lines, waited out the supply settling, and probed the identities.
    /// Reference settling after `configure` is also shared and lives in `bring_up`.
    /// The chip is already reset and in SDATAC from [`Self::probe_identity`].
    pub(super) fn initialize(&mut self) -> Result<()> {
        self.configure()
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

        // `configure()` puts every channel of the driving chip into the RLD
        // derivation for real electrode use. With the test signal active, that
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

    /// Pulls this chip's START pin high. START is per-chip so a warm recovery here
    /// never restarts the other chip into the post-START death hazard.
    pub(super) fn start_conversion(&mut self) -> Result<()> {
        self.start.set_high()?;
        Ok(())
    }

    /// Pulls this chip's START pin low. Used by [`Self::warm_recover`].
    pub(super) fn stop_conversion(&mut self) -> Result<()> {
        self.start.set_low()?;
        Ok(())
    }

    /// Warm-recovers the front end after a mid-session death: conversions stopped,
    /// RESET/SDATAC/reconfigure, RDATAC, START again. The bring-up campaign
    /// (documentation/ads1298-bringup-2026-07-31/) established that the front end
    /// dies stochastically under multi-channel conversion and that this recovery
    /// restores it. The register work here takes a few milliseconds; the true cost
    /// per recovery is the reference-settling discard window acquisition applies
    /// afterwards.
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
        // CONFIG1: high resolution, per-chip readback, fMOD/256, clock output off —
        // with CLKSEL at 3V3 the CLK pins are 3-stated and unconnected.
        assert_eq!(device_config1().to_byte(), 0xC4);

        // CONFIG2: the internal test signal generator off.
        assert_eq!(NORMAL_CONFIG2.to_byte(), 0x00);

        // CONFIG3: internal reference on, and the bias drive only where the board
        // carries the jumper. Chip 0 drives, chip 1 does not, and both driving at
        // once is the failure this pins against (see RIGHT_LEG_DRIVE_MODE).
        let [drive_zero, drive_one] = RIGHT_LEG_DRIVE_MODE;
        assert_eq!(config3_for_mode(drive_zero).to_byte(), 0xC6);
        assert_eq!(config3_for_mode(drive_one).to_byte(), 0xC0);
        assert_eq!(
            RIGHT_LEG_DRIVE_MODE
                .iter()
                .filter(|mode| !matches!(mode, RightLegDriveMode::Disabled))
                .count(),
            1
        );

        // RLD_SENSP/N: all eight channels of the driving chip, none of the other's.
        assert_eq!(
            RightLegDriveSensePositive(sense_channels_for_mode(drive_zero)).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSenseNegative(sense_channels_for_mode(drive_zero)).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSensePositive(sense_channels_for_mode(drive_one)).to_byte(),
            0x00
        );
        assert_eq!(
            RightLegDriveSenseNegative(sense_channels_for_mode(drive_one)).to_byte(),
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
            Config3::from_byte(config3_for_mode(RIGHT_LEG_DRIVE_MODE[0]).to_byte()),
            Some(config3_for_mode(RIGHT_LEG_DRIVE_MODE[0]))
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
        // 2.048 MHz internal oscillator / 1024 in high-resolution fMOD/256.
        assert_eq!(SAMPLE_RATE_HZ, 2000);
    }
}
