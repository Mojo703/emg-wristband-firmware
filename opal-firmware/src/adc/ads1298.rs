//! Driver for one TI ADS1298 8-channel ADC; the front end runs two of them.
//!
//! Datasheet: <https://www.ti.com/lit/ds/symlink/ads1298.pdf> (SBAS459K).
//!
//! The two chips share one SPI bus and the RESET, START, and PWDN lines; each has
//! its own chip select and DRDY. One chip's internal 2.048 MHz oscillator clocks
//! both (see [`ClockRole`]). The shared lines and the sequencing live in
//! [`super::ChipPair`]; this type owns everything per-chip.

use anyhow::Result;
use esp_idf_svc::hal::delay::{Ets, FreeRtos};
use esp_idf_svc::hal::gpio::{Input, Output, PinDriver};
use esp_idf_svc::hal::spi::{Operation, SpiDeviceDriver, SpiDriver};
use log::warn;
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

const COMMAND_RESET: u8 = 0x06;
const COMMAND_RDATAC: u8 = 0x10;
const COMMAND_SDATAC: u8 = 0x11;
const COMMAND_RREG_BASE: u8 = 0x20;
const COMMAND_WREG_BASE: u8 = 0x40;

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
/// The clock output is off here; [`ClockRole::Source`] turns it on for the one chip
/// that drives CLK. The reset default is `0x06` — low power at fMOD/1024 — which converts at 250 SPS, so a chip
/// that has silently reverted is visible both in a readback and in the DRDY edge
/// rate; acquisition's slow-period recovery trigger keys on exactly that.
pub(super) const fn device_config1() -> Config1 {
    Config1 {
        high_resolution: true,
        multiple_readback: true,
        output_clock_enabled: false,
        data_rate: DataRate::ModulatorClockOver256,
    }
}

/// How a chip drives the subject bias reference.
///
/// The drive is per chip, and exactly one chip may hold it. Each chip has its own
/// amplifier, its own compensation network (R1/C2 for IC1, R2/C11 for IC2), and its
/// own driven electrode (BIAS_DRV1 on FPC2, BIAS_DRV2 on FPC4). Two powered amplifiers are two closed loops on one arm, each
/// summing a different eight channels and each forcing the body toward its own
/// RLDREF through its own electrode, so neither can see the other except as a
/// disturbance. The datasheet's multi-device arrangement, master RLDOUT into slave
/// RLDIN with the slave amplifier down, is not wired here: RLDOUT ties straight to
/// RLDIN on each chip.
///
/// One amplifier covers both chips because they share ground, so driving the arm
/// toward that reference lowers common mode everywhere. The driving chip's
/// RLD_SENSP/N still only sums its own eight channels, so the loop nulls what its
/// own electrodes see and the other chip benefits through the body. Expect the two
/// to improve by different amounts.
///
/// Either reference mode suits this board. AVDD is +2.5 V and AVSS is -2.5 V, so
/// mid-supply is 0 V, and the grounded RLDREF pin presents exactly the reference the
/// internal option would generate. With no body on the electrodes the loop is open
/// and the driving chip's output sits at a rail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RightLegDriveMode {
    /// Amplifier powered, reference taken from the RLDREF pin (`0xC6`). The pin is
    /// grounded, which is mid-supply on this board's bipolar rails, so this mode is
    /// correct here.
    ExternalReference,
    /// Amplifier powered down entirely (`0xC0`).
    Disabled,
}

const CONFIGURATION_REGISTER_COUNT: usize = 23;

impl RightLegDriveMode {
    /// CONFIG3 with the internal 2.4 V reference and this bias-drive mode.
    const fn config3(self) -> Config3 {
        let drive_enabled = !matches!(self, Self::Disabled);
        Config3 {
            internal_reference_enabled: true,
            // The 2.4 V reference is what `preprocess` converts codes against.
            four_volt_reference: false,
            right_leg_drive_measurement: false,
            right_leg_drive_reference_internal: false,
            right_leg_drive_enabled: drive_enabled,
            // Lead-off sense rides on the same amplifier.
            right_leg_drive_lead_off_sense: drive_enabled,
        }
    }

    /// Channels summed into this chip's bias drive: all when powered, none otherwise.
    const fn sense_channels(self) -> ChannelMask {
        match self {
            Self::Disabled => ChannelMask::NONE,
            _ => ChannelMask::ALL,
        }
    }

    /// The complete normal register recipe for a chip using this bias-drive mode.
    pub(super) fn expected_configuration(self) -> [(Register, u8); CONFIGURATION_REGISTER_COUNT] {
        let sense = self.sense_channels();
        [
            (Register::Config1, device_config1().to_byte()),
            (Register::Config3, self.config3().to_byte()),
            (
                Register::RldSensP,
                RightLegDriveSensePositive(sense).to_byte(),
            ),
            (
                Register::RldSensN,
                RightLegDriveSenseNegative(sense).to_byte(),
            ),
            (Register::Config2, NORMAL_CONFIG2.to_byte()),
            (Register::Loff, LEAD_OFF_CONTROL.to_byte()),
            (Register::Ch1Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch2Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch3Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch4Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch5Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch6Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch7Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (Register::Ch8Set, NORMAL_CHANNEL_SETTINGS.to_byte()),
            (
                Register::LoffSensP,
                LeadOffSensePositive(lead_off_channels()).to_byte(),
            ),
            (
                Register::LoffSensN,
                LeadOffSenseNegative(lead_off_channels()).to_byte(),
            ),
            (Register::LoffFlip, LeadOffFlip(ChannelMask::NONE).to_byte()),
            (Register::Gpio, GeneralPurposeInputOutputOff.to_byte()),
            (Register::Pace, PaceDetectOff.to_byte()),
            (Register::Resp, RESPIRATION_OFF.to_byte()),
            (Register::Config4, CONFIG4.to_byte()),
            (Register::Wct1, WilsonCenterTerminalOneOff.to_byte()),
            (Register::Wct2, WilsonCenterTerminalTwoOff.to_byte()),
        ]
    }
}

/// Which chip drives, in chip order: IC1 drives through BIAS_DRV1.
///
/// TODO(bias drive): the powered loop is not bench-validated. Scope BIAS_DRV with the
/// survival harness before trusting a session, since RLD load was never exonerated in
/// the conversion-death campaign, and leave chip 1's bias electrode off the arm until
/// a powered-down RLDOUT is confirmed high-impedance.
pub(super) const RIGHT_LEG_DRIVE_MODE: [RightLegDriveMode; DEVICE_COUNT] = [
    RightLegDriveMode::ExternalReference,
    RightLegDriveMode::Disabled,
];

/// Which end of the shared CLK net a chip sits on, fixed by its CLKSEL strap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ClockRole {
    /// CLKSEL high: converts on its internal oscillator and drives it out of CLK
    /// (CONFIG1 `CLK_EN`). A RESET clears `CLK_EN`, which stops the follower.
    Source,
    /// CLKSEL low: converts on the CLK pin and cannot decode a command without it.
    Follower,
}

/// Chip 0 is IC1 (CLKSEL to GND), chip 1 is IC2 (CLKSEL to 3V3).
pub(super) const CLOCK_ROLE: [ClockRole; DEVICE_COUNT] = [ClockRole::Follower, ClockRole::Source];

impl ClockRole {
    pub(super) const fn config1(self) -> Config1 {
        Config1 {
            output_clock_enabled: matches!(self, ClockRole::Source),
            ..device_config1()
        }
    }
}

/// One chip's whole register recipe: its bias-drive mode's, with CONFIG1 set for its
/// clock role.
pub(super) fn chip_configuration(
    right_leg_drive: RightLegDriveMode,
    clock_role: ClockRole,
) -> [(Register, u8); CONFIGURATION_REGISTER_COUNT] {
    let mut configuration = right_leg_drive.expected_configuration();
    for (register, value) in &mut configuration {
        if *register == Register::Config1 {
            *value = clock_role.config1().to_byte();
        }
    }
    configuration
}

/// CONFIG2 with the internal test signal generator off, which is normal operation.
const NORMAL_CONFIG2: Config2 = Config2 {
    chopping_frequency_constant: false,
    test_signal_enabled: false,
    test_signal_amplitude: TestSignalAmplitude::Single,
    test_signal_frequency: TestSignalFrequency::PulsedSlow,
};

/// LOFF: DC lead-off detection from the 6 nA current sources, comparators at the
/// widest thresholds (SBAS459K table 21). The pull-resistor mode would put 10 MΩ to
/// each rail across a dry electrode: up to 250 nA, and a quarter volt of offset per
/// megaohm of skin. The current source costs 6 mV per megaohm.
const LEAD_OFF_CONTROL: LeadOffControl = LeadOffControl {
    comparator_threshold: LeadOffComparatorThreshold::NinetyFive,
    pull_resistor_mode: false,
    current: LeadOffCurrent::Six,
    detection: LeadOffDetection::DirectCurrent,
};

/// CHnSET for normal operation: channel powered up, gain 6, reading its electrode pair.
const NORMAL_CHANNEL_SETTINGS: ChannelSettings = ChannelSettings {
    powered_down: false,
    gain: Gain::Six,
    input: ChannelInput::Electrode,
};

/// Whether the streaming recipe watches for lead-off. It does not: against dry
/// electrodes the 6 nA sources push most channels past full scale at gain 6, so
/// lead-off runs only as an AC snapshot ([`Ads1298Device::set_lead_off`]).
pub(crate) const LEAD_OFF_ENABLED: bool = false;

/// CONFIG4: continuous conversion; lead-off comparators per [`LEAD_OFF_ENABLED`].
const CONFIG4: Config4 = Config4 {
    respiration_frequency: RespirationFrequency::SixtyFourKilohertz,
    single_shot: false,
    wilson_center_terminal_to_right_leg_drive: false,
    lead_off_comparators_enabled: LEAD_OFF_ENABLED,
};

/// LOFF for a snapshot: the 10 MΩ pull resistors alternating at fDR / 4. On the
/// bench they put a uniform 11.3 mV tone on every pad in free air (the flex's 1 nF
/// filter capacitors set it) where the 6 nA sources gave an uneven 0.5 to 2.4 mV.
/// The excitation averages to zero, so unlike DC it leaves no charge behind.
const ALTERNATING_LEAD_OFF_CONTROL: LeadOffControl = LeadOffControl {
    detection: LeadOffDetection::AlternatingCurrent,
    pull_resistor_mode: true,
    ..LEAD_OFF_CONTROL
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
    /// This chip's entry in [`RIGHT_LEG_DRIVE_MODE`]. Only one chip may hold a
    /// powered mode; the reasoning is on that constant.
    right_leg_drive: RightLegDriveMode,
    clock_role: ClockRole,
}

impl Ads1298Device {
    pub(super) fn frame_spi_handle(&self) -> esp_idf_svc::sys::spi_device_handle_t {
        self.frame_spi.device()
    }

    pub(super) fn new(
        command_spi: SpiDevice,
        frame_spi: SpiDevice,
        mut chip_select: PinDriver<'static, Output>,
        drdy: PinDriver<'static, Input>,
        right_leg_drive: RightLegDriveMode,
        clock_role: ClockRole,
    ) -> Result<Self> {
        chip_select.set_high()?;
        Ok(Self {
            command_spi,
            frame_spi,
            chip_select,
            drdy,
            right_leg_drive,
            clock_role,
        })
    }

    pub(super) fn clock_role(&self) -> ClockRole {
        self.clock_role
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

    /// Whether DRDY is low: a conversion is waiting to be read.
    pub(super) fn data_is_ready(&self) -> bool {
        self.drdy.is_low()
    }

    /// This chip's CS pin number. The interrupt-side reader drives CS straight
    /// from the GPIO output registers; the driver holds this `PinDriver` so the
    /// pad stays configured as an output and nothing else can claim it.
    pub(super) fn chip_select_pin(&self) -> u8 {
        self.chip_select.pin()
    }

    /// Resets this chip alone by opcode, then waits out the 18 tCLK lockout
    /// (SBAS459K §9.3.2.3). The shared RESET pin is [`super::ChipPair`]'s.
    pub(super) fn reset_by_command(&mut self) -> Result<()> {
        self.send_command(COMMAND_RESET)?;
        FreeRtos::delay_ms(1);
        Ok(())
    }

    /// Switches AC lead-off excitation on every input on, or back to the streaming
    /// recipe. SDATAC only.
    pub(super) fn set_lead_off(&mut self, on: bool) -> Result<()> {
        let (control, inputs) = match on {
            true => (ALTERNATING_LEAD_OFF_CONTROL, ChannelMask::ALL),
            false => (LEAD_OFF_CONTROL, lead_off_channels()),
        };
        self.write_register(control)?;
        self.write_register(LeadOffSensePositive(inputs))?;
        self.write_register(LeadOffSenseNegative(inputs))
    }

    /// Writes this chip's CONFIG1, which on the clock source starts CLK.
    pub(super) fn write_config1(&mut self) -> Result<()> {
        self.write_register(self.clock_role.config1())
    }

    /// Helper for sending SPI commands
    fn send_command(&mut self, command: u8) -> Result<()> {
        self.with_selection(|device| {
            device.command_spi.write(&[command])?;
            Ok(())
        })
    }

    /// Disables read data continuous mode so that registers can be configured
    pub(super) fn stop_read_data_continuous(&mut self) -> Result<()> {
        self.send_command(COMMAND_SDATAC)
    }

    /// Enables read data continuous mode. Once started, each DRDY
    /// falling edge means a frame is ready to clock out
    pub(super) fn read_data_continuous(&mut self) -> Result<()> {
        self.send_command(COMMAND_RDATAC)
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
                Operation::Write(&[COMMAND_WREG_BASE | reg.addr()]),
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
                Operation::Write(&[COMMAND_RREG_BASE | reg.addr()]),
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

    /// Every register as the chip holds it, address order. SDATAC only, before
    /// RDATAC starts the stream. A register whose read fails stays `None`.
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

    // Writes and verifies this device's full register set. Must be called after
    // `stop_read_data_continuous()` (SDATAC), since registers can't be accessed while streaming.
    pub(super) fn configure(&mut self) -> Result<()> {
        let expected = chip_configuration(self.right_leg_drive, self.clock_role);
        for &(register, value) in &expected {
            self.write_register_byte(register, value)?;
        }
        for (register, expected) in expected {
            let read = self.read_register(register)?;
            if read != expected {
                anyhow::bail!(
                    "{} wrote {expected:#04x}, reads {read:#04x}",
                    register.name()
                );
            }
        }

        Ok(())
    }

    /// Leaves a freshly reset chip in SDATAC and returns whatever its ID register says.
    /// The caller has already reset it and waited out the 18 tCLK lockout.
    pub(super) fn probe_identity(&mut self) -> Result<u8> {
        self.stop_read_data_continuous()?;
        FreeRtos::delay_ms(1);
        self.read_register(Register::Id)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte `configure` puts on the wire, pinned by number with the field names
    /// beside it — change a field on purpose and this test names the byte that moved.
    #[test]
    fn configure_writes_the_bytes_the_bench_validated() {
        let configuration = RIGHT_LEG_DRIVE_MODE[0].expected_configuration();
        assert_eq!(configuration.len(), 23);
        for (index, (register, _)) in configuration.iter().enumerate() {
            assert!(
                configuration[..index]
                    .iter()
                    .all(|(seen, _)| seen != register),
                "{register:?} appears twice"
            );
        }

        // CONFIG1: high resolution, per-chip readback, fMOD/256, clock output off.
        assert_eq!(device_config1().to_byte(), 0xC4);

        // CONFIG2: the internal test signal generator off.
        assert_eq!(NORMAL_CONFIG2.to_byte(), 0x00);

        // CONFIG3: internal reference on, and the bias drive on one chip. Chip 0
        // drives, chip 1 does not, and both driving at once is the failure this
        // pins against (see RIGHT_LEG_DRIVE_MODE).
        let [drive_zero, drive_one] = RIGHT_LEG_DRIVE_MODE;
        assert_eq!(drive_zero.config3().to_byte(), 0xC6);
        assert_eq!(drive_one.config3().to_byte(), 0xC0);
        assert_eq!(
            RIGHT_LEG_DRIVE_MODE
                .iter()
                .filter(|mode| !matches!(mode, RightLegDriveMode::Disabled))
                .count(),
            1
        );

        // RLD_SENSP/N: all eight channels of the driving chip, none of the other's.
        assert_eq!(
            RightLegDriveSensePositive(drive_zero.sense_channels()).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSenseNegative(drive_zero.sense_channels()).to_byte(),
            0xFF
        );
        assert_eq!(
            RightLegDriveSensePositive(drive_one.sense_channels()).to_byte(),
            0x00
        );
        assert_eq!(
            RightLegDriveSenseNegative(drive_one.sense_channels()).to_byte(),
            0x00
        );

        // LOFF: DC lead-off detection from the 6 nA current sources.
        assert_eq!(LEAD_OFF_CONTROL.to_byte(), 0x03);

        // CH1SET..CH8SET: powered up, gain 6, on the electrode.
        assert_eq!(NORMAL_CHANNEL_SETTINGS.to_byte(), 0x00);

        // LOFF_SENSP/N and LOFF_FLIP: nothing sensed while streaming.
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
        assert_eq!(ALTERNATING_LEAD_OFF_CONTROL.to_byte(), 0x11);
    }

    /// Exactly one chip drives CLK, and each chip's recipe carries its own CONFIG1.
    #[test]
    fn only_the_clock_source_drives_clk() {
        assert_eq!(ClockRole::Source.config1().to_byte(), 0xE4);
        assert_eq!(ClockRole::Follower.config1().to_byte(), 0xC4);
        assert_eq!(
            CLOCK_ROLE
                .iter()
                .filter(|role| **role == ClockRole::Source)
                .count(),
            1
        );
        for (drive, clock) in RIGHT_LEG_DRIVE_MODE.into_iter().zip(CLOCK_ROLE) {
            let configuration = chip_configuration(drive, clock);
            assert_eq!(
                configuration[0],
                (Register::Config1, clock.config1().to_byte())
            );
        }
    }

    /// Every bias-drive mode's byte, pinned so switching the knob later cannot
    /// silently change anything else.
    #[test]
    fn each_right_leg_drive_mode_emits_its_datasheet_byte() {
        use RightLegDriveMode::{Disabled, ExternalReference};

        assert_eq!(ExternalReference.config3().to_byte(), 0xC6);
        assert_eq!(Disabled.config3().to_byte(), 0xC0);
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
            Config3::from_byte(RIGHT_LEG_DRIVE_MODE[0].config3().to_byte()),
            Some(RIGHT_LEG_DRIVE_MODE[0].config3())
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
