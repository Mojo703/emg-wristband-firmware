//! ADS1298 register map, and a typed value for every register the driver writes.
//!
//! Datasheet: TI SBAS459K, §9.6 "Register Map".
//!
//! # Why the values are types
//!
//! The driver used to write bare `u8` literals: `write_register(Register::Config1,
//! 0xE4)`. That is three separate ways to be wrong at one call site -- the wrong
//! address, a byte belonging to a different register, or a reserved bit dropped -- and
//! the only way to check any of it was to hold the datasheet open beside the code. A
//! wrong byte here does not fail; it converts at the wrong rate, or leaves an amplifier
//! powered down, and shows up days later as data that looks almost right.
//!
//! So each register gets a struct with named fields, and [`RegisterValue`] carries the
//! address along with the value. Three things follow:
//!
//! * The address cannot be mismatched to the value, because the value owns it.
//! * The read-only registers (ID, LOFF_STATP, LOFF_STATN) are unwritable, because no
//!   value type names them and [`super::ads1298::Ads1298Device::write_register`] only
//!   takes value types.
//! * Bits the datasheet requires be written as 1 are OR'd in by `to_byte` rather than
//!   remembered at each call site.
//!
//! Everything here is `const fn` over plain enums and bools, so the generated code is
//! the same byte literal the driver used to write by hand.
//!
//! The active driver retains only register encodings it writes or checks in the
//! current serial-production configuration.

use core::fmt;

use super::channel::Channel;

/// The conversion clock each chip generates for itself: the internal oscillator
/// (CLKSEL strapped to 3V3 on both boards). Nominal — each chip misses it by its
/// own oscillator error. See [`super::INTERNAL_OSCILLATOR_HZ`].
const CLOCK_HZ: u32 = super::INTERNAL_OSCILLATOR_HZ;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Register {
    Id = 0x00,
    Config1 = 0x01,
    Config2 = 0x02,
    Config3 = 0x03,
    Loff = 0x04,
    Ch1Set = 0x05,
    Ch2Set = 0x06,
    Ch3Set = 0x07,
    Ch4Set = 0x08,
    Ch5Set = 0x09,
    Ch6Set = 0x0A,
    Ch7Set = 0x0B,
    Ch8Set = 0x0C,
    RldSensP = 0x0D,
    RldSensN = 0x0E,
    LoffSensP = 0x0F,
    LoffSensN = 0x10,
    LoffFlip = 0x11,
    // The lead-off status arrives in every frame's status word, which `super::status`
    // decodes, so nothing reads these two registers directly. They stay for the
    // complete map, and because naming them here is what documents that they are
    // read-only: no `RegisterValue` claims either address, so neither can be written.
    LoffStatP = 0x12,
    LoffStatN = 0x13,
    Gpio = 0x14,
    Pace = 0x15,
    Resp = 0x16,
    Config4 = 0x17,
    Wct1 = 0x18,
    Wct2 = 0x19,
}

impl Register {
    /// The whole map in address order — what a provenance snapshot walks.
    pub(super) const ALL: [Register; 26] = [
        Register::Id,
        Register::Config1,
        Register::Config2,
        Register::Config3,
        Register::Loff,
        Register::Ch1Set,
        Register::Ch2Set,
        Register::Ch3Set,
        Register::Ch4Set,
        Register::Ch5Set,
        Register::Ch6Set,
        Register::Ch7Set,
        Register::Ch8Set,
        Register::RldSensP,
        Register::RldSensN,
        Register::LoffSensP,
        Register::LoffSensN,
        Register::LoffFlip,
        Register::LoffStatP,
        Register::LoffStatN,
        Register::Gpio,
        Register::Pace,
        Register::Resp,
        Register::Config4,
        Register::Wct1,
        Register::Wct2,
    ];

    pub(super) const fn addr(self) -> u8 {
        self as u8
    }

    /// The datasheet's own name for this register (SBAS459K §9.6), so a snapshot
    /// off the wire reads as the register map rather than as addresses.
    pub(super) const fn name(self) -> &'static str {
        match self {
            Register::Id => "ID",
            Register::Config1 => "CONFIG1",
            Register::Config2 => "CONFIG2",
            Register::Config3 => "CONFIG3",
            Register::Loff => "LOFF",
            Register::Ch1Set => "CH1SET",
            Register::Ch2Set => "CH2SET",
            Register::Ch3Set => "CH3SET",
            Register::Ch4Set => "CH4SET",
            Register::Ch5Set => "CH5SET",
            Register::Ch6Set => "CH6SET",
            Register::Ch7Set => "CH7SET",
            Register::Ch8Set => "CH8SET",
            Register::RldSensP => "RLD_SENSP",
            Register::RldSensN => "RLD_SENSN",
            Register::LoffSensP => "LOFF_SENSP",
            Register::LoffSensN => "LOFF_SENSN",
            Register::LoffFlip => "LOFF_FLIP",
            Register::LoffStatP => "LOFF_STATP",
            Register::LoffStatN => "LOFF_STATN",
            Register::Gpio => "GPIO",
            Register::Pace => "PACE",
            Register::Resp => "RESP",
            Register::Config4 => "CONFIG4",
            Register::Wct1 => "WCT1",
            Register::Wct2 => "WCT2",
        }
    }

    /// The CHnSET register for a channel. CH1SET..CH8SET are eight consecutive
    /// addresses from 0x05, and [`Channel`] is already range-checked, so this is total.
    pub(super) const fn channel_settings(channel: Channel) -> Register {
        match channel.index() {
            0 => Register::Ch1Set,
            1 => Register::Ch2Set,
            2 => Register::Ch3Set,
            3 => Register::Ch4Set,
            4 => Register::Ch5Set,
            5 => Register::Ch6Set,
            6 => Register::Ch7Set,
            // `Channel` cannot hold anything past 7, so this arm is the last channel
            // rather than a fallback for bad input.
            _ => Register::Ch8Set,
        }
    }
}

/// A value that can be written to exactly one register.
///
/// The address is a const on the value type, which is what makes a read-only register
/// unwritable: there is no type whose `ADDRESS` is `Id`, `LoffStatP` or `LoffStatN`.
/// CH1SET..CH8SET are the exception -- eight addresses sharing one value -- and they go
/// through [`super::ads1298::Ads1298Device::write_channel_settings`] instead.
pub(super) trait RegisterValue: Copy + Sized {
    const ADDRESS: Register;

    /// Bits the datasheet requires be written as 1 whatever the fields say.
    /// [`RegisterValue::to_byte`] ORs these in unconditionally, so a reserved bit
    /// cannot be lost by forgetting it at a call site.
    const RESERVED_ONES: u8;

    fn to_byte(self) -> u8;

    /// Decodes a byte read back from the chip, or `None` if it holds an encoding the
    /// datasheet does not define -- an undefined field value, a reserved bit that
    /// should read 1 reading 0, or one that should read 0 reading 1.
    ///
    /// This returns `Option` rather than the infallible `Self` a write-side type would
    /// suggest, because the only bytes that reach it come off the wire. Rounding a
    /// readback of `GAIN = 0b111` into a legal gain would hide exactly the kind of
    /// fault a readback exists to catch.
    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Self>;
}

// ---------------------------------------------------------------------------
// Shared field types
// ---------------------------------------------------------------------------

/// A set of channels within one device, laid out as the ADS1298 lays out its
/// per-channel registers: bit *n* is channel *n*.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ChannelMask(u8);

impl ChannelMask {
    pub(super) const NONE: ChannelMask = ChannelMask(0x00);
    pub(super) const ALL: ChannelMask = ChannelMask(0xFF);

    pub(super) const fn from_bits(bits: u8) -> ChannelMask {
        ChannelMask(bits)
    }

    pub(super) const fn bits(self) -> u8 {
        self.0
    }

    pub(super) const fn contains(self, channel: Channel) -> bool {
        self.0 & channel.bit() != 0
    }

    pub(super) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for ChannelMask {
    /// Names the set channels, because the reason this type exists is that
    /// `0xC00000` in a log line has to be decoded by hand before it means anything.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(formatter, "none");
        }
        let mut first = true;
        for channel in Channel::ALL {
            if self.contains(channel) {
                if !first {
                    write!(formatter, ",")?;
                }
                write!(formatter, "{channel}")?;
                first = false;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CONFIG1 (0x01)
// ---------------------------------------------------------------------------

/// CONFIG1 `DR[2:0]` -- the modulator-clock divider that sets the output data rate.
///
/// `0b111` is reserved and has no variant, which is why
/// [`DataRate::from_bits`] is fallible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DataRate {
    #[cfg(test)]
    ModulatorClockOver16 = 0b000,
    #[cfg(test)]
    ModulatorClockOver32 = 0b001,
    #[cfg(test)]
    ModulatorClockOver64 = 0b010,
    #[cfg(test)]
    ModulatorClockOver128 = 0b011,
    ModulatorClockOver256 = 0b100,
    #[cfg(test)]
    ModulatorClockOver512 = 0b101,
    #[cfg(test)]
    ModulatorClockOver1024 = 0b110,
}

impl DataRate {
    const fn bits(self) -> u8 {
        self as u8
    }

    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<DataRate> {
        match bits {
            0b000 => Some(DataRate::ModulatorClockOver16),
            0b001 => Some(DataRate::ModulatorClockOver32),
            0b010 => Some(DataRate::ModulatorClockOver64),
            0b011 => Some(DataRate::ModulatorClockOver128),
            0b100 => Some(DataRate::ModulatorClockOver256),
            0b101 => Some(DataRate::ModulatorClockOver512),
            0b110 => Some(DataRate::ModulatorClockOver1024),
            _ => None,
        }
    }

    /// The output data rate this divider produces, in samples per second.
    ///
    /// fMOD is fCLK/4 in high-resolution mode and fCLK/8 in low power, and `DR`
    /// selects fMOD divided by `16 << DR`. The mode is therefore half the answer: the
    /// same `DR` converts at half the rate in low power, which is what makes the
    /// power-on default (low power, fMOD/1024) come out at 250 SPS.
    pub(super) const fn samples_per_second(self, high_resolution: bool) -> u32 {
        let modulator_clock_hz = if high_resolution {
            CLOCK_HZ / 4
        } else {
            CLOCK_HZ / 8
        };
        modulator_clock_hz / (16u32 << self.bits())
    }
}

/// CONFIG1 -- conversion mode, readback mode, clock output, data rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config1 {
    /// `HR` (bit 7). High-resolution mode; clearing it halves fMOD and so halves the
    /// conversion rate for a given `data_rate`.
    pub(super) high_resolution: bool,
    /// `DAISY_EN` (bit 6). The datasheet's 1 is "multiple readback", meaning each chip
    /// is read over its own chip select -- how this board is wired. 0 would daisy-chain
    /// the two chips' data out through a single DOUT.
    pub(super) multiple_readback: bool,
    /// `CLK_EN` (bit 5). Drives the internal oscillator out of the CLK pin. This is
    /// how chip A clocks chip B, so exactly one chip may set it.
    pub(super) output_clock_enabled: bool,
    /// `DR[2:0]` (bits 2:0).
    pub(super) data_rate: DataRate,
}

/// Bits 4:3 of CONFIG1 are reserved and must be written 0.
#[cfg(test)]
const CONFIG1_RESERVED_ZEROS: u8 = 0b0001_1000;

impl RegisterValue for Config1 {
    const ADDRESS: Register = Register::Config1;
    const RESERVED_ONES: u8 = 0x00;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.high_resolution as u8) << 7)
            | ((self.multiple_readback as u8) << 6)
            | ((self.output_clock_enabled as u8) << 5)
            | self.data_rate.bits()
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Config1> {
        if byte & CONFIG1_RESERVED_ZEROS != 0 {
            return None;
        }
        Some(Config1 {
            high_resolution: byte & (1 << 7) != 0,
            multiple_readback: byte & (1 << 6) != 0,
            output_clock_enabled: byte & (1 << 5) != 0,
            data_rate: DataRate::from_bits(byte & 0b111)?,
        })
    }
}

// ---------------------------------------------------------------------------
// CONFIG2 (0x02)
// ---------------------------------------------------------------------------

/// CONFIG2 `TEST_AMP` (bit 2) -- the internal test signal's amplitude, as a multiple of
/// the datasheet's `(VREFP - VREFN) / 2400`, which is 1 mV at the 2.4 V reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TestSignalAmplitude {
    Single = 0,
    #[cfg(test)]
    Double = 1,
}

/// CONFIG2 `TEST_FREQ[1:0]` (bits 1:0) -- how fast the internal test signal alternates.
/// `0b10` is not a defined encoding and has no variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TestSignalFrequency {
    /// Square wave at fCLK / 2^21, about 1 Hz at the 2.048 MHz clock.
    PulsedSlow = 0b00,
    /// Square wave at fCLK / 2^20, twice `PulsedSlow`.
    #[cfg(test)]
    PulsedFast = 0b01,
    /// The signal is held at one level instead of alternating.
    #[cfg(test)]
    HeldAtDirectCurrent = 0b11,
}

impl TestSignalFrequency {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<TestSignalFrequency> {
        match bits {
            0b00 => Some(TestSignalFrequency::PulsedSlow),
            0b01 => Some(TestSignalFrequency::PulsedFast),
            0b11 => Some(TestSignalFrequency::HeldAtDirectCurrent),
            _ => None,
        }
    }
}

/// CONFIG2 -- the internal test signal generator and the WCT chopping clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config2 {
    /// `WCT_CHOP` (bit 5). Holds the Wilson-center-terminal chopping frequency
    /// constant instead of letting it vary with the data rate.
    pub(super) chopping_frequency_constant: bool,
    /// `INT_TEST` (bit 4). Generates the test signal internally. Without it the test
    /// signal has to be driven in on a pin, which this board does not do, so a channel
    /// muxed to the test signal with this clear reads nothing.
    pub(super) test_signal_enabled: bool,
    /// `TEST_AMP` (bit 2).
    pub(super) test_signal_amplitude: TestSignalAmplitude,
    /// `TEST_FREQ[1:0]` (bits 1:0).
    pub(super) test_signal_frequency: TestSignalFrequency,
}

/// Bits 7:6 and bit 3 of CONFIG2 are reserved and must be written 0.
#[cfg(test)]
const CONFIG2_RESERVED_ZEROS: u8 = 0b1100_1000;

impl RegisterValue for Config2 {
    const ADDRESS: Register = Register::Config2;
    const RESERVED_ONES: u8 = 0x00;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.chopping_frequency_constant as u8) << 5)
            | ((self.test_signal_enabled as u8) << 4)
            | ((self.test_signal_amplitude as u8) << 2)
            | (self.test_signal_frequency as u8)
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Config2> {
        if byte & CONFIG2_RESERVED_ZEROS != 0 {
            return None;
        }
        Some(Config2 {
            chopping_frequency_constant: byte & (1 << 5) != 0,
            test_signal_enabled: byte & (1 << 4) != 0,
            test_signal_amplitude: if byte & (1 << 2) != 0 {
                TestSignalAmplitude::Double
            } else {
                TestSignalAmplitude::Single
            },
            test_signal_frequency: TestSignalFrequency::from_bits(byte & 0b11)?,
        })
    }
}

// ---------------------------------------------------------------------------
// CONFIG3 (0x03)
// ---------------------------------------------------------------------------

/// CONFIG3 -- reference buffer and right-leg-drive amplifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config3 {
    /// `PD_REFBUF` (bit 7). Powers up the internal reference buffer. Clearing it means
    /// the reference has to be supplied externally.
    pub(super) internal_reference_enabled: bool,
    /// `VREF_4V` (bit 5). Selects the 4 V reference; clear selects 2.4 V, which is
    /// what [`super::preprocess`] converts codes against.
    pub(super) four_volt_reference: bool,
    /// `RLD_MEAS` (bit 4). Routes the right-leg-drive signal to a channel for
    /// measurement.
    pub(super) right_leg_drive_measurement: bool,
    /// `RLDREF_INT` (bit 3). Generates the right-leg-drive reference internally
    /// (mid-supply). Clear means it is fed in on the RLDREF pin.
    ///
    /// Open hardware question, raised by a datasheet audit and not yet answered: chip A
    /// runs with `right_leg_drive_enabled` set and this clear, so its amplifier sources
    /// its reference externally. That is only correct if the board actually puts a
    /// divider on the RLDREF pin. If it does not, the amplifier is referenced to
    /// whatever that pin floats to. Setting this to `true` would be the firmware-side
    /// fix, but the board is what needs checking first, so the value stays as it is.
    pub(super) right_leg_drive_reference_internal: bool,
    /// `PD_RLD` (bit 2). Powers up the right-leg-drive buffer.
    pub(super) right_leg_drive_enabled: bool,
    /// `RLD_LOFF_SENS` (bit 1). Enables the right-leg-drive lead-off sense function.
    pub(super) right_leg_drive_lead_off_sense: bool,
}

/// Bit 6 of CONFIG3 is reserved and must always be written 1.
///
/// Bit 0 is `RLD_STAT`, which is read-only -- it reports whether the right-leg-drive
/// lead is connected. There is no field for it: it is written 0, and on readback it is
/// ignored rather than rejected, because a 1 there is the chip reporting a lead state
/// and not a corrupted byte.
const CONFIG3_RESERVED_ONE: u8 = 0b0100_0000;

impl RegisterValue for Config3 {
    const ADDRESS: Register = Register::Config3;
    const RESERVED_ONES: u8 = CONFIG3_RESERVED_ONE;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.internal_reference_enabled as u8) << 7)
            | ((self.four_volt_reference as u8) << 5)
            | ((self.right_leg_drive_measurement as u8) << 4)
            | ((self.right_leg_drive_reference_internal as u8) << 3)
            | ((self.right_leg_drive_enabled as u8) << 2)
            | ((self.right_leg_drive_lead_off_sense as u8) << 1)
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Config3> {
        if byte & CONFIG3_RESERVED_ONE == 0 {
            return None;
        }
        Some(Config3 {
            internal_reference_enabled: byte & (1 << 7) != 0,
            four_volt_reference: byte & (1 << 5) != 0,
            right_leg_drive_measurement: byte & (1 << 4) != 0,
            right_leg_drive_reference_internal: byte & (1 << 3) != 0,
            right_leg_drive_enabled: byte & (1 << 2) != 0,
            right_leg_drive_lead_off_sense: byte & (1 << 1) != 0,
        })
    }
}

// ---------------------------------------------------------------------------
// LOFF (0x04)
// ---------------------------------------------------------------------------

/// LOFF `COMP_TH[2:0]` (bits 7:5) -- the lead-off comparator's trip points, as a
/// percentage of the supply. `NinetyFive` compares against 95% and 5%.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LeadOffComparatorThreshold {
    NinetyFivePercent = 0b000,
    #[cfg(test)]
    NinetyTwoAndAHalfPercent = 0b001,
    #[cfg(test)]
    NinetyPercent = 0b010,
    #[cfg(test)]
    EightySevenAndAHalfPercent = 0b011,
    #[cfg(test)]
    EightyFivePercent = 0b100,
    #[cfg(test)]
    EightyPercent = 0b101,
    #[cfg(test)]
    SeventyFivePercent = 0b110,
    #[cfg(test)]
    SeventyPercent = 0b111,
}

impl LeadOffComparatorThreshold {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<LeadOffComparatorThreshold> {
        match bits {
            0b000 => Some(LeadOffComparatorThreshold::NinetyFivePercent),
            0b001 => Some(LeadOffComparatorThreshold::NinetyTwoAndAHalfPercent),
            0b010 => Some(LeadOffComparatorThreshold::NinetyPercent),
            0b011 => Some(LeadOffComparatorThreshold::EightySevenAndAHalfPercent),
            0b100 => Some(LeadOffComparatorThreshold::EightyFivePercent),
            0b101 => Some(LeadOffComparatorThreshold::EightyPercent),
            0b110 => Some(LeadOffComparatorThreshold::SeventyFivePercent),
            _ => Some(LeadOffComparatorThreshold::SeventyPercent),
        }
    }
}

/// LOFF `ILEAD_OFF[1:0]` (bits 3:2) -- the excitation current the lead-off detector
/// pushes through the electrode, in nanoamps. Only used in current-source mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LeadOffCurrent {
    SixNanoamps = 0b00,
    #[cfg(test)]
    TwelveNanoamps = 0b01,
    #[cfg(test)]
    EighteenNanoamps = 0b10,
    #[cfg(test)]
    TwentyFourNanoamps = 0b11,
}

impl LeadOffCurrent {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<LeadOffCurrent> {
        match bits {
            0b00 => Some(LeadOffCurrent::SixNanoamps),
            0b01 => Some(LeadOffCurrent::TwelveNanoamps),
            0b10 => Some(LeadOffCurrent::EighteenNanoamps),
            _ => Some(LeadOffCurrent::TwentyFourNanoamps),
        }
    }
}

/// LOFF `FLEAD_OFF[1:0]` (bits 1:0) -- how lead-off is excited. `0b01` and `0b10` are
/// not defined encodings and have no variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LeadOffDetection {
    /// Alternating-current detection at a quarter of the data rate.
    #[cfg(test)]
    AlternatingCurrent = 0b00,
    /// Direct-current detection, which is what the driver uses: the comparators simply
    /// watch the electrode's DC level.
    DirectCurrent = 0b11,
}

impl LeadOffDetection {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<LeadOffDetection> {
        match bits {
            0b00 => Some(LeadOffDetection::AlternatingCurrent),
            0b11 => Some(LeadOffDetection::DirectCurrent),
            _ => None,
        }
    }
}

/// LOFF -- how the lead-off detector excites and judges an electrode. Which channels
/// it watches lives in [`LeadOffSensePositive`] / [`LeadOffSenseNegative`], and whether
/// the comparators are powered at all lives in [`Config4`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LeadOffControl {
    /// `COMP_TH[2:0]` (bits 7:5).
    pub(super) comparator_threshold: LeadOffComparatorThreshold,
    /// `VLEAD_OFF_EN` (bit 4). Selects pull-up/pull-down resistor mode; clear selects
    /// current-source mode, where `current` applies.
    pub(super) pull_resistor_mode: bool,
    /// `ILEAD_OFF[1:0]` (bits 3:2).
    pub(super) current: LeadOffCurrent,
    /// `FLEAD_OFF[1:0]` (bits 1:0).
    pub(super) detection: LeadOffDetection,
}

impl RegisterValue for LeadOffControl {
    const ADDRESS: Register = Register::Loff;
    const RESERVED_ONES: u8 = 0x00;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.comparator_threshold as u8) << 5)
            | ((self.pull_resistor_mode as u8) << 4)
            | ((self.current as u8) << 2)
            | (self.detection as u8)
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<LeadOffControl> {
        Some(LeadOffControl {
            comparator_threshold: LeadOffComparatorThreshold::from_bits(byte >> 5)?,
            pull_resistor_mode: byte & (1 << 4) != 0,
            current: LeadOffCurrent::from_bits((byte >> 2) & 0b11)?,
            detection: LeadOffDetection::from_bits(byte & 0b11)?,
        })
    }
}

// ---------------------------------------------------------------------------
// CH1SET..CH8SET (0x05..0x0C)
// ---------------------------------------------------------------------------

/// CHnSET `GAIN[2:0]` (bits 6:4) -- the programmable gain amplifier's setting.
///
/// Note that the encoding does not start at 1: `0b000` is gain 6, which is why writing
/// an all-zero CHnSET gives a working channel at the gain
/// [`super::preprocess`] converts against. `0b111` is not a legal encoding and has no
/// variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Gain {
    Six = 0b000,
    #[cfg(test)]
    One = 0b001,
    #[cfg(test)]
    Two = 0b010,
    #[cfg(test)]
    Three = 0b011,
    #[cfg(test)]
    Four = 0b100,
    #[cfg(test)]
    Eight = 0b101,
    #[cfg(test)]
    Twelve = 0b110,
}

impl Gain {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<Gain> {
        match bits {
            0b000 => Some(Gain::Six),
            0b001 => Some(Gain::One),
            0b010 => Some(Gain::Two),
            0b011 => Some(Gain::Three),
            0b100 => Some(Gain::Four),
            0b101 => Some(Gain::Eight),
            0b110 => Some(Gain::Twelve),
            _ => None,
        }
    }
}

/// CHnSET `MUX[2:0]` (bits 2:0) -- what the channel's amplifier is connected to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChannelInput {
    /// The electrode pair. The only setting that reads a subject.
    Electrode = 0b000,
    /// Both inputs tied to the same mid-supply point, so the channel reads its own
    /// noise and offset. Used on the channels that are not carrying the test signal.
    Shorted = 0b001,
    #[cfg(test)]
    RightLegDriveMeasurement = 0b010,
    #[cfg(test)]
    SupplyMeasurement = 0b011,
    #[cfg(test)]
    TemperatureSensor = 0b100,
    /// The internal square wave, which only exists when
    /// [`Config2::test_signal_enabled`] is set.
    TestSignal = 0b101,
    #[cfg(test)]
    RightLegDrivePositiveDriver = 0b110,
    #[cfg(test)]
    RightLegDriveNegativeDriver = 0b111,
}

impl ChannelInput {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<ChannelInput> {
        match bits {
            0b000 => Some(ChannelInput::Electrode),
            0b001 => Some(ChannelInput::Shorted),
            0b010 => Some(ChannelInput::RightLegDriveMeasurement),
            0b011 => Some(ChannelInput::SupplyMeasurement),
            0b100 => Some(ChannelInput::TemperatureSensor),
            0b101 => Some(ChannelInput::TestSignal),
            0b110 => Some(ChannelInput::RightLegDrivePositiveDriver),
            _ => Some(ChannelInput::RightLegDriveNegativeDriver),
        }
    }
}

/// CHnSET -- one channel's power state, gain and input source.
///
/// This is the one register value with no `ADDRESS`: eight registers share the layout,
/// so the channel is a separate argument to
/// [`super::ads1298::Ads1298Device::write_channel_settings`] and the address comes from
/// [`Register::channel_settings`]. Making the address const here would have meant eight
/// near-identical types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChannelSettings {
    /// `PD` (bit 7). Powers the channel down.
    pub(super) powered_down: bool,
    /// `GAIN[2:0]` (bits 6:4).
    pub(super) gain: Gain,
    /// `MUX[2:0]` (bits 2:0).
    pub(super) input: ChannelInput,
}

/// Bit 3 of CHnSET is reserved and must be written 0.
#[cfg(test)]
const CHANNEL_SETTINGS_RESERVED_ZEROS: u8 = 0b0000_1000;

impl ChannelSettings {
    /// Matches [`RegisterValue::RESERVED_ONES`]; CHnSET has no bits forced to 1.
    const RESERVED_ONES: u8 = 0x00;

    pub(super) const fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.powered_down as u8) << 7)
            | ((self.gain as u8) << 4)
            | (self.input as u8)
    }

    /// Symmetry with [`RegisterValue::from_byte`]; CHnSET has no `ADDRESS`, so it
    /// cannot implement the trait.
    #[cfg(test)]
    pub(super) const fn from_byte(byte: u8) -> Option<ChannelSettings> {
        if byte & CHANNEL_SETTINGS_RESERVED_ZEROS != 0 {
            return None;
        }
        let gain = match Gain::from_bits((byte >> 4) & 0b111) {
            Some(gain) => gain,
            None => return None,
        };
        let input = match ChannelInput::from_bits(byte & 0b111) {
            Some(input) => input,
            None => return None,
        };
        Some(ChannelSettings {
            powered_down: byte & (1 << 7) != 0,
            gain,
            input,
        })
    }
}

// ---------------------------------------------------------------------------
// The channel-bitmask registers: RLD_SENSP/N, LOFF_SENSP/N, LOFF_FLIP
// ---------------------------------------------------------------------------

/// Defines a register whose value is exactly a [`ChannelMask`].
///
/// Five registers share that shape. Each gets its own newtype rather than one shared
/// `ChannelMask: RegisterValue` impl, because the address lives on the type: with one
/// impl there would be one address, and writing the RLD mask to the LOFF register would
/// compile.
macro_rules! channel_mask_register {
    ($(#[$attribute:meta])* $name:ident => $register:ident) => {
        $(#[$attribute])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(super) struct $name(pub(super) ChannelMask);

        impl RegisterValue for $name {
            const ADDRESS: Register = Register::$register;
            const RESERVED_ONES: u8 = 0x00;

            fn to_byte(self) -> u8 {
                Self::RESERVED_ONES | self.0.bits()
            }

            #[cfg(test)]
            fn from_byte(byte: u8) -> Option<Self> {
                Some($name(ChannelMask::from_bits(byte)))
            }
        }
    };
}

channel_mask_register! {
    /// RLD_SENSP -- which channels' positive electrodes are summed into the right-leg
    /// drive.
    RightLegDriveSensePositive => RldSensP
}

channel_mask_register! {
    /// RLD_SENSN -- which channels' negative electrodes are summed into the right-leg
    /// drive.
    RightLegDriveSenseNegative => RldSensN
}

channel_mask_register! {
    /// LOFF_SENSP -- which channels' positive electrodes the lead-off detector watches.
    LeadOffSensePositive => LoffSensP
}

channel_mask_register! {
    /// LOFF_SENSN -- which channels' negative electrodes the lead-off detector watches.
    LeadOffSenseNegative => LoffSensN
}

channel_mask_register! {
    /// LOFF_FLIP -- which channels swap the direction of their lead-off excitation
    /// current between the positive and negative electrode.
    LeadOffFlip => LoffFlip
}

// ---------------------------------------------------------------------------
// RESP (0x16)
// ---------------------------------------------------------------------------

/// RESP `RESP_PH[2:0]` (bits 4:2) -- the phase offset between the respiration
/// modulation and demodulation clocks, in equal steps across half a cycle. The step
/// size depends on the modulation frequency selected in [`Config4`], so the variants
/// are named by step rather than by degrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RespirationPhase {
    Zero = 0b000,
    #[cfg(test)]
    One = 0b001,
    #[cfg(test)]
    Two = 0b010,
    #[cfg(test)]
    Three = 0b011,
    #[cfg(test)]
    Four = 0b100,
    #[cfg(test)]
    Five = 0b101,
    #[cfg(test)]
    Six = 0b110,
    #[cfg(test)]
    Seven = 0b111,
}

impl RespirationPhase {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<RespirationPhase> {
        match bits {
            0b000 => Some(RespirationPhase::Zero),
            0b001 => Some(RespirationPhase::One),
            0b010 => Some(RespirationPhase::Two),
            0b011 => Some(RespirationPhase::Three),
            0b100 => Some(RespirationPhase::Four),
            0b101 => Some(RespirationPhase::Five),
            0b110 => Some(RespirationPhase::Six),
            _ => Some(RespirationPhase::Seven),
        }
    }
}

/// RESP `RESP_CTRL[1:0]` (bits 1:0) -- where the respiration modulation signals come
/// from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RespirationControl {
    /// No respiration measurement. The only setting this firmware uses: an EMG
    /// wristband has no thoracic impedance to measure.
    None = 0b00,
    #[cfg(test)]
    External = 0b01,
    #[cfg(test)]
    InternalSignals = 0b10,
    #[cfg(test)]
    UserGeneratedSignals = 0b11,
}

impl RespirationControl {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<RespirationControl> {
        match bits {
            0b00 => Some(RespirationControl::None),
            0b01 => Some(RespirationControl::External),
            0b10 => Some(RespirationControl::InternalSignals),
            _ => Some(RespirationControl::UserGeneratedSignals),
        }
    }
}

/// RESP -- the respiration modulation/demodulation block, which this board leaves off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Respiration {
    /// `RESP_DEMOD_EN1` (bit 7).
    pub(super) demodulation_enabled: bool,
    /// `RESP_MOD_EN1` (bit 6).
    pub(super) modulation_enabled: bool,
    /// `RESP_PH[2:0]` (bits 4:2).
    pub(super) phase: RespirationPhase,
    /// `RESP_CTRL[1:0]` (bits 1:0).
    pub(super) control: RespirationControl,
}

/// Bit 5 of RESP is reserved and must always be written 1. This is the whole of the
/// `0x20` the driver writes to a block it does not use.
const RESPIRATION_RESERVED_ONE: u8 = 0b0010_0000;

impl RegisterValue for Respiration {
    const ADDRESS: Register = Register::Resp;
    const RESERVED_ONES: u8 = RESPIRATION_RESERVED_ONE;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.demodulation_enabled as u8) << 7)
            | ((self.modulation_enabled as u8) << 6)
            | ((self.phase as u8) << 2)
            | (self.control as u8)
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Respiration> {
        if byte & RESPIRATION_RESERVED_ONE == 0 {
            return None;
        }
        Some(Respiration {
            demodulation_enabled: byte & (1 << 7) != 0,
            modulation_enabled: byte & (1 << 6) != 0,
            phase: RespirationPhase::from_bits((byte >> 2) & 0b111)?,
            control: RespirationControl::from_bits(byte & 0b11)?,
        })
    }
}

// ---------------------------------------------------------------------------
// CONFIG4 (0x17)
// ---------------------------------------------------------------------------

/// CONFIG4 `RESP_FREQ[2:0]` (bits 7:5) -- the respiration modulation frequency. Only
/// meaningful when [`Respiration::control`] is not `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RespirationFrequency {
    SixtyFourKilohertz = 0b000,
    #[cfg(test)]
    ThirtyTwoKilohertz = 0b001,
    #[cfg(test)]
    SixteenKilohertz = 0b010,
    #[cfg(test)]
    EightKilohertz = 0b011,
    #[cfg(test)]
    FourKilohertz = 0b100,
    #[cfg(test)]
    TwoKilohertz = 0b101,
    #[cfg(test)]
    OneKilohertz = 0b110,
    #[cfg(test)]
    FiveHundredHertz = 0b111,
}

impl RespirationFrequency {
    #[cfg(test)]
    const fn from_bits(bits: u8) -> Option<RespirationFrequency> {
        match bits {
            0b000 => Some(RespirationFrequency::SixtyFourKilohertz),
            0b001 => Some(RespirationFrequency::ThirtyTwoKilohertz),
            0b010 => Some(RespirationFrequency::SixteenKilohertz),
            0b011 => Some(RespirationFrequency::EightKilohertz),
            0b100 => Some(RespirationFrequency::FourKilohertz),
            0b101 => Some(RespirationFrequency::TwoKilohertz),
            0b110 => Some(RespirationFrequency::OneKilohertz),
            _ => Some(RespirationFrequency::FiveHundredHertz),
        }
    }
}

/// CONFIG4 -- conversion mode and the lead-off comparator power switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config4 {
    /// `RESP_FREQ[2:0]` (bits 7:5).
    pub(super) respiration_frequency: RespirationFrequency,
    /// `SINGLE_SHOT` (bit 3). One conversion per START pulse instead of continuous
    /// conversion. Clear is what the streaming path needs.
    pub(super) single_shot: bool,
    /// `WCT_TO_RLD` (bit 2). Routes the Wilson center terminal into the right-leg
    /// drive.
    pub(super) wilson_center_terminal_to_right_leg_drive: bool,
    /// `PD_LOFF_COMP` (bit 1). Powers up the lead-off comparators. Without this the
    /// status word's lead-off bits never set, whatever LOFF_SENSP/N say.
    pub(super) lead_off_comparators_enabled: bool,
}

/// Bits 4 and 0 of CONFIG4 are reserved and must be written 0.
#[cfg(test)]
const CONFIG4_RESERVED_ZEROS: u8 = 0b0001_0001;

impl RegisterValue for Config4 {
    const ADDRESS: Register = Register::Config4;
    const RESERVED_ONES: u8 = 0x00;

    fn to_byte(self) -> u8 {
        Self::RESERVED_ONES
            | ((self.respiration_frequency as u8) << 5)
            | ((self.single_shot as u8) << 3)
            | ((self.wilson_center_terminal_to_right_leg_drive as u8) << 2)
            | ((self.lead_off_comparators_enabled as u8) << 1)
    }

    #[cfg(test)]
    fn from_byte(byte: u8) -> Option<Config4> {
        if byte & CONFIG4_RESERVED_ZEROS != 0 {
            return None;
        }
        Some(Config4 {
            respiration_frequency: RespirationFrequency::from_bits(byte >> 5)?,
            single_shot: byte & (1 << 3) != 0,
            wilson_center_terminal_to_right_leg_drive: byte & (1 << 2) != 0,
            lead_off_comparators_enabled: byte & (1 << 1) != 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Blocks this board does not use: GPIO, PACE, WCT1, WCT2
// ---------------------------------------------------------------------------

/// Defines a register the firmware only ever writes all-zero.
///
/// These blocks are switched off and stay off, so a field-by-field model of them would
/// be documentation of hardware nobody here uses, and dead code the moment it was
/// written. A type with exactly one state still buys the two things that matter: the
/// address travels with the value, and `0x00` cannot be written to the wrong register.
macro_rules! all_zero_register {
    ($(#[$attribute:meta])* $name:ident => $register:ident) => {
        $(#[$attribute])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(super) struct $name;

        impl RegisterValue for $name {
            const ADDRESS: Register = Register::$register;
            const RESERVED_ONES: u8 = 0x00;

            fn to_byte(self) -> u8 {
                Self::RESERVED_ONES
            }

            #[cfg(test)]
            fn from_byte(byte: u8) -> Option<Self> {
                if byte == 0x00 {
                    Some($name)
                } else {
                    None
                }
            }
        }
    };
}

all_zero_register! {
    /// GPIO, all-zero: the four general-purpose pins are unused on this board. The
    /// power-on default is `0x0F` (all four configured as inputs), so this write is
    /// not a no-op even though the value is zero.
    GeneralPurposeInputOutputOff => Gpio
}

all_zero_register! {
    /// PACE, all-zero: pace-detect is an ECG feature with nothing to do on a wristband.
    PaceDetectOff => Pace
}

all_zero_register! {
    /// WCT1, all-zero: the Wilson center terminal amplifiers are powered down and no
    /// channel is routed to one.
    WilsonCenterTerminalOneOff => Wct1
}

all_zero_register! {
    /// WCT2, all-zero, for the same reason as WCT1.
    WilsonCenterTerminalTwoOff => Wct2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adc::channel::CHANNELS_PER_DEVICE;

    #[test]
    fn channel_settings_addresses_are_consecutive_from_ch1set() {
        for channel in Channel::ALL {
            assert_eq!(
                Register::channel_settings(channel).addr(),
                Register::Ch1Set.addr() + channel.index() as u8
            );
        }
        assert_eq!(Channel::ALL.len(), CHANNELS_PER_DEVICE);
    }

    #[test]
    fn reserved_ones_survive_an_all_clear_value() {
        // The point of `RESERVED_ONES`: a value with every field off still has to
        // carry the bits the datasheet forces high.
        let config3 = Config3 {
            internal_reference_enabled: false,
            four_volt_reference: false,
            right_leg_drive_measurement: false,
            right_leg_drive_reference_internal: false,
            right_leg_drive_enabled: false,
            right_leg_drive_lead_off_sense: false,
        };
        assert_eq!(config3.to_byte(), 0x40);

        let respiration = Respiration {
            demodulation_enabled: false,
            modulation_enabled: false,
            phase: RespirationPhase::Zero,
            control: RespirationControl::None,
        };
        assert_eq!(respiration.to_byte(), 0x20);
    }

    #[test]
    fn data_rate_matches_the_datasheet_worked_examples() {
        // High resolution: fMOD = 2.048 MHz / 4, over 256 is the datasheet's own
        // worked example, 2000 SPS.
        assert_eq!(
            DataRate::ModulatorClockOver256.samples_per_second(true),
            2000
        );
        // Low power halves fMOD, so the same divider halves the rate.
        assert_eq!(
            DataRate::ModulatorClockOver256.samples_per_second(false),
            1000
        );
        // The power-on default, CONFIG1 = 0x06: low power, fMOD/1024, the
        // datasheet's 250 SPS.
        assert_eq!(
            DataRate::ModulatorClockOver1024.samples_per_second(false),
            250
        );
    }

    #[test]
    fn power_on_default_config1_decodes_as_the_datasheet_describes() {
        // 0x06 is the reset value, and recognising it in a readback is how a chip that
        // has quietly reverted announces itself.
        let default = Config1::from_byte(0x06).expect("0x06 is a legal CONFIG1");
        assert!(!default.high_resolution);
        assert!(!default.multiple_readback);
        assert!(!default.output_clock_enabled);
        assert_eq!(default.data_rate, DataRate::ModulatorClockOver1024);
        assert_eq!(default.data_rate.samples_per_second(false), 250);
    }

    #[test]
    fn undefined_encodings_do_not_decode() {
        // DR = 0b111 is reserved.
        assert!(Config1::from_byte(0xE7).is_none());
        // CONFIG1 bits 4:3 are reserved-zero.
        assert!(Config1::from_byte(0xEC).is_none());
        // CONFIG3 bit 6 must read back as 1.
        assert!(Config3::from_byte(0x80).is_none());
        // RESP bit 5 must read back as 1.
        assert!(Respiration::from_byte(0x00).is_none());
        // CHnSET GAIN = 0b111 is not a legal encoding.
        assert!(ChannelSettings::from_byte(0x70).is_none());
        // FLEAD_OFF = 0b01 is not a defined encoding.
        assert!(LeadOffControl::from_byte(0x11).is_none());
    }

    #[test]
    fn channel_masks_name_their_set_channels() {
        assert_eq!(ChannelMask::NONE.to_string(), "none");
        assert_eq!(ChannelMask::ALL.to_string(), "0,1,2,3,4,5,6,7");
        assert_eq!(ChannelMask::from_bits(0b1000_1001).to_string(), "0,3,7");
        assert!(ChannelMask::ALL.contains(Channel::checked(5)));
        assert!(!ChannelMask::NONE.contains(Channel::checked(5)));
    }
}
