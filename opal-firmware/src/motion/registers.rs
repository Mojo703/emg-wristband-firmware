//! Register maps for the MinIMU-9 v6's two chips, and a typed value for every
//! register the drivers write.
//!
//! Datasheets: ST DS12140 (LSM6DSO) §8–9 and DocID024204 (LIS3MDL) §7–8.
//!
//! The values are types for the same reason as in `adc::registers`: a bare byte
//! can be wrong three ways at one call site (wrong address, a byte for a different
//! register, a dropped bit), and a wrong byte here does not fail — it samples at
//! the wrong range and shows up as a gate that never trips. So each written
//! register is a struct with named fields, and its value type carries its address.
//! The two chips get separate address types, so a value for one cannot be sent to
//! the other.
//!
//! Options the shipped configuration does not use are `#[cfg(test)]`: the full
//! tables stay checked against the datasheet, and picking another option in
//! `config.rs` means deleting that one attribute.

// ---------------------------------------------------------------------------
// LSM6DSO: accelerometer and gyroscope
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InertialRegister {
    WhoAmI = 0x0F,
    Ctrl1Xl = 0x10,
    Ctrl2G = 0x11,
    Ctrl3C = 0x12,
    Ctrl9Xl = 0x18,
    StatusReg = 0x1E,
    /// The first of twelve output bytes: gyroscope X, Y, Z, then accelerometer
    /// X, Y, Z, each a little-endian `i16`.
    OutxLG = 0x22,
}

impl InertialRegister {
    pub(super) const fn address(self) -> u8 {
        self as u8
    }
}

/// WHO_AM_I's fixed value.
pub(super) const INERTIAL_IDENTITY: u8 = 0x6C;

/// STATUS_REG `XLDA` (bit 0) and `GDA` (bit 1): a new sample is waiting.
pub(super) const STATUS_NEW_SAMPLE: u8 = 0b11;

/// Bytes in one gyroscope-plus-accelerometer burst from [`InertialRegister::OutxLG`].
pub(super) const SAMPLE_BYTES: usize = 12;

/// A value that can be written to exactly one LSM6DSO register. The address is a
/// const on the type, so the read-only registers (WHO_AM_I, STATUS_REG, the
/// outputs) have no value type and cannot be written.
pub(super) trait InertialRegisterValue: Copy {
    const ADDRESS: InertialRegister;
    fn to_byte(self) -> u8;
}

/// Output data rate, shared by both sensors: `ODR[3:0]`, bits 7:4 of CTRL1_XL and
/// CTRL2_G, high-performance mode. `0b0000` is power-down and has no variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputDataRate {
    #[cfg(test)]
    Hertz12Point5 = 0b0001,
    #[cfg(test)]
    Hertz26 = 0b0010,
    #[cfg(test)]
    Hertz52 = 0b0011,
    Hertz104 = 0b0100,
    #[cfg(test)]
    Hertz208 = 0b0101,
    #[cfg(test)]
    Hertz416 = 0b0110,
    #[cfg(test)]
    Hertz833 = 0b0111,
    #[cfg(test)]
    Hertz1660 = 0b1000,
}

impl OutputDataRate {
    const fn bits(self) -> u8 {
        self as u8
    }

    pub(crate) const fn hertz(self) -> f32 {
        match self {
            #[cfg(test)]
            OutputDataRate::Hertz12Point5 => 12.5,
            #[cfg(test)]
            OutputDataRate::Hertz26 => 26.0,
            #[cfg(test)]
            OutputDataRate::Hertz52 => 52.0,
            OutputDataRate::Hertz104 => 104.0,
            #[cfg(test)]
            OutputDataRate::Hertz208 => 208.0,
            #[cfg(test)]
            OutputDataRate::Hertz416 => 416.0,
            #[cfg(test)]
            OutputDataRate::Hertz833 => 833.0,
            #[cfg(test)]
            OutputDataRate::Hertz1660 => 1660.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CTRL1_XL (0x10)
// ---------------------------------------------------------------------------

/// CTRL1_XL `FS[1:0]_XL` (bits 3:2), with CTRL8_XL `XL_FS_MODE` at its reset 0.
/// The codes are not in size order; the datasheet's table is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccelerometerRange {
    #[cfg(test)]
    TwoG = 0b00,
    #[cfg(test)]
    SixteenG = 0b01,
    #[cfg(test)]
    FourG = 0b10,
    EightG = 0b11,
}

impl AccelerometerRange {
    const fn bits(self) -> u8 {
        self as u8
    }

    pub(crate) const fn full_scale_g(self) -> f32 {
        match self {
            #[cfg(test)]
            AccelerometerRange::TwoG => 2.0,
            #[cfg(test)]
            AccelerometerRange::FourG => 4.0,
            AccelerometerRange::EightG => 8.0,
            #[cfg(test)]
            AccelerometerRange::SixteenG => 16.0,
        }
    }

    /// Sensitivity, DS12140 table 2.
    pub(crate) const fn g_per_count(self) -> f32 {
        match self {
            #[cfg(test)]
            AccelerometerRange::TwoG => 0.061e-3,
            #[cfg(test)]
            AccelerometerRange::FourG => 0.122e-3,
            AccelerometerRange::EightG => 0.244e-3,
            #[cfg(test)]
            AccelerometerRange::SixteenG => 0.488e-3,
        }
    }
}

/// CTRL1_XL -- accelerometer data rate and range. `LPF2_XL_EN` (bit 1) stays 0:
/// the gate wants the raw first-stage output, not a second low-pass. Bit 0 is
/// reserved 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ctrl1Xl {
    pub(super) output_data_rate: OutputDataRate,
    pub(super) range: AccelerometerRange,
}

impl InertialRegisterValue for Ctrl1Xl {
    const ADDRESS: InertialRegister = InertialRegister::Ctrl1Xl;

    fn to_byte(self) -> u8 {
        (self.output_data_rate.bits() << 4) | (self.range.bits() << 2)
    }
}

// ---------------------------------------------------------------------------
// CTRL2_G (0x11)
// ---------------------------------------------------------------------------

/// CTRL2_G `FS[1:0]_G` (bits 3:2) and `FS_125` (bit 1), as the three-bit field
/// `[3:1]`. ±125 °/s is its own bit rather than an FS_G code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GyroscopeRange {
    #[cfg(test)]
    OneHundredTwentyFiveDegreesPerSecond = 0b001,
    #[cfg(test)]
    TwoHundredFiftyDegreesPerSecond = 0b000,
    #[cfg(test)]
    FiveHundredDegreesPerSecond = 0b010,
    OneThousandDegreesPerSecond = 0b100,
    #[cfg(test)]
    TwoThousandDegreesPerSecond = 0b110,
}

impl GyroscopeRange {
    const fn bits(self) -> u8 {
        self as u8
    }

    pub(crate) const fn full_scale_degrees_per_second(self) -> f32 {
        match self {
            #[cfg(test)]
            GyroscopeRange::OneHundredTwentyFiveDegreesPerSecond => 125.0,
            #[cfg(test)]
            GyroscopeRange::TwoHundredFiftyDegreesPerSecond => 250.0,
            #[cfg(test)]
            GyroscopeRange::FiveHundredDegreesPerSecond => 500.0,
            GyroscopeRange::OneThousandDegreesPerSecond => 1000.0,
            #[cfg(test)]
            GyroscopeRange::TwoThousandDegreesPerSecond => 2000.0,
        }
    }

    /// Sensitivity, DS12140 table 2.
    pub(crate) const fn degrees_per_second_per_count(self) -> f32 {
        match self {
            #[cfg(test)]
            GyroscopeRange::OneHundredTwentyFiveDegreesPerSecond => 4.375e-3,
            #[cfg(test)]
            GyroscopeRange::TwoHundredFiftyDegreesPerSecond => 8.75e-3,
            #[cfg(test)]
            GyroscopeRange::FiveHundredDegreesPerSecond => 17.5e-3,
            GyroscopeRange::OneThousandDegreesPerSecond => 35.0e-3,
            #[cfg(test)]
            GyroscopeRange::TwoThousandDegreesPerSecond => 70.0e-3,
        }
    }
}

/// CTRL2_G -- gyroscope data rate and range. Bit 0 is reserved 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ctrl2G {
    pub(super) output_data_rate: OutputDataRate,
    pub(super) range: GyroscopeRange,
}

impl InertialRegisterValue for Ctrl2G {
    const ADDRESS: InertialRegister = InertialRegister::Ctrl2G;

    fn to_byte(self) -> u8 {
        (self.output_data_rate.bits() << 4) | (self.range.bits() << 1)
    }
}

// ---------------------------------------------------------------------------
// CTRL3_C (0x12)
// ---------------------------------------------------------------------------

/// CTRL3_C -- interface behaviour. The other bits (BOOT, interrupt polarity and
/// drive, 3-wire SPI) stay at their reset 0s: the carrier wires none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ctrl3C {
    /// `BDU` (bit 6). The output registers hold still between the low and high
    /// byte reads of one sample, so a burst never stitches two samples together.
    pub(super) block_data_update: bool,
    /// `IF_INC` (bit 2). Register address auto-increment on bursts; on at reset,
    /// written anyway so the readback states the whole intent.
    pub(super) address_increment: bool,
    /// `SW_RESET` (bit 0). Restores every control register to its default and
    /// clears itself when done.
    pub(super) software_reset: bool,
}

/// CTRL3_C's `SW_RESET` bit, for polling the reset's completion.
pub(super) const CTRL3_C_SOFTWARE_RESET: u8 = 1 << 0;

impl InertialRegisterValue for Ctrl3C {
    const ADDRESS: InertialRegister = InertialRegister::Ctrl3C;

    fn to_byte(self) -> u8 {
        ((self.block_data_update as u8) << 6)
            | ((self.address_increment as u8) << 2)
            | (self.software_reset as u8)
    }
}

// ---------------------------------------------------------------------------
// CTRL9_XL (0x18)
// ---------------------------------------------------------------------------

/// CTRL9_XL -- the I3C interface switch. `DEN_X`/`DEN_Y`/`DEN_Z` (bits 7:5) are
/// kept at their reset 1s; they only matter with the DEN pin, which the carrier
/// does not break out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ctrl9Xl {
    /// `I3C_disable` (bit 1). The part otherwise watches the bus for I3C framing;
    /// ST recommends turning that off on a pure I2C bus, so a glitch cannot be
    /// taken for an I3C command.
    pub(super) i3c_disabled: bool,
}

/// CTRL9_XL bits 7:5 at their reset values.
const CTRL9_XL_DEN_AXES_RESET: u8 = 0b1110_0000;

impl InertialRegisterValue for Ctrl9Xl {
    const ADDRESS: InertialRegister = InertialRegister::Ctrl9Xl;

    fn to_byte(self) -> u8 {
        CTRL9_XL_DEN_AXES_RESET | ((self.i3c_disabled as u8) << 1)
    }
}

// ---------------------------------------------------------------------------
// LIS3MDL: magnetometer, kept powered down
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MagnetometerRegister {
    WhoAmI = 0x0F,
    CtrlReg1 = 0x20,
    CtrlReg3 = 0x22,
}

impl MagnetometerRegister {
    pub(super) const fn address(self) -> u8 {
        self as u8
    }
}

/// WHO_AM_I's fixed value.
pub(super) const MAGNETOMETER_IDENTITY: u8 = 0x3D;

/// A value that can be written to exactly one LIS3MDL register.
pub(super) trait MagnetometerRegisterValue: Copy {
    const ADDRESS: MagnetometerRegister;
    fn to_byte(self) -> u8;
}

/// CTRL_REG3 `MD[1:0]` (bits 1:0). Only power-down is ever written; the band has
/// no use for a magnetometer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MagnetometerMode {
    PowerDown = 0b11,
}

/// CTRL_REG3 -- operating mode. `I2C_DISABLE` (bit 7) stays 0 even though nothing
/// talks to the chip again: it survives an ESP reboot, and the next boot's
/// identity check would then fail. `LP` and `SIM` stay at their reset 0s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CtrlReg3 {
    pub(super) mode: MagnetometerMode,
}

impl MagnetometerRegisterValue for CtrlReg3 {
    const ADDRESS: MagnetometerRegister = MagnetometerRegister::CtrlReg3;

    fn to_byte(self) -> u8 {
        self.mode as u8
    }
}

/// CTRL_REG1 -- temperature sensor, performance mode, data rate, self-test. With
/// the chip in power-down only the temperature sensor is worth naming; the rest
/// is written as its slowest, lowest-power 0s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CtrlReg1 {
    /// `TEMP_EN` (bit 7).
    pub(super) temperature_sensor_enabled: bool,
}

impl MagnetometerRegisterValue for CtrlReg1 {
    const ADDRESS: MagnetometerRegister = MagnetometerRegister::CtrlReg1;

    fn to_byte(self) -> u8 {
        (self.temperature_sensor_enabled as u8) << 7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCELEROMETER_RANGES: [AccelerometerRange; 4] = [
        AccelerometerRange::TwoG,
        AccelerometerRange::FourG,
        AccelerometerRange::EightG,
        AccelerometerRange::SixteenG,
    ];

    const GYROSCOPE_RANGES: [GyroscopeRange; 5] = [
        GyroscopeRange::OneHundredTwentyFiveDegreesPerSecond,
        GyroscopeRange::TwoHundredFiftyDegreesPerSecond,
        GyroscopeRange::FiveHundredDegreesPerSecond,
        GyroscopeRange::OneThousandDegreesPerSecond,
        GyroscopeRange::TwoThousandDegreesPerSecond,
    ];

    /// In code order, `0b0001` to `0b1000`.
    const OUTPUT_DATA_RATES: [OutputDataRate; 8] = [
        OutputDataRate::Hertz12Point5,
        OutputDataRate::Hertz26,
        OutputDataRate::Hertz52,
        OutputDataRate::Hertz104,
        OutputDataRate::Hertz208,
        OutputDataRate::Hertz416,
        OutputDataRate::Hertz833,
        OutputDataRate::Hertz1660,
    ];

    #[test]
    fn each_data_rate_code_doubles_the_rate() {
        for pair in OUTPUT_DATA_RATES.windows(2) {
            let (previous, next) = (pair[0], pair[1]);
            assert_eq!(next.bits(), previous.bits() + 1, "{previous:?} to {next:?}");
            // The datasheet's rates are rounded, so the doubling is only close.
            let ratio = next.hertz() / previous.hertz();
            assert!((1.95..2.1).contains(&ratio), "{previous:?} to {next:?}: {ratio}");
        }
    }

    #[test]
    fn the_shipped_configuration_writes_the_datasheet_bytes() {
        // 104 Hz (0100) and ±8 g (11): 0100_1100.
        let accelerometer = Ctrl1Xl {
            output_data_rate: OutputDataRate::Hertz104,
            range: AccelerometerRange::EightG,
        };
        assert_eq!(accelerometer.to_byte(), 0x4C);
        // 104 Hz (0100) and ±1000 °/s (FS_G 10, FS_125 0): 0100_1000.
        let gyroscope = Ctrl2G {
            output_data_rate: OutputDataRate::Hertz104,
            range: GyroscopeRange::OneThousandDegreesPerSecond,
        };
        assert_eq!(gyroscope.to_byte(), 0x48);
        let interface = Ctrl3C {
            block_data_update: true,
            address_increment: true,
            software_reset: false,
        };
        assert_eq!(interface.to_byte(), 0x44);
        assert_eq!(Ctrl9Xl { i3c_disabled: true }.to_byte(), 0xE2);
        let mode = CtrlReg3 {
            mode: MagnetometerMode::PowerDown,
        };
        assert_eq!(mode.to_byte(), 0x03);
        let temperature = CtrlReg1 {
            temperature_sensor_enabled: false,
        };
        assert_eq!(temperature.to_byte(), 0x00);
    }

    #[test]
    fn the_out_of_order_and_odd_codes_land_where_the_datasheet_puts_them() {
        let sixteen_g = Ctrl1Xl {
            output_data_rate: OutputDataRate::Hertz26,
            range: AccelerometerRange::SixteenG,
        };
        assert_eq!(sixteen_g.to_byte(), 0x24);
        let one_hundred_twenty_five = Ctrl2G {
            output_data_rate: OutputDataRate::Hertz12Point5,
            range: GyroscopeRange::OneHundredTwentyFiveDegreesPerSecond,
        };
        assert_eq!(one_hundred_twenty_five.to_byte(), 0x12);
        let two_thousand = Ctrl2G {
            output_data_rate: OutputDataRate::Hertz1660,
            range: GyroscopeRange::TwoThousandDegreesPerSecond,
        };
        assert_eq!(two_thousand.to_byte(), 0x8C);
    }

    #[test]
    fn sensitivity_times_full_count_is_about_full_scale() {
        for range in ACCELEROMETER_RANGES {
            let ratio = range.g_per_count() * i16::MAX as f32 / range.full_scale_g();
            assert!((ratio - 1.0).abs() < 0.01, "{range:?}: {ratio}");
        }
        for range in GYROSCOPE_RANGES {
            let ratio = range.degrees_per_second_per_count() * i16::MAX as f32
                / range.full_scale_degrees_per_second();
            // The gyroscope's sensitivities leave about 15% headroom past the
            // nominal range.
            assert!((1.0..1.2).contains(&ratio), "{range:?}: {ratio}");
        }
    }
}
