//! The register map as types.
//!
//! Each register is a zero-sized type carrying its address and access rights, so the
//! datasheet's read-only columns are enforced by the compiler rather than by care:
//! `write_register(register::Status, ..)` is a type error, not a bug found at the bench.
//!
//! Addresses are from SLOS854D Table 3. Only the registers this driver touches are
//! declared; the audio-to-vibe, auto-calibration-result and LRA-specific registers are
//! deliberately absent, since a name here is a claim that the driver understands the
//! register.

/// A register that exists in the map.
pub trait Register {
    const ADDRESS: u8;
    /// The datasheet's name, used to say which register an I2C failure was on.
    const NAME: &'static str;
}

pub trait ReadableRegister: Register {}
pub trait WritableRegister: Register {}

macro_rules! register {
    ($(#[$documentation:meta])* $name:ident, $address:expr, $datasheet_name:literal, ReadOnly) => {
        $(#[$documentation])*
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl Register for $name {
            const ADDRESS: u8 = $address;
            const NAME: &'static str = $datasheet_name;
        }

        impl ReadableRegister for $name {}
    };
    ($(#[$documentation:meta])* $name:ident, $address:expr, $datasheet_name:literal, ReadWrite) => {
        $(#[$documentation])*
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl Register for $name {
            const ADDRESS: u8 = $address;
            const NAME: &'static str = $datasheet_name;
        }

        impl ReadableRegister for $name {}
        impl WritableRegister for $name {}
    };
}

register!(Status, 0x00, "STATUS", ReadOnly);
register!(Mode, 0x01, "MODE", ReadWrite);
register!(
    RealtimePlaybackInput,
    0x02,
    "REAL_TIME_PLAYBACK_INPUT",
    ReadWrite
);

register!(LibrarySelection, 0x03, "LIBRARY_SELECTION", ReadWrite);

register!(
    /// Base of the eight-slot waveform sequencer, 0x04 through 0x0B. The slots are only
    /// ever written as one block, which the chip's sequential addressing allows in a
    /// single transaction (SLOS854D section 8.5.3.2), so the seven registers above the
    /// base need no names of their own.
    WaveformSequencer,
    0x04,
    "WAVEFORM_SEQUENCER",
    ReadWrite
);

register!(Go, 0x0C, "GO", ReadWrite);
register!(RatedVoltage, 0x16, "RATED_VOLTAGE", ReadWrite);
register!(OverdriveClamp, 0x17, "OD_CLAMP", ReadWrite);
register!(FeedbackControl, 0x1A, "FEEDBACK_CONTROL", ReadWrite);
register!(Control3, 0x1D, "CONTROL3", ReadWrite);
