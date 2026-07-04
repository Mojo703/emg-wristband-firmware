//! ADS1298 register map
//!
//! Only a subset is being used by the current bring-up sequence.
//! The rest is kept as a reference for when lead-off detection, gain, 
//! and reference config get implemented.

#[allow(dead_code)]
#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum Register {
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
    pub const fn addr(self) -> u8 {
        self as u8
    }
}