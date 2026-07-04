//! ADS1298 SPI command opcodes.
//!
//! Most commands are standalone one-byte commands. RREG and WREG are base
//! opcodes that must be OR'd with a register address.

#[allow(dead_code)]
pub const WAKEUP: u8 = 0x02;

#[allow(dead_code)]
pub const STANDBY: u8 = 0x04;

pub const RESET: u8 = 0x06;

#[allow(dead_code)]
pub const START: u8 = 0x08;

#[allow(dead_code)]
pub const STOP: u8 = 0x0A;

pub const RDATAC: u8 = 0x10;
pub const SDATAC: u8 = 0x11;

#[allow(dead_code)]
pub const RDATA: u8 = 0x12;

/// Base opcode for register reads. Must be combined with `Register::addr()` using bitwise OR.
pub const RREG_BASE: u8 = 0x20;

/// Base opcode for register writes. Must be combined with `Register::addr()` using bitwise OR.
pub const WREG_BASE: u8 = 0x40;