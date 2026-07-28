//! ADS1298 SPI command opcodes.
//!
//! Most commands are standalone one-byte commands. RREG and WREG are base
//! opcodes that must be OR'd with a register address.

pub(crate) const WAKEUP: u8 = 0x02;

pub(crate) const STANDBY: u8 = 0x04;

pub(crate) const RESET: u8 = 0x06;

/// Reserved for future on-demand conversion control
pub(crate) const START: u8 = 0x08;

/// Reserved for future on-demand conversion control
pub(crate) const STOP: u8 = 0x0A;

pub(crate) const RDATAC: u8 = 0x10;
pub(crate) const SDATAC: u8 = 0x11;

/// Reserved for future on-demand conversion control
pub(crate) const RDATA: u8 = 0x12;

/// Base opcode for register reads. Must be combined with `Register::addr()` using bitwise OR.
pub(crate) const RREG_BASE: u8 = 0x20;

/// Base opcode for register writes. Must be combined with `Register::addr()` using bitwise OR.
pub(crate) const WREG_BASE: u8 = 0x40;