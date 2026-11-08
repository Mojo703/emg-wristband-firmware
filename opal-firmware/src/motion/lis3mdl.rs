//! The LIS3MDL magnetometer, which the band does not use.
//!
//! It shares the carrier's supply and sits a few centimetres from the electrodes,
//! and while it measures it fires set/reset current pulses through its sensing
//! straps every conversion. The one job here is to make sure the chip is in
//! power-down and to prove it by reading the mode back. After [`power_down`]
//! nothing addresses it again.

use anyhow::{bail, Result};

use super::registers::{
    CtrlReg1, CtrlReg3, MagnetometerMode, MagnetometerRegister, MagnetometerRegisterValue,
    MAGNETOMETER_IDENTITY,
};
use super::ImuBus;

/// Confirms the chip is there, puts it in power-down, turns its temperature sensor
/// off, and verifies both by reading back. The mode goes first, so conversions
/// have already stopped when the rest is written. Power-down is also the reset
/// mode; it is written anyway, so a chip left running by an earlier image stops.
pub(super) fn power_down(bus: &mut ImuBus, address: u8) -> Result<()> {
    let identity = bus.read_byte(address, MagnetometerRegister::WhoAmI.address())?;
    if identity != MAGNETOMETER_IDENTITY {
        bail!(
            "LIS3MDL at {address:#04x}: WHO_AM_I reads {identity:#04x}, expected \
             {MAGNETOMETER_IDENTITY:#04x} (wiring, power, or the SA0 strap)"
        );
    }
    write_verified(
        bus,
        address,
        CtrlReg3 {
            mode: MagnetometerMode::PowerDown,
        },
    )?;
    write_verified(
        bus,
        address,
        CtrlReg1 {
            temperature_sensor_enabled: false,
        },
    )
}

fn write_verified<V: MagnetometerRegisterValue>(
    bus: &mut ImuBus,
    address: u8,
    value: V,
) -> Result<()> {
    let byte = value.to_byte();
    bus.write_byte(address, V::ADDRESS.address(), byte)?;
    let read = bus.read_byte(address, V::ADDRESS.address())?;
    if read != byte {
        bail!("LIS3MDL {:?} wrote {byte:#04x}, reads {read:#04x}", V::ADDRESS);
    }
    Ok(())
}
