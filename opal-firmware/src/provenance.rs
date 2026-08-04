//! What this device is, for the dashboard to record alongside a session: the
//! build that is running and the register set the front end actually came up
//! with.
//!
//! The build identity is stamped in by `build.rs` at compile time. The register
//! snapshot is filled once, by [`crate::adc::bring_up`], into the fixed array
//! below — no heap, and nothing per frame. The wire form is built only when a
//! link asks for it, which is on connect and on a config change.

use crate::adc::{register_identity, DEVICE_COUNT, REGISTER_COUNT};
use protocol::{AnalogFrontEnd, DeviceProvenance, FirmwareBuild, RegisterReadback};
use std::sync::Mutex;

/// Every chip's registers as they read back after configuration. `None` for a
/// register whose read failed, and for every register of a chip that never came
/// up — an absent front end reports absent values rather than plausible ones.
static FRONT_END_REGISTERS: Mutex<[[Option<u8>; REGISTER_COUNT]; DEVICE_COUNT]> =
    Mutex::new([[None; REGISTER_COUNT]; DEVICE_COUNT]);

/// Publish one chip's post-configuration register readback, address order.
pub(crate) fn record_front_end(chip: usize, values: [Option<u8>; REGISTER_COUNT]) {
    if let Some(slot) = FRONT_END_REGISTERS.lock().unwrap().get_mut(chip) {
        *slot = values;
    }
}

/// The provenance a [`protocol::Frame::DeviceHello`] carries.
pub(crate) fn device() -> DeviceProvenance {
    let registers = *FRONT_END_REGISTERS.lock().unwrap();
    DeviceProvenance {
        firmware: FirmwareBuild {
            crate_version: env!("CARGO_PKG_VERSION").into(),
            git_commit: env!("FIRMWARE_GIT_COMMIT").into(),
            working_tree_modified: env!("FIRMWARE_WORKING_TREE_MODIFIED") == "true",
            built_at: env!("FIRMWARE_BUILT_AT").into(),
        },
        analog_front_ends: registers
            .iter()
            .enumerate()
            .map(|(chip, values)| AnalogFrontEnd {
                chip: chip as u8,
                registers: values
                    .iter()
                    .enumerate()
                    .map(|(index, &value)| {
                        let (name, address) = register_identity(index);
                        RegisterReadback {
                            name: name.into(),
                            address,
                            value,
                        }
                    })
                    .collect(),
            })
            .collect(),
    }
}
