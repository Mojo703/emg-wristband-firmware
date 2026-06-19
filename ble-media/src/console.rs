//! Serial-console input over USB-Serial-JTAG.
//!
//! The ESP32-S3-Zero's console is USB-Serial-JTAG (the USB-C port). esp-idf's
//! default `stdin` over that link does not deliver bytes, so we install the
//! USB-Serial-JTAG driver and read from it directly.
//!
//! For bring-up the media keys are driven by single characters typed into the
//! serial monitor. Later the gesture classifier produces [`MediaKey`]s directly
//! and this module can be dropped or kept as a debug input.

use core::ffi::c_void;

use anyhow::Result;
use esp_idf_svc::sys;

/// One-line usage shown at boot and on `h`/`?`.
pub const HELP: &str = "keys: p=play/pause  n=next  b=back  +=vol up  -=vol down  m=mute  h=help";

/// Install the USB-Serial-JTAG driver so [`read_byte`] receives console input.
/// Must be called once before reading.
pub fn init() -> Result<()> {
    let mut cfg = sys::usb_serial_jtag_driver_config_t {
        tx_buffer_size: 256,
        rx_buffer_size: 256,
    };
    sys::esp!(unsafe { sys::usb_serial_jtag_driver_install(&mut cfg) })?;
    // Route stdio/ESP_LOG through the driver as well. Without this, logs still
    // use the polling path and contend with the driver for the peripheral, so
    // output from other tasks (e.g. the NimBLE host task's connect/disconnect
    // handlers) gets dropped.
    unsafe { sys::esp_vfs_usb_serial_jtag_use_driver() };
    Ok(())
}

/// Block up to `timeout_ms` for one console byte; `None` on timeout.
///
/// A FreeRTOS tick is 1 ms here (`CONFIG_FREERTOS_HZ=1000`), so `timeout_ms`
/// doubles as the tick count.
pub(crate) fn read_byte_blocking(timeout_ms: u32) -> Option<u8> {
    let mut byte = 0u8;
    let read = unsafe {
        sys::usb_serial_jtag_read_bytes(&mut byte as *mut u8 as *mut c_void, 1, timeout_ms)
    };
    (read > 0).then_some(byte)
}
