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

use crate::media::MediaKey;

/// A parsed console command.
#[derive(Clone, Copy, Debug)]
pub enum Command {
    /// Send a media key.
    Media(MediaKey),
    /// Print the help text.
    Help,
}

/// One-line usage shown at boot and on `h`/`?`.
pub const HELP: &str =
    "keys: p=play/pause  n=next  b=back  +=vol up  -=vol down  m=mute  h=help";

/// Install the USB-Serial-JTAG driver so [`read_byte`] receives console input.
/// Must be called once before reading.
pub fn init() -> Result<()> {
    let mut cfg = sys::usb_serial_jtag_driver_config_t {
        tx_buffer_size: 256,
        rx_buffer_size: 256,
    };
    sys::esp!(unsafe { sys::usb_serial_jtag_driver_install(&mut cfg) })?;
    Ok(())
}

/// Block up to `timeout_ms` for one console byte; `None` on timeout.
///
/// A FreeRTOS tick is 1 ms here (`CONFIG_FREERTOS_HZ=1000`), so `timeout_ms`
/// doubles as the tick count.
pub fn read_byte(timeout_ms: u32) -> Option<u8> {
    let mut byte = 0u8;
    let read = unsafe {
        sys::usb_serial_jtag_read_bytes(&mut byte as *mut u8 as *mut c_void, 1, timeout_ms)
    };
    (read > 0).then_some(byte)
}

/// Map a console byte to a command. `Ok(None)` for whitespace (ignored),
/// `Err(c)` for anything unrecognized so the caller can warn.
pub fn parse(byte: u8) -> Result<Option<Command>, char> {
    let c = byte as char;
    match c.to_ascii_lowercase() {
        'p' => Ok(Some(Command::Media(MediaKey::PlayPause))),
        'n' => Ok(Some(Command::Media(MediaKey::NextTrack))),
        'b' => Ok(Some(Command::Media(MediaKey::PrevTrack))),
        '+' | '=' => Ok(Some(Command::Media(MediaKey::VolumeUp))),
        '-' | '_' => Ok(Some(Command::Media(MediaKey::VolumeDown))),
        'm' => Ok(Some(Command::Media(MediaKey::Mute))),
        'h' | '?' => Ok(Some(Command::Help)),
        '\r' | '\n' | ' ' | '\t' => Ok(None),
        other => Err(other),
    }
}
