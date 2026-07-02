//! Logs as protocol frames. The USB CDC channel carries framed CBOR, so there is no
//! text console anymore (`CONFIG_ESP_CONSOLE_NONE`); instead `log` records buffer
//! here and the serve loop drains them to the dashboard as `Frame::Log` over whatever
//! link is active. The buffer is bounded and drops oldest, so early-boot logs survive
//! until the first link comes up without ever growing unbounded.

use log::{Level, LevelFilter, Log, Metadata, Record};
use protocol::{Frame, LogLevel};
use std::collections::VecDeque;
use std::sync::Mutex;

/// Boot logs wait here for the first link; at 4 predictions/s of steady-state logging
/// this is minutes of headroom.
const BUFFER_CAP: usize = 64;

static BUFFER: Mutex<VecDeque<Frame>> = Mutex::new(VecDeque::new());
static LOGGER: FrameLogger = FrameLogger;

struct FrameLogger;

impl Log for FrameLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let level = match record.level() {
            Level::Error => LogLevel::Error,
            Level::Warn => LogLevel::Warn,
            Level::Info => LogLevel::Info,
            Level::Debug | Level::Trace => LogLevel::Debug,
        };
        let frame = Frame::Log {
            t_us: unsafe { esp_idf_svc::sys::esp_timer_get_time() } as u64,
            level,
            message: format!("{}", record.args()),
        };
        let mut buffer = BUFFER.lock().unwrap();
        if buffer.len() >= BUFFER_CAP {
            buffer.pop_front();
        }
        buffer.push_back(frame);
    }

    fn flush(&self) {}
}

/// Install as the global logger. Replaces `EspLogger` (whose output would go to the
/// now-disabled console).
pub fn init() {
    log::set_logger(&LOGGER).expect("logger installed twice");
    log::set_max_level(LevelFilter::Info);
}

/// Take everything logged since the last drain, oldest first.
pub fn drain() -> Vec<Frame> {
    BUFFER.lock().unwrap().drain(..).collect()
}
