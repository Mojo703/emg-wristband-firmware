//! Logs for protocol delivery. The USB CDC channel carries framed CBOR, so there is
//! no text console anymore (`CONFIG_ESP_CONSOLE_NONE`); instead compact records wait
//! here and the serve loop turns them into `Frame::Log` values at the active link.
//! The 64 record descriptors live in an inline array that cannot allocate or grow;
//! only each record's formatted message owns heap storage. Full buffers drop oldest.

use log::{Level, LevelFilter, Log, Metadata, Record};
use protocol::{Frame, LogLevel};
use std::sync::Mutex;

/// Boot logs wait here for the first link; at 4 predictions/s of steady-state logging
/// this is minutes of headroom.
const BUFFER_CAP: usize = 64;

static LOGGER: FrameLogger = FrameLogger {
    buffer: Mutex::new(LogRing::new()),
};

type LogRing = FixedRing<LogRecord, BUFFER_CAP>;

struct FrameLogger {
    buffer: Mutex<LogRing>,
}

#[derive(Debug)]
pub(crate) struct LogRecord {
    t_us: u64,
    level: LogLevel,
    message: String,
}

impl LogRecord {
    pub(crate) fn into_frame(self) -> Frame {
        Frame::Log {
            t_us: self.t_us,
            level: self.level,
            message: self.message,
        }
    }

    pub(crate) fn from_frame(frame: Frame) -> Self {
        let Frame::Log {
            t_us,
            level,
            message,
        } = frame
        else {
            unreachable!("logger only constructs Frame::Log values")
        };
        Self {
            t_us,
            level,
            message,
        }
    }
}

/// A fixed-capacity FIFO whose descriptor storage is entirely inline. Moving a
/// `LogRecord` into or out of a slot never allocates queue storage.
struct FixedRing<T, const CAPACITY: usize> {
    slots: [Option<T>; CAPACITY],
    head: usize,
    len: usize,
}

impl<T, const CAPACITY: usize> FixedRing<T, CAPACITY> {
    const fn new() -> Self {
        Self {
            slots: [const { None }; CAPACITY],
            head: 0,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push_back(&mut self, record: T) {
        if self.len == CAPACITY {
            self.slots[self.head] = Some(record);
            self.head = (self.head + 1) % CAPACITY;
            return;
        }
        let tail = (self.head + self.len) % CAPACITY;
        self.slots[tail] = Some(record);
        self.len += 1;
    }

    fn push_front(&mut self, record: T) {
        if self.len == CAPACITY {
            // The restored record is oldest, so drop-oldest immediately evicts it.
            return;
        }
        self.head = (self.head + CAPACITY - 1) % CAPACITY;
        self.slots[self.head] = Some(record);
        self.len += 1;
    }

    fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let record = self.slots[self.head].take();
        self.head = (self.head + 1) % CAPACITY;
        self.len -= 1;
        record
    }
}

impl Log for FrameLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        #[cfg(feature = "diagnostic-console")]
        println!("[{}] {}", record.level(), record.args());
        let level = match record.level() {
            Level::Error => LogLevel::Error,
            Level::Warn => LogLevel::Warn,
            Level::Info => LogLevel::Info,
            Level::Debug | Level::Trace => LogLevel::Debug,
        };
        let record = LogRecord {
            t_us: unsafe { esp_idf_svc::sys::esp_timer_get_time() } as u64,
            level,
            message: format!("{}", record.args()),
        };
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push_back(record);
    }

    fn flush(&self) {}
}

/// Install as the global logger. Replaces `EspLogger` (whose output would go to the
/// now-disabled console).
pub fn init() {
    log::set_logger(&LOGGER).expect("logger installed twice");
    log::set_max_level(LevelFilter::Info);
}

/// Number of records currently pending. A sender snapshots this before its loop so
/// logs produced during transport I/O remain queued for the next send interval.
pub fn len() -> usize {
    LOGGER.buffer.lock().unwrap().len()
}

/// Pop one compact record, oldest first. The lock is released before return, so it
/// can never span frame construction, encoding, or transport I/O.
pub fn pop() -> Option<LogRecord> {
    LOGGER.buffer.lock().unwrap().pop_front()
}

/// Restore a record rejected by a dead link ahead of records logged while it was in
/// flight. The fixed 64-record cap still applies with the same drop-oldest policy.
pub fn restore(record: LogRecord) {
    LOGGER.buffer.lock().unwrap().push_front(record);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::type_name;
    use core::mem::size_of;

    fn record(sequence: u64) -> LogRecord {
        LogRecord {
            t_us: sequence,
            level: LogLevel::Info,
            message: sequence.to_string(),
        }
    }

    fn sequences(ring: &mut LogRing) -> Vec<u64> {
        let mut values = Vec::new();
        while let Some(record) = ring.pop_front() {
            values.push(record.t_us);
        }
        values
    }

    #[test]
    fn sixty_four_records_retain_order_and_sixty_fifth_drops_oldest() {
        let mut ring = LogRing::new();
        for sequence in 0..BUFFER_CAP as u64 {
            ring.push_back(record(sequence));
        }
        assert_eq!(
            sequences(&mut ring),
            (0..BUFFER_CAP as u64).collect::<Vec<_>>()
        );

        for sequence in 0..=BUFFER_CAP as u64 {
            ring.push_back(record(sequence));
        }
        assert_eq!(
            sequences(&mut ring),
            (1..=BUFFER_CAP as u64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn failed_record_is_restored_ahead_of_logs_added_during_send() {
        let mut ring = LogRing::new();
        for sequence in 0..3 {
            ring.push_back(record(sequence));
        }

        let delivered = ring.pop_front().unwrap();
        let failed = ring.pop_front().unwrap();
        ring.push_back(record(3));
        ring.push_front(failed);

        assert_eq!(delivered.t_us, 0);
        assert_eq!(sequences(&mut ring), vec![1, 2, 3]);
    }

    #[test]
    fn queue_operations_do_not_allocate_storage_after_initialization() {
        let mut ring = LogRing::new();
        let records: [LogRecord; BUFFER_CAP] =
            std::array::from_fn(|sequence| record(sequence as u64));
        let mut interval = crate::allocation::begin_interval();

        for record in records {
            ring.push_back(record);
        }
        for _ in 0..BUFFER_CAP {
            let record = ring.pop_front().unwrap();
            ring.push_back(record);
        }

        let allocation = crate::allocation::take_interval(&mut interval);
        assert_eq!(allocation.requests, 0);
    }

    #[test]
    fn repeated_fill_drain_restore_never_allocates_or_requests_large_storage() {
        let mut ring = LogRing::new();
        for sequence in 0..BUFFER_CAP as u64 {
            ring.push_back(record(sequence));
        }
        let mut interval = crate::allocation::begin_interval();

        for _ in 0..10_000 {
            let failed = ring.pop_front().unwrap();
            ring.push_front(failed);
            let delivered = ring.pop_front().unwrap();
            ring.push_back(delivered);
        }

        let allocation = crate::allocation::take_interval(&mut interval);
        assert_eq!(allocation.requests, 0);
        assert!(allocation.maximum_requested_bytes < 12_288);
    }

    #[test]
    fn stored_record_and_ring_are_compact_and_frame_free() {
        assert!(size_of::<LogRecord>() <= 40);
        assert!(size_of::<LogRing>() <= BUFFER_CAP * 40 + 32);
        assert!(size_of::<LogRecord>() < size_of::<Frame>());
        assert!(type_name::<LogRing>().contains("LogRecord"));
        assert!(!type_name::<LogRing>().contains("Frame"));
    }
}
