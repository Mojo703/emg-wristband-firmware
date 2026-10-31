//! The run log: what each boot did, kept in NVS so it outlives a flat battery.
//!
//! A run claims one of [`SLOTS`] records on its first save, a minute in, and
//! rewrites it every minute after. Boots that die sooner, such as brownout
//! loops on an empty cell, never claim a slot, so they cannot push the battery
//! run out. Every boot prints all the stored runs first.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

use anyhow::Result;
use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};
use esp_idf_svc::sys;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};

const NAMESPACE: &str = "battery";
const BOOT_COUNT_KEY: &str = "boots";
const NEXT_SLOT_KEY: &str = "next";
const SLOTS: u32 = 6;
const RECENT_ERRORS: usize = 10;
const ERROR_MESSAGE_CHARACTERS: usize = 80;
const RECORD_BYTES_MAX: usize = 2048;

#[derive(Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    pub uptime_seconds: u32,
    pub message: String,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct RunRecord {
    pub boot: u32,
    pub reset_reason: u32,
    pub uptime_seconds: u32,
    pub paired_seconds: u32,
    pub events_sent: u32,
    pub events_dropped: u32,
    pub events_failed: u32,
    pub connections: u32,
    pub disconnects: u32,
    pub adc_frames: u32,
    pub adc_missed_frames: u32,
    pub adc_bad_frames: u32,
    pub adc_stalls: u32,
    pub adc_recoveries: u32,
    pub adc_bring_up_failures: u32,
    pub errors_total: u32,
    pub recent_errors: Vec<ErrorEntry>,
}

impl fmt::Display for RunRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "boot {}: ran {} (paired {}), started by {}",
            self.boot,
            Clock(self.uptime_seconds),
            Clock(self.paired_seconds),
            reset_reason_name(self.reset_reason),
        )?;
        writeln!(
            formatter,
            "  events: {} sent, {} dropped with no phone, {} failed; {} connections, {} disconnects",
            self.events_sent,
            self.events_dropped,
            self.events_failed,
            self.connections,
            self.disconnects,
        )?;
        writeln!(
            formatter,
            "  ADC: {} frames, {} missed, {} bad, {} stalls, {} recoveries, {} failed bring-ups",
            self.adc_frames,
            self.adc_missed_frames,
            self.adc_bad_frames,
            self.adc_stalls,
            self.adc_recoveries,
            self.adc_bring_up_failures,
        )?;
        write!(formatter, "  errors: {} in total", self.errors_total)?;
        for entry in &self.recent_errors {
            write!(
                formatter,
                "\n    at {}: {}",
                Clock(entry.uptime_seconds),
                entry.message
            )?;
        }
        Ok(())
    }
}

/// Seconds as `h:mm:ss`.
pub struct Clock(pub u32);

impl fmt::Display for Clock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let seconds = self.0;
        write!(
            formatter,
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    }
}

pub fn uptime_seconds() -> u32 {
    (unsafe { sys::esp_timer_get_time() } / 1_000_000) as u32
}

struct ErrorRing {
    total: u32,
    recent: VecDeque<ErrorEntry>,
}

static ERRORS: Mutex<ErrorRing> = Mutex::new(ErrorRing {
    total: 0,
    recent: VecDeque::new(),
});

/// Log an error and keep it for the run record. Callable from any thread.
pub fn error(message: String) {
    error!("{message}");
    let entry = ErrorEntry {
        uptime_seconds: uptime_seconds(),
        message: message.chars().take(ERROR_MESSAGE_CHARACTERS).collect(),
    };
    let mut errors = ERRORS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    errors.total += 1;
    if errors.recent.len() == RECENT_ERRORS {
        errors.recent.pop_front();
    }
    errors.recent.push_back(entry);
}

/// Copy the error ring into `record`.
pub fn fill_errors(record: &mut RunRecord) {
    let errors = ERRORS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    record.errors_total = errors.total;
    record.recent_errors.clear();
    record.recent_errors.extend(errors.recent.iter().cloned());
}

pub struct RunLog {
    nvs: EspDefaultNvs,
    boot: u32,
    slot: Option<u32>,
    encoded: Vec<u8>,
}

impl RunLog {
    pub fn open(partition: EspDefaultNvsPartition) -> Result<Self> {
        let nvs = EspDefaultNvs::new(partition, NAMESPACE, true)?;
        let boot = nvs.get_u32(BOOT_COUNT_KEY)?.unwrap_or(0) + 1;
        nvs.set_u32(BOOT_COUNT_KEY, boot)?;
        Ok(Self {
            nvs,
            boot,
            slot: None,
            encoded: Vec::with_capacity(RECORD_BYTES_MAX),
        })
    }

    pub fn boot(&self) -> u32 {
        self.boot
    }

    /// Print every stored run, oldest first, then the longest.
    pub fn print_stored(&self) {
        let mut runs: Vec<RunRecord> = (0..SLOTS).filter_map(|slot| self.load(slot)).collect();
        if runs.is_empty() {
            info!("no stored runs");
            return;
        }
        runs.sort_by_key(|run| run.boot);
        info!("===== {} stored runs =====", runs.len());
        for run in &runs {
            info!("{run}");
        }
        if let Some(longest) = runs.iter().max_by_key(|run| run.uptime_seconds) {
            info!(
                "===== longest: boot {}, {} =====",
                longest.boot,
                Clock(longest.uptime_seconds)
            );
        }
    }

    fn load(&self, slot: u32) -> Option<RunRecord> {
        let mut buffer = [0u8; RECORD_BYTES_MAX];
        match self.nvs.get_blob(&slot_key(slot), &mut buffer) {
            Ok(Some(bytes)) => ciborium::from_reader(bytes).ok(),
            Ok(None) => None,
            Err(error) => {
                warn!("run slot {slot} unreadable: {error}");
                None
            }
        }
    }

    pub fn save(&mut self, record: &RunRecord) {
        if let Err(error) = self.try_save(record) {
            self::error(format!("run log save failed: {error:#}"));
        }
    }

    fn try_save(&mut self, record: &RunRecord) -> Result<()> {
        let slot = match self.slot {
            Some(slot) => slot,
            None => {
                let next = self.nvs.get_u32(NEXT_SLOT_KEY)?.unwrap_or(0);
                self.nvs.set_u32(NEXT_SLOT_KEY, next.wrapping_add(1))?;
                let slot = next % SLOTS;
                info!("run log: boot {} records into slot {slot}", self.boot);
                self.slot = Some(slot);
                slot
            }
        };
        self.encoded.clear();
        ciborium::into_writer(record, &mut self.encoded)?;
        self.nvs.set_blob(&slot_key(slot), &self.encoded)?;
        Ok(())
    }
}

fn slot_key(slot: u32) -> String {
    format!("run{slot}")
}

pub fn reset_reason_name(reason: u32) -> &'static str {
    match reason {
        sys::esp_reset_reason_t_ESP_RST_POWERON => "power-on",
        sys::esp_reset_reason_t_ESP_RST_EXT => "external pin",
        sys::esp_reset_reason_t_ESP_RST_SW => "software restart",
        sys::esp_reset_reason_t_ESP_RST_PANIC => "panic",
        sys::esp_reset_reason_t_ESP_RST_INT_WDT => "interrupt watchdog",
        sys::esp_reset_reason_t_ESP_RST_TASK_WDT => "task watchdog",
        sys::esp_reset_reason_t_ESP_RST_WDT => "other watchdog",
        sys::esp_reset_reason_t_ESP_RST_DEEPSLEEP => "deep-sleep wake",
        sys::esp_reset_reason_t_ESP_RST_BROWNOUT => "brownout",
        sys::esp_reset_reason_t_ESP_RST_SDIO => "SDIO",
        sys::esp_reset_reason_t_ESP_RST_USB => "USB",
        sys::esp_reset_reason_t_ESP_RST_JTAG => "JTAG",
        _ => "unknown",
    }
}
