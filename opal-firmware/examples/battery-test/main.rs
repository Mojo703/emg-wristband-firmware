//! Battery-life test for the XIAO ESP32-S3 motherboard.
//!
//! Draws the band's working current until the cell is flat: both ADS1298s
//! stream at 2000 SPS on core 1, the CPU runs at the product's 240 MHz, and
//! every five seconds a random volume key goes to the paired phone over BLE
//! HID. There is no inference, because this board carries no calibration.
//!
//! Every minute the run's uptime, event counts and recent errors are written
//! to NVS, and every boot prints the stored runs. After the battery dies, plug
//! the board in, open the monitor and press RESET to read how long it lasted.
//!
//! Pin map, from the Motherboard PCB schematic (XIAO pad, GPIO):
//!
//! | Net    | Pad  | GPIO |
//! |--------|------|------|
//! | DRDY2  | D0   | 1    |
//! | MISO   | D1   | 2    |
//! | CS1    | D2   | 3    |
//! | SCLK   | D6   | 43   |
//! | START  | D7   | 44   |
//! | CS2    | D8   | 7    |
//! | RESET  | D9   | 8    |
//! | MOSI   | D10  | 9    |
//! | DRDY1  | MTDO | 40   |
//! | PWDN   | MTMS | 42   |

mod adc;
mod record;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ble_media::nimble::NimbleRadio;
use ble_media::phone::{Delivery, Phone, PhoneState, RadioClaim};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::{self, esp};
use log::{info, warn};
use protocol::MediaKey;

use record::{Clock, RunLog, RunRecord};

const DEVICE_NAME: &str = "EMG Battery Test";
const EVENT_INTERVAL: Duration = Duration::from_secs(5);
const SAVE_INTERVAL: Duration = Duration::from_secs(60);
/// The product's BLE tick, so a key's hold behaves the same.
const TICK_MILLISECONDS: u32 = 5;
const EVENT_KEYS: [MediaKey; 2] = [MediaKey::VolumeUp, MediaKey::VolumeDown];

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let reset_reason = unsafe { sys::esp_reset_reason() };
    let mut run_log = RunLog::open(EspDefaultNvsPartition::take()?)?;
    run_log.print_stored();
    info!(
        "=== battery test, boot {}, started by {} ===",
        run_log.boot(),
        record::reset_reason_name(reset_reason)
    );

    let mut phone = match NimbleRadio::bring_up(DEVICE_NAME) {
        Ok(radio) => Phone::new(radio),
        Err(error) => {
            record::error(format!("BLE stack refused at boot: {error:#}"));
            Phone::unavailable(format!("{error:#}"))
        }
    };
    phone.set_enabled(true, RadioClaim::Free);

    let peripherals = Peripherals::take()?;
    let pins = peripherals.pins;
    // Pads 39-42 come out of reset owned by JTAG; gpio_config hands DRDY1 (40)
    // and PWDN (42) to the GPIO matrix.
    for (pin, mode) in [
        (40, sys::gpio_mode_t_GPIO_MODE_INPUT),
        (42, sys::gpio_mode_t_GPIO_MODE_OUTPUT),
    ] {
        let config = sys::gpio_config_t {
            pin_bit_mask: 1u64 << pin,
            mode,
            pull_up_en: sys::gpio_pullup_t_GPIO_PULLUP_DISABLE,
            pull_down_en: sys::gpio_pulldown_t_GPIO_PULLDOWN_DISABLE,
            intr_type: sys::gpio_int_type_t_GPIO_INTR_DISABLE,
        };
        esp!(unsafe { sys::gpio_config(&config) })?;
    }
    adc::start(
        peripherals.spi2,
        adc::Wiring {
            sclk: pins.gpio43.into(),
            data_in: pins.gpio9.into(),
            data_out: pins.gpio2.into(),
            power_down: pins.gpio42.into(),
            reset: pins.gpio8.into(),
            start: pins.gpio44.into(),
            chip_select: [pins.gpio3.into(), pins.gpio7.into()],
            data_ready: [pins.gpio40.into(), pins.gpio1.into()],
        },
    )?;

    let watchdog = sys::esp_task_wdt_config_t {
        timeout_ms: 5000,
        idle_core_mask: 0,
        trigger_panic: true,
    };
    if unsafe { sys::esp_task_wdt_reconfigure(&watchdog) } != sys::ESP_OK {
        warn!("task watchdog reconfigure failed");
    }
    unsafe { sys::esp_task_wdt_add(std::ptr::null_mut()) };

    let mut run = RunRecord {
        boot: run_log.boot(),
        reset_reason,
        ..Default::default()
    };
    let mut previous_state = phone.state();
    let mut paired_time = Duration::ZERO;
    let started = Instant::now();
    let mut last_tick = started;
    let mut next_event = started + EVENT_INTERVAL;
    let mut next_save = started + SAVE_INTERVAL;

    loop {
        FreeRtos::delay_ms(TICK_MILLISECONDS);
        unsafe { sys::esp_task_wdt_reset() };
        let now = Instant::now();
        let elapsed = now - last_tick;
        last_tick = now;
        phone.tick(elapsed);

        let state = phone.state();
        if state == PhoneState::Paired {
            paired_time += elapsed;
        }
        if state != previous_state {
            if state == PhoneState::Paired {
                run.connections += 1;
                info!("phone paired");
            } else if previous_state == PhoneState::Paired {
                run.disconnects += 1;
                record::error(format!("phone link lost, now {state:?}"));
            } else {
                info!("phone: {state:?}");
            }
            if let Some(reason) = phone.reason() {
                warn!("phone: {reason}");
            }
            previous_state = state;
        }

        if now >= next_event {
            next_event += EVENT_INTERVAL;
            let key = EVENT_KEYS[unsafe { sys::esp_random() } as usize % EVENT_KEYS.len()];
            match phone.press(key) {
                Delivery::Sent => {
                    run.events_sent += 1;
                    info!("event {}: sent {key:?}", run.events_sent);
                }
                Delivery::Dropped(state) => {
                    run.events_dropped += 1;
                    info!("no paired phone ({state:?}); dropped {key:?}");
                }
                Delivery::Failed(reason) => {
                    run.events_failed += 1;
                    record::error(format!("{key:?} failed: {reason}"));
                }
            }
        }

        if now >= next_save {
            next_save += SAVE_INTERVAL;
            run.uptime_seconds = record::uptime_seconds();
            run.paired_seconds = paired_time.as_secs() as u32;
            let counters = &adc::COUNTERS;
            run.adc_frames = counters.frames.load(Ordering::Relaxed);
            run.adc_missed_frames = counters.missed_frames.load(Ordering::Relaxed);
            run.adc_bad_frames = counters.bad_frames.load(Ordering::Relaxed);
            run.adc_stalls = counters.stalls.load(Ordering::Relaxed);
            run.adc_recoveries = counters.recoveries.load(Ordering::Relaxed);
            run.adc_bring_up_failures = counters.bring_up_failures.load(Ordering::Relaxed);
            record::fill_errors(&mut run);
            run_log.save(&run);
            info!(
                "uptime {}, {state:?}, heap {} free\n{run}",
                Clock(run.uptime_seconds),
                unsafe { sys::esp_get_free_heap_size() },
            );
        }
    }
}
