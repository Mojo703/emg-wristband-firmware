//! Deep-sleep bench for a bare ESP32-S3-Zero: holding the BOOT button puts the chip to
//! sleep, any press wakes it, and the onboard LED shows how it came up.
//!
//! | LED   | Meaning                                              |
//! |-------|------------------------------------------------------|
//! | blue  | awake after power-on or reset                        |
//! | green | awake after the button woke it from deep sleep       |
//! | red   | awake after deep sleep, woken by something else      |
//! | white | button held 4 s; release to sleep                    |
//! | dark  | asleep                                               |
//!
//! A counter in RTC slow memory, which stays powered in deep sleep, counts boots for as
//! long as that memory keeps its contents. After boot the LED blinks that count in the
//! wake colour, then holds the colour.
//!
//! Deep sleep drops the USB-Serial-JTAG link, so the LED is the result. To reflash a
//! sleeping board, hold BOOT and tap RESET.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{PinDriver, Pull};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::sys::{self, esp};
use log::info;

#[allow(unused_imports)]
#[path = "../src/feedback/led.rs"]
mod led;

use led::{Color, IndicatorLed};

#[link_section = ".rtc.data"]
static BOOT_COUNT: AtomicU32 = AtomicU32::new(0);

const BUTTON: sys::gpio_num_t = sys::gpio_num_t_GPIO_NUM_0;
const INDICATOR: sys::gpio_num_t = sys::gpio_num_t_GPIO_NUM_21;

const POLL_MILLISECONDS: u32 = 10;
const SLEEP_HOLD: Duration = Duration::from_secs(4);
const BLINK: Duration = Duration::from_millis(200);

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    // SAFETY: plain register writes on a pin nothing else owns yet.
    esp!(unsafe { sys::gpio_hold_dis(INDICATOR) })?;

    let peripherals = Peripherals::take()?;
    let button = PinDriver::input(peripherals.pins.gpio0, Pull::Up)?;
    let mut led = IndicatorLed::bring_up(peripherals.pins.gpio21)?;

    // SAFETY: reads a value the sleep module latched at boot.
    let cause = unsafe { sys::esp_sleep_get_wakeup_cause() };
    let awake = match cause {
        sys::esp_sleep_source_t_ESP_SLEEP_WAKEUP_UNDEFINED => Color::BLUE,
        sys::esp_sleep_source_t_ESP_SLEEP_WAKEUP_EXT0 => Color::GREEN,
        _ => Color::RED,
    };
    let boots = BOOT_COUNT.load(Ordering::Relaxed) + 1;
    BOOT_COUNT.store(boots, Ordering::Relaxed);
    info!("awake, wake cause {cause}, boot {boots}");
    for _ in 0..boots {
        show_for(&mut led, awake, BLINK)?;
        show_for(&mut led, Color::OFF, BLINK)?;
    }

    // The press that woke the chip may still be held; only a fresh press counts.
    show_until(&mut led, awake, || button.is_high())?;
    let mut held_since = None;
    let mut was_low = false;
    show_until(&mut led, awake, || {
        let low = button.is_low();
        if low != was_low {
            info!("button {}", if low { "down" } else { "up" });
            was_low = low;
        }
        if !low {
            held_since = None;
            return false;
        }
        held_since.get_or_insert_with(Instant::now).elapsed() >= SLEEP_HOLD
    })?;

    info!("held {SLEEP_HOLD:?}; boot {boots}; sleeping on release");
    // Wake is level-triggered on low, so sleeping with the button down would wake at once.
    show_until(&mut led, Color::WHITE, || button.is_high())?;
    led.show(Color::OFF, 0)?;
    drop(led);
    drop(button);

    // SAFETY: both pins' drivers are dropped above, and these calls only configure
    // pad and sleep registers before the chip powers down.
    unsafe {
        // Held low so a floating data line cannot clock garbage into the LED.
        esp!(sys::gpio_reset_pin(INDICATOR))?;
        esp!(sys::gpio_set_direction(
            INDICATOR,
            sys::gpio_mode_t_GPIO_MODE_OUTPUT
        ))?;
        esp!(sys::gpio_set_level(INDICATOR, 0))?;
        esp!(sys::gpio_hold_en(INDICATOR))?;

        esp!(sys::rtc_gpio_pullup_en(BUTTON))?;
        esp!(sys::rtc_gpio_pulldown_dis(BUTTON))?;
        esp!(sys::esp_sleep_enable_ext0_wakeup(BUTTON, 0))?;
        sys::esp_deep_sleep_start()
    }
}

fn show_until(
    led: &mut IndicatorLed,
    color: Color,
    mut done: impl FnMut() -> bool,
) -> anyhow::Result<()> {
    while !done() {
        led.show(color, u8::MAX)?;
        FreeRtos::delay_ms(POLL_MILLISECONDS);
    }
    Ok(())
}

fn show_for(led: &mut IndicatorLed, color: Color, duration: Duration) -> anyhow::Result<()> {
    let start = Instant::now();
    show_until(led, color, || start.elapsed() >= duration)
}
