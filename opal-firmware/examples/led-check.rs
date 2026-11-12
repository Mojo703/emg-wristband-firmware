//! Lights the band's WS2812B-2020 on GPIO5 in solid red, green, blue and white,
//! two seconds each, and logs each colour as it goes out. Nothing else is
//! brought up. If the log runs and the LED stays dark, the fault is the wiring
//! or the part; if red and green come out swapped, so is the byte order.

use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::sys;
use log::info;

#[allow(unused_imports, dead_code)]
#[path = "../src/feedback/led.rs"]
mod led;

use led::{Color, IndicatorLed};

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    // SAFETY: register writes on pads nothing has claimed yet.
    unsafe {
        sys::gpio_deep_sleep_hold_dis();
        sys::gpio_force_unhold_all();
    }

    // SAFETY: plain pad configuration on a pin nothing else owns yet.
    unsafe {
        let pad = sys::gpio_num_t_GPIO_NUM_5;
        info!("GPIO5 hold before check: {}", sys::gpio_hold_dis(pad));
        sys::esp!(sys::gpio_reset_pin(pad))?;
        sys::esp!(sys::gpio_set_direction(
            pad,
            sys::gpio_mode_t_GPIO_MODE_INPUT_OUTPUT
        ))?;
        for level in [1, 0, 1, 0] {
            sys::esp!(sys::gpio_set_level(pad, level))?;
            FreeRtos::delay_ms(5);
            info!(
                "GPIO5 driven {level}, pad reads {}",
                sys::gpio_get_level(pad)
            );
        }
        sys::esp!(sys::gpio_reset_pin(pad))?;
    }

    let peripherals = Peripherals::take()?;
    let mut led = IndicatorLed::bring_up(peripherals.pins.gpio5)?;
    info!("=== LED check on GPIO5 ===");
    loop {
        for (name, color) in [
            ("red", Color::RED),
            ("green", Color::GREEN),
            ("blue", Color::new(0, 0, 255)),
            ("white", Color::WHITE),
            ("off", Color::OFF),
        ] {
            info!("showing {name}");
            for _ in 0..400 {
                led.show(color, u8::MAX)?;
                FreeRtos::delay_ms(5);
            }
        }
    }
}
