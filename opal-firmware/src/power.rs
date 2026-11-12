//! The power button on D5 (GPIO6), wired to 3V3, and deep sleep.
//!
//! What a press means is [`feedback_vocabulary::PowerButton`]'s decision; this is
//! the pin and the sleep sequence. Sleep is deep sleep: the ADS1298s go into
//! power-down with PWDN held low, the LED data line is held low so it cannot clock
//! in a colour, and ext0 wakes the chip on the button's high level. Waking is a
//! reset, so the band boots exactly as it does from power-on.
//!
//! Deep sleep drops the USB-Serial-JTAG link. To reflash a sleeping band, press the
//! button or RESET.

use anyhow::Result;
use esp_idf_svc::hal::gpio::{Gpio6, Input, PinDriver, Pull};
use esp_idf_svc::sys::{self, esp, gpio_num_t};

const BUTTON: gpio_num_t = sys::gpio_num_t_GPIO_NUM_6;
const INDICATOR: gpio_num_t = sys::gpio_num_t_GPIO_NUM_5;
const ADC_POWER_DOWN: gpio_num_t = sys::gpio_num_t_GPIO_NUM_42;

/// Releases every pad hold left by the last sleep. Must run before any pin is
/// configured: a held pad ignores its driver, so the ADS1298s would stay powered
/// down and the LED dark. Each pad by name: `gpio_force_unhold_all` leaves the
/// per-pad hold bit `gpio_hold_en` sets, and that bit outlives every reset short
/// of a power cycle.
pub(crate) fn release_sleep_holds() {
    // SAFETY: register writes on pads nothing has claimed yet.
    unsafe {
        sys::gpio_deep_sleep_hold_dis();
        for pad in [INDICATOR, ADC_POWER_DOWN] {
            if let Err(error) = esp!(sys::gpio_hold_dis(pad)) {
                log::warn!("releasing GPIO{pad} from its sleep hold failed ({error})");
            }
        }
    }
}

pub(crate) struct ButtonPin {
    pin: PinDriver<'static, Input>,
}

impl ButtonPin {
    /// The button pulls the pad to 3V3; the internal pull-down holds it low while
    /// the button is up.
    pub(crate) fn bring_up(pin: Gpio6<'static>) -> Result<Self> {
        Ok(Self {
            pin: PinDriver::input(pin, Pull::Down)?,
        })
    }

    pub(crate) fn is_pressed(&self) -> bool {
        self.pin.is_high()
    }

    /// Powers the band down and sleeps until the next press. The LED must already
    /// be showing black.
    pub(crate) fn deep_sleep(self) -> ! {
        drop(self.pin);
        if let Err(error) = arm_and_hold() {
            log::error!("sleep setup failed ({error:#}); restarting instead");
            // SAFETY: no preconditions.
            unsafe { sys::esp_restart() };
        }
        // SAFETY: pads are held and the wake source is armed.
        unsafe { sys::esp_deep_sleep_start() }
    }
}

fn arm_and_hold() -> Result<()> {
    // SAFETY: pad and sleep register writes moments before the chip powers down.
    // Whatever still owns these pins never runs again.
    unsafe {
        esp!(sys::gpio_set_level(ADC_POWER_DOWN, 0))?;
        esp!(sys::gpio_hold_en(ADC_POWER_DOWN))?;
        // The RMT idles low, so this latches the data line low.
        esp!(sys::gpio_hold_en(INDICATOR))?;
        // GPIO42 is a digital pad; its hold only lasts through deep sleep with
        // this on. It holds every digital pad, so the SPI lines keep their idle
        // levels too.
        sys::gpio_deep_sleep_hold_en();

        // Level-triggered: a button still down here would wake the chip at once.
        // The vocabulary only reaches sleep after a release.
        esp!(sys::rtc_gpio_pulldown_en(BUTTON))?;
        esp!(sys::rtc_gpio_pullup_dis(BUTTON))?;
        esp!(sys::esp_sleep_enable_ext0_wakeup(BUTTON, 1))?;
    }
    Ok(())
}
