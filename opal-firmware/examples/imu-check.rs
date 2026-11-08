//! Bring-up check for the MinIMU-9 v6 IMU on the XIAO ESP32-S3 motherboard.
//!
//! Powers the magnetometer down and proves it, brings the accelerometer and
//! gyroscope up, then logs once a second what the motion gate would have seen:
//! acceleration and rotation extremes, how many samples crossed the thresholds,
//! and the latest acceleration on the band's axes for checking the mounting. The
//! gate only observes; nothing here acts on it.
//!
//! Wiring, XIAO pad to IMU pin. D4 and D5 are unused on the motherboard (pin map
//! in `examples/battery-test/main.rs`).
//!
//! | XIAO      | IMU                                           |
//! |-----------|-----------------------------------------------|
//! | 3V3       | VDD. VIN stays unconnected; never power both  |
//! | GND       | GND                                           |
//! | D4, GPIO5 | SDA                                           |
//! | D5, GPIO6 | SCL                                           |
//! | none      | SA0 floating, pulled high on the carrier      |
//!
//! Every setting is in `src/motion/config.rs`. Run with `cargo imu-check`; the
//! hardware-free tests run with `cargo test-device --example imu-check`.

#[path = "../src/motion/mod.rs"]
mod motion;

use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::sys;
use log::{info, warn};

use motion::config::MotionConfig;
use motion::gate::{self, PeriodSummary};
use motion::lsm6dso::Sample;

/// Faster than the 9.6 ms sample period at 104 Hz, so no sample is skipped.
const POLL_MILLISECONDS: u32 = 5;

const SUMMARY_PERIOD: Duration = Duration::from_secs(1);

/// Consecutive read failures before the check stops: past any transient.
const FAILURES_BEFORE_GIVING_UP: u32 = 3;

fn main() -> Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let config = MotionConfig::DEFAULT;
    let peripherals = Peripherals::take()?;
    info!("=== IMU check: I2C1, SDA GPIO5 (D4), SCL GPIO6 (D5) ===");
    let mut imu = motion::bring_up(
        peripherals.i2c1,
        peripherals.pins.gpio5,
        peripherals.pins.gpio6,
        &config,
    )?;
    info!(
        "gate thresholds (placeholders): |a| more than {} g from 1 g, or |ω| over {} °/s",
        config.thresholds.acceleration_deviation_g,
        config.thresholds.angular_rate_degrees_per_second
    );

    let mut summary = PeriodSummary::EMPTY;
    let mut latest_band_acceleration_g = [0.0f32; 3];
    let mut summary_started = Instant::now();
    let mut consecutive_failures = 0;
    loop {
        FreeRtos::delay_ms(POLL_MILLISECONDS);
        match imu.read_sample() {
            Ok(Some(sample)) => {
                consecutive_failures = 0;
                let band = Sample {
                    acceleration_g: config.mounting.to_band(sample.acceleration_g),
                    angular_rate_degrees_per_second: config
                        .mounting
                        .to_band(sample.angular_rate_degrees_per_second),
                };
                summary.record(&gate::assess(&band, &config.thresholds));
                latest_band_acceleration_g = band.acceleration_g;
            }
            Ok(None) => {}
            Err(error) => {
                consecutive_failures += 1;
                warn!("read failed ({consecutive_failures} in a row): {error:#}");
                if consecutive_failures >= FAILURES_BEFORE_GIVING_UP {
                    bail!("{FAILURES_BEFORE_GIVING_UP} reads in a row failed; stopping");
                }
            }
        }

        if summary_started.elapsed() >= SUMMARY_PERIOD {
            report(&summary, latest_band_acceleration_g);
            summary = PeriodSummary::EMPTY;
            summary_started = Instant::now();
        }
    }
}

fn report(summary: &PeriodSummary, [x, y, z]: [f32; 3]) {
    if summary.samples == 0 {
        warn!("no samples in the last period");
        return;
    }
    info!(
        "{} samples, {} over threshold | |a| {:.3}..{:.3} g, |ω| max {:.1} °/s | \
         band a ({x:+.3}, {y:+.3}, {z:+.3}) g",
        summary.samples,
        summary.excessive_samples,
        summary.minimum_acceleration_g,
        summary.maximum_acceleration_g,
        summary.maximum_angular_rate_degrees_per_second,
    );
}
