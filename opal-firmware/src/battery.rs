//! Battery charge from the divider on GPIO4 (XIAO D3): 22 kΩ from the cell to
//! the pin, 56 kΩ from the pin to ground.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use esp_idf_svc::hal::adc::attenuation;
use esp_idf_svc::hal::adc::oneshot::config::{AdcChannelConfig, Calibration};
use esp_idf_svc::hal::adc::oneshot::{AdcChannelDriver, AdcDriver};
use esp_idf_svc::hal::adc::{ADC1, ADCCH3, ADCU1};
use esp_idf_svc::hal::gpio::Gpio4;

const DIVIDER_TOP_OHMS: u32 = 22_000;
const DIVIDER_BOTTOM_OHMS: u32 = 56_000;
const SAMPLES_PER_READING: u32 = 16;
pub(crate) const READING_INTERVAL: Duration = Duration::from_secs(60);
/// How far a reading must climb before the reported charge goes up, so load
/// changes do not make a discharging battery look like it is charging.
const RISE_MARGIN_PERCENT: u8 = 3;
/// The point on the discharge curve the band reports as full.
const FULL_CURVE_PERCENT: u32 = 96;

/// Resting voltage of a single LiPo cell against its charge, highest first.
const DISCHARGE_CURVE: [(u32, u8); 21] = [
    (4200, 100),
    (4150, 95),
    (4110, 90),
    (4080, 85),
    (4020, 80),
    (3980, 75),
    (3950, 70),
    (3910, 65),
    (3870, 60),
    (3850, 55),
    (3840, 50),
    (3820, 45),
    (3800, 40),
    (3790, 35),
    (3770, 30),
    (3750, 25),
    (3730, 20),
    (3710, 15),
    (3690, 10),
    (3610, 5),
    (3270, 0),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Reading {
    pub cell_millivolts: u32,
    pub percent: u8,
}

pub(crate) struct Battery {
    channel: AdcChannelDriver<'static, ADCCH3<ADCU1>, AdcDriver<'static, ADCU1>>,
    reported_percent: Option<u8>,
    last_reading_at: Option<Instant>,
}

impl Battery {
    pub(crate) fn start(adc: ADC1<'static>, pin: Gpio4<'static>) -> Result<Self> {
        let driver = AdcDriver::new(adc).context("taking ADC1")?;
        let config = AdcChannelConfig {
            // 0 to 3.1 V on the S3; a full cell puts 3.02 V on the pin.
            attenuation: attenuation::DB_12,
            calibration: Calibration::Curve,
            ..AdcChannelConfig::new()
        };
        let channel =
            AdcChannelDriver::new(driver, pin, &config).context("configuring the battery pin")?;
        Ok(Self {
            channel,
            reported_percent: None,
            last_reading_at: None,
        })
    }

    /// A reading when one is due, at most once per [`READING_INTERVAL`].
    pub(crate) fn poll(&mut self) -> Result<Option<Reading>> {
        if self
            .last_reading_at
            .is_some_and(|at| at.elapsed() < READING_INTERVAL)
        {
            return Ok(None);
        }
        self.last_reading_at = Some(Instant::now());
        let mut pin_millivolts_total = 0;
        for _ in 0..SAMPLES_PER_READING {
            pin_millivolts_total +=
                u32::from(self.channel.read().context("reading the battery pin")?);
        }
        let cell_millivolts = cell_millivolts_from_pin(pin_millivolts_total / SAMPLES_PER_READING);
        let percent = settled_percent(
            self.reported_percent,
            reported_charge(percent_from_cell(cell_millivolts)),
        );
        self.reported_percent = Some(percent);
        Ok(Some(Reading {
            cell_millivolts,
            percent,
        }))
    }
}

fn cell_millivolts_from_pin(pin_millivolts: u32) -> u32 {
    pin_millivolts * (DIVIDER_TOP_OHMS + DIVIDER_BOTTOM_OHMS) / DIVIDER_BOTTOM_OHMS
}

fn percent_from_cell(cell_millivolts: u32) -> u8 {
    let (full_millivolts, _) = DISCHARGE_CURVE[0];
    if cell_millivolts >= full_millivolts {
        return 100;
    }
    for pair in DISCHARGE_CURVE.windows(2) {
        let [(upper_millivolts, upper_percent), (lower_millivolts, lower_percent)] = pair else {
            unreachable!("windows(2) yields pairs");
        };
        if cell_millivolts >= *lower_millivolts {
            let span = upper_millivolts - lower_millivolts;
            let above = cell_millivolts - lower_millivolts;
            let rise = u32::from(upper_percent - lower_percent) * above / span;
            return lower_percent + rise as u8;
        }
    }
    0
}

fn reported_charge(curve_percent: u8) -> u8 {
    (u32::from(curve_percent) * 100 / FULL_CURVE_PERCENT).min(100) as u8
}

fn settled_percent(reported: Option<u8>, measured: u8) -> u8 {
    match reported {
        Some(reported) if measured > reported && measured < reported + RISE_MARGIN_PERCENT => {
            reported
        }
        _ => measured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divider_scales_pin_to_cell() {
        assert_eq!(cell_millivolts_from_pin(3016), 4200);
    }

    #[test]
    fn curve_endpoints_and_interpolation() {
        assert_eq!(percent_from_cell(4300), 100);
        assert_eq!(percent_from_cell(4200), 100);
        assert_eq!(percent_from_cell(3840), 50);
        assert_eq!(percent_from_cell(3860), 57);
        assert_eq!(percent_from_cell(3270), 0);
        assert_eq!(percent_from_cell(3000), 0);
    }

    #[test]
    fn ninety_six_on_the_curve_reports_full() {
        assert_eq!(reported_charge(100), 100);
        assert_eq!(reported_charge(96), 100);
        assert_eq!(reported_charge(95), 98);
        assert_eq!(reported_charge(48), 50);
        assert_eq!(reported_charge(0), 0);
    }

    #[test]
    fn small_rises_hold_and_falls_pass() {
        assert_eq!(settled_percent(None, 60), 60);
        assert_eq!(settled_percent(Some(60), 62), 60);
        assert_eq!(settled_percent(Some(60), 63), 63);
        assert_eq!(settled_percent(Some(60), 58), 58);
    }
}
