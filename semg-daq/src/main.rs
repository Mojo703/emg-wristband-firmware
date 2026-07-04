//! ESP32-S3 sEMG data-acquisition firmware.
//!
//! Brings up two ADS1298 ADCs wired in Cascade configuration (shared
//! SCLK/DIN/DOUT, independent CS and DRDY per device) and streams decoded
//! frames out as log lines. This is a bring-up skeleton: the SPI/GPIO pin
//! assignments below are placeholders and registers are not fully configured yet.

mod acquisition;
mod ads1298;
mod config;
mod registers;
mod spi_commands;

use anyhow::Result;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{PinDriver, Pull};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::spi::{config::Config as SpiConfig, SpiDeviceDriver, SpiDriver, SpiDriverConfig};
use esp_idf_svc::hal::units::FromValueType;
use log::info;

use ads1298::{Ads1298Device, Ads1298Pair};

fn main() -> Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let cfg = config::CONFIG;
    info!("ADS1298 starting sample rate: {} Hz", cfg.sample_rate_hz);

    let peripherals = Peripherals::take()?;
    let pins = peripherals.pins;

    // **Pin assignments below are placeholders for now**
    let spi_driver = SpiDriver::new(
        peripherals.spi2,
        pins.gpio12, // SCLK — shared
        pins.gpio11, // DIN (MOSI) — shared
        Some(pins.gpio13), // DOUT (MISO) — shared
        &SpiDriverConfig::new(),
    )?;

    // ADS1298 requires SPI mode 1 (CPOL=0, CPHA=1)
    // baudrate will likely be configured higher once SPI
    // communication is confirmed 
    let spi_config = SpiConfig::new()
        .baudrate(1000000.Hz())
        .data_mode(esp_idf_svc::hal::spi::config::MODE_1);
    let spi_dev1 = SpiDeviceDriver::new(&spi_driver, Some(pins.gpio10), &spi_config)?; // CS 1
    let spi_dev2 = SpiDeviceDriver::new(&spi_driver, Some(pins.gpio6), &spi_config)?; // CS 2

    let reset_n = PinDriver::output(pins.gpio8)?; // shared RESET_N
    let start = PinDriver::output(pins.gpio7)?; // shared START

    // DRDY is actively driven by the ADS1298, so no internal pull resistor is needed
    let drdy1 = PinDriver::input(pins.gpio9, Pull::Floating)?; // device 1's own DRDY
    let drdy2 = PinDriver::input(pins.gpio5, Pull::Floating)?; // device 2's own DRDY

    let adc1 = Ads1298Device::new(spi_dev1, drdy1);
    let adc2 = Ads1298Device::new(spi_dev2, drdy2);
    let mut pair = Ads1298Pair::new(adc1, adc2, reset_n, start);

    // Initialization sequence for both ADS1298
    power_up(&mut pair)?;

    pair.adc1.read_data_continuous()?;
    pair.adc2.read_data_continuous()?;
    pair.start_conversion()?;
    info!("Conversion started. Entering acquisition loop.");

    acquisition::run(&mut pair)
}

/// Brings both devices out of reset and out of their power-on RDATAC state,
/// leaving them ready for register configuration.
/// This function is still incomplete and depends on the final schematics
fn power_up(pair: &mut Ads1298Pair<'_>) -> Result<()> {
    pair.hardware_reset()?;
    // Device powers up streaming (RDATAC), SDATAC must come first or register
    // writes below will not work
    pair.adc1.stop_read_data_continuous()?;
    pair.adc2.stop_read_data_continuous()?;
    FreeRtos::delay_ms(1);

    // TODO: Configure CONFIG1-4, CHnSET (gain, lead config, data rate) registers
    // once schematics are finalized
    Ok(())
}
