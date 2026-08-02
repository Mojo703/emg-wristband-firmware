//! Bench harness for the DRV2605L breakout: bring the chip up on I2C, then cycle
//! through the candidate gesture-feedback patterns so they can be told apart by feel.
//!
//! There is nothing to type at the console. Flash it, hold the motor, and listen to the
//! log for which pattern is playing.

use anyhow::Result;
use drv2605l::{
    overdrive_clamp_register_value, rated_voltage_register_value, Drv2605l, LibraryEffect,
    SequenceStep, MOTOR_OVERDRIVE_CLAMP, MOTOR_RATED_VOLTAGE,
};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::PinDriver;
use esp_idf_svc::hal::i2c::{I2cConfig, I2cDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::units::FromValueType;

/// Standard mode. The register traffic here is a handful of bytes per pattern, so the
/// bus speed is nowhere near the limit and the slower edges are kinder to flying leads.
const BUS_KILOHERTZ: u32 = 100;

/// How long the user gets to feel one pattern before the next begins.
const GAP_BETWEEN_PATTERNS_MILLISECONDS: u32 = 2000;

/// Playback is expected to finish well inside this; exceeding it means the GO bit never
/// cleared, which is worth a log line rather than a hang.
const PLAYBACK_TIMEOUT_MILLISECONDS: u32 = 3000;

const PLAYBACK_POLL_MILLISECONDS: u32 = 10;

/// Long enough to catch on any timebase, which is the whole point of the segment.
const FULL_POWER_MILLISECONDS: u32 = 1000;

/// Full-scale forward drive in real-time playback.
const FULL_POWER_AMPLITUDE: u8 = 255;

struct HapticPattern {
    name: &'static str,
    steps: &'static [SequenceStep],
}

/// The candidate set. Each entry is one wristband cue; retune by editing the steps.
///
/// The doubles and triples are built out of single clicks with explicit pauses rather
/// than the ROM's own double-click effects, because the ROM doubles are tight enough to
/// read as one longer event on a small rotor. A pause of 100 ms is far enough apart to
/// count. Pause durations are checked when this table is compiled, so a wait the
/// sequencer could not encode is a build error rather than a surprise at the bench.
const PATTERNS: [HapticPattern; 10] = [
    HapticPattern {
        name: "single_strong_click",
        steps: &[SequenceStep::Effect(
            LibraryEffect::StrongClickOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "double_click",
        steps: &[
            SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent),
            SequenceStep::pause(100),
            SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent),
        ],
    },
    HapticPattern {
        name: "triple_click",
        steps: &[
            SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent),
            SequenceStep::pause(100),
            SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent),
            SequenceStep::pause(100),
            SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent),
        ],
    },
    HapticPattern {
        name: "sharp_tick",
        steps: &[SequenceStep::Effect(
            LibraryEffect::SharpTickOneOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "soft_bump",
        steps: &[SequenceStep::Effect(LibraryEffect::SoftBumpSixtyPercent)],
    },
    HapticPattern {
        name: "short_buzz",
        steps: &[SequenceStep::Effect(
            LibraryEffect::TransitionHumOneOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "long_alert_buzz",
        steps: &[SequenceStep::Effect(
            LibraryEffect::SevenHundredFiftyMillisecondsAlertOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "pulsing_strong",
        steps: &[SequenceStep::Effect(
            LibraryEffect::PulsingStrongOneOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "ramp_up",
        steps: &[SequenceStep::Effect(
            LibraryEffect::TransitionRampUpLongSmoothOneZeroToOneHundredPercent,
        )],
    },
    HapticPattern {
        name: "ramp_down",
        steps: &[SequenceStep::Effect(
            LibraryEffect::TransitionRampDownLongSmoothOneOneHundredToZeroPercent,
        )],
    },
];

fn main() -> Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let peripherals = Peripherals::take()?;

    // IN/TRIG is wired to the board but unused: internal-trigger mode fires from the GO
    // register instead. Driven low rather than left floating so a stray edge cannot look
    // like an external trigger if the mode is ever changed.
    let mut trigger_pin = PinDriver::output(peripherals.pins.gpio4)?;
    trigger_pin.set_low()?;

    let configuration = I2cConfig::new().baudrate(BUS_KILOHERTZ.kHz().into());
    let i2c = I2cDriver::new(
        peripherals.i2c0,
        peripherals.pins.gpio5,
        peripherals.pins.gpio6,
        &configuration,
    )?;

    let mut haptics = Drv2605l::new(i2c)?;
    let status = haptics.read_status()?;
    log::info!(
        "DRV2605L up: status 0x{:02X}, device identifier {}, overtemperature {}, \
         overcurrent {}",
        status.raw,
        status.device_identifier,
        status.overtemperature,
        status.overcurrent
    );
    log_drive_configuration();

    report_diagnostics(&mut haptics)?;

    let mut lap = 1u32;
    loop {
        log::info!("lap {lap}");
        for (index, pattern) in PATTERNS.iter().enumerate() {
            log::info!("pattern {}/{}: {}", index + 1, PATTERNS.len(), pattern.name);
            haptics.play_sequence(pattern.steps)?;
            wait_for_playback(&mut haptics, pattern.name)?;
            report_any_fault(&mut haptics, pattern.name)?;
            FreeRtos::delay_ms(GAP_BETWEEN_PATTERNS_MILLISECONDS);
        }

        play_full_power_steady(&mut haptics)?;
        FreeRtos::delay_ms(GAP_BETWEEN_PATTERNS_MILLISECONDS);

        lap += 1;
    }
}

/// What the voltage constants actually became once quantised into the registers, so the
/// monitor shows the programmed value rather than the requested one.
fn log_drive_configuration() {
    let rated = rated_voltage_register_value(MOTOR_RATED_VOLTAGE);
    let clamp = overdrive_clamp_register_value(MOTOR_OVERDRIVE_CLAMP);
    log::info!(
        "drive configuration: rated voltage {} mV -> RATED_VOLTAGE {rated} (0x{rated:02X}), \
         overdrive clamp {} mV -> OD_CLAMP {clamp} (0x{clamp:02X})",
        MOTOR_RATED_VOLTAGE.millivolts(),
        MOTOR_OVERDRIVE_CLAMP.millivolts(),
    );
    log::info!("RATED_VOLTAGE is ignored in open loop; OD_CLAMP alone sets full-scale output");
}

/// A steady full-scale drive, held long enough to measure. Every ROM effect is tens of
/// milliseconds of pulse-width-modulated output, which is awkward to catch on a scope; a
/// flat second of it is not.
fn play_full_power_steady(haptics: &mut Drv2605l<I2cDriver<'static>>) -> Result<()> {
    log::info!(
        "scope aid: full_power_steady, amplitude {FULL_POWER_AMPLITUDE} held for \
         {FULL_POWER_MILLISECONDS} ms"
    );

    let mut realtime = haptics.start_realtime_playback()?;
    realtime.set_amplitude(FULL_POWER_AMPLITUDE)?;
    FreeRtos::delay_ms(FULL_POWER_MILLISECONDS);
    realtime.finish()?;

    // The heaviest sustained drive in the whole loop, so the likeliest place for a trip.
    report_any_fault(haptics, "full_power_steady")
}

/// Both fault flags latch, so this catches a trip that happened part-way through rather
/// than only one still active now. Silent when clear.
fn report_any_fault(haptics: &mut Drv2605l<I2cDriver<'static>>, name: &str) -> Result<()> {
    let status = haptics.read_status()?;
    if status.has_fault() {
        log::error!(
            "FAULT after {}: overcurrent {}, overtemperature {} (status 0x{:02X})",
            name,
            status.overcurrent,
            status.overtemperature,
            status.raw
        );
    }
    Ok(())
}

/// Runs the built-in diagnostics once and says loudly what it found. A failure is not
/// fatal here: the patterns still run afterwards, because a scope on OUT+/OUT- learns
/// more from a chip that is trying and failing than from a binary that gave up at boot.
fn report_diagnostics(haptics: &mut Drv2605l<I2cDriver<'static>>) -> Result<()> {
    match haptics.run_diagnostics() {
        Ok(outcome) if outcome.passed && !outcome.overcurrent && !outcome.overtemperature => {
            log::info!(
                "diagnostics passed (status 0x{:02X}) -- but a pass proves little in \
                 open-loop ERM, where no back-EMF is measured; it passes with nothing \
                 connected at all",
                outcome.raw_status
            );
        }
        Ok(outcome) => {
            log::error!("================ DIAGNOSTICS FAILED ================");
            log::error!(
                "actuator check {}, overcurrent {}, overtemperature {} (status 0x{:02X})",
                if outcome.passed { "passed" } else { "FAILED" },
                outcome.overcurrent,
                outcome.overtemperature,
                outcome.raw_status
            );
            if outcome.overcurrent {
                log::error!(
                    "overcurrent means the load is below the 4 ohm detection threshold; \
                     the part wants 8 ohms minimum, and a servo motor winding is far under that"
                );
            } else if !outcome.passed {
                log::error!("actuator reads as absent, shorted, or out of range");
            }
            log::error!("continuing into the pattern loop anyway so the outputs can be scoped");
            log::error!("====================================================");
        }
        Err(error) => {
            log::error!("================ DIAGNOSTICS ERROR =================");
            log::error!("could not run diagnostics: {error}");
            log::error!("continuing into the pattern loop anyway so the outputs can be scoped");
            log::error!("====================================================");
        }
    }
    Ok(())
}

fn wait_for_playback(haptics: &mut Drv2605l<I2cDriver<'static>>, name: &str) -> Result<()> {
    let mut waited = 0;
    while haptics.is_playing()? {
        if waited >= PLAYBACK_TIMEOUT_MILLISECONDS {
            log::warn!("{name} still reports playing after {waited} ms; cancelling");
            haptics.stop()?;
            return Ok(());
        }
        FreeRtos::delay_ms(PLAYBACK_POLL_MILLISECONDS);
        waited += PLAYBACK_POLL_MILLISECONDS;
    }
    Ok(())
}
