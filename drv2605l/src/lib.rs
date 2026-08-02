//! One DRV2605L haptic driver, open-loop ERM, fired from the host over I2C.
//!
//! Datasheet: TI SLOS854D.
//!
//! The scope is the ROM effect library and the eight-slot waveform sequencer, plus
//! real-time playback as a small extra. External trigger, PWM, analog and audio-to-vibe
//! modes are absent: the IN/TRIG pin is not part of this design, so the GO register is
//! the only way a waveform starts.
//!
//! Auto-calibration is not implemented either. The motor is a brushed rotor pulled from
//! a 9 g servo, which is close enough to a 3 V ERM that Library B's open-loop waveforms
//! land on it directly, and open-loop drive needs no back-EMF measurement to work.
//!
//! Datasheet constraints are carried by types rather than checked at run time: registers
//! know whether they are writable ([`register`]), a [`Pause`] cannot hold a duration the
//! sequencer could not encode, and [`Volts`] cannot exceed what the voltage registers can
//! express. All three are `const`-checked, so a bad pattern table fails to compile
//! instead of failing at the bench.

#![no_std]

use embedded_hal::i2c::I2c;

mod library_effect;
pub mod register;

pub use library_effect::LibraryEffect;

use register::{ReadableRegister, Register, WritableRegister};

/// Fixed by the part; the DRV2605L has no address-select pin.
pub const DEVICE_ADDRESS: u8 = 0x5A;

/// Value of `DEVICE_ID[2:0]` in STATUS that identifies this part rather than one of its
/// siblings (SLOS854D Table 4: 3 is the DRV2605, 4 the DRV2604, 6 the DRV2604L).
const DEVICE_IDENTIFIER: u8 = 7;

const MODE_DEVICE_RESET: u8 = 1 << 7;
const MODE_STANDBY: u8 = 1 << 6;
const MODE_INTERNAL_TRIGGER: u8 = 0;
const MODE_REALTIME_PLAYBACK: u8 = 5;
const MODE_DIAGNOSTICS: u8 = 6;

const GO_BIT: u8 = 1 << 0;

/// `DIAG_RESULT`. Set means the routine *failed*, so the sense is inverted on the way
/// into [`DeviceStatus`].
const STATUS_DIAGNOSTICS_FAILED: u8 = 1 << 3;
const STATUS_OVERTEMPERATURE: u8 = 1 << 1;
const STATUS_OVERCURRENT: u8 = 1 << 0;

/// Bit 7 of FEEDBACK_CONTROL: 0 selects ERM, 1 selects LRA.
const FEEDBACK_CONTROL_SELECT_LRA: u8 = 1 << 7;

const CONTROL3_ERM_OPEN_LOOP: u8 = 1 << 5;
const CONTROL3_UNSIGNED_REALTIME_DATA: u8 = 1 << 3;

/// `LIBRARY_SEL` value 2. Library B is the 3 V ERM library with the shortest brake time
/// of the 3 V set (SLOS854D Table 1), which is what makes its clicks feel crisp rather
/// than smeared on a small rotor.
const LIBRARY_SELECTION_ERM_B: u8 = 2;

/// Slots 0x04 through 0x0B.
pub const SEQUENCER_SLOTS: usize = 8;

/// Set in a sequencer slot, this reinterprets the low seven bits as a wait time instead
/// of a waveform identifier.
const SEQUENCER_WAIT_FLAG: u8 = 1 << 7;

/// Reset is not instant and the device may not acknowledge while it is in progress, so
/// the poll tolerates bus errors until this many attempts have gone by. Each attempt is
/// a full I2C transaction, hundreds of microseconds even at 400 kHz, which is what
/// paces the loop -- the driver has no clock of its own and takes no delay provider.
const RESET_POLL_ATTEMPTS: u32 = 500;

/// Paced the same way as [`RESET_POLL_ATTEMPTS`]. Diagnostics drives the actuator for
/// real, so it takes longer than a reset and gets a correspondingly larger budget.
const DIAGNOSTICS_POLL_ATTEMPTS: u32 = 2000;

/// Amplitude that means "stop driving" in real-time playback.
///
/// Not zero. `BIDIR_INPUT` defaults to 1, and in an open-loop mode that makes the input
/// bidirectional: 50% is no output, 100% is full-scale forward, and **0% is full-scale
/// reverse**, which is a hard brake rather than silence (SLOS854D Table 25). Mid-scale is
/// the neutral point.
pub const REALTIME_AMPLITUDE_SILENT: u8 = 128;

/// A voltage the DRV2605L can actually be programmed to, held as whole millivolts.
///
/// Millivolts rather than `f32` because the register conversions are then exact integer
/// arithmetic that works in `const` context on any target, with no floating point in the
/// driver at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Volts(u16);

impl Volts {
    /// Both voltage registers are eight bits, and the coarser of the two scales is
    /// OD_CLAMP at 21.59 mV per step, so 255 steps is the highest either can express
    /// (SLOS854D Equations 4 and 6).
    pub const MAXIMUM_MILLIVOLTS: u16 = 5505;

    /// Panics at compile time when used in a `const`, which is the point: an
    /// unrepresentable drive voltage should never reach a register.
    pub const fn from_millivolts(millivolts: u16) -> Self {
        assert!(
            millivolts <= Self::MAXIMUM_MILLIVOLTS,
            "drive voltage exceeds what the DRV2605L voltage registers can express"
        );
        Self(millivolts)
    }

    pub const fn millivolts(self) -> u16 {
        self.0
    }
}

/// Nominal drive voltage of the motor.
///
/// Inert in open loop -- RATED_VOLTAGE is explicitly ignored there and OD_CLAMP sets the
/// full-scale reference instead (SLOS854D section 8.5.2.1). Written anyway so a register
/// dump describes the motor it is driving.
pub const MOTOR_RATED_VOLTAGE: Volts = Volts::from_millivolts(3000);

/// Full-scale reference for open-loop drive, and the only amplitude control that does
/// anything in this configuration.
///
/// Set above the 3.3 V rail on purpose. Open-loop output is a duty cycle against the
/// supply, so asking for more than the rail saturates the duty at full scale and gets the
/// loudest drive the part can produce. Safe only because the intended actuator is a low
/// impedance motor being characterised on a scope; raising this with a delicate actuator
/// attached would not be.
pub const MOTOR_OVERDRIVE_CLAMP: Volts = Volts::from_millivolts(3600);

/// SLOS854D Equation 4, closed-loop ERM: `V = 21.18 mV x RATED_VOLTAGE`.
const RATED_VOLTAGE_MICROVOLTS_PER_STEP: u32 = 21_180;

/// SLOS854D Equation 6, open-loop ERM: `V = 21.59 mV x OD_CLAMP`.
const OVERDRIVE_CLAMP_MICROVOLTS_PER_STEP: u32 = 21_590;

pub const fn rated_voltage_register_value(voltage: Volts) -> u8 {
    register_steps(voltage, RATED_VOLTAGE_MICROVOLTS_PER_STEP)
}

pub const fn overdrive_clamp_register_value(voltage: Volts) -> u8 {
    register_steps(voltage, OVERDRIVE_CLAMP_MICROVOLTS_PER_STEP)
}

const fn register_steps(voltage: Volts, microvolts_per_step: u32) -> u8 {
    let microvolts = voltage.millivolts() as u32 * 1000;
    let steps = (microvolts + microvolts_per_step / 2) / microvolts_per_step;
    if steps > u8::MAX as u32 {
        u8::MAX
    } else {
        steps as u8
    }
}

/// Which half of a register transaction failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusAccess {
    Read,
    Write,
}

/// What a driver call can fail on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error<BusError> {
    /// The I2C transaction itself failed. The register is named so a wiring fault reads
    /// differently from a chip that stopped answering mid-sequence.
    Bus {
        register: &'static str,
        access: BusAccess,
        source: BusError,
    },
    /// `DEV_RESET` never self-cleared.
    ResetTimedOut,
    /// The diagnostics routine was started but GO never self-cleared, so the result in
    /// STATUS would not have been valid to read.
    DiagnosticsTimedOut,
    /// STATUS answered, but with some other part's identifier.
    UnexpectedDevice { device_identifier: u8 },
    /// More steps than there are sequencer slots.
    SequenceTooLong { steps: usize },
}

impl<BusError: core::fmt::Debug> core::fmt::Display for Error<BusError> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Bus {
                register,
                access,
                source,
            } => {
                let verb = match access {
                    BusAccess::Read => "read of",
                    BusAccess::Write => "write to",
                };
                write!(formatter, "I2C {verb} {register} failed: {source:?}")
            }
            Error::ResetTimedOut => write!(formatter, "device reset never completed"),
            Error::DiagnosticsTimedOut => {
                write!(formatter, "diagnostics routine never completed")
            }
            Error::UnexpectedDevice { device_identifier } => write!(
                formatter,
                "STATUS reported device identifier {device_identifier}, expected \
                 {DEVICE_IDENTIFIER} for the DRV2605L"
            ),
            Error::SequenceTooLong { steps } => write!(
                formatter,
                "sequence of {steps} steps exceeds the {SEQUENCER_SLOTS} sequencer slots"
            ),
        }
    }
}

impl<BusError: core::fmt::Debug> core::error::Error for Error<BusError> {}

/// A decoded STATUS byte.
///
/// Reading STATUS is destructive, and unevenly so, which decides how the caller has to
/// use this (SLOS854D Table 4 and section 8.4.4.4):
///
/// - `diagnostics_passed` and `overtemperature` come from flags the datasheet says clear
///   upon read. One read consumes them.
/// - `overcurrent` behaves differently: it "remains asserted until the short is removed",
///   so it survives being read while the fault is still there.
///
/// Everything therefore has to be decoded from a single [`Drv2605l::read_status`], never
/// from two reads of the same event -- the second read would come back clean and look
/// like the fault had gone away.
///
/// The two fault flags are latched rather than sampled, which is what makes a readback
/// after a waveform finishes worth doing at all: a trip that happened in the middle of an
/// effect is still visible afterwards. Note that output shorts are only detected while
/// the device is actually driving. A short present while the device sits idle goes
/// unnoticed until the next waveform runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceStatus {
    pub raw: u8,
    /// `DEVICE_ID[2:0]`.
    pub device_identifier: u8,
    /// `DIAG_RESULT` inverted: the register bit is set on *failure*.
    pub diagnostics_passed: bool,
    pub overtemperature: bool,
    pub overcurrent: bool,
}

impl DeviceStatus {
    pub const fn decode(raw: u8) -> Self {
        Self {
            raw,
            device_identifier: raw >> 5,
            diagnostics_passed: raw & STATUS_DIAGNOSTICS_FAILED == 0,
            overtemperature: raw & STATUS_OVERTEMPERATURE != 0,
            overcurrent: raw & STATUS_OVERCURRENT != 0,
        }
    }

    /// Whether either fault flag is set, which is the question the bench asks after every
    /// pattern.
    pub const fn has_fault(&self) -> bool {
        self.overtemperature || self.overcurrent
    }
}

/// What the built-in actuator diagnostics found.
///
/// The fault flags travel with the result because they come out of the same single STATUS
/// read that carries `passed`; reading them separately would lose them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosticsOutcome {
    /// False means the actuator is absent, shorted, timing out, or returning
    /// out-of-range back-EMF. The chip does not say which.
    pub passed: bool,
    pub overtemperature: bool,
    pub overcurrent: bool,
    pub raw_status: u8,
}

/// A sequencer wait, in units the sequencer can actually store.
///
/// The register holds a count of 10 ms units in seven bits, so a pause is a multiple of
/// 10 ms up to 1270 ms and nothing else. Constructing one is `const`, so a pattern table
/// asking for 15 ms fails to build rather than failing a range check on the bench.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pause(u8);

impl Pause {
    pub const RESOLUTION_MILLISECONDS: u16 = 10;
    pub const MAXIMUM_MILLISECONDS: u16 = Self::RESOLUTION_MILLISECONDS * 127;

    pub const fn from_milliseconds(milliseconds: u16) -> Self {
        assert!(
            milliseconds % Self::RESOLUTION_MILLISECONDS == 0,
            "sequencer pauses have 10 ms resolution"
        );
        assert!(
            milliseconds <= Self::MAXIMUM_MILLISECONDS,
            "sequencer pauses cannot exceed 1270 ms"
        );
        Self((milliseconds / Self::RESOLUTION_MILLISECONDS) as u8)
    }

    pub const fn milliseconds(self) -> u16 {
        self.0 as u16 * Self::RESOLUTION_MILLISECONDS
    }

    const fn encode(self) -> u8 {
        SEQUENCER_WAIT_FLAG | self.0
    }
}

/// One entry in the waveform sequencer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStep {
    Effect(LibraryEffect),
    Pause(Pause),
}

impl SequenceStep {
    /// Shorthand for a pause step. `const`, so out-of-range durations are build errors.
    pub const fn pause(milliseconds: u16) -> Self {
        SequenceStep::Pause(Pause::from_milliseconds(milliseconds))
    }

    /// The byte this step occupies in a sequencer slot. Cannot fail: both variants are
    /// already constrained to what the slot can hold.
    pub const fn encode(self) -> u8 {
        match self {
            SequenceStep::Effect(effect) => effect as u8,
            SequenceStep::Pause(pause) => pause.encode(),
        }
    }
}

pub struct Drv2605l<Bus> {
    i2c: Bus,
}

impl<Bus: I2c> Drv2605l<Bus> {
    /// Resets the device, confirms it is a DRV2605L, sets it up for open-loop ERM
    /// playback out of Library B under internal trigger, and leaves it out of standby
    /// ready for [`Self::play_sequence`].
    pub fn new(i2c: Bus) -> Result<Self, Error<Bus::Error>> {
        let mut driver = Self { i2c };
        driver.reset()?;

        let device_identifier = driver.read_status()?.device_identifier;
        if device_identifier != DEVICE_IDENTIFIER {
            return Err(Error::UnexpectedDevice { device_identifier });
        }

        driver.write_register(
            register::RatedVoltage,
            rated_voltage_register_value(MOTOR_RATED_VOLTAGE),
        )?;
        driver.write_register(
            register::OverdriveClamp,
            overdrive_clamp_register_value(MOTOR_OVERDRIVE_CLAMP),
        )?;

        driver.update_register(register::FeedbackControl, |value| {
            value & !FEEDBACK_CONTROL_SELECT_LRA
        })?;

        // Real-time playback data is signed out of reset, which spends half the range
        // driving the rotor backwards. Unsigned makes the amplitude argument a plain
        // 0-255 intensity; see REALTIME_AMPLITUDE_SILENT for where the neutral point
        // lands once BIDIR_INPUT is taken into account.
        driver.update_register(register::Control3, |value| {
            value | CONTROL3_ERM_OPEN_LOOP | CONTROL3_UNSIGNED_REALTIME_DATA
        })?;

        driver.write_register(register::LibrarySelection, LIBRARY_SELECTION_ERM_B)?;

        // Writing MODE last is what leaves standby, so nothing plays out of a
        // half-configured register file.
        driver.write_register(register::Mode, MODE_INTERNAL_TRIGGER)?;
        Ok(driver)
    }

    /// Loads the sequencer and fires it. Returns as soon as GO is set; playback runs on
    /// the device, so poll [`Self::is_playing`] to know when it has finished.
    pub fn play_sequence(&mut self, steps: &[SequenceStep]) -> Result<(), Error<Bus::Error>> {
        if steps.len() > SEQUENCER_SLOTS {
            return Err(Error::SequenceTooLong { steps: steps.len() });
        }

        // A zero identifier ends the sequence, so every slot the caller did not fill has
        // to be cleared -- otherwise the tail of the previous pattern plays too.
        let mut slots = [0u8; SEQUENCER_SLOTS];
        for (slot, step) in slots.iter_mut().zip(steps) {
            *slot = step.encode();
        }
        self.write_sequencer_slots(&slots)?;

        self.write_register(register::Go, GO_BIT)
    }

    pub fn play_effect(&mut self, effect: LibraryEffect) -> Result<(), Error<Bus::Error>> {
        self.play_sequence(&[SequenceStep::Effect(effect)])
    }

    /// GO stays high until the whole sequence has played out.
    pub fn is_playing(&mut self) -> Result<bool, Error<Bus::Error>> {
        Ok(self.read_register(register::Go)? & GO_BIT != 0)
    }

    /// Cancels playback part-way through.
    pub fn stop(&mut self) -> Result<(), Error<Bus::Error>> {
        self.write_register(register::Go, 0)
    }

    pub fn enter_standby(&mut self) -> Result<(), Error<Bus::Error>> {
        self.update_register(register::Mode, |value| value | MODE_STANDBY)
    }

    pub fn exit_standby(&mut self) -> Result<(), Error<Bus::Error>> {
        self.update_register(register::Mode, |value| value & !MODE_STANDBY)
    }

    /// One STATUS read, decoded. See [`DeviceStatus`] on why the result of a single read
    /// has to be kept rather than re-read.
    pub fn read_status(&mut self) -> Result<DeviceStatus, Error<Bus::Error>> {
        Ok(DeviceStatus::decode(self.read_register(register::Status)?))
    }

    /// Runs the chip's own actuator diagnostics, which drive the output briefly and
    /// report whether anything is connected and behaving.
    ///
    /// **A pass means very little in open-loop ERM.** The datasheet describes the routine
    /// unconditionally (sections 8.3.2.6 and 8.6.2) and never says it depends on the loop
    /// mode, but the check it documents -- actuator "not present or shorted, timing out,
    /// or giving out-of-range back-EMF" -- is a back-EMF measurement, and open-loop ERM
    /// drive never measures back-EMF. On this bench the routine passed with nothing at
    /// all connected to OUT+/OUT-, so treat a pass as uninformative and only a failure as
    /// evidence. Confirming an actuator is present needs a scope or an ammeter, not this.
    ///
    /// Restores internal-trigger mode before returning, including when the routine fails,
    /// so a failed diagnostic does not leave the chip stuck in diagnostics mode.
    pub fn run_diagnostics(&mut self) -> Result<DiagnosticsOutcome, Error<Bus::Error>> {
        self.write_register(register::Mode, MODE_DIAGNOSTICS)?;
        self.write_register(register::Go, GO_BIT)?;

        let mut completed = false;
        for _ in 0..DIAGNOSTICS_POLL_ATTEMPTS {
            if self.read_register(register::Go)? & GO_BIT == 0 {
                completed = true;
                break;
            }
        }

        // The result is only meaningful once GO has self-cleared, but the mode has to be
        // put back either way.
        let outcome = if completed {
            let status = self.read_status()?;
            Some(DiagnosticsOutcome {
                passed: status.diagnostics_passed,
                overtemperature: status.overtemperature,
                overcurrent: status.overcurrent,
                raw_status: status.raw,
            })
        } else {
            None
        };

        self.write_register(register::Mode, MODE_INTERNAL_TRIGGER)?;
        outcome.ok_or(Error::DiagnosticsTimedOut)
    }

    /// Enters real-time playback and hands back the only handle that can set an
    /// amplitude, so driving the output directly is impossible unless the chip is
    /// actually in the mode that honours it.
    ///
    /// The output stays silent until the first [`RealtimePlayback::set_amplitude`].
    /// Finish with [`RealtimePlayback::finish`]; there is no `Drop` shortcut, because
    /// leaving the mode is I2C traffic that can fail and a silently swallowed error here
    /// would leave the part driving.
    pub fn start_realtime_playback(
        &mut self,
    ) -> Result<RealtimePlayback<'_, Bus>, Error<Bus::Error>> {
        self.write_register(register::RealtimePlaybackInput, REALTIME_AMPLITUDE_SILENT)?;
        self.write_register(register::Mode, MODE_REALTIME_PLAYBACK)?;
        Ok(RealtimePlayback { driver: self })
    }

    fn reset(&mut self) -> Result<(), Error<Bus::Error>> {
        self.write_register(register::Mode, MODE_DEVICE_RESET)?;
        for _ in 0..RESET_POLL_ATTEMPTS {
            if let Ok(mode) = self.read_register(register::Mode) {
                if mode & MODE_DEVICE_RESET == 0 {
                    return Ok(());
                }
            }
        }
        Err(Error::ResetTimedOut)
    }

    fn update_register<R: ReadableRegister + WritableRegister + Copy>(
        &mut self,
        target: R,
        change: impl FnOnce(u8) -> u8,
    ) -> Result<(), Error<Bus::Error>> {
        let value = self.read_register(target)?;
        self.write_register(target, change(value))
    }

    fn read_register<R: ReadableRegister>(&mut self, _target: R) -> Result<u8, Error<Bus::Error>> {
        let mut value = [0u8; 1];
        self.i2c
            .write_read(DEVICE_ADDRESS, &[R::ADDRESS], &mut value)
            .map_err(|source| Error::Bus {
                register: R::NAME,
                access: BusAccess::Read,
                source,
            })?;
        Ok(value[0])
    }

    fn write_register<R: WritableRegister>(
        &mut self,
        _target: R,
        value: u8,
    ) -> Result<(), Error<Bus::Error>> {
        self.i2c
            .write(DEVICE_ADDRESS, &[R::ADDRESS, value])
            .map_err(|source| Error::Bus {
                register: R::NAME,
                access: BusAccess::Write,
                source,
            })
    }

    /// All eight slots in one transaction. The chip advances its own register pointer
    /// across a multi-byte write (SLOS854D section 8.5.3.2).
    fn write_sequencer_slots(
        &mut self,
        slots: &[u8; SEQUENCER_SLOTS],
    ) -> Result<(), Error<Bus::Error>> {
        let mut payload = [0u8; SEQUENCER_SLOTS + 1];
        payload[0] = register::WaveformSequencer::ADDRESS;
        payload[1..].copy_from_slice(slots);
        self.i2c
            .write(DEVICE_ADDRESS, &payload)
            .map_err(|source| Error::Bus {
                register: register::WaveformSequencer::NAME,
                access: BusAccess::Write,
                source,
            })
    }
}

/// Borrow-guard proving the chip is in real-time playback mode.
///
/// Holding one borrows the driver, so ROM playback cannot be started behind its back, and
/// it is the only route to the amplitude register.
pub struct RealtimePlayback<'a, Bus: I2c> {
    driver: &'a mut Drv2605l<Bus>,
}

impl<Bus: I2c> RealtimePlayback<'_, Bus> {
    /// Unsigned intensity. [`REALTIME_AMPLITUDE_SILENT`] is the neutral point and 255 is
    /// full-scale forward drive; below neutral the part drives in reverse to brake.
    pub fn set_amplitude(&mut self, amplitude: u8) -> Result<(), Error<Bus::Error>> {
        self.driver
            .write_register(register::RealtimePlaybackInput, amplitude)
    }

    /// Silences the output and returns to internal-trigger mode.
    pub fn finish(self) -> Result<(), Error<Bus::Error>> {
        self.driver
            .write_register(register::RealtimePlaybackInput, REALTIME_AMPLITUDE_SILENT)?;
        self.driver
            .write_register(register::Mode, MODE_INTERNAL_TRIGGER)
    }
}
