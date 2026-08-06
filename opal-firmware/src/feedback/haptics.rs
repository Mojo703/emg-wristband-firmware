//! The DRV2605L haptic output.
//!
//! The `drv2605l` crate knows the part; this module adds the two things a wristband
//! needs and a bench harness does not — a bus that cannot block forever, and the
//! discipline to stop talking to a chip that has stopped answering.
//!
//! Playback is asynchronous: [`Haptics::play`] loads the sequencer and returns, the
//! chip runs the pattern for 100–500 ms, and [`Haptics::poll_playback`] notices when
//! it is done. Blocking on it, as the bench harness does, would freeze the indicator
//! LED for the length of every buzz.

use drv2605l::{Drv2605l, SequenceStep};
use embedded_hal::i2c::{ErrorType, I2c, Operation, SevenBitAddress};
use esp_idf_svc::hal::delay::TickType;
use esp_idf_svc::hal::gpio::{InputPin, OutputPin};
use esp_idf_svc::hal::i2c::{I2c as I2cPeripheral, I2cConfig, I2cDriver, I2cError};
use esp_idf_svc::hal::units::FromValueType;
use log::{info, warn};
use std::time::Duration;

/// Standard mode. A handful of bytes per pattern is nowhere near the limit, and the
/// slower edges are kinder to the flying leads this rides on.
const BUS_KILOHERTZ: u32 = 100;

/// The longest any single transaction may hold the feedback thread.
///
/// `embedded-hal`'s blanket implementation for `I2cDriver` passes `BLOCK`, which
/// waits forever, so a board browning out mid-transfer would take the indicator LED
/// with it. Long enough that a healthy bus never trips it.
const TRANSACTION_TIMEOUT: Duration = Duration::from_millis(50);

/// Consecutive bus failures before the output gives up for this boot. Retrying
/// forever against an absent board costs a stalled tick every 20 ms for as long as
/// the device runs; three in a row is past any plausible transient.
const FAILURES_BEFORE_GIVING_UP: u32 = 3;

/// An [`I2cDriver`] whose transactions time out. Exists only to replace the `BLOCK`
/// in `esp-idf-hal`'s own `embedded-hal` implementation with [`TRANSACTION_TIMEOUT`].
pub struct TimeBoundedI2cBus<'d> {
    driver: I2cDriver<'d>,
    timeout: u32,
}

impl<'d> TimeBoundedI2cBus<'d> {
    pub fn new(driver: I2cDriver<'d>, timeout: Duration) -> Self {
        Self {
            driver,
            timeout: TickType::from(timeout).ticks(),
        }
    }
}

impl ErrorType for TimeBoundedI2cBus<'_> {
    type Error = I2cError;
}

impl I2c<SevenBitAddress> for TimeBoundedI2cBus<'_> {
    fn read(&mut self, address: u8, buffer: &mut [u8]) -> Result<(), Self::Error> {
        self.driver
            .read(address, buffer, self.timeout)
            .map_err(I2cError::other)
    }

    fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), Self::Error> {
        self.driver
            .write(address, bytes, self.timeout)
            .map_err(I2cError::other)
    }

    fn write_read(
        &mut self,
        address: u8,
        bytes: &[u8],
        buffer: &mut [u8],
    ) -> Result<(), Self::Error> {
        self.driver
            .write_read(address, bytes, buffer, self.timeout)
            .map_err(I2cError::other)
    }

    fn transaction(
        &mut self,
        address: u8,
        operations: &mut [Operation<'_>],
    ) -> Result<(), Self::Error> {
        self.driver
            .transaction(address, operations, self.timeout)
            .map_err(I2cError::other)
    }
}

/// What a playback poll learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Playback {
    Running,
    /// Nothing running, so the fault flags the last pattern latched are there to read.
    Finished,
    /// The bus failed, or the chip has been given up on. Send it nothing more this
    /// pass.
    Unknown,
}

/// The motor, or the memory of one that stopped answering.
pub struct Haptics {
    driver: Drv2605l<TimeBoundedI2cBus<'static>>,
    playing: bool,
    consecutive_failures: u32,
}

impl Haptics {
    /// Brings the chip up on its own I2C bus. Allocates the bus interrupt on the
    /// calling core, which is why the feedback thread does it — see [`crate::cores`].
    pub fn bring_up(
        i2c: impl I2cPeripheral + 'static,
        data: impl InputPin + OutputPin + 'static,
        clock: impl InputPin + OutputPin + 'static,
    ) -> anyhow::Result<Self> {
        let driver = I2cDriver::new(
            i2c,
            data,
            clock,
            &I2cConfig::new().baudrate(BUS_KILOHERTZ.kHz().into()),
        )?;
        let bus = TimeBoundedI2cBus::new(driver, TRANSACTION_TIMEOUT);
        let driver = Drv2605l::new(bus).map_err(|error| anyhow::anyhow!("{error}"))?;
        Ok(Self {
            driver,
            playing: false,
            consecutive_failures: 0,
        })
    }

    /// Starts `steps` playing, cutting off whatever was running: a cue describes what
    /// just happened, so finishing a stale pattern first would invert the two.
    pub fn play(&mut self, steps: &[SequenceStep]) {
        if self.has_given_up() {
            return;
        }
        match self.driver.play_sequence(steps) {
            Ok(()) => {
                self.playing = true;
                self.consecutive_failures = 0;
            }
            Err(error) => self.record_failure(format_args!("play failed: {error}")),
        }
    }

    /// Whether a pattern is still running. One register read while playing, nothing
    /// while idle. Three-way because "finished" and "could not ask" want opposite
    /// things: the first is the edge fault flags are read on, the second is a bus to
    /// leave alone.
    pub fn poll_playback(&mut self) -> Playback {
        if self.has_given_up() {
            return Playback::Unknown;
        }
        if !self.playing {
            return Playback::Finished;
        }
        match self.driver.is_playing() {
            Ok(playing) => {
                self.playing = playing;
                self.consecutive_failures = 0;
                if playing {
                    Playback::Running
                } else {
                    Playback::Finished
                }
            }
            Err(error) => {
                self.playing = false;
                self.record_failure(format_args!("playback poll failed: {error}"));
                Playback::Unknown
            }
        }
    }

    /// Reads and reports the latched fault flags. Only meaningful right after a
    /// pattern has driven the output: the part cannot see a short while idle, and
    /// reading STATUS consumes the overtemperature flag.
    pub fn report_faults(&mut self) {
        if self.has_given_up() {
            return;
        }
        match self.driver.read_status() {
            Ok(status) if status.has_fault() => warn!(
                "haptics fault after playback: overtemperature {}, overcurrent {}",
                status.overtemperature, status.overcurrent
            ),
            Ok(_) => self.consecutive_failures = 0,
            Err(error) => self.record_failure(format_args!("status read failed: {error}")),
        }
    }

    fn has_given_up(&self) -> bool {
        self.consecutive_failures >= FAILURES_BEFORE_GIVING_UP
    }

    /// Counts a bus failure, saying so once on the one that crosses the threshold.
    /// Further failures are silent — that is the point of giving up.
    fn record_failure(&mut self, detail: std::fmt::Arguments) {
        self.consecutive_failures += 1;
        if self.consecutive_failures == FAILURES_BEFORE_GIVING_UP {
            warn!(
                "haptics {detail}; {FAILURES_BEFORE_GIVING_UP} in a row, giving up for this boot"
            );
        } else if !self.has_given_up() {
            info!("haptics {detail}");
        }
    }
}
