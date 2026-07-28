//! Sampling loop: poll DRDY, clock out a frame, hand it off to gesture recognition.
//!
//! For this skeleton the "hand off" is just a log line. Once the pipeline has been
//! tested and has somewhere to send samples, I'll replace `log_frame` with that call.

use anyhow::Result;
use log::{info, warn};
use sampling_utils::convert::code_to_voltage;
use sampling_utils::loff::loff_flagged;

use crate::ads1298::{Ads1298Pair, Frame, CHANNELS_PER_DEVICE};

// VREF and PGA gain currently written into the CHnSET/CONFIG3 registers
// Update these if those register values change.
const VREF: f32 = 2.4;
const GAIN: f32 = 6.0;

/// Runs forever for now, reading one frame per DRDY pulse
pub fn run(chain: &mut Ads1298Pair<'_>) -> ! {
    loop {
        match chain.data_ready() {
            Ok(true) => match chain.read_frame() {
                Ok(frame) => log_frame(&frame),
                Err(e) => warn!("frame read failed: {e}"),
            },
            Ok(false) => {}
            Err(e) => warn!("DRDY poll failed: {e}"),
        }
    }
}

fn log_frame(frame: &Frame) {
    for (i, sample) in frame.devices.iter().enumerate() {
        let mut volts = [0.0f32; CHANNELS_PER_DEVICE];
        for (ch, slot) in volts.iter_mut().enumerate() {
            *slot = if loff_flagged(sample.status, ch) {
                0.0
            } else {
                code_to_voltage(sample.channels[ch], VREF, GAIN)
            };
        }
        info!("dev{i} status={:#08x} volts={:?}", sample.status, volts);
    }
}

