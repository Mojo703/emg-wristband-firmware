//! Sampling loop: poll DRDY, clock out a frame, hand it off to gesture recognition.
//!
//! For this skeleton the "hand off" is just a log line. Once the pipeline has been
//! tested and has somewhere to send samples, I'll replace `log_frame` with that call.

use log::{info, warn};

use crate::ads1298::{Ads1298Pair, Frame};

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
        info!("dev{i} status={:#08x} channels={:?}", sample.status, sample.channels);
    }
}
