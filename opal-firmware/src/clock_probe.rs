//! Production USB timing probe. It does not provide a long-horizon clock mapping.

use esp_idf_svc::hal::gpio::{AnyOutputPin, Output, PinDriver};
use log::warn;
use protocol::Frame;

pub(crate) struct ClockProbeAdapter {
    marker: Option<PinDriver<'static, Output>>,
}

impl ClockProbeAdapter {
    pub(crate) fn new(pin: AnyOutputPin<'static>) -> Self {
        let marker = PinDriver::output(pin)
            .and_then(|mut marker| {
                marker.set_low()?;
                Ok(marker)
            })
            .map_err(|error| warn!("clock probe GPIO3 unavailable: {error}"))
            .ok();
        Self { marker }
    }

    pub(crate) fn respond(
        &mut self,
        sequence: u32,
        host_send_nanoseconds: u64,
        acquisition_sample: u64,
    ) -> Frame {
        if let Some(marker) = self.marker.as_mut() {
            let _ = marker.set_high();
        }
        let device_receive_microseconds = crate::device_now_us();
        let device_send_microseconds = crate::device_now_us();
        if let Some(marker) = self.marker.as_mut() {
            let _ = marker.set_low();
        }
        Frame::ClockProbeResponse {
            sequence,
            host_send_nanoseconds,
            device_receive_microseconds,
            device_send_microseconds,
            acquisition_sample,
        }
    }
}
