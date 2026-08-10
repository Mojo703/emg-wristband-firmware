//! The dashboard's measurement code, split out from the relay binary so the
//! offline tools in `src/bin/` measure a recording with the same estimator the
//! live panel measures the stream with.

pub mod calibration_library;
pub mod guided_session;
pub mod session_report;
pub mod signal_quality;
pub mod timing;
