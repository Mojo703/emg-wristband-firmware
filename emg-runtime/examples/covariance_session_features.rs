use std::io::Write;

use emg_runtime::band_features::{CHANNEL_COUNT, WINDOW_SAMPLES};
use emg_runtime::covariance_features::{
    CovarianceFeaturePipeline, CovarianceVariant, FEATURE_COUNT,
};

const MICROVOLTS_PER_COUNT: f32 = 12.207031;
const RECORD_VALUES: usize = CHANNEL_COUNT * WINDOW_SAMPLES;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, label, session, output] = arguments.as_slice() else {
        panic!("usage: covariance_session_features <variant> <emg.i16> <output.npy>");
    };
    let variant = CovarianceVariant::named(label).expect("known variant");
    let bytes = std::fs::read(session).expect("session readable");
    let values: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();

    let mut pipeline = CovarianceFeaturePipeline::with_variant(MICROVOLTS_PER_COUNT, variant);
    let mut features: Vec<f32> = Vec::new();
    let mut sweeps = [0u64; 4];
    for record in values.chunks_exact(RECORD_VALUES) {
        for instant in 0..WINDOW_SAMPLES {
            let mut sample = [0i16; CHANNEL_COUNT];
            for (channel, value) in sample.iter_mut().enumerate() {
                *value = record[channel * WINDOW_SAMPLES + instant];
            }
            if let Some(window) = pipeline.push(&sample) {
                features.extend_from_slice(window);
                for (total, &count) in sweeps.iter_mut().zip(pipeline.last_window_sweeps().iter()) {
                    *total += u64::from(count);
                }
            }
        }
    }

    let windows = features.len() / FEATURE_COUNT;
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({windows}, {FEATURE_COUNT}), }}"
    );
    while (header.len() + 11) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut file = std::fs::File::create(output).expect("output writable");
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for value in &features {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    let per_window: Vec<String> = sweeps
        .iter()
        .map(|total| format!("{:.2}", *total as f64 / windows as f64))
        .collect();
    println!(
        "{label} {windows} windows, mean sweeps per band {}",
        per_window.join(" ")
    );
}
