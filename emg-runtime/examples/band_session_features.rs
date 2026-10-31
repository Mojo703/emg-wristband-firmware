//! Exports the band's 64 band-power features for a recorded session, using the
//! reference gains of one calibration slot, or the prior image's rows as raw
//! features with their labels.
//!
//!     cargo run --release --example band_session_features -- session <partition.bin> <slot> <emg.i16> <features.npy>
//!     cargo run --release --example band_session_features -- prior <partition.bin> <rows.npy> <labels.npy>

use std::io::Write;

use emg_runtime::band_features::{
    BandFeaturePipeline, CHANNEL_COUNT, FEATURE_COUNT, WINDOW_SAMPLES,
};
use emg_runtime::flash_image::{self, PriorImage, SLOT_BYTES, SLOT_OFFSETS};
use emg_runtime::streaming_fit::ROW_STRIDE;

const MICROVOLTS_PER_COUNT: f32 = 12.207031;
const RECORD_VALUES: usize = CHANNEL_COUNT * WINDOW_SAMPLES;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    match arguments.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [_, "session", partition, slot, session, output] => export_session(
            &std::fs::read(partition).expect("partition readable"),
            slot.parse().expect("slot index"),
            &std::fs::read(session).expect("session readable"),
            output,
        ),
        [_, "prior", partition, rows, labels] => {
            export_prior(&std::fs::read(partition).expect("partition readable"), rows, labels)
        }
        _ => panic!(
            "usage: band_session_features session <partition> <slot> <emg.i16> <out.npy> | prior <partition> <rows.npy> <labels.npy>"
        ),
    }
}

fn export_session(partition: &[u8], slot: usize, bytes: &[u8], output: &str) {
    let prior = PriorImage::parse(partition).expect("prior image");
    let live = flash_image::parse_slot(
        slot,
        &partition[SLOT_OFFSETS[slot]..SLOT_OFFSETS[slot] + SLOT_BYTES],
        prior.hash(),
    )
    .expect("slot");
    let mut pipeline = BandFeaturePipeline::new(MICROVOLTS_PER_COUNT, live.record.reference_gains);
    let values: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let mut features = Vec::new();
    for record in values.chunks_exact(RECORD_VALUES) {
        for instant in 0..WINDOW_SAMPLES {
            let mut sample = [0i16; CHANNEL_COUNT];
            for (channel, value) in sample.iter_mut().enumerate() {
                *value = record[channel * WINDOW_SAMPLES + instant];
            }
            if let Some(window) = pipeline.push(&sample) {
                features.extend_from_slice(&window);
            }
        }
    }
    write_npy(
        output,
        "<f4",
        &[features.len() / FEATURE_COUNT, FEATURE_COUNT],
        |file| {
            for value in &features {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        },
    );
    eprintln!(
        "slot {slot} (sequence {}): {} windows",
        live.record.sequence,
        features.len() / FEATURE_COUNT
    );
}

fn export_prior(partition: &[u8], rows_path: &str, labels_path: &str) {
    let prior = PriorImage::parse(partition).expect("prior image");
    let quantization = prior.quantization();
    let standardization = prior.standardization();
    let bytes = prior.rows().bytes();
    let count = bytes.len() / ROW_STRIDE;
    let mut raw = Vec::with_capacity(count * FEATURE_COUNT);
    let mut labels = Vec::with_capacity(count);
    for row in bytes.chunks_exact(ROW_STRIDE) {
        for feature in 0..FEATURE_COUNT {
            let code = f32::from(row[feature] as i8);
            let standardized = code * quantization.scale[feature] + quantization.offset[feature];
            raw.push(
                standardized * standardization.deviation[feature] + standardization.mean[feature],
            );
        }
        labels.push(row[FEATURE_COUNT]);
    }
    write_npy(rows_path, "<f4", &[count, FEATURE_COUNT], |file| {
        for value in &raw {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    });
    write_npy(labels_path, "|u1", &[count], |file| {
        file.write_all(&labels).unwrap()
    });
    eprintln!("prior: {count} rows, {} classes", prior.class_count());
}

fn write_npy(path: &str, descriptor: &str, shape: &[usize], body: impl FnOnce(&mut std::fs::File)) {
    let dimensions = match shape {
        [length] => format!("({length},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': {dimensions}, }}");
    while (header.len() + 11) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut file = std::fs::File::create(path).expect("output writable");
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    body(&mut file);
}
