//! Device-side checks on the `emg-runtime` inference path.
//!
//! These live in a firmware crate because the SIMD kernels only exist on Xtensa; off
//! target both names resolve to the scalar oracle and the comparison is vacuous.
//!
//! The fixture blob is included under `cfg(test)`, so its verification windows link
//! into the libtest binary only.
//!
//! ```sh
//! cargo test-device
//! ```

use emg_runtime::model::{ForwardResult, Model, VerifyBatch};
use emg_runtime::tensor::{AlignedI8, I8Activation, Rng};
use emg_runtime::{layers, mac};

#[repr(align(16))]
struct AlignedBlob<Bytes: ?Sized>(Bytes);

static FIXTURE_BLOB: &AlignedBlob<[u8]> =
    &AlignedBlob(*include_bytes!("../../emg-runtime/data/model_int8.bin"));

/// Integer ratios, so the comparison never materialises a float constant: the Xtensa
/// backend mishandles float constant pools in the test profile.
const TOP1_FLOOR_PERCENT: usize = 85;
const AGREEMENT_FLOOR_PERCENT: usize = 90;

fn predicted_class(logits: &[i32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn reference_class(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[test]
fn simd_dot_product_matches_scalar_oracle() {
    for &length in &[16usize, 32, 64, 128, 256] {
        let mut rng = Rng::new(0xC0DE + length as u32);
        let weights = AlignedI8::from_slice(&rng.fill_i8(length));
        let activations = AlignedI8::from_slice(&rng.fill_i8(length));

        assert_eq!(
            mac::dot_i8_scalar(weights.as_slice(), activations.as_slice()),
            mac::dot_i8_simd(weights.as_slice(), activations.as_slice()),
            "length {length}"
        );
    }
}

#[test]
fn simd_depthwise_matches_scalar_oracle() {
    let kernel = 25;
    let requantize = layers::Requantize {
        mult: 1,
        shift: 12,
        relu: true,
    };

    for &(steps, channels) in &[(32usize, 16usize), (64, 32), (125, 64), (250, 16)] {
        let mut rng = Rng::new(0xDEAD + steps as u32);
        let input = I8Activation::synthetic(steps, channels, 0xBEEF + steps as u32);
        let weights = AlignedI8::from_slice(&rng.fill_i8(kernel * channels));
        let bias = rng.fill_i32_small(channels);

        let mut padded = I8Activation::zeros(1, 1);
        let mut scalar_output = I8Activation::zeros(1, 1);
        let mut simd_output = I8Activation::zeros(1, 1);

        layers::depthwise_scalar(
            &input,
            weights.as_slice(),
            &bias,
            kernel,
            2,
            requantize,
            &mut padded,
            &mut scalar_output,
        );
        layers::depthwise_simd(
            &input,
            weights.as_slice(),
            &bias,
            kernel,
            2,
            requantize,
            &mut padded,
            &mut simd_output,
        );

        assert_eq!(
            scalar_output.as_slice(),
            simd_output.as_slice(),
            "steps {steps}, channels {channels}"
        );
    }
}

/// Top-1 says the model is still accurate; agreement says the int8 path still tracks
/// the float path it was quantized from. A bad quantization moves agreement first.
#[test]
fn device_forward_pass_matches_float_reference() {
    let mut model = Model::load(&FIXTURE_BLOB.0);
    let mut batch = VerifyBatch::new(&FIXTURE_BLOB.0);
    let total = batch.total;
    assert!(total > 0, "fixture blob carries no verification windows");

    let mut correct = 0usize;
    let mut agreements = 0usize;
    while let Some(window) = batch.next_window() {
        let ForwardResult::Logits(logits) = model.forward(&window.input);
        let predicted = predicted_class(&logits);
        if predicted == window.label as usize {
            correct += 1;
        }
        if predicted == reference_class(&window.float_logits) {
            agreements += 1;
        }
    }

    assert!(
        correct * 100 >= total * TOP1_FLOOR_PERCENT,
        "top-1 {correct}/{total}"
    );
    assert!(
        agreements * 100 >= total * AGREEMENT_FLOOR_PERCENT,
        "agreement {agreements}/{total}"
    );
}
