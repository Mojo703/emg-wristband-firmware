//! On-the-fly training augmentation that simulates inter-subject / placement
//! variability (0010), applied to the fit batch only. Two transforms survived the
//! 0010 sweep:
//!
//! - magnitude warp (sigma=0.3): scale each channel by a smooth random gain curve
//!   (K knots interpolated to T), simulating the electrode contact / amplitude
//!   differences that dominate inter-subject variation.
//! - channel dropout (probability=0.1): zero a channel at random, inverted-scale.
//!
//! Rotate (−15 pt), noise, time-warp, and mixup were tested and dropped (0010).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::Rng;

/// Winning augmentation combo from 0010/0013: warp sigma=0.3 + channel-dropout 0.1.
#[derive(Clone)]
pub(crate) struct AugmentConfig {
    pub(crate) warp_sigma: f64,
    pub(crate) channel_dropout: f64,
    pub(crate) num_knots: usize,
}

impl AugmentConfig {
    /// The proven combo (`--augment`).
    pub(crate) fn on() -> Self {
        Self {
            warp_sigma: 0.3,
            channel_dropout: 0.1,
            num_knots: 5,
        }
    }

    /// No augmentation (`--augment` absent).
    pub(crate) fn off() -> Self {
        Self {
            warp_sigma: 0.0,
            channel_dropout: 0.0,
            num_knots: 5,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.warp_sigma > 0.0 || self.channel_dropout > 0.0
    }

    /// True if the warp transform (which needs the [K,T] knot basis) is on.
    pub(crate) fn needs_basis(&self) -> bool {
        self.warp_sigma > 0.0
    }

    pub(crate) fn describe(&self) -> String {
        if !self.enabled() {
            return "none".into();
        }
        let mut parts = Vec::new();
        if self.warp_sigma > 0.0 {
            parts.push(format!("warp{:.2}", self.warp_sigma));
        }
        if self.channel_dropout > 0.0 {
            parts.push(format!("cdrop{:.2}", self.channel_dropout));
        }
        parts.join("+")
    }
}

/// Fixed [K,T] linear-interpolation basis mapping K knots → T samples, so a warp
/// envelope is `knots[B*C,K] @ basis[K,T]`.
pub(crate) fn warp_basis(num_knots: usize, time: usize, device: &Device) -> Result<Tensor> {
    let mut weights = vec![0f32; num_knots * time];
    for time_index in 0..time {
        let position = if time > 1 {
            time_index as f32 / (time - 1) as f32 * (num_knots - 1) as f32
        } else {
            0.0
        };
        let lower_knot = position.floor() as usize;
        let fraction = position - lower_knot as f32;
        weights[lower_knot * time + time_index] += 1.0 - fraction;
        if lower_knot + 1 < num_knots {
            weights[(lower_knot + 1) * time + time_index] += fraction;
        }
    }
    Ok(Tensor::from_vec(weights, (num_knots, time), device)?)
}

/// Apply enabled transforms to `inputs` [B,1,C,T]. `basis` is required iff warp is on.
pub(crate) fn apply(
    inputs: &Tensor,
    config: &AugmentConfig,
    basis: Option<&Tensor>,
    _rng: &mut impl Rng,
    device: &Device,
) -> Result<Tensor> {
    let (batch_size, _, channels, time) = inputs.dims4()?;
    let mut output = inputs.clone();

    if config.warp_sigma > 0.0 {
        let basis = basis.expect("warp basis required");
        let num_knots = basis.dim(0)?;
        // knots ~ N(0,1) per (sample, channel); envelope = knots @ basis.
        let knot_values = Tensor::randn(0f32, 1f32, (batch_size * channels, num_knots), device)?;
        let envelope = knot_values.matmul(basis)?.reshape((batch_size, 1, channels, time))?;
        let gain =
            (envelope * config.warp_sigma)?.broadcast_add(&Tensor::ones((1,), DType::F32, device)?)?;
        output = output.mul(&gain)?;
    }

    if config.channel_dropout > 0.0 {
        let keep_probability = 1.0 - config.channel_dropout;
        let random = Tensor::rand(0f32, 1f32, (batch_size, 1, channels, 1), device)?;
        let keep_threshold = Tensor::full(keep_probability as f32, (1, 1, 1, 1), device)?;
        let mask = random.broadcast_lt(&keep_threshold)?.to_dtype(DType::F32)?;
        // inverted dropout: scale kept channels by 1/keep so the expected scale holds
        output = output.broadcast_mul(&mask)?.affine(1.0 / keep_probability, 0.0)?;
    }

    Ok(output.contiguous()?)
}
