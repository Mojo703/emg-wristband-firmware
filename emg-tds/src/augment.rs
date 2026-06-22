//! On-the-fly training augmentation that simulates inter-subject / placement
//! variability (engineering-logs/0010). Applied to the fit batch only.
//!
//! Only the two transforms that earned their place in the 0010 sweep survive:
//!
//! - **magnitude warp** (σ=0.3): multiply each channel by a smooth random gain
//!   curve (low-frequency, K knots linearly interpolated to T). Simulates electrode
//!   contact / amplitude differences across people — the dominant inter-subject
//!   factor in the electrode-shift literature.
//! - **channel dropout** (p=0.1): zero a channel at random (inverted-scale).
//!
//! Rotate (−15 pt), noise, time-warp, and mixup were tested and dropped (0010).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::Rng;

/// Winning augmentation combo from 0010/0013: warp σ=0.3 + channel-dropout 0.1.
#[derive(Clone)]
pub(crate) struct AugCfg {
    pub(crate) warp_sigma: f64,
    pub(crate) chan_dropout: f64,
    pub(crate) knots: usize,
}

impl AugCfg {
    /// The proven combo (`--augment`).
    pub(crate) fn on() -> Self {
        Self {
            warp_sigma: 0.3,
            chan_dropout: 0.1,
            knots: 5,
        }
    }

    /// No augmentation (`--augment` absent).
    pub(crate) fn off() -> Self {
        Self {
            warp_sigma: 0.0,
            chan_dropout: 0.0,
            knots: 5,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.warp_sigma > 0.0 || self.chan_dropout > 0.0
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
        if self.chan_dropout > 0.0 {
            parts.push(format!("cdrop{:.2}", self.chan_dropout));
        }
        parts.join("+")
    }
}

/// Fixed [K,T] linear-interpolation basis mapping K knots → T samples, so a warp
/// envelope is `knots[B*C,K] @ basis[K,T]`.
pub(crate) fn warp_basis(knots: usize, t: usize, device: &Device) -> Result<Tensor> {
    let mut w = vec![0f32; knots * t];
    for ti in 0..t {
        let p = if t > 1 {
            ti as f32 / (t - 1) as f32 * (knots - 1) as f32
        } else {
            0.0
        };
        let kl = p.floor() as usize;
        let frac = p - kl as f32;
        w[kl * t + ti] += 1.0 - frac;
        if kl + 1 < knots {
            w[(kl + 1) * t + ti] += frac;
        }
    }
    Ok(Tensor::from_vec(w, (knots, t), device)?)
}

/// Apply enabled transforms to xb [B,1,C,T]. `basis` is required iff warp is on.
pub(crate) fn apply(
    xb: &Tensor,
    cfg: &AugCfg,
    basis: Option<&Tensor>,
    _rng: &mut impl Rng,
    device: &Device,
) -> Result<Tensor> {
    let (b, _, c, t) = xb.dims4()?;
    let mut x = xb.clone();

    if cfg.warp_sigma > 0.0 {
        let basis = basis.expect("warp basis required");
        let k = basis.dim(0)?;
        // knots ~ N(0,1) per (sample, channel); envelope = knots @ basis.
        let knots = Tensor::randn(0f32, 1f32, (b * c, k), device)?;
        let env = knots.matmul(basis)?.reshape((b, 1, c, t))?;
        let gain =
            (env * cfg.warp_sigma)?.broadcast_add(&Tensor::ones((1,), DType::F32, device)?)?;
        x = x.mul(&gain)?;
    }

    if cfg.chan_dropout > 0.0 {
        let keep = 1.0 - cfg.chan_dropout;
        let r = Tensor::rand(0f32, 1f32, (b, 1, c, 1), device)?;
        let keep_t = Tensor::full(keep as f32, (1, 1, 1, 1), device)?;
        let mask = r.broadcast_lt(&keep_t)?.to_dtype(DType::F32)?;
        // inverted dropout: scale kept channels by 1/keep so the expected scale holds
        x = x.broadcast_mul(&mask)?.affine(1.0 / keep, 0.0)?;
    }

    Ok(x.contiguous()?)
}
