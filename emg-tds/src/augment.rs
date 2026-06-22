//! On-the-fly training augmentation that simulates inter-subject / placement
//! variability (engineering-logs/0010). Applied to the fit batch only.
//!
//! - **magnitude warp**: multiply each channel by a smooth random gain curve
//!   (low-frequency, K knots linearly interpolated to T). Simulates electrode
//!   contact / amplitude differences across people — the dominant inter-subject
//!   factor in the electrode-shift literature.
//! - **noise**: additive Gaussian.
//! - **rotate**: cyclically shift the 16 channels. Physically, the wristband worn
//!   rotated on the arm — a placement augmentation specific to a circular array.

use anyhow::Result;
use candle_core::{Device, Tensor, D};
use rand::Rng;

#[derive(Clone)]
pub struct AugCfg {
    pub warp_sigma: f64,
    pub noise_sigma: f64,
    pub rotate: bool,
    /// Time-warp strength (smooth random resampling of the time axis). 0 = off.
    pub time_warp_sigma: f64,
    /// Per-channel dropout probability (zero a channel, inverted-scale). 0 = off.
    pub chan_dropout: f64,
    pub knots: usize,
}

impl AugCfg {
    pub fn enabled(&self) -> bool {
        self.warp_sigma > 0.0
            || self.noise_sigma > 0.0
            || self.rotate
            || self.time_warp_sigma > 0.0
            || self.chan_dropout > 0.0
    }

    /// True if any enabled transform needs the [K,T] knot basis.
    pub fn needs_basis(&self) -> bool {
        self.warp_sigma > 0.0 || self.time_warp_sigma > 0.0
    }

    pub fn describe(&self) -> String {
        if !self.enabled() {
            return "none".into();
        }
        let mut parts = Vec::new();
        if self.warp_sigma > 0.0 {
            parts.push(format!("warp{:.2}", self.warp_sigma));
        }
        if self.noise_sigma > 0.0 {
            parts.push(format!("noise{:.2}", self.noise_sigma));
        }
        if self.rotate {
            parts.push("rotate".into());
        }
        if self.time_warp_sigma > 0.0 {
            parts.push(format!("twarp{:.2}", self.time_warp_sigma));
        }
        if self.chan_dropout > 0.0 {
            parts.push(format!("cdrop{:.2}", self.chan_dropout));
        }
        parts.join("+")
    }
}

/// Fixed [K,T] linear-interpolation basis mapping K knots → T samples, so a warp
/// envelope is `knots[B*C,K] @ basis[K,T]`.
pub fn warp_basis(knots: usize, t: usize, device: &Device) -> Result<Tensor> {
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
pub fn apply(
    xb: &Tensor,
    cfg: &AugCfg,
    basis: Option<&Tensor>,
    rng: &mut impl Rng,
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
        let gain = (env * cfg.warp_sigma)?.broadcast_add(&Tensor::ones((1,), candle_core::DType::F32, device)?)?;
        x = x.mul(&gain)?;
    }

    if cfg.noise_sigma > 0.0 {
        let noise = (Tensor::randn(0f32, 1f32, xb.shape(), device)? * cfg.noise_sigma)?;
        x = x.add(&noise)?;
    }

    if cfg.rotate {
        // one random cyclic channel shift for the whole batch (cheap; varies
        // across batches/epochs).
        let s = rng.gen_range(0..c);
        if s != 0 {
            let top = x.narrow(2, s, c - s)?;
            let bottom = x.narrow(2, 0, s)?;
            x = Tensor::cat(&[&top, &bottom], 2)?;
        }
    }

    if cfg.time_warp_sigma > 0.0 {
        let basis = basis.expect("warp basis required");
        let k = basis.dim(0)?;
        // Smooth positive speed curve over time; cumulative → monotonic warp path,
        // normalized to [0, T-1]. One warp per batch (cheap; varies across batches).
        let speeds = (Tensor::randn(0f32, 1f32, (1, k), device)? * cfg.time_warp_sigma)?.exp()?;
        let curve = speeds.matmul(basis)?.reshape((t,))?; // [T]
        let cum = curve.cumsum(0)?;
        let total = cum.narrow(0, t - 1, 1)?.to_vec1::<f32>()?[0];
        let pos: Vec<f32> = cum
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v / total * (t - 1) as f32)
            .collect();
        let mut lo = vec![0u32; t];
        let mut hi = vec![0u32; t];
        let mut frac = vec![0f32; t];
        for (i, &p) in pos.iter().enumerate() {
            let l = p.floor().clamp(0.0, (t - 1) as f32) as usize;
            lo[i] = l as u32;
            hi[i] = (l + 1).min(t - 1) as u32;
            frac[i] = p - l as f32;
        }
        let lo = Tensor::from_vec(lo, t, device)?;
        let hi = Tensor::from_vec(hi, t, device)?;
        let fr = Tensor::from_vec(frac, (1, 1, 1, t), device)?;
        let xl = x.index_select(&lo, 3)?;
        let xh = x.index_select(&hi, 3)?;
        let one_minus = fr.affine(-1.0, 1.0)?; // 1 - frac
        x = (xl.broadcast_mul(&one_minus)? + xh.broadcast_mul(&fr)?)?;
    }

    if cfg.chan_dropout > 0.0 {
        let keep = 1.0 - cfg.chan_dropout;
        let r = Tensor::rand(0f32, 1f32, (b, 1, c, 1), device)?;
        let keep_t = Tensor::full(keep as f32, (1, 1, 1, 1), device)?;
        let mask = r.broadcast_lt(&keep_t)?.to_dtype(candle_core::DType::F32)?;
        // inverted dropout: scale kept channels by 1/keep so the expected scale holds
        x = x.broadcast_mul(&mask)?.affine(1.0 / keep, 0.0)?;
    }

    let _ = D::Minus1;
    Ok(x.contiguous()?)
}
