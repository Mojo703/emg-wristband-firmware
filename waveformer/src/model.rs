//! Faithful Rust/candle port of WaveFormer's supervised classification path
//! (arXiv:2506.11168, github.com/ForeverBlue816/WaveFormer — `Waveformer_base`).
//!
//! Flow: input (B,1,C,T) → PatchEmbed (Conv kernel/stride = patch_size) →
//! WTConv2d learnable-wavelet block → flatten patches → prepend cls token →
//! depth× Transformer blocks with RoPE attention → LayerNorm → take cls →
//! fc_norm → linear head. The reference's decoder / contrastive / forecasting /
//! domain machinery is SSL-pretraining scaffolding and is intentionally omitted.
//!
//! candle note: conv_transpose2d has no `groups`, so WTConv's depthwise inverse
//! wavelet transform is done as input-dilation + grouped conv2d with a spatially
//! flipped kernel (the standard transposed-conv ≡ conv identity).

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

#[derive(Clone, Copy)]
pub struct Config {
    pub input_channels: usize, // C (EMG electrodes)
    pub time_steps: usize,     // T (window length)
    pub patch_h: usize,        // patch_size[0]
    pub patch_w: usize,        // patch_size[1]
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub wt_levels: usize,
    pub out_dim: usize,
    /// Head parameter name. Use distinct names ("head" for classification,
    /// "pose_head" for pose pretraining) so a checkpoint's encoder weights load
    /// by matching name while the wrong-task head is simply skipped.
    pub head_name: &'static str,
    pub use_wavelet: bool,
}

impl Config {
    /// `Waveformer_base` for 16-channel, 500-sample EMG windows.
    pub fn base_emg(input_channels: usize, time_steps: usize, num_classes: usize) -> Self {
        Self {
            input_channels,
            time_steps,
            patch_h: 1,
            patch_w: 100,
            embed_dim: 256,
            depth: 6,
            num_heads: 8,
            mlp_ratio: 1,
            wt_levels: 3,
            out_dim: num_classes,
            head_name: "head",
            use_wavelet: true,
        }
    }

    /// Pose-MSE pretraining variant: 20-d regression head, distinct name.
    pub fn pose_pretrain(input_channels: usize, time_steps: usize) -> Self {
        let mut c = Self::base_emg(input_channels, time_steps, 20);
        c.head_name = "pose_head";
        c
    }
}

// ---------------------------------------------------------------------------
// small tensor helpers
// ---------------------------------------------------------------------------

/// Reverse a tensor along `dim` (used to spatially flip conv kernels).
fn reverse(t: &Tensor, dim: usize) -> Result<Tensor> {
    let n = t.dim(dim)?;
    let idx: Vec<u32> = (0..n as u32).rev().collect();
    let idx = Tensor::from_vec(idx, n, t.device())?;
    t.index_select(&idx, dim)
}

/// Insert `factor-1` zeros between spatial elements (dims 2,3): h→factor*h-(factor-1).
/// Implemented by interleaving zeros via stack+reshape, then trimming the trailing zero.
fn dilate2d(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor == 1 {
        return Ok(x.clone());
    }
    let mut out = x.clone();
    for dim in [2usize, 3usize] {
        let (b, c, h, w) = out.dims4()?;
        let z = out.zeros_like()?;
        // stack [out, zeros*(factor-1)] along a new axis after `dim`, then merge.
        let mut parts = vec![out.unsqueeze(dim + 1)?];
        for _ in 1..factor {
            parts.push(z.unsqueeze(dim + 1)?);
        }
        let stacked = Tensor::cat(&parts, dim + 1)?; // shape with size `factor` at dim+1
                                                     // merge dim and dim+1
        let new = if dim == 2 {
            stacked.reshape((b, c, h * factor, w))?
        } else {
            stacked.reshape((b, c, h, w * factor))?
        };
        // trim the trailing inserted zeros: keep factor*size-(factor-1)
        let keep = new.dim(dim)? - (factor - 1);
        out = new.narrow(dim, 0, keep)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// WTConv2d — learnable wavelet, multi-level, depthwise
// ---------------------------------------------------------------------------

struct ScaleModule {
    weight: Tensor, // [1, ch, 1, 1]
}
impl ScaleModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.weight)
    }
}

struct WTConv2d {
    levels: usize,
    wt_filter: Tensor,  // [4C,1,2,2] decomposition (depthwise, groups=C)
    iwt_filter: Tensor, // [4C,1,2,2] reconstruction
    base_w: Tensor,     // [C,1,3,3] depthwise base conv
    base_scale: ScaleModule,
    wconv_w: Vec<Tensor>, // each [4C,1,3,3] depthwise over the 4 subbands
    wscale: Vec<ScaleModule>,
}

impl WTConv2d {
    fn new(c: usize, levels: usize, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        // Haar (db1) filters, kernel 2x2, tiled per channel → [4C,1,2,2].
        // Kept as fixed tensors for now (the reference makes them learnable;
        // see engineering log — a faithful follow-up will register them as Vars).
        let wt_filter = haar_filter(c, false, &dev, vb.dtype())?;
        let iwt_filter = haar_filter(c, true, &dev, vb.dtype())?;
        let randn = candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 };
        let base_w = vb.get_with_hints((c, 1, 3, 3), "base_conv.weight", randn)?;
        let base_scale = ScaleModule {
            weight: vb.get_with_hints((1, c, 1, 1), "base_scale", candle_nn::Init::Const(1.0))?,
        };
        let mut wconv_w = Vec::new();
        let mut wscale = Vec::new();
        for l in 0..levels {
            wconv_w.push(vb.get_with_hints((4 * c, 1, 3, 3), &format!("wavelet_convs.{l}.weight"), randn)?);
            wscale.push(ScaleModule {
                weight: vb.get_with_hints(
                    (1, 4 * c, 1, 1),
                    &format!("wavelet_scale.{l}"),
                    candle_nn::Init::Const(0.1),
                )?,
            });
        }
        Ok(Self {
            levels,
            wt_filter,
            iwt_filter,
            base_w,
            base_scale,
            wconv_w,
            wscale,
        })
    }

    // depthwise forward wavelet transform: [B,C,H,W] → [B,C,4,H',W']
    fn wt(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, _h, _w) = x.dims4()?;
        let out = x.conv2d(&self.wt_filter, 1, 2, 1, c)?; // pad=1, stride=2, groups=C
        let (_, _, nh, nw) = out.dims4()?;
        out.reshape((b, c, 4, nh, nw))
    }

    // depthwise inverse wavelet transform: [B,C,4,h,w] → [B,C,H,W]
    fn iwt(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let (b, c, h, w) = (dims[0], dims[1], dims[3], dims[4]);
        let inp = x.reshape((b, c * 4, h, w))?; // [B,4C,h,w], ordering [c*4+s]
                                                // transposed conv (stride 2, pad 1) via dilation + grouped conv2d:
        let dil = dilate2d(&inp, 2)?; // [B,4C,2h-1,2w-1]
                                      // weight [C,4,2,2] = reshape(iwt_filter) then spatial-flip
        let w_conv = self.iwt_filter.reshape((c, 4, 2, 2))?;
        let w_conv = reverse(&reverse(&w_conv, 2)?, 3)?;
        // grouped conv2d: groups=C, in=4C, out=C, pad = k-1-p = 0
        dil.conv2d(&w_conv, 0, 1, 1, c)
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let _ = train; // high-freq dropout omitted (eval-equivalent); add for training later
        let mut x_ll_levels: Vec<Tensor> = Vec::new();
        let mut x_h_levels: Vec<Tensor> = Vec::new();
        let mut shapes: Vec<(usize, usize)> = Vec::new();

        let mut curr = x.clone();
        for l in 0..self.levels {
            let (b, c, h, w) = curr.dims4()?;
            shapes.push((h, w));
            // pad to even
            let pad_h = h % 2;
            let pad_w = w % 2;
            if pad_h != 0 {
                curr = curr.pad_with_zeros(2, 0, pad_h)?;
            }
            if pad_w != 0 {
                curr = curr.pad_with_zeros(3, 0, pad_w)?;
            }
            let dec = self.wt(&curr)?; // [B,C,4,h',w']
            let dd = dec.dims().to_vec();
            let (hh, ww) = (dd[3], dd[4]);
            let flat = dec.reshape((b, c * 4, hh, ww))?;
            let flat = flat.conv2d(&self.wconv_w[l], 1, 1, 1, c * 4)?; // depthwise k3 pad1
            let flat = self.wscale[l].forward(&flat)?;
            let dec = flat.reshape((b, c, 4, hh, ww))?;
            // split LL / HF
            let lf = dec.narrow(2, 0, 1)?; // [B,C,1,h',w']
            let hf = dec.narrow(2, 1, 3)?; // [B,C,3,h',w']
            curr = lf.squeeze(2)?; // next level operates on LL
            x_ll_levels.push(curr.clone());
            x_h_levels.push(hf);
        }

        // inverse, deepest first
        let mut next_ll: Option<Tensor> = None;
        for l in (0..self.levels).rev() {
            let ll = match &next_ll {
                Some(n) => (&x_ll_levels[l] + n)?,
                None => x_ll_levels[l].clone(),
            };
            let cat4 = Tensor::cat(&[ll.unsqueeze(2)?, x_h_levels[l].clone()], 2)?;
            let recon = self.iwt(&cat4)?;
            let (ho, wo) = shapes[l];
            let recon = recon.narrow(2, 0, ho)?.narrow(3, 0, wo)?;
            next_ll = Some(recon);
        }
        let x_tag = next_ll.unwrap();

        let (_, c, _, _) = x.dims4()?;
        let x_base = x.conv2d(&self.base_w, 1, 1, 1, c)?; // depthwise k3 pad1
        let x_base = self.base_scale.forward(&x_base)?;
        x_base + x_tag
    }
}

/// Haar (db1) 2D filters [4,2,2] (LL,LH,HL,HH) tiled to [4C,1,2,2]. `inverse`
/// uses the reconstruction (time-reversed) variant; for orthonormal Haar the
/// reconstruction filters equal the (flipped) decomposition filters.
fn haar_filter(c: usize, _inverse: bool, dev: &Device, dtype: DType) -> Result<Tensor> {
    let s = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)
    let lo = [s, s];
    let hi = [-s, s];
    let outer = |a: [f32; 2], b: [f32; 2]| {
        // a along rows (kh), b along cols (kw): f[i][j] = a[i]*b[j]
        vec![a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    };
    let mut data = Vec::with_capacity(4 * 2 * 2);
    for f in [outer(lo, lo), outer(lo, hi), outer(hi, lo), outer(hi, hi)] {
        data.extend_from_slice(&f);
    }
    let base = Tensor::from_vec(data, (4, 1, 2, 2), dev)?.to_dtype(dtype)?;
    // tile per channel → [4C,1,2,2]
    let tiled = base.broadcast_as((c, 4, 1, 2, 2))?.reshape((4 * c, 1, 2, 2))?;
    tiled.contiguous()
}

// ---------------------------------------------------------------------------
// RoPE attention
// ---------------------------------------------------------------------------

/// Apply rotary embedding to the last dim of q/k shaped [B,H,N,Dr] (Dr even).
fn apply_rope(x: &Tensor, base: f64) -> Result<Tensor> {
    let (_b, _h, n, d) = x.dims4()?;
    let dev = x.device();
    let half = d / 2;
    let pos: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let inv: Vec<f32> = (0..half)
        .map(|i| 1f32 / (base as f32).powf((2 * i) as f32 / d as f32))
        .collect();
    let pos = Tensor::from_vec(pos, (n, 1), dev)?;
    let inv = Tensor::from_vec(inv, (1, half), dev)?;
    let freqs = pos.matmul(&inv)?; // [N, half]
    let cos = freqs.cos()?.reshape((1, 1, n, half))?.to_dtype(x.dtype())?;
    let sin = freqs.sin()?.reshape((1, 1, n, half))?.to_dtype(x.dtype())?;
    // split even/odd over last dim: reshape [...,half,2]
    let xr = x.reshape((x.dim(0)?, x.dim(1)?, n, half, 2))?;
    let x_even = xr.narrow(4, 0, 1)?.squeeze(4)?; // [B,H,N,half]
    let x_odd = xr.narrow(4, 1, 1)?.squeeze(4)?;
    let out_even = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
    let out_odd = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;
    // interleave back
    let stacked = Tensor::stack(&[out_even, out_odd], 4)?; // [B,H,N,half,2]
    stacked.reshape((x.dim(0)?, x.dim(1)?, n, d))
}

struct RoPEAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}
impl RoPEAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // [B,N,3C]
        let qkv = qkv
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?; // [3,B,H,N,hd]
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let q = apply_rope(&q, 10000.0)?;
        let k = apply_rope(&k, 10000.0)?;
        let attn = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * self.scale)?;
        let attn = softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?; // [B,H,N,hd]
        let out = out.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}
impl Mlp {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, dim, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: RoPEAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}
impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        Ok(Self {
            norm1: layer_norm(d, 1e-6, vb.pp("norm1"))?,
            attn: RoPEAttention::new(d, cfg.num_heads, vb.pp("attn"))?,
            norm2: layer_norm(d, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(d, d * cfg.mlp_ratio, vb.pp("mlp"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?)?)?;
        &x + self.mlp.forward(&self.norm2.forward(&x)?)?
    }
}

// ---------------------------------------------------------------------------
// WaveFormer
// ---------------------------------------------------------------------------

pub struct WaveFormer {
    cfg: Config,
    patch_w: Tensor, // conv1d weight [embed_dim, 1, patch_w]
    patch_b: Tensor, // [embed_dim]
    patch_norm: LayerNorm,
    wtconv: Option<WTConv2d>,
    wt_pointwise: Option<Tensor>, // DSConv 1x1: [D,D,1,1]
    cls_token: Tensor,            // [1,1,D]
    blocks: Vec<Block>,
    norm: LayerNorm,
    fc_norm: LayerNorm,
    head: Linear,
}

impl WaveFormer {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embed_dim;
        assert_eq!(cfg.patch_h, 1, "this port assumes patch height 1");
        // IMPORTANT: vb.get() defaults to Init::Const(0.) in candle — a zero patch
        // conv makes every patch embedding input-independent (model collapses to
        // predicting the mean). Init with fan-in scaled noise (~xavier; fan_in =
        // 1*patch_w). Bias may stay zero.
        let patch_w = vb.get_with_hints(
            (d, 1, cfg.patch_w),
            "patch_embed.proj.weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (cfg.patch_w as f64).sqrt(),
            },
        )?;
        let patch_b = vb.get_with_hints((d,), "patch_embed.proj.bias", candle_nn::Init::Const(0.0))?;
        let patch_norm = layer_norm(d, 1e-5, vb.pp("patch_embed.norm"))?;
        let (wtconv, wt_pointwise) = if cfg.use_wavelet {
            let dw = WTConv2d::new(d, cfg.wt_levels, vb.pp("wavelet_conv.depthwise"))?;
            // DSConvWT pointwise: 1x1 conv [D,D,1,1] mixing channels (groups=1).
            let pw = vb.get_with_hints(
                (d, d, 1, 1),
                "wavelet_conv.pointwise.weight",
                candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 },
            )?;
            (Some(dw), Some(pw))
        } else {
            (None, None)
        };
        let cls_token =
            vb.get_with_hints((1, 1, d), "cls_token", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 })?;
        let mut blocks = Vec::new();
        for i in 0..cfg.depth {
            blocks.push(Block::new(&cfg, vb.pp(format!("blocks.{i}")))?);
        }
        let norm = layer_norm(d, 1e-6, vb.pp("norm"))?;
        let fc_norm = layer_norm(d, 1e-6, vb.pp("fc_norm"))?;
        let head = linear(d, cfg.out_dim, vb.pp(cfg.head_name))?;
        Ok(Self {
            cfg,
            patch_w,
            patch_b,
            patch_norm,
            wtconv,
            wt_pointwise,
            cls_token,
            blocks,
            norm,
            fc_norm,
            head,
        })
    }

    // PatchEmbed via conv1d over each channel row (kernel/stride = patch_w).
    // x: (B,1,C,T) → (B, D, C, T/patch_w)
    fn patch_embed(&self, x: &Tensor) -> Result<Tensor> {
        let (b, one, c, t) = x.dims4()?;
        assert_eq!(one, 1);
        let rows = x.reshape((b * c, 1, t))?; // each channel-row as 1d signal
        let pw = self.cfg.patch_w;
        let emb = rows.conv1d(&self.patch_w, 0, pw, 1, 1)?; // [B*C, D, T/pw]
        let emb = emb.broadcast_add(&self.patch_b.reshape((1, self.cfg.embed_dim, 1))?)?;
        let wp = t / pw;
        let d = self.cfg.embed_dim;
        // → (B, C, D, Wp) → (B, D, C, Wp)
        let emb = emb.reshape((b, c, d, wp))?.permute((0, 2, 1, 3))?;
        // PatchEmbed applies LayerNorm over D then GELU (channels-last)
        let emb = emb.permute((0, 2, 3, 1))?.contiguous()?; // (B,C,Wp,D)
        let emb = self.patch_norm.forward(&emb)?.gelu()?;
        emb.permute((0, 3, 1, 2))?.contiguous() // (B,D,C,Wp)
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut feat = self.patch_embed(x)?; // (B,D,C,Wp)
        if let Some(wt) = &self.wtconv {
            feat = wt.forward(&feat, train)?;
            if let Some(pw) = &self.wt_pointwise {
                feat = feat.conv2d(pw, 0, 1, 1, 1)?; // 1x1 channel mix
            }
        }
        let (b, d, h, w) = feat.dims4()?;
        // flatten patches → (B, N, D)
        let tokens = feat.reshape((b, d, h * w))?.transpose(1, 2)?.contiguous()?;
        let cls = self.cls_token.broadcast_as((b, 1, d))?;
        let mut x = Tensor::cat(&[cls, tokens], 1)?; // (B,1+N,D)
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        x = self.norm.forward(&x)?;
        let cls_feat = x.narrow(1, 0, 1)?.squeeze(1)?; // (B,D)
        let cls_feat = self.fc_norm.forward(&cls_feat)?;
        self.head.forward(&cls_feat)
    }

    /// Number of patch tokens (excluding cls) for the configured input.
    pub fn num_patches(&self) -> usize {
        self.cfg.input_channels * (self.cfg.time_steps / self.cfg.patch_w)
    }
}
