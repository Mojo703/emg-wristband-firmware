# 0006 — WaveFormer Rust/candle port: architecture + forward pass

**Date:** 2026-06-21
**Crate:** `EMG-Wristband/waveformer/` (candle 0.10). `cargo run -- forward-test`.

## Goal

Stand up a faithful, runnable WaveFormer **classification** path in Rust as the
accuracy-ceiling baseline ([0005](0005-sota-review-and-waveformer.md)), validated
end-to-end on random input before wiring data/training.

## Method

Cloned the reference (github.com/ForeverBlue816/WaveFormer) and ported the
supervised path module-for-module against `model.py` / `util/patch_embed.py`:

- **PatchEmbed** — reference uses a non-square `Conv2d` kernel/stride `(1,100)`;
  candle's conv2d is square-only, so implemented as `conv1d` over each channel
  row (reshape `(B,1,C,T)`→`(B·C,1,T)`, stride 100), then LayerNorm+GELU. Output
  `(B,D,C,T/100)`, identical to the reference.
- **WTConv2d** — learnable-wavelet depthwise block: forward DWT via grouped
  `conv2d` (Haar db1, 2×2, stride 2, groups=C); per-level depthwise 3×3 + scale;
  3-level decompose/reconstruct. The inverse DWT is depthwise grouped
  transposed-conv, which candle lacks → implemented via the transposed-conv ≡
  conv identity: **input dilation (zero-insertion) + grouped `conv2d` with a
  spatially-flipped kernel** (`dilate2d` + `reverse` helpers). Plus the DSConv
  1×1 pointwise channel mix.
- **RoPE attention** — `qkv`→split heads, rotate Q/K over the full head_dim
  (even/odd pairing, base 1e4), scaled dot-product softmax, proj. Matches
  `RoPEAttention`/`RotaryEmbedding`.
- **Block** — pre-norm: `x+attn(ln1(x))`, `x+mlp(ln2(x))`, mlp_ratio 1, LN eps 1e-6.
- **Head** — final LN → take cls → fc_norm → linear. embed 256, depth 6, heads 8.

## Measured

```
input  : [2, 1, 16, 500]
patches: 80 (+1 cls)        # 16 channels × 5 time-patches
logits : [2, 5]
params : 2.50M  (2.40M without wavelet)
forward-test OK
```

Shapes are correct end-to-end; both wavelet and no-wavelet variants run. **2.50M
params vs the paper's 3.1M** — the gap is the SSL projector/predictor (~0.5M,
1024-dim) + decoder, which are pretraining-only and intentionally omitted. The
classification encoder is faithfully sized.

## Faithfulness deviations (tracked, to revisit before claiming the ceiling)

1. **Wavelet filters are fixed Haar, not learnable.** The reference registers the
   dec/rec filters as `nn.Parameter(requires_grad=True)`. Here they're constant
   tensors. Fixed Haar is a sound starting wavelet; making them `Var`s is a
   one-line change once training exists. *Likely small effect.*
2. **High-frequency dropout omitted** (eval-equivalent; add for training).
3. **GELU**: used candle `gelu()` (tanh approx) vs timm's erf GELU — switch to
   `gelu_erf` for exactness. *Negligible.*
4. **pos_embed_x/y unused** — matches the reference's *released* classification
   `forward` (it relies on RoPE only; the learned pos-embeds are defined but not
   added in `forward_encoder_all_patches`). The reference also **comments out the
   wavelet** in that forward; we wire it in, since it's the paper's contribution
   and the source of the headline accuracy. Flagging the discrepancy explicitly.
5. Raw conv weights are randn-init here; proper xavier/trunc-normal init (per
   `initialize_weights`) to be added with the training stage.

## Next steps

- **0007 — data + training.** Wire the `emg-gesture-class` Hyser/Pinch loaders
  (or read the same windows), add xavier init + HF dropout + learnable wavelet
  Vars, AdamW + the reference schedule, and train the supervised ceiling.
  Report cross-subject and **per-user inter-session** accuracy vs the 0005 table
  (target: approach WaveFormer's ~82% DB6 / beat our 70%).
- **Numerical cross-check:** before trusting the ceiling, validate the Rust
  WTConv against the PyTorch reference on a fixed input (the reference repo is at
  `/tmp/WaveFormer`).
- Only after the ceiling is known: downscale to the ESP32-S3 budget and re-enter
  the latency loop (0001–0003).
