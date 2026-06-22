# emg-tds

Depthwise-separable conv encoder (Rust/candle) for sEMG gesture recognition, with
swappable pose/classifier heads for pose-pretrain → head-swap transfer. Replaces
the retired WaveFormer ceiling experiment — see
[`engineering-logs/0008`](../engineering-logs/0008-waveformer-dead-end-emg2pose-and-tds.md)
for the why.

## Architecture

4 depthwise-separable conv1d blocks — depthwise temporal conv (k=25, keeps the 16
channels separate) → pointwise 1×1 (mixes channels) → BatchNorm → ReLU, strided
2× per block (16→32→64→128→128) — then global average pool → linear head. ~0.04M
params. The encoder is shared across tasks; only the head name differs
(`cls_head` / `pose_head`), so a finetune loads the encoder by name and skips the
wrong-task head.

This design — BatchNorm (not LayerNorm), large kernels, no early channel collapse
— is what lifted test accuracy from 45% (the first TDS attempt) to **80.7%** on
the cross-subject Hyser 5-class split, past the prior CNN's 70%.

## Build & run

CPU (builds anywhere): `cargo run -- forward-test`

GPU (CUDA 13.x): set the cudarc ABI pin (auto-detect rejects 13.3):

```
CUDARC_CUDA_VERSION=13020 cargo build --release --features cuda
```

Data is the `.npy` export from `emg-gesture-class` (same format as the old
`waveformer/data/`): `train_x|y.npy`, `test_x|y.npy` for classification,
`pose_x|y.npy` for pretrain.

```
# supervised classifier, best-checkpointed + early stopped
CUDARC_CUDA_VERSION=13020 ./target/release/emg-tds train \
  --data-dir ../waveformer/data --epochs 300 --lr 1e-3 \
  --patience 30 --out checkpoints/best.safetensors

# pose-regression pretrain (per-dim z-scored targets), then head-swap finetune
./target/release/emg-tds pretrain --data-dir ../waveformer/data --out checkpoints/pose.safetensors
./target/release/emg-tds train --data-dir ../waveformer/data --init checkpoints/pose.safetensors
```

`train` early-stops when test accuracy hasn't gained ≥`--min-delta` for
`--patience` evals, and saves the best model to `--out`.
