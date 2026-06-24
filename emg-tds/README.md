# emg-tds

A depthwise-separable (Time-Depth-Separable) conv encoder in Rust/candle for sEMG
gesture recognition, with swappable pose and classifier heads so a pose-pretrained
encoder can be head-swapped into a classifier. It replaces the retired WaveFormer
ceiling experiment; see
[`../engineering-logs/0008`](../engineering-logs/0008-waveformer-dead-end-emg2pose-and-tds.md)
for why. The [`dashboard`](../dashboard) reuses this crate's `Classifier` for
inference.

## Build and run

On CPU it builds anywhere:

```sh
cargo run -- forward-test
```

On GPU (CUDA 13.x), pin the cudarc ABI target, since auto-detect rejects 13.3:

```sh
CUDARC_CUDA_VERSION=13020 cargo build --release --features cuda
```

## Commands

```sh
# supervised classifier with the winning warp + channel-dropout, early-stopped
CUDARC_CUDA_VERSION=13020 ./target/release/emg-tds train \
  --data-dir ../waveformer/data --epochs 300 --augment \
  --out checkpoints/best.safetensors

# open-set: 5 commands plus grouped negatives, balanced, with the reject report
./target/release/emg-tds train --data-dir ../waveformer/data \
  --n-commands 5 --balance --augment

# pose-regression pretrain (per-dim z-scored targets), then head-swap finetune
./target/release/emg-tds pretrain --data-dir ../waveformer/data --out checkpoints/pose.safetensors
./target/release/emg-tds train --data-dir ../waveformer/data --init checkpoints/pose.safetensors
```

`forward-test` builds with random weights and runs one forward pass to check
shapes. `train` selects on held-out training subjects (`--val-subjects`, never the
test subjects), early-stops after `--patience` epochs with no gain, and saves the
best model to `--out`. `--augment` applies warp σ=0.3 and channel-dropout 0.1.
`--n-commands N` treats labels at or above N as grouped negatives and prints the
reject report. `pretrain` runs pose regression on emg2pose windows with
per-dimension z-scored targets; without the z-scoring the mean-pose error floored
out, as log 0007 records.

Weight-decay, eval frequency, and the early-stop delta are constants now, not CLI
flags; engineering-logs 0010 through 0013 record why.

## Architecture

Four depthwise-separable conv1d blocks make up the encoder. Each block runs a
depthwise temporal conv (k=25, which keeps the 16 channels separate), then a
pointwise 1×1 that mixes channels, then BatchNorm and ReLU, strided 2× per block
(16→32→64→128→128). A global average pool and a linear head follow, for about
0.04M params. The encoder is shared across tasks and only the head name differs
(`cls_head` or `pose_head`), so a finetune loads the encoder by name and skips the
wrong-task head.

Three choices carry the accuracy: BatchNorm rather than LayerNorm, large kernels,
and no early channel collapse. These lifted test accuracy from 45% on the
first TDS attempt to 80.7% on the cross-subject Hyser 5-class split, past the prior
CNN's 70%.

## Data and source

Data lives in [`../waveformer/data`](../waveformer): `train_x|y.npy` and
`test_x|y.npy` for classification, `pose_x|y.npy` for pretrain. The source is
`src/model.rs` for the encoder and heads, `src/data.rs` for
the loaders, `src/augment.rs` for the warp and channel-dropout, `src/lib.rs` for
the `Classifier` inference path, and `src/main.rs` for the CLI.
