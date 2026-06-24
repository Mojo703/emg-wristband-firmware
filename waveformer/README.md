# waveformer

> Retired. This was an accuracy-ceiling experiment, a Rust/candle port of
> WaveFormer. It was shelved when the depthwise-separable encoder in
> [`../emg-tds`](../emg-tds) beat it. See
> [`../engineering-logs/0008`](../engineering-logs/0008-waveformer-dead-end-emg2pose-and-tds.md)
> for the post-mortem. Use `emg-tds` for new model work.

It is kept for two reasons. The build still documents the WaveFormer port, and its
`data/` directory holds the canonical `.npy` dataset that `emg-tds` also consumes
(`--data-dir ../waveformer/data`), so do not delete `data/`. That export is
`train_x|y.npy` and `test_x|y.npy` for classification, and `pose_x|y.npy` and
`pose_seq.npy` for pose pretrain, the same layout `emg-tds` reads.

```sh
cargo run -- forward-test                                        # CPU shape check
CUDARC_CUDA_VERSION=13020 cargo build --release --features cuda  # GPU (CUDA 13.x)
```

`CUDARC_CUDA_VERSION=13020` pins the cudarc ABI target; CUDA 13.3 is compatible but
auto-detect rejects it. The subcommands are `forward-test` (one forward pass on
random weights) and `train` (supervised training on the exported Hyser windows).
`src/model.rs` is the architecture, and `src/data.rs` the `.npy` loaders.
