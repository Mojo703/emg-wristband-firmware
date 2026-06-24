# CLAUDE.md: waveformer

A retired accuracy-ceiling experiment (Rust/candle WaveFormer port), superseded by
`../emg-tds`; see `../engineering-logs/0008`. See `README.md` for the build
commands.

Agent notes:

- Use `../emg-tds` for new model work, not this.
- Don't delete `data/`. It is the canonical `.npy` export that `emg-tds` also reads
  via `--data-dir ../waveformer/data`.
