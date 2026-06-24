# CLAUDE.md: emg-tds

The current gesture model: a depthwise-separable (TDS) conv encoder in Rust/candle
with swappable classifier and pose heads. See `README.md` for build commands, the
subcommands, and the architecture. The `dashboard` reuses this crate's
`Classifier`.

Agent notes:

- Don't relitigate the logs. The architecture, the augmentation choice,
  calibration, and the negative-class handling were each tested and settled in
  `../engineering-logs/0008` through `0013`. Training constants (weight-decay, eval
  frequency, early-stop delta) are constants rather than flags for that reason.
  Read the relevant log before changing any of them.
- No temporal voting in the model. It was a crutch that ruined a past solution; the
  decision smoothing belongs in the reject pipeline (`dashboard`/firmware), not
  here.
- The encoder is shared by head name (`cls_head` / `pose_head`): a finetune loads
  the encoder by name and skips the wrong-task head. Keep that naming intact or
  transfer breaks.
