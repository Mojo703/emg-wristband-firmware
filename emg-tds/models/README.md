# models

Released, tracked model weights. Each file here is a curated release with a
descriptive, versioned name — not training scratch.

Training writes to `../checkpoints/` (gitignored, regenerable). When a run is good
enough to keep, copy it here under a name that says what it is, and bump the
version. Downstream defaults point at these files, so renaming one is a breaking
change.

| File | What it is |
|------|------------|
| `gesture-classifier-v1.safetensors` | The current TDS gesture classifier. Loaded by the dashboard at runtime (`EMG_CHECKPOINT` default) and the source the int8 device export quantizes. |
