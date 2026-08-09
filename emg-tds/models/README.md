# models

Released, tracked model weights. Each file here has a descriptive, versioned
name. Training scratch does not belong here.

Training writes to `../checkpoints/` (gitignored, regenerable). Promote a useful
run under a descriptive name, then bump the version. The scoring and export
command defaults point at these files. Update those defaults when renaming a
model.

| File | What it is |
|------|------------|
| `gesture-classifier-v1.safetensors` | The current TDS gesture classifier. The host scoring commands load it by default, and the int8 device export quantizes it. |
