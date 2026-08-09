# emg-runtime

The on-device inference path for the wristband: int8 kernels for the `emg-tds`
classifier with hand-written ESP32-S3 SIMD and a scalar fallback off target, the
model-free reject pipeline that turns one window's logits into a decision, and the
grid aligner that puts the two ADS1298s' independently clocked sample streams on
one time base. `opal-firmware` is the consumer.

The crate is `no_std` plus `alloc` and depends on nothing but
[`protocol`](../protocol) and `libm`, so it carries no ESP dependency and builds on
the host as readily as on Xtensa. That is deliberate: the same code that runs on the
device can be run, tested, and compared against a reference on a laptop.

`cargo test` runs the host suite. Off Xtensa the SIMD entry points resolve to the
scalar oracle, so their comparison would prove nothing. The separate
`emg-runtime-esp32s3-tests` package runs those checks on the device with
`cargo test-device`.

## The model

`src/model.rs` implements one architecture and only one: the four-block
depthwise-separable one-dimensional convolutional encoder `emg-tds` trains, with
BatchNorm folded in, ReLU, channels 16 → 32 → 64 → 128 → 128, a global average pool
over time, and a linear head to five classes. Each block is a strided depthwise
convolution followed by a pointwise (1×1) convolution; accumulation is `i32` and
each stage requantizes back to int8 through a fixed-point multiply and shift taken
from the blob.

The weights are not compiled in. `Model::load` takes a byte slice — the blob
`emg-tds export-int8` writes — and reads the architecture out of its header, so the
crate stays blob-agnostic and the firmware decides where the bytes come from. The
blob currently in `data/` describes 500 time steps of 16 channels (250 ms at the
front end's 2000 Hz), kernel 25, stride 2, four blocks, five classes.

Two things about `load` are worth knowing before touching it.

It borrows rather than copies. The weight tensors and biases are slices into the
caller's blob, which on the device is memory-mapped flash that the SIMD kernels can
read directly. The thirty-odd kilobytes are worth more as heap headroom than as
RAM-speed operands. The consequence is a contract: the blob must be 16-byte aligned
as a whole, which is why the firmware wraps `include_bytes!` in an
`#[repr(align(16))]` struct, and the exporter pads every section to the alignment
`load` asserts.

It allocates the activation buffers once. A forward pass runs through three scratch
tensors sized for their worst case across the block walk and reused every inference,
because the device runs an inference four times a second and repeated
multi-kilobyte allocations fragment a heap whose free margin is already thin. An
allocation failure mid-forward is an abort with no console.

`forward` returns raw `i32` logits. Two scale factors come off the blob for the
caller: `input_scale`, the normalised units per int8 count at which a window must be
quantized, and `logit_scale`, which converts the returned logits to the float scale
the softmax expects.

## The kernels, and the oracle they are checked against

`src/mac.rs` and `src/layers.rs` hold the arithmetic, each hot kernel written twice.

The pointwise convolutions, the head, and everything else MAC-heavy funnel through
one `i8`·`i8` → `i32` dot product. On Xtensa it uses the ESP32-S3's accumulator
`ACCX`: `ee.vmulas.s8.accx` multiplies sixteen int8 lanes and adds the sum of the
products straight into one 40-bit accumulator, so a dot product is a load-and-multiply
loop and a single register read at the end, with no lane reduction. The loop is
software-pipelined through the fused load-and-multiply form. The depthwise
convolutions instead use `QACC`, sixteen independent 20-bit accumulators, which
processes sixteen channels per vector instruction.

Off Xtensa the same names — `mac::dot_i8`, `layers::depthwise` — resolve to plain
scalar Rust. That fallback is not a courtesy for host builds; `dot_i8_scalar` and
`depthwise_scalar` are the correctness oracle. The device tests generate random
weights and activations and assert the SIMD result is bit-exact against the scalar
one, then run the whole forward pass over the exported verification windows and hold
it to a top-1 floor and to agreement with the float reference it was quantized from.

Every SIMD entry point requires 16-byte-aligned operands whose length is a whole
number of sixteen-lane vectors. `src/tensor.rs` is what guarantees it: `AlignedI8`
owns int8 storage on a 16-aligned base, zero-padded up to a multiple of sixteen
bytes, and because every channel count in this model is itself a multiple of
sixteen, each row of an activation is aligned too. The asserts in `Model::load` catch
an unpadded or misaligned blob at load time rather than as an alignment fault deep
inside a vector load.

## The reject pipeline

`src/pipeline.rs` is the decision spine, and it is model-free — no weights, no
learned threshold — which is why it sits beside the inference rather than inside it.

`softmax` normalises one window's logits. `RejectPipeline::step` then takes the
softmax over the command classes and reads two things out of it: the argmax, and the
reject score, which is the largest command probability. A window whose reject score
clears the threshold `tau` extends a streak for that command; a window below the
threshold, or one whose argmax is a different command, resets it. The command
latches once the streak reaches three consecutive windows holding the same argmax
(`RejectPipeline::NEEDED`), and the wake state follows from the streak: `Idle` at
zero, `Arming` on the way up, `Active` once latched.

Three of three is a decision spine, not an accuracy patch. It exists to keep a
single confident-looking window from firing a media key, and the cost — the latency
of two further windows — is paid deliberately. `tau` is public and the firmware
moves it when the dashboard changes the sensitivity preset.

## The grid aligner

`src/alignment.rs` solves a problem that has nothing to do with the model, but
belongs to the same host-testable core. The two ADS1298s convert on their own
internal oscillators, so their samples tick at almost the same rate and slip
continuously in phase, and anything that wants one sixteen-channel time series has
to decide which of chip A's samples goes with which of chip B's.

`GridAligner` makes that decision explicit. It lays a fixed grid at the nominal rate
on the device clock, anchored on the first accepted frame, and for each tick takes
from each present source the queued frame nearest that tick, if one falls inside the
acceptance window. What it could not pair, it counts: `surplus_dropped` when a frame
is skipped because a later one sat at least as near (that source's clock runs fast),
`duplicated` when a frame serves two ticks (it runs slow), and `missing` when
nothing lands in the window at all. Those counters are the measured oscillator
error rather than an assumed one, and the firmware reports them as telemetry.

Two behaviours are load-bearing for the consumer. A source must be announced present
before it can contribute and marked absent promptly when it stops, because the grid
waits on every present source; marking a source absent also clears its queue, since
frames from after a reset describe a different epoch. And when every present source
is in a gap, a short outage is emitted with empty slots — a couple of missed reads
must not void the 250 ms window being built around them — while a sustained one has
its ticks skipped and counted instead, so the consumer sees an honest discontinuity
rather than manufactured all-zero data.

## `data/`

| File | What it is |
|------|------------|
| `model_int8.bin` | The shipping blob, about 36 KB. `opal-firmware` embeds it with `include_bytes!` behind a 16-byte-aligned wrapper. |
| `model_int8_verify.bin` | The same blob with a balanced batch of 32 labeled test windows and their float logits appended, about 286 KB. It links only into the device test binary, where `VerifyBatch` streams the windows one at a time so the batch never has to fit in the device's heap. |

Both are written by `emg-tds export-int8`, which targets these paths by default.
They are checked in, so a fresh clone flashes a working model without a training
run.
