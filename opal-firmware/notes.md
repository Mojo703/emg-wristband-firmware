# Notes:

Overall, I don't like the overuse of pub. Many of them can be replaced with pub(crate), to limit visibility. And in most of those cases, we are breaking encapsulation.

Make sure to fmt all of the files before working.

Rust analyzer in vscode does not work right now.

We assume the error type in many locations: converting an esp-idf error into our own anyhow str error. We should match to make sure our assumptions don't change. And it they do, we have a compiler error showing the problem location.

## config.rs

CompileConfig should have Options, not sentinal values

SENSITIVITY_LEVELS should be an enum.

Settings.keymap should probably not be a Vec. Array instead. We want to avoid allocations.

Settings.defaults should be the Default trait instead.

Settings.to_wire should use an Into or From trait instead.

default_keymap should be Default on some struct.

Store should probably be its own mod file.


## frames.rs

I don't like the loose fns: emg, prediction, events, key_name. They should be methods on relevant structs.

## logger.rs

I don't like the static BUFFER. I understand (but don't prefer) the static LOGGER.

Why does flush exist if it doesn't do anything.

Why do we have two sources for log level??


## main.rs

To reduce OTA size, we should consider moving the model out of the firmware (so no include_bytes).

I don't like the static config values.

The fn main is way too long. We need to factor logic out into sane units.

I don't like the current tangled state logic for the transports: serial_claim, fresh_claim, etc. It must be simplified.

I don't like that we deal with the transports in the main loop. This should be encapsulated completely somewhere else. Especially handling the different Controls in different ways!

I don't like the annouce fn.

We probably want to support button and status LED, so our current blocking system is inconvenient.

I don't like that you used clippy allow. This is a mark of awful code.

Why is softmax defined in main.rs as a loose fn? Why are we passing around logits as a primitive type? I also don't like the Vec usage.

Why is reset_reason defined in main.rs as a loose fn?

> Important! justify why model needs to use the heap and take ownership of the layer data (Vecs vs slices). Why can't we just use references to the existing model in flash/memory? We have full control over the model file, so we can massage it to make it more convenient.

## provider.rs

Ok for now. We will soon have a basic driver for the ADC.

## transport.rs

The transport impl should probably decide which controls its meant to convey.

We might want to turn this into: ./transport/{mod.rs, wifi.rs, serial.rs} instead of the 'comment namespace' that we are doing right now.

I don't like how many loose fns the tcp section has.


## wifi.rs

I don't like how many loose fns this mod has.

We assume the error type. Use a match statement so if error variants are ever added, we can see compile issues (GOOD!).




