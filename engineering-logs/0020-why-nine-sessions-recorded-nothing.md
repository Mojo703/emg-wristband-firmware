# 0020 — Why nine sessions recorded nothing

**Date:** 2026-08-04
**Crates:** `dashboard` (`session_report`, `signal_quality`), `opal-firmware` (`adc::ads1298`).
**Hardware:** rev A bodged board, both ADS1298s, subject bias drive OFF. Board files
read from the KiCad project in `~/Downloads/ADC PCB stuff/`.

## Purpose

[0019](0019-what-the-band-actually-records.md) settled what three recordings
contained. Nine now exist, and every one of them comes back `NOT USABLE for
training`. This entry settles why, separates the sessions that failed for
electrical reasons from the ones that failed for procedural reasons, and traces
the bias drive to the node that breaks it. It also retires two claims from 0019's
next steps that six more sessions do not support.

## Method

`session_report` over all nine directories in `dashboard/sessions/`, on one build,
reading per-channel floors, mains, offsets, at-rail fractions, label integrity and
the cue-locked response grid. The floor is interharmonic band power over
20–450 Hz. The mains fundamental is measured rather than assumed, and every bin
within 12 Hz of a harmonic is discarded, so mains and its sidebands sit outside
the number.

The board side is a netlist exported with `kicad-cli sch export netlist`, read
pin by pin rather than from the schematic image.

## Measured

Duration is EMG samples. Floors are per-chip medians over live channels.

| Session | Dur s | chip0 live / floor µV | chip1 live / floor µV | Railed | Mains µV | Cues | Schedule |
|---|---|---|---|---|---|---|---|
| 11-27-57 Matthew | 129 | 2/8, 72.9 | 7/8, 27.6 | 0,2,3,4,5,6,9 | 212–2772 | 47 | matches |
| 12-10-38 Matthew | 7 | 8/8, 36.2 | 7/8, 16.6 | 9 | 98–4352 | 0 | no anchor |
| 14-22-07 Matthew | 236 | 8/8, 219.1 | 7/8, 123.4 | 9 | 500–5896 | 84 | matches |
| 17-20-21 Matthew | 59 | 8/8, 109.2 | 7/8, 38.4 | 9 | 202–3728 | 17 | disagrees |
| 17-46-47 Matthew | 229 | 8/8, 64.7 | 7/8, 37.2 | 9 | 290–2666 | 80 | matches |
| 19-53-37 Cole | 89 | 5/8, 5.6 | 8/8, 6.4 | 4,5,7 | 6–16 | 30 | disagrees |
| 21-03-56 Matthew | 99 | 4/8, 37.0 | 7/8, 41.3 | 1,2,3,7,11 | 292–1591 | 36 | matches |
| 21-06-57 Test | 59 | 8/8, 8.6 | 8/8, 7.9 | none | 70–457 | 11 | disagrees |
| 22-02-50 Matthew | 144 | 4/8, 176.7 | 7/8, 42.4 | 1,2,3,7,11 | 66–2659 | 54 | matches |

The first two sessions ran at 3.0518 µV per count and the other seven at 12.2070,
so their offset spans are ±100 mV against ±400 mV. That is a scale change, not
drift.

Two sessions had good contact. Cole put all 13 live channels under the 10 µV
limit, and the Test session put 14 of 16 under it with nothing railed. Both are
also the sessions with the fewest cues: Cole played 30 of 156 authored notes and
Test played 11 of 80, and neither completed.

Cole's floor flatters one of its chips. Its five passing channels on chip 0 sat
at the rail between 25.8% and 35.4% of the time, so they were clipping rather
than quiet, and a clipped channel reports little interharmonic power. Chip 1 ran
7.4% to 26.7%, with two channels never at the rail. The Test
session is the only one of the nine with clean contact on both chips.

The five worst sessions are the five with mains in the thousands. Cole and Test
measure 6–16 µV and 70–457 µV; every other session measures hundreds to
thousands. Floor and mains move together, which is what a common-mode problem
looks like.

Gestures do appear. The 14:22 session cleared seven of 75 channel-by-gesture
cells for wrist pronation, and six of those peak at true lag. The 17:46 session
cleared three. The 11:27 session cleared two for pinky pinch, both peaking
correctly. The most recent session cleared two that both fail the lag test, one
of them peaking 500 ms early and never falling below 1.17 anywhere in ±4 s.

Railing follows whoever is wearing the band. Channel 9 railed alone in four
consecutive sessions, which 0019 read as a hard fault. It then stopped: Cole
railed 4, 5 and 7, the Test session railed nothing, and the last two Matthew
sessions railed 1, 2, 3, 7 and 11. Nothing on this board is chronically dead.

## The bias drive, traced

`CONFIG3` reads `0xC0` on both chips and `RLD_SENSP`/`RLD_SENSN` read `0x00`, so
the amplifier is powered down with no channels feeding its common-mode sense.
That alone explains why nothing rejects mains. The board explains why it cannot
simply be switched on.

From the netlist:

| Node | Connects to |
|---|---|
| `IC1.63` RLDOUT | shorted to `IC1.62` RLDIN |
| that node | `R1` 1 MΩ and `C2` 1 nF, both out to `BIAS_DRV` |
| `BIAS_DRV` | `C11` 1 nF to ground, and `RN1` 4.7 kΩ onward |
| `RN1` other end | `BIAS`, which reaches `J5` pins 3 and 4, the electrode |
| `IC1.61` RLDINV | `BIAS_INV`, which reaches `J4` pin 1 and nothing else |
| `IC1.60` RLDREF | ground |

The compensation network sits between the output and the electrode, and RLDINV,
the amplifier's inverting input, terminates on a header pin. An amplifier whose
inverting input floats has no closed loop, so powering this one drives its output
to a rail. That is the errata 0019 named, located.

Grounding RLDREF is correct here, which closes 0019's note to check it. AVDD is
+2.5 V and AVSS is −2.5 V, so mid-supply is 0 V. The external reference mode
presents what the internal one would generate.

The fix is one wire, from `J4` pin 1 to the `BIAS_DRV` node, reachable at `R1`'s
outer pad or `RN1` pin 8. That puts `R1 ∥ C2` between RLDOUT and RLDINV, the
topology SBAS459K figure 94 asks for. The electrode stays tapped off the same
node through `RN1`, and no trace needs cutting.

## Analysis

Nine sessions divide into two failures with different causes.

Six failed electrically. Their floors sit two to twenty times over the limit
because the arm has no reference. Nothing sits on `J5` pins 1 and 2, the
connector's ground, and the bias drive that would otherwise hold the body is off
and unwired. High-impedance inputs with no return path drift to whatever charge
is on them. That is why the offsets reach ±400 mV and why the railed set moves
with every re-don, and it is why several millivolts of mains arrive
differentially.

Three failed procedurally. Cole and the Test session had the contact everyone has
been trying to get, and threw it away by ending after a fifth and an eighth of
their tracks. The 17:20 session played 17 of 68 cues. Good electrodes and no
labels is the same amount of training data as bad electrodes and full labels.

The detection-limit column is what separates the two readings. On Cole the
recording could have resolved 2.8 µV of added activity and still found nothing,
which is a real negative. On the 22:02 session the best channel could not have
resolved less than 8.9 µV, and most could not have gone below 40–400 µV against
surface EMG of 14–20 µV. Its silence says nothing about the subject.

Filtering cannot recover any of it. The floor already excludes mains and its
sidebands, so a notch removes something the measurement never counted.

## Next steps

- Put an electrode on `J5` pin 1 or 2. It needs no board change and no firmware
  change, and nobody has tried it.
- Prepare the skin. Nine sessions record `skin_prep: false`.
- Then the jumper, then the loop validated on the bench with a scope on
  `BIAS_DRV`, then `RightLegDriveMode` off `Disabled`. In that order.
- Record a full track on good contact. No session has yet done both.
- 0019's channel 9 hard-fault call is retired here. Its ±12 Hz mains guard is
  already shipped, and `signal_quality.rs` has a test holding the sidebands out.
