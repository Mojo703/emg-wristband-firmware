# Donning the wristband

One page, in order, every time. Skipping the skin steps is the leading
suspect for sessions that never resolved spatially; the electrode panel
check is what catches a bad seat before it costs a calibration.

## Prepare (first don of the day)

- [ ] 1. Shave the dorsal wrist if stubbled (that side has hair and the
      worse noise floor).
- [ ] 2. Abrade lightly with NuPrep or a fine pad, leaving skin slightly pink, not
      raw. This is the step that matters most and the one people skip.
- [ ] 3. Wipe with 70% isopropyl alcohol; let it dry completely.

## Seat

- [ ] 4. Ring the two modules around the wrist, one module inner, one
      outer, hook-and-loop snug: no gap under any electrode, no skin
      blanching. Every electrode flat on skin.
- [ ] 5. Note band offset and rotation; enter them in the dashboard's
      setup form and press re-donned (the timestamp must move).
- [ ] 6. Sit still ten seconds, then check the electrode panel: every
      channel live, none flat or railed, with plausible offset and noise.
      Lead-off detection is currently unavailable. A missing lead-off value
      means unknown, not good contact. Re-seat or re-wipe a suspect channel now; the
      firmware cannot yet identify a lifted electrode.

## Calibrate (per don; reuse is disabled)

- [ ] 7. Pole or grippable stick within reach.
- [ ] 8. Dashboard → Calibrate → Start. Stand still, arm supported,
      until the first prompt (~60 s).
- [ ] 9. Answer prompts in the fixed order: tip forward, tip back, tip
      in, tip out, thumb up. Hold each for ~1.5 s, with the thumb extended for
      the whole hold. Silence = counted; soft bump = same gesture again.
- [ ] 10. Triple bump: grab the pole. Same order, thumb gripping.
- [ ] 11. Green flash + click-then-bump: done. Enable Phone BLE in the
      dashboard, pair if needed, and wait for `paired`. Try each gesture
      thumb-up (keys fire) and gripped (nothing fires).

## If it goes wrong

- A gesture's own rhythm followed by a bump = that class failed: lift
  the band clear, wipe the skin, re-seat, recalibrate. The previous
  calibration is still installed; a failed run never leaves the device
  worse.
- Any `booting (panic)` line in the Logs panel: note what you were
  doing and tell the bench; the device recovers by itself.
