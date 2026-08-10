/**
 * Smooth presentation of a timestamped authoritative playhead.
 *
 * The backend remains the owner of position and whether time advances. Between
 * its sparse snapshots we run the last observed velocity locally. Small clock
 * or delivery errors are paid back over a short horizon; large discontinuities
 * (a reconnect, seek, or long-stale stream) snap instead of animating through a
 * lie. Extrapolation is bounded so a dead connection cannot make the game keep
 * playing indefinitely.
 */
export class PresentationPlayhead {
  readonly #correctionHorizonMs: number;
  readonly #snapThresholdMs: number;
  readonly #maxExtrapolationMs: number;

  #positionMs = 0;
  #anchorAtMs = 0;
  #advanceAfterMs = 0;
  #advanceForMs = 0;
  #advancing = false;
  #correctionMs = 0;
  #initialized = false;

  constructor({
    correctionHorizonMs = 250,
    snapThresholdMs = 1_000,
    maxExtrapolationMs = 1_500,
  }: {
    correctionHorizonMs?: number;
    snapThresholdMs?: number;
    maxExtrapolationMs?: number;
  } = {}) {
    this.#correctionHorizonMs = correctionHorizonMs;
    this.#snapThresholdMs = snapThresholdMs;
    this.#maxExtrapolationMs = maxExtrapolationMs;
  }

  observe(
    positionMs: number,
    observedAtUnixMs: number,
    advancing: boolean,
    nowUnixMs: number,
    forceSnap = false,
  ): void {
    const presented = this.value(nowUnixMs);
    const sampleAge = nowUnixMs - observedAtUnixMs;
    const stale = sampleAge > this.#maxExtrapolationMs;
    const clockMovedBack = this.#initialized && nowUnixMs < this.#anchorAtMs;
    const targetNow = positionMs + (advancing ? Math.min(Math.max(0, sampleAge), this.#maxExtrapolationMs) : 0);
    const error = targetNow - presented;
    const snap = forceSnap || !this.#initialized || stale || clockMovedBack || Math.abs(error) >= this.#snapThresholdMs;

    this.#positionMs = snap ? targetNow : presented;
    this.#anchorAtMs = nowUnixMs;
    this.#advancing = advancing;
    this.#correctionMs = snap ? 0 : error;
    // A future audible observation (normally the exact t=0 anchor) stays at
    // its position until that instant. A delayed sample may extrapolate only
    // the unused portion of the stale budget.
    this.#advanceAfterMs = advancing ? Math.max(0, -sampleAge) : 0;
    this.#advanceForMs = advancing
      ? Math.max(0, this.#maxExtrapolationMs - Math.max(0, sampleAge))
      : 0;
    this.#initialized = true;
  }

  value(nowMs: number): number {
    if (!this.#initialized) return 0;
    const elapsed = Math.max(0, nowMs - this.#anchorAtMs);
    const extrapolated = this.#advancing
      ? Math.min(Math.max(0, elapsed - this.#advanceAfterMs), this.#advanceForMs)
      : 0;
    const correctionShare = Math.min(1, elapsed / this.#correctionHorizonMs);
    return Math.max(0, this.#positionMs + extrapolated + this.#correctionMs * correctionShare);
  }
}
