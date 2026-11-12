// What the wearer is told about electrode quality before calibrating.
//
// The check grades the band's AC lead-off snapshot: the 500 Hz excitation tone
// each channel carries, which grows as the path through a pad opens up, and the
// share of frames it sat at a rail. That catches a pad whose solder joint has
// failed. `gradeChannel` grades the dashboard's rested noise floor and headroom
// instead, and is kept for comparison.

import type { LeadOffSnapshot } from '../band/protocol.ts';
import type { ChannelQuality, QualityReport } from '../collect/signalQuality.ts';

/** How far a channel is from clean: 0 clean, 1 unusable, `null` before a check. */
export type Contact = number | null;

/** Twice the dashboard's 10 µV limit: this band's floor sits high and it works anyway. */
const NOISE_CLEAN_MICROVOLTS = 20;
const NOISE_UNUSABLE_MICROVOLTS = 40;
/** Of about 400 mV full scale. Unmeasured guesses until recordings place them. */
const HEADROOM_CLEAN_MILLIVOLTS = 100;
const HEADROOM_UNUSABLE_MILLIVOLTS = 25;
const SATURATED_UNUSABLE_FRACTION = 0.01;

/** Above this a channel reads as noisy; at 1 it is too noisy to use. */
export const NOISY = 0.5;

const clamp = (value: number) => Math.min(1, Math.max(0, value));

export function gradeChannel(channel: ChannelQuality): number {
  if (channel.saturatedFraction > SATURATED_UNUSABLE_FRACTION) return 1;
  const noise =
    (channel.noiseFloorMicrovolts - NOISE_CLEAN_MICROVOLTS) / (NOISE_UNUSABLE_MICROVOLTS - NOISE_CLEAN_MICROVOLTS);
  const headroom =
    (HEADROOM_CLEAN_MILLIVOLTS - channel.headroomMillivolts) / (HEADROOM_CLEAN_MILLIVOLTS - HEADROOM_UNUSABLE_MILLIVOLTS);
  return clamp(Math.max(noise, headroom));
}

/** Measured on the band: good pads read about 7 mV, adequate ones up to 11 mV,
 * and a pad off the skin 11.3-11.5 mV. */
const TONE_CLEAN_MILLIVOLTS = 7;
const TONE_OPEN_MILLIVOLTS = 11.3;
const RAILED_CLEAN_PERCENT = 10;
const RAILED_UNUSABLE_PERCENT = 50;

export function gradeLeadOff(toneMicrovolts: number, railedPercent: number): number {
  const tone = (toneMicrovolts / 1000 - TONE_CLEAN_MILLIVOLTS) / (TONE_OPEN_MILLIVOLTS - TONE_CLEAN_MILLIVOLTS);
  const railed = (railedPercent - RAILED_CLEAN_PERCENT) / (RAILED_UNUSABLE_PERCENT - RAILED_CLEAN_PERCENT);
  return clamp(Math.max(tone, railed));
}

export function contactFromSnapshot(snapshot: LeadOffSnapshot): Contact[] {
  return snapshot.amplitudeMicrovolts.map((tone, channel) => gradeLeadOff(tone, snapshot.railedPercent[channel] ?? 0));
}

export function contactFromQuality(report: QualityReport): Contact[] {
  return report.channels.map(gradeChannel);
}

const CHANNELS_PER_MODULE = 8;
/** Module 0 sits under the pod on the back of the wrist; module 1 opposite it. */
const MODULE_NAMES = ['pads under the pod', 'pads on the inside of your wrist'] as const;
/** Most of a module failing means the module is off the skin, not a pad. */
const MODULE_OFF_SKIN = 6;

function count(contact: readonly Contact[], test: (badness: number) => boolean): number {
  return contact.filter((value) => value !== null && test(value)).length;
}

const unusable = (badness: number) => badness >= 1;
const noisy = (badness: number) => badness > NOISY && badness < 1;

function channels(number: number): string {
  return number === 1 ? '1 channel' : `${number} channels`;
}

export function contactSuggestion(contact: readonly Contact[]): string {
  if (contact.some((badness) => badness === null)) return 'The signal hasn\u2019t been checked yet.';
  const bad = count(contact, unusable);
  const fair = count(contact, noisy);
  if (bad === 0 && fair === 0) return 'All 16 channels look clean.';
  if (bad === 0) {
    return `${channels(fair)} ${fair === 1 ? 'is' : 'are'} a little noisy. Calibration will still work; tightening the strap one notch may help.`;
  }
  const perModule = [0, 1].map((module) =>
    count(contact.slice(module * CHANNELS_PER_MODULE, (module + 1) * CHANNELS_PER_MODULE), unusable),
  );
  const offSkin = perModule.findIndex((number) => number >= MODULE_OFF_SKIN);
  if (offSkin !== -1) {
    return `The ${MODULE_NAMES[offSkin]} aren\u2019t picking up a clean signal. Turn the band so they sit flat against your wrist, then tighten the strap one notch.`;
  }
  return `${channels(bad)} ${bad === 1 ? 'is' : 'are'} too noisy to use. Keep your arm relaxed and still, then tighten the strap one notch or reseat the band.`;
}

/** Green through orange to red as a channel goes from clean to unusable. */
export function contactColor(badness: Contact): string {
  if (badness === null) return 'var(--line)';
  if (badness <= 0.5) return `color-mix(in oklab, var(--fair) ${Math.round(badness * 200)}%, var(--good))`;
  return `color-mix(in oklab, var(--bad) ${Math.round((badness - 0.5) * 200)}%, var(--fair))`;
}
