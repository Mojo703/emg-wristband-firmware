import { CHANNEL_COUNT, SAMPLE_RATE_HZ, type EmgWindow } from '../band/emgStream.ts';
import { MICROVOLTS_PER_WIRE_COUNT } from './session.ts';

export const ANALYSIS_LENGTH = 4096;
export const BAND_LOW_HERTZ = 20;
export const BAND_HIGH_HERTZ = 450;
export const MAINS_GUARD_HERTZ = 12;
export const NOISE_FLOOR_LIMIT_MICROVOLTS = 10;
const MAINS_SEARCH_LOW_HERTZ = 45;
const MAINS_SEARCH_HIGH_HERTZ = 65;
const SATURATION_FRACTION_OF_FULL_SCALE = 0.98;
const SATURATION_HISTORY_MILLIS = 10_000;
const REPORT_INTERVAL_MILLIS = 1_000;
const FULL_SCALE_COUNTS = 32768;
const CHANNELS_PER_SOURCE = 8;

export interface ChannelQuality {
  readonly noiseFloorMicrovolts: number;
  readonly mainsMicrovolts: number;
  readonly offsetMillivolts: number;
  readonly headroomMillivolts: number;
  readonly saturatedFraction: number;
}

export interface QualityReport {
  readonly mainsHertz: number;
  readonly channels: readonly ChannelQuality[];
}

export interface Spectrum {
  readonly density: Float64Array;
  readonly binWidth: number;
}

const hann = Float64Array.from({ length: ANALYSIS_LENGTH }, (_, index) =>
  0.5 - 0.5 * Math.cos((2 * Math.PI * index) / ANALYSIS_LENGTH),
);
const hannPower = hann.reduce((sum, weight) => sum + weight * weight, 0);

function forwardTransform(real: Float64Array, imaginary: Float64Array): void {
  const length = real.length;
  for (let source = 1, target = 0; source < length; source += 1) {
    let bit = length >> 1;
    while (target & bit) {
      target ^= bit;
      bit >>= 1;
    }
    target |= bit;
    if (source < target) {
      [real[source], real[target]] = [real[target]!, real[source]!];
      [imaginary[source], imaginary[target]] = [imaginary[target]!, imaginary[source]!];
    }
  }
  for (let span = 2; span <= length; span <<= 1) {
    const half = span / 2;
    for (let offset = 0; offset < half; offset += 1) {
      const angle = (-2 * Math.PI * offset) / span;
      const weightReal = Math.cos(angle);
      const weightImaginary = Math.sin(angle);
      for (let start = 0; start < length; start += span) {
        const low = start + offset;
        const high = low + half;
        const productReal = weightReal * real[high]! - weightImaginary * imaginary[high]!;
        const productImaginary = weightReal * imaginary[high]! + weightImaginary * real[high]!;
        real[high] = real[low]! - productReal;
        imaginary[high] = imaginary[low]! - productImaginary;
        real[low] = real[low]! + productReal;
        imaginary[low] = imaginary[low]! + productImaginary;
      }
    }
  }
}

export function spectrumOf(microvolts: Float64Array, sampleRate = SAMPLE_RATE_HZ): Spectrum {
  const length = microvolts.length;
  const mean = microvolts.reduce((sum, value) => sum + value, 0) / length;
  const real = microvolts.map((value, index) => (value - mean) * hann[index]!);
  const imaginary = new Float64Array(length);
  forwardTransform(real, imaginary);
  const scale = 1 / (hannPower * sampleRate);
  const density = new Float64Array(length / 2 + 1);
  for (let bin = 0; bin <= length / 2; bin += 1) {
    const power = (real[bin]! ** 2 + imaginary[bin]! ** 2) * scale;
    density[bin] = bin === 0 || bin === length / 2 ? power : 2 * power;
  }
  return { density, binWidth: sampleRate / length };
}

function peakMainsPower(spectrum: Spectrum): number {
  const low = Math.floor(MAINS_SEARCH_LOW_HERTZ / spectrum.binWidth);
  const high = Math.min(Math.floor(MAINS_SEARCH_HIGH_HERTZ / spectrum.binWidth), spectrum.density.length - 1);
  let peak = 0;
  for (let bin = low; bin <= high; bin += 1) peak = Math.max(peak, spectrum.density[bin]!);
  return peak;
}

export function mainsFundamentalHertz(spectrum: Spectrum): number {
  const { density, binWidth } = spectrum;
  const low = Math.max(Math.floor(MAINS_SEARCH_LOW_HERTZ / binWidth), 1);
  const high = Math.min(Math.floor(MAINS_SEARCH_HIGH_HERTZ / binWidth), density.length - 2);
  if (low >= high) return 0;
  let peak = low;
  for (let bin = low; bin <= high; bin += 1) if (density[bin]! > density[peak]!) peak = bin;
  const before = density[peak - 1]!;
  const at = density[peak]!;
  const after = density[peak + 1]!;
  const curvature = before - 2 * at + after;
  const shift = curvature === 0 ? 0 : (0.5 * (before - after)) / curvature;
  return (peak + Math.min(Math.max(shift, -0.5), 0.5)) * binWidth;
}

function nearMainsHarmonic(frequency: number, mainsHertz: number): boolean {
  if (mainsHertz <= 0) return false;
  const harmonic = Math.max(Math.round(frequency / mainsHertz), 1);
  return Math.abs(frequency - harmonic * mainsHertz) <= MAINS_GUARD_HERTZ;
}

export function bandPowers(spectrum: Spectrum, mainsHertz: number): { broadband: number; mains: number } {
  let keptPower = 0;
  let keptBins = 0;
  let mainsPower = 0;
  let bandBins = 0;
  spectrum.density.forEach((power, bin) => {
    const frequency = bin * spectrum.binWidth;
    if (frequency < BAND_LOW_HERTZ || frequency > BAND_HIGH_HERTZ) return;
    bandBins += 1;
    if (nearMainsHarmonic(frequency, mainsHertz)) {
      mainsPower += power;
    } else {
      keptPower += power;
      keptBins += 1;
    }
  });
  const broadband = keptBins === 0 ? 0 : (keptPower / keptBins) * bandBins * spectrum.binWidth;
  return { broadband: Math.sqrt(broadband), mains: Math.sqrt(mainsPower * spectrum.binWidth) };
}

function missingAt(missing: Uint8Array, steps: number, source: number, step: number): boolean {
  const index = source * Math.ceil(steps / 8) + Math.floor(step / 8);
  return index < missing.length && (missing[index]! & (1 << step % 8)) !== 0;
}

export class SignalQualityMonitor {
  #recent = Array.from({ length: CHANNEL_COUNT }, () => new Int16Array(ANALYSIS_LENGTH));
  #written = 0;
  #saturation: { at: number; samples: number; perChannel: Uint32Array }[] = [];
  #lastReport: number | null = null;

  add(window: EmgWindow, at: number): QualityReport | null {
    const steps = window.stepsPerChannel;
    const limit = FULL_SCALE_COUNTS * SATURATION_FRACTION_OF_FULL_SCALE;
    const perChannel = new Uint32Array(CHANNEL_COUNT);
    for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
      const buffer = this.#recent[channel]!;
      const source = Math.floor(channel / CHANNELS_PER_SOURCE);
      let held = this.#written === 0 ? 0 : buffer[(this.#written - 1) % ANALYSIS_LENGTH]!;
      for (let step = 0; step < steps; step += 1) {
        let count = held;
        if (!missingAt(window.missing, steps, source, step)) {
          count = window.samples[channel * steps + step]!;
          held = count;
          if (Math.abs(count) >= limit) perChannel[channel]! += 1;
        }
        buffer[(this.#written + step) % ANALYSIS_LENGTH] = count;
      }
    }
    this.#written += steps;
    this.#saturation.push({ at, samples: steps, perChannel });
    while (this.#saturation.length > 0 && at - this.#saturation[0]!.at > SATURATION_HISTORY_MILLIS) {
      this.#saturation.shift();
    }
    const due = this.#lastReport === null || at - this.#lastReport >= REPORT_INTERVAL_MILLIS;
    if (!due || this.#written < ANALYSIS_LENGTH) return null;
    this.#lastReport = at;
    return this.#report();
  }

  #report(): QualityReport {
    const scale = MICROVOLTS_PER_WIRE_COUNT;
    const ordered = this.#recent.map((buffer) => {
      const microvolts = new Float64Array(ANALYSIS_LENGTH);
      for (let index = 0; index < ANALYSIS_LENGTH; index += 1) {
        microvolts[index] = buffer[(this.#written + index) % ANALYSIS_LENGTH]! * scale;
      }
      return microvolts;
    });
    const spectra = ordered.map((microvolts) => spectrumOf(microvolts));
    const loudest = spectra.reduce((best, spectrum) => (peakMainsPower(spectrum) > peakMainsPower(best) ? spectrum : best));
    const mainsHertz = mainsFundamentalHertz(loudest);
    const fullScaleMillivolts = (FULL_SCALE_COUNTS * scale) / 1000;
    const historySamples = this.#saturation.reduce((sum, tally) => sum + tally.samples, 0);
    const channels = spectra.map((spectrum, channel) => {
      const powers = bandPowers(spectrum, mainsHertz);
      const offsetMillivolts = ordered[channel]!.reduce((sum, value) => sum + value, 0) / ANALYSIS_LENGTH / 1000;
      const saturated = this.#saturation.reduce((sum, tally) => sum + tally.perChannel[channel]!, 0);
      return {
        noiseFloorMicrovolts: powers.broadband,
        mainsMicrovolts: powers.mains,
        offsetMillivolts,
        headroomMillivolts: Math.max(fullScaleMillivolts - Math.abs(offsetMillivolts), 0),
        saturatedFraction: historySamples === 0 ? 0 : saturated / historySamples,
      };
    });
    return { mainsHertz, channels };
  }
}
