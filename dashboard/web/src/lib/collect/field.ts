// The falling-blocks playfield: a pure canvas renderer plus the small immutable
// model it draws from.
//
// Each note is a *hold block*: its leading (bottom) edge crossing the hit line
// means "begin the gesture", the block is held while it passes, and its
// trailing (top) edge crossing means "release". While a hold is in progress the
// block's bottom is pinned to the hit line and the rest keeps falling, so the
// visible block shrinks until the release edge lands — the consumption is the
// hold-time countdown.
//
// Nothing in here holds game state. Every frame is a function of the beatmap
// (fixed for the session), the audio element's position, and the streak the
// backend's note results imply — so the panel can be unmounted, refreshed, or
// re-rendered at any moment and it paints the same thing. Position is always a
// TrackMilliseconds read from the audio element; the wall clock never enters the
// drawing.
//
// Future: model evaluation will render *predicted* gesture segments as a second
// block layer beside the cued one (same geometry, distinct styling), so keep
// block drawing generic over its source.
import type { CollectionClass, Note, TrackMilliseconds } from '../protocol';

/** How long a block's leading edge is on screen before it reaches the hit line. */
export const FALL_DURATION_MILLISECONDS = 3000;

/** How long a block keeps fading after its release. */
const FADE_DURATION_MILLISECONDS = 320;

/** Vertical space under the hit line, reserved for the target rings. */
const TARGET_BAND_HEIGHT = 108;

const RING_RADIUS = 26;
const RING_THICKNESS = 5;
const LANE_INSET = 10;
const BLOCK_CORNER_RADIUS = 8;
/** A zero-length block would be invisible; every hold renders at least this tall. */
const MINIMUM_BLOCK_HEIGHT = 14;

/** Streak at which the ring reaches its brightest tone. */
const STREAK_FULL = 12;

/** One cue's hold on a lane's timeline, in track milliseconds. */
export interface Block {
  readonly at: number;
  readonly release: number;
}

/** One lane: a collection class and its hold blocks, ascending by onset. Built
 * once per beatmap; ascending order is what makes the visible window and the
 * delivered count binary searches instead of scans. */
export interface Lane {
  readonly classId: string;
  readonly label: string;
  /** Palette name from the backend — resolve with `theme.color()`, never a literal. */
  readonly colorName: string;
  readonly blocks: readonly Block[];
}

export interface Playfield {
  readonly lanes: readonly Lane[];
  readonly totalNotes: number;
}

/** Lanes follow the catalog order. A note whose class is not in the catalog is
 * dropped rather than guessed at. `BeatmapFrame.notes` arrives time-ordered and
 * non-overlapping (the protocol guard enforces it), so each lane's blocks come
 * out ascending without being sorted here. */
export function buildPlayfield(
  classes: readonly CollectionClass[],
  notes: readonly Note[],
): Playfield {
  const byClassId = new Map<string, Block[]>();
  for (const collectionClass of classes) {
    byClassId.set(collectionClass.id, []);
  }
  let totalNotes = 0;
  for (const note of notes) {
    const blocks = byClassId.get(note.class_id);
    if (blocks === undefined) continue;
    blocks.push({ at: note.at, release: note.at + note.hold });
    totalNotes += 1;
  }
  const lanes = classes.map((collectionClass): Lane => ({
    classId: collectionClass.id,
    label: collectionClass.label,
    colorName: collectionClass.color,
    blocks: byClassId.get(collectionClass.id) ?? [],
  }));
  return { lanes, totalNotes };
}

/** Canvas chrome (anything not per-class), resolved from app.css's --canvas-*
 * custom properties so the playfield tracks the active theme. */
export interface FieldChrome {
  readonly background: string;
  readonly grid: string;
  readonly gridFaint: string;
  readonly label: string;
  readonly textStrong: string;
}

export interface FieldFrame {
  readonly playfield: Playfield;
  /** Resolved CSS colours, one per lane, in lane order. */
  readonly laneColors: readonly string[];
  readonly chrome: FieldChrome;
  readonly position: TrackMilliseconds;
  readonly streak: number;
  readonly width: number;
  readonly height: number;
}

/** How many blocks begin at or before `value`. */
function countOnsetsAtOrBefore(blocks: readonly Block[], value: number): number {
  let low = 0;
  let high = blocks.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((blocks[middle]?.at ?? 0) <= value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

/** Index of the first block whose release is at or after `value` — blocks are
 * non-overlapping, so releases ascend exactly like onsets. */
function firstReleasedAtOrAfter(blocks: readonly Block[], value: number): number {
  let low = 0;
  let high = blocks.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((blocks[middle]?.release ?? 0) < value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

/** Parses `#rrggbb` (the palette's format). Returns null for anything else so an
 * unexpected colour string is passed through untouched instead of mangled. */
function parseHex(color: string): readonly [number, number, number] | null {
  if (!/^#[0-9a-f]{6}$/i.test(color)) return null;
  return [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16),
  ];
}

/** Mixes a colour toward white. The streak reward is this and nothing else: the
 * ring keeps its class hue and gets brighter, so lanes stay identifiable. */
function towardWhite(color: string, amount: number): string {
  const parsed = parseHex(color);
  if (parsed === null) return color;
  const mix = (channel: number): number => Math.round(channel + (255 - channel) * amount);
  const [red, green, blue] = parsed;
  return `rgb(${mix(red)} ${mix(green)} ${mix(blue)})`;
}

function withAlpha(color: string, alpha: number): string {
  const parsed = parseHex(color);
  if (parsed === null) return color;
  const [red, green, blue] = parsed;
  return `rgb(${red} ${green} ${blue} / ${alpha})`;
}

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const limit = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + limit, y);
  context.arcTo(x + width, y, x + width, y + height, limit);
  context.arcTo(x + width, y + height, x, y + height, limit);
  context.arcTo(x, y + height, x, y, limit);
  context.arcTo(x, y, x + width, y, limit);
  context.closePath();
}

export function renderField(context: CanvasRenderingContext2D, frame: FieldFrame): void {
  const { playfield, laneColors, chrome, position, streak, width, height } = frame;
  context.clearRect(0, 0, width, height);
  context.fillStyle = chrome.background;
  context.fillRect(0, 0, width, height);

  const laneCount = playfield.lanes.length;
  if (laneCount === 0 || width <= 0 || height <= 0) return;

  const hitLineY = Math.max(40, height - TARGET_BAND_HEIGHT);
  const ringCenterY = hitLineY + TARGET_BAND_HEIGHT / 2;
  const laneWidth = width / laneCount;
  // The vertical position a track moment renders at right now: the hit line at
  // `position`, the top of the field one fall-duration later.
  const yOf = (moment: number): number =>
    hitLineY * (1 - (moment - position) / FALL_DURATION_MILLISECONDS);
  const windowEnd = position + FALL_DURATION_MILLISECONDS;
  const windowStart = position - FADE_DURATION_MILLISECONDS;
  const brightness = Math.min(Math.max(streak, 0) / STREAK_FULL, 1) * 0.62;

  // Lane separators and the hit line: the only two static marks on the field.
  context.strokeStyle = chrome.gridFaint;
  context.lineWidth = 1;
  context.beginPath();
  for (let lane = 1; lane < laneCount; lane += 1) {
    const x = Math.round(lane * laneWidth) + 0.5;
    context.moveTo(x, 0);
    context.lineTo(x, hitLineY);
  }
  context.stroke();

  context.strokeStyle = chrome.grid;
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(0, hitLineY);
  context.lineTo(width, hitLineY);
  context.stroke();

  for (let laneIndex = 0; laneIndex < laneCount; laneIndex += 1) {
    const lane = playfield.lanes[laneIndex];
    if (lane === undefined) continue;
    const color = laneColors[laneIndex] ?? chrome.label;
    const laneLeft = laneIndex * laneWidth;
    const blockLeft = laneLeft + LANE_INSET;
    const blockWidth = Math.max(4, laneWidth - LANE_INSET * 2);

    // Faint lane wash so an empty lane still reads as a lane.
    context.fillStyle = withAlpha(color, 0.05);
    context.fillRect(laneLeft, 0, laneWidth, hitLineY);

    let holding = 0;
    const firstVisible = firstReleasedAtOrAfter(lane.blocks, windowStart);
    for (let index = firstVisible; index < lane.blocks.length; index += 1) {
      const block = lane.blocks[index];
      if (block === undefined) break;
      if (block.at > windowEnd) break;

      // The hold in progress: bottom pinned to the line, top still falling.
      const inHold = position >= block.at && position <= block.release;
      if (inHold) holding = 1;

      const top = Math.max(yOf(block.release), -BLOCK_CORNER_RADIUS);
      const bottom = Math.min(yOf(block.at), hitLineY);
      const blockHeight = Math.max(bottom - top, MINIMUM_BLOCK_HEIGHT);

      // Fade in entering at the top; fade out once released.
      const entering = Math.min(1, (windowEnd - block.at) / 220);
      const sinceRelease = position - block.release;
      const leaving =
        sinceRelease <= 0 ? 1 : Math.max(0, 1 - sinceRelease / FADE_DURATION_MILLISECONDS);
      const alpha = Math.max(0, Math.min(1, entering)) * leaving;
      if (alpha <= 0.01) continue;

      // A held block reads brighter with a hard press edge; an approaching one
      // is quieter with a visible onset edge at its bottom.
      context.fillStyle = withAlpha(color, (inHold ? 0.95 : 0.55) * alpha);
      roundedRect(
        context,
        blockLeft,
        bottom - blockHeight,
        blockWidth,
        blockHeight,
        BLOCK_CORNER_RADIUS,
      );
      context.fill();

      // Onset edge: the "press now" line at the bottom of the block, visible
      // until the press has happened.
      if (!inHold && position < block.at) {
        context.fillStyle = withAlpha(color, alpha);
        context.fillRect(blockLeft, bottom - 3, blockWidth, 3);
      }
    }

    drawTargetRing(context, {
      centerX: laneLeft + laneWidth / 2,
      centerY: ringCenterY,
      color,
      chrome,
      // Cue delivery, not scoring: the share of this lane's holds already begun.
      delivered:
        lane.blocks.length === 0
          ? 0
          : countOnsetsAtOrBefore(lane.blocks, position) / lane.blocks.length,
      brightness,
      // The ring glows for the whole hold — it is the "keep holding" cue.
      impact: holding,
    });
  }
}

interface TargetRing {
  readonly centerX: number;
  readonly centerY: number;
  readonly color: string;
  readonly chrome: FieldChrome;
  readonly delivered: number;
  readonly brightness: number;
  readonly impact: number;
}

function drawTargetRing(context: CanvasRenderingContext2D, ring: TargetRing): void {
  const { centerX, centerY, color, chrome, delivered, brightness, impact } = ring;
  const bright = towardWhite(color, brightness);

  // Track.
  context.strokeStyle = chrome.gridFaint;
  context.lineWidth = RING_THICKNESS;
  context.beginPath();
  context.arc(centerX, centerY, RING_RADIUS, 0, Math.PI * 2);
  context.stroke();

  // Progress, clockwise from twelve o'clock.
  const fraction = Math.min(Math.max(delivered, 0), 1);
  if (fraction > 0) {
    const start = -Math.PI / 2;
    context.strokeStyle = bright;
    context.lineWidth = RING_THICKNESS;
    context.lineCap = 'round';
    context.beginPath();
    context.arc(centerX, centerY, RING_RADIUS, start, start + fraction * Math.PI * 2);
    context.stroke();
    context.lineCap = 'butt';
  }

  // While a hold is in progress the ring's centre fills — the sustained "keep
  // holding" signal, mirroring the block pinned to the line above it.
  const core = RING_RADIUS - RING_THICKNESS - 2;
  context.fillStyle = withAlpha(color, 0.12 + 0.5 * impact);
  context.beginPath();
  context.arc(centerX, centerY, core * (0.55 + 0.35 * impact), 0, Math.PI * 2);
  context.fill();
}
