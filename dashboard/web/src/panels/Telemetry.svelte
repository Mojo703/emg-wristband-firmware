<script lang="ts">
  // Device telemetry: the periodic measurements the firmware reports as
  // `Frame::Telemetry` (per-chip edge timing, aligner accounting, inference
  // performance), accumulated for the browser session in `live.telemetry`.
  // One card per metric — except min/mean/max families of one quantity
  // (`edge_period_min_us` / `_mean_us` / `_max_us`), which fold into a single
  // card drawn as a min–max band with the mean line over it. The stream is
  // loss-tolerant, so gaps between samples are normal.
  import { live, type TelemetrySample } from '../lib/socket.svelte';

  const WIDTH = 240;
  const HEIGHT = 48;
  const PAD = 3;

  type StatName = 'min' | 'mean' | 'max';
  const STAT_ORDER: readonly StatName[] = ['min', 'mean', 'max'];

  interface MetricCard {
    /// The metric name, or the group's recombined name.
    readonly title: string;
    /// 'family' folds min/mean/max of one quantity into a band + mean line;
    /// 'series' draws one colored line per labeled entity (chip0/chip1).
    readonly kind: 'single' | 'family' | 'series';
    /// One entry per series; `label` is null only for kind 'single'.
    readonly stats: readonly { label: string | null; series: readonly TelemetrySample[] }[];
  }

  // `<base>_<stat>_<unit>` (or `<base>_<stat>`) → family key `<base>_<unit>`.
  function familyOf(name: string): { key: string; stat: StatName } | null {
    const match = /^(.+)_(min|mean|max)(_.+)?$/.exec(name);
    if (!match) return null;
    return { key: match[1]! + (match[3] ?? ''), stat: match[2] as StatName };
  }

  // Every way `name` splits around one run of digits. Names that share a split
  // key — identical text except that index — belong to one series group:
  // chip0_missing/chip1_missing key as "chip*_missing" with labels chip0/chip1,
  // and any future example0/example1/example2 family groups the same way with
  // no knowledge of the word "example" here. Grouping on ANY digit run is a
  // deliberate decision, not a loose match: a band_50_hz/band_60_hz or
  // p95_us/p99_us family is wanted on one card too. A metric whose digits must
  // not group has to spell them out ("two_of_three"), which fits the repo's
  // no-abbreviations naming anyway.
  function indexSplits(name: string): { key: string; label: string; index: number }[] {
    const splits: { key: string; label: string; index: number }[] = [];
    for (const match of name.matchAll(/\d+/g)) {
      const prefix = name.slice(0, match.index);
      const suffix = name.slice(match.index + match[0].length);
      splits.push({
        key: `${prefix}*${suffix}`,
        label: `${prefix}${match[0]}`.replace(/_+$/, '') || match[0],
        index: Number(match[0]),
      });
    }
    return splits;
  }

  function cardsOf(metrics: Record<string, TelemetrySample[]>): MetricCard[] {
    const names = Object.keys(metrics);
    // Count each candidate key's members, then assign every name to its
    // best-populated key (leftmost split on ties, via stable order).
    const members = new Map<string, number>();
    for (const name of names) {
      for (const split of indexSplits(name)) {
        members.set(split.key, (members.get(split.key) ?? 0) + 1);
      }
    }
    const groups = new Map<string, { label: string; index: number; name: string }[]>();
    const ungrouped: string[] = [];
    for (const name of names) {
      let best: { key: string; label: string; index: number } | null = null;
      for (const split of indexSplits(name)) {
        if ((members.get(split.key) ?? 0) < 2) continue;
        if (best === null || members.get(split.key)! > members.get(best.key)!) {
          best = split;
        }
      }
      if (best === null) {
        ungrouped.push(name);
      } else {
        const group = groups.get(best.key) ?? [];
        group.push({ label: best.label, index: best.index, name });
        groups.set(best.key, group);
      }
    }
    // A candidate can lose its partners to a better-populated key; a group of
    // one is not a group.
    for (const [key, group] of [...groups]) {
      if (group.length < 2) {
        groups.delete(key);
        for (const entry of group) ungrouped.push(entry.name);
      }
    }

    const families = new Map<string, Map<StatName, readonly TelemetrySample[]>>();
    const singles: MetricCard[] = [];
    for (const name of ungrouped) {
      const family = familyOf(name);
      if (family) {
        const stats = families.get(family.key) ?? new Map();
        stats.set(family.stat, metrics[name]!);
        families.set(family.key, stats);
      } else {
        singles.push({
          title: name,
          kind: 'single',
          stats: [{ label: null, series: metrics[name]! }],
        });
      }
    }
    const cards: MetricCard[] = [];
    for (const [key, group] of groups) {
      cards.push({
        title: key,
        kind: 'series',
        stats: group
          .sort((a, b) => a.index - b.index)
          .map((entry) => ({ label: entry.label, series: metrics[entry.name]! })),
      });
    }
    for (const [key, stats] of families) {
      if (stats.size >= 2) {
        cards.push({
          title: key,
          kind: 'family',
          stats: STAT_ORDER.filter((stat) => stats.has(stat)).map((stat) => ({
            label: stat,
            series: stats.get(stat)!,
          })),
        });
      } else {
        // A lone `_mean_` (or similar) is just a metric with a long name.
        const [stat, series] = [...stats.entries()][0]!;
        cards.push({ title: `${key} (${stat})`, kind: 'single', stats: [{ label: null, series }] });
      }
    }
    return [...singles, ...cards].sort((a, b) => a.title.localeCompare(b.title));
  }

  const sources = $derived(Object.keys(live.telemetry).sort());

  interface Hover {
    readonly source: string;
    readonly title: string;
    readonly index: number;
    readonly x: number;
  }
  let hover = $state<Hover | null>(null);

  interface Frame2d {
    readonly t0: number;
    readonly t1: number;
    readonly low: number;
    readonly high: number;
  }

  // Joint bounds over every series in the card, so band and line share axes.
  function bounds(card: MetricCard): Frame2d {
    let low = Infinity;
    let high = -Infinity;
    let t0 = Infinity;
    let t1 = -Infinity;
    for (const { series } of card.stats) {
      for (const sample of series) {
        if (sample.value < low) low = sample.value;
        if (sample.value > high) high = sample.value;
        if (sample.t_us < t0) t0 = sample.t_us;
        if (sample.t_us > t1) t1 = sample.t_us;
      }
    }
    if (high === low) {
      low -= 1;
      high += 1;
    }
    return { t0, t1, low, high };
  }

  function x(frame: Frame2d, t_us: number): number {
    if (frame.t1 === frame.t0) return WIDTH / 2;
    return PAD + ((t_us - frame.t0) / (frame.t1 - frame.t0)) * (WIDTH - 2 * PAD);
  }

  function y(frame: Frame2d, value: number): number {
    return PAD + (1 - (value - frame.low) / (frame.high - frame.low)) * (HEIGHT - 2 * PAD);
  }

  function linePath(frame: Frame2d, series: readonly TelemetrySample[]): string {
    return series
      .map(
        (sample, index) =>
          `${index === 0 ? 'M' : 'L'}${x(frame, sample.t_us).toFixed(1)},${y(frame, sample.value).toFixed(1)}`,
      )
      .join(' ');
  }

  // Forward along the upper series, back along the lower: the min–max band.
  function bandPath(
    frame: Frame2d,
    lower: readonly TelemetrySample[],
    upper: readonly TelemetrySample[],
  ): string {
    const count = Math.min(lower.length, upper.length);
    if (count === 0) return '';
    const forward = upper
      .slice(0, count)
      .map(
        (sample, index) =>
          `${index === 0 ? 'M' : 'L'}${x(frame, sample.t_us).toFixed(1)},${y(frame, sample.value).toFixed(1)}`,
      );
    const back = lower
      .slice(0, count)
      .reverse()
      .map((sample) => `L${x(frame, sample.t_us).toFixed(1)},${y(frame, sample.value).toFixed(1)}`);
    return [...forward, ...back, 'Z'].join(' ');
  }

  function labeledSeries(card: MetricCard, label: string): readonly TelemetrySample[] | null {
    return card.stats.find((entry) => entry.label === label)?.series ?? null;
  }

  /// The band's edges for a 'family' card: min and max when both exist, else
  /// whichever exists paired with mean.
  function bandEdges(
    card: MetricCard,
  ): { lower: readonly TelemetrySample[]; upper: readonly TelemetrySample[] } | null {
    if (card.kind !== 'family') return null;
    const lower = labeledSeries(card, 'min') ?? labeledSeries(card, 'mean')!;
    const upper = labeledSeries(card, 'max') ?? labeledSeries(card, 'mean')!;
    return { lower, upper };
  }

  /// The series the hover snaps to and the family/single cards draw as their
  /// line: mean when present, else the first series.
  function primary(card: MetricCard): readonly TelemetrySample[] {
    return labeledSeries(card, 'mean') ?? card.stats[0]!.series;
  }

  function formatValue(value: number): string {
    if (Number.isInteger(value)) return value.toLocaleString();
    return Math.abs(value) >= 100 ? value.toFixed(1) : value.toPrecision(4);
  }

  function uptime(t_us: number): string {
    const seconds = t_us / 1e6;
    if (seconds < 100) return `${seconds.toFixed(1)} s`;
    return `${(seconds / 60).toFixed(1)} min`;
  }

  /// The card's headline values, at `index` from the tail (hover) or newest:
  /// one labeled part per series, in series order.
  function valuesAt(
    card: MetricCard,
    index: number | null,
  ): { label: string | null; text: string }[] {
    const parts: { label: string | null; text: string }[] = [];
    for (const { label, series } of card.stats) {
      const sample = series[index ?? series.length - 1];
      if (sample === undefined) continue;
      parts.push({
        label,
        text: label === null ? formatValue(sample.value) : `${label} ${formatValue(sample.value)}`,
      });
    }
    return parts;
  }

  function onHover(event: PointerEvent, source: string, card: MetricCard): void {
    const series = primary(card);
    if (series.length === 0) return;
    const frame = bounds(card);
    const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
    const pointerX = ((event.clientX - rect.left) / rect.width) * WIDTH;
    let index = 0;
    let nearest = Infinity;
    for (let i = 0; i < series.length; i++) {
      const distance = Math.abs(x(frame, series[i]!.t_us) - pointerX);
      if (distance < nearest) {
        nearest = distance;
        index = i;
      }
    }
    hover = { source, title: card.title, index, x: x(frame, series[index]!.t_us) };
  }

  function hoverFor(source: string, card: MetricCard): Hover | null {
    return hover !== null && hover.source === source && hover.title === card.title ? hover : null;
  }
</script>

<div class="telemetry">
  {#if sources.length === 0}
    <div class="muted empty">No telemetry from the selected device yet.</div>
  {:else}
    {#each sources as source (source)}
      <section>
        <h2>{source}</h2>
        <div class="grid">
          {#each cardsOf(live.telemetry[source]!) as card (card.title)}
            {@const frame = bounds(card)}
            {@const band = bandEdges(card)}
            {@const hovered = hoverFor(source, card)}
            {@const reference = primary(card)}
            <div class="card">
              <div class="metric-name">{card.title}</div>
              <div class="value">
                {#each valuesAt(card, hovered?.index ?? null) as part, index (part.label ?? '')}
                  <span class="value-part">
                    {#if card.kind === 'series'}
                      <span class="dot series-{index}"></span>
                    {/if}
                    {part.text}
                  </span>
                {/each}
              </div>
              <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
              <svg
                role="img"
                aria-label="{source} {card.title} history"
                viewBox="0 0 {WIDTH} {HEIGHT}"
                onpointermove={(event) => onHover(event, source, card)}
                onpointerleave={() => (hover = null)}
              >
                {#if card.kind === 'series'}
                  {#each card.stats as entry, index (entry.label)}
                    <path class="line series-{index}" d={linePath(frame, entry.series)} />
                  {/each}
                {:else}
                  {#if band}
                    <path class="band" d={bandPath(frame, band.lower, band.upper)} />
                  {/if}
                  <path class="line series-0" d={linePath(frame, reference)} />
                {/if}
                {#if hovered}
                  <line x1={hovered.x} y1="0" x2={hovered.x} y2={HEIGHT} />
                {/if}
              </svg>
              <div class="caption muted">
                {#if hovered}
                  at {uptime(reference[hovered.index]!.t_us)} uptime
                {:else}
                  {reference.length} samples · latest {uptime(
                    reference[reference.length - 1]!.t_us,
                  )}
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </section>
    {/each}
  {/if}
</div>

<style>
  .telemetry {
    height: 100%;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  section h2 {
    margin: 0 0 8px;
    font-size: 15px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 8px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px 8px;
  }
  .metric-name {
    font-family: ui-monospace, monospace;
    font-size: 12px;
    color: var(--muted-foreground);
  }
  .value {
    font-size: 15px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    margin: 2px 0 6px;
  }
  svg {
    display: block;
    width: 100%;
    height: 48px;
  }
  svg .line {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  /* Color follows the entity: series order is fixed (sorted labels), so a
     chip keeps its hue whatever else the card shows. */
  svg .line.series-0 {
    stroke: var(--chart-1);
  }
  svg .line.series-1 {
    stroke: var(--chart-2);
  }
  svg .line.series-2 {
    stroke: var(--chart-3);
  }
  svg .band {
    fill: var(--chart-1);
    opacity: 0.18;
    stroke: none;
  }
  .value-part {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-right: 10px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    display: inline-block;
  }
  .dot.series-0 {
    background: var(--chart-1);
  }
  .dot.series-1 {
    background: var(--chart-2);
  }
  .dot.series-2 {
    background: var(--chart-3);
  }
  svg line {
    stroke: var(--muted-foreground);
    stroke-width: 1;
  }
  .caption {
    margin-top: 4px;
    font-size: 11px;
  }
  .muted {
    color: var(--muted-foreground);
  }
  .empty {
    padding: 12px 4px;
  }
</style>
