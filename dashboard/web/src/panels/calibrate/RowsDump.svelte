<script lang="ts">
  // Pull a stored slot back off the device so a calibration that went wrong in
  // the field can be replayed at a desk. The device answers a request with a run
  // of rows at a time; this asks for the next run until the slot is complete,
  // then hands the operator the bytes verbatim — the record and the rows as they
  // sit on flash, never a re-encoding, so the host fits on exactly what the
  // device fitted on.
  import { Button } from '$lib/components/ui/button/index.js';
  import { api, on, type CalibrationRowsDumpFrame } from '../../lib/socket.svelte';
  import { precisionName, priorHashText } from './text';

  /** Wearer slots, per the flash layout. The device holds two and evicts the
   * lowest sequence; either may be the interesting one after a bad run. */
  const SLOTS: readonly number[] = [0, 1];

  /** Rows to ask for per frame. The device caps this at what its encode buffer
   * holds, so asking high costs nothing and asking low costs round trips. */
  const ROWS_PER_REQUEST = 256;

  /** A slot that stops answering has to end as an error rather than a spinner. */
  const REPLY_TIMEOUT_MILLISECONDS = 5000;

  interface Transfer {
    readonly slot: number;
    readonly sequence: number;
    readonly valid: boolean;
    readonly priorHash: number;
    readonly record: Uint8Array;
    readonly rowStride: number;
    readonly precision: number;
    readonly totalRows: number;
    readonly rowsReceived: number;
  }

  let slot = $state(SLOTS[0]!);
  let transfer = $state<Transfer | null>(null);
  let failure = $state<string | null>(null);
  // The bytes themselves are not state: nothing renders them, and copying a
  // slot's worth of rows on every frame would be the only expensive thing here.
  let chunks: Uint8Array[] = [];
  let timer: number | null = null;

  function clearTimer(): void {
    if (timer !== null) {
      window.clearTimeout(timer);
      timer = null;
    }
  }

  function armTimer(): void {
    clearTimer();
    timer = window.setTimeout(() => {
      timer = null;
      failure = 'The device did not answer.';
      transfer = null;
      chunks = [];
    }, REPLY_TIMEOUT_MILLISECONDS);
  }

  function request(firstRow: number): void {
    api.requestCalibrationRows(slot, firstRow, ROWS_PER_REQUEST);
    armTimer();
  }

  function start(): void {
    failure = null;
    transfer = null;
    chunks = [];
    request(0);
  }

  function download(bytes: Uint8Array, name: string): void {
    const url = URL.createObjectURL(new Blob([bytes as BlobPart]));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function concatenate(parts: readonly Uint8Array[]): Uint8Array {
    const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
    const joined = new Uint8Array(total);
    let offset = 0;
    for (const part of parts) {
      joined.set(part, offset);
      offset += part.byteLength;
    }
    return joined;
  }

  // Two files, each the slot's own bytes: a container of our invention would be
  // one more thing for the host tool to agree with.
  function deliver(complete: Transfer): void {
    const stem = `calibration-slot${complete.slot}-seq${complete.sequence}`;
    download(complete.record, `${stem}-record.bin`);
    download(concatenate(chunks), `${stem}-rows.bin`);
  }

  function accept(dump: CalibrationRowsDumpFrame): void {
    // The armed timer is what says a request of ours is outstanding. Without it
    // this is another page's reply, or a late answer to a request that already
    // timed out, and appending it would corrupt a later download.
    if (timer === null) return;
    if (dump.slot !== slot) return;
    const expected = transfer?.rowsReceived ?? 0;
    if (dump.first_row !== expected) {
      clearTimer();
      failure = `The device sent rows from ${dump.first_row} when ${expected} were held.`;
      transfer = null;
      chunks = [];
      return;
    }
    if (dump.first_row === 0) chunks = [];
    chunks.push(dump.rows);
    const next: Transfer = {
      slot: dump.slot,
      sequence: dump.sequence,
      valid: dump.valid,
      priorHash: dump.prior_hash,
      record: dump.record,
      rowStride: dump.row_stride,
      precision: dump.precision,
      totalRows: dump.total_rows,
      rowsReceived: dump.first_row + dump.row_count,
    };
    transfer = next;
    if (next.rowsReceived >= next.totalRows) {
      clearTimer();
      deliver(next);
      return;
    }
    // A device that answers with no rows and claims more exist would otherwise
    // loop asking for the same run forever.
    if (dump.row_count === 0) {
      clearTimer();
      failure = 'The device stopped sending rows before the slot was complete.';
      return;
    }
    request(next.rowsReceived);
  }

  $effect(() => on('calibrationRowsDump', accept));
  $effect(() => clearTimer);

  const running = $derived(transfer !== null && transfer.rowsReceived < transfer.totalRows);
</script>

<section class="card">
  <div class="field-label">Stored slots</div>
  <p class="muted">
    Downloads a slot's record and rows as the device stored them, for replay on the host.
  </p>
  <div class="row">
    {#each SLOTS as candidate (candidate)}
      <button
        class="chip"
        aria-pressed={slot === candidate}
        disabled={running}
        onclick={() => (slot = candidate)}
      >
        Slot {candidate}
      </button>
    {/each}
    <Button variant="outline" disabled={running} onclick={start}>
      {running ? 'Downloading…' : 'Download slot'}
    </Button>
  </div>
  {#if failure !== null}
    <p class="warn">{failure}</p>
  {/if}
  {#if transfer !== null}
    <p>
      Slot {transfer.slot}, sequence {transfer.sequence}: {transfer.rowsReceived} of
      {transfer.totalRows} rows, {transfer.rowStride} bytes each,
      {precisionName(transfer.precision)}.
    </p>
    <!-- Which prior the rows were standardized against. A host that replays them
         under a different prior is fitting on numbers that mean something else. -->
    <p class="muted">Prior {priorHashText(transfer.priorHash)}.</p>
    {#if !transfer.valid}
      <p class="warn">
        The slot's CRC or prior hash did not check out. The rows are here to look at;
        nothing in them can be trusted.
      </p>
    {/if}
  {/if}
</section>

<style>
  /* Layout only; cards, chips and text colours come from the global styles. */
  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .card p {
    margin: 0;
  }

  .row {
    margin-bottom: 0;
  }
</style>
