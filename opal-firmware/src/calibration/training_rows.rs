//! The flash-resident calibration training rows: find the `training`
//! partition, map it, and hand its packed rows to a fit.
//!
//! They live in flash because they do not fit anywhere else. The full training
//! set is 9654 rows — 604 KB even at int8 — against a device whose free heap is
//! a couple of hundred kilobytes, so a calibration that wants the prior data
//! cannot hold it. Mapping rather than reading keeps it that way: the fit walks
//! the rows through the flash cache in place, and the only RAM cost is this
//! struct.
//!
//! The image is written separately (`espflash write-bin`), never by the app, and
//! `firmware-bench/PROTOCOL.md` documents its layout. Nothing here parses rows —
//! `emg_runtime::calibration::StaticFeatureRows` owns that — so this module's
//! whole job is to produce a byte slice that is exactly a whole number of rows
//! at a precision the caller can name.

use anyhow::{bail, Context, Result};
use emg_runtime::calibration::{FeaturePrecision, Int8Quantization};
use emg_runtime::flash_image::{
    self, ImageError, LiveSlot, PriorImage, SlotRecord, StandardizationVariant, PARTITION_BYTES,
    SLOT_BYTES, SLOT_COUNT, SLOT_CRC_OFFSET, SLOT_OFFSETS, SLOT_ROWS_OFFSET,
};
use emg_runtime::streaming_fit::{RowSource, ROW_STRIDE};
use esp_idf_svc::sys::EspError;
use log::{info, warn};

/// Marks the partition as carrying a row image at all. A partition that was
/// never written reads as erased flash (`0xFF`), and a fit that took that for
/// rows would train on 14000 rows of garbage and report a plausible wall time
/// for it.
const MAGIC: [u8; 8] = *b"OPALROWS";

/// The layout this build understands.
const VERSION: u32 = 1;

/// Fixed header, then the int8 constants, then the rows. The constants sit in
/// the image rather than arriving with the fit command because they are a
/// property of how these particular rows were quantized: an image and the
/// affine that decodes it cannot be allowed to travel separately.
const HEADER_BYTES: usize = 32;
const QUANTIZATION_BYTES: usize = 64 * 2 * 4;
const ROWS_OFFSET: usize = HEADER_BYTES + QUANTIZATION_BYTES;

/// A mapped training-row image. The mapping is released when this is dropped,
/// so the borrow of [`Self::rows`] cannot outlive it.
pub(crate) struct TrainingRows {
    handle: esp_idf_svc::sys::esp_partition_mmap_handle_t,
    /// The mapped window, starting at the image's first byte.
    mapped: &'static [u8],
    row_count: usize,
    row_bytes: usize,
    precision: FeaturePrecision,
    quantization: Int8Quantization,
}

impl TrainingRows {
    /// Map the `training` partition's rows, or explain why not. A missing or
    /// unwritten partition is a plain absence, not a fault: a device with no
    /// image simply fits on live rows alone.
    pub(crate) fn map() -> Result<Option<TrainingRows>> {
        let label = c"training";
        let partition = unsafe {
            esp_idf_svc::sys::esp_partition_find_first(
                esp_idf_svc::sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
                esp_idf_svc::sys::esp_partition_subtype_t_ESP_PARTITION_SUBTYPE_ANY,
                label.as_ptr(),
            )
        };
        if partition.is_null() {
            info!("no training partition; calibration fits on live rows only");
            return Ok(None);
        }
        let partition_size = unsafe { (*partition).size } as usize;

        // The header is read rather than mapped: its row count decides how much
        // to map, and mapping the whole 960 KB partition to find out would ask
        // the MMU for address space the image may not need.
        let mut header = [0u8; ROWS_OFFSET];
        EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_read(
                partition,
                0,
                header.as_mut_ptr().cast(),
                header.len(),
            )
        })
        .context("read the training partition header")?;

        if header[..MAGIC.len()] != MAGIC {
            info!("training partition holds no row image; live rows only");
            return Ok(None);
        }
        let version = read_u32(&header, 8);
        if version != VERSION {
            bail!("training image is version {version}, this build reads {VERSION}");
        }
        let precision = match read_u32(&header, 12) {
            0 => FeaturePrecision::Float32,
            1 => FeaturePrecision::Float16,
            2 => FeaturePrecision::Int8,
            other => bail!("training image names precision {other}"),
        };
        let row_count = read_u32(&header, 16) as usize;
        let row_bytes = read_u32(&header, 20) as usize;
        // The stride is in the image so a mismatch is caught here, where it can
        // be named, rather than as rows that decode into nonsense.
        if row_bytes != precision.row_bytes() {
            bail!(
                "training image says {row_bytes} bytes per row, this build packs {}",
                precision.row_bytes()
            );
        }
        let rows_bytes = row_count
            .checked_mul(row_bytes)
            .context("training image row count overflows")?;
        if ROWS_OFFSET + rows_bytes > partition_size {
            bail!(
                "training image claims {row_count} rows ({rows_bytes} bytes), partition holds {partition_size}"
            );
        }
        let quantization = Int8Quantization::from_bits(&header[HEADER_BYTES..ROWS_OFFSET])
            .context("training image quantization constants")?;

        // Mapped from offset zero so the mapping needs no alignment argument;
        // the rows are indexed inside the window instead.
        let length = ROWS_OFFSET + rows_bytes;
        let mut pointer = core::ptr::null();
        let mut handle = 0;
        EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_mmap(
                partition,
                0,
                length,
                esp_idf_svc::sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
                &mut pointer,
                &mut handle,
            )
        })
        .with_context(|| format!("map {length} bytes of the training partition"))?;

        // Safe: the mapping stays live until `munmap` in `Drop`, and the slice
        // never escapes this struct except by reborrow through `&self`.
        let mapped = unsafe { core::slice::from_raw_parts(pointer.cast::<u8>(), length) };
        info!(
            "training rows mapped: {row_count} rows of {row_bytes} bytes ({} KB) at {precision:?}",
            rows_bytes / 1024
        );
        Ok(Some(TrainingRows {
            handle,
            mapped,
            row_count,
            row_bytes,
            precision,
            quantization,
        }))
    }

    /// The packed rows, exactly a whole number of them.
    pub(crate) fn rows(&self) -> &[u8] {
        &self.mapped[ROWS_OFFSET..ROWS_OFFSET + self.row_count * self.row_bytes]
    }

    pub(crate) fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) fn precision(&self) -> FeaturePrecision {
        self.precision
    }

    pub(crate) fn quantization(&self) -> Int8Quantization {
        self.quantization
    }

    /// Read every row byte once and return how long it took, in microseconds.
    ///
    /// A fit walks these rows once per step — 250 times — so on the full set it
    /// pulls well over a hundred megabytes through the flash cache, and that
    /// bandwidth is invisible in the fit's wall time. One pass measured here is
    /// what lets the wall time be attributed: multiply by the step count and
    /// whatever is left over is arithmetic. The accumulator is returned into a
    /// black box so the read cannot be optimized away.
    pub(crate) fn walk_microseconds(&self) -> u32 {
        let started = crate::device_now_us();
        let mut checksum = 0u32;
        for byte in self.rows() {
            checksum = checksum.wrapping_add(*byte as u32);
        }
        core::hint::black_box(checksum);
        (crate::device_now_us() - started) as u32
    }
}

impl Drop for TrainingRows {
    fn drop(&mut self) {
        unsafe { esp_idf_svc::sys::esp_partition_munmap(self.handle) };
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Find the `training` partition, or `None` if this build was flashed without
/// one.
fn find_partition() -> Option<*const esp_idf_svc::sys::esp_partition_t> {
    let label = c"training";
    let partition = unsafe {
        esp_idf_svc::sys::esp_partition_find_first(
            esp_idf_svc::sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
            esp_idf_svc::sys::esp_partition_subtype_t_ESP_PARTITION_SUBTYPE_ANY,
            label.as_ptr(),
        )
    };
    (!partition.is_null()).then_some(partition)
}

/// The v2 calibration partition: the shipped prior and the two wearer slots.
///
/// The whole partition is mapped as one read-only window, so the prior's rows
/// and a slot's rows are the same kind of borrow and the fit walks both in
/// place. `emg_runtime::flash_image` owns every offset and every validity rule;
/// this type owns the mapping, the erase and the writes.
///
/// # Why the writes take `&mut self`
///
/// A mapping cannot be live across a write to what it maps, and a fit cannot
/// hold a row borrow across a flush. Both are the same rule, and the borrow
/// checker is the cheapest place to enforce it: [`CalibrationPartition::prior`]
/// and [`CalibrationPartition::slot`] borrow `&self`, every write takes
/// `&mut self`, so a caller that tries to flush mid-pass does not compile.
///
/// # What a write costs
///
/// Every write and erase stalls the other core and blocks all non-IRAM code.
/// The erase is one call during the announced settling phase; a row flush is a
/// few small writes into pre-erased flash, **roughly 1–4 ms of cache-down
/// each**. Scheduling them between rounds, never inside a labeling window, is
/// the caller's job — each method returns the microseconds it stalled for so
/// telemetry can prove it happened where it was supposed to.
pub(super) struct CalibrationPartition {
    partition: *const esp_idf_svc::sys::esp_partition_t,
    handle: esp_idf_svc::sys::esp_partition_mmap_handle_t,
    mapped: &'static [u8],
    /// Rows flushed into each slot since it was erased.
    ///
    /// Tracked here rather than passed in, because it is the one number a
    /// caller cannot afford to get wrong: reading one row past what was
    /// actually written hands the fit erased flash, whose label byte is 0xFF,
    /// and a label of 255 indexes off the end of a 12-class probability vector.
    flushed: [usize; SLOT_COUNT],
}

impl CalibrationPartition {
    /// Map the partition and check the prior image. `None` when there is no
    /// partition or it holds no v2 prior — a plain absence, and the device
    /// simply has nothing to calibrate against.
    pub(super) fn map() -> Result<Option<CalibrationPartition>> {
        let Some(partition) = find_partition() else {
            info!("no training partition; calibration has no prior");
            return Ok(None);
        };
        let size = unsafe { (*partition).size } as usize;
        if size < PARTITION_BYTES {
            bail!("training partition is {size} bytes, the v2 layout needs {PARTITION_BYTES}");
        }
        let mut mapped = Self {
            partition,
            handle: 0,
            mapped: &[],
            flushed: [0; SLOT_COUNT],
        };
        mapped.open()?;
        match PriorImage::parse(mapped.mapped) {
            Ok(prior) => {
                info!(
                    "v2 prior mapped: {} rows x {} classes, hash {:08x}, {:?} standardization",
                    prior.row_count(),
                    prior.class_count(),
                    prior.hash(),
                    prior.standardization_variant(),
                );
                if prior.hash() != prior.computed_hash() {
                    bail!(
                        "prior image hash {:08x} does not match its own bytes ({:08x})",
                        prior.hash(),
                        prior.computed_hash()
                    );
                }
                // The image declares which statistics standardize live rows,
                // and this firmware implements exactly one of them. Accepting
                // an image that asks for the other and then standardizing by
                // the frozen prior anyway is the failure the header exists to
                // prevent: the rows and the recipe that reads them would have
                // travelled separately, and nothing downstream could tell.
                if prior.standardization_variant() != StandardizationVariant::FrozenPrior {
                    bail!(
                        "prior image asks for {:?} standardization; this firmware implements only \
                         {:?}, so it must not run this image",
                        prior.standardization_variant(),
                        StandardizationVariant::FrozenPrior
                    );
                }
            }
            Err(ImageError::Absent) => {
                info!("training partition holds no v2 prior image");
                return Ok(None);
            }
            Err(error) => bail!("v2 prior image: {}", error.as_str()),
        }
        Ok(Some(mapped))
    }

    fn open(&mut self) -> Result<()> {
        let mut pointer = core::ptr::null();
        let mut handle = 0;
        EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_mmap(
                self.partition,
                0,
                PARTITION_BYTES,
                esp_idf_svc::sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
                &mut pointer,
                &mut handle,
            )
        })
        .context("map the training partition")?;
        // Safe: the mapping stays live until `close`, and the slice never
        // escapes this struct except by reborrow through `&self`.
        self.mapped = unsafe { core::slice::from_raw_parts(pointer.cast::<u8>(), PARTITION_BYTES) };
        self.handle = handle;
        Ok(())
    }

    fn close(&mut self) {
        if self.handle != 0 {
            unsafe { esp_idf_svc::sys::esp_partition_munmap(self.handle) };
            self.handle = 0;
            self.mapped = &[];
        }
    }

    pub(super) fn prior(&self) -> PriorImage<'_> {
        PriorImage::parse(self.mapped).expect("checked at map time")
    }

    fn slot_bytes(&self, index: usize) -> &[u8] {
        let at = SLOT_OFFSETS[index];
        &self.mapped[at..at + SLOT_BYTES]
    }

    /// Validate one slot against the mapped prior. Every failure is named, so a
    /// dead slot is reported as dead rather than skipped.
    pub(super) fn slot(&self, index: usize) -> Result<LiveSlot<'_>, ImageError> {
        flash_image::parse_slot(index, self.slot_bytes(index), self.prior().hash())
    }

    /// The stored calibration to run, if any: the live slot with the highest
    /// sequence.
    pub(super) fn newest_slot(&self) -> Option<LiveSlot<'_>> {
        (0..SLOT_COUNT)
            .filter_map(|index| self.slot(index).ok())
            .max_by_key(|slot| slot.record.sequence)
    }

    /// The sequence each slot carries, or `None` where the slot is empty or its
    /// record does not survive its CRC. Eviction and the next sequence number
    /// are both read off this, so both callers see one answer.
    pub(super) fn sequences(&self) -> [Option<u32>; flash_image::SLOT_COUNT] {
        core::array::from_fn(|index| self.slot(index).ok().map(|slot| slot.record.sequence))
    }

    /// Whether a slot is already erased, so a calibration can start without
    /// erasing anything.
    ///
    /// A read, and the reason the erase moved to boot: erasing 128 KB is
    /// roughly forty-eight sector erases, each of which suspends the other core
    /// and takes the flash cache down with it for tens of milliseconds. That is
    /// survivable with nothing else running and fatal beside a front end
    /// servicing DRDY at 2 kHz — the bench never saw it because the bench has
    /// no acquisition.
    pub(super) fn slot_is_erased(&self, index: usize) -> bool {
        self.slot_bytes(index).iter().all(|byte| *byte == 0xFF)
    }

    /// Erase a whole slot in one call, before collection begins.
    ///
    /// This is the only erase in a calibration and the longest stall in it —
    /// 32 sectors of 4 KB. Nothing else may run during it, which is why the
    /// protocol puts it inside the announced settling phase. Returns the
    /// microseconds it took.
    pub(super) fn erase_slot_region(&mut self, index: usize) -> Result<u32> {
        self.close();
        let started = crate::device_now_us();
        let result = EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_erase_range(
                self.partition,
                SLOT_OFFSETS[index],
                SLOT_BYTES,
            )
        });
        let elapsed = (crate::device_now_us() - started) as u32;
        self.open()?;
        result.with_context(|| format!("erase calibration slot {index}"))?;
        self.flushed[index] = 0;
        info!("calibration slot {index} erased in {elapsed} us");
        Ok(elapsed)
    }

    /// Flush buffered rows into a pre-erased slot, starting at row
    /// `first_row`.
    ///
    /// `bytes` is whole rows in the v2 packing — `RowBuffer::as_bytes`. The
    /// caller owns the schedule: this must sit in the gap between rounds, with
    /// the fitter quiescent, because it stalls the other core. Returns the
    /// microseconds it stalled for.
    pub(super) fn append_rows_buffered(&mut self, index: usize, bytes: &[u8]) -> Result<u32> {
        if bytes.len() % ROW_STRIDE != 0 {
            bail!("{} bytes is not a whole number of rows", bytes.len());
        }
        let rows = bytes.len() / ROW_STRIDE;
        let first_row = self.flushed[index];
        let capacity = flash_image::slot_row_capacity();
        if first_row + rows > capacity {
            bail!(
                "rows {first_row}..{} past the slot's {capacity}",
                first_row + rows
            );
        }
        let at = SLOT_OFFSETS[index] + SLOT_ROWS_OFFSET + first_row * ROW_STRIDE;
        let elapsed = self
            .write(at, bytes)
            .with_context(|| format!("append {rows} rows to calibration slot {index}"))?;
        self.flushed[index] += rows;
        Ok(elapsed)
    }

    /// Rows flushed into a slot since it was erased.
    pub(super) fn flushed_row_count(&self, index: usize) -> usize {
        self.flushed[index]
    }

    /// The rows flushed into a slot so far, as a source the fit can walk.
    ///
    /// This is the live half of the training set during a calibration. The
    /// wearer's rows live in flash from the moment they are flushed, so the
    /// only RAM the collection needs is one round's worth of buffer — a whole
    /// slot is 184 KB and does not fit beside wifi and acquisition.
    ///
    /// The record is not committed yet and there is nothing to validate
    /// against: no magic, no CRC, no sequence. The extent comes from this
    /// type's own count of what it wrote, which is why
    /// [`CalibrationPartition::append_rows_buffered`] no longer takes a row
    /// offset — the one way to corrupt a fit here is to disagree with flash
    /// about how many rows are in it.
    ///
    /// The borrow is what enforces the write discipline: this takes `&self` and
    /// every write takes `&mut self`, so a fit holding these rows across a
    /// flush does not compile.
    pub(super) fn flushed_rows(&self, index: usize) -> RowSource<'_> {
        let at = SLOT_OFFSETS[index] + SLOT_ROWS_OFFSET;
        let end = at + self.flushed[index] * ROW_STRIDE;
        RowSource::new(&self.mapped[at..end]).expect("whole rows by construction")
    }

    /// Commit a slot: the metadata block, then the CRC last.
    ///
    /// The order is what makes a torn write detectable. A crash before the
    /// metadata leaves an erased magic; a crash between the metadata and the
    /// CRC leaves an erased CRC word, which cannot match. Either way the slot
    /// is dead and the previous calibration stays installed.
    ///
    /// The rows must already be flushed. `live_row_count` has to be exactly how
    /// many were — the CRC covers that many rows, so a count past the last
    /// flush would checksum erased bytes and validate on the way back in — so
    /// this refuses any count but [`CalibrationPartition::flushed_row_count`].
    pub(super) fn commit_record(
        &mut self,
        index: usize,
        record: &SlotRecord,
        live_row_count: usize,
    ) -> Result<u32> {
        if live_row_count != self.flushed[index] {
            bail!(
                "committing {live_row_count} rows to slot {index} but {} were flushed",
                self.flushed[index]
            );
        }
        let metadata = record.to_metadata_block(live_row_count);
        let at = SLOT_OFFSETS[index];
        let mut elapsed = self
            .write(at, &metadata)
            .with_context(|| format!("write calibration slot {index} metadata"))?;

        let covered = flash_image::covered_bytes(live_row_count);
        let crc = {
            let slot = self.slot_bytes(index);
            flash_image::crc32(&slot[..covered])
        };
        elapsed += self
            .write(at + SLOT_CRC_OFFSET, &crc.to_le_bytes())
            .with_context(|| format!("commit calibration slot {index}"))?;
        info!(
            "calibration slot {index} committed: sequence {}, {live_row_count} rows, crc {crc:08x}, {elapsed} us of writes",
            record.sequence
        );
        Ok(elapsed)
    }

    /// One write, with the mapping dropped across it. Returns microseconds.
    fn write(&mut self, at: usize, bytes: &[u8]) -> Result<u32> {
        self.close();
        let started = crate::device_now_us();
        let result = EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_write(
                self.partition,
                at,
                bytes.as_ptr().cast(),
                bytes.len(),
            )
        });
        let elapsed = (crate::device_now_us() - started) as u32;
        self.open()?;
        result?;
        Ok(elapsed)
    }
}

impl Drop for CalibrationPartition {
    fn drop(&mut self) {
        self.close();
    }
}

/// A failure to map is not a reason to refuse to boot: report it and fit on
/// live rows. Returning the absence rather than the error keeps the one
/// decision — join flash or not — in one place.
pub(crate) fn map_or_warn() -> Option<TrainingRows> {
    match TrainingRows::map() {
        Ok(rows) => rows,
        Err(error) => {
            warn!("training rows unavailable: {error:#}");
            None
        }
    }
}
