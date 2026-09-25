//! The wearer's calibrated model, read once at boot from the `training`
//! partition and never written.
//!
//! `emg_runtime::flash_image` owns the layout and every validity rule. This
//! module maps the partition, picks the resident slot, copies its model and
//! reference gains out, and unmaps again.

use anyhow::{bail, Context, Result};
use calibration_flow::CALIBRATION_MODEL_CLASS_COUNT;
use emg_runtime::band_features::CHANNEL_COUNT;
use emg_runtime::calibration::{CalibrationModel, INPUT_COUNT};
use emg_runtime::flash_image::{
    self, ImageError, PriorImage, SlotRecord, SlotRole, StandardizationVariant, PARTITION_BYTES,
    SLOT_BYTES, SLOT_OFFSETS,
};
use emg_runtime::resident_selector::{PhysicalSlot, StoreSelector, StoredIdentity, StoredRole};
use esp_idf_svc::sys::EspError;
use log::{info, warn};

pub(crate) struct Resident {
    pub model: CalibrationModel,
    pub gains: [f32; CHANNEL_COUNT],
    pub slot: usize,
    pub sequence: u32,
}

fn is_usable(record: &SlotRecord) -> bool {
    record.class_count == CALIBRATION_MODEL_CLASS_COUNT
        && record.weights.len() == INPUT_COUNT * CALIBRATION_MODEL_CLASS_COUNT
        && record.reference_gains.iter().all(|value| value.is_finite())
        && record.mean.iter().all(|value| value.is_finite())
        && record
            .deviation
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && record.weights.iter().all(|value| value.is_finite())
}

/// Load the resident model. `Ok(None)` when the partition holds no prior or
/// no saved model: the band then runs without classifying.
pub(crate) fn load() -> Result<Option<Resident>> {
    let Some(mapping) = Mapping::open()? else {
        info!("no training partition; nothing to classify with");
        return Ok(None);
    };
    let bytes = mapping.bytes();
    let prior = match PriorImage::parse(bytes) {
        Ok(prior) => prior,
        Err(ImageError::Absent) => {
            info!("training partition holds no prior image");
            return Ok(None);
        }
        Err(error) => bail!("prior image: {}", error.as_str()),
    };
    check_prior(&prior)?;

    let slot_bytes = |slot: usize| &bytes[SLOT_OFFSETS[slot]..SLOT_OFFSETS[slot] + SLOT_BYTES];
    let stored = PhysicalSlot::ALL.map(|physical| {
        let slot = physical.index();
        match flash_image::parse_slot(slot, slot_bytes(slot), prior.hash()) {
            Ok(live) => Some(StoredIdentity {
                physical,
                generation: live.record.sequence,
                crc: live.crc,
                role: match live.record.role {
                    SlotRole::Resident => StoredRole::Resident,
                    SlotRole::ExportableCandidate => StoredRole::ExportableCandidate,
                    SlotRole::Inactive => StoredRole::Inactive,
                },
            }),
            Err(ImageError::Absent) => None,
            Err(error) => {
                warn!("slot {slot} unreadable: {}", error.as_str());
                None
            }
        }
    });
    let Some(resident) = StoreSelector::recover(stored).resident() else {
        info!("no saved calibration in either slot");
        return Ok(None);
    };
    let slot = resident.physical.index();
    let live = flash_image::parse_slot(slot, slot_bytes(slot), prior.hash())
        .map_err(|error| anyhow::anyhow!("slot {slot}: {}", error.as_str()))?;
    let record = &live.record;
    if !is_usable(record) {
        bail!(
            "resident sequence {} in slot {slot} is not a usable {CALIBRATION_MODEL_CLASS_COUNT}-class model",
            record.sequence
        );
    }
    Ok(Some(Resident {
        model: CalibrationModel::from_parts(
            record.class_count,
            &record.mean,
            &record.deviation,
            &record.weights,
        ),
        gains: record.reference_gains,
        slot,
        sequence: record.sequence,
    }))
}

fn check_prior(prior: &PriorImage<'_>) -> Result<()> {
    if prior.class_count() != CALIBRATION_MODEL_CLASS_COUNT {
        bail!(
            "prior has {} classes, this build scores {CALIBRATION_MODEL_CLASS_COUNT}",
            prior.class_count()
        );
    }
    if prior.hash() != prior.computed_hash() {
        bail!(
            "prior hash {:08x} does not match its bytes ({:08x})",
            prior.hash(),
            prior.computed_hash()
        );
    }
    if prior.standardization_variant() != StandardizationVariant::FrozenPrior {
        bail!(
            "prior asks for {:?} standardization; this firmware implements only {:?}",
            prior.standardization_variant(),
            StandardizationVariant::FrozenPrior
        );
    }
    Ok(())
}

/// The partition mapped read-only; unmapped on drop, so nothing borrowed from
/// it outlives the boot-time copy.
struct Mapping {
    handle: esp_idf_svc::sys::esp_partition_mmap_handle_t,
    bytes: &'static [u8],
}

impl Mapping {
    fn open() -> Result<Option<Self>> {
        let label = c"training";
        // SAFETY: the label is a valid NUL-terminated string and the returned
        // pointer is either null or a partition table entry valid for the
        // program's lifetime.
        let partition = unsafe {
            esp_idf_svc::sys::esp_partition_find_first(
                esp_idf_svc::sys::esp_partition_type_t_ESP_PARTITION_TYPE_DATA,
                esp_idf_svc::sys::esp_partition_subtype_t_ESP_PARTITION_SUBTYPE_ANY,
                label.as_ptr(),
            )
        };
        if partition.is_null() {
            return Ok(None);
        }
        // SAFETY: checked non-null above; partition entries are static.
        let size = unsafe { (*partition).size } as usize;
        if size < PARTITION_BYTES {
            bail!("training partition is {size} bytes, the layout needs {PARTITION_BYTES}");
        }
        let mut pointer = core::ptr::null();
        let mut handle = 0;
        // SAFETY: the partition pointer is valid and the out-pointers are live locals.
        EspError::convert(unsafe {
            esp_idf_svc::sys::esp_partition_mmap(
                partition,
                0,
                PARTITION_BYTES,
                esp_idf_svc::sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
                &mut pointer,
                &mut handle,
            )
        })
        .context("map the training partition")?;
        // SAFETY: the mapping covers PARTITION_BYTES and stays live until drop;
        // `bytes()` only lends it for the lifetime of `&self`.
        let bytes = unsafe { core::slice::from_raw_parts(pointer.cast::<u8>(), PARTITION_BYTES) };
        Ok(Some(Self { handle, bytes }))
    }

    fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful mmap and is unmapped once.
        unsafe { esp_idf_svc::sys::esp_partition_munmap(self.handle) };
    }
}
