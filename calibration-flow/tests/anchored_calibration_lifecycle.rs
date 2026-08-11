use calibration_flow::{
    anchored_class_index, anchored_labeled_span, AnchoredFitPlan, AnchoredFitStage,
    AnchoredRecipeProgress, AnchoredSong, AnchoredSongAction, AnchoredSongIdentity, Constants,
    RepEvidence, SongState, ACTIVE_CALIBRATION_CLASSES, ACTIVE_GESTURE_COUNT,
    ANCHORED_COMMAND_TARGET, ANCHORED_NO_OP_TARGET, CALIBRATION_MODEL_CLASS_COUNT,
    MAX_ANCHORED_SONG_CHUNK_CUES,
};
use emg_runtime::band_features::FEATURE_COUNT;
use emg_runtime::flash_image::{
    self, parse_slot, resident_promotion_crc, SlotRecord, SlotRole, SLOT_CRC_OFFSET,
    SLOT_PROMOTION_CRC_OFFSET, SLOT_ROLE_OFFSET, SLOT_ROWS_OFFSET,
};
use emg_runtime::streaming_fit::{
    FitCheckpoint, Fitter, RowBuffer, RowSource, Standardization, StandardizedQuantization,
    ROW_STRIDE,
};
use protocol::{
    CalibrationCueId, CalibrationRunId, CalibrationRunKey, CalibrationScheduleEntry,
    CalibrationScheduleRevision, CalibrationSessionId, DurationMilliseconds, TrackMilliseconds,
};

const CLASS_COUNT: usize = CALIBRATION_MODEL_CLASS_COUNT;
const CUES_PER_SONG: usize = 52;
const RECIPE_REP_COUNT: usize =
    ACTIVE_GESTURE_COUNT * ANCHORED_COMMAND_TARGET as usize + ANCHORED_NO_OP_TARGET as usize;
const RECIPE_ROW_COUNT: usize = RECIPE_REP_COUNT * Constants::DEFAULT.labeled_windows as usize;

fn run_key() -> CalibrationRunKey {
    CalibrationRunKey {
        session_id: CalibrationSessionId::new(41).unwrap(),
        run_id: CalibrationRunId::new(7).unwrap(),
    }
}

fn song_entries() -> Vec<CalibrationScheduleEntry> {
    let mut remaining = [10usize, 10, 6, 6, 5, 5, 5, 5];
    let mut entries = Vec::with_capacity(CUES_PER_SONG);
    while entries.len() < CUES_PER_SONG {
        for (class_index, (gesture, modifier)) in ACTIVE_CALIBRATION_CLASSES.into_iter().enumerate()
        {
            if remaining[class_index] == 0 {
                continue;
            }
            let index = entries.len();
            entries.push(CalibrationScheduleEntry {
                cue_id: CalibrationCueId::new(index as u32 + 1).unwrap(),
                gesture,
                modifier,
                track_offset: TrackMilliseconds::new(index as u32 * 2_000),
                hold: DurationMilliseconds::new(1_500),
            });
            remaining[class_index] -= 1;
        }
    }
    entries
}

fn nor_program(destination: &mut [u8], source: &[u8]) {
    assert_eq!(destination.len(), source.len());
    for (stored, desired) in destination.iter_mut().zip(source) {
        assert_eq!(
            *stored | *desired,
            *stored,
            "test attempted a forbidden NOR 0-to-1 transition"
        );
        *stored &= *desired;
    }
}

fn append_rep(slot: &mut [u8], row_count: &mut usize, entry: CalibrationScheduleEntry) {
    let constants = Constants::DEFAULT;
    let label = anchored_class_index(entry).expect("test cue is active") as u8;
    let standardization = Standardization {
        mean: [0.0; FEATURE_COUNT],
        deviation: [1.0; FEATURE_COUNT],
    };
    let mut rows = RowBuffer::with_capacity(constants.labeled_windows as usize);
    for window in 0..constants.labeled_windows {
        let mut features = [label as f32 + 1.0; FEATURE_COUNT];
        features[window as usize % FEATURE_COUNT] += 0.25;
        assert!(rows.push_calibration(
            &features,
            &standardization,
            &StandardizedQuantization::IDENTITY,
            label,
            ACTIVE_GESTURE_COUNT,
        ));
    }
    assert_eq!(rows.len(), constants.labeled_windows as usize);
    let first = SLOT_ROWS_OFFSET + *row_count * ROW_STRIDE;
    let end = first + rows.as_bytes().len();
    nor_program(&mut slot[first..end], rows.as_bytes());
    *row_count += rows.len();
    assert!(*row_count <= RECIPE_ROW_COUNT);
}

fn run_song(
    song: &mut AnchoredSong,
    revision: u32,
    progress: &mut AnchoredRecipeProgress,
    slot: &mut [u8],
    row_count: &mut usize,
    fit_plan: &mut AnchoredFitPlan,
    reject: impl Fn(CalibrationScheduleEntry) -> bool,
) {
    let constants = Constants::DEFAULT;
    let entries = song_entries();
    let identity = AnchoredSongIdentity::new(
        run_key(),
        CalibrationScheduleRevision::new(revision).unwrap(),
        format!("two-song-fixture-r{revision}"),
        entries.len() as u32,
    )
    .unwrap();
    song.begin_upload(identity.clone()).unwrap();
    for (chunk_index, chunk) in entries.chunks(MAX_ANCHORED_SONG_CHUNK_CUES).enumerate() {
        song.upload_chunk(
            &identity,
            (chunk_index * MAX_ANCHORED_SONG_CHUNK_CUES) as u32,
            chunk,
        )
        .unwrap();
    }
    let acknowledged = u64::from(revision) * 1_000_000_000;
    let acquisition_sample = u64::from(revision) * 10_000;
    let anchor = song
        .commit(&identity, acknowledged, acquisition_sample)
        .unwrap();

    for &entry in &entries {
        let opens =
            anchor.device_monotonic_microseconds + u64::from(entry.track_offset.get()) * 1_000;
        song.heartbeat(&identity, opens).unwrap();
        assert_eq!(
            song.poll(opens).unwrap(),
            Some(AnchoredSongAction::OpenCue {
                entry,
                device_monotonic_microseconds: opens,
            })
        );

        let span = anchored_labeled_span(constants, anchor, opens);
        assert_eq!(span.window_count, constants.labeled_windows);
        assert_eq!(span.windows().count(), constants.labeled_windows as usize);
        let invalid = reject(entry);
        let evidence = RepEvidence {
            windows_present: constants.labeled_windows - u32::from(invalid),
            windows_expected: span.window_count,
            ..RepEvidence::default()
        };
        let retains = !invalid && progress.retains_next(entry);

        let closes = opens + u64::from(entry.hold.get()) * 1_000;
        assert!(matches!(
            song.poll(closes).unwrap(),
            Some(AnchoredSongAction::CloseCue { entry: closed, .. }) if closed == entry
        ));
        let result = song
            .record_closed_evidence(evidence, if retains { constants.rows_per_rep() } else { 0 })
            .unwrap();
        if invalid {
            assert!(result.is_err());
            progress.record_rejected(entry);
        } else {
            assert!(result.is_ok());
            let recorded_retention = progress.record_accepted(entry);
            assert_eq!(recorded_retention, retains);
            if retains {
                append_rep(slot, row_count, entry);
                *fit_plan = fit_plan.request_checkpoint();
            }
        }
    }
    let last_close = anchor.device_monotonic_microseconds
        + u64::from(entries.last().unwrap().track_offset.get() + 1_500) * 1_000;
    assert_eq!(
        song.poll(last_close + 1).unwrap(),
        Some(AnchoredSongAction::Completed)
    );
    assert_eq!(song.state(), SongState::Completed);
}

#[test]
fn imperfect_song_continue_surplus_final_polish_crc_and_save_complete_in_seconds() {
    let mut song = AnchoredSong::new(run_key());
    let mut progress = AnchoredRecipeProgress::default();
    let mut slot = vec![0xFF; flash_image::SLOT_BYTES];
    let mut row_count = 0usize;
    let mut fit_plan = AnchoredFitPlan::Idle;

    // Song one is intentionally imperfect: each gesture's first soft-grip cue
    // is short. Every other class reaches its exact target.
    run_song(
        &mut song,
        1,
        &mut progress,
        &mut slot,
        &mut row_count,
        &mut fit_plan,
        |entry| matches!(entry.cue_id.get(), 2 | 6),
    );
    assert_eq!(
        row_count,
        (CUES_PER_SONG - ACTIVE_GESTURE_COUNT) * Constants::DEFAULT.rows_per_rep() as usize
    );
    assert_eq!(fit_plan, AnchoredFitPlan::Checkpoint);

    // Complete one coalesced checkpoint, just as recovery-gap work can finish
    // while the operator chooses a Continue track.
    let (stage, remainder) = fit_plan.take_first().unwrap();
    assert_eq!(stage, AnchoredFitStage::Checkpoint);
    fit_plan = remainder;
    song.fit_checkpoint_completed();

    // Song two fills every deficit. Its extra clean command cues are visible
    // successes but must not append a single surplus row.
    run_song(
        &mut song,
        2,
        &mut progress,
        &mut slot,
        &mut row_count,
        &mut fit_plan,
        |_| false,
    );
    assert_eq!(progress.retained_rep_count(), RECIPE_REP_COUNT as u32);
    assert_eq!(row_count, RECIPE_ROW_COUNT);
    assert_eq!(
        song.retained_progress().accepted_rows as usize,
        RECIPE_ROW_COUNT
    );

    // Save is the terminal edge: the queued checkpoint is owed first, then
    // exactly one final polish. Merely completing either song never requested
    // candidate metadata and the slot header is still erased.
    assert!(slot[..SLOT_ROWS_OFFSET].iter().all(|byte| *byte == 0xFF));
    fit_plan = fit_plan.request_finalization();
    assert_eq!(fit_plan, AnchoredFitPlan::CheckpointThenFinalize);
    let mut stages = Vec::new();
    while let Some((stage, remaining)) = fit_plan.take_first() {
        stages.push(stage);
        if stage == AnchoredFitStage::Checkpoint {
            song.fit_checkpoint_completed();
        }
        fit_plan = remaining;
    }
    assert_eq!(
        stages,
        [AnchoredFitStage::Checkpoint, AnchoredFitStage::FinalPolish]
    );

    // Exercise the production fitter over the exact packed live-row image.
    let rows_end = SLOT_ROWS_OFFSET + row_count * ROW_STRIDE;
    let source = RowSource::new(&slot[SLOT_ROWS_OFFSET..rows_end]).unwrap();
    let mut checkpoint = FitCheckpoint::zeroed(CLASS_COUNT);
    let mut fitter = Fitter::new(CLASS_COUNT);
    assert_eq!(
        fitter.resume_fit(
            &mut checkpoint,
            &StandardizedQuantization::IDENTITY,
            &[source],
            stages.len(),
        ),
        stages.len()
    );
    assert!(checkpoint.weights().iter().all(|value| value.is_finite()));

    // Candidate metadata and CRC are programmed only now, bytewise as NOR.
    let prior_hash = 0x8cf4_9324;
    let mut record = SlotRecord::empty(CLASS_COUNT);
    record.sequence = 8;
    record.role = SlotRole::ExportableCandidate;
    record.prior_hash = prior_hash;
    record.reference_gains = [1.25; 16];
    record.weights.copy_from_slice(checkpoint.weights());
    let metadata = record.to_metadata_block(row_count);
    nor_program(&mut slot[..SLOT_ROWS_OFFSET], &metadata);
    let covered = flash_image::covered_bytes(row_count);
    let candidate_crc = flash_image::crc32(&slot[..covered]);
    nor_program(
        &mut slot[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + 4],
        &candidate_crc.to_le_bytes(),
    );
    let candidate = parse_slot(1, &slot, prior_hash).unwrap();
    assert_eq!(candidate.record.role, SlotRole::ExportableCandidate);
    assert_eq!(candidate.rows().len(), row_count);
    assert_eq!(candidate.record.reference_gains, [1.25; 16]);

    // Save's promotion is promotion-CRC first and role bit last. Before the
    // role edge a reboot still sees a candidate; after it, the same exact rows
    // and model are a CRC-valid resident.
    let promotion_crc = resident_promotion_crc(&slot, row_count).unwrap();
    nor_program(
        &mut slot[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + 4],
        &promotion_crc.to_le_bytes(),
    );
    assert_eq!(
        parse_slot(1, &slot, prior_hash).unwrap().record.role,
        SlotRole::ExportableCandidate
    );
    nor_program(
        &mut slot[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + 4],
        &SlotRole::Resident.value().to_le_bytes(),
    );
    let resident = parse_slot(1, &slot, prior_hash).unwrap();
    assert_eq!(resident.record.role, SlotRole::Resident);
    assert_eq!(resident.rows().len(), row_count);
    assert_eq!(resident.crc, promotion_crc);
}
