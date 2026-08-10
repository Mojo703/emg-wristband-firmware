//! Drives the firmware validation bench from the host: stream a recorded
//! session into a bare ESP32-S3, load or fit a calibration model on it, replay
//! feature rows through it, and write everything it sends back to a directory.
//!
//! Every step is a subcommand and every subcommand is non-interactive, because
//! the parity orchestration calls this in a loop over sessions and fold models.
//! Device state — the loaded model, the stored calibration rows — survives
//! between invocations, so a case is a sequence of calls; `script` runs such a
//! sequence over one connection when the port open is worth avoiding.
//!
//! `firmware-bench/PROTOCOL.md` documents the frames and the sequences. The
//! arithmetic is not this tool's business: it moves the fixtures' exact bytes.

mod capture;
mod numpy;
mod partition_v2;
mod sources;

use anyhow::{bail, Context, Result};
use capture::Capture;
use clap::{Parser, Subcommand};
use emg_runtime::band_features::FEATURE_COUNT;
use emg_runtime::flash_image::{
    build_prior, whole_partition, PriorBuildInputs, PriorImage, StandardizationVariant,
};
use emg_runtime::streaming_fit::{Standardization, StandardizedQuantization};
use playback_host::link::Link;
use protocol::{
    CalibrationCueId, CalibrationGesture, CalibrationModifier, CalibrationRunId, CalibrationRunKey,
    CalibrationScheduleEntry, CalibrationScheduleRevision, CalibrationSessionId,
    DurationMilliseconds, Frame, TrackMilliseconds, CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES,
    PLAYBACK_CHANNEL_COUNT, PLAYBACK_MAX_CHUNK_SAMPLES,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Sample instants per chunk. A quarter of the 500-sample window, so a chunk
/// never straddles a window boundary — which is what makes the device's
/// per-window timing exact — and 4000 bytes, which leaves the credit window
/// three chunks deep inside the device's receive ring.
const DEFAULT_CHUNK_SAMPLES: usize = 125;

/// How long any single reply may take before the run is called dead. Generous:
/// a fit occupies the device for seconds and answers nothing meanwhile.
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);

/// The fit result alone waits longer: a full live-plus-flash fit walks
/// ~10k rows 250 times on a 240 MHz core, measured in whole minutes.
const FIT_RESULT_TIMEOUT: Duration = Duration::from_secs(900);

/// How long to keep draining after the last frame the caller was waiting for,
/// so trailing status and feature frames land in the capture.
const SETTLE: Duration = Duration::from_millis(400);

#[derive(Parser)]
#[command(about = "Host end of the ESP32-S3 firmware validation bench")]
struct Arguments {
    /// The device's USB serial port.
    #[arg(long, default_value = "/dev/ttyACM0")]
    port: PathBuf,
    /// Where captured output is written.
    #[arg(long, default_value = "bench-output")]
    output: PathBuf,
    #[command(subcommand)]
    step: Step,
}

#[derive(Subcommand, Clone)]
enum Step {
    /// Stream a recorded session's samples and capture the features it produces.
    Stream {
        /// A session manifest from `firmware-bench/fixtures/sessions/<name>`.
        #[arg(long)]
        manifest: PathBuf,
        /// The `emg.i16` to stream. Defaults to the path in the manifest.
        #[arg(long)]
        raw: Option<PathBuf>,
        #[arg(long, default_value_t = DEFAULT_CHUNK_SAMPLES)]
        chunk_samples: usize,
        /// Stop after this many 500-sample windows instead of the whole session.
        #[arg(long)]
        windows: Option<usize>,
    },
    /// Upload and run one complete device-anchored calibration song.
    CalibrationSong {
        /// Fixture manifests whose cue spans form one authored song. The first
        /// is thumb-up and every following manifest is thumb-down.
        #[arg(long, required = true)]
        manifest: Vec<PathBuf>,
        /// Exact guided-session identity chosen by the caller.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
        session_id: u64,
        /// Exact firmware-run identity chosen by the caller.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        run_id: u32,
        /// Fresh schedule revision for this upload.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        schedule_revision: u32,
        /// Maximum time to await a song result or interruption.
        #[arg(long, default_value_t = 120)]
        finish_seconds: u64,
    },
    /// Retain accepted evidence and make the next authored song eligible.
    CalibrationContinue {
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
        session_id: u64,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        run_id: u32,
    },
    /// Activate a numerically and storage-valid candidate calibration.
    CalibrationSave {
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
        session_id: u64,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        run_id: u32,
    },
    /// Discard the candidate and retain the previous resident calibration.
    CalibrationDiscard {
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..))]
        session_id: u64,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        run_id: u32,
    },
    /// Install a host-fitted calibration model.
    LoadModel {
        /// A model directory from `firmware-bench/fixtures/models/<name>`.
        #[arg(long)]
        model: PathBuf,
    },
    /// Score stored feature rows through the loaded model.
    Replay {
        /// Feature rows: a `.npy` of shape (rows, 64), or a flat little-endian
        /// `f32` blob.
        #[arg(long)]
        rows: PathBuf,
        #[arg(long, default_value_t = 0)]
        first_window: u32,
    },
    /// Fit calibration on the device from a model directory's training rows.
    Fit {
        #[arg(long)]
        model: PathBuf,
        /// Feature-row storage precision.
        #[arg(long, default_value = "f32")]
        precision: String,
        /// The `i8` affine constants, needed only at `i8` precision. Defaults to
        /// `feature_quantization.json` beside the fixtures.
        #[arg(long)]
        quantization: Option<PathBuf>,
        /// Store at most this many rows, to find where the device runs out.
        #[arg(long)]
        rows: Option<usize>,
        /// Stream only these sessions' rows. Repeatable; the complement of the
        /// rows placed in the v2 prior image.
        #[arg(long = "session")]
        sessions: Vec<String>,
        /// Stream only these roles' rows. Repeatable.
        #[arg(long = "role")]
        roles: Vec<String>,
        /// Join the flash training partition. Off measures how many rows RAM
        /// holds; on measures the live-plus-flash split. Two experiments, and
        /// the device refuses rather than silently running the other one.
        #[arg(long)]
        static_rows: bool,
    },
    /// Ask for a status frame.
    Status,
    /// Drop the device's session, model, and stored rows.
    Reset,
    /// Run one step per line of a file, over a single connection.
    Script {
        #[arg(long)]
        plan: PathBuf,
    },
    /// Build the v2 prior image: the same rows, standardized by the prior's
    /// own statistics and quantized to int8 against a standardized-feature
    /// affine, plus the warm-start weights the device fits from. Touches no
    /// port.
    BuildPartitionV2 {
        #[arg(long)]
        model: PathBuf,
        /// Leave these sessions' rows out — they are the ones a wearer
        /// collects live. Repeatable.
        #[arg(long = "exclude-session")]
        exclude_sessions: Vec<String>,
        /// Leave these roles' rows out. Repeatable.
        #[arg(long = "exclude-role")]
        exclude_roles: Vec<String>,
        /// Work package V's constants file, for the standardization variant it
        /// chose. Defaults to the fixtures' `calibration_constants.json` if it
        /// exists, and to the frozen prior otherwise.
        #[arg(long)]
        constants: Option<PathBuf>,
        /// Pack at most this many rows.
        #[arg(long)]
        rows: Option<usize>,
        /// Write the prior region alone rather than the whole partition. The
        /// whole partition is the default because flashing it also puts both
        /// wearer slots into a known erased state.
        #[arg(long)]
        prior_only: bool,
        #[arg(long)]
        image: PathBuf,
    },
    /// Report what a built partition image contains: the prior's header and
    /// hash, and each slot's sequence and whether it is whole.
    InspectPartition {
        #[arg(long)]
        image: PathBuf,
    },
    /// Read the fixtures a run would send and report what they contain, without
    /// touching the port. The hardware is serialized between workers, so this is
    /// how a fixture set is checked before a slot on it is spent.
    Inspect {
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        quantization: Option<PathBuf>,
        #[arg(long, default_value_t = DEFAULT_CHUNK_SAMPLES)]
        chunk_samples: usize,
    },
    /// Capture a fixed number of serial clock probes, then exit.
    #[cfg(feature = "clock-probe")]
    ClockCapture {
        #[arg(long, default_value_t = 1_000, value_parser = clap::value_parser!(u32).range(1..))]
        count: u32,
        #[arg(long, default_value_t = 10)]
        interval_milliseconds: u64,
        #[arg(long, default_value_t = 2_000)]
        timeout_milliseconds: u64,
        #[arg(long, default_value = "default-output")]
        output_device: String,
        #[arg(long, default_value_t = 0)]
        pause_revision: u64,
        #[arg(long, default_value_t = 10_000)]
        scheduling_horizon_milliseconds: u64,
        /// Board-evidence threshold. Omit to produce a provisional, non-GO fit.
        #[arg(long)]
        maximum_device_uncertainty_microseconds: Option<f64>,
        /// Board-evidence threshold for acquisition-sample mapping. Omit for provisional.
        #[arg(long)]
        maximum_acquisition_uncertainty_samples: Option<f64>,
    },
    /// Replay a clock capture and print its original accept/reject decision.
    #[cfg(feature = "clock-probe")]
    ClockReplay { capture: PathBuf },
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    #[cfg(feature = "clock-probe")]
    if let Step::ClockCapture {
        count,
        interval_milliseconds,
        timeout_milliseconds,
        output_device,
        pause_revision,
        scheduling_horizon_milliseconds,
        maximum_device_uncertainty_microseconds,
        maximum_acquisition_uncertainty_samples,
    } = &arguments.step
    {
        return playback_host::clock_capture::capture(
            playback_host::clock_capture::CaptureRequest {
                port_path: &arguments.port,
                output: &arguments.output,
                count: *count,
                interval: Duration::from_millis(*interval_milliseconds),
                timeout: Duration::from_millis(*timeout_milliseconds),
                output_device,
                pause_revision: *pause_revision,
                scheduling_horizon: Duration::from_millis(*scheduling_horizon_milliseconds),
                maximum_device_uncertainty_microseconds: *maximum_device_uncertainty_microseconds,
                maximum_acquisition_uncertainty_samples: *maximum_acquisition_uncertainty_samples,
            },
        );
    }
    #[cfg(feature = "clock-probe")]
    if let Step::ClockReplay { capture } = &arguments.step {
        println!(
            "accepted={}",
            playback_host::clock_capture::replay(capture)?
        );
        return Ok(());
    }
    if let Step::BuildPartitionV2 {
        model,
        exclude_sessions,
        exclude_roles,
        constants,
        rows,
        prior_only,
        image,
    } = &arguments.step
    {
        return build_partition_v2(
            model,
            constants.as_deref(),
            *rows,
            *prior_only,
            image,
            &sources::Selection {
                sessions: exclude_sessions.clone(),
                roles: exclude_roles.clone(),
                exclude: true,
            },
        );
    }
    if let Step::InspectPartition { image } = &arguments.step {
        let bytes = std::fs::read(image).with_context(|| format!("read {}", image.display()))?;
        println!("{} ({} bytes)", image.display(), bytes.len());
        print!("{}", partition_v2::describe(&bytes)?);
        return Ok(());
    }
    if let Step::Inspect {
        manifest,
        model,
        quantization,
        chunk_samples,
    } = &arguments.step
    {
        return inspect(
            manifest.as_deref(),
            model.as_deref(),
            quantization.as_deref(),
            *chunk_samples,
        );
    }
    let mut link = Link::open(&arguments.port)?;
    let mut capture = Capture::new(&arguments.output);

    let steps = match &arguments.step {
        Step::Script { plan } => read_plan(plan)?,
        step => vec![step.clone()],
    };
    let mut failure = None;
    for step in steps {
        if let Err(error) = run(&step, &mut link, &mut capture) {
            eprintln!("step failed: {error:#}");
            failure = Some(error);
            break;
        }
    }

    capture.write()?;
    eprintln!("wrote {}", arguments.output.display());
    match failure {
        Some(error) => Err(error),
        None if capture.failed() => bail!("the device refused part of the run; see errors.json"),
        None => Ok(()),
    }
}

/// One step per non-empty, non-comment line, parsed with the same grammar the
/// command line uses so a plan file and a shell loop cannot drift apart.
fn read_plan(path: &Path) -> Result<Vec<Step>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let steps: Vec<Step> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut words = vec!["playback-host"];
            words.extend(line.split_whitespace());
            Ok(Arguments::try_parse_from(words)
                .with_context(|| format!("plan line: {line}"))?
                .step)
        })
        .collect::<Result<_>>()?;
    if steps
        .iter()
        .filter(|step| matches!(step, Step::Fit { .. }))
        .count()
        > 1
    {
        bail!("plan contains more than one fit step");
    }
    if steps
        .iter()
        .filter(|step| matches!(step, Step::CalibrationSong { .. }))
        .count()
        > 1
    {
        bail!("plan contains more than one calibration-song step");
    }
    Ok(steps)
}

fn run(step: &Step, link: &mut Link, capture: &mut Capture) -> Result<()> {
    match step {
        Step::Stream {
            manifest,
            raw,
            chunk_samples,
            windows,
        } => stream(
            link,
            capture,
            manifest,
            raw.as_deref(),
            *chunk_samples,
            *windows,
        ),
        Step::CalibrationSong {
            manifest,
            session_id,
            run_id,
            schedule_revision,
            finish_seconds,
        } => calibration_song(
            link,
            capture,
            manifest,
            calibration_run(*session_id, *run_id),
            calibration_revision(*schedule_revision),
            *finish_seconds,
        ),
        Step::CalibrationContinue { session_id, run_id } => {
            link.send(&Frame::CalibrationContinue {
                run: calibration_run(*session_id, *run_id),
            })?;
            settle(link, capture, SETTLE);
            Ok(())
        }
        Step::CalibrationSave { session_id, run_id } => {
            link.send(&Frame::CalibrationSave {
                run: calibration_run(*session_id, *run_id),
            })?;
            await_resident_activation(link, capture)
        }
        Step::CalibrationDiscard { session_id, run_id } => {
            link.send(&Frame::CalibrationDiscard {
                run: calibration_run(*session_id, *run_id),
            })?;
            settle(link, capture, SETTLE);
            Ok(())
        }
        Step::LoadModel { model } => load_model(link, capture, model),
        Step::Replay { rows, first_window } => replay(link, capture, rows, *first_window),
        Step::Fit {
            model,
            precision,
            quantization,
            rows,
            sessions,
            roles,
            static_rows,
        } => fit(
            link,
            capture,
            &FitRequest {
                directory: model,
                precision,
                quantization_path: quantization.as_deref(),
                row_limit: *rows,
                selection: sources::Selection {
                    sessions: sessions.clone(),
                    roles: roles.clone(),
                    exclude: false,
                },
                use_static_rows: *static_rows,
            },
        ),
        Step::Status => {
            link.send(&Frame::BenchStatusRequest {})?;
            settle(link, capture, SETTLE);
            Ok(())
        }
        Step::Reset => {
            link.send(&Frame::BenchReset {})?;
            settle(link, capture, SETTLE);
            Ok(())
        }
        Step::Script { plan } => bail!("nested script {}", plan.display()),
        Step::Inspect { .. } => bail!("inspect does not run over a link"),
        Step::BuildPartitionV2 { .. } => bail!("build-partition-v2 does not run over a link"),
        Step::InspectPartition { .. } => bail!("inspect-partition does not run over a link"),
        #[cfg(feature = "clock-probe")]
        Step::ClockCapture { .. } => bail!("clock-capture owns its serial link"),
        #[cfg(feature = "clock-probe")]
        Step::ClockReplay { .. } => bail!("clock-replay does not run over a link"),
    }
}

/// The sessions whose rows are the wearer's own, collected live at calibration
/// time, and which therefore must not ship in the prior.
///
/// The bench's v1 image included them because it replayed the golden matrix as
/// recorded, where every row was training data. In the product they are the two
/// same-don blocks a wearer performs — the thumb-up commands and the thumb-down
/// no-ops — so shipping them would train the prior on the data the calibration
/// is about to collect, and would make every stored calibration look better
/// than it is.
const DEFAULT_LIVE_SESSIONS: [&str; 2] =
    ["2026-08-07T22-08-47_Matthew", "2026-08-07T22-16-46_Matthew"];

/// Build the v2 prior image from a model directory.
///
/// The rows are standardized here, once, by the statistics the host fit
/// produced — the same operation `standardized_training_rows.npy` records, and
/// the test in `partition_v2` holds this path to those bits. What the device
/// receives is therefore a set of rows it never has to standardize again.
fn build_partition_v2(
    directory: &Path,
    constants_path: Option<&Path>,
    row_limit: Option<usize>,
    prior_only: bool,
    image_path: &Path,
    selection: &sources::Selection,
) -> Result<()> {
    let raw = numpy::read(directory.join("training_rows.npy"))?;
    if raw.columns()? != FEATURE_COUNT {
        bail!("training rows have {} columns", raw.columns()?);
    }
    let labels = numpy::read(directory.join("training_labels.npy"))?;
    let calibration = calibration_fixtures(directory, constants_path)?;
    let warm_start = numpy::read(calibration.directory.join("calibration_prior_weights.npy"))
        .context("the prior model V fits over the prior rows alone")?;
    let mean = numpy::read(calibration.directory.join("calibration_prior_mean.npy"))?;
    let deviation = numpy::read(
        calibration
            .directory
            .join("calibration_prior_deviation.npy"),
    )?;
    let available = raw.rows();
    if labels.values.len() != available {
        bail!("{available} rows but {} labels", labels.values.len());
    }
    let class_count = warm_start.columns()?;

    // Standardize before selecting, so the statistics are the fit's own over
    // the whole set — the prior's statistics, which is what the device scores
    // through — and not the statistics of whatever subset ships.
    let statistics = Standardization {
        mean: mean.values.try_into().map_err(|values: Vec<f32>| {
            anyhow::anyhow!(
                "prior mean has {} values, want {FEATURE_COUNT}",
                values.len()
            )
        })?,
        deviation: deviation.values.try_into().map_err(|values: Vec<f32>| {
            anyhow::anyhow!(
                "prior deviation has {} values, want {FEATURE_COUNT}",
                values.len()
            )
        })?,
    };
    let standardized = partition_v2::standardize(&raw.values, &statistics);

    let breakdown = sources::read(directory, available)?;
    // Naming nothing means the product's split, not "ship everything": a
    // default that silently included the wearer's own rows is the mistake this
    // guards against.
    let default_selection;
    let selection = if selection.sessions.is_empty() && selection.roles.is_empty() {
        default_selection = sources::Selection {
            sessions: DEFAULT_LIVE_SESSIONS
                .iter()
                .map(|name| name.to_string())
                .collect(),
            roles: Vec::new(),
            exclude: true,
        };
        println!(
            "excluding the live sessions by default: {}",
            DEFAULT_LIVE_SESSIONS.join(", ")
        );
        &default_selection
    } else {
        selection
    };
    let (kept, indices) = selection.resolve(&breakdown)?;
    let selected_rows = sources::gather(&standardized, FEATURE_COUNT, &indices);
    let selected_labels: Vec<u8> = indices
        .iter()
        .map(|index| labels.values[*index] as u8)
        .collect();
    // A row stores its class scale and nothing else. The divisor is the count
    // of rows of that class present at a checkpoint, which grows every round,
    // so it is formed at fit time — the alternative would need a row in flash
    // to be rewritten, which flash does not do.
    let command_classes = command_class_count(directory)?;
    let selected_weights = prior_class_scales(&selected_labels, command_classes);
    let selected = indices.len();
    let rows = row_limit.unwrap_or(selected).min(selected);
    println!("prior rows from:\n  {}", sources::describe(&kept));
    if rows < selected {
        println!("  cut to {rows} of {selected} rows by --rows");
    }

    // The command columns are NOT zeroed. They carry no rows, but the softmax
    // gradient drives them negative over the prior fit, and that is what makes
    // the prior assert "not a command" everywhere — a device running the prior
    // alone puts at most 0.221 of its mass on the command classes and so cannot
    // commit. Zeroing them would throw that away. This is V's measured finding
    // in ARITHMETIC.md's calibration section, not a choice available here.
    let warm_start_weights = warm_start.values.clone();
    let largest_command = warm_start_weights
        .chunks_exact(class_count)
        .flat_map(|input| input[..command_classes.min(class_count)].iter())
        .fold(f32::MIN, |top, value| top.max(*value));
    println!(
        "warm-start weights: {class_count} classes from calibration_prior_weights.npy, command columns kept (largest {largest_command:e})"
    );

    let variant = standardization_variant(directory, constants_path)?;
    // One scale for every feature, zero offset: full scale at ten prior
    // deviations. Fitted to standardized values, so the per-feature constants
    // in feature_quantization.json — fitted to raw features — must not be used.
    let quantization = StandardizedQuantization::uniform(calibration.row_quantization_scale);
    println!(
        "row quantization: offset 0, scale {} for all {} features",
        calibration.row_quantization_scale, FEATURE_COUNT
    );
    let prior = build_prior(&PriorBuildInputs {
        class_count,
        standardized: &selected_rows[..rows * FEATURE_COUNT],
        labels: &selected_labels[..rows],
        class_scales: &selected_weights[..rows],
        warm_start_weights: &warm_start_weights,
        standardization: &statistics,
        quantization: &quantization,
        variant,
    })
    .map_err(|error| anyhow::anyhow!(error))?;

    // The safety property, checked on the rows the prior deliberately excludes:
    // the wearer's own reps, which are the closest thing to what a device
    // running the prior alone will actually see. If the prior can commit on
    // those, a failed calibration is worse than no calibration.
    let live_indices: Vec<usize> = (0..available)
        .filter(|index| !indices.contains(index))
        .collect();
    if !live_indices.is_empty() {
        let live_rows = sources::gather(&raw.values, FEATURE_COUNT, &live_indices);
        let (worst_total, worst_single) = partition_v2::command_mass(
            &live_rows,
            &statistics,
            &quantization,
            &warm_start_weights,
            class_count,
            command_classes,
        );
        println!(
            "prior alone over the {} live rows: at most {worst_total:.3} of a row's mass on the command classes, largest single class {worst_single:.3}, tau {}",
            live_indices.len(),
            partition_v2::REJECT_TAU,
        );
        if worst_single >= partition_v2::REJECT_TAU {
            bail!(
                "the prior model can commit: a command class reaches {worst_single:.3} against tau {}. \
                 A device whose calibration failed runs this model alone and must not be able to fire",
                partition_v2::REJECT_TAU
            );
        }
    }

    let image = if prior_only {
        prior
    } else {
        whole_partition(&prior).map_err(|error| anyhow::anyhow!(error))?
    };
    std::fs::write(image_path, &image)
        .with_context(|| format!("write {}", image_path.display()))?;
    let prior_hash = PriorImage::parse(&image)
        .map_err(|error| anyhow::anyhow!("built prior did not parse: {}", error.as_str()))?
        .hash();
    println!(
        "wrote {} : {rows} rows x {class_count} classes at 72 B, hash {:08x}, {} bytes",
        image_path.display(),
        prior_hash,
        image.len(),
    );
    print!("{}", partition_v2::describe(&image)?);
    println!(
        "flash with: espflash write-bin 0x310000 {}",
        image_path.display()
    );
    Ok(())
}

/// Where V's calibration constants and the prior model live, and the one
/// number from them the row packing needs.
struct CalibrationFixtures {
    directory: PathBuf,
    row_quantization_scale: f32,
    standardization_variant: String,
}

/// `calibration_constants.json` and the `.npy` files beside it. Defaults to the
/// fixtures directory the model sits under.
fn calibration_fixtures(
    model_directory: &Path,
    constants_path: Option<&Path>,
) -> Result<CalibrationFixtures> {
    let path = constants_path.map(Path::to_path_buf).unwrap_or_else(|| {
        model_directory
            .join("..")
            .join("..")
            .join("calibration_constants.json")
    });
    let directory = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let text = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "read {} — work package V publishes it, and the v2 image cannot be built without it",
            path.display()
        )
    })?;
    let json: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    // The bit pattern, not the decimal: the decimal in the file is the f32's
    // shortest representation, and reading it back through f64 is one rounding
    // away from what the device multiplies by.
    let scale = json["row_quantization"]["scale_bits"]
        .as_str()
        .map(|bits| {
            u32::from_str_radix(bits.trim_start_matches("0x"), 16)
                .map(f32::from_bits)
                .with_context(|| format!("row_quantization.scale_bits {bits}"))
        })
        .transpose()?
        .with_context(|| format!("{} names no row_quantization.scale_bits", path.display()))?;
    Ok(CalibrationFixtures {
        directory,
        row_quantization_scale: scale,
        standardization_variant: json["standardization"]["variant"]
            .as_str()
            .with_context(|| format!("{} names no standardization.variant", path.display()))?
            .to_string(),
    })
}

/// A row's class scale: 0.4 for a no-op class, 1.0 otherwise.
///
/// This is all a row stores. The fit divides by the count of rows of that class
/// present at each checkpoint, and that count grows with every collection
/// round — so it cannot live in a row, because a row in flash can never be
/// rewritten. ARITHMETIC.md calls the convention `checkpoint_counts`.
fn prior_class_scales(labels: &[u8], command_classes: usize) -> Vec<f32> {
    const NO_OP_SCALE: f32 = 0.4;
    labels
        .iter()
        .map(|label| {
            let class = *label as usize;
            // Classes run commands, then no-ops, then rest, five of each of the
            // first two.
            if (command_classes..2 * command_classes).contains(&class) {
                NO_OP_SCALE
            } else {
                1.0
            }
        })
        .collect()
}

/// How many of the classes are commands, from `model.json`. Row weights depend
/// on which classes are no-ops, so the count has to come from the model rather
/// than a constant here.
fn command_class_count(model_directory: &Path) -> Result<usize> {
    let path = model_directory.join("model.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parse model.json")?;
    json["number_of_commands"]
        .as_u64()
        .map(|count| count as usize)
        .with_context(|| format!("{} names no number_of_commands", path.display()))
}

/// Which statistics standardize the live rows: work package V's choice, read
/// from its constants file. The frozen prior is the fallback, because it is the
/// variant the plan describes and the one that needs no recomputation.
fn standardization_variant(
    model_directory: &Path,
    constants_path: Option<&Path>,
) -> Result<StandardizationVariant> {
    let calibration = calibration_fixtures(model_directory, constants_path)?;
    let name = calibration.standardization_variant.as_str();
    let variant = match name {
        "frozen_prior" | "frozen" => StandardizationVariant::FrozenPrior,
        "recomputed_per_round" | "recomputed" => StandardizationVariant::RecomputedPerRound,
        other => bail!("standardization must be frozen_prior or recomputed_per_round, not {other}"),
    };
    println!("standardization: {name}");
    Ok(variant)
}

/// Load everything a run would send and report it. Deliberately does the real
/// work — the same manifest read, the same model assembly, the same transpose —
/// so a fixture this accepts is one the streaming path will accept too.
fn inspect(
    manifest_path: Option<&Path>,
    model: Option<&Path>,
    quantization: Option<&Path>,
    chunk_samples: usize,
) -> Result<()> {
    if let Some(path) = manifest_path {
        let manifest = read_manifest(path)?;
        let scale = f32::from_bits(u32::from_le_bytes(
            manifest.constants[..4].try_into().expect("four bytes"),
        ));
        let gains: Vec<f32> = manifest.constants[4..]
            .chunks_exact(4)
            .map(|bits| f32::from_bits(u32::from_le_bytes(bits.try_into().expect("four bytes"))))
            .collect();
        println!("session {}", manifest.session);
        println!(
            "  {} records of {} samples = {} samples, {} windows",
            manifest.records,
            manifest.samples_per_record,
            manifest.records * manifest.samples_per_record,
            manifest.records * manifest.samples_per_record / 500
        );
        println!(
            "  scale_uv {scale} ({} bytes of constants)",
            manifest.constants.len()
        );
        println!(
            "  gains {:?} .. {:?}",
            &gains[..2],
            &gains[gains.len() - 2..]
        );
        println!(
            "  chunks of {chunk_samples} samples = {} bytes, {} per record",
            protocol::playback_chunk_bytes(chunk_samples),
            manifest.samples_per_record / chunk_samples.max(1)
        );
        if manifest.samples_per_record % chunk_samples != 0 {
            bail!("chunk_samples {chunk_samples} does not divide the record");
        }
        let raw = std::fs::metadata(&manifest.raw_path)
            .with_context(|| format!("stat {}", manifest.raw_path.display()))?;
        println!(
            "  raw stream {} ({} bytes, {} records available)",
            manifest.raw_path.display(),
            raw.len(),
            raw.len() as usize / (manifest.samples_per_record * PLAYBACK_CHANNEL_COUNT * 2)
        );
    }
    if let Some(directory) = model {
        let (class_count, bits) = model_bits(directory)?;
        println!("model {}", directory.display());
        println!("  {class_count} classes, {} bytes", bits.len());
        let rows = numpy::read(directory.join("training_rows.npy"))?;
        println!("  {} training rows of {}", rows.rows(), rows.columns()?);
        // The breakdown a split is cut from, so the v2 prior selection can be
        // checked before an image is built.
        for source in sources::read(directory, rows.rows())? {
            println!(
                "    {} {} {} rows: {} B in the v2 prior",
                source.session,
                source.role,
                source.rows,
                source.rows * emg_runtime::streaming_fit::ROW_STRIDE,
            );
        }
        println!(
            "  v2 prior holds {} standardized int8 rows",
            emg_runtime::flash_image::prior_row_capacity(),
        );
    }
    let quantization_path = quantization
        .map(Path::to_path_buf)
        .or_else(|| model.map(default_quantization_path));
    if let Some(path) = quantization_path {
        if path.exists() {
            let bits = read_quantization(&path)?;
            println!("quantization {} : {} bytes", path.display(), bits.len());
        }
    }
    Ok(())
}

/// A session manifest, as far as this tool reads it.
struct Manifest {
    session: String,
    raw_path: PathBuf,
    samples_per_record: usize,
    records: usize,
    /// Scale and the sixteen reference gains, already as little-endian bits.
    constants: Vec<u8>,
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parse manifest")?;

    let session = json["session"]
        .as_str()
        .context("manifest has no session")?
        .to_string();
    let raw = &json["raw_stream"];
    let samples_per_record = raw["samples_per_record"]
        .as_u64()
        .context("no samples_per_record")? as usize;
    let records = raw["records"].as_u64().context("no records")? as usize;
    let channels = raw["channels"].as_u64().context("no channels")? as usize;
    if channels != PLAYBACK_CHANNEL_COUNT {
        bail!("manifest has {channels} channels, the device expects {PLAYBACK_CHANNEL_COUNT}");
    }

    // The manifest publishes both the decimal value and the exact bit pattern
    // of every constant. The bits are what cross the wire: a decimal that
    // round-trips to a different low bit would move the whole filter chain, and
    // the comparison this feeds exists to catch differences that small.
    let mut constants = Vec::new();
    constants.extend_from_slice(&scale_bits(&json)?.to_le_bytes());
    let gain_bits = json["reference"]["gain_bits"]
        .as_array()
        .context("manifest has no reference.gain_bits")?;
    if gain_bits.len() != PLAYBACK_CHANNEL_COUNT {
        bail!(
            "manifest lists {} gains, want {PLAYBACK_CHANNEL_COUNT}",
            gain_bits.len()
        );
    }
    for entry in gain_bits {
        let text = entry.as_str().context("gain_bits entry is not a string")?;
        let bits = u32::from_str_radix(text.trim_start_matches("0x"), 16)
            .with_context(|| format!("gain bits {text}"))?;
        constants.extend_from_slice(&bits.to_le_bytes());
    }

    Ok(Manifest {
        session,
        raw_path: PathBuf::from(raw["path"].as_str().context("no raw_stream.path")?),
        samples_per_record,
        records,
        constants,
    })
}

/// `scale_uv` as bits.
///
/// The gains come with their exact bit patterns; the scale is published as a
/// decimal, and its decimal is not exactly representable in `f32`. Narrowing it
/// here is not a loss: the host simulation does `np.float32(scale_uv)` from the
/// same decimal, and both narrowings are round-to-nearest-even, so the device
/// gets the bits the host used. A `scale_uv_bits` field is preferred when the
/// fixtures carry one, which removes the argument entirely.
fn scale_bits(json: &serde_json::Value) -> Result<u32> {
    if let Some(text) = json["scale_uv_bits"].as_str() {
        return u32::from_str_radix(text.trim_start_matches("0x"), 16)
            .with_context(|| format!("scale_uv_bits {text}"));
    }
    let scale = json["scale_uv"]
        .as_f64()
        .context("manifest has no scale_uv")?;
    Ok((scale as f32).to_bits())
}

fn stream(
    link: &mut Link,
    capture: &mut Capture,
    manifest_path: &Path,
    raw_override: Option<&Path>,
    chunk_samples: usize,
    window_limit: Option<usize>,
) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    if chunk_samples == 0 || chunk_samples > PLAYBACK_MAX_CHUNK_SAMPLES {
        bail!("chunk_samples must be 1..={PLAYBACK_MAX_CHUNK_SAMPLES}");
    }
    if manifest.samples_per_record % chunk_samples != 0 {
        bail!(
            "chunk_samples {chunk_samples} must divide the {}-sample record, or a chunk \
             straddles a window boundary and the device's per-window timing smears",
            manifest.samples_per_record
        );
    }
    let raw_path = raw_override.unwrap_or(&manifest.raw_path);
    let raw = std::fs::read(raw_path).with_context(|| format!("read {}", raw_path.display()))?;

    let record_values = manifest.samples_per_record * PLAYBACK_CHANNEL_COUNT;
    let available = raw.len() / (record_values * 2);
    if available < manifest.records {
        bail!(
            "{} holds {available} records, the manifest says {}",
            raw_path.display(),
            manifest.records
        );
    }
    let records = window_limit
        .unwrap_or(manifest.records)
        .min(manifest.records);
    let sample_count = records * manifest.samples_per_record;

    link.send(&Frame::PlaybackBegin {
        session: manifest.session.clone(),
        sample_count: sample_count as u32,
        chunk_samples: chunk_samples as u32,
        constants: manifest.constants.clone(),
    })?;
    let (mut next_sequence, mut free_chunks) = await_credit(link, capture)?;

    eprintln!(
        "streaming {} : {records} records, {sample_count} samples, chunks of {chunk_samples}",
        manifest.session
    );
    let started = Instant::now();
    let mut sent: u32 = 0;
    let mut stalls = 0u32;
    let mut chunk = vec![0u8; chunk_samples * PLAYBACK_CHANNEL_COUNT * 2];

    for record in 0..records {
        let base = record * record_values * 2;
        for offset in (0..manifest.samples_per_record).step_by(chunk_samples) {
            // The recording is record-major — within a record the channels are
            // whole runs of 500 samples — and the device wants time-major
            // interleaved instants. The transpose happens here so the device's
            // hot path reads a chunk straight through.
            for step in 0..chunk_samples {
                for channel in 0..PLAYBACK_CHANNEL_COUNT {
                    let source = base + (channel * manifest.samples_per_record + offset + step) * 2;
                    let target = (step * PLAYBACK_CHANNEL_COUNT + channel) * 2;
                    chunk[target] = raw[source];
                    chunk[target + 1] = raw[source + 1];
                }
            }

            while sent >= next_sequence + free_chunks {
                stalls += 1;
                let (sequence, chunks) = await_credit(link, capture)?;
                next_sequence = sequence;
                free_chunks = chunks;
            }
            link.send(&Frame::PlaybackSamples {
                sequence: sent,
                samples: chunk.clone(),
            })?;
            sent += 1;

            // Credits and features arrive continuously; taking them here keeps
            // the reader's channel from growing to the size of the session.
            for frame in link.drain() {
                capture.accept(frame);
            }
            if let Some((sequence, chunks)) = capture.credit.take() {
                next_sequence = sequence;
                free_chunks = chunks;
            }
            if capture.failed() {
                bail!("the device abandoned the session at chunk {sent}");
            }
        }
    }

    link.send(&Frame::PlaybackEnd {})?;
    // The final status is the device saying it has drained everything, so it is
    // what "the stream is done" means here rather than the last write returning.
    await_status(link, capture)?;

    if capture.dropped_chunks().unwrap_or(0) != 0 {
        bail!(
            "the device refused {} chunks; the run lost samples",
            capture.dropped_chunks().unwrap_or(0)
        );
    }

    let seconds = started.elapsed().as_secs_f64();
    let bytes = sent as f64 * (chunk_samples * PLAYBACK_CHANNEL_COUNT * 2) as f64;
    eprintln!(
        "streamed {sent} chunks in {seconds:.1} s: {:.0} kB/s, {stalls} credit stalls, \
         {} windows captured",
        bytes / seconds / 1024.0,
        capture.windows()
    );
    Ok(())
}

fn load_model(link: &mut Link, capture: &mut Capture, directory: &Path) -> Result<()> {
    let (class_count, bits) = model_bits(directory)?;
    link.send(&Frame::BenchModelLoad {
        class_count: class_count as u32,
        model: bits,
    })?;
    settle(link, capture, SETTLE);
    if capture.failed() {
        bail!("the device refused the model");
    }
    eprintln!("loaded {} : {class_count} classes", directory.display());
    Ok(())
}

/// Assemble the model blob `CalibrationModel::from_bits` reads: mean,
/// deviation, then the weight matrix in the order numpy stored it.
fn model_bits(directory: &Path) -> Result<(usize, Vec<u8>)> {
    let mean = numpy::read(directory.join("standardization_mean.npy"))?;
    let deviation = numpy::read(directory.join("standardization_deviation.npy"))?;
    let weights = numpy::read(directory.join("weights.npy"))?;
    let class_count = weights.columns()?;
    if weights.rows() != protocol::BENCH_FEATURE_COUNT + 1 {
        bail!(
            "weights have {} rows, want {} features plus a bias row",
            weights.rows(),
            protocol::BENCH_FEATURE_COUNT
        );
    }
    let mut bits = mean.to_bits();
    bits.extend(deviation.to_bits());
    bits.extend(weights.to_bits());
    Ok((class_count, bits))
}

fn replay(link: &mut Link, capture: &mut Capture, rows: &Path, first_window: u32) -> Result<()> {
    let bits = feature_rows(rows)?;
    let row_bytes = protocol::BENCH_FEATURE_COUNT * 4;
    let count = bits.len() / row_bytes;
    // Batched so one frame stays well inside the device's 18 KB encode buffer
    // and its receive ring; the device answers with decisions per batch.
    const ROWS_PER_FRAME: usize = 32;
    for (batch, chunk) in bits.chunks(ROWS_PER_FRAME * row_bytes).enumerate() {
        link.send(&Frame::BenchReplayRows {
            first_window: first_window + (batch * ROWS_PER_FRAME) as u32,
            rows: chunk.to_vec(),
        })?;
        settle(link, capture, SETTLE);
        if capture.failed() {
            bail!("the device refused a replay batch");
        }
    }
    eprintln!("replayed {count} rows from {}", rows.display());
    Ok(())
}

/// Feature rows from a `.npy` of shape (rows, 64) or from a flat little-endian
/// `f32` blob — the second is what this tool's own `features.f32` capture is, so
/// a session's device-computed features can be replayed straight back.
fn feature_rows(path: &Path) -> Result<Vec<u8>> {
    let row_bytes = protocol::BENCH_FEATURE_COUNT * 4;
    if path.extension().is_some_and(|extension| extension == "npy") {
        let array = numpy::read(path)?;
        if array.columns()? != protocol::BENCH_FEATURE_COUNT {
            bail!(
                "{} has {} columns, want {}",
                path.display(),
                array.columns()?,
                protocol::BENCH_FEATURE_COUNT
            );
        }
        return Ok(array.to_bits());
    }
    let bits = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bits.len() % row_bytes != 0 {
        bail!("{} is not whole 64-feature rows", path.display());
    }
    Ok(bits)
}

/// One `fit` step's inputs. A struct rather than a long argument list because
/// the row selection and the static-rows switch are the two things a case
/// varies, and they are easy to transpose as positionals.
struct FitRequest<'a> {
    directory: &'a Path,
    precision: &'a str,
    quantization_path: Option<&'a Path>,
    row_limit: Option<usize>,
    selection: sources::Selection,
    use_static_rows: bool,
}

fn fit(link: &mut Link, capture: &mut Capture, request: &FitRequest<'_>) -> Result<()> {
    let FitRequest {
        directory,
        precision,
        quantization_path,
        row_limit,
        selection,
        use_static_rows,
    } = request;
    let (quantization_path, row_limit, use_static_rows) =
        (*quantization_path, *row_limit, *use_static_rows);
    let selector = match *precision {
        "f32" => 0u8,
        "f16" => 1,
        "i8" => 2,
        other => bail!("precision must be f32, f16 or i8, not {other}"),
    };
    let features = numpy::read(directory.join("training_rows.npy"))?;
    let labels = numpy::read(directory.join("training_labels.npy"))?;
    let weights = numpy::read(directory.join("row_weights.npy"))?;
    if features.columns()? != protocol::BENCH_FEATURE_COUNT {
        bail!("training rows have {} columns", features.columns()?);
    }
    let available = features.rows();
    if labels.values.len() != available || weights.values.len() != available {
        bail!(
            "{available} rows but {} labels and {} weights",
            labels.values.len(),
            weights.values.len()
        );
    }
    // Which of the model's rows this run streams. The complement goes to flash,
    // so the two selections have to be exact: the source breakdown is read and
    // checked against the matrix it describes rather than assumed.
    let breakdown = sources::read(directory, available)?;
    let (kept, indices) = selection.resolve(&breakdown)?;
    eprintln!("streaming rows from:\n  {}", sources::describe(&kept));
    let features_selected =
        sources::gather(&features.values, protocol::BENCH_FEATURE_COUNT, &indices);
    let labels_selected: Vec<f32> = indices.iter().map(|index| labels.values[*index]).collect();
    let weights_selected: Vec<f32> = indices.iter().map(|index| weights.values[*index]).collect();

    let selected = indices.len();
    let rows = row_limit.unwrap_or(selected).min(selected);
    let class_count = numpy::read(directory.join("weights.npy"))?.columns()?;

    let quantization = if selector == 2 {
        let path = quantization_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| default_quantization_path(directory));
        read_quantization(&path)?
    } else {
        Vec::new()
    };

    link.send(&Frame::BenchFitBegin {
        row_capacity: rows as u32,
        precision: selector,
        class_count: class_count as u32,
        quantization,
    })?;
    settle(link, capture, SETTLE);
    if capture.failed() {
        bail!("the device refused the store");
    }

    // Batched to keep each frame a few kilobytes: a row is 256 bytes of
    // features, and the device's receive ring is what bounds the frame.
    const ROWS_PER_FRAME: usize = 12;
    let mut sent = 0usize;
    while sent < rows {
        let batch = ROWS_PER_FRAME.min(rows - sent);
        let start = sent * protocol::BENCH_FEATURE_COUNT;
        let end = (sent + batch) * protocol::BENCH_FEATURE_COUNT;
        link.send(&Frame::BenchFitRows {
            labels: labels_selected[sent..sent + batch]
                .iter()
                .map(|label| *label as u8)
                .collect(),
            row_weights: weights_selected[sent..sent + batch]
                .iter()
                .flat_map(|weight| weight.to_bits().to_le_bytes())
                .collect(),
            rows: features_selected[start..end]
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        })?;
        sent += batch;
        for frame in link.drain() {
            capture.accept(frame);
        }
        if capture.failed() {
            bail!("the device stopped accepting rows after {sent}");
        }
    }
    // Nothing paces the row frames — only the sample stream is credit-gated —
    // so the device's own count is what confirms they all landed. A fit that
    // silently trained on fewer rows than were sent would produce a model that
    // disagrees with the host for a reason no comparison would name.
    link.send(&Frame::BenchStatusRequest {})?;
    settle(link, capture, SETTLE);
    match capture.stored_rows() {
        Some(stored) if stored as usize == sent => {}
        Some(stored) => bail!(
            "sent {sent} rows, the device holds {stored} ({} chunks refused)",
            capture.dropped_chunks().unwrap_or(0)
        ),
        None => bail!("the device did not report how many rows it holds"),
    }
    eprintln!(
        "stored {sent} rows at {precision}; fitting {}",
        if use_static_rows {
            "over live rows plus the flash partition"
        } else {
            "over live rows only"
        }
    );

    let started = Instant::now();
    link.send(&Frame::BenchFitRun { use_static_rows })?;
    while !capture.has_fit_result() {
        if started.elapsed() > FIT_RESULT_TIMEOUT {
            bail!("no fit result after {:?}", started.elapsed());
        }
        let Some(frame) = link.receive(Duration::from_secs(2)) else {
            continue;
        };
        capture.accept(frame);
        if capture.failed() {
            bail!("the device refused the fit");
        }
    }
    settle(link, capture, SETTLE);
    Ok(())
}

/// `feature_quantization.json` lives at the fixtures root, two levels above a
/// model directory.
fn default_quantization_path(model_directory: &Path) -> PathBuf {
    model_directory
        .parent()
        .and_then(Path::parent)
        .unwrap_or(Path::new("."))
        .join("feature_quantization.json")
}

/// The `i8` affine as the device wants it: per-feature offsets then per-feature
/// scales, from the fixture's exact bit patterns.
fn read_quantization(path: &Path) -> Result<Vec<u8>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parse quantization")?;
    let mut bits = Vec::new();
    for field in ["offset_bits", "scale_bits"] {
        let entries = json[field]
            .as_array()
            .with_context(|| format!("{} has no {field}", path.display()))?;
        if entries.len() != protocol::BENCH_FEATURE_COUNT {
            bail!("{field} has {} entries", entries.len());
        }
        for entry in entries {
            let text = entry.as_str().context("bits entry is not a string")?;
            let value = u32::from_str_radix(text.trim_start_matches("0x"), 16)
                .with_context(|| format!("{field} entry {text}"))?;
            bits.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bits)
}

/// Block until the device grants credit, folding everything else into the
/// capture on the way.
fn await_credit(link: &Link, capture: &mut Capture) -> Result<(u32, u32)> {
    if let Some(credit) = capture.credit.take() {
        return Ok(credit);
    }
    loop {
        let Some(frame) = link.receive(REPLY_TIMEOUT) else {
            bail!("no credit from the device");
        };
        capture.accept(frame);
        if capture.failed() {
            bail!("the device refused the session");
        }
        if let Some(credit) = capture.credit.take() {
            return Ok(credit);
        }
    }
}

/// Block until a status frame arrives, which is how the device says it has
/// drained what it was sent.
fn await_status(link: &Link, capture: &mut Capture) -> Result<()> {
    let deadline = Instant::now() + REPLY_TIMEOUT;
    while Instant::now() < deadline {
        let Some(frame) = link.receive(Duration::from_millis(500)) else {
            continue;
        };
        let was_status = matches!(frame, Frame::BenchStatus { .. });
        capture.accept(frame);
        if was_status {
            settle(link, capture, SETTLE);
            return Ok(());
        }
    }
    bail!("the device sent no closing status")
}

/// Drain whatever arrives for a moment, so trailing frames are captured.
fn settle(link: &Link, capture: &mut Capture, window: Duration) {
    let deadline = Instant::now() + window;
    while Instant::now() < deadline {
        match link.receive(Duration::from_millis(50)) {
            Some(frame) => capture.accept(frame),
            None => continue,
        }
    }
}

/// Which `CalibrationGesture` a session's `class_id` stands for.
///
/// The two thumb-modifier sessions name the same five wrist motions
/// differently — `thumb_up_pronation` in the thumb-up session, plain
/// `wrist_pronation` in the base one — and the hold gesture is `thumb_up_hold`
/// in one and `thumb_extension` in the other. The device's canonical five are
/// what the schedule has to speak, so the mapping lives here rather than in a
/// manifest nobody would think to check.
fn gesture_for(class_id: &str) -> Option<CalibrationGesture> {
    let base = class_id.strip_prefix("thumb_up_").unwrap_or(class_id);
    match base {
        "wrist_pronation" | "pronation" => Some(CalibrationGesture::WristPronation),
        "wrist_supination" | "supination" => Some(CalibrationGesture::WristSupination),
        "wrist_radial_deviation" | "radial_deviation" => {
            Some(CalibrationGesture::WristRadialDeviation)
        }
        "wrist_ulnar_deviation" | "ulnar_deviation" => {
            Some(CalibrationGesture::WristUlnarDeviation)
        }
        "thumb_extension" | "hold" => Some(CalibrationGesture::ThumbExtension),
        _ => None,
    }
}

const FIXTURE_SAMPLE_MILLISECONDS: u64 = 2;
const CALIBRATION_HOLD_MILLISECONDS: u32 = 1_500;

fn calibration_run(session_id: u64, run_id: u32) -> CalibrationRunKey {
    CalibrationRunKey {
        session_id: CalibrationSessionId::new(session_id).expect("clap rejects zero session ids"),
        run_id: CalibrationRunId::new(run_id).expect("clap rejects zero run ids"),
    }
}

fn calibration_revision(revision: u32) -> CalibrationScheduleRevision {
    CalibrationScheduleRevision::new(revision).expect("clap rejects zero schedule revisions")
}

/// Construct the complete, immutable schedule before writing a byte to the
/// device. A fixture provides cue positions, not device labels: the resulting
/// entries use audio-relative milliseconds and the fixed Tuesday hold.
fn anchored_schedule(
    manifest_paths: &[PathBuf],
) -> Result<(Vec<CalibrationScheduleEntry>, String)> {
    let mut entries = Vec::new();
    let mut sample_offset = 0u64;
    for (manifest_index, path) in manifest_paths.iter().enumerate() {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text).context("parse manifest")?;
        let spans = json["cue_spans"]
            .as_array()
            .context("manifest has no cue_spans; it cannot provide a calibration song")?;
        if spans.is_empty() {
            bail!("{} has no cue spans", path.display());
        }
        let modifier = if manifest_index == 0 {
            CalibrationModifier::ThumbUp
        } else {
            CalibrationModifier::ThumbDown
        };
        for span in spans {
            let class_id = span["class_id"]
                .as_str()
                .context("cue span has no class_id")?;
            let Some(gesture) = gesture_for(class_id) else {
                bail!("cue class {class_id:?} has no calibration gesture");
            };
            let start = span["start"].as_u64().context("cue span has no start")?;
            let track_milliseconds = (sample_offset + start)
                .checked_mul(FIXTURE_SAMPLE_MILLISECONDS)
                .context("fixture track offset overflow")?;
            let track_milliseconds = u32::try_from(track_milliseconds)
                .context("fixture track offset exceeds protocol range")?;
            let cue_id = u32::try_from(entries.len() + 1)
                .ok()
                .and_then(CalibrationCueId::new)
                .context("too many calibration schedule entries")?;
            entries.push(CalibrationScheduleEntry {
                cue_id,
                gesture,
                modifier,
                track_offset: TrackMilliseconds::new(track_milliseconds),
                hold: DurationMilliseconds::new(CALIBRATION_HOLD_MILLISECONDS),
            });
        }
        let manifest = read_manifest(path)?;
        sample_offset = sample_offset
            .checked_add((manifest.records * manifest.samples_per_record) as u64)
            .context("fixture sample offset overflow")?;
    }
    if entries.is_empty() {
        bail!("the manifests contain no calibration cues");
    }
    let canonical =
        serde_json::to_vec(&entries).context("serialize complete calibration schedule")?;
    let content_identity = format!("sha256:{:x}", Sha256::digest(canonical));
    Ok((entries, content_identity))
}

/// Upload one complete schedule, require the matching device anchor, then
/// maintain the calibration-specific liveness clock until the terminal song
/// event. No streamed fixture samples participate in label timing.
fn calibration_song(
    link: &mut Link,
    capture: &mut Capture,
    manifest_paths: &[PathBuf],
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    finish_seconds: u64,
) -> Result<()> {
    let (entries, content_identity) = anchored_schedule(manifest_paths)?;
    for frame in schedule_upload_frames(run, schedule_revision, &content_identity, &entries)? {
        link.send(&frame)?;
    }

    let deadline = Instant::now() + Duration::from_secs(finish_seconds);
    let accepted = await_schedule_accepted(link, capture, deadline)?;
    validate_schedule_acceptance(&accepted, run, schedule_revision, &content_identity)?;
    eprintln!(
        "schedule accepted: device anchor {} us, acquisition sample {}, content {}",
        accepted.anchor_device_monotonic_microseconds,
        accepted.acquisition_sample,
        accepted.content_identity
    );

    let heartbeat = link.start_calibration_heartbeats(run, schedule_revision)?;
    while Instant::now() < deadline {
        let Some(frame) = link.receive(Duration::from_millis(250)) else {
            continue;
        };
        capture.accept(frame);
        if capture.has_calibration_song_terminal_event() {
            return capture.assert_calibration_song_completed();
        }
    }
    drop(heartbeat);
    bail!("no calibration song result or interruption inside {finish_seconds}s")
}

/// Build the only legal transport sequence for one immutable schedule. Keeping
/// this separate from serial I/O makes the 130-cue boundary auditable without
/// pretending a host-side test can emulate device scheduling.
fn schedule_upload_frames(
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    content_identity: &str,
    entries: &[CalibrationScheduleEntry],
) -> Result<Vec<Frame>> {
    if entries.is_empty() {
        bail!("a calibration song must contain at least one cue");
    }
    let total_count = u32::try_from(entries.len()).context("too many schedule entries")?;
    let mut frames = Vec::with_capacity(
        2 + entries
            .len()
            .div_ceil(CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES),
    );
    frames.push(Frame::CalibrationScheduleBegin {
        run,
        schedule_revision,
        content_identity: content_identity.into(),
        total_count,
    });
    for (chunk_index, entries) in entries
        .chunks(CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
        .enumerate()
    {
        frames.push(Frame::CalibrationScheduleChunk {
            run,
            schedule_revision,
            content_identity: content_identity.into(),
            total_count,
            first_entry: u32::try_from(chunk_index * CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
                .expect("entry count already fits u32"),
            entries: entries.to_vec(),
        });
    }
    frames.push(Frame::CalibrationScheduleCommit {
        run,
        schedule_revision,
        content_identity: content_identity.into(),
        total_count,
    });
    Ok(frames)
}

fn validate_schedule_acceptance(
    accepted: &protocol::CalibrationScheduleAccepted,
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    content_identity: &str,
) -> Result<()> {
    if accepted.run != run
        || accepted.schedule_revision != schedule_revision
        || accepted.content_identity != content_identity
    {
        bail!("device accepted a different calibration schedule identity");
    }
    if !accepted.is_exactly_three_seconds_ahead() {
        bail!("device calibration anchor is not exactly three seconds after acknowledgement");
    }
    Ok(())
}

fn await_schedule_accepted(
    link: &Link,
    capture: &mut Capture,
    deadline: Instant,
) -> Result<protocol::CalibrationScheduleAccepted> {
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Some(frame) = link.receive(remaining.min(Duration::from_millis(250))) else {
            continue;
        };
        if let Frame::CalibrationScheduleAccepted { accepted } = &frame {
            return Ok(accepted.clone());
        }
        capture.accept(frame);
    }
    bail!("device did not acknowledge the complete calibration schedule")
}

fn await_resident_activation(link: &Link, capture: &mut Capture) -> Result<()> {
    let deadline = Instant::now() + REPLY_TIMEOUT;
    while Instant::now() < deadline {
        let Some(frame) = link.receive(Duration::from_millis(250)) else {
            continue;
        };
        capture.accept(frame);
        if capture.has_valid_resident_activation() {
            return Ok(());
        }
    }
    bail!("device did not acknowledge calibration resident activation")
}

#[cfg(test)]
mod tests {
    use super::{
        anchored_schedule, calibration_revision, calibration_run, read_plan,
        schedule_upload_frames, validate_schedule_acceptance, Arguments, Parser, Step,
    };
    use crate::capture::Capture;
    use protocol::{
        CalibrationClassCounts, CalibrationCueId, CalibrationGesture, CalibrationModifier,
        CalibrationScheduleAccepted, CalibrationScheduleEntry, CalibrationSongInterruption,
        CalibrationSongInterruptionReason, DurationMilliseconds, Frame, TrackMilliseconds,
    };
    use std::path::PathBuf;

    fn write_plan(contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "playback-host-plan-{}-{}.txt",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn plan_rejects_more_than_one_fit_step() {
        let path = write_plan("fit --model first\nfit --model second\n");
        let error = read_plan(&path).err().expect("duplicate fit must fail");
        std::fs::remove_file(path).unwrap();

        assert!(format!("{error:#}").contains("more than one fit step"));
    }

    #[test]
    fn plan_rejects_more_than_one_calibration_song_step() {
        let path = write_plan(
            "calibration-song --manifest first.json\ncalibration-song --manifest second.json\n",
        );
        let error = read_plan(&path)
            .err()
            .expect("duplicate calibration song must fail");
        std::fs::remove_file(path).unwrap();

        assert!(format!("{error:#}").contains("more than one calibration-song step"));
    }

    #[test]
    fn anchored_schedule_has_fixed_holds_and_a_stable_content_identity() {
        let manifest = std::env::temp_dir().join(format!(
            "playback-host-anchored-schedule-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &manifest,
            r#"{
                "session":"fixture",
                "raw_stream":{"path":"fixture.i16","samples_per_record":500,"records":20,"channels":16},
                "scale_uv":1.0,
                "reference":{"gain_bits":["0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000","0x00000000"]},
                "cue_spans":[
                    {"class_id":"thumb_up_pronation","start":100,"stop":200},
                    {"class_id":"thumb_up_hold","start":1100,"stop":1200}
                ]
            }"#,
        )
        .unwrap();
        let (entries, identity) = anchored_schedule(std::slice::from_ref(&manifest)).unwrap();
        let (_, repeated_identity) = anchored_schedule(std::slice::from_ref(&manifest)).unwrap();
        std::fs::remove_file(manifest).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].track_offset.get(), 200);
        assert_eq!(entries[1].track_offset.get(), 2_200);
        assert!(entries.iter().all(|entry| entry.hold.get() == 1_500));
        assert!(identity.starts_with("sha256:"));
        assert_eq!(identity, repeated_identity);
    }

    fn entries(count: u32) -> Vec<CalibrationScheduleEntry> {
        (1..=count)
            .map(|cue_id| CalibrationScheduleEntry {
                cue_id: CalibrationCueId::new(cue_id).unwrap(),
                gesture: CalibrationGesture::WristPronation,
                modifier: CalibrationModifier::ThumbUp,
                track_offset: TrackMilliseconds::new(cue_id * 2_000),
                hold: DurationMilliseconds::new(1_500),
            })
            .collect()
    }

    #[test]
    fn a_130_cue_song_uploads_as_32_32_32_32_2_with_one_content_identity() {
        let run = calibration_run(1, 2);
        let revision = calibration_revision(3);
        let frames =
            schedule_upload_frames(run, revision, "sha256:complete-song", &entries(130)).unwrap();
        assert_eq!(frames.len(), 7);
        assert!(matches!(
            &frames[0],
            Frame::CalibrationScheduleBegin { run: frame_run, schedule_revision, content_identity, total_count }
                if *frame_run == run && *schedule_revision == revision
                    && content_identity == "sha256:complete-song" && *total_count == 130
        ));
        let chunks: Vec<_> = frames[1..6]
            .iter()
            .map(|frame| match frame {
                Frame::CalibrationScheduleChunk {
                    run: frame_run,
                    schedule_revision,
                    content_identity,
                    total_count,
                    first_entry,
                    entries,
                } => {
                    assert_eq!(*frame_run, run);
                    assert_eq!(*schedule_revision, revision);
                    assert_eq!(content_identity, "sha256:complete-song");
                    assert_eq!(*total_count, 130);
                    (*first_entry, entries.len())
                }
                _ => panic!("expected calibration schedule chunk"),
            })
            .collect();
        assert_eq!(
            chunks,
            vec![(0, 32), (32, 32), (64, 32), (96, 32), (128, 2)]
        );
        assert!(matches!(
            &frames[6],
            Frame::CalibrationScheduleCommit { run: frame_run, schedule_revision, content_identity, total_count }
                if *frame_run == run && *schedule_revision == revision
                    && content_identity == "sha256:complete-song" && *total_count == 130
        ));
    }

    #[test]
    fn a_short_nonempty_song_still_has_begin_chunk_and_commit() {
        let run = calibration_run(1, 2);
        let revision = calibration_revision(3);
        let frames =
            schedule_upload_frames(run, revision, "sha256:short-song", &entries(1)).unwrap();

        assert_eq!(frames.len(), 3);
        assert!(matches!(
            &frames[0],
            Frame::CalibrationScheduleBegin { content_identity, total_count, .. }
                if content_identity == "sha256:short-song" && *total_count == 1
        ));
        assert!(matches!(
            &frames[1],
            Frame::CalibrationScheduleChunk { first_entry, entries, content_identity, total_count, .. }
                if *first_entry == 0 && entries.len() == 1
                    && content_identity == "sha256:short-song" && *total_count == 1
        ));
        assert!(matches!(
            &frames[2],
            Frame::CalibrationScheduleCommit { content_identity, total_count, .. }
                if content_identity == "sha256:short-song" && *total_count == 1
        ));
    }

    #[test]
    fn accepted_anchor_must_echo_the_complete_identity_and_be_exactly_three_seconds_ahead() {
        let run = calibration_run(4, 5);
        let revision = calibration_revision(6);
        let accepted = CalibrationScheduleAccepted {
            run,
            schedule_revision: revision,
            content_identity: "sha256:complete-song".into(),
            acknowledged_device_monotonic_microseconds: 70,
            anchor_device_monotonic_microseconds: 3_000_070,
            acquisition_sample: 123,
        };
        assert!(
            validate_schedule_acceptance(&accepted, run, revision, "sha256:complete-song").is_ok()
        );
        assert!(
            validate_schedule_acceptance(&accepted, run, revision, "sha256:other-song").is_err()
        );

        let mut wrong_anchor = accepted;
        wrong_anchor.anchor_device_monotonic_microseconds += 1;
        assert!(
            validate_schedule_acceptance(&wrong_anchor, run, revision, "sha256:complete-song")
                .is_err()
        );
    }

    #[test]
    fn interruption_continues_with_a_new_revision_and_never_reuploads_the_old_one() {
        let run = calibration_run(8, 9);
        let interrupted_revision = calibration_revision(10);
        let retry_revision = calibration_revision(11);
        let mut capture = Capture::new("unused");
        capture.accept(Frame::CalibrationSongInterrupted {
            interruption: CalibrationSongInterruption {
                run,
                schedule_revision: interrupted_revision,
                content_identity: "sha256:interrupted-song".into(),
                reason: CalibrationSongInterruptionReason::HeartbeatTimeout,
                open_cue: Some(CalibrationCueId::new(1).unwrap()),
                counts: vec![CalibrationClassCounts {
                    gesture: CalibrationGesture::ThumbExtension,
                    modifier: CalibrationModifier::ThumbUp,
                    accepted_count: 1,
                    rejected_count: 0,
                    target_count: 10,
                    deficit_count: 9,
                }],
            },
        });
        assert!(capture.has_calibration_song_terminal_event());
        assert!(capture.assert_calibration_song_completed().is_err());
        let continue_frame = Frame::CalibrationContinue { run };
        let retry =
            schedule_upload_frames(run, retry_revision, "sha256:retry-song", &entries(2)).unwrap();

        assert!(
            matches!(continue_frame, Frame::CalibrationContinue { run: frame_run } if frame_run == run)
        );
        assert_ne!(retry_revision, interrupted_revision);
        assert!(matches!(
            &retry[0],
            Frame::CalibrationScheduleBegin { schedule_revision, .. } if *schedule_revision == retry_revision
        ));
        assert!(retry.iter().all(|frame| !matches!(
            frame,
            Frame::CalibrationScheduleBegin { schedule_revision, .. }
            | Frame::CalibrationScheduleChunk { schedule_revision, .. }
            | Frame::CalibrationScheduleCommit { schedule_revision, .. }
                if *schedule_revision == interrupted_revision
        )));
    }

    #[test]
    fn the_legacy_partition_builder_is_not_a_cli_command() {
        assert!(Arguments::try_parse_from(["playback-host", "build-partition"]).is_err());
        assert!(matches!(
            Arguments::try_parse_from([
                "playback-host",
                "build-partition-v2",
                "--model",
                "model",
                "--image",
                "image.bin"
            ])
            .unwrap()
            .step,
            Step::BuildPartitionV2 { .. }
        ));
    }
}
