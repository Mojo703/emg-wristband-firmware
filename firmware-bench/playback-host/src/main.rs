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
mod link;
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
use link::Link;
use protocol::{
    CalibrationGesture, CalibrationOutcome, Frame, CALIBRATION_SCHEDULE_ENTRY_BYTES,
    PLAYBACK_CHANNEL_COUNT, PLAYBACK_MAX_CHUNK_SAMPLES,
};
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
    /// Run a whole on-device calibration against a recorded session.
    ///
    /// The scripted wearer: the device's own state machine runs every phase,
    /// validity check, and fit, and the recording's cue spans stand in for a
    /// person. This is the hardware-validation vehicle — deterministic, because
    /// the flow is paced by the session's sample indices rather than by the
    /// wall clock, so the same bytes give the same run every time.
    Calibrate {
        /// A session manifest whose `cue_spans` become the prompt schedule.
        /// Repeatable, and **order matters**: the state machine collects the
        /// thumb-up block first, so the thumb-up session comes first. The two
        /// are spliced into one sample space, so a full run is
        /// `--manifest .../22-08-47.../manifest.json --manifest .../22-16-46.../manifest.json`.
        #[arg(long, required = true)]
        manifest: Vec<PathBuf>,
        #[arg(long, default_value_t = DEFAULT_CHUNK_SAMPLES)]
        chunk_samples: usize,
        /// How long to wait after the last sample for the run to finish. The
        /// polish passes happen in this window, so it has to cover them.
        #[arg(long, default_value_t = 120)]
        finish_seconds: u64,
        /// Request an abort as soon as the first fit checkpoint has begun.
        #[arg(long, conflicts_with = "abort_after_fit_passes")]
        abort_after_fit_start: bool,
        /// Request an abort after a fit checkpoint reports at least this many
        /// completed passes.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        abort_after_fit_passes: Option<u32>,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalibrationAbort {
    FitStarted,
    FitPasses(u32),
}

impl CalibrationAbort {
    fn from_args(after_start: bool, after_passes: Option<u32>) -> Option<Self> {
        if after_start {
            Some(Self::FitStarted)
        } else {
            after_passes.map(Self::FitPasses)
        }
    }

    fn reached(self, fit_passes_done: u32, fit_passes_planned: u32) -> bool {
        if fit_passes_planned == 0 {
            return false;
        }
        match self {
            Self::FitStarted => true,
            Self::FitPasses(threshold) => fit_passes_done >= threshold,
        }
    }
}

struct AbortController {
    policy: CalibrationAbort,
    requested: bool,
}

impl AbortController {
    fn new(policy: CalibrationAbort) -> Self {
        Self {
            policy,
            requested: false,
        }
    }

    fn request_if_due(&mut self, link: &mut Link, capture: &Capture) -> Result<()> {
        if self.requested || capture.has_calibration_result() {
            return Ok(());
        }
        let Some((phase, done, planned)) = capture
            .calibration_fit_progress()
            .rev()
            .find(|(_, done, planned)| self.policy.reached(*done, *planned))
        else {
            return Ok(());
        };
        eprintln!("requesting calibration abort in {phase:?} after {done}/{planned} fit passes");
        link.send(&Frame::CalibrationAbort {})?;
        self.requested = true;
        Ok(())
    }
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
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
        .filter(|step| matches!(step, Step::Calibrate { .. }))
        .count()
        > 1
    {
        bail!("plan contains more than one calibrate step");
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
            None,
        ),
        Step::Calibrate {
            manifest,
            chunk_samples,
            finish_seconds,
            abort_after_fit_start,
            abort_after_fit_passes,
        } => calibrate(
            link,
            capture,
            manifest,
            *chunk_samples,
            *finish_seconds,
            CalibrationAbort::from_args(*abort_after_fit_start, *abort_after_fit_passes),
        ),
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
    mut abort: Option<&mut AbortController>,
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
    if let Some(controller) = abort.as_mut() {
        controller.request_if_due(link, capture)?;
    }

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
            if let Some(controller) = abort.as_mut() {
                controller.request_if_due(link, capture)?;
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
    if let Some(controller) = abort.as_mut() {
        controller.request_if_due(link, capture)?;
    }

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

/// The prompt schedule a session's cue spans describe, shifted into the run's
/// own sample space.
///
/// Sample indices straight out of the manifest: the device labels by sample
/// index on its own grid, and the recording's cue clock is the same clock, so
/// nothing is converted and nothing rounds. `sample_offset` is how many
/// instants the sessions before this one contributed — a spliced run is one
/// monotonic space, because the labeling arithmetic cannot have time run
/// backwards halfway through.
fn cue_schedule(
    manifest_path: &Path,
    sample_offset: u64,
    block: u8,
) -> Result<(Vec<u8>, [u32; 5])> {
    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).context("parse manifest")?;
    let spans = json["cue_spans"]
        .as_array()
        .context("manifest has no cue_spans; this session cannot script a calibration")?;
    if spans.is_empty() {
        bail!("the manifest's cue_spans is empty; there is nothing to prompt from");
    }

    let mut entries = Vec::with_capacity(spans.len() * CALIBRATION_SCHEDULE_ENTRY_BYTES);
    let mut skipped = Vec::new();
    let mut written = 0usize;
    let mut per_gesture = [0u32; 5];
    for span in spans {
        let class_id = span["class_id"]
            .as_str()
            .context("cue span has no class_id")?;
        let Some(gesture) = gesture_for(class_id) else {
            // Named rather than counted: a session whose classes do not map is
            // the wrong session for this, and a silent zero-length schedule is
            // how that becomes a confusing device refusal instead.
            if !skipped.contains(&class_id.to_string()) {
                skipped.push(class_id.to_string());
            }
            continue;
        };
        let start = span["start"].as_u64().context("cue span has no start")?;
        let stop = span["stop"].as_u64().context("cue span has no stop")?;
        entries.extend_from_slice(&((start + sample_offset) as u32).to_le_bytes());
        entries.extend_from_slice(&((stop.saturating_sub(start)) as u32).to_le_bytes());
        entries.push(gesture.index());
        // The round this cue answers: its position among that gesture's cues
        // in this block. Load-bearing, not informational — the device looks a
        // cue up by (gesture, block, round), so that a rejected rep's retry
        // takes a spare rather than eating the next round's cue and starving
        // the block's last round many rounds later.
        entries.push(per_gesture[gesture.index() as usize] as u8);
        entries.push(block);
        entries.push(0);
        per_gesture[gesture.index() as usize] += 1;
        written += 1;
    }
    if !skipped.is_empty() {
        eprintln!("skipped cue classes with no calibration gesture: {skipped:?}");
    }
    if written == 0 {
        bail!("no cue span in this session maps to a calibration gesture");
    }
    eprintln!(
        "cue schedule: {written} prompts from {}",
        manifest_path.display()
    );
    Ok((entries, per_gesture))
}

/// Send the schedule, start the run, stream every session in turn, and wait
/// for it to finish.
fn calibrate(
    link: &mut Link,
    capture: &mut Capture,
    manifest_paths: &[PathBuf],
    chunk_samples: usize,
    finish_seconds: u64,
    abort: Option<CalibrationAbort>,
) -> Result<()> {
    // Both ends count from zero. The device's calibration sample space spans
    // sessions but resets here, so the reset has to come before the run
    // starts, not between the sessions.
    link.send(&Frame::BenchReset {})?;
    settle(link, capture, SETTLE);

    // The first session is the thumb-up block, the rest are thumb-down: the
    // state machine collects thumb-up first, so that is the order the
    // manifests have to be given in.
    const THUMB_UP_ROUNDS: u32 = 10;
    const THUMB_DOWN_ROUNDS: u32 = 12;
    let mut entries = Vec::new();
    let mut offset = 0u64;
    let mut per_block: Vec<(u8, [u32; 5])> = Vec::new();
    for (index, path) in manifest_paths.iter().enumerate() {
        let block = u8::from(index > 0);
        let manifest = read_manifest(path)?;
        let (bytes, per_gesture) = cue_schedule(path, offset, block)?;
        entries.extend_from_slice(&bytes);
        per_block.push((block, per_gesture));
        offset += (manifest.records * manifest.samples_per_record) as u64;
    }

    // Per gesture per block, not a total: a run ends when the gesture it is
    // asking for has no cue left, so a schedule with plenty of prompts overall
    // and none for ulnar deviation stops just as early. Worth knowing before a
    // hardware slot is spent rather than after.
    for (block, per_gesture) in &per_block {
        let wanted = if *block == 0 {
            THUMB_UP_ROUNDS
        } else {
            THUMB_DOWN_ROUNDS
        };
        let name = if *block == 0 {
            "thumb-up"
        } else {
            "thumb-down"
        };
        let thinnest = per_gesture.iter().copied().min().unwrap_or(0);
        if thinnest < wanted {
            eprintln!(
                "note: the {name} block has {thinnest} cues for its thinnest gesture against the \
                 {wanted} rounds its floor wants; the run ends when that gesture runs out and \
                 reports what it collected"
            );
        } else if thinnest == wanted {
            eprintln!(
                "note: the {name} block has exactly {wanted} cues for its thinnest gesture, so \
                 there is no slack — the first rejected rep for it ends the block"
            );
        }
    }

    // Batched so no frame outgrows what the device decodes comfortably. The
    // device refuses a schedule with a gap in its entry numbering, so a lost
    // frame stops the run rather than shifting every later prompt.
    const ENTRIES_PER_FRAME: usize = 64;
    for (index, batch) in entries
        .chunks(ENTRIES_PER_FRAME * CALIBRATION_SCHEDULE_ENTRY_BYTES)
        .enumerate()
    {
        link.send(&Frame::CalibrationCueSchedule {
            first_entry: (index * ENTRIES_PER_FRAME) as u32,
            entries: batch.to_vec(),
        })?;
    }
    link.send(&Frame::CalibrationStart {
        scripted_wearer: true,
    })?;
    settle(link, capture, SETTLE);
    if capture.failed() {
        bail!("the device refused the calibration");
    }

    // No reset between sessions: that would restart the calibration's sample
    // space in the middle of the run. Each `playback_begin` still builds a
    // fresh filter pipeline, which is right — filters carry state across a
    // session and must start from zero for the next one.
    let mut abort_controller = abort.map(AbortController::new);
    for path in manifest_paths {
        stream(
            link,
            capture,
            path,
            None,
            chunk_samples,
            None,
            abort_controller.as_mut(),
        )?;
        if capture.has_calibration_result() {
            break;
        }
    }

    // The last samples are not the end of the run: the polish passes and the
    // slot commit happen after them, and how long that takes is the number
    // this whole exercise exists to measure. So wait for the result frame
    // rather than for a fixed settle.
    eprintln!("streamed; waiting up to {finish_seconds}s for the run to finish");
    let deadline = Instant::now() + Duration::from_secs(finish_seconds);
    while Instant::now() < deadline && !capture.has_calibration_result() {
        let Some(frame) = link.receive(Duration::from_millis(500)) else {
            continue;
        };
        capture.accept(frame);
        if let Some(controller) = abort_controller.as_mut() {
            controller.request_if_due(link, capture)?;
        }
    }
    // Leave nothing running on the device for the next step to trip over.
    // Before the verdict, so a device mid-run is stopped either way.
    if abort_controller.is_none() {
        link.send(&Frame::CalibrationAbort {})?;
    }
    settle(link, capture, SETTLE);

    // The verdict. A run that aborted, failed its fit, lost its front end or
    // ran out of storage reports that in its result rather than as a
    // `bench_error` — so reading only the error list exits zero on every one
    // of them, and an orchestration cannot tell a calibration that installed
    // from one that gave up in round three.
    if let Some(controller) = abort_controller {
        if !controller.requested {
            bail!("the calibration ended before the configured fit abort milestone was reached");
        }
        return capture.assert_aborted_calibration();
    }
    match capture.calibration_outcome() {
        Some(CalibrationOutcome::Installed) => Ok(()),
        Some(outcome) => bail!("the calibration ended {outcome:?} without installing"),
        // The states collected so far still carry the pass timing, and they
        // are already written; the run is still a failure.
        None => bail!("no result frame inside {finish_seconds}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::{read_plan, Arguments, CalibrationAbort, Parser, Step};
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
    fn fit_started_waits_for_a_planned_checkpoint() {
        assert!(!CalibrationAbort::FitStarted.reached(0, 0));
        assert!(CalibrationAbort::FitStarted.reached(0, 16));
    }

    #[test]
    fn fit_pass_abort_waits_for_the_threshold() {
        let abort = CalibrationAbort::FitPasses(3);
        assert!(!abort.reached(2, 16));
        assert!(abort.reached(3, 16));
        assert!(abort.reached(4, 16));
    }

    #[test]
    fn plan_rejects_more_than_one_fit_step() {
        let path = write_plan("fit --model first\nfit --model second\n");
        let error = read_plan(&path).err().expect("duplicate fit must fail");
        std::fs::remove_file(path).unwrap();

        assert!(format!("{error:#}").contains("more than one fit step"));
    }

    #[test]
    fn plan_rejects_more_than_one_calibrate_step() {
        let path =
            write_plan("calibrate --manifest first.json\ncalibrate --manifest second.json\n");
        let error = read_plan(&path)
            .err()
            .expect("duplicate calibrate must fail");
        std::fs::remove_file(path).unwrap();

        assert!(format!("{error:#}").contains("more than one calibrate step"));
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
