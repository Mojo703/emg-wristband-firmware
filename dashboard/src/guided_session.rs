//! Shared ownership, presence, and reconnect projection for wearer-guided modes.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::watch;

macro_rules! revision_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u64);

        impl $name {
            pub const fn from_wire(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

revision_type!(SnapshotRevision);
revision_type!(RunRevision);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuidedSessionId(u64);

impl GuidedSessionId {
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 {
            None
        } else {
            Some(Self(value))
        }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceConnectionIdentity {
    pub device_id: String,
    pub connection_token: u64,
}

impl DeviceConnectionIdentity {
    pub fn new(device_id: impl Into<String>, connection_token: u64) -> Self {
        Self {
            device_id: device_id.into(),
            connection_token,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuidedMode {
    Collection,
    Calibration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidedSessionBinding {
    pub session_id: GuidedSessionId,
    pub run_revision: RunRevision,
    pub mode: GuidedMode,
    pub device: Option<DeviceConnectionIdentity>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidedFailureKind {
    DependencyFailed,
    TaskFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidedFailure {
    pub run_revision: RunRevision,
    pub mode: GuidedMode,
    pub kind: GuidedFailureKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidedSessionSnapshot {
    pub revision: SnapshotRevision,
    pub run_revision: RunRevision,
    pub visible_collection_views: usize,
    pub visible_calibration_views: usize,
    pub lifecycle: GuidedSessionLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuidedSessionLifecycle {
    Idle {
        calibration: Option<protocol::GuidedCalibrationSnapshot>,
    },
    Collection {
        binding: GuidedSessionBinding,
    },
    Calibration {
        binding: GuidedSessionBinding,
        calibration: Option<protocol::GuidedCalibrationSnapshot>,
    },
    CollectionFailed {
        failure: GuidedFailure,
    },
    CalibrationFailed {
        failure: GuidedFailure,
        calibration: Option<protocol::GuidedCalibrationSnapshot>,
    },
}

impl GuidedSessionSnapshot {
    pub const fn visible_views(&self, mode: GuidedMode) -> usize {
        match mode {
            GuidedMode::Collection => self.visible_collection_views,
            GuidedMode::Calibration => self.visible_calibration_views,
        }
    }

    pub const fn active(&self) -> Option<&GuidedSessionBinding> {
        match &self.lifecycle {
            GuidedSessionLifecycle::Collection { binding }
            | GuidedSessionLifecycle::Calibration { binding, .. } => Some(binding),
            GuidedSessionLifecycle::Idle { .. }
            | GuidedSessionLifecycle::CollectionFailed { .. }
            | GuidedSessionLifecycle::CalibrationFailed { .. } => None,
        }
    }

    pub const fn failure(&self) -> Option<&GuidedFailure> {
        match &self.lifecycle {
            GuidedSessionLifecycle::CollectionFailed { failure }
            | GuidedSessionLifecycle::CalibrationFailed { failure, .. } => Some(failure),
            GuidedSessionLifecycle::Idle { .. }
            | GuidedSessionLifecycle::Collection { .. }
            | GuidedSessionLifecycle::Calibration { .. } => None,
        }
    }

    pub const fn calibration(&self) -> Option<&protocol::GuidedCalibrationSnapshot> {
        match &self.lifecycle {
            GuidedSessionLifecycle::Idle { calibration }
            | GuidedSessionLifecycle::Calibration { calibration, .. }
            | GuidedSessionLifecycle::CalibrationFailed { calibration, .. } => calibration.as_ref(),
            GuidedSessionLifecycle::Collection { .. }
            | GuidedSessionLifecycle::CollectionFailed { .. } => None,
        }
    }

    pub fn to_wire(&self) -> protocol::GuidedSessionSnapshot {
        let lifecycle = match &self.lifecycle {
            GuidedSessionLifecycle::Idle { calibration } => protocol::GuidedSessionState::Idle {
                calibration: calibration.clone(),
            },
            GuidedSessionLifecycle::Collection { binding } => {
                protocol::GuidedSessionState::Collection {
                    session_id: binding.session_id.get(),
                    device_id: binding
                        .device
                        .as_ref()
                        .map(|device| device.device_id.clone()),
                }
            }
            GuidedSessionLifecycle::Calibration {
                binding,
                calibration,
            } => protocol::GuidedSessionState::Calibration {
                session_id: binding.session_id.get(),
                device_id: binding
                    .device
                    .as_ref()
                    .map(|device| device.device_id.clone()),
                calibration: calibration.clone(),
            },
            GuidedSessionLifecycle::CollectionFailed { failure } => {
                protocol::GuidedSessionState::CollectionFailed {
                    kind: failure.kind.into(),
                    detail: failure.detail.clone(),
                }
            }
            GuidedSessionLifecycle::CalibrationFailed {
                failure,
                calibration,
            } => protocol::GuidedSessionState::CalibrationFailed {
                kind: failure.kind.into(),
                detail: failure.detail.clone(),
                calibration: calibration.clone(),
            },
        };
        protocol::GuidedSessionSnapshot {
            revision: self.revision.get(),
            run_revision: self.run_revision.get(),
            visible_collection_views: self.visible_collection_views as u64,
            visible_calibration_views: self.visible_calibration_views as u64,
            lifecycle,
        }
    }
}

impl From<GuidedFailureKind> for protocol::GuidedFailureKind {
    fn from(value: GuidedFailureKind) -> Self {
        match value {
            GuidedFailureKind::DependencyFailed => Self::DependencyFailed,
            GuidedFailureKind::TaskFailed => Self::TaskFailed,
        }
    }
}

impl From<GuidedMode> for protocol::GuidedMode {
    fn from(value: GuidedMode) -> Self {
        match value {
            GuidedMode::Collection => Self::Collection,
            GuidedMode::Calibration => Self::Calibration,
        }
    }
}

impl From<protocol::GuidedMode> for GuidedMode {
    fn from(value: protocol::GuidedMode) -> Self {
        match value {
            protocol::GuidedMode::Collection => Self::Collection,
            protocol::GuidedMode::Calibration => Self::Calibration,
        }
    }
}

pub trait GuidedModeAdapter: Send + Sync + 'static {
    fn available(&self) -> bool {
        true
    }

    fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding);

    fn handle_intent(
        &self,
        _session: Option<&GuidedSessionBinding>,
        _device: Option<DeviceConnectionIdentity>,
        _request: GuidedIntentRequest,
    ) -> Result<(), CoordinatorError> {
        Err(CoordinatorError::ModeUnavailable(GuidedMode::Calibration))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRequest {
    pub expected_revision: SnapshotRevision,
    pub mode: GuidedMode,
    pub device: Option<DeviceConnectionIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidedIntentRequest {
    pub expected_revision: SnapshotRevision,
    pub expected_run_revision: RunRevision,
    pub expected_session_id: Option<GuidedSessionId>,
    pub action: protocol::GuidedSessionAction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorError {
    StaleRevision {
        expected: SnapshotRevision,
        received: SnapshotRevision,
    },
    LeaseHeld {
        mode: GuidedMode,
    },
    ModeUnavailable(GuidedMode),
    StaleRun {
        expected: RunRevision,
        received: RunRevision,
    },
    StaleSession {
        expected: Option<GuidedSessionId>,
        received: Option<GuidedSessionId>,
    },
    RevisionExhausted,
    LeaseMismatch,
    DeviceRequired,
    AdapterTaskUnavailable,
    AdapterTaskBusy,
    CalibrationTrackRequired,
    UnknownCalibrationTrack,
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for CoordinatorError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionExit {
    Completed,
    OperatorStopped,
    DependencyFailed(String),
    TaskFailed(String),
}

struct CoordinatorState {
    snapshot: GuidedSessionSnapshot,
    next_connection_id: u64,
    connections: HashMap<u64, Option<GuidedMode>>,
    adapters: HashMap<GuidedMode, Weak<dyn GuidedModeAdapter>>,
}

struct CoordinatorInner {
    state: Mutex<CoordinatorState>,
    snapshots: watch::Sender<GuidedSessionSnapshot>,
}

#[derive(Clone)]
pub struct GuidedSessionCoordinator {
    inner: Arc<CoordinatorInner>,
}

impl Default for GuidedSessionCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl GuidedSessionCoordinator {
    pub fn new() -> Self {
        let snapshot = GuidedSessionSnapshot {
            revision: SnapshotRevision(0),
            run_revision: RunRevision(0),
            visible_collection_views: 0,
            visible_calibration_views: 0,
            lifecycle: GuidedSessionLifecycle::Idle { calibration: None },
        };
        let (snapshots, _) = watch::channel(snapshot.clone());
        Self {
            inner: Arc::new(CoordinatorInner {
                state: Mutex::new(CoordinatorState {
                    snapshot,
                    next_connection_id: 0,
                    connections: HashMap::new(),
                    adapters: HashMap::new(),
                }),
                snapshots,
            }),
        }
    }

    pub fn set_mode_adapter<T>(&self, mode: GuidedMode, adapter: Arc<T>)
    where
        T: GuidedModeAdapter,
    {
        let adapter: Arc<dyn GuidedModeAdapter> = adapter;
        self.inner
            .state
            .lock()
            .unwrap()
            .adapters
            .insert(mode, Arc::downgrade(&adapter));
    }

    pub fn snapshot(&self) -> GuidedSessionSnapshot {
        self.inner.state.lock().unwrap().snapshot.clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<GuidedSessionSnapshot> {
        self.inner.snapshots.subscribe()
    }

    pub fn acquire(&self, request: SessionRequest) -> Result<SessionLease, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        check_revision(&state.snapshot, request.expected_revision)?;
        self.acquire_locked(&mut state, request.mode, request.device)
    }

    fn acquire_locked(
        &self,
        state: &mut CoordinatorState,
        mode: GuidedMode,
        device: Option<DeviceConnectionIdentity>,
    ) -> Result<SessionLease, CoordinatorError> {
        if let Some(active) = state.snapshot.active() {
            return Err(CoordinatorError::LeaseHeld { mode: active.mode });
        }
        let adapter_available = state
            .adapters
            .get(&mode)
            .and_then(Weak::upgrade)
            .is_some_and(|adapter| adapter.available());
        if !adapter_available {
            return Err(CoordinatorError::ModeUnavailable(mode));
        }
        let next_snapshot_revision = next_snapshot_revision(&state.snapshot)?;
        let run = state
            .snapshot
            .run_revision
            .0
            .checked_add(1)
            .ok_or(CoordinatorError::RevisionExhausted)?;
        let binding = GuidedSessionBinding {
            session_id: GuidedSessionId(run),
            run_revision: RunRevision(run),
            mode,
            device,
        };
        state.snapshot.run_revision = binding.run_revision;
        state.snapshot.lifecycle = match mode {
            GuidedMode::Collection => GuidedSessionLifecycle::Collection {
                binding: binding.clone(),
            },
            GuidedMode::Calibration => GuidedSessionLifecycle::Calibration {
                binding: binding.clone(),
                calibration: None,
            },
        };
        publish_locked(&self.inner, state, next_snapshot_revision);
        Ok(SessionLease {
            coordinator: Arc::downgrade(&self.inner),
            binding,
            lifecycle: LeaseLifecycle::Active,
        })
    }

    pub fn acquire_current(
        &self,
        mode: GuidedMode,
        device: Option<DeviceConnectionIdentity>,
    ) -> Result<SessionLease, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        self.acquire_locked(&mut state, mode, device)
    }

    pub fn update_calibration(
        &self,
        binding: &GuidedSessionBinding,
        calibration: protocol::GuidedCalibrationSnapshot,
    ) -> Result<GuidedSessionSnapshot, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        let next_revision = next_snapshot_revision(&state.snapshot)?;
        match &mut state.snapshot.lifecycle {
            GuidedSessionLifecycle::Calibration {
                binding: active,
                calibration: current,
            } if active == binding => *current = Some(calibration),
            _ => return Err(CoordinatorError::LeaseMismatch),
        }
        publish_locked(&self.inner, &mut state, next_revision);
        Ok(state.snapshot.clone())
    }

    pub fn update_idle_calibration(
        &self,
        calibration: protocol::GuidedCalibrationSnapshot,
    ) -> Result<GuidedSessionSnapshot, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        let next_revision = next_snapshot_revision(&state.snapshot)?;
        match state.snapshot.lifecycle {
            GuidedSessionLifecycle::Idle { .. }
            | GuidedSessionLifecycle::CollectionFailed { .. }
            | GuidedSessionLifecycle::CalibrationFailed { .. } => {
                state.snapshot.lifecycle = GuidedSessionLifecycle::Idle {
                    calibration: Some(calibration),
                };
            }
            GuidedSessionLifecycle::Collection { .. }
            | GuidedSessionLifecycle::Calibration { .. } => {
                return Err(CoordinatorError::LeaseMismatch);
            }
        }
        publish_locked(&self.inner, &mut state, next_revision);
        Ok(state.snapshot.clone())
    }

    pub fn handle_intent(&self, request: GuidedIntentRequest) -> Result<(), CoordinatorError> {
        self.handle_intent_for_device(request, None)
    }

    pub fn handle_intent_for_device(
        &self,
        request: GuidedIntentRequest,
        device: Option<DeviceConnectionIdentity>,
    ) -> Result<(), CoordinatorError> {
        let (adapter, active) = {
            let state = self.inner.state.lock().unwrap();
            check_intent_identity(&state.snapshot, &request)?;
            let adapter = state
                .adapters
                .get(&GuidedMode::Calibration)
                .and_then(Weak::upgrade)
                .filter(|adapter| adapter.available())
                .ok_or(CoordinatorError::ModeUnavailable(GuidedMode::Calibration))?;
            (adapter, state.snapshot.active().cloned())
        };
        adapter.handle_intent(active.as_ref(), device, request)
    }

    pub fn acquire_for_intent(
        &self,
        request: &GuidedIntentRequest,
        mode: GuidedMode,
        device: Option<DeviceConnectionIdentity>,
    ) -> Result<SessionLease, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        check_intent_identity(&state.snapshot, request)?;
        self.acquire_locked(&mut state, mode, device)
    }

    pub fn publish_calibration(
        &self,
        expected_revision: SnapshotRevision,
        expected_run_revision: RunRevision,
        expected_session_id: Option<GuidedSessionId>,
        calibration: protocol::GuidedCalibrationSnapshot,
    ) -> Result<GuidedSessionSnapshot, CoordinatorError> {
        let mut state = self.inner.state.lock().unwrap();
        check_identity(
            &state.snapshot,
            expected_revision,
            expected_run_revision,
            expected_session_id,
        )?;
        let next_revision = next_snapshot_revision(&state.snapshot)?;
        match &mut state.snapshot.lifecycle {
            GuidedSessionLifecycle::Idle {
                calibration: current,
            }
            | GuidedSessionLifecycle::Calibration {
                calibration: current,
                ..
            }
            | GuidedSessionLifecycle::CalibrationFailed {
                calibration: current,
                ..
            } => *current = Some(calibration),
            GuidedSessionLifecycle::Collection { .. }
            | GuidedSessionLifecycle::CollectionFailed { .. } => {
                return Err(CoordinatorError::LeaseMismatch);
            }
        }
        publish_locked(&self.inner, &mut state, next_revision);
        Ok(state.snapshot.clone())
    }

    pub fn connect_browser(&self) -> GuidedBrowserConnection {
        let id = {
            let mut state = self.inner.state.lock().unwrap();
            state.next_connection_id = state.next_connection_id.wrapping_add(1);
            let id = state.next_connection_id;
            state.connections.insert(id, None);
            id
        };
        GuidedBrowserConnection {
            coordinator: Arc::downgrade(&self.inner),
            id,
            lifecycle: BrowserConnectionLifecycle::Attached,
        }
    }

    #[cfg(test)]
    fn set_revisions_for_test(&self, snapshot: u64, run: u64) {
        let mut state = self.inner.state.lock().unwrap();
        state.snapshot.revision = SnapshotRevision(snapshot);
        state.snapshot.run_revision = RunRevision(run);
        self.inner.snapshots.send_replace(state.snapshot.clone());
    }
}

fn check_revision(
    snapshot: &GuidedSessionSnapshot,
    received: SnapshotRevision,
) -> Result<(), CoordinatorError> {
    if snapshot.revision == received {
        Ok(())
    } else {
        Err(CoordinatorError::StaleRevision {
            expected: snapshot.revision,
            received,
        })
    }
}

fn check_intent_identity(
    snapshot: &GuidedSessionSnapshot,
    request: &GuidedIntentRequest,
) -> Result<(), CoordinatorError> {
    check_identity(
        snapshot,
        request.expected_revision,
        request.expected_run_revision,
        request.expected_session_id,
    )
}

fn check_identity(
    snapshot: &GuidedSessionSnapshot,
    expected_revision: SnapshotRevision,
    expected_run_revision: RunRevision,
    expected_session_id: Option<GuidedSessionId>,
) -> Result<(), CoordinatorError> {
    check_revision(snapshot, expected_revision)?;
    if snapshot.run_revision != expected_run_revision {
        return Err(CoordinatorError::StaleRun {
            expected: snapshot.run_revision,
            received: expected_run_revision,
        });
    }
    let expected_session = snapshot.active().map(|active| active.session_id);
    if expected_session != expected_session_id {
        return Err(CoordinatorError::StaleSession {
            expected: expected_session,
            received: expected_session_id,
        });
    }
    Ok(())
}

fn next_snapshot_revision(
    snapshot: &GuidedSessionSnapshot,
) -> Result<SnapshotRevision, CoordinatorError> {
    snapshot
        .revision
        .0
        .checked_add(1)
        .map(SnapshotRevision)
        .ok_or(CoordinatorError::RevisionExhausted)
}

fn publish_locked(
    inner: &CoordinatorInner,
    state: &mut CoordinatorState,
    revision: SnapshotRevision,
) {
    state.snapshot.revision = revision;
    inner.snapshots.send_replace(state.snapshot.clone());
}

pub struct SessionLease {
    coordinator: Weak<CoordinatorInner>,
    binding: GuidedSessionBinding,
    lifecycle: LeaseLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseLifecycle {
    Active,
    Finished,
}

impl fmt::Debug for SessionLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionLease")
            .field("binding", &self.binding)
            .field("lifecycle", &self.lifecycle)
            .finish()
    }
}

impl SessionLease {
    pub fn binding(&self) -> &GuidedSessionBinding {
        &self.binding
    }

    pub fn finish(mut self, outcome: SessionExit) {
        self.lifecycle = LeaseLifecycle::Finished;
        release(&self.coordinator, &self.binding, outcome);
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        if matches!(self.lifecycle, LeaseLifecycle::Active) {
            release(
                &self.coordinator,
                &self.binding,
                SessionExit::TaskFailed("session task exited without an outcome".into()),
            );
        }
    }
}

fn release(
    coordinator: &Weak<CoordinatorInner>,
    binding: &GuidedSessionBinding,
    outcome: SessionExit,
) {
    let Some(inner) = coordinator.upgrade() else {
        return;
    };
    let mut state = inner.state.lock().unwrap();
    if state.snapshot.active() != Some(binding) {
        return;
    }
    let Ok(next_revision) = next_snapshot_revision(&state.snapshot) else {
        return;
    };
    let calibration = match &mut state.snapshot.lifecycle {
        GuidedSessionLifecycle::Calibration { calibration, .. } => calibration.take(),
        _ => None,
    };
    state.snapshot.lifecycle = match outcome {
        SessionExit::Completed | SessionExit::OperatorStopped => {
            GuidedSessionLifecycle::Idle { calibration }
        }
        SessionExit::DependencyFailed(detail) => failed_lifecycle(
            binding,
            GuidedFailureKind::DependencyFailed,
            detail,
            calibration,
        ),
        SessionExit::TaskFailed(detail) => {
            failed_lifecycle(binding, GuidedFailureKind::TaskFailed, detail, calibration)
        }
    };
    publish_locked(&inner, &mut state, next_revision);
}

fn failed_lifecycle(
    binding: &GuidedSessionBinding,
    kind: GuidedFailureKind,
    detail: String,
    calibration: Option<protocol::GuidedCalibrationSnapshot>,
) -> GuidedSessionLifecycle {
    let failure = GuidedFailure {
        run_revision: binding.run_revision,
        mode: binding.mode,
        kind,
        detail,
    };
    match binding.mode {
        GuidedMode::Collection => GuidedSessionLifecycle::CollectionFailed { failure },
        GuidedMode::Calibration => GuidedSessionLifecycle::CalibrationFailed {
            failure,
            calibration,
        },
    }
}

pub struct GuidedBrowserConnection {
    coordinator: Weak<CoordinatorInner>,
    id: u64,
    lifecycle: BrowserConnectionLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrowserConnectionLifecycle {
    Attached,
    Removed,
}

impl GuidedBrowserConnection {
    pub fn set_visible_mode(
        &self,
        visible_mode: Option<GuidedMode>,
    ) -> Result<GuidedSessionSnapshot, CoordinatorError> {
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(CoordinatorError::RevisionExhausted);
        };
        let (snapshot, pause) = {
            let mut state = inner.state.lock().unwrap();
            let Some(previous) = state.connections.get(&self.id).copied() else {
                return Ok(state.snapshot.clone());
            };
            if previous == visible_mode {
                return Ok(state.snapshot.clone());
            }
            let next_revision = next_snapshot_revision(&state.snapshot)?;
            if let Some(mode) = previous {
                decrement_visible(&mut state.snapshot, mode);
            }
            if let Some(mode) = visible_mode {
                increment_visible(&mut state.snapshot, mode);
            }
            state.connections.insert(self.id, visible_mode);
            let pause = previous.and_then(|mode| pause_target(&state, mode));
            publish_locked(&inner, &mut state, next_revision);
            (state.snapshot.clone(), pause)
        };
        invoke_pause(pause);
        Ok(snapshot)
    }

    pub fn disconnect(mut self) {
        self.remove();
    }

    fn remove(&mut self) {
        if matches!(self.lifecycle, BrowserConnectionLifecycle::Removed) {
            return;
        }
        self.lifecycle = BrowserConnectionLifecycle::Removed;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let pause = {
            let mut state = inner.state.lock().unwrap();
            let Some(visible_mode) = state.connections.get(&self.id).copied() else {
                return;
            };
            let Some(mode) = visible_mode else {
                state.connections.remove(&self.id);
                return;
            };
            let Ok(next_revision) = next_snapshot_revision(&state.snapshot) else {
                return;
            };
            state.connections.remove(&self.id);
            decrement_visible(&mut state.snapshot, mode);
            let pause = pause_target(&state, mode);
            publish_locked(&inner, &mut state, next_revision);
            pause
        };
        invoke_pause(pause);
    }
}

impl Drop for GuidedBrowserConnection {
    fn drop(&mut self) {
        self.remove();
    }
}

type PauseTarget = Option<(Arc<dyn GuidedModeAdapter>, GuidedSessionBinding)>;

fn increment_visible(snapshot: &mut GuidedSessionSnapshot, mode: GuidedMode) {
    match mode {
        GuidedMode::Collection => snapshot.visible_collection_views += 1,
        GuidedMode::Calibration => snapshot.visible_calibration_views += 1,
    }
}

fn decrement_visible(snapshot: &mut GuidedSessionSnapshot, mode: GuidedMode) {
    match mode {
        GuidedMode::Collection => snapshot.visible_collection_views -= 1,
        GuidedMode::Calibration => snapshot.visible_calibration_views -= 1,
    }
}

fn pause_target(state: &CoordinatorState, departed_mode: GuidedMode) -> PauseTarget {
    if state.snapshot.visible_views(departed_mode) != 0 {
        return None;
    }
    let active = state.snapshot.active().cloned()?;
    if active.mode != departed_mode {
        return None;
    }
    let adapter = state.adapters.get(&active.mode)?.upgrade()?;
    Some((adapter, active))
}

fn invoke_pause(target: PauseTarget) {
    if let Some((adapter, session)) = target {
        adapter.pause_for_no_visible_views(&session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct FakeModeAdapter {
        paused: Mutex<Vec<GuidedSessionId>>,
        actions: Mutex<Vec<protocol::GuidedSessionAction>>,
    }

    impl GuidedModeAdapter for FakeModeAdapter {
        fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
            self.paused.lock().unwrap().push(session.session_id);
        }

        fn handle_intent(
            &self,
            _session: Option<&GuidedSessionBinding>,
            _device: Option<DeviceConnectionIdentity>,
            request: GuidedIntentRequest,
        ) -> Result<(), CoordinatorError> {
            self.actions.lock().unwrap().push(request.action);
            Ok(())
        }
    }

    fn device() -> DeviceConnectionIdentity {
        DeviceConnectionIdentity::new("opal-1", 9)
    }

    fn coordinator() -> (GuidedSessionCoordinator, Arc<FakeModeAdapter>) {
        let coordinator = GuidedSessionCoordinator::new();
        let adapter = Arc::new(FakeModeAdapter::default());
        coordinator.set_mode_adapter(GuidedMode::Collection, adapter.clone());
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        (coordinator, adapter)
    }

    #[test]
    fn collection_and_calibration_are_mutually_exclusive() {
        let (coordinator, _adapter) = coordinator();
        let collection = coordinator
            .acquire(SessionRequest {
                expected_revision: coordinator.snapshot().revision,
                mode: GuidedMode::Collection,
                device: Some(device()),
            })
            .unwrap();

        assert_eq!(
            coordinator
                .acquire(SessionRequest {
                    expected_revision: coordinator.snapshot().revision,
                    mode: GuidedMode::Calibration,
                    device: Some(device()),
                })
                .unwrap_err(),
            CoordinatorError::LeaseHeld {
                mode: GuidedMode::Collection,
            }
        );

        collection.finish(SessionExit::Completed);
        let calibration = coordinator
            .acquire(SessionRequest {
                expected_revision: coordinator.snapshot().revision,
                mode: GuidedMode::Calibration,
                device: Some(device()),
            })
            .unwrap();
        assert_eq!(calibration.binding().run_revision.get(), 2);
        calibration.finish(SessionExit::OperatorStopped);
    }

    #[test]
    fn stale_intents_do_not_mutate_the_projection() {
        let (coordinator, _adapter) = coordinator();
        let stale = coordinator.snapshot().revision;
        let connection = coordinator.connect_browser();
        connection
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        let before = coordinator.snapshot();

        assert_eq!(
            coordinator
                .acquire(SessionRequest {
                    expected_revision: stale,
                    mode: GuidedMode::Collection,
                    device: Some(device()),
                })
                .unwrap_err(),
            CoordinatorError::StaleRevision {
                expected: before.revision,
                received: stale,
            }
        );
        assert_eq!(coordinator.snapshot(), before);
    }

    #[test]
    fn guided_actions_require_the_exact_snapshot_run_and_session_identity() {
        let (coordinator, adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Calibration, Some(device()))
            .unwrap();
        let snapshot = coordinator.snapshot();
        let request = GuidedIntentRequest {
            expected_revision: snapshot.revision,
            expected_run_revision: snapshot.run_revision,
            expected_session_id: snapshot.active().map(|active| active.session_id),
            action: protocol::GuidedSessionAction::PauseCalibration,
        };

        coordinator.handle_intent(request.clone()).unwrap();
        assert_eq!(
            adapter.actions.lock().unwrap().as_slice(),
            &[protocol::GuidedSessionAction::PauseCalibration]
        );

        let mut stale_run = request.clone();
        stale_run.expected_run_revision = RunRevision::from_wire(0);
        assert!(matches!(
            coordinator.handle_intent(stale_run),
            Err(CoordinatorError::StaleRun { .. })
        ));

        let mut stale_session = request;
        stale_session.expected_session_id = None;
        assert!(matches!(
            coordinator.handle_intent(stale_session),
            Err(CoordinatorError::StaleSession { .. })
        ));
        assert_eq!(adapter.actions.lock().unwrap().len(), 1);
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn adapter_mutation_rechecks_identity_after_dispatch_validation() {
        let (coordinator, _adapter) = coordinator();
        let before = coordinator.snapshot();
        let request = GuidedIntentRequest {
            expected_revision: before.revision,
            expected_run_revision: before.run_revision,
            expected_session_id: None,
            action: protocol::GuidedSessionAction::StartCalibration,
        };
        let connection = coordinator.connect_browser();
        connection
            .set_visible_mode(Some(GuidedMode::Calibration))
            .unwrap();

        assert!(matches!(
            coordinator.acquire_for_intent(&request, GuidedMode::Calibration, Some(device())),
            Err(CoordinatorError::StaleRevision { .. })
        ));
        assert!(coordinator.snapshot().active().is_none());
    }

    #[test]
    fn only_the_last_visible_guided_view_pauses_the_active_mode() {
        let (coordinator, adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Collection, Some(device()))
            .unwrap();
        let generic_socket = coordinator.connect_browser();
        let first = coordinator.connect_browser();
        let second = coordinator.connect_browser();

        first
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        second
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        generic_socket.disconnect();
        assert!(adapter.paused.lock().unwrap().is_empty());

        first.set_visible_mode(None).unwrap();
        assert!(adapter.paused.lock().unwrap().is_empty());
        second.set_visible_mode(None).unwrap();

        assert_eq!(
            adapter.paused.lock().unwrap().as_slice(),
            &[lease.binding().session_id]
        );
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn a_lagged_observer_resynchronizes_from_one_complete_snapshot() {
        let (coordinator, _adapter) = coordinator();
        let mut snapshots = coordinator.subscribe();
        let lease = coordinator
            .acquire_current(GuidedMode::Calibration, Some(device()))
            .unwrap();
        for _ in 0..20 {
            let connection = coordinator.connect_browser();
            connection
                .set_visible_mode(Some(GuidedMode::Calibration))
                .unwrap();
        }

        assert!(snapshots.has_changed().unwrap());
        let resynchronized = snapshots.borrow_and_update().clone();
        assert_eq!(resynchronized, coordinator.snapshot());
        assert_eq!(resynchronized.visible_views(GuidedMode::Calibration), 0);
        assert_eq!(resynchronized.active(), Some(lease.binding()));
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn reconnect_snapshot_contains_the_whole_current_projection() {
        let (coordinator, _adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Collection, Some(device()))
            .unwrap();
        let connection = coordinator.connect_browser();
        connection
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.active(), Some(lease.binding()));
        assert_eq!(snapshot.run_revision, lease.binding().run_revision);
        assert_eq!(snapshot.visible_views(GuidedMode::Collection), 1);
        assert_eq!(snapshot.failure(), None);
        lease.finish(SessionExit::Completed);
    }

    #[test]
    fn task_exit_without_an_outcome_releases_the_lease_and_publishes_failure() {
        let (coordinator, _adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Calibration, Some(device()))
            .unwrap();
        let run_revision = lease.binding().run_revision;
        drop(lease);

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.active(), None);
        assert_eq!(
            snapshot.failure(),
            Some(&GuidedFailure {
                run_revision,
                mode: GuidedMode::Calibration,
                kind: GuidedFailureKind::TaskFailed,
                detail: "session task exited without an outcome".into(),
            })
        );
        assert!(coordinator
            .acquire_current(GuidedMode::Collection, Some(device()))
            .is_ok());
    }

    #[test]
    fn dependency_failure_releases_the_lease_and_remains_in_the_snapshot() {
        let (coordinator, _adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Collection, Some(device()))
            .unwrap();
        let run_revision = lease.binding().run_revision;
        lease.finish(SessionExit::DependencyFailed("device link ended".into()));

        assert_eq!(
            coordinator.snapshot().failure(),
            Some(&GuidedFailure {
                run_revision,
                mode: GuidedMode::Collection,
                kind: GuidedFailureKind::DependencyFailed,
                detail: "device link ended".into(),
            })
        );
        assert!(coordinator.snapshot().active().is_none());
    }

    #[test]
    fn publishing_a_new_idle_calibration_clears_the_previous_run_failure() {
        let (coordinator, _adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Calibration, Some(device()))
            .unwrap();
        lease.finish(SessionExit::TaskFailed("old run failed".into()));
        assert!(coordinator.snapshot().failure().is_some());

        coordinator
            .update_idle_calibration(protocol::GuidedCalibrationSnapshot::Setup {
                tracks: Vec::new(),
                selected_track_id: None,
            })
            .unwrap();

        let snapshot = coordinator.snapshot();
        assert!(matches!(
            snapshot.lifecycle,
            GuidedSessionLifecycle::Idle {
                calibration: Some(protocol::GuidedCalibrationSnapshot::Setup { .. })
            }
        ));
        assert!(snapshot.failure().is_none());
    }

    #[test]
    fn presence_is_connection_local_mode_scoped_and_idempotent() {
        let (coordinator, adapter) = coordinator();
        let lease = coordinator
            .acquire_current(GuidedMode::Collection, Some(device()))
            .unwrap();
        let collection = coordinator.connect_browser();
        let calibration = coordinator.connect_browser();

        let first = collection
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        let duplicate = collection
            .set_visible_mode(Some(GuidedMode::Collection))
            .unwrap();
        assert_eq!(duplicate.revision, first.revision);
        calibration
            .set_visible_mode(Some(GuidedMode::Calibration))
            .unwrap();
        drop(collection);

        assert_eq!(
            adapter.paused.lock().unwrap().as_slice(),
            &[lease.binding().session_id]
        );
        assert_eq!(
            coordinator
                .snapshot()
                .visible_views(GuidedMode::Calibration),
            1
        );
        assert_eq!(
            coordinator.snapshot().visible_views(GuidedMode::Collection),
            0
        );
        lease.finish(SessionExit::Completed);
    }

    struct UnavailableAdapter;

    impl GuidedModeAdapter for UnavailableAdapter {
        fn available(&self) -> bool {
            false
        }

        fn pause_for_no_visible_views(&self, _session: &GuidedSessionBinding) {}
    }

    #[test]
    fn unavailable_mode_adapter_cannot_acquire_a_lease() {
        let coordinator = GuidedSessionCoordinator::new();
        let adapter = Arc::new(UnavailableAdapter);
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter);
        let before = coordinator.snapshot();

        assert_eq!(
            coordinator
                .acquire_current(GuidedMode::Calibration, Some(device()))
                .unwrap_err(),
            CoordinatorError::ModeUnavailable(GuidedMode::Calibration)
        );
        assert_eq!(coordinator.snapshot(), before);
    }

    #[test]
    fn exhausted_snapshot_revision_rejects_before_mutation() {
        let (coordinator, _adapter) = coordinator();
        coordinator.set_revisions_for_test(u64::MAX, 7);
        let before = coordinator.snapshot();

        assert_eq!(
            coordinator
                .acquire_current(GuidedMode::Collection, Some(device()))
                .unwrap_err(),
            CoordinatorError::RevisionExhausted
        );
        assert_eq!(coordinator.snapshot(), before);
    }
}
