//! Link routing: exactly one link carries the data stream at a time, and the most
//! recently established link wins. A dashboard probing the serial port claims it
//! (heartbeats keep the claim alive; silence, unplug, or a stalled write releases
//! it), and TCP carries the stream otherwise. The claim/stall/cooldown decisions
//! live in the [`policy`] module, tested on-device; this module owns the
//! transports and the wifi dialer thread and applies those decisions.

mod policy;
mod wifi;

use crate::config::Settings;
use crate::logger;
use crate::transport::{
    Control, SerialTransport, TcpTransport, Transport, SERIAL_CLAIM_TIMEOUT,
    SERIAL_HOST_ABSENCE_GRACE, SERIAL_RECLAIM_COOLDOWN,
};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use log::{info, warn};
use policy::{ClaimOutcome, SerialClaimPolicy};
use protocol::Frame;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// Which link is carrying the stream is a fact about the wearer's device, not
// about these transports, so the type lives with the rest of the vocabulary
// that faces them and is re-exported here for the code that sets it.
pub(crate) use feedback_vocabulary::ActiveLink;

/// Owns both dashboard links and routes the stream over the active one.
pub struct Links {
    serial: SerialTransport,
    tcp: Option<TcpTransport>,
    tcp_deliveries: mpsc::Receiver<TcpTransport>,
    /// Stands the dialer thread down while a dashboard holds the serial link.
    want_tcp: Arc<AtomicBool>,
    wifi_worker: Option<WifiWorker>,
    /// The original lease stays here while a worker uses an effective reborrow.
    modem: Option<Modem<'static>>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    claim: SerialClaimPolicy,
    /// Changes whenever a newly established transport becomes eligible to carry
    /// frames. Current-level frames use it to replay after reconnect.
    generation: u32,
}

const WIFI_WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);
const TCP_READERS_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

struct WifiWorker {
    stop: Arc<AtomicBool>,
    done: mpsc::Receiver<()>,
    join: JoinHandle<anyhow::Result<()>>,
}

struct WorkerDone(Option<mpsc::Sender<()>>);

impl Drop for WorkerDone {
    fn drop(&mut self) {
        if let Some(done) = self.0.take() {
            let _ = done.send(());
        }
    }
}

/// A conclusively stopped wifi worker whose cleanup reported an error.
/// The original modem remains recoverable because no effective borrow is live.
#[must_use]
pub struct WifiShutdownError {
    modem: Modem<'static>,
    reason: String,
}

impl WifiShutdownError {
    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn into_modem(self) -> Modem<'static> {
        self.modem
    }
}

impl core::fmt::Debug for WifiShutdownError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("WifiShutdownError")
            .field("reason", &self.reason)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Display for WifiShutdownError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl std::error::Error for WifiShutdownError {}

impl Links {
    /// Construct the primary serial + BLE mode while retaining the modem lease.
    pub fn serial_only(
        serial: SerialTransport,
        modem: Modem<'static>,
        sysloop: EspSystemEventLoop,
        nvs: EspDefaultNvsPartition,
    ) -> Self {
        let (_, tcp_deliveries) = mpsc::channel();
        Self {
            serial,
            tcp: None,
            tcp_deliveries,
            want_tcp: Arc::new(AtomicBool::new(false)),
            wifi_worker: None,
            modem: Some(modem),
            sysloop,
            nvs,
            claim: SerialClaimPolicy::new(
                SERIAL_CLAIM_TIMEOUT,
                SERIAL_RECLAIM_COOLDOWN,
                SERIAL_HOST_ABSENCE_GRACE,
            ),
            generation: 0,
        }
    }

    /// Start wifi and dashboard dialing as an explicit runtime transition.
    /// Dashboard addresses are numeric to exclude uncancellable synchronous DNS.
    pub fn start_wifi(&mut self, settings: &Settings) -> anyhow::Result<()> {
        if self.wifi_worker.is_some() {
            anyhow::bail!("wifi is already running");
        }
        if settings.wifi_ssid.is_empty() {
            anyhow::bail!("wifi SSID is empty");
        }
        let server_addr: SocketAddr = settings
            .server_addr
            .parse()
            .map_err(|error| anyhow::anyhow!("dashboard address must be IP:port: {error}"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let want_tcp = Arc::new(AtomicBool::new(true));
        let (deliveries, tcp_deliveries) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();

        let modem = self.modem.as_mut().expect("Links always retains modem");
        // SAFETY: `Links` retains the original token while `wifi_worker` exists.
        // The effective borrow never escapes that worker, which drops its wifi
        // driver (and completes esp_wifi_deinit) before signaling `done`.
        let modem_borrow: Modem<'static> =
            unsafe { std::mem::transmute::<Modem<'_>, Modem<'static>>(modem.reborrow()) };
        let join = spawn_link_thread(
            modem_borrow,
            self.sysloop.clone(),
            self.nvs.clone(),
            settings,
            server_addr,
            Arc::clone(&want_tcp),
            Arc::clone(&stop),
            deliveries,
            done_tx,
        )?;
        self.want_tcp = want_tcp;
        self.tcp_deliveries = tcp_deliveries;
        self.wifi_worker = Some(WifiWorker { stop, done, join });
        Ok(())
    }

    /// One iteration of link upkeep: adopt or discard a freshly dialed TCP link,
    /// drain controls from both links, and expire a stale serial claim. Link
    /// management (probes and heartbeats) is handled here, announcing with the
    /// current `settings`; config controls are returned in arrival order for the
    /// caller to apply.
    pub fn poll(&mut self, device_id: &str, settings: &Settings) -> Vec<Control> {
        // A freshly dialed TCP link. If a dashboard claimed serial in the meantime,
        // discard it (dropping closes the socket; the thread stays stood down).
        if let Ok(transport) = self.tcp_deliveries.try_recv() {
            if self.claim.is_claimed() {
                drop(transport);
            } else {
                self.tcp = Some(transport);
                self.generation = self.generation.wrapping_add(1);
                if let Some(transport) = self.tcp.as_mut() {
                    let _ = announce(transport, device_id, settings);
                }
            }
        }

        let mut config_controls = Vec::new();
        while let Some(control) = self.serial.poll() {
            match control {
                Control::Probe {} => {
                    let outcome = self.claim.on_probe(Instant::now());
                    // A probe denotes a fresh dashboard session even when the old
                    // serial lease has not expired yet.
                    self.generation = self.generation.wrapping_add(1);
                    self.apply_claim_outcome(outcome, device_id, settings);
                }
                Control::Heartbeat {} => {
                    if let Some(outcome) = self.claim.on_heartbeat(Instant::now()) {
                        if outcome.became_claimed {
                            self.generation = self.generation.wrapping_add(1);
                        }
                        self.apply_claim_outcome(outcome, device_id, settings);
                    }
                }
                other => config_controls.push(other),
            }
        }
        if let Some(transport) = self.tcp.as_mut() {
            while let Some(control) = transport.poll() {
                match control {
                    Control::Probe {} | Control::Heartbeat {} => {} // serial-only frames
                    other => config_controls.push(other),
                }
            }
        }

        // Expire a serial claim when heartbeats stop or the cable is gone. The cause
        // is logged because the two point at opposite ends of the link.
        if let Some(reason) = self
            .claim
            .expire(Instant::now(), self.serial.host_present())
        {
            info!("serial link released ({reason:?})");
            self.want_tcp.store(true, Ordering::SeqCst);
        }

        config_controls
    }

    /// Route one window's frames over the active link: a claimed serial link wins,
    /// TCP otherwise. With neither, skip sending — frames for this window are lost
    /// (they're a live stream) but logs stay queued in their bounded buffer for
    /// whichever link appears first. Logs go first (reliable, tiny), then `hello`
    /// (present when config changed), then the window's data. Returns `true` only
    /// when an active transport accepted the entire send.
    pub fn send_window(&mut self, hello: Option<&Frame>, frames: &[Frame]) -> bool {
        let mut delivered = false;
        let serial_active = self.claim.is_claimed();
        if serial_active || self.tcp.is_some() {
            let Self { serial, tcp, .. } = self;
            let active: &mut dyn Transport = if serial_active {
                serial
            } else {
                tcp.as_mut().expect("tcp checked above")
            };

            let mut ok = true;
            // Logs are the record of what went wrong, so a dead link must not eat
            // them: restore the failed record; everything behind it remains queued.
            let pending_logs = logger::len();
            for _ in 0..pending_logs {
                let Some(log_record) = logger::pop() else {
                    break;
                };
                let log_frame = log_record.into_frame();
                if active.send(&log_frame).is_err() {
                    logger::restore(logger::LogRecord::from_frame(log_frame));
                    ok = false;
                    break;
                }
            }
            // Telemetry rides after the logs, fire-and-forget: a report a dead
            // link failed to send is not restored — the next interval re-reports,
            // and every counter in it is cumulative (see `telemetry`).
            if ok {
                for telemetry_frame in crate::telemetry::drain() {
                    if active.send(&telemetry_frame).is_err() {
                        ok = false;
                        break;
                    }
                }
            }
            // Stop at the first failure: every further send would block its full
            // timeout against the same dead link (and the data is a live stream —
            // this window is stale by the next iteration anyway).
            if ok {
                if let Some(hello) = hello {
                    ok = active.send(hello).is_ok();
                }
            }
            for frame in frames {
                if ok {
                    ok = active.send(frame).is_ok();
                }
            }
            delivered = ok;

            if !ok {
                if serial_active {
                    info!("serial write stalled; releasing claim");
                    self.claim.on_stall(Instant::now());
                    self.want_tcp.store(true, Ordering::SeqCst);
                } else {
                    warn!("dashboard link lost; redialing");
                }
            }
        }
        if self
            .tcp
            .as_ref()
            .is_some_and(|transport| !transport.alive_handle().load(Ordering::SeqCst))
        {
            self.tcp = None;
        }
        delivered
    }

    /// The link the next window would go out on, by [`Self::send_window`]'s own rule.
    ///
    /// A TCP transport dialed while serial held the claim never reaches `self.tcp`
    /// ([`Self::poll`] drops it on arrival), and a dead socket is cleared at the end
    /// of `send_window`, which the serve loop calls every iteration even with nothing
    /// to send. So this cannot name a link the stream is not using, and a link that
    /// dies while idle still turns up here.
    pub fn active_link(&self) -> ActiveLink {
        if self.claim.is_claimed() {
            ActiveLink::Serial
        } else if self.tcp.is_some() {
            ActiveLink::Wifi
        } else {
            ActiveLink::None
        }
    }

    /// Identity of the currently established delivery opportunity. It advances
    /// on reconnect even when the active link has the same variant as before.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Request worker shutdown and hang up any current or delivered TCP link.
    /// [`Self::stop_wifi`] performs the bounded wait and joins it.
    pub fn request_wifi_stop(&mut self) {
        self.want_tcp.store(false, Ordering::SeqCst);
        if let Some(worker) = self.wifi_worker.as_ref() {
            worker.stop.store(true, Ordering::SeqCst);
        }
        self.tcp = None;
        while let Ok(transport) = self.tcp_deliveries.try_recv() {
            drop(transport);
        }
    }

    /// Stop TCP and wifi without relinquishing the modem lease. Even when cleanup
    /// reports an error after join, `self` still owns the now-unborrowed modem.
    pub fn stop_wifi(&mut self) -> anyhow::Result<()> {
        self.request_wifi_stop();
        let Some(worker) = self.wifi_worker.as_ref() else {
            return Ok(());
        };
        match worker.done.recv_timeout(WIFI_WORKER_SHUTDOWN_TIMEOUT) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {
                anyhow::bail!("wifi worker did not stop within {WIFI_WORKER_SHUTDOWN_TIMEOUT:?}");
            }
        }

        let worker = self.wifi_worker.take().expect("worker checked above");
        worker
            .join
            .join()
            .map_err(|_| anyhow::anyhow!("wifi worker panicked during shutdown"))??;
        Ok(())
    }

    /// Stop TCP and wifi, wait for `esp_wifi_deinit`, and recover the original modem.
    /// A returned error also owns the modem; a live worker timeout aborts instead.
    pub fn shutdown(mut self) -> Result<Modem<'static>, WifiShutdownError> {
        if let Err(error) = self.stop_wifi() {
            if self.wifi_worker.is_some() {
                log::error!(
                    "wifi worker remained live during consuming shutdown ({error}); aborting"
                );
                std::process::abort();
            }
            return Err(WifiShutdownError {
                modem: self
                    .modem
                    .take()
                    .expect("dead worker releases effective modem borrow"),
                reason: format!("{error:#}"),
            });
        }
        Ok(self.modem.take().expect("wifi stopped before modem return"))
    }

    /// Act on an accepted probe or heartbeat: on a fresh claim, hang up TCP and
    /// stand the dialer down; announce when the policy asks for the hello.
    fn apply_claim_outcome(&mut self, outcome: ClaimOutcome, device_id: &str, settings: &Settings) {
        if outcome.became_claimed {
            info!("serial link claimed by dashboard");
            self.want_tcp.store(false, Ordering::SeqCst);
            self.tcp = None; // dropping hangs up; the backend sees a clean close
        }
        if outcome.announce {
            let _ = announce(&mut self.serial, device_id, settings);
        }
    }
}

impl Drop for Links {
    fn drop(&mut self) {
        if let Err(error) = self.stop_wifi() {
            if self.wifi_worker.is_some() {
                log::error!("wifi lifecycle could not stop safely ({error}); aborting");
                std::process::abort();
            }
            warn!("wifi shutdown completed with an error ({error})");
        }
    }
}

/// The background thread that keeps wifi associated and delivers dialed TCP
/// transports to the serve loop. Stands down (and stays associated but idle) while
/// `want_tcp` is false — i.e. while a dashboard holds the serial link.
fn spawn_link_thread(
    modem: Modem<'static>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    settings: &Settings,
    server_addr: SocketAddr,
    want_tcp: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    deliveries: mpsc::Sender<TcpTransport>,
    done: mpsc::Sender<()>,
) -> anyhow::Result<JoinHandle<anyhow::Result<()>>> {
    let ssid = settings.wifi_ssid.clone();
    let psk = settings.wifi_psk.clone();
    // Wifi bring-up and association run deep into esp-idf; they previously lived on
    // the 24 KB main task, so give this thread real headroom (8 KB overflowed).
    crate::cores::spawn_pinned(crate::cores::WIFI_LINK_MANAGEMENT_CORE, || {
        std::thread::Builder::new()
            .stack_size(20480)
            .spawn(move || {
                let _done = WorkerDone(Some(done));
                let mut wifi = wifi::start(modem, sysloop, nvs, &ssid, &psk)?;
                let mut current: Option<Arc<AtomicBool>> = None;
                let mut readers = Vec::new();
                while !stop.load(Ordering::SeqCst) {
                    TcpTransport::reap_finished_readers(&mut readers);
                    let delivered_alive = current
                        .as_ref()
                        .is_some_and(|alive| alive.load(Ordering::SeqCst));
                    // Reap before redial and never stack a new reader behind one
                    // still closing. This bounds supervision state during operation
                    // even if socket closure consumes its full read timeout.
                    if !want_tcp.load(Ordering::SeqCst) || delivered_alive || !readers.is_empty() {
                        delay_cancelable(&stop, Duration::from_millis(500));
                        continue;
                    }
                    if !wifi::ensure_connected(&mut wifi, &stop) {
                        delay_cancelable(&stop, Duration::from_secs(3));
                        continue;
                    }
                    match TcpTransport::connect_cancelable(server_addr, &stop) {
                        Ok(mut transport) => {
                            readers.push(transport.take_reader());
                            if stop.load(Ordering::SeqCst) {
                                drop(transport);
                                break;
                            }
                            info!("connected to dashboard at {server_addr}");
                            current = Some(transport.alive_handle());
                            if deliveries.send(transport).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            if stop.load(Ordering::SeqCst) {
                                break;
                            }
                            warn!("dial {server_addr} failed ({e}); retrying");
                            delay_cancelable(&stop, Duration::from_secs(2));
                        }
                    }
                }

                TcpTransport::reap_finished_readers(&mut readers);
                let readers_deadline = Instant::now() + TCP_READERS_SHUTDOWN_TIMEOUT;
                for reader in readers {
                    let Some(remaining) = readers_deadline.checked_duration_since(Instant::now())
                    else {
                        log::error!("TCP readers exceeded shutdown deadline; aborting");
                        std::process::abort();
                    };
                    if let Err(error) = reader.join(remaining) {
                        if Instant::now() >= readers_deadline {
                            log::error!("{error}; aborting before wifi deinit");
                            std::process::abort();
                        }
                        warn!("{error}");
                    }
                }
                let stop_result = wifi::stop(&mut wifi);
                drop(wifi); // WifiDriver::drop calls esp_wifi_deinit.
                stop_result
            })
    })?
    .map_err(Into::into)
}

fn delay_cancelable(stop: &AtomicBool, duration: Duration) {
    let deadline = Instant::now() + duration;
    while !stop.load(Ordering::SeqCst) {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            break;
        };
        FreeRtos::delay_ms(remaining.min(Duration::from_millis(50)).as_millis() as u32);
    }
}

/// Send the current identity + functional config.
fn announce(
    transport: &mut dyn Transport,
    device_id: &str,
    settings: &Settings,
) -> anyhow::Result<()> {
    transport.send(&Frame::DeviceHello {
        device_id: device_id.into(),
        config: settings.to_wire(),
        provenance: crate::provenance::device(),
    })
}
