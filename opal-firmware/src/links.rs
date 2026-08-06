//! Link routing: exactly one link carries the data stream at a time, and the most
//! recently established link wins. A dashboard probing the serial port claims it
//! (heartbeats keep the claim alive; silence, unplug, or a stalled write releases
//! it), and TCP carries the stream otherwise. The claim/stall/cooldown decisions
//! live in the `link_policy` module, tested on-device; this module owns the
//! transports and the wifi dialer thread and applies those decisions.

use crate::config::Settings;
use crate::link_policy::{ClaimOutcome, SerialClaimPolicy};
use crate::logger;
use crate::transport::{
    Control, SerialTransport, TcpTransport, Transport, SERIAL_CLAIM_TIMEOUT,
    SERIAL_HOST_ABSENCE_GRACE, SERIAL_RECLAIM_COOLDOWN,
};
use crate::wifi;
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use log::{info, warn};
use protocol::Frame;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// Which link is carrying the stream right now.
///
/// [`Links::send_window`]'s routing rule as a value, so anything reacting to the link
/// reads the same fact the data follows rather than a second copy of the rule. A
/// level, not an edge: callers wanting transitions diff it themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveLink {
    None,
    Serial,
    Wifi,
}

impl ActiveLink {
    /// Whether the stream is going anywhere. Which link is a development fact — a
    /// wearer only ever has wifi — so anything facing them asks this instead.
    pub fn is_connected(self) -> bool {
        self != ActiveLink::None
    }
}

/// Owns both dashboard links and routes the stream over the active one.
pub struct Links {
    serial: SerialTransport,
    tcp: Option<TcpTransport>,
    tcp_deliveries: mpsc::Receiver<TcpTransport>,
    /// Stands the dialer thread down while a dashboard holds the serial link.
    want_tcp: Arc<AtomicBool>,
    claim: SerialClaimPolicy,
    /// The one wire-encoding buffer, threaded through every [`Transport::send`].
    /// Reserved once at boot while the heap is still unfragmented: a bulk EMG frame
    /// encodes to ~24 KB, and the steady-state heap's largest free block hovers
    /// around 31 KB, so allocating that per send is an out-of-memory abort waiting
    /// on fragmentation luck.
    scratch: Vec<u8>,
}

/// [`Links::scratch`]'s boot-time reservation: a typical worst encoded EMG frame —
/// one railed chip packing at ~3 varint bytes per sample, the other near one — plus
/// CBOR structure and framing. A frame beyond it (both chips railed) grows the
/// buffer once and the capacity sticks; reserving the absolute worst case up front
/// costs 8 KB of permanent headroom against a heap where TCP's link threads need
/// two 8 KB contiguous stacks at dial time.
const ENCODE_SCRATCH_BYTES: usize = 18 * 1024;

impl Links {
    /// Take ownership of the serial link and, when wifi credentials are configured,
    /// spawn the background thread that keeps wifi associated and dials the
    /// dashboard.
    pub fn new(
        serial: SerialTransport,
        modem: Modem<'static>,
        sysloop: EspSystemEventLoop,
        nvs: EspDefaultNvsPartition,
        settings: &Settings,
    ) -> anyhow::Result<Self> {
        let want_tcp = Arc::new(AtomicBool::new(!settings.wifi_ssid.is_empty()));
        let (deliveries_sender, tcp_deliveries) = mpsc::channel();
        if !settings.wifi_ssid.is_empty() {
            spawn_link_thread(
                modem,
                sysloop,
                nvs,
                settings,
                Arc::clone(&want_tcp),
                deliveries_sender,
            )?;
        } else {
            info!("no wifi configured; serial only");
        }
        Ok(Self {
            serial,
            tcp: None,
            tcp_deliveries,
            want_tcp,
            claim: SerialClaimPolicy::new(
                SERIAL_CLAIM_TIMEOUT,
                SERIAL_RECLAIM_COOLDOWN,
                SERIAL_HOST_ABSENCE_GRACE,
            ),
            scratch: Vec::with_capacity(ENCODE_SCRATCH_BYTES),
        })
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
                if let Some(transport) = self.tcp.as_mut() {
                    let _ = announce(transport, &mut self.scratch, device_id, settings);
                }
            }
        }

        let mut config_controls = Vec::new();
        while let Some(control) = self.serial.poll() {
            match control {
                Control::Probe {} => {
                    let outcome = self.claim.on_probe(Instant::now());
                    self.apply_claim_outcome(outcome, device_id, settings);
                }
                Control::Heartbeat {} => {
                    if let Some(outcome) = self.claim.on_heartbeat(Instant::now()) {
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
            info!("serial link released ({reason:?}); resuming wifi");
            self.want_tcp.store(true, Ordering::SeqCst);
        }

        config_controls
    }

    /// Route one window's frames over the active link: a claimed serial link wins,
    /// TCP otherwise. With neither, skip sending — frames for this window are lost
    /// (they're a live stream) but logs stay queued in their bounded buffer for
    /// whichever link appears first. Logs go first (reliable, tiny), then `hello`
    /// (present when config changed), then the window's data.
    pub fn send_window(&mut self, hello: Option<&Frame>, frames: &[Frame]) {
        let serial_active = self.claim.is_claimed();
        if serial_active || self.tcp.is_some() {
            // Destructured so the active transport and the shared encode scratch can
            // be borrowed at once.
            let Self {
                serial,
                tcp,
                scratch,
                ..
            } = self;
            let active: &mut dyn Transport = if serial_active {
                serial
            } else {
                tcp.as_mut().expect("tcp checked above")
            };

            let mut ok = true;
            // Logs are the record of what went wrong, so a dead link must not eat
            // them: put the failed record and everything behind it back for the
            // next link.
            let mut pending_logs = logger::drain().into_iter();
            while let Some(log_frame) = pending_logs.next() {
                if active.send(&log_frame, scratch).is_err() {
                    let mut unsent = vec![log_frame];
                    unsent.extend(pending_logs);
                    logger::restore(unsent);
                    ok = false;
                    break;
                }
            }
            // Telemetry rides after the logs, fire-and-forget: a report a dead
            // link failed to send is not restored — the next interval re-reports,
            // and every counter in it is cumulative (see `telemetry`).
            if ok {
                for telemetry_frame in crate::telemetry::drain() {
                    if active.send(&telemetry_frame, scratch).is_err() {
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
                    ok = active.send(hello, scratch).is_ok();
                }
            }
            for frame in frames {
                if ok {
                    ok = active.send(frame, scratch).is_ok();
                }
            }

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

    /// Act on an accepted probe or heartbeat: on a fresh claim, hang up TCP and
    /// stand the dialer down; announce when the policy asks for the hello.
    fn apply_claim_outcome(&mut self, outcome: ClaimOutcome, device_id: &str, settings: &Settings) {
        if outcome.became_claimed {
            info!("serial link claimed by dashboard");
            self.want_tcp.store(false, Ordering::SeqCst);
            self.tcp = None; // dropping hangs up; the backend sees a clean close
        }
        if outcome.announce {
            let _ = announce(&mut self.serial, &mut self.scratch, device_id, settings);
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
    want_tcp: Arc<AtomicBool>,
    deliveries: mpsc::Sender<TcpTransport>,
) -> anyhow::Result<()> {
    let ssid = settings.wifi_ssid.clone();
    let psk = settings.wifi_psk.clone();
    let server_addr = settings.server_addr.clone();
    // Wifi bring-up and association run deep into esp-idf; they previously lived on
    // the 24 KB main task, so give this thread real headroom (8 KB overflowed).
    crate::cores::spawn_pinned(crate::cores::WIFI_LINK_MANAGEMENT_CORE, || {
        std::thread::Builder::new()
            .stack_size(20480)
            .spawn(move || {
                let mut wifi = match wifi::start(modem, sysloop, nvs, &ssid, &psk) {
                    Ok(wifi) => wifi,
                    Err(e) => {
                        warn!("wifi failed to start ({e}); serial only");
                        return;
                    }
                };
                let mut current: Option<Arc<AtomicBool>> = None;
                loop {
                    let delivered_alive = current
                        .as_ref()
                        .is_some_and(|alive| alive.load(Ordering::SeqCst));
                    if !want_tcp.load(Ordering::SeqCst) || delivered_alive {
                        FreeRtos::delay_ms(500);
                        continue;
                    }
                    if !wifi::ensure_connected(&mut wifi) {
                        FreeRtos::delay_ms(3000);
                        continue;
                    }
                    match TcpTransport::connect(&server_addr) {
                        Ok(transport) => {
                            info!("connected to dashboard at {server_addr}");
                            current = Some(transport.alive_handle());
                            if deliveries.send(transport).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            warn!("dial {server_addr} failed ({e}); retrying");
                            FreeRtos::delay_ms(2000);
                        }
                    }
                }
            })
    })??;
    Ok(())
}

/// Send the current identity + functional config.
fn announce(
    transport: &mut dyn Transport,
    scratch: &mut Vec<u8>,
    device_id: &str,
    settings: &Settings,
) -> anyhow::Result<()> {
    transport.send(
        &Frame::DeviceHello {
            device_id: device_id.into(),
            config: settings.to_wire(),
            provenance: crate::provenance::device(),
        },
        scratch,
    )
}
