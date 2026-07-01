//! The link to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi) or a UART (serial), but the
//! framing is identical — a 4-byte little-endian length, then that many CBOR bytes —
//! so the rest of the firmware is transport-agnostic.

use anyhow::Result;
use esp_idf_svc::hal::uart::UartDriver;
use protocol::{Binding, Frame};
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::sync::Arc;

/// The control frames the device accepts (browser → backend → device). A dedicated,
/// float-free mirror of the relevant `protocol::Frame` variants: decoding the full
/// `Frame` would force ciborium's f16→f32 float path to compile, which the Xtensa
/// LLVM backend cannot codegen. The device never receives float-bearing frames, so
/// this subset is sufficient and keeps that path out of the firmware entirely.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Control {
    SetSensitivity { level: String },
    SetKeymap { bindings: Vec<Binding> },
    SetWifi { ssid: String, psk: String },
}

/// One end of the dashboard link.
pub trait Transport {
    fn send(&mut self, frame: &Frame) -> Result<()>;
    /// Non-blocking: the next control frame from the dashboard, if any is ready.
    fn poll(&mut self) -> Option<Control>;
}

fn encode(frame: &Frame) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf).expect("CBOR encode");
    buf
}

fn decode(bytes: &[u8]) -> Option<Control> {
    ciborium::from_reader(bytes).ok()
}

// ----------------------------------------------------------------------------
// TCP (wifi)
// ----------------------------------------------------------------------------

/// Dials the dashboard over TCP. A reader thread decodes inbound frames into a
/// channel so [`Transport::poll`] stays non-blocking.
///
/// The socket is shared through an `Arc` rather than duplicated: `TcpStream::try_clone`
/// maps to `dup()`, which lwIP does not implement (ENOSYS) on ESP-IDF. `Read` and
/// `Write` are both available on `&TcpStream`, so one fd serves both threads.
pub struct TcpTransport {
    writer: Arc<TcpStream>,
    rx: mpsc::Receiver<Control>,
}

impl TcpTransport {
    pub fn connect(addr: &str) -> Result<Self> {
        let stream = Arc::new(TcpStream::connect(addr)?);
        stream.set_nodelay(true).ok();
        let reader = Arc::clone(&stream);
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .stack_size(6144)
            .spawn(move || read_loop(reader, tx))?;
        Ok(Self { writer: stream, rx })
    }
}

fn read_loop(reader: Arc<TcpStream>, tx: mpsc::Sender<Control>) {
    let mut reader = &*reader;
    loop {
        let mut len = [0u8; 4];
        if reader.read_exact(&mut len).is_err() {
            break;
        }
        let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
        if reader.read_exact(&mut buf).is_err() {
            break;
        }
        if let Some(frame) = decode(&buf) {
            if tx.send(frame).is_err() {
                break;
            }
        }
    }
}

impl Transport for TcpTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        let bytes = encode(frame);
        let mut writer: &TcpStream = &self.writer;
        writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        self.rx.try_recv().ok()
    }
}

// ----------------------------------------------------------------------------
// Serial (UART)
// ----------------------------------------------------------------------------

/// Reads/writes framed CBOR over a UART. Single-threaded: `poll` drains whatever
/// bytes are available into an accumulator and returns one frame once complete.
pub struct SerialTransport {
    uart: UartDriver<'static>,
    accumulator: Vec<u8>,
}

impl SerialTransport {
    pub fn new(uart: UartDriver<'static>) -> Self {
        Self { uart, accumulator: Vec::new() }
    }
}

impl Transport for SerialTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        let bytes = encode(frame);
        write_all_uart(&self.uart, &(bytes.len() as u32).to_le_bytes())?;
        write_all_uart(&self.uart, &bytes)?;
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        let mut chunk = [0u8; 256];
        loop {
            match self.uart.read(&mut chunk, 0) {
                Ok(0) | Err(_) => break,
                Ok(n) => self.accumulator.extend_from_slice(&chunk[..n]),
            }
        }
        if self.accumulator.len() < 4 {
            return None;
        }
        let n = u32::from_le_bytes([
            self.accumulator[0],
            self.accumulator[1],
            self.accumulator[2],
            self.accumulator[3],
        ]) as usize;
        if self.accumulator.len() < 4 + n {
            return None;
        }
        let frame = decode(&self.accumulator[4..4 + n]);
        self.accumulator.drain(0..4 + n);
        frame
    }
}

fn write_all_uart(uart: &UartDriver, mut bytes: &[u8]) -> Result<()> {
    while !bytes.is_empty() {
        let written = uart.write(bytes)?;
        bytes = &bytes[written..];
    }
    Ok(())
}
