//! CBOR encode/decode for protocol frames. One place so the WebSocket and any
//! debug dump agree on the wire format.

use anyhow::Result;
use protocol::Frame;

pub fn encode(frame: &Frame) -> Vec<u8> {
    let mut buffer = Vec::new();
    // Frames are small/owned; encoding only fails on an allocator failure.
    ciborium::into_writer(frame, &mut buffer).expect("CBOR encode");
    buffer
}

pub fn decode(bytes: &[u8]) -> Result<Frame> {
    Ok(ciborium::from_reader(bytes)?)
}
