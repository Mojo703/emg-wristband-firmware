pub const EMG_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c03";

pub const CHUNK_HEADER_LEN: usize = 6;
pub const RECORD_HEADER_LEN: usize = 12;

pub const MAX_NOTIFICATION_LEN: usize = 244;

#[derive(Debug, Clone, Copy)]
pub struct WindowRecord<'a> {
    pub first_step_micros: u64,
    pub steps_per_channel: u16,
    pub missing: &'a [u8],
    pub packed_samples: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    NotificationTooShort(usize),
    TooManyChunks(usize),
}

impl core::fmt::Display for ChunkError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotificationTooShort(length) => write!(
                formatter,
                "a {length}-byte notification has no room after the {CHUNK_HEADER_LEN}-byte header"
            ),
            Self::TooManyChunks(count) => write!(formatter, "{count} chunks do not fit a u16"),
        }
    }
}

impl std::error::Error for ChunkError {}

impl WindowRecord<'_> {
    pub fn encoded_len(&self) -> usize {
        RECORD_HEADER_LEN + self.missing.len() + self.packed_samples.len()
    }

    pub fn chunk_count(&self, notification_len: usize) -> Result<u16, ChunkError> {
        let payload = chunk_payload(notification_len)?;
        let count = self.encoded_len().div_ceil(payload);
        u16::try_from(count).map_err(|_| ChunkError::TooManyChunks(count))
    }

    pub fn write_chunk(
        &self,
        sequence: u16,
        index: u16,
        count: u16,
        notification_len: usize,
        out: &mut [u8],
    ) -> usize {
        let payload = notification_len - CHUNK_HEADER_LEN;
        let start = usize::from(index) * payload;
        let end = (start + payload).min(self.encoded_len());
        out[0..2].copy_from_slice(&sequence.to_le_bytes());
        out[2..4].copy_from_slice(&index.to_le_bytes());
        out[4..6].copy_from_slice(&count.to_le_bytes());

        let header = self.header();
        let segments: [&[u8]; 3] = [&header, self.missing, self.packed_samples];
        let mut written = CHUNK_HEADER_LEN;
        let mut segment_start = 0;
        for segment in segments {
            let segment_end = segment_start + segment.len();
            let from = start.max(segment_start);
            let to = end.min(segment_end);
            if from < to {
                let slice = &segment[from - segment_start..to - segment_start];
                out[written..written + slice.len()].copy_from_slice(slice);
                written += slice.len();
            }
            segment_start = segment_end;
        }
        written
    }

    fn header(&self) -> [u8; RECORD_HEADER_LEN] {
        let mut header = [0u8; RECORD_HEADER_LEN];
        header[0..8].copy_from_slice(&self.first_step_micros.to_le_bytes());
        header[8..10].copy_from_slice(&self.steps_per_channel.to_le_bytes());
        header[10..12].copy_from_slice(&(self.missing.len() as u16).to_le_bytes());
        header
    }
}

fn chunk_payload(notification_len: usize) -> Result<usize, ChunkError> {
    notification_len
        .checked_sub(CHUNK_HEADER_LEN)
        .filter(|&payload| payload > 0)
        .ok_or(ChunkError::NotificationTooShort(notification_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reassemble(record: &WindowRecord, notification_len: usize) -> Vec<u8> {
        let count = record.chunk_count(notification_len).unwrap();
        let mut joined = Vec::new();
        for index in 0..count {
            let mut out = [0u8; MAX_NOTIFICATION_LEN];
            let used = record.write_chunk(7, index, count, notification_len, &mut out);
            assert!(used <= notification_len);
            assert_eq!(&out[0..2], &7u16.to_le_bytes());
            assert_eq!(&out[2..4], &index.to_le_bytes());
            assert_eq!(&out[4..6], &count.to_le_bytes());
            joined.extend_from_slice(&out[CHUNK_HEADER_LEN..used]);
        }
        joined
    }

    #[test]
    fn a_record_encodes_to_the_documented_layout() {
        let record = WindowRecord {
            first_step_micros: 0x0102_0304_0506_0708,
            steps_per_channel: 500,
            missing: &[0xAA, 0xBB],
            packed_samples: &[1, 2, 3],
        };
        assert_eq!(
            reassemble(&record, MAX_NOTIFICATION_LEN),
            [
                0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0xF4, 0x01, 2, 0, 0xAA, 0xBB, 1, 2,
                3
            ]
        );
    }

    #[test]
    fn chunks_split_across_segment_boundaries() {
        let missing: Vec<u8> = (0..126).map(|byte| byte as u8).collect();
        let packed: Vec<u8> = (0..9_000).map(|byte| (byte * 7) as u8).collect();
        let record = WindowRecord {
            first_step_micros: 42,
            steps_per_channel: 500,
            missing: &missing,
            packed_samples: &packed,
        };
        for notification_len in [7, 20, 100, 185, MAX_NOTIFICATION_LEN] {
            let joined = reassemble(&record, notification_len);
            assert_eq!(joined.len(), record.encoded_len());
            assert_eq!(
                &joined[RECORD_HEADER_LEN..RECORD_HEADER_LEN + 126],
                &missing[..]
            );
            assert_eq!(&joined[RECORD_HEADER_LEN + 126..], &packed[..]);
        }
    }

    #[test]
    fn a_notification_with_no_room_is_refused() {
        let record = WindowRecord {
            first_step_micros: 0,
            steps_per_channel: 1,
            missing: &[],
            packed_samples: &[],
        };
        assert_eq!(
            record.chunk_count(CHUNK_HEADER_LEN),
            Err(ChunkError::NotificationTooShort(CHUNK_HEADER_LEN))
        );
        assert_eq!(record.chunk_count(CHUNK_HEADER_LEN + 1), Ok(12));
    }
}
