//! Which of a device's eight analog inputs a value refers to.

use core::fmt;

/// Analog input channels on one ADS1298.
pub(crate) const CHANNELS_PER_DEVICE: usize = 8;

/// ADS1298s on the front end.
///
/// Two self-clocked boards, each on its own SPI bus with its own chip select,
/// START, RESET, PWDN, and DRDY. [`model_slot`] covers all sixteen slots, the
/// conditioner sizes itself to match, and `preprocess` documents the
/// synchronisation the pair has to satisfy for the layout to be honest.
pub(crate) const DEVICE_COUNT: usize = 2;

/// One of the two analog-front-end boards, as wired in `main`: A is the
/// right-column ribbon on SPI2, B the left-column ribbon on SPI3. This is
/// hardware identity — it keys the wiring, the aligner source slot, and the
/// core plan (`crate::cores`) — and stays put even when any of those
/// assignments change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Board {
    A,
    B,
}

impl Board {
    /// Both boards, in device-index order.
    pub(crate) const ALL: [Board; DEVICE_COUNT] = [Board::A, Board::B];

    /// This board's aligner source slot and [`model_slot`] block.
    pub(crate) const fn device_index(self) -> usize {
        match self {
            Board::A => 0,
            Board::B => 1,
        }
    }
}

/// Which of the model's input slots device `device_index` fills with `channel`.
///
/// Devices take contiguous blocks in index order, so device 0 owns slots 0..8 and
/// device 1 owns slots 8..16. That ordering is not arbitrary — it is the order the
/// two-chip cascade presented its channels in when the training set was recorded, and
/// the model learned a spatial pattern across them. Reordering it silently costs
/// accuracy in a way no test here would catch.
pub(crate) const fn model_slot(device_index: usize, channel: Channel) -> usize {
    device_index * CHANNELS_PER_DEVICE + channel.index()
}

/// A zero-indexed channel on one device, guaranteed to sit in
/// `0..CHANNELS_PER_DEVICE`.
///
/// The guarantee is the whole point. The range used to be checked by a `debug_assert!`
/// at each use site and by a hand-rolled `const _: () = assert!(...)` in `main`, which
/// meant every new use site had to remember to check again. Constructing the value is
/// now the only place the range is looked at, so the CHnSET address arithmetic, the
/// status-word bit shifts, and the channel array indexing can all treat it as settled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Channel(u8);

impl Channel {
    /// Every channel, low to high. Bitmask types walk this to name their set bits, so
    /// the order here is the order flags appear in a log line.
    pub(crate) const ALL: [Channel; CHANNELS_PER_DEVICE] = [
        Channel(0),
        Channel(1),
        Channel(2),
        Channel(3),
        Channel(4),
        Channel(5),
        Channel(6),
        Channel(7),
    ];

    /// `None` for an index outside `0..CHANNELS_PER_DEVICE`.
    #[cfg(test)]
    pub(crate) const fn new(index: u8) -> Option<Channel> {
        if (index as usize) < CHANNELS_PER_DEVICE {
            Some(Channel(index))
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) const fn checked(index: u8) -> Channel {
        match Self::new(index) {
            Some(channel) => channel,
            None => panic!("channel index must be within 0..CHANNELS_PER_DEVICE"),
        }
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    /// This channel's bit in the registers and status fields that carry one bit per
    /// channel: channel *n* is bit *n*.
    pub(crate) const fn bit(self) -> u8 {
        1 << self.0
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_channels_are_rejected() {
        assert!(Channel::new(0).is_some());
        assert!(Channel::new(7).is_some());
        assert!(Channel::new(8).is_none());
        assert!(Channel::new(255).is_none());
    }

    #[test]
    fn channel_bits_are_one_per_channel() {
        assert_eq!(Channel::checked(0).bit(), 0x01);
        assert_eq!(Channel::checked(7).bit(), 0x80);
    }

    #[test]
    fn all_channels_are_in_index_order() {
        for (index, channel) in Channel::ALL.iter().enumerate() {
            assert_eq!(channel.index(), index);
        }
    }
}
