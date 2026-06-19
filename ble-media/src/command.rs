use crate::media;

/// A parsed console command.
#[derive(Clone, Copy, Debug)]
pub enum Command {
    /// Send a media key.
    Media(media::MediaKey),
    /// Print the help text.
    Help,
}

impl TryFrom<char> for Command {
    type Error = char;

    fn try_from(value: char) -> core::prelude::v1::Result<Self, Self::Error> {
        use media::MediaKey as MK;

        match value.to_ascii_lowercase() {
            'p' => Ok(Command::Media(MK::PlayPause)),
            'n' => Ok(Command::Media(MK::NextTrack)),
            'b' => Ok(Command::Media(MK::PrevTrack)),
            '+' | '=' => Ok(Command::Media(MK::VolumeUp)),
            '-' | '_' => Ok(Command::Media(MK::VolumeDown)),
            'm' => Ok(Command::Media(MK::Mute)),
            'h' | '?' => Ok(Command::Help),
            other => Err(other),
        }
    }
}
