use protocol::MediaKey;

/// A parsed console command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// Send a media key.
    Media(MediaKey),
    /// Turn the phone peripheral on or off — the bench's stand-in for the
    /// dashboard button, so the state machine can be driven without one.
    Phone(bool),
    /// Print the help text.
    Help,
}

impl TryFrom<char> for Command {
    type Error = char;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value.to_ascii_lowercase() {
            'p' => Ok(Command::Media(MediaKey::PlayPause)),
            'n' => Ok(Command::Media(MediaKey::NextTrack)),
            'b' => Ok(Command::Media(MediaKey::PrevTrack)),
            '+' | '=' => Ok(Command::Media(MediaKey::VolumeUp)),
            '-' | '_' => Ok(Command::Media(MediaKey::VolumeDown)),
            'm' => Ok(Command::Media(MediaKey::Mute)),
            'e' => Ok(Command::Phone(true)),
            'd' => Ok(Command::Phone(false)),
            'h' | '?' => Ok(Command::Help),
            other => Err(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enable and disable are one keystroke each and next to each other on the
    /// keyboard, so the pair is worth a test that says which is which.
    #[test]
    fn the_toggle_keys_parse_to_the_state_they_name() {
        assert_eq!(Command::try_from('e'), Ok(Command::Phone(true)));
        assert_eq!(Command::try_from('D'), Ok(Command::Phone(false)));
        assert_eq!(
            Command::try_from('p'),
            Ok(Command::Media(MediaKey::PlayPause))
        );
        assert_eq!(Command::try_from('z'), Err('z'));
    }
}
