//! Error type shared by the whole crate.

use std::fmt;
use std::io;

/// Everything that can go wrong while compressing or decompressing.
///
/// Decoding never panics on hostile input: every malformed archive maps to
/// one of these variants.
#[derive(Debug)]
pub enum Error {
    /// Underlying I/O failure (not including a short read, see `Truncated`).
    Io(io::Error),
    /// The first four bytes are not `NZIP`.
    BadMagic,
    /// The archive was written by a format version this build cannot read.
    UnsupportedVersion(u8),
    /// Unknown model identifier in the header.
    UnsupportedModel(u8),
    /// Header flag bits that this build does not understand.
    UnsupportedFlags(u16),
    /// A header field is outside its legal range.
    InvalidHeader(&'static str),
    /// The header CRC-32 does not match.
    HeaderChecksum,
    /// Options passed to the encoder are outside the legal range.
    InvalidOptions(&'static str),
    /// The archive asks for more model memory than the caller allows.
    MemoryLimit { required: u64, limit: u64 },
    /// The archive declares more output than the caller allows.
    OutputLimit { declared: u64, limit: u64 },
    /// Structurally invalid block or payload.
    Corrupt(&'static str),
    /// A decoded block does not match its CRC-32.
    BlockChecksum { block: u64 },
    /// The whole-content SHA-256 does not match.
    ContentHash,
    /// The archive ends before it should.
    Truncated,
    /// Bytes follow the end of the archive.
    TrailingData,
    /// The input stream length differs from the length given to the encoder.
    LengthMismatch,
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::BadMagic => write!(f, "not a NeuroZip archive (bad magic)"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Error::UnsupportedModel(m) => write!(f, "unsupported model identifier {m}"),
            Error::UnsupportedFlags(x) => write!(f, "unsupported codec flags {x:#06x}"),
            Error::InvalidHeader(why) => write!(f, "invalid header: {why}"),
            Error::HeaderChecksum => write!(f, "header checksum mismatch"),
            Error::InvalidOptions(why) => write!(f, "invalid options: {why}"),
            Error::MemoryLimit { required, limit } => write!(
                f,
                "archive needs {required} bytes of model memory, limit is {limit}"
            ),
            Error::OutputLimit { declared, limit } => write!(
                f,
                "archive declares {declared} output bytes, limit is {limit}"
            ),
            Error::Corrupt(why) => write!(f, "corrupt archive: {why}"),
            Error::BlockChecksum { block } => write!(f, "checksum mismatch in block {block}"),
            Error::ContentHash => write!(f, "content SHA-256 mismatch"),
            Error::Truncated => write!(f, "archive is truncated"),
            Error::TrailingData => write!(f, "unexpected data after end of archive"),
            Error::LengthMismatch => write!(f, "input length differs from the declared length"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
