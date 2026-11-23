//! The `.nz` container: header, block headers and trailer.
//!
//! The authoritative description is `docs/FORMAT.md`; this module is its
//! implementation. All multi-byte integers are little-endian.
//!
//! ```text
//! archive := header block* trailer
//!
//! header (28 bytes)
//!    0  4  magic            "NZIP"
//!    4  1  format version   1
//!    5  1  model id         0 = none, 1 = byte context model, 2 = bit mixing model
//!    6  2  codec flags      bit 0 = independent blocks; others must be 0
//!    8  1  model param A    model 1: max order;  model 2: hash bits
//!    9  1  model param B    model 1: hash bits;  model 2: match window bits
//!   10  1  model param C    model 2: feature switches; otherwise 0
//!   11  1  reserved         must be 0
//!   12  4  block size       raw bytes per block, 1..=MAX_BLOCK_SIZE
//!   16  8  original length  total raw bytes
//!   24  4  CRC-32 of bytes 0..24
//!
//! block (13-byte header + payload)
//!    0  1  codec            0 = stored, 1 = model + range coder
//!    1  4  raw length
//!    5  4  payload length
//!    9  4  CRC-32 of the raw block
//!   13  .  payload
//!
//! trailer (32 bytes)
//!    SHA-256 of the complete original data
//! ```

use crate::crc32::crc32;
use crate::error::{Error, Result};
use crate::model::ctx::CtxParams;
use crate::model::mix::MixParams;

pub const MAGIC: [u8; 4] = *b"NZIP";
pub const FORMAT_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 28;
pub const BLOCK_HEADER_LEN: usize = 13;
pub const TRAILER_LEN: usize = 32;

/// Largest permitted block size (64 MiB). Bounds decoder buffer memory.
pub const MAX_BLOCK_SIZE: u32 = 1 << 26;

/// Codec flag: the model is reset at the start of every block, so blocks
/// can be decoded independently of one another.
pub const FLAG_INDEPENDENT_BLOCKS: u16 = 1 << 0;
/// All flag bits this version understands.
pub const KNOWN_FLAGS: u16 = FLAG_INDEPENDENT_BLOCKS;

pub const MODEL_ID_NONE: u8 = 0;
pub const MODEL_ID_CTX: u8 = 1;
pub const MODEL_ID_MIX: u8 = 2;

/// Block codec identifiers.
pub const CODEC_STORED: u8 = 0;
pub const CODEC_RANGE: u8 = 1;
/// Range-coded after the x86 filter (mixing model only).
pub const CODEC_RANGE_X86: u8 = 2;

/// Which predictor an archive uses, with its parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpec {
    /// No model: every block is stored verbatim.
    None,
    /// The byte-level adaptive context model.
    Ctx(CtxParams),
    /// The bit-level mixing model with a match model.
    Mix(MixParams),
}

impl ModelSpec {
    /// Heap bytes the decoder must allocate for this model.
    pub fn memory_bytes(&self) -> u64 {
        match self {
            ModelSpec::None => 0,
            ModelSpec::Ctx(p) => p.memory_bytes(),
            ModelSpec::Mix(p) => p.memory_bytes(),
        }
    }

    pub fn is_valid(&self) -> bool {
        match self {
            ModelSpec::None => true,
            ModelSpec::Ctx(p) => p.is_valid(),
            ModelSpec::Mix(p) => p.is_valid(),
        }
    }

    fn to_fields(self) -> (u8, u8, u8, u8) {
        match self {
            ModelSpec::None => (MODEL_ID_NONE, 0, 0, 0),
            ModelSpec::Ctx(p) => (MODEL_ID_CTX, p.max_order, p.hash_bits, 0),
            ModelSpec::Mix(p) => (MODEL_ID_MIX, p.hash_bits, p.match_bits, p.flags),
        }
    }

    fn from_fields(id: u8, a: u8, b: u8, c: u8) -> Result<Self> {
        if id != MODEL_ID_MIX && c != 0 {
            return Err(Error::InvalidHeader("model parameter C must be zero"));
        }
        match id {
            MODEL_ID_MIX => {
                let p = MixParams {
                    hash_bits: a,
                    match_bits: b,
                    flags: c,
                };
                if !p.is_valid() {
                    return Err(Error::InvalidHeader("mixing model parameters out of range"));
                }
                Ok(ModelSpec::Mix(p))
            }
            MODEL_ID_NONE => {
                if a != 0 || b != 0 {
                    return Err(Error::InvalidHeader("model parameters must be zero"));
                }
                Ok(ModelSpec::None)
            }
            MODEL_ID_CTX => {
                let p = CtxParams {
                    max_order: a,
                    hash_bits: b,
                };
                if !p.is_valid() {
                    return Err(Error::InvalidHeader(
                        "context model parameters out of range",
                    ));
                }
                Ok(ModelSpec::Ctx(p))
            }
            other => Err(Error::UnsupportedModel(other)),
        }
    }
}

/// Parsed archive header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub model: ModelSpec,
    pub flags: u16,
    pub block_size: u32,
    pub original_len: u64,
}

impl Header {
    pub fn independent_blocks(&self) -> bool {
        self.flags & FLAG_INDEPENDENT_BLOCKS != 0
    }

    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let (id, a, b, c) = self.model.to_fields();
        let mut out = [0u8; HEADER_LEN];
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = FORMAT_VERSION;
        out[5] = id;
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8] = a;
        out[9] = b;
        out[10] = c;
        // 11 reserved, zero.
        out[12..16].copy_from_slice(&self.block_size.to_le_bytes());
        out[16..24].copy_from_slice(&self.original_len.to_le_bytes());
        let crc = crc32(&out[..24]);
        out[24..28].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse and fully validate a header.
    pub fn parse(bytes: &[u8; HEADER_LEN]) -> Result<Self> {
        if bytes[0..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        if bytes[4] != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(bytes[4]));
        }
        let stored_crc = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        if crc32(&bytes[..24]) != stored_crc {
            return Err(Error::HeaderChecksum);
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(Error::UnsupportedFlags(flags));
        }
        let model = ModelSpec::from_fields(bytes[5], bytes[8], bytes[9], bytes[10])?;
        if bytes[11] != 0 {
            return Err(Error::InvalidHeader("reserved byte must be zero"));
        }
        let block_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if block_size == 0 || block_size > MAX_BLOCK_SIZE {
            return Err(Error::InvalidHeader("block size out of range"));
        }
        let mut len = [0u8; 8];
        len.copy_from_slice(&bytes[16..24]);
        Ok(Header {
            model,
            flags,
            block_size,
            original_len: u64::from_le_bytes(len),
        })
    }
}

/// Parsed block header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    pub codec: u8,
    pub raw_len: u32,
    pub payload_len: u32,
    pub crc: u32,
}

impl BlockHeader {
    pub fn to_bytes(&self) -> [u8; BLOCK_HEADER_LEN] {
        let mut out = [0u8; BLOCK_HEADER_LEN];
        out[0] = self.codec;
        out[1..5].copy_from_slice(&self.raw_len.to_le_bytes());
        out[5..9].copy_from_slice(&self.payload_len.to_le_bytes());
        out[9..13].copy_from_slice(&self.crc.to_le_bytes());
        out
    }

    /// Decode the fields. Semantic validation happens in the decoder, which
    /// knows the expected block length.
    pub fn parse(b: &[u8; BLOCK_HEADER_LEN]) -> Self {
        BlockHeader {
            codec: b[0],
            raw_len: u32::from_le_bytes([b[1], b[2], b[3], b[4]]),
            payload_len: u32::from_le_bytes([b[5], b[6], b[7], b[8]]),
            crc: u32::from_le_bytes([b[9], b[10], b[11], b[12]]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Header {
        Header {
            model: ModelSpec::Ctx(CtxParams {
                max_order: 4,
                hash_bits: 20,
            }),
            flags: FLAG_INDEPENDENT_BLOCKS,
            block_size: 1 << 18,
            original_len: 0x0123_4567_89AB,
        }
    }

    /// Re-seal a hand-edited header so only the edited field is at fault.
    fn reseal(b: &mut [u8; HEADER_LEN]) {
        let crc = crc32(&b[..24]);
        b[24..28].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn header_roundtrip() {
        let h = sample();
        assert_eq!(Header::parse(&h.to_bytes()).unwrap(), h);
        let none = Header {
            model: ModelSpec::None,
            flags: 0,
            block_size: 1,
            original_len: 0,
        };
        assert_eq!(Header::parse(&none.to_bytes()).unwrap(), none);
    }

    #[test]
    fn every_single_bit_flip_is_rejected() {
        let good = sample().to_bytes();
        for bit in 0..HEADER_LEN * 8 {
            let mut bad = good;
            bad[bit / 8] ^= 1 << (bit % 8);
            assert!(Header::parse(&bad).is_err(), "bit {bit} accepted");
        }
    }

    #[test]
    fn field_validation() {
        let good = sample().to_bytes();
        let edit = |f: &dyn Fn(&mut [u8; HEADER_LEN])| {
            let mut b = good;
            f(&mut b);
            reseal(&mut b);
            Header::parse(&b)
        };
        assert!(matches!(edit(&|b| b[0] = b'X'), Err(Error::BadMagic)));
        assert!(matches!(
            edit(&|b| b[4] = 2),
            Err(Error::UnsupportedVersion(2))
        ));
        assert!(matches!(
            edit(&|b| b[5] = 9),
            Err(Error::UnsupportedModel(9))
        ));
        assert!(matches!(
            edit(&|b| b[6] = 0x82),
            Err(Error::UnsupportedFlags(_))
        ));
        assert!(matches!(edit(&|b| b[8] = 9), Err(Error::InvalidHeader(_))));
        assert!(matches!(edit(&|b| b[9] = 25), Err(Error::InvalidHeader(_))));
        assert!(matches!(edit(&|b| b[9] = 7), Err(Error::InvalidHeader(_))));
        assert!(matches!(edit(&|b| b[10] = 1), Err(Error::InvalidHeader(_))));
        assert!(matches!(
            edit(&|b| b[12..16].copy_from_slice(&0u32.to_le_bytes())),
            Err(Error::InvalidHeader(_))
        ));
        assert!(matches!(
            edit(&|b| b[12..16].copy_from_slice(&(MAX_BLOCK_SIZE + 1).to_le_bytes())),
            Err(Error::InvalidHeader(_))
        ));
        // Model "none" must not carry parameters.
        assert!(matches!(edit(&|b| b[5] = 0), Err(Error::InvalidHeader(_))));
        // A bad CRC alone.
        let mut b = good;
        b[27] ^= 0xFF;
        assert!(matches!(Header::parse(&b), Err(Error::HeaderChecksum)));
    }

    #[test]
    fn block_header_roundtrip() {
        let h = BlockHeader {
            codec: CODEC_RANGE,
            raw_len: 262_144,
            payload_len: 70_001,
            crc: 0xDEAD_BEEF,
        };
        assert_eq!(BlockHeader::parse(&h.to_bytes()), h);
    }
}
