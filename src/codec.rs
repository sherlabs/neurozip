//! Compression and decompression of whole `.nz` archives.
//!
//! Data is processed one block at a time, so working memory is the model
//! (fixed by its parameters) plus a small multiple of the block size.
//!
//! # Model synchronisation
//!
//! Unless `independent_blocks` is set the model carries over from block to
//! block. It is updated with **every** byte of every block, whichever codec
//! the block ends up using, in both encoder and decoder. That is what keeps
//! the two sides identical when a block falls back to stored form.

use std::io::{self, Read, Write};

use sha2::{Digest, Sha256};

use crate::crc32::crc32;
use crate::error::{Error, Result};
use crate::format::{
    BLOCK_HEADER_LEN, BlockHeader, CODEC_RANGE, CODEC_RANGE_X86, CODEC_STORED,
    FLAG_INDEPENDENT_BLOCKS, HEADER_LEN, Header, MAX_BLOCK_SIZE, ModelSpec, TRAILER_LEN,
};
use crate::freq::{Freqs, SYMBOLS};
use crate::model::Predictor;
use crate::model::ctx::CtxModel;
use crate::model::mix::{FLAG_NO_X86, MixModel, MixParams, PROB_BITS};
use crate::rangecoder::TOTAL_BITS;
use crate::rangecoder::{RangeDecoder, RangeEncoder};
use crate::x86;

/// Encoder settings. All of them are recorded in the archive header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub model: ModelSpec,
    /// Raw bytes per block, `1..=MAX_BLOCK_SIZE`.
    pub block_size: u32,
    /// Reset the model at each block boundary.
    pub independent_blocks: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            model: ModelSpec::Mix(MixParams::default()),
            block_size: 1 << 18,
            independent_blocks: false,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.block_size == 0 || self.block_size > MAX_BLOCK_SIZE {
            return Err(Error::InvalidOptions("block size out of range"));
        }
        if !self.model.is_valid() {
            return Err(Error::InvalidOptions("model parameters out of range"));
        }
        Ok(())
    }
}

/// Resource limits applied by the decoder before it allocates anything
/// sized by the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Largest model the decoder will allocate, in bytes.
    pub max_model_memory: u64,
    /// Largest declared original length the decoder will accept.
    pub max_output_len: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_model_memory: 1 << 30,
            max_output_len: u64::MAX,
        }
    }
}

/// What happened during one compress or decompress call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Bytes of original data.
    pub original_len: u64,
    /// Bytes of archive, including header, block headers and trailer.
    pub archive_len: u64,
    /// Blocks coded with the model and range coder.
    pub blocks_coded: u64,
    /// Blocks stored verbatim.
    pub blocks_stored: u64,
}

/// The instantiated predictor for an archive.
enum Model {
    None,
    Ctx(Box<CtxModel>),
    Mix(Box<MixModel>),
}

/// Scale a 12-bit model probability to the coder's 16-bit range.
const PROB_SHIFT: u32 = TOTAL_BITS - PROB_BITS;

/// Whether this archive's model uses the x86 filter at all.
fn x86_enabled(spec: ModelSpec) -> bool {
    matches!(spec, ModelSpec::Mix(p) if p.flags & FLAG_NO_X86 == 0)
}

/// The filtered form of `block` if the filter applies to it, else `None`.
/// The decision depends only on the raw bytes, so both sides agree on it
/// even for blocks that end up stored.
fn x86_view(block: &[u8], offset: u64, enabled: bool) -> Option<Vec<u8>> {
    (enabled && x86::wanted(block)).then(|| {
        let mut filtered = block.to_vec();
        x86::encode(&mut filtered, offset);
        filtered
    })
}

impl Model {
    fn new(spec: ModelSpec) -> Self {
        match spec {
            ModelSpec::None => Model::None,
            ModelSpec::Ctx(p) => Model::Ctx(Box::new(CtxModel::new(p))),
            ModelSpec::Mix(p) => Model::Mix(Box::new(MixModel::new(p))),
        }
    }

    fn reset(&mut self) {
        match self {
            Model::None => {}
            Model::Ctx(m) => m.reset(),
            Model::Mix(m) => m.reset(),
        }
    }

    /// Try to range-code `block`; `None` means "store it instead". The
    /// model has seen the whole block either way.
    fn encode_block(&mut self, block: &[u8]) -> Option<Vec<u8>> {
        match self {
            Model::None => None,
            Model::Ctx(m) => encode_block(m.as_mut(), block),
            Model::Mix(m) => encode_block_bits(m, block, true),
        }
    }

    /// Decode a range-coded block of `raw_len` bytes.
    fn decode_block(&mut self, payload: &[u8], raw_len: usize) -> Result<Vec<u8>> {
        match self {
            Model::None => Err(Error::Corrupt(
                "range-coded block in an archive with no model",
            )),
            Model::Ctx(m) => decode_block(m.as_mut(), payload, raw_len),
            Model::Mix(m) => decode_block_bits(m, payload, raw_len),
        }
    }

    /// Feed a stored block to the model without predicting.
    fn observe(&mut self, block: &[u8]) {
        match self {
            Model::None => {}
            Model::Ctx(m) => {
                for &b in block {
                    m.update(b);
                }
            }
            Model::Mix(m) => {
                for &b in block {
                    m.update_byte(b);
                }
            }
        }
    }
}

/// Range-code `block` bit by bit with the mixing model. With `fallback`,
/// gives up (returning `None`) as soon as the output cannot be smaller
/// than the block; the model still observes the whole block.
fn encode_block_bits(model: &mut MixModel, block: &[u8], fallback: bool) -> Option<Vec<u8>> {
    let mut enc = RangeEncoder::with_capacity(block.len() / 2 + 16);
    for (i, &byte) in block.iter().enumerate() {
        if fallback && enc.bytes_emitted() >= block.len() {
            for &b in &block[i..] {
                model.update_byte(b);
            }
            return None;
        }
        for shift in (0..8).rev() {
            let bit = (byte >> shift) & 1;
            enc.encode_bit(bit == 1, model.p1() << PROB_SHIFT);
            model.update(bit as u32);
        }
    }
    let payload = enc.finish();
    (!fallback || payload.len() < block.len()).then_some(payload)
}

fn decode_block_bits(model: &mut MixModel, payload: &[u8], raw_len: usize) -> Result<Vec<u8>> {
    let mut dec =
        RangeDecoder::new(payload).ok_or(Error::Corrupt("range coder payload is malformed"))?;
    let mut out = Vec::with_capacity(raw_len.min(payload.len().saturating_mul(64)));
    for _ in 0..raw_len {
        if dec.overrun() {
            return Err(Error::Corrupt("range coder payload is too short"));
        }
        let mut byte = 0u8;
        for _ in 0..8 {
            let bit = dec.decode_bit(model.p1() << PROB_SHIFT);
            model.update(bit as u32);
            byte = (byte << 1) | bit as u8;
        }
        out.push(byte);
    }
    if !dec.finish() {
        return Err(Error::Corrupt("range coder payload has the wrong length"));
    }
    Ok(out)
}

fn encode_block<P: Predictor>(model: &mut P, block: &[u8]) -> Option<Vec<u8>> {
    let mut enc = RangeEncoder::with_capacity(block.len() / 2 + 16);
    let mut freqs: Freqs = [0; SYMBOLS];
    for (i, &byte) in block.iter().enumerate() {
        // A coded block is only used if it is strictly smaller than the raw
        // block. Once that is impossible, stop coding and just let the
        // model observe the remainder.
        if enc.bytes_emitted() >= block.len() {
            for &b in &block[i..] {
                model.update(b);
            }
            return None;
        }
        model.predict(&mut freqs);
        let cum: u32 = freqs[..byte as usize].iter().map(|&f| f as u32).sum();
        enc.encode(cum, freqs[byte as usize] as u32);
        model.update(byte);
    }
    let payload = enc.finish();
    (payload.len() < block.len()).then_some(payload)
}

fn decode_block<P: Predictor>(model: &mut P, payload: &[u8], raw_len: usize) -> Result<Vec<u8>> {
    let mut dec =
        RangeDecoder::new(payload).ok_or(Error::Corrupt("range coder payload is malformed"))?;
    let mut out = Vec::with_capacity(raw_len.min(payload.len().saturating_mul(64)));
    let mut freqs: Freqs = [0; SYMBOLS];
    for _ in 0..raw_len {
        // A valid payload is consumed exactly, so running past its end
        // proves corruption. Stopping here keeps the work done on a forged
        // block proportional to the bytes actually supplied.
        if dec.overrun() {
            return Err(Error::Corrupt("range coder payload is too short"));
        }
        model.predict(&mut freqs);
        let target = dec.target();
        let mut cum = 0u32;
        let mut sym = 0usize;
        // The table sums to TOTAL > target, so the loop always finds a
        // symbol before running off the end.
        for (s, &f) in freqs.iter().enumerate() {
            let next = cum + f as u32;
            if target < next {
                sym = s;
                break;
            }
            cum = next;
        }
        dec.consume(cum, freqs[sym] as u32);
        model.update(sym as u8);
        out.push(sym as u8);
    }
    if !dec.finish() {
        return Err(Error::Corrupt("range coder payload has the wrong length"));
    }
    Ok(out)
}

/// `read_exact` that reports a short read as [`Error::Truncated`].
fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::Truncated
        } else {
            Error::Io(e)
        }
    })
}

/// True if the reader has no more bytes.
fn at_eof<R: Read>(r: &mut R) -> Result<bool> {
    let mut probe = [0u8; 1];
    loop {
        match r.read(&mut probe) {
            Ok(0) => return Ok(true),
            Ok(_) => return Ok(false),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e)),
        }
    }
}

/// Compress exactly `len` bytes from `reader` into `writer`.
///
/// The length is written into the header up front, so it must be known. If
/// the reader yields fewer or more bytes the call fails with
/// [`Error::LengthMismatch`] (the partial output must then be discarded).
pub fn compress_stream<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    len: u64,
    opts: &Options,
) -> Result<Stats> {
    opts.validate()?;
    let header = Header {
        model: opts.model,
        flags: if opts.independent_blocks {
            FLAG_INDEPENDENT_BLOCKS
        } else {
            0
        },
        block_size: opts.block_size,
        original_len: len,
    };
    writer.write_all(&header.to_bytes())?;

    let mut stats = Stats {
        original_len: len,
        archive_len: (HEADER_LEN + TRAILER_LEN) as u64,
        ..Stats::default()
    };
    let mut model = Model::new(opts.model);
    let x86_on = x86_enabled(opts.model);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; (opts.block_size as u64).min(len) as usize];
    let mut remaining = len;

    while remaining > 0 {
        let n = (opts.block_size as u64).min(remaining) as usize;
        read_exact(&mut reader, &mut buf[..n]).map_err(|e| match e {
            Error::Truncated => Error::LengthMismatch,
            other => other,
        })?;
        let block = &buf[..n];
        hasher.update(block);
        if opts.independent_blocks {
            model.reset();
        }
        // The model is shown the filtered form of blocks that look like
        // x86 code; the stored fallback always keeps the raw bytes.
        let filtered = x86_view(block, len - remaining, x86_on);
        let coded = model.encode_block(filtered.as_deref().unwrap_or(block));
        let (codec, payload): (u8, &[u8]) = match &coded {
            Some(p) if filtered.is_some() => (CODEC_RANGE_X86, p),
            Some(p) => (CODEC_RANGE, p),
            None => (CODEC_STORED, block),
        };
        let bh = BlockHeader {
            codec,
            raw_len: n as u32,
            payload_len: payload.len() as u32,
            crc: crc32(block),
        };
        writer.write_all(&bh.to_bytes())?;
        writer.write_all(payload)?;
        stats.archive_len += (BLOCK_HEADER_LEN + payload.len()) as u64;
        if codec == CODEC_STORED {
            stats.blocks_stored += 1;
        } else {
            stats.blocks_coded += 1;
        }
        remaining -= n as u64;
    }

    if !at_eof(&mut reader)? {
        return Err(Error::LengthMismatch);
    }
    let digest: [u8; TRAILER_LEN] = hasher.finalize().into();
    writer.write_all(&digest)?;
    writer.flush()?;
    Ok(stats)
}

/// Decompress one archive from `reader` into `writer`.
///
/// Each block is written only after its CRC has been verified. If a later
/// check fails the call returns an error and whatever was already written
/// must be discarded by the caller. The reader must end exactly at the end
/// of the archive.
pub fn decompress_stream<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    limits: &Limits,
) -> Result<Stats> {
    let mut hb = [0u8; HEADER_LEN];
    read_exact(&mut reader, &mut hb)?;
    let header = Header::parse(&hb)?;

    let required = header.model.memory_bytes();
    if required > limits.max_model_memory {
        return Err(Error::MemoryLimit {
            required,
            limit: limits.max_model_memory,
        });
    }
    if header.original_len > limits.max_output_len {
        return Err(Error::OutputLimit {
            declared: header.original_len,
            limit: limits.max_output_len,
        });
    }

    let mut stats = Stats {
        original_len: header.original_len,
        archive_len: (HEADER_LEN + TRAILER_LEN) as u64,
        ..Stats::default()
    };
    let mut model = Model::new(header.model);
    let x86_on = x86_enabled(header.model);
    let mut hasher = Sha256::new();
    let mut payload = Vec::new();
    let mut remaining = header.original_len;
    let mut block_index = 0u64;

    while remaining > 0 {
        let mut bb = [0u8; BLOCK_HEADER_LEN];
        read_exact(&mut reader, &mut bb)?;
        let bh = BlockHeader::parse(&bb);

        // Blocks are canonical: every block but the last is exactly
        // `block_size` long, so the raw length is fully determined.
        let expected = (header.block_size as u64).min(remaining);
        if bh.raw_len as u64 != expected {
            return Err(Error::Corrupt("block length does not match the header"));
        }
        let raw_len = bh.raw_len as usize;
        match bh.codec {
            CODEC_STORED => {
                if bh.payload_len != bh.raw_len {
                    return Err(Error::Corrupt("stored block with mismatched lengths"));
                }
            }
            CODEC_RANGE | CODEC_RANGE_X86 => {
                // The encoder only emits a coded block when it is strictly
                // smaller than the raw block; this also bounds the buffer.
                if (bh.payload_len as usize) < RangeDecoder::MIN_LEN || bh.payload_len >= bh.raw_len
                {
                    return Err(Error::Corrupt("range-coded block with impossible length"));
                }
                if bh.codec == CODEC_RANGE_X86 && !x86_on {
                    return Err(Error::Corrupt(
                        "x86-filtered block in an archive without the filter",
                    ));
                }
            }
            _ => return Err(Error::Corrupt("unknown block codec")),
        }

        // Grow the buffer only as data actually arrives, so a forged length
        // cannot force a large allocation on a short input.
        payload.clear();
        let got = (&mut reader)
            .take(bh.payload_len as u64)
            .read_to_end(&mut payload)?;
        if got != bh.payload_len as usize {
            return Err(Error::Truncated);
        }

        if header.independent_blocks() {
            model.reset();
        }
        let offset = header.original_len - remaining;
        let mut decoded;
        let raw: &[u8] = if bh.codec == CODEC_STORED {
            // Show the model exactly what the encoder showed it.
            match x86_view(&payload, offset, x86_on) {
                Some(filtered) => model.observe(&filtered),
                None => model.observe(&payload),
            }
            stats.blocks_stored += 1;
            &payload
        } else {
            decoded = model.decode_block(&payload, raw_len)?;
            if bh.codec == CODEC_RANGE_X86 {
                x86::decode(&mut decoded, offset);
            }
            stats.blocks_coded += 1;
            &decoded
        };
        if crc32(raw) != bh.crc {
            return Err(Error::BlockChecksum { block: block_index });
        }
        hasher.update(raw);
        writer.write_all(raw)?;

        stats.archive_len += (BLOCK_HEADER_LEN + payload.len()) as u64;
        remaining -= raw_len as u64;
        block_index += 1;
    }

    let mut trailer = [0u8; TRAILER_LEN];
    read_exact(&mut reader, &mut trailer)?;
    let digest: [u8; TRAILER_LEN] = hasher.finalize().into();
    if digest != trailer {
        return Err(Error::ContentHash);
    }
    if !at_eof(&mut reader)? {
        return Err(Error::TrailingData);
    }
    writer.flush()?;
    Ok(stats)
}

/// A variant of the encoder measured for ablation studies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ablation {
    /// Store a block raw when range coding does not shrink it (the real
    /// encoder always does).
    pub fallback: bool,
    /// Stop learning after this many input bytes: the model still follows
    /// the context but its statistics are frozen. `None` learns throughout.
    pub freeze_after: Option<u64>,
}

impl Default for Ablation {
    /// The real encoder's behaviour.
    fn default() -> Self {
        Ablation {
            fallback: true,
            freeze_after: None,
        }
    }
}

/// Exact size in bytes, all metadata included, of the archive an encoder
/// variant would write for `data`.
///
/// This runs the real model and range coder and counts their output. With
/// `Ablation::default()` the result equals `compress(data, opts)?.len()`.
/// Other variants are measurements of sizes only: the `.nz` format has no
/// way to say "no fallback" or "frozen model", so no decodable archive of
/// that kind exists.
pub fn ablation_size(data: &[u8], opts: &Options, ablation: &Ablation) -> Result<u64> {
    opts.validate()?;
    let mut total = (HEADER_LEN + TRAILER_LEN) as u64;
    let freeze_at = ablation.freeze_after.unwrap_or(u64::MAX);
    let mut model = match opts.model {
        ModelSpec::None => None,
        ModelSpec::Ctx(p) => Some(CtxModel::new(p)),
        ModelSpec::Mix(p) => {
            // The mixing model's ablations are header switches (real,
            // decodable archives). Only the fallback variant applies here.
            if ablation.freeze_after.is_some() {
                return Err(Error::InvalidOptions(
                    "freeze-after applies to the ctx model; use --mix-off mixer for mix",
                ));
            }
            let mut m = MixModel::new(p);
            let x86_on = x86_enabled(opts.model);
            let mut offset = 0u64;
            for block in data.chunks(opts.block_size as usize) {
                if opts.independent_blocks {
                    m.reset();
                }
                let filtered = x86_view(block, offset, x86_on);
                let view = filtered.as_deref().unwrap_or(block);
                let coded = encode_block_bits(&mut m, view, ablation.fallback);
                total += (BLOCK_HEADER_LEN + coded.map_or(block.len(), |c| c.len())) as u64;
                offset += block.len() as u64;
            }
            return Ok(total);
        }
    };
    let mut freqs: Freqs = [0; SYMBOLS];
    let mut pos = 0u64;
    for block in data.chunks(opts.block_size as usize) {
        total += BLOCK_HEADER_LEN as u64;
        let Some(m) = model.as_mut() else {
            total += block.len() as u64;
            continue;
        };
        if opts.independent_blocks {
            m.reset();
        }
        let mut enc = RangeEncoder::with_capacity(block.len() / 2 + 16);
        for &byte in block {
            m.predict(&mut freqs);
            let cum: u32 = freqs[..byte as usize].iter().map(|&f| f as u32).sum();
            enc.encode(cum, freqs[byte as usize] as u32);
            if pos < freeze_at {
                m.update(byte);
            } else {
                m.advance(byte);
            }
            pos += 1;
        }
        let coded = enc.finish().len();
        total += if ablation.fallback {
            coded.min(block.len())
        } else {
            coded
        } as u64;
    }
    Ok(total)
}

/// Compress a byte slice into a new archive.
pub fn compress(data: &[u8], opts: &Options) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() / 2 + HEADER_LEN + TRAILER_LEN + BLOCK_HEADER_LEN);
    compress_stream(data, &mut out, data.len() as u64, opts)?;
    Ok(out)
}

/// Decompress a complete archive held in memory.
pub fn decompress(archive: &[u8], limits: &Limits) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_stream(archive, &mut out, limits)?;
    Ok(out)
}

/// SHA-256 of `data`, the same digest the archive trailer carries.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}
