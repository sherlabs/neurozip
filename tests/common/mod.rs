//! Helpers shared by the integration tests.
#![allow(dead_code)]

use neurozip::{CtxParams, MixParams, ModelSpec, Options};

/// Deterministic pseudo-random bytes (splitmix64).
pub fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Deterministic compressible, text-like bytes.
pub fn text_bytes(len: usize, seed: u64) -> Vec<u8> {
    const WORDS: [&str; 16] = [
        "range", "coder", "context", "model", "the", "of", "and", "predict", "update", "byte",
        "block", "archive", "lossless", "entropy", "symbol", "table",
    ];
    let noise = random_bytes(len / 2 + 16, seed);
    let mut out = Vec::with_capacity(len + 16);
    let mut i = 0;
    while out.len() < len {
        let r = noise[i % noise.len()] as usize;
        i += 1;
        out.extend_from_slice(WORDS[r % WORDS.len()].as_bytes());
        out.push(match r >> 4 {
            0 => b'\n',
            1 => b',',
            _ => b' ',
        });
    }
    out.truncate(len);
    out
}

/// Small-memory context model options so tests stay fast.
pub fn ctx_opts(max_order: u8, hash_bits: u8, block_size: u32) -> Options {
    Options {
        model: ModelSpec::Ctx(CtxParams {
            max_order,
            hash_bits,
        }),
        block_size,
        independent_blocks: false,
    }
}

/// Small-memory mixing model options.
pub fn mix_opts(hash_bits: u8, match_bits: u8, flags: u8, block_size: u32) -> Options {
    Options {
        model: ModelSpec::Mix(MixParams {
            hash_bits,
            match_bits,
            flags,
        }),
        block_size,
        independent_blocks: false,
    }
}
