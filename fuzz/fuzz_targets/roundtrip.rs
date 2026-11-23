//! Any input, any options: compress then decompress must give it back.
#![no_main]

use libfuzzer_sys::fuzz_target;
use neurozip::{CtxParams, Limits, MixParams, ModelSpec, Options, compress, decompress};

fuzz_target!(|input: &[u8]| {
    if input.len() < 4 {
        return;
    }
    let (cfg, data) = input.split_at(4);
    let opts = Options {
        model: if cfg[0] % 16 == 0 {
            ModelSpec::None
        } else if cfg[0] % 2 == 1 {
            ModelSpec::Mix(MixParams {
                hash_bits: 10 + cfg[1] % 3,
                match_bits: 12 + cfg[1] % 4,
                flags: cfg[0] >> 4,
            })
        } else {
            ModelSpec::Ctx(CtxParams {
                max_order: cfg[0] % 9,
                hash_bits: 8 + cfg[1] % 5,
            })
        },
        block_size: 1 + u16::from_le_bytes([cfg[2], cfg[3] & 0x0F]) as u32,
        independent_blocks: cfg[3] & 0x80 != 0,
    };
    let archive = compress(data, &opts).expect("compress");
    let restored = decompress(&archive, &Limits::default()).expect("decompress");
    assert!(restored == data, "round trip mismatch");
});
