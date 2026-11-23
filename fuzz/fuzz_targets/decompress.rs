//! Arbitrary bytes into the decoder: must return, never panic or hang.
//!
//! The first 24 input bytes are used as header fields and given a valid
//! CRC, so the fuzzer gets past the header check and into the block and
//! range decoders instead of being stopped by the checksum every time.
#![no_main]

use libfuzzer_sys::fuzz_target;
use neurozip::crc32::crc32;
use neurozip::{Limits, decompress};

fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        // Small enough that the fuzzer cannot ask for huge models.
        max_model_memory: 64 << 20,
        max_output_len: 1 << 20,
    };
    let _ = decompress(data, &limits);

    if data.len() >= 24 {
        let mut sealed = data[..24].to_vec();
        sealed[..4].copy_from_slice(b"NZIP");
        sealed[4] = 1;
        let crc = crc32(&sealed);
        sealed.extend_from_slice(&crc.to_le_bytes());
        sealed.extend_from_slice(&data[24..]);
        let _ = decompress(&sealed, &limits);
    }
});
