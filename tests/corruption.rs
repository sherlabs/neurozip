//! Hostile and damaged input: the decoder must return an error (never
//! panic, never hang, never over-allocate) and must never hand back wrong
//! data as if it were right.

mod common;

use common::{ctx_opts, mix_opts, random_bytes, text_bytes};
use neurozip::crc32::crc32;
use neurozip::format::{BLOCK_HEADER_LEN, HEADER_LEN, TRAILER_LEN};
use neurozip::{Error, Limits, ModelSpec, Options, compress, decompress};
use proptest::prelude::*;

/// Both model types, small enough to mutate exhaustively.
fn model_opts() -> [Options; 2] {
    [ctx_opts(3, 10, 600), mix_opts(10, 12, 0, 600)]
}

fn limits() -> Limits {
    Limits::default()
}

/// The safety property: a damaged archive either fails or (when the damage
/// happens not to matter, e.g. in bits the range decoder never needs)
/// yields exactly the original. It never yields anything else.
fn assert_safe(mutated: &[u8], original: &[u8]) -> bool {
    match decompress(mutated, &limits()) {
        Err(_) => true,
        Ok(out) => {
            assert!(out == original, "damaged archive decoded to different data");
            false
        }
    }
}

/// A small archive with both a range-coded and a stored block.
fn sample() -> (Vec<u8>, Vec<u8>) {
    sample_with(&ctx_opts(3, 10, 600))
}

fn sample_with(opts: &Options) -> (Vec<u8>, Vec<u8>) {
    let mut data = text_bytes(600, 1);
    data.extend(random_bytes(300, 2));
    let archive = compress(&data, opts).unwrap();
    (data, archive)
}

fn reseal_header(archive: &mut [u8]) {
    let crc = crc32(&archive[..24]);
    archive[24..28].copy_from_slice(&crc.to_le_bytes());
}

#[test]
fn every_single_bit_flip() {
    for opts in model_opts() {
        bit_flips(&opts);
    }
}

fn bit_flips(opts: &Options) {
    let (data, archive) = sample_with(opts);
    let mut rejected = 0usize;
    for bit in 0..archive.len() * 8 {
        let mut bad = archive.clone();
        bad[bit / 8] ^= 1 << (bit % 8);
        if assert_safe(&bad, &data) {
            rejected += 1;
        }
    }
    // Nearly every flip must be caught outright. (A flip is harmless only
    // in the few low-order flush bits of a range-coded payload.)
    let total = archive.len() * 8;
    assert!(
        rejected * 1000 >= total * 995,
        "{rejected} of {total} rejected"
    );
}

#[test]
fn every_single_byte_replacement_in_headers() {
    let (data, archive) = sample();
    for pos in 0..HEADER_LEN + BLOCK_HEADER_LEN {
        for value in 0..=255u8 {
            if archive[pos] == value {
                continue;
            }
            let mut bad = archive.clone();
            bad[pos] = value;
            assert!(assert_safe(&bad, &data), "byte {pos} = {value} accepted");
        }
    }
}

#[test]
fn every_truncation_is_rejected() {
    for opts in model_opts() {
        truncations(&opts);
    }
}

fn truncations(opts: &Options) {
    let (_, archive) = sample_with(opts);
    for len in 0..archive.len() {
        match decompress(&archive[..len], &limits()) {
            Err(_) => {}
            Ok(_) => panic!("truncation to {len} bytes accepted"),
        }
    }
    assert!(matches!(decompress(&[], &limits()), Err(Error::Truncated)));
}

#[test]
fn trailing_data_is_rejected() {
    let (_, mut archive) = sample();
    archive.push(0);
    assert!(matches!(
        decompress(&archive, &limits()),
        Err(Error::TrailingData)
    ));
    // Two archives back to back are not one archive.
    let (_, a) = sample();
    let doubled = [a.clone(), a].concat();
    assert!(matches!(
        decompress(&doubled, &limits()),
        Err(Error::TrailingData)
    ));
}

#[test]
fn specific_errors() {
    let (_, archive) = sample();

    let mut bad = archive.clone();
    bad[0] = b'X';
    assert!(matches!(decompress(&bad, &limits()), Err(Error::BadMagic)));

    let mut bad = archive.clone();
    bad[4] = 99;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::UnsupportedVersion(99))
    ));

    let mut bad = archive.clone();
    bad[20] ^= 1;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::HeaderChecksum)
    ));

    // Damage the trailer only.
    let mut bad = archive.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::ContentHash)
    ));

    // Damage the stored block's payload (the last block before the trailer).
    let mut bad = archive.clone();
    let pos = bad.len() - TRAILER_LEN - 10;
    bad[pos] ^= 0x55;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::BlockChecksum { block: 1 })
    ));

    // Unknown block codec.
    let mut bad = archive.clone();
    bad[HEADER_LEN] = 7;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::Corrupt(_))
    ));
}

#[test]
fn forged_lengths_do_not_allocate_or_spin() {
    let (_, archive) = sample();

    // Claim an enormous original length with a valid header checksum.
    let mut bad = archive.clone();
    bad[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    reseal_header(&mut bad);
    assert!(decompress(&bad, &limits()).is_err());

    // Same, with the largest block size, so the first block claims 64 MiB.
    let mut bad = archive.clone();
    bad[12..16].copy_from_slice(&(1u32 << 26).to_le_bytes());
    bad[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes());
    reseal_header(&mut bad);
    let bh = HEADER_LEN;
    bad[bh + 1..bh + 5].copy_from_slice(&(1u32 << 26).to_le_bytes());
    let start = std::time::Instant::now();
    assert!(decompress(&bad, &limits()).is_err());
    assert!(
        start.elapsed().as_secs() < 5,
        "decoder spent too long on a forged block"
    );

    // An output limit rejects it before any work.
    let small = Limits {
        max_output_len: 100,
        ..Limits::default()
    };
    assert!(matches!(
        decompress(&archive, &small),
        Err(Error::OutputLimit { .. })
    ));
}

#[test]
fn memory_limit_is_enforced_before_allocation() {
    // Header asking for the largest model: 7 hashed orders x 512 MiB.
    let mut archive = compress(b"hello", &ctx_opts(2, 8, 64)).unwrap();
    archive[8] = 8;
    archive[9] = 24;
    reseal_header(&mut archive);
    match decompress(&archive, &limits()) {
        Err(Error::MemoryLimit { required, limit }) => {
            assert!(required > 3 << 30);
            assert_eq!(limit, 1 << 30);
        }
        other => panic!("expected MemoryLimit, got {other:?}"),
    }

    // A tight limit rejects even a modest model; a sufficient one accepts.
    let data = text_bytes(500, 3);
    let archive = compress(&data, &ctx_opts(4, 12, 256)).unwrap();
    let need = ctx_opts(4, 12, 256).model.memory_bytes();
    let tight = Limits {
        max_model_memory: need - 1,
        ..Limits::default()
    };
    assert!(matches!(
        decompress(&archive, &tight),
        Err(Error::MemoryLimit { .. })
    ));
    let enough = Limits {
        max_model_memory: need,
        ..Limits::default()
    };
    assert_eq!(decompress(&archive, &enough).unwrap(), data);
}

#[test]
fn range_block_without_a_model_is_rejected() {
    let (_, mut archive) = sample();
    // Rewrite the header to claim "no model" while a range-coded block follows.
    archive[5] = 0;
    archive[8] = 0;
    archive[9] = 0;
    reseal_header(&mut archive);
    assert!(matches!(
        decompress(&archive, &limits()),
        Err(Error::Corrupt(_))
    ));
}

#[test]
fn non_canonical_blocks_are_rejected() {
    // A stored-only archive is easy to edit by hand.
    let data = random_bytes(100, 5);
    let opts = Options {
        model: ModelSpec::None,
        block_size: 60,
        independent_blocks: false,
    };
    let archive = compress(&data, &opts).unwrap();
    assert_eq!(decompress(&archive, &limits()).unwrap(), data);

    // Shrink the first block's declared raw length.
    let mut bad = archive.clone();
    bad[HEADER_LEN + 1..HEADER_LEN + 5].copy_from_slice(&59u32.to_le_bytes());
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::Corrupt(_))
    ));

    // Stored block whose payload length disagrees with its raw length.
    let mut bad = archive.clone();
    bad[HEADER_LEN + 5..HEADER_LEN + 9].copy_from_slice(&61u32.to_le_bytes());
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::Corrupt(_))
    ));

    // Range block claiming to be no smaller than the raw data.
    let mut bad = archive.clone();
    bad[HEADER_LEN] = 1;
    assert!(matches!(
        decompress(&bad, &limits()),
        Err(Error::Corrupt(_))
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    /// Arbitrary bytes are never a panic and essentially never an archive.
    #[test]
    fn prop_garbage_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = decompress(&bytes, &limits());
    }

    /// Garbage behind a valid header reaches the block and range decoders.
    #[test]
    fn prop_garbage_after_valid_header(
        body in proptest::collection::vec(any::<u8>(), 0..600),
        order in 0u8..=5,
        len in 0u64..5000,
    ) {
        let mut archive = compress(
            b"",
            &if order == 5 { mix_opts(10, 12, 0, 512) } else { ctx_opts(order, 9, 512) },
        ).unwrap();
        archive.truncate(HEADER_LEN);
        archive[16..24].copy_from_slice(&len.to_le_bytes());
        reseal_header(&mut archive);
        archive.extend_from_slice(&body);
        let _ = decompress(&archive, &limits());
    }

    /// Garbage payload behind a structurally valid range-coded block header.
    #[test]
    fn prop_garbage_range_payload(
        mut payload in proptest::collection::vec(any::<u8>(), 5..300),
        raw_len in 301u32..2000,
    ) {
        payload[0] = 0;
        let mut archive = compress(
            b"",
            &if raw_len % 2 == 0 { mix_opts(10, 12, 0, 4096) } else { ctx_opts(3, 9, 4096) },
        ).unwrap();
        archive.truncate(HEADER_LEN);
        archive[16..24].copy_from_slice(&(raw_len as u64).to_le_bytes());
        reseal_header(&mut archive);
        archive.push(1);
        archive.extend_from_slice(&raw_len.to_le_bytes());
        archive.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        archive.extend_from_slice(&[0; 4]);
        archive.extend_from_slice(&payload);
        archive.extend_from_slice(&[0; TRAILER_LEN]);
        prop_assert!(decompress(&archive, &limits()).is_err());
    }

    /// Random multi-byte damage to real archives is always safe.
    #[test]
    fn prop_mutations_are_safe(
        seed in any::<u64>(),
        len in 0usize..3000,
        edits in proptest::collection::vec((any::<usize>(), any::<u8>(), 0u8..3), 1..6),
    ) {
        let data = if seed % 3 == 0 { random_bytes(len, seed) } else { text_bytes(len, seed) };
        let archive = compress(
            &data,
            &if seed % 2 == 0 { mix_opts(10, 12, (seed % 16) as u8, 700) } else { ctx_opts((seed % 6) as u8, 9, 700) },
        ).unwrap();
        let mut bad = archive.clone();
        for (pos, value, kind) in edits {
            let pos = pos % bad.len();
            match kind {
                0 => bad[pos] = value,
                1 => bad.insert(pos, value),
                _ => { bad.remove(pos); }
            }
            if bad.is_empty() { break; }
        }
        if bad != archive {
            assert_safe(&bad, &data);
        }
    }
}
