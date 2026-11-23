//! Exact reconstruction across data shapes, sizes and every option.

mod common;

use common::{ctx_opts, mix_opts, random_bytes, text_bytes};
use neurozip::format::{BLOCK_HEADER_LEN, HEADER_LEN, TRAILER_LEN};
use neurozip::{
    CtxParams, Error, Limits, MixParams, ModelSpec, Options, compress, compress_stream, decompress,
    decompress_stream, sha256,
};
use proptest::prelude::*;

/// Compress, decompress, and check bytes and digest. Returns the archive.
fn roundtrip(data: &[u8], opts: &Options) -> Vec<u8> {
    let archive = compress(data, opts).expect("compress");
    let restored = decompress(&archive, &Limits::default()).expect("decompress");
    assert_eq!(restored.len(), data.len(), "length differs");
    assert!(restored == data, "content differs");
    assert_eq!(sha256(&restored), sha256(data));
    archive
}

fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut mixed = text_bytes(40_000, 3);
    mixed.extend(random_bytes(40_000, 4));
    mixed.extend(vec![0u8; 40_000]);
    mixed.extend(text_bytes(40_000, 5));
    vec![
        ("empty", vec![]),
        ("one byte", vec![0x42]),
        ("two bytes", vec![0, 255]),
        ("zeros", vec![0u8; 100_000]),
        ("ones", vec![0xFF; 70_001]),
        ("all byte values", (0..=255u8).cycle().take(5000).collect()),
        ("text", text_bytes(150_000, 1)),
        ("random", random_bytes(100_000, 2)),
        ("mixed", mixed),
        ("short text", b"hello hello hello hello hello".to_vec()),
    ]
}

#[test]
fn default_options() {
    for (name, data) in corpus() {
        // Default hash_bits allocates ~96 MiB; do it once per input.
        let archive = roundtrip(&data, &Options::default());
        eprintln!("{name}: {} -> {}", data.len(), archive.len());
    }
}

#[test]
fn every_order() {
    for order in 0..=8u8 {
        for (name, data) in corpus() {
            let archive = roundtrip(&data, &ctx_opts(order, 12, 1 << 14));
            assert!(!archive.is_empty(), "{name}");
        }
    }
}

#[test]
fn mix_model_every_switch_combination() {
    for flags in 0..32u8 {
        for (name, data) in corpus() {
            let archive = roundtrip(&data, &mix_opts(10, 12, flags, 1 << 14));
            assert!(!archive.is_empty(), "{name}");
        }
    }
}

#[test]
fn mix_model_parameter_extremes() {
    let data = text_bytes(60_000, 9);
    roundtrip(&data, &mix_opts(10, 12, 0, 1 << 16));
    roundtrip(&data, &mix_opts(20, 24, 0, 1 << 16));
    // A match window far smaller than the data and than a block.
    let mut repeats = random_bytes(5000, 3).repeat(30);
    repeats.extend(text_bytes(20_000, 4));
    roundtrip(&repeats, &mix_opts(12, 12, 0, 1 << 15));
}

#[test]
fn mix_model_beats_the_byte_model_on_repeats() {
    // Long repeats of noise: only a match model can exploit them.
    let data = random_bytes(4000, 7).repeat(20);
    let mix = roundtrip(&data, &mix_opts(12, 16, 0, 1 << 16));
    let ctx = roundtrip(&data, &ctx_opts(4, 12, 1 << 16));
    assert!(
        mix.len() * 3 < ctx.len(),
        "mix {} ctx {}",
        mix.len(),
        ctx.len()
    );
    assert!(mix.len() < data.len() / 10);
}

#[test]
fn hash_bits_extremes() {
    let data = text_bytes(60_000, 9);
    roundtrip(&data, &ctx_opts(4, 8, 1 << 16));
    roundtrip(&data, &ctx_opts(3, 22, 1 << 16));
}

#[test]
fn block_size_boundaries() {
    let data = text_bytes(4096, 7);
    for block_size in [1u32, 2, 5, 6, 7, 100, 4095, 4096, 4097, 1 << 20] {
        for len in [0usize, 1, 99, 100, 101, 200, 4095, 4096] {
            roundtrip(&data[..len], &ctx_opts(3, 10, block_size));
        }
    }
}

#[test]
fn stored_model() {
    let opts = Options {
        model: ModelSpec::None,
        block_size: 1000,
        independent_blocks: false,
    };
    for (_, data) in corpus() {
        let archive = roundtrip(&data, &opts);
        let blocks = data.len().div_ceil(1000);
        assert_eq!(
            archive.len(),
            HEADER_LEN + TRAILER_LEN + blocks * BLOCK_HEADER_LEN + data.len()
        );
    }
}

#[test]
fn independent_blocks() {
    let mut opts = ctx_opts(4, 12, 8192);
    let data = text_bytes(100_000, 11);
    let carried = roundtrip(&data, &opts);
    opts.independent_blocks = true;
    let independent = roundtrip(&data, &opts);
    // Resetting the model at each block throws information away.
    assert!(independent.len() > carried.len());
}

#[test]
fn compresses_redundant_data() {
    let text = text_bytes(200_000, 13);
    let archive = roundtrip(&text, &ctx_opts(4, 16, 1 << 18));
    assert!(archive.len() < text.len() / 2, "got {}", archive.len());

    let zeros = vec![0u8; 1 << 20];
    let archive = roundtrip(&zeros, &ctx_opts(2, 10, 1 << 18));
    assert!(archive.len() < 4096, "got {}", archive.len());
}

#[test]
fn incompressible_data_expands_by_metadata_only() {
    let data = random_bytes(300_000, 17);
    let opts = ctx_opts(4, 14, 1 << 16);
    let archive = roundtrip(&data, &opts);
    let blocks = data.len().div_ceil(1 << 16);
    assert_eq!(
        archive.len(),
        HEADER_LEN + TRAILER_LEN + blocks * BLOCK_HEADER_LEN + data.len(),
        "every block should fall back to stored"
    );
}

#[test]
fn model_stays_in_sync_across_stored_blocks() {
    // Alternate compressible and incompressible blocks so coded blocks
    // follow stored ones; any encoder/decoder model drift breaks decoding.
    let mut data = Vec::new();
    for i in 0..12u64 {
        if i % 2 == 0 {
            data.extend(text_bytes(4096, i));
        } else {
            data.extend(random_bytes(4096, i));
        }
    }
    for opts in [ctx_opts(4, 12, 4096), mix_opts(12, 14, 0, 4096)] {
        let archive = compress(&data, &opts).unwrap();
        let mut out = Vec::new();
        let stats = decompress_stream(&archive[..], &mut out, &Limits::default()).unwrap();
        assert_eq!(out, data);
        assert_eq!(stats.blocks_coded, 6);
        assert_eq!(stats.blocks_stored, 6);
        assert_eq!(stats.archive_len, archive.len() as u64);
    }
}

#[test]
fn output_is_deterministic() {
    let data = text_bytes(80_000, 21);
    let opts = ctx_opts(4, 14, 1 << 14);
    assert_eq!(
        compress(&data, &opts).unwrap(),
        compress(&data, &opts).unwrap()
    );
}

/// Guards the bitstream: if this fails, the format or the model changed and
/// `FORMAT_VERSION` (or the model identifier) must change with it.
#[test]
fn golden_archive_digest() {
    let mut data = text_bytes(50_000, 1);
    data.extend(random_bytes(3000, 2));
    data.extend(vec![7u8; 5000]);
    let archive = compress(&data, &ctx_opts(4, 12, 1 << 14)).unwrap();
    let hex: String = sha256(&archive)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        (archive.len(), hex.as_str()),
        GOLDEN,
        "bitstream changed: bump the format version or model id, then update GOLDEN"
    );
}
/// The same guard for the mixing model.
#[test]
fn golden_mix_archive_digest() {
    let mut data = text_bytes(50_000, 1);
    data.extend(random_bytes(3000, 2));
    data.extend(vec![7u8; 5000]);
    data.extend(text_bytes(20_000, 1));
    let archive = compress(&data, &mix_opts(12, 14, 0, 1 << 14)).unwrap();
    let hex: String = sha256(&archive)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        (archive.len(), hex.as_str()),
        GOLDEN_MIX,
        "bitstream changed: bump the format version or model id, then update GOLDEN_MIX"
    );
}
const GOLDEN_MIX: (usize, &str) = (
    9670,
    "d0d1e894749fcd3a9a2a16ed84c7badee171186372b75020a31112cfc81d0834",
);

const GOLDEN: (usize, &str) = (
    9666,
    "3eb4a12dbe372ec82de209d9849eec20b916f6e4e98003f0065ae7515d9c1b34",
);

#[test]
fn stream_reports_length_mismatch() {
    let data = text_bytes(1000, 1);
    let opts = ctx_opts(2, 10, 256);
    let mut out = Vec::new();
    assert!(matches!(
        compress_stream(&data[..], &mut out, 1001, &opts),
        Err(Error::LengthMismatch)
    ));
    let mut out = Vec::new();
    assert!(matches!(
        compress_stream(&data[..], &mut out, 999, &opts),
        Err(Error::LengthMismatch)
    ));
}

#[test]
fn invalid_options_are_rejected() {
    let bad = [
        ctx_opts(9, 12, 1024),
        ctx_opts(4, 7, 1024),
        ctx_opts(4, 25, 1024),
        ctx_opts(4, 12, 0),
        ctx_opts(4, 12, (1 << 26) + 1),
        mix_opts(9, 16, 0, 1024),
        mix_opts(25, 16, 0, 1024),
        mix_opts(12, 11, 0, 1024),
        mix_opts(12, 29, 0, 1024),
        mix_opts(12, 16, 32, 1024),
    ];
    for opts in bad {
        assert!(matches!(
            compress(b"x", &opts),
            Err(Error::InvalidOptions(_))
        ));
    }
}

/// A reader that hands out data in awkward, irregular pieces.
struct Dribble<'a> {
    data: &'a [u8],
    step: usize,
}

impl std::io::Read for Dribble<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.step = self.step % 7 + 1;
        let n = self.step.min(buf.len()).min(self.data.len());
        buf[..n].copy_from_slice(&self.data[..n]);
        self.data = &self.data[n..];
        Ok(n)
    }
}

#[test]
fn streaming_with_short_reads() {
    let data = text_bytes(20_000, 31);
    let opts = ctx_opts(3, 10, 3000);
    let mut archive = Vec::new();
    let reader = Dribble {
        data: &data,
        step: 0,
    };
    compress_stream(reader, &mut archive, data.len() as u64, &opts).unwrap();
    assert_eq!(archive, compress(&data, &opts).unwrap());

    let mut out = Vec::new();
    let reader = Dribble {
        data: &archive,
        step: 3,
    };
    decompress_stream(reader, &mut out, &Limits::default()).unwrap();
    assert_eq!(out, data);
}

fn arb_data() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Arbitrary bytes.
        proptest::collection::vec(any::<u8>(), 0..3000),
        // Small alphabet: highly compressible.
        proptest::collection::vec(0u8..4, 0..6000),
        // Repeated chunk.
        (proptest::collection::vec(any::<u8>(), 1..40), 1usize..300)
            .prop_map(|(chunk, n)| chunk.repeat(n)),
        // Runs.
        proptest::collection::vec((any::<u8>(), 1usize..600), 0..20).prop_map(|runs| {
            runs.into_iter()
                .flat_map(|(b, n)| std::iter::repeat_n(b, n))
                .collect()
        }),
    ]
}

fn arb_opts() -> impl Strategy<Value = Options> {
    (
        0u8..=8,
        8u8..=12,
        1u32..5000,
        any::<bool>(),
        0u8..10,
        0u8..32,
    )
        .prop_map(
            |(max_order, hash_bits, block_size, independent_blocks, pick, flags)| Options {
                model: if pick == 0 {
                    ModelSpec::None
                } else if pick < 6 {
                    ModelSpec::Mix(MixParams {
                        hash_bits: hash_bits.max(10),
                        match_bits: 12 + max_order % 3,
                        flags,
                    })
                } else {
                    ModelSpec::Ctx(CtxParams {
                        max_order,
                        hash_bits,
                    })
                },
                block_size,
                independent_blocks,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn prop_roundtrip(data in arb_data(), opts in arb_opts()) {
        let archive = compress(&data, &opts).unwrap();
        let restored = decompress(&archive, &Limits::default()).unwrap();
        prop_assert_eq!(&restored, &data);
        // Never worse than storing: metadata is the only overhead.
        let blocks = data.len().div_ceil(opts.block_size as usize);
        prop_assert!(
            archive.len() <= HEADER_LEN + TRAILER_LEN + blocks * BLOCK_HEADER_LEN + data.len()
        );
    }
}

#[test]
fn ablation_default_matches_real_archive_size() {
    use neurozip::{Ablation, ablation_size};
    let mut opts_list = vec![
        ctx_opts(4, 12, 4096),
        ctx_opts(0, 8, 1000),
        mix_opts(12, 14, 0, 4096),
        mix_opts(10, 12, 15, 1500),
    ];
    let mut independent = ctx_opts(3, 10, 2048);
    independent.independent_blocks = true;
    opts_list.push(independent);
    opts_list.push(Options {
        model: ModelSpec::None,
        block_size: 777,
        independent_blocks: false,
    });
    for opts in &opts_list {
        for (name, data) in corpus() {
            let real = compress(&data, opts).unwrap().len() as u64;
            let measured = ablation_size(&data, opts, &Ablation::default()).unwrap();
            assert_eq!(measured, real, "{name}");
        }
    }
}

#[test]
fn ablation_variants_cost_what_they_should() {
    use neurozip::{Ablation, ablation_size};
    let opts = ctx_opts(4, 12, 4096);
    let base = Ablation::default();

    // Without fallback, incompressible data grows beyond stored size.
    let noise = random_bytes(50_000, 1);
    let with = ablation_size(&noise, &opts, &base).unwrap();
    let without = ablation_size(
        &noise,
        &opts,
        &Ablation {
            fallback: false,
            ..base
        },
    )
    .unwrap();
    assert!(without > with);

    // A model frozen early predicts worse than one that keeps learning,
    // and one that never learns cannot compress at all.
    let text = text_bytes(100_000, 2);
    let full = ablation_size(&text, &opts, &base).unwrap();
    let frozen = |n| {
        ablation_size(
            &text,
            &opts,
            &Ablation {
                freeze_after: Some(n),
                ..base
            },
        )
        .unwrap()
    };
    assert!(frozen(1_000) > full);
    assert!(frozen(0) > text.len() as u64);
    assert_eq!(frozen(text.len() as u64), full);
}

/// Synthetic x86-like data: many CALL instructions to a few targets.
fn fake_code(len: usize, seed: u64) -> Vec<u8> {
    let noise = random_bytes(len, seed);
    let mut out = noise.clone();
    let mut i = 0;
    while i + 12 < len {
        let target = 0x1000 + (noise[i] as usize % 8) * 0x40;
        let rel = (target as i64 - (i as i64 + 5)) as i32;
        out[i] = 0xE8;
        out[i + 1..i + 5].copy_from_slice(&rel.to_le_bytes());
        // Low-entropy filler between calls.
        for b in &mut out[i + 5..i + 12] {
            *b &= 0x07;
        }
        i += 12;
    }
    out
}

#[test]
fn x86_filter_roundtrips_and_helps() {
    let data = fake_code(60_000, 5);
    // Blocks smaller and larger than the data, and one that splits calls.
    for block_size in [4096u32, 7001, 1 << 16] {
        let with = roundtrip(&data, &mix_opts(12, 14, 0, block_size));
        let without = roundtrip(&data, &mix_opts(12, 14, 16, block_size));
        assert!(
            with.len() < without.len(),
            "{} vs {}",
            with.len(),
            without.len()
        );
    }
    // Filtered blocks next to stored ones keep the model in step.
    let mut mixed = fake_code(8192, 6);
    mixed.extend(random_bytes(8192, 7));
    mixed.extend(fake_code(8192, 8));
    mixed.extend(text_bytes(8192, 9));
    roundtrip(&mixed, &mix_opts(12, 14, 0, 8192));
    roundtrip(&mixed, &mix_opts(12, 14, 0, 5000));
}
