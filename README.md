# NeuroZip

An experimental general-purpose lossless compressor. The question it is
built to answer: can a deterministic, online-learned predictor combined
with integer range coding beat classical compressors on held-out data?

**Status.** The compressor, the archive format, an online-trained mixing
model and a benchmark against seven other configurations are built and
tested. A stand-alone neural predictor (MLP) and fallback to a classical
codec are not built. **Nothing here is a new algorithm**: every technique
comes from the PAQ, lpaq and zpaq line of context-mixing compressors (see
`docs/PRIOR_ART.md`).

**Measured result** (`docs/REPORT.md`; 14 files the model had never been
run on, one machine). Mean bits per byte over the five compressible data
types, lower is better:

| Compressor | Mean bpb |
|---|---:|
| NeuroZip | 1.636 |
| zpaq -m5 | 1.664 |
| lpaq1 (level 6) | 1.715 |
| PPMd (7-Zip) | 1.903 |
| Brotli -q 11 | 1.977 |
| xz -9 | 2.005 |
| zstd -19 | 2.060 |

NeuroZip is the smallest on 8 of the 11 compressible files. It is also
slow (under 1 MB/s in both directions on the loaded test machine, about
1 MB/s when idle) and needs about 180 MiB of memory, the same class as
zpaq. cmix and the PAQ8 family were not benchmarked and are expected to
produce smaller files still.

## Findings and conclusion

Findings (all in `docs/REPORT.md`, raw numbers in `bench/results/`):

* A from-scratch bit-level context-mixing model with an integer range
  coder, no stored weights and exact lossless round trips (SHA-256 checked)
  beats zstd -19, Brotli -q 11, xz -9, PPMd, lpaq1 and zpaq -m5 on size
  over the unseen final test set: 1.636 bits per byte against 1.664 for the
  next best (zpaq), about 1.7% smaller; 17% smaller than the best classical
  tool. It is smallest on 8 of 11 compressible files.
* The gains came from known PAQ-family parts (bit histories, sparse and
  word contexts, match model, two-layer mixer, correction tables, x86
  filter) combined and tuned on a separate development set. **No novel
  algorithm was found or claimed.**
* **cmix is better.** On one file (`json_apache_builds.json`, 127,275 bytes)
  cmix produced 4,397 bytes against NeuroZip's 5,952 (NeuroZip 35% larger),
  at 286 s CPU and 8.6 GiB. That was a compress-only check on a single
  file; the PAQ8 family was not run.
* The price is speed and memory: about 1 MB/s both ways and about 180 MiB.
  Timings were taken on a loaded machine and are approximate; sizes are
  not affected.
* The first test set was not re-run with the final model, the fuzz targets
  were compiled but never run, and power was not measured.

Conclusion: the experiment's question is answered for the classical tools
and for lpaq1/zpaq: yes, a deterministic online-learned predictor with
range coding beats them on held-out data, at zpaq-class speed and memory.
It does not beat cmix, and the result is engineering, not a new method.
The project's stretch goal (a better engine than all available ones with a
novel idea) is **not met**. The untested next step is an LSTM-style neural
input to the mixer, which is where cmix gets its edge; it would cost a
large slowdown. That work is parked here.

## What exists

| Piece | File |
|-------|------|
| Integer range encoder / decoder | `src/rangecoder.rs` |
| Deterministic normalisation to strictly positive frequency tables | `src/freq.rs` |
| `Predictor` contract | `src/model/mod.rs` |
| Bit-level mixing model with match model (the default) | `src/model/mix.rs` |
| Reversible x86 call/jump filter | `src/x86.rs` |
| Byte-level context model, orders 0..N (the earlier baseline) | `src/model/ctx.rs` |
| `.nz` container (magic, version, model id, flags, lengths, checksums) | `src/format.rs`, `docs/FORMAT.md` |
| Block codec, stored fallback, resource limits | `src/codec.rs` |
| CLI | `src/main.rs` |
| libFuzzer targets | `fuzz/` |
| Benchmark: corpora, runner, report generator | `bench/` |
| Benchmark report and charts | `docs/REPORT.md`, `docs/charts/` |

Properties:

* **Lossless and verified.** Each block carries a CRC-32, the archive ends
  with a SHA-256 of the original data, and the decoder checks both.
* **Deterministic.** Integer arithmetic only, `#![forbid(unsafe_code)]`.
  The encoder and decoder update the model identically after every bit.
  No weights or model state are stored in the archive; the mixer starts
  from a fixed state and learns from the data being coded.
* **Bounded memory.** Model memory is fixed by two header parameters
  (default about 180 MiB) and the decoder refuses archives that ask for
  more than its limit before allocating. Data is processed block by block.
* **Rejects damaged input.** Malformed archives produce an error, never a
  panic and never silently wrong output.
* **Never much bigger than the input.** Blocks that do not compress are
  stored; worst case is 60 bytes plus 13 per block.

## Build and test

```
cargo build --release
cargo test --release
```

83 tests: unit tests for each module, round trips over every model order
and block boundary, a property test over arbitrary data and options,
exhaustive single-bit corruption and truncation of sample archives for
both models, property tests feeding garbage to the decoder, pinned archive
digests that guard each model's bitstream, and end-to-end CLI tests.

The property tests run a few hundred cases by default. For a longer run:

```
PROPTEST_CASES=20000 cargo test --release prop_
```

Coverage-guided fuzzing needs a nightly toolchain and `cargo-fuzz`:

```
cargo +nightly fuzz run decompress
cargo +nightly fuzz run roundtrip
```

## Use

```
neurozip compress   file            # writes file.nz
neurozip decompress file.nz         # writes file
neurozip compress   file --model ctx # the earlier byte-level model
neurozip bench      a.txt b.bin
neurozip --help
```

`bench` compresses and decompresses in memory, checks the round trip, and
prints archive size (all metadata included), bits per byte and single-run
throughput. `--json` gives one JSON object per line.

`neurozip ablate` reports the size an encoder variant would produce
(`--mix-off match,apm,word,x86,mixer`, `--no-fallback`,
`--independent-blocks`; for the ctx model also `--orders` and
`--freeze-after <percent>`).

## Benchmark

```
python3 bench/fetch_corpora.py   # pinned test and development files
sh bench/build_tools.sh          # builds lpaq1
python3 bench/run_bench.py --sets final --output final.json
python3 bench/run_bench.py --sets fresh --skip-ablations --output fresh.json
python3 bench/make_report.py     # docs/REPORT.md and charts
```

Needs `zstd`, `brotli`, `xz`, `7zz` and `zpaq` on the PATH and Python 3
(standard library only). Python is used for the harness and charts, not
in the compressor. There are three test sets; only the last (`final`) was
unseen by the current model, and the report says so.

## Development set

`bench/fetch_corpora.py` also builds `bench/dev/` (pinned in
`bench/dev.lock.json`): different books, projects, JSON files, log sources
and binaries from the benchmark. It is the only data used for tuning;
`bench/devscore.sh` prints the score on it. The benchmark files are never
used to choose a constant.

## Known limits

* Speed: about 1 MB/s in both directions on an idle machine. Decoding
  re-runs the model, so it can never be fast in the way zstd or xz are.
* Memory: about 180 MiB at the defaults, for both directions.
* Table sizes are fixed, so very large inputs will fill them; this has not
  been measured beyond 4 MiB files.
* Probabilities are 12 bits, which caps the best possible ratio at a few
  thousand to one.
* The format stores the length up front, so compressing from stdin
  buffers the whole input.
* Release builds have integer overflow checks off for speed; the test
  suite runs with them on.
* The fuzz targets compile but have not been run under libFuzzer here
  (no nightly toolchain on the development machine).
* The benchmark's speed figures were taken on a machine that was not
  idle; see the report.

## Not built yet

* A stand-alone neural predictor (small quantised MLP) as another mixer
  input, with the "neural only" ablation.
* Per-block fallback to a classical codec (zstd).
* A full comparison against cmix and the PAQ8 family.

## Licence

MIT or Apache-2.0, at your option: see `LICENSE-MIT` and
`LICENSE-APACHE`. Contributions are accepted under the same terms.

Third-party notes: the benchmark downloads test files (books, source
code, JSON, logs, binaries) from their original sources at run time and
does not redistribute them; each keeps its own licence. Comparison tools
(zstd, Brotli, xz, 7-Zip, zpaq, lpaq1, cmix) are used as external programs
and none of their code is included. No PAQ code was copied (see
`docs/PRIOR_ART.md`, which also has patent notes; not legal advice).
