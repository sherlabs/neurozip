# Prior art and patent notes

NeuroZip makes **no novelty claim**. Every technique in the code so far is
decades old and well documented. This file records where each one comes
from, what the later phases will be building on, and which patent areas to
stay aware of.

This is an engineering note written from general knowledge of the field.
It is not a patent search and not legal advice. Citations are from memory
and should be checked against the sources before being quoted elsewhere.

## What Phase 1 uses

| Component | Prior art |
|-----------|-----------|
| Range coding | G. N. N. Martin, "Range encoding: an algorithm for removing redundancy from a digitised message" (1979). Arithmetic coding: Rissanen; Pasco (1976); Witten, Neal & Cleary, CACM (1987). |
| Carry-propagating byte-wise range coder with a 64-bit `low` and cached byte | The construction used in LZMA (Igor Pavlov, 7-Zip) and described by Schindler (1998) and Subbotin (carryless variant). `src/rangecoder.rs` is an independent implementation of that public design. |
| Finite-context byte prediction, blending orders | PPM: Cleary & Witten (1984). Escape method C: Moffat (1990). The blending here is the interpolated form of the Witten-Bell estimate (Witten & Bell, 1991), without PPM's symbol exclusion. |
| Hashed context tables with check tags, count halving | Standard in PPM, CTW and PAQ-family implementations (Mahoney's PAQ series, 2002 onward). |
| Power-of-two frequency totals, strictly positive normalised tables | Universal in table-based entropy coders (e.g. zstd's FSE normalisation by Collet). |
| Block container with stored fallback, CRC-32, SHA-256 | Conventional (gzip, xz, zstd frame formats). |

## What the mixing model uses

`src/model/mix.rs` follows the architecture of Matt Mahoney's **lpaq1**
(2007) and the PAQ family it comes from: bitwise prediction, per-context
adaptive probability counters in nibble-sized hashed buckets, a match
model for long repeats, mixing in the logistic ("stretch") domain by
gated linear networks trained online with the coding error, and adaptive
probability maps (secondary symbol estimation) after the mixer. The
`squash`/`stretch` construction and the 12-bit probability convention are
PAQ's. The code was written for this project and differs in detail
(direct probability counters instead of bit-history state machines, a
different mixer layout), which most likely makes it weaker than lpaq, not
stronger. The PAQ sources are GPL; no PAQ code was copied.

Gated linear mixing was later analysed in its own right by Veness et al.,
"Gated Linear Networks" (2017), and Mattern, "Mixing Strategies in Data
Compression" (2012).

### Later additions to the mixing model

All of these are established techniques; none is original to this project.

* **Bit histories mapped to probabilities.** Each context node keeps the
  last few bits seen in that context, and an adaptive table turns that
  history into a probability. This is indirect context modelling: PAQ's
  bit-history states with adaptive state maps, and zpaq's ICM and ISSE
  components (Mahoney). NeuroZip stores the raw recent bits instead of a
  state-machine state, which is simpler and most likely weaker for long
  histories.
* **Sparse and word contexts**, including the two-word context: standard
  in PAQ8 and lpaq.
* **x86 call/jump filter** (`src/x86.rs`): the E8/E9 transform, used in
  7-Zip (BCJ, Pavlov), PAQ8 and zpaq. Written independently here; the
  backward-encode, forward-decode order follows the same reasoning as the
  published implementations.

On the files measured (`docs/REPORT.md`) the result is smaller output than
lpaq1 and zpaq at the settings tested. That is an engineering result from
combining known parts and tuning them, not a new algorithm, and it says
nothing about cmix or the PAQ8 family, which were not run.

## What later work would build on

These are the systems any "online-learned predictor + entropy coder"
experiment has to be compared with, and they set realistic expectations:

* **PAQ / lpaq / ZPAQ** (Mahoney and others): bit-level context mixing,
  where many models' predictions are combined by small online-trained
  networks working in the logistic domain, in fixed-point arithmetic, with
  no weights transmitted. This is the closest established prior art to the
  planned linear mixer.
* **cmix** (Knoll) and **lstm-compress**: add LSTM predictors to context
  mixing.
* **NNCP** (Bellard, 2019 and 2021): LSTM and Transformer models trained
  online during compression, with deterministic arithmetic so the decoder
  reproduces the encoder exactly. This directly demonstrates the idea in
  this project's goal.
* **DeepZip** (Goyal et al., 2018), **DZip** (2021), **TRACE** (2022),
  **CMIX**-derived entries on the Large Text Compression Benchmark, and
  "Language Modeling Is Compression" (Delétang et al., 2023).
* Schmidhuber & Heil, "Sequential neural text compression" (1996), an
  early neural predictor plus arithmetic coder.

Known result to keep in mind: on text, strong context-mixing and neural
compressors beat xz and zstd on ratio by a wide margin, at speeds
thousands of times slower. A new experiment in this space is therefore
expected to trade speed for ratio; whether NeuroZip does so competitively
is exactly what Phase 3 must measure.

## Patent notes

* **Arithmetic and range coding.** The well-known IBM, Mitsubishi and
  related arithmetic-coding patents from the 1980s and early 1990s have
  expired. Martin's 1979 range coder paper predates them and has long been
  treated as unencumbered.
* **ANS.** Not used. There has been patent activity around asymmetric
  numeral systems (a Google application that was abandoned, and a
  Microsoft patent on rANS-related encoding granted in 2022). Staying on
  range coding avoids that area entirely. If ANS is ever considered, get a
  proper review first.
* **PPM and context mixing.** The foundational PPM work is from 1984 and
  1990. PAQ is open source with a long public history. Specific commercial
  PPM variants and products have had patents; this code implements only
  the textbook interpolated estimate.
* **Neural compression.** This is an active filing area (learned image and
  video codecs in particular, and some filings on neural probability
  models for entropy coding). Before Phase 2 results are published or the
  code is used commercially, a real search should cover: online-trained
  neural predictors for lossless coding, deterministic or quantised
  training for encoder/decoder agreement, and model mixing. The published
  work listed above is useful as dated prior art for those ideas.
* **Classical codecs for fallback and benchmarks** (zstd, Brotli, xz):
  open source under permissive licences; zstd and Brotli come with patent
  grants or non-assertion statements from their sponsors. Check the
  licence of whichever implementation is linked in Phase 2.

## Rules this project follows

1. No performance or novelty claim without a measurement in the
   repository that reproduces it.
2. Every borrowed technique is named here with its source.
3. Results from quick local runs are labelled as such and kept separate
   from benchmark results on held-out corpora.
