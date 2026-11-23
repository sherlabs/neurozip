# The `.nz` archive format, version 1

Status: experimental. The format may change; any change to the bitstream
must change the format version or the model identifier. The test
`golden_archive_digest` fails if the bitstream changes without that.

All multi-byte integers are little-endian. An archive holds exactly one
byte string.

```
archive := header block* trailer
```

## Header (28 bytes)

| Offset | Size | Field | Meaning |
|-------:|-----:|-------|---------|
| 0  | 4 | magic | `4E 5A 49 50` (`NZIP`) |
| 4  | 1 | format version | `1` |
| 5  | 1 | model identifier | `0` none, `1` byte context model, `2` bit mixing model |
| 6  | 2 | codec flags | bit 0: independent blocks. All other bits must be 0 |
| 8  | 1 | model parameter A | model 1: highest order, 0..=8. Model 2: hash bits, 10..=24 |
| 9  | 1 | model parameter B | model 1: hash bits, 8..=24. Model 2: match window bits, 12..=28 |
| 10 | 1 | model parameter C | model 2: feature switches (below). Otherwise 0 |
| 11 | 1 | reserved | must be 0 |
| 12 | 4 | block size | raw bytes per block, 1..=67108864 (64 MiB) |
| 16 | 8 | original length | total raw bytes |
| 24 | 4 | header CRC | CRC-32 (IEEE) of bytes 0..24 |

For model identifier 0 all model parameters must be 0.

A decoder must reject: wrong magic, unknown version, CRC mismatch, unknown
flag bits, unknown model identifier, out-of-range parameters, non-zero
reserved bytes, out-of-range block size.

Before allocating, a decoder computes the model's memory from the
parameters and rejects the archive if it exceeds the caller's limit.

## Blocks

The original data is cut into blocks of `block size` bytes; only the last
may be shorter. There are `ceil(original length / block size)` blocks, so
an empty input has none.

| Offset | Size | Field | Meaning |
|-------:|-----:|-------|---------|
| 0  | 1 | codec | `0` stored, `1` model + range coder, `2` x86 filter + model + range coder |
| 1  | 4 | raw length | bytes this block decodes to |
| 5  | 4 | payload length | bytes of payload that follow |
| 9  | 4 | block CRC | CRC-32 of the raw block |
| 13 | n | payload | |

Rules a decoder enforces:

* `raw length` equals `min(block size, bytes still expected)`.
* codec 0: `payload length == raw length`; the payload is the raw block.
* codec 1: `5 <= payload length < raw length`, and the model identifier
  is not 0.
* codec 2: as codec 1, and the model must be model 2 with its x86 filter
  switch not set.
* any other codec value is an error.

The encoder uses codec 1 or 2 only when its output is strictly smaller
than the raw block, so an archive is never larger than the input plus 60
bytes plus 13 bytes per block.

### The x86 filter (codec 2)

Defined by `src/x86.rs`. With model 2, a block is *filtered* when at least
1 byte in 256 starts a rewritable instruction: a byte `E8` or `E9` whose
fourth following byte is `00` or `FF` (counted left to right, skipping the
five bytes of each one found). Filtering replaces the three bytes after
each such opcode by their value plus the opcode's offset in the whole
input, modulo 2^24, visiting positions from the last to the first. The
inverse visits positions from the first to the last and subtracts.

For a filtered block the model codes, and learns from, the filtered bytes;
the decoder applies the inverse after decoding. The block CRC and the
trailer hash always cover the original bytes. If a filtered block does not
shrink it is stored unfiltered (codec 0); the decoder then repeats the
same test on the stored bytes and shows the model the filtered form, so
both sides stay in step.

### Codec 1 payload

The payload is a range coder stream (see `src/rangecoder.rs`): 32-bit
range, carry propagation, frequency total 2^16. Its first byte is 0.

With model 2, each byte is coded as eight binary decisions, most
significant bit first. For each bit the model gives a 12-bit probability
`p` (1..=4095) that the bit is 1; the bit is coded with `p * 16` out of
65536 using the coder's binary step (`bound = (range >> 16) * (65536 - p1)`;
a 0 takes `[0, bound)`, a 1 takes the rest); then the model is updated
with the bit.

With model 1, for each byte of the block, in order:

1. the model produces a table of 256 frequencies, each at least 1, summing
   to 65536;
2. the byte is coded in the interval given by that table, symbols in
   increasing byte order;
3. the model is updated with the byte.

A decoder must consume the payload exactly; a payload that is too short or
has bytes left over is an error.

### Model state across blocks

If the independent-blocks flag is clear, model state carries from one
block to the next, and the model is updated with every byte of every
block, **including stored blocks**. If the flag is set, the model returns
to its initial state at the start of every block.

No model state or learned weight is ever stored in the archive. Encoder
and decoder both start from the same defined initial state.

## Trailer (32 bytes)

SHA-256 of the complete original data. Nothing may follow the trailer.

## Model 1: adaptive context model

Defined by `src/model/ctx.rs`; summarised here.

State: the last 8 bytes (initially all zero) and count tables for each
order from 0 to the highest order `N`.

* Order 0: 256 counts. Order 1: 256 counts for each previous byte. A
  context's counts are all replaced by `ceil(c / 2)` when their total
  reaches 16384.
* Order `k >= 2`: a table of `2^hash_bits` slots. The context (the last
  `k` bytes as an integer, most recent byte lowest) is hashed:
  `h = ctx * 0x9E3779B97F4A7C15; h ^= h >> 32; h *= 0xD6E8FEB86659FD93`
  (64-bit wrapping). The slot index is the top `hash_bits` bits of `h` and
  the 16-bit check tag is the next 16 bits. A slot holds the tag and up to
  15 (byte, count) pairs. Updating a slot whose tag differs clears it
  first. A new byte in a full slot replaces the first pair with the
  smallest count. When a count reaches 255 all counts in the slot are
  replaced by `ceil(c / 2)`.

Prediction, with `P` an array of 256 integers scaled by 2^24:

1. `P[s] = 2^24 / 256` for all `s`.
2. For each order from 0 to `N` whose current context has counts (for
   hashed orders: the slot is non-empty and the tag matches), with total
   `T` and `D` distinct bytes:
   `unit = floor(2^24 / (T + D))`,
   `P[s] = floor(P[s] * D * unit / 2^24) + count[s] * unit`.
3. Normalise (`src/freq.rs`): `W = sum(P)`,
   `scale = floor(65280 * 2^32 / W)`,
   `f[s] = 1 + floor(P[s] * scale / 2^32)`, and `65536 - sum(f)` is added
   to the entry with the largest `P` (lowest byte value on a tie).

Memory: `131072 + (N - 1) * 32 * 2^hash_bits` bytes for `N >= 1`.

## Model 2: bit-level mixing model

Defined by `src/model/mix.rs`, which is the normative description; this is
a summary. All arithmetic is integer.

Feature switches (model parameter C): bit 0 no match model, bit 1 no
correction tables, bit 2 no word context, bit 3 mixer learning off, bit 4
no x86 filter. Other bits must be 0. The full model is 0.

Per bit:

1. **Contexts.** Order 0 (direct table), order 1 (direct table) and ten
   hashed contexts: orders 2, 3, 4, 5, 6 and 8; the current word; the
   current word with the previous word; and two sparse contexts (the two
   bytes before the last one; the last byte with the one three back).
   Hashed tables hold 64-byte buckets: a 32-bit check tag and the 15 nodes
   of one nibble's decision tree.
2. **Nodes.** Each order-1 and hashed node is a `u32`: a 16-bit
   probability, a 4-bit count and a 12-bit history of the most recent bits
   seen in that context (behind a leading 1, at most 11 bits). The counter
   update is `p += (target - p) * floor(131072 / (2n + 3)) >> 16`, then
   `n = min(n + 1, 15)`.
3. **Inputs.** 26 values in the stretch domain (`ln(p / (1 - p))`, 8
   fractional bits, clamped to +-2047): the order-0 counter; for each of
   the eleven nodes, its counter (0 if never updated) and the output of a
   *history map*, a slowly adapting table that turns the node's bit history
   into a probability (one table per context kind and bit position); the
   match model's prediction; a counter indexed by the match model's
   predicted byte; and a constant 256.
4. **Match model.** A hash of the last 6 bytes indexes a table of
   positions in a ring buffer of `2^match bits` bytes. A candidate is
   accepted if at least 6 of the preceding bytes agree (up to 32 are
   checked); the match then extends while its predicted byte is correct.
   Its input is the stretch of an adaptive counter chosen by match length,
   signed by the predicted bit.
5. **Mixer.** Three first-layer weight sets, selected by the partial
   current byte, the match-length bucket and the previous byte, each give
   `dot = sum(x * w) >> 16` (weights have 16 fractional bits, initial
   value 9830). A second layer, selected by the partial current byte,
   combines the three (initial weight 21845). Probabilities come from
   `squash`, a 33-point interpolated logistic table. Training:
   `w += (x * (target - p) * rate + 32768) >> 16`, with rate 16 for the
   first layer and 8 for the second.
6. **Correction tables.** Two tables of 65536 contexts by 33 interpolation
   points refine the mixed probability (update shift 6); the output is
   `(p + a1 + 2 * a2 + 2) >> 2`, clamped to 1..=4095.

Memory: `640 * 2^hash bits + 2 * 2^match bits` bytes plus about 11 MiB
of fixed tables (about 180 MiB at the defaults).
