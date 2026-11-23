//! Integer range coder (carry-propagating, byte-oriented).
//!
//! This is the classic 32-bit range coder with a 64-bit `low` and deferred
//! carry propagation, the same construction used by LZMA's `rc`. Symbols are
//! coded against cumulative frequency tables whose total is the fixed power
//! of two [`TOTAL`], so the only division is one `code / r` in the decoder.
//!
//! All arithmetic is integer and fully specified, so encoder and decoder
//! agree bit for bit on every platform.
//!
//! Stream layout: the first byte is always zero, followed by the coded
//! bytes. An encoder that performed `n` renormalisation shifts emits exactly
//! `n + 5` bytes and a decoder of the same symbol sequence consumes exactly
//! `n + 5` bytes; [`RangeDecoder::finish`] uses this to reject payloads that
//! are too short or too long.

/// Number of bits of probability resolution.
pub const TOTAL_BITS: u32 = 16;
/// Sum of every frequency table handed to the coder.
pub const TOTAL: u32 = 1 << TOTAL_BITS;

const TOP: u32 = 1 << 24;

/// Range encoder writing into an in-memory buffer.
pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(cap: usize) -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::with_capacity(cap),
        }
    }

    /// Bytes emitted so far. Up to `cache_size` further bytes are pending,
    /// and `finish` adds the flush, so this is a lower bound on the final
    /// size.
    pub fn bytes_emitted(&self) -> usize {
        self.out.len()
    }

    /// Encode a symbol occupying `[cum, cum + freq)` out of [`TOTAL`].
    ///
    /// # Panics
    /// Panics if `freq == 0` or `cum + freq > TOTAL`; either would produce an
    /// undecodable stream, so it is a programming error, never a data error.
    #[inline]
    pub fn encode(&mut self, cum: u32, freq: u32) {
        assert!(
            freq > 0 && cum <= TOTAL && freq <= TOTAL - cum,
            "bad interval"
        );
        let r = self.range >> TOTAL_BITS;
        self.low += r as u64 * cum as u64;
        self.range = r * freq;
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Encode one binary decision. `p1` is the probability that the bit is
    /// 1, out of [`TOTAL`], and must be in `1..TOTAL`.
    ///
    /// Unlike [`encode`](Self::encode) the upper symbol receives the whole
    /// remainder of the range, so no code space is wasted.
    #[inline]
    pub fn encode_bit(&mut self, bit: bool, p1: u32) {
        assert!(p1 > 0 && p1 < TOTAL, "bad probability");
        let bound = (self.range >> TOTAL_BITS) * (TOTAL - p1);
        if bit {
            self.low += bound as u64;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[inline]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut byte = self.cache;
            loop {
                self.out.push(byte.wrapping_add(carry));
                byte = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Flush and return the coded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

/// Range decoder over a complete payload.
pub struct RangeDecoder<'a> {
    code: u32,
    range: u32,
    r: u32,
    input: &'a [u8],
    pos: usize,
    overrun: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Minimum length of any valid payload.
    pub const MIN_LEN: usize = 5;

    /// Returns `None` if the payload cannot have come from [`RangeEncoder`].
    pub fn new(input: &'a [u8]) -> Option<Self> {
        if input.len() < Self::MIN_LEN || input[0] != 0 {
            return None;
        }
        let code = u32::from_be_bytes([input[1], input[2], input[3], input[4]]);
        Some(RangeDecoder {
            code,
            range: 0xFFFF_FFFF,
            r: 0,
            input,
            pos: 5,
            overrun: false,
        })
    }

    /// Value in `0..TOTAL` identifying the next symbol: the symbol whose
    /// interval `[cum, cum + freq)` contains it.
    ///
    /// For corrupt input the true quotient can exceed `TOTAL - 1`; it is
    /// clamped so the caller always finds a symbol and nothing can underflow.
    #[inline]
    pub fn target(&mut self) -> u32 {
        self.r = self.range >> TOTAL_BITS;
        (self.code / self.r).min(TOTAL - 1)
    }

    /// Consume the symbol found from the preceding [`target`](Self::target)
    /// call. Requires `cum <= target < cum + freq`.
    #[inline]
    pub fn consume(&mut self, cum: u32, freq: u32) {
        debug_assert!(freq > 0 && cum + freq <= TOTAL);
        // cum <= target <= code / r, hence r * cum <= code: no underflow.
        self.code -= self.r * cum;
        self.range = self.r * freq;
        self.normalize();
    }

    /// Decode one binary decision coded with
    /// [`RangeEncoder::encode_bit`] and the same `p1`.
    #[inline]
    pub fn decode_bit(&mut self, p1: u32) -> bool {
        debug_assert!(p1 > 0 && p1 < TOTAL);
        let bound = (self.range >> TOTAL_BITS) * (TOTAL - p1);
        // On corrupt input `code` may be at or beyond `range`; that simply
        // decodes as a 1 and cannot underflow.
        let bit = self.code >= bound;
        if bit {
            self.code -= bound;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        self.normalize();
        bit
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            let byte = match self.input.get(self.pos) {
                Some(&b) => {
                    self.pos += 1;
                    b
                }
                None => {
                    self.overrun = true;
                    0
                }
            };
            // On corrupt input `code` may exceed `range`; the shift simply
            // drops the high bits. No panic, and checksums catch the damage.
            self.code = (self.code << 8) | byte as u32;
            self.range <<= 8;
        }
    }

    /// True once the decoder has needed a byte beyond the end of the
    /// payload. A valid payload never causes this, so callers can stop early.
    #[inline]
    pub fn overrun(&self) -> bool {
        self.overrun
    }

    /// True if the decoder consumed the payload exactly: no missing bytes
    /// and none left over.
    pub fn finish(&self) -> bool {
        !self.overrun && self.pos == self.input.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator for tests (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next() % n as u64) as u32
        }
    }

    /// Build a strictly positive table of `n` symbols summing to TOTAL.
    fn random_table(rng: &mut Rng, n: usize) -> Vec<u32> {
        let mut f = vec![1u32; n];
        let mut left = TOTAL - n as u32;
        // Sometimes extremely skewed, sometimes flat.
        let skew = rng.below(3);
        for (i, slot) in f.iter_mut().enumerate() {
            if left == 0 {
                break;
            }
            let take = match skew {
                0 => rng.below(left + 1),
                1 => rng.below(left.min(64) + 1),
                _ => left / (n - i) as u32,
            };
            *slot += take;
            left -= take;
        }
        f[rng.below(n as u32) as usize] += left;
        assert_eq!(f.iter().sum::<u32>(), TOTAL);
        f
    }

    fn cum_of(f: &[u32], s: usize) -> u32 {
        f[..s].iter().sum()
    }

    fn find(f: &[u32], target: u32) -> (usize, u32) {
        let mut cum = 0;
        for (s, &x) in f.iter().enumerate() {
            if target < cum + x {
                return (s, cum);
            }
            cum += x;
        }
        unreachable!("tables sum to TOTAL");
    }

    #[test]
    fn empty_stream() {
        let bytes = RangeEncoder::new().finish();
        assert_eq!(bytes.len(), 5);
        let dec = RangeDecoder::new(&bytes).unwrap();
        assert!(dec.finish());
    }

    #[test]
    fn roundtrip_changing_tables() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for round in 0..300 {
            let n_syms = 2 + rng.below(255) as usize;
            let len = rng.below(3000) as usize;
            let mut tables = Vec::with_capacity(len);
            let mut syms = Vec::with_capacity(len);
            let mut enc = RangeEncoder::new();
            for _ in 0..len {
                let f = random_table(&mut rng, n_syms);
                // Pick symbols proportional to probability half the time, so
                // both likely and very unlikely symbols are exercised.
                let s = if rng.below(2) == 0 {
                    find(&f, rng.below(TOTAL)).0
                } else {
                    rng.below(n_syms as u32) as usize
                };
                enc.encode(cum_of(&f, s), f[s]);
                tables.push(f);
                syms.push(s);
            }
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for (f, &s) in tables.iter().zip(&syms) {
                let (got, cum) = find(f, dec.target());
                assert_eq!(got, s, "round {round}");
                dec.consume(cum, f[got]);
            }
            assert!(dec.finish(), "round {round}: payload not consumed exactly");
        }
    }

    #[test]
    fn extreme_probabilities() {
        // Two symbols: one with the minimum frequency, one with the rest.
        let f = [1u32, TOTAL - 1];
        for pattern in 0..4u32 {
            let mut enc = RangeEncoder::new();
            let syms: Vec<usize> = (0..50_000u32)
                .map(|i| match pattern {
                    0 => 0,
                    1 => 1,
                    2 => (i % 2) as usize,
                    _ => usize::from(i % 997 == 0),
                })
                .collect();
            for &s in &syms {
                enc.encode(cum_of(&f, s), f[s]);
            }
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for &s in &syms {
                let (got, cum) = find(&f, dec.target());
                assert_eq!(got, s);
                dec.consume(cum, f[got]);
            }
            assert!(dec.finish());
        }
    }

    #[test]
    fn carry_propagation_through_ff_runs() {
        // Always coding the last symbol of a table keeps `low` hugging the
        // top of the interval, which produces long runs of pending 0xFF
        // bytes and exercises the carry path.
        let mut rng = Rng(42);
        let mut enc = RangeEncoder::new();
        let mut log = Vec::new();
        for i in 0..200_000u32 {
            let f = if i % 3 == 0 {
                vec![TOTAL - 1, 1]
            } else {
                let tail = 1 + rng.below(5);
                vec![TOTAL - 1 - tail, 1, tail]
            };
            let s = if rng.below(50) == 0 { 0 } else { f.len() - 1 };
            enc.encode(cum_of(&f, s), f[s]);
            log.push((f, s));
        }
        let bytes = enc.finish();
        assert!(bytes.windows(3).any(|w| w == [0xFF, 0xFF, 0xFF]));
        let mut dec = RangeDecoder::new(&bytes).unwrap();
        for (f, s) in &log {
            let (got, cum) = find(f, dec.target());
            assert_eq!(got, *s);
            dec.consume(cum, f[got]);
        }
        assert!(dec.finish());
    }

    #[test]
    fn binary_roundtrip() {
        let mut rng = Rng(0xB17);
        for round in 0..200 {
            let len = rng.below(5000) as usize;
            let mode = rng.below(4);
            let mut enc = RangeEncoder::new();
            let mut log = Vec::with_capacity(len);
            for _ in 0..len {
                let p1 = match mode {
                    0 => 1 + rng.below(TOTAL - 1),
                    1 => 1 + rng.below(8),
                    2 => TOTAL - 1 - rng.below(8),
                    _ => TOTAL / 2,
                };
                // Mix likely and unlikely outcomes.
                let bit = match rng.below(3) {
                    0 => rng.below(TOTAL) < p1,
                    1 => true,
                    _ => false,
                };
                enc.encode_bit(bit, p1);
                log.push((bit, p1));
            }
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for &(bit, p1) in &log {
                assert_eq!(dec.decode_bit(p1), bit, "round {round}");
            }
            assert!(dec.finish(), "round {round}");
        }
    }

    #[test]
    fn binary_coding_is_efficient() {
        // 100k bits that are 1 with probability 1/16, coded with that
        // probability: the entropy is about 0.337 bits each.
        let mut rng = Rng(5);
        let mut enc = RangeEncoder::new();
        for _ in 0..100_000 {
            enc.encode_bit(rng.below(16) == 0, TOTAL / 16);
        }
        let bits = enc.finish().len() as f64 * 8.0;
        assert!((32_000.0..35_500.0).contains(&bits), "{bits} bits");
    }

    #[test]
    fn binary_decoder_never_panics_on_garbage() {
        let mut rng = Rng(11);
        for _ in 0..2000 {
            let len = 5 + rng.below(100) as usize;
            let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            bytes[0] = 0;
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for _ in 0..2000 {
                dec.decode_bit(1 + rng.below(TOTAL - 1));
            }
        }
    }

    #[test]
    fn decoder_rejects_bad_framing() {
        assert!(RangeDecoder::new(&[]).is_none());
        assert!(RangeDecoder::new(&[0, 0, 0, 0]).is_none());
        assert!(RangeDecoder::new(&[1, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn decoder_never_panics_on_garbage() {
        let mut rng = Rng(7);
        for _ in 0..2000 {
            let len = 5 + rng.below(200) as usize;
            let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            bytes[0] = 0;
            let mut dec = RangeDecoder::new(&bytes).unwrap();
            for _ in 0..400 {
                let f = random_table(&mut rng, 256);
                let (s, cum) = find(&f, dec.target());
                dec.consume(cum, f[s]);
            }
        }
    }

    #[test]
    fn length_mismatch_is_detected() {
        let f = [TOTAL / 2, TOTAL / 2];
        let mut enc = RangeEncoder::new();
        for i in 0..1000usize {
            let s = (i * 7 % 3) & 1;
            enc.encode(cum_of(&f, s), f[s]);
        }
        let bytes = enc.finish();
        let run = |payload: &[u8]| {
            let mut dec = RangeDecoder::new(payload).unwrap();
            for _ in 0..1000 {
                let (s, cum) = find(&f, dec.target());
                dec.consume(cum, f[s]);
            }
            dec.finish()
        };
        assert!(run(&bytes));
        assert!(!run(&bytes[..bytes.len() - 1]));
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(!run(&longer));
    }
}
