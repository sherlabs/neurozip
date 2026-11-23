//! Adaptive byte-level context model, orders 0 through `max_order`.
//!
//! This is the non-neural baseline. For each order `k` the model keeps
//! counts of which bytes followed the current `k`-byte context, and blends
//! the orders from lowest to highest with interpolated Witten-Bell
//! smoothing (the escape estimate known as method C in the PPM literature):
//!
//! ```text
//! P_k(s) = (c_k(s) + D_k * P_{k-1}(s)) / (T_k + D_k)
//! ```
//!
//! where `c_k(s)` is the count of `s` in the order-`k` context, `T_k` the
//! sum of counts, `D_k` the number of distinct bytes seen there, and
//! `P_{-1}` is uniform. A context with no statistics leaves the estimate
//! unchanged.
//!
//! Everything is fixed-point integer arithmetic with `P` scaled by
//! `2^P_BITS`. The result goes through [`normalize`], which makes it
//! strictly positive and exactly [`TOTAL`](crate::rangecoder::TOTAL).
//!
//! # Storage and memory bound
//!
//! * order 0: 256 dense 16-bit counts;
//! * order 1: 256 x 256 dense 16-bit counts;
//! * each order `k >= 2`: a table of `2^hash_bits` 32-byte slots indexed by
//!   a hash of the context. A slot holds a 16-bit check tag and up to 15
//!   `(byte, count)` pairs. A tag mismatch replaces the slot.
//!
//! Memory is therefore fixed at construction: see [`CtxParams::memory_bytes`].

use super::Predictor;
use crate::freq::{Freqs, SYMBOLS, normalize};
use sha2::{Digest, Sha256};

/// Fixed-point scale of intermediate probabilities.
const P_BITS: u32 = 24;
/// Dense contexts halve their counts when the total reaches this value.
const DENSE_LIMIT: u32 = 1 << 14;
/// Number of `(byte, count)` pairs in a hashed slot.
const SLOT_SYMS: usize = 15;
/// A hashed slot halves its counts when one of them reaches this value.
const SLOT_COUNT_LIMIT: u8 = 255;

/// Highest context order this model supports.
pub const MAX_ORDER: u8 = 8;
/// Smallest and largest permitted `hash_bits`.
pub const MIN_HASH_BITS: u8 = 8;
pub const MAX_HASH_BITS: u8 = 24;

/// Construction parameters; stored in the archive header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxParams {
    /// Highest context order, `0..=MAX_ORDER`.
    pub max_order: u8,
    /// log2 of the number of slots per hashed order.
    pub hash_bits: u8,
}

impl Default for CtxParams {
    fn default() -> Self {
        CtxParams {
            max_order: 4,
            hash_bits: 20,
        }
    }
}

impl CtxParams {
    pub fn is_valid(&self) -> bool {
        self.max_order <= MAX_ORDER && (MIN_HASH_BITS..=MAX_HASH_BITS).contains(&self.hash_bits)
    }

    /// Number of hashed tables (orders 2 and up).
    fn hashed_orders(&self) -> u64 {
        (self.max_order as u64).saturating_sub(1)
    }

    /// Exact heap memory the model allocates, in bytes.
    pub fn memory_bytes(&self) -> u64 {
        let dense1 = if self.max_order >= 1 {
            (SYMBOLS * SYMBOLS * 2) as u64
        } else {
            0
        };
        dense1 + self.hashed_orders() * ((std::mem::size_of::<Slot>() as u64) << self.hash_bits)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    tag: u16,
    syms: [u8; SLOT_SYMS],
    /// Counts; entries are packed, the first zero ends the list.
    cnts: [u8; SLOT_SYMS],
}

const _: () = assert!(std::mem::size_of::<Slot>() == 32);

impl Slot {
    const EMPTY: Slot = Slot {
        tag: 0,
        syms: [0; SLOT_SYMS],
        cnts: [0; SLOT_SYMS],
    };

    #[inline]
    fn matches(&self, tag: u16) -> bool {
        self.cnts[0] != 0 && self.tag == tag
    }

    #[inline]
    fn learn(&mut self, tag: u16, byte: u8) {
        if !self.matches(tag) {
            *self = Slot::EMPTY;
            self.tag = tag;
            self.syms[0] = byte;
            self.cnts[0] = 1;
            return;
        }
        let mut min_i = 0;
        let mut min_c = u8::MAX;
        for i in 0..SLOT_SYMS {
            let c = self.cnts[i];
            if c == 0 {
                self.syms[i] = byte;
                self.cnts[i] = 1;
                return;
            }
            if self.syms[i] == byte {
                self.cnts[i] = c + 1;
                if c + 1 == SLOT_COUNT_LIMIT {
                    for x in self.cnts.iter_mut() {
                        *x = x.div_ceil(2);
                    }
                }
                return;
            }
            if c < min_c {
                min_c = c;
                min_i = i;
            }
        }
        // Slot is full: evict the first entry with the smallest count.
        self.syms[min_i] = byte;
        self.cnts[min_i] = 1;
    }
}

/// Scale `p` by the escape probability `distinct / (total + distinct)` and
/// return the fixed-point weight of one count.
#[inline]
fn escape_scale(p: &mut [u32; SYMBOLS], total: u32, distinct: u32) -> u32 {
    let unit = (1u32 << P_BITS) / (total + distinct);
    let mult = (distinct * unit) as u64;
    for x in p.iter_mut() {
        *x = ((*x as u64 * mult) >> P_BITS) as u32;
    }
    unit
}

/// Blend one dense context into `p`.
#[inline]
fn blend_dense(p: &mut [u32; SYMBOLS], counts: &[u16], total: u32, distinct: u32) {
    if total == 0 {
        return;
    }
    let unit = escape_scale(p, total, distinct);
    for (x, &c) in p.iter_mut().zip(counts) {
        *x += c as u32 * unit;
    }
}

/// Count one occurrence in a dense context, halving when the limit is hit.
#[inline]
fn learn_dense(counts: &mut [u16], total: &mut u32, distinct: &mut u16, byte: u8) {
    let c = &mut counts[byte as usize];
    if *c == 0 {
        *distinct += 1;
    }
    *c += 1;
    *total += 1;
    if *total >= DENSE_LIMIT {
        let mut t = 0u32;
        for x in counts.iter_mut() {
            *x = x.div_ceil(2);
            t += *x as u32;
        }
        *total = t;
    }
}

/// The context-mixing baseline model. See the module documentation.
pub struct CtxModel {
    params: CtxParams,
    /// The last eight bytes, most recent in the low byte. Starts at zero.
    hist: u64,
    o0: [u16; SYMBOLS],
    o0_total: u32,
    o0_distinct: u16,
    /// `o1[prev * 256 + byte]`; empty when `max_order == 0`.
    o1: Vec<u16>,
    o1_total: [u32; SYMBOLS],
    o1_distinct: [u16; SYMBOLS],
    /// `tables[k - 2]` serves order `k`.
    tables: Vec<Vec<Slot>>,
}

impl CtxModel {
    /// # Panics
    /// Panics if `params` is not valid; callers validate first.
    pub fn new(params: CtxParams) -> Self {
        assert!(params.is_valid(), "invalid CtxParams");
        let o1 = if params.max_order >= 1 {
            vec![0u16; SYMBOLS * SYMBOLS]
        } else {
            Vec::new()
        };
        let tables = (0..params.hashed_orders())
            .map(|_| vec![Slot::EMPTY; 1usize << params.hash_bits])
            .collect();
        CtxModel {
            params,
            hist: 0,
            o0: [0; SYMBOLS],
            o0_total: 0,
            o0_distinct: 0,
            o1,
            o1_total: [0; SYMBOLS],
            o1_distinct: [0; SYMBOLS],
            tables,
        }
    }

    pub fn params(&self) -> CtxParams {
        self.params
    }

    /// Slot index and check tag for the current order-`k` context, `k >= 2`.
    #[inline]
    fn slot_pos(&self, k: usize) -> (usize, u16) {
        let ctx = if k >= 8 {
            self.hist
        } else {
            self.hist & ((1u64 << (8 * k)) - 1)
        };
        let mut h = ctx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
        h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let bits = self.params.hash_bits as u32;
        ((h >> (64 - bits)) as usize, (h >> (48 - bits)) as u16)
    }

    /// SHA-256 over the complete model state. For tests and debugging: two
    /// models with equal digests make identical predictions forever.
    pub fn state_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([self.params.max_order, self.params.hash_bits]);
        h.update(self.hist.to_le_bytes());
        let mut put16 = |v: &[u16]| {
            for x in v {
                h.update(x.to_le_bytes());
            }
        };
        put16(&self.o0);
        put16(&[self.o0_distinct]);
        put16(&self.o1);
        put16(&self.o1_distinct);
        h.update(self.o0_total.to_le_bytes());
        for t in &self.o1_total {
            h.update(t.to_le_bytes());
        }
        for table in &self.tables {
            for slot in table {
                h.update(slot.tag.to_le_bytes());
                h.update(slot.syms);
                h.update(slot.cnts);
            }
        }
        h.finalize().into()
    }
}

impl Predictor for CtxModel {
    fn predict(&mut self, out: &mut Freqs) {
        let mut p = [(1u32 << P_BITS) / SYMBOLS as u32; SYMBOLS];

        blend_dense(&mut p, &self.o0, self.o0_total, self.o0_distinct as u32);

        if self.params.max_order >= 1 {
            let c1 = (self.hist & 0xFF) as usize;
            blend_dense(
                &mut p,
                &self.o1[c1 * SYMBOLS..(c1 + 1) * SYMBOLS],
                self.o1_total[c1],
                self.o1_distinct[c1] as u32,
            );
        }

        for k in 2..=self.params.max_order as usize {
            let (idx, tag) = self.slot_pos(k);
            let slot = &self.tables[k - 2][idx];
            if !slot.matches(tag) {
                continue;
            }
            let mut total = 0u32;
            let mut distinct = 0u32;
            for &c in &slot.cnts {
                if c == 0 {
                    break;
                }
                total += c as u32;
                distinct += 1;
            }
            let unit = escape_scale(&mut p, total, distinct);
            for i in 0..distinct as usize {
                p[slot.syms[i] as usize] += slot.cnts[i] as u32 * unit;
            }
        }

        normalize(&p, out);
    }

    fn update(&mut self, byte: u8) {
        learn_dense(
            &mut self.o0,
            &mut self.o0_total,
            &mut self.o0_distinct,
            byte,
        );

        if self.params.max_order >= 1 {
            let c1 = (self.hist & 0xFF) as usize;
            learn_dense(
                &mut self.o1[c1 * SYMBOLS..(c1 + 1) * SYMBOLS],
                &mut self.o1_total[c1],
                &mut self.o1_distinct[c1],
                byte,
            );
        }

        for k in 2..=self.params.max_order as usize {
            let (idx, tag) = self.slot_pos(k);
            self.tables[k - 2][idx].learn(tag, byte);
        }

        self.hist = (self.hist << 8) | byte as u64;
    }

    fn advance(&mut self, byte: u8) {
        self.hist = (self.hist << 8) | byte as u64;
    }

    fn reset(&mut self) {
        self.hist = 0;
        self.o0 = [0; SYMBOLS];
        self.o0_total = 0;
        self.o0_distinct = 0;
        self.o1.fill(0);
        self.o1_total = [0; SYMBOLS];
        self.o1_distinct = [0; SYMBOLS];
        for t in &mut self.tables {
            t.fill(Slot::EMPTY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rangecoder::TOTAL;

    fn sample(len: usize, seed: u64) -> Vec<u8> {
        // Text-like data with repeats plus some noise.
        let words: [&[u8]; 6] = [b"the ", b"quick ", b"brown ", b"fox ", b"jumps ", b"\n"];
        let mut s = seed;
        let mut out = Vec::new();
        while out.len() < len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as usize;
            if r.is_multiple_of(11) {
                out.push((r >> 8) as u8);
            } else {
                out.extend_from_slice(words[r % words.len()]);
            }
        }
        out.truncate(len);
        out
    }

    fn small(order: u8) -> CtxParams {
        CtxParams {
            max_order: order,
            hash_bits: 10,
        }
    }

    #[test]
    fn output_is_always_a_valid_table() {
        for order in 0..=MAX_ORDER {
            let mut m = CtxModel::new(small(order));
            let mut f = [0u16; SYMBOLS];
            for &b in &sample(20_000, order as u64) {
                m.predict(&mut f);
                assert!(f.iter().all(|&x| x >= 1));
                assert_eq!(f.iter().map(|&x| x as u32).sum::<u32>(), TOTAL);
                m.update(b);
            }
        }
    }

    #[test]
    fn two_instances_stay_identical() {
        let data = sample(30_000, 99);
        let mut a = CtxModel::new(small(4));
        let mut b = CtxModel::new(small(4));
        let (mut fa, mut fb) = ([0u16; SYMBOLS], [0u16; SYMBOLS]);
        for &x in &data {
            a.predict(&mut fa);
            b.predict(&mut fb);
            assert_eq!(fa, fb);
            a.update(x);
            b.update(x);
        }
        assert_eq!(a.state_digest(), b.state_digest());
    }

    #[test]
    fn predict_does_not_change_state() {
        let data = sample(30_000, 5);
        let mut with = CtxModel::new(small(5));
        let mut without = CtxModel::new(small(5));
        let mut f = [0u16; SYMBOLS];
        for &x in &data {
            with.predict(&mut f);
            with.predict(&mut f);
            with.update(x);
            without.update(x);
        }
        assert_eq!(with.state_digest(), without.state_digest());
    }

    #[test]
    fn reset_restores_initial_state() {
        let fresh = CtxModel::new(small(4)).state_digest();
        let mut m = CtxModel::new(small(4));
        for &x in &sample(10_000, 1) {
            m.update(x);
        }
        assert_ne!(m.state_digest(), fresh);
        m.reset();
        assert_eq!(m.state_digest(), fresh);
    }

    #[test]
    fn learns_a_repeating_pattern() {
        let mut m = CtxModel::new(small(4));
        let pattern = b"abcdefgh";
        let mut f = [0u16; SYMBOLS];
        for i in 0..4000 {
            m.update(pattern[i % pattern.len()]);
        }
        m.predict(&mut f);
        // After many repetitions the next byte ('a') should be near certain.
        assert!(
            f[b'a' as usize] as u32 > TOTAL * 9 / 10,
            "got {}",
            f[b'a' as usize]
        );
    }

    #[test]
    fn dense_counts_survive_long_runs() {
        // Far more than DENSE_LIMIT and u16::MAX occurrences of one byte.
        let mut m = CtxModel::new(small(2));
        let mut f = [0u16; SYMBOLS];
        for _ in 0..300_000 {
            m.update(0);
        }
        m.predict(&mut f);
        assert!(f[0] as u32 > TOTAL * 9 / 10);
    }

    #[test]
    fn memory_bytes_matches_allocation() {
        let p = CtxParams {
            max_order: 4,
            hash_bits: 12,
        };
        let m = CtxModel::new(p);
        let actual: usize = m.o1.len() * 2 + m.tables.iter().map(|t| t.len() * 32).sum::<usize>();
        assert_eq!(p.memory_bytes(), actual as u64);
        assert_eq!(
            CtxParams {
                max_order: 0,
                hash_bits: 12
            }
            .memory_bytes(),
            0
        );
    }
}
