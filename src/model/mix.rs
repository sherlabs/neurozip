//! Bit-level context-mixing model with a match model ("mix").
//!
//! Each byte is coded as eight binary decisions, most significant bit
//! first. For every bit a set of simple predictors each give a probability
//! that the bit is 1; an online-trained two-layer linear mixer combines
//! them in the logistic domain; two adaptive correction tables refine the
//! result. This is the architecture of the PAQ / lpaq family (see
//! `docs/PRIOR_ART.md`); the code here is an independent implementation.
//!
//! Predictors (the mixer's inputs):
//!
//! * an order-0 counter;
//! * for order 1 and ten hashed contexts (orders 2, 3, 4, 5, 6 and 8, the
//!   current word, the current and previous word, and two sparse contexts):
//!   a counter, and a prediction from the context's recent bit history;
//! * a match model: the longest recent repeat of the current context
//!   predicts the byte that followed it last time;
//! * the match model's predicted byte used as a context;
//! * a constant bias.
//!
//! Everything is integer fixed point. There are no stored weights: the
//! mixer starts from a fixed initial state and learns from the data being
//! coded, identically in encoder and decoder.
//!
//! # Fixed-point conventions
//!
//! * probabilities are 12 bits (`0..4096` means `0..1`);
//! * `stretch(p) = ln(p / (1 - p))` and its inverse `squash` use 8
//!   fractional bits, clamped to `-2047..=2047`;
//! * mixer weights use 16 fractional bits;
//! * a plain counter is a `u32`: a 22-bit probability above a 10-bit count;
//! * a context node is a `u32`: 16-bit probability, 4-bit count and a
//!   12-bit history of the most recent bits seen in that context.

use sha2::{Digest, Sha256};

/// Bits of probability the model outputs.
pub const PROB_BITS: u32 = 12;

/// Feature switches, stored in the archive header. Used for ablations.
pub const FLAG_NO_MATCH: u8 = 1 << 0;
pub const FLAG_NO_APM: u8 = 1 << 1;
pub const FLAG_NO_WORD: u8 = 1 << 2;
pub const FLAG_FROZEN_MIXER: u8 = 1 << 3;
/// Do not apply the x86 filter to blocks that look like machine code.
pub const FLAG_NO_X86: u8 = 1 << 4;
pub const KNOWN_FLAGS: u8 =
    FLAG_NO_MATCH | FLAG_NO_APM | FLAG_NO_WORD | FLAG_FROZEN_MIXER | FLAG_NO_X86;

pub const MIN_HASH_BITS: u8 = 10;
pub const MAX_HASH_BITS: u8 = 24;
pub const MIN_MATCH_BITS: u8 = 12;
pub const MAX_MATCH_BITS: u8 = 28;

/// Hashed context tables: orders 2, 3, 4, 6 and the word context.
const N_HASHED: usize = 10;
const WORD_TABLE: usize = 4;
/// Contexts that keep a bit history as well as a counter: o1 and the hashed.
const N_IND: usize = 1 + N_HASHED;
/// Mixer inputs: o0; a counter and a history prediction for each of o1 and
/// the five hashed contexts; match; match-byte context; bias.
const N_INPUTS: usize = 4 + 2 * N_IND;
const IN_MATCH: usize = 1 + 2 * N_IND;
const IN_MATCH_BYTE: usize = 2 + 2 * N_IND;
const IN_BIAS: usize = 3 + 2 * N_IND;
/// First-layer mixers, each with its own weight-set selector.
const N_MIXERS: usize = 3;
/// Weight sets: 256 by partial byte, 32 by match length, 256 by last byte.
const SEL_BASE: [usize; N_MIXERS] = [0, 256, 288];
const N_SETS: usize = 544;

/// A counter stops slowing its adaptation after this many updates.
const COUNT_LIMIT: u32 = 24;
/// Counter with probability one half and no observations.
const NODE_INIT: u32 = 1 << 31;
/// Context node: 16-bit probability, 4-bit count, 12-bit bit history. The
/// history is the most recent bits seen in that context behind a leading 1.
const CTX_INIT: u32 = (1 << 31) | 1;
const CTX_LIMIT: u32 = 15;
const HIST_SIZE: usize = 4096;
/// History maps are close to stationary, so they adapt slowly.
const MAP_LIMIT: u32 = 255;
/// Mixer learning rates (see `train_weights`).
const LR_LAYER1: i32 = 16;
const LR_LAYER2: i32 = 8;
/// Initial first-layer weight (0.3) and second-layer weight (1/3).
const W1_INIT: i32 = 9_830;
const W2_INIT: i32 = 21_845;
const W_LIMIT: i32 = 1 << 24;
/// The match model needs this many bytes of agreeing context.
const MATCH_MIN: u64 = 6;
/// How far back a new match is verified; it grows byte by byte after that.
const MATCH_VERIFY: u32 = 32;
const MATCH_MAX: u32 = 65_535;
const APM_RATE: u32 = 6;
const APM_CONTEXTS: usize = 65_536;

/// `squash` sampled every 0.5 from -8 to 8: `round(4096 / (1 + e^-x))`.
const SQUASH_TABLE: [i32; 33] = [
    1, 2, 4, 6, 10, 17, 27, 45, 74, 120, 194, 311, 488, 747, 1102, 1546, 2048, 2550, 2994, 3349,
    3608, 3785, 3902, 3976, 4022, 4051, 4069, 4079, 4086, 4090, 4092, 4094, 4095,
];

/// Logistic function: stretch-domain value to 12-bit probability.
#[inline]
const fn squash(d: i32) -> i32 {
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 0;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (SQUASH_TABLE[i] * (128 - w) + SQUASH_TABLE[i + 1] * w + 64) >> 7
}

/// Inverse of `squash`, tabulated for every 12-bit probability.
const STRETCH: [i16; 4096] = {
    let mut t = [0i16; 4096];
    let mut pi = 0usize;
    let mut x = -2047;
    while x <= 2047 {
        let v = squash(x) as usize;
        while pi <= v {
            t[pi] = x as i16;
            pi += 1;
        }
        x += 1;
    }
    while pi < 4096 {
        t[pi] = 2047;
        pi += 1;
    }
    t
};

#[inline]
fn stretch(p: i32) -> i32 {
    STRETCH[p as usize] as i32
}

/// Adaptation rate after `n` updates: `2^16 / (n + 1.5)`.
const RATE: [u32; 1024] = {
    let mut t = [0u32; 1024];
    let mut n = 0;
    while n < 1024 {
        t[n] = (131_072 / (2 * n + 3)) as u32;
        n += 1;
    }
    t
};

/// Move a counter toward `bit`, quickly at first and more slowly later.
#[inline]
fn train(node: &mut u32, bit: u32, limit: u32) {
    let n = *node & 1023;
    let p = (*node >> 10) as i64;
    let target = (bit as i64) << 22;
    let p = (p + (((target - p) * RATE[n as usize] as i64) >> 16)).clamp(0, (1 << 22) - 1);
    let n = if n < limit { n + 1 } else { n };
    *node = ((p as u32) << 10) | n;
}

/// Update a context node: its counter, and its bit history.
#[inline]
fn ctx_train(node: &mut u32, bit: u32) {
    let n = (*node >> 12) & 15;
    let p = (*node >> 16) as i64;
    let target = (bit as i64) << 16;
    let p = (p + (((target - p) * RATE[n as usize] as i64) >> 16)).clamp(0, 65_535);
    let n = if n < CTX_LIMIT { n + 1 } else { n };
    let h = *node & 0xFFF;
    let h = if h >= 0x800 {
        0x800 | (((h << 1) | bit) & 0x7FF)
    } else {
        (h << 1) | bit
    };
    *node = ((p as u32) << 16) | (n << 12) | h;
}

/// Mixer input from a context node's counter.
#[inline]
fn ctx_input(node: u32) -> i32 {
    if (node >> 12) & 15 == 0 {
        0
    } else {
        stretch((node >> 20) as i32)
    }
}

/// Mixer input from a counter: nothing until it has seen data.
#[inline]
fn input(node: u32) -> i32 {
    if node & 1023 == 0 {
        0
    } else {
        stretch((node >> 20) as i32)
    }
}

#[inline]
fn hash64(x: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 29)
}

/// Fifteen counters for one context and one nibble: the binary tree of
/// the four bit decisions. One cache line.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Bucket {
    tag: u32,
    nodes: [u32; 15],
}

const _: () = assert!(std::mem::size_of::<Bucket>() == 64);

impl Bucket {
    const EMPTY: Bucket = Bucket {
        tag: 0,
        nodes: [CTX_INIT; 15],
    };
}

/// Find the bucket for `key`, claiming one if it is not present. Two
/// neighbouring buckets are candidates; the less used one is replaced.
#[inline]
fn find(table: &mut [Bucket], bits: u32, key: u64) -> usize {
    let i = (key >> (64 - bits)) as usize;
    let tag = key as u32;
    if table[i].tag == tag {
        return i;
    }
    let j = i ^ 1;
    if table[j].tag == tag {
        return j;
    }
    let victim = if ((table[j].nodes[0] >> 12) & 15) < ((table[i].nodes[0] >> 12) & 15) {
        j
    } else {
        i
    };
    table[victim] = Bucket {
        tag,
        nodes: [CTX_INIT; 15],
    };
    victim
}

/// Construction parameters; stored in the archive header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixParams {
    /// log2 of the number of buckets in each hashed context table.
    pub hash_bits: u8,
    /// log2 of the match model's history window in bytes.
    pub match_bits: u8,
    /// Feature switches (`FLAG_*`); zero for the full model.
    pub flags: u8,
}

impl Default for MixParams {
    fn default() -> Self {
        MixParams {
            hash_bits: 18,
            match_bits: 22,
            flags: 0,
        }
    }
}

impl MixParams {
    pub fn is_valid(&self) -> bool {
        (MIN_HASH_BITS..=MAX_HASH_BITS).contains(&self.hash_bits)
            && (MIN_MATCH_BITS..=MAX_MATCH_BITS).contains(&self.match_bits)
            && self.flags & !KNOWN_FLAGS == 0
    }

    /// Heap memory the model allocates, in bytes.
    pub fn memory_bytes(&self) -> u64 {
        let tables = (N_HASHED as u64 * 64) << self.hash_bits;
        let history = 1u64 << self.match_bits;
        let match_table = 4u64 << (self.match_bits - 2);
        let fixed = (256 + 65_536 + 2 * 65_536 + 8 * N_IND as u64 * HIST_SIZE as u64) * 4 // counters, history maps
            + (N_SETS * N_INPUTS + 256 * N_MIXERS) as u64 * 4 // mixer weights
            + 2 * (APM_CONTEXTS as u64) * 33 * 2; // correction tables
        tables + history + match_table + fixed
    }
}

/// Adaptive correction table: maps (context, probability) to a refined
/// probability, interpolating between 33 points along the stretch axis.
struct Apm {
    t: Vec<u16>,
    idx: usize,
}

impl Apm {
    fn new() -> Self {
        let mut t = vec![0u16; APM_CONTEXTS * 33];
        for (i, v) in t.iter_mut().enumerate() {
            let j = (i % 33) as i32;
            *v = (squash((j - 16) * 128) * 16) as u16;
        }
        Apm { t, idx: 0 }
    }

    /// Refine 12-bit probability `p` in context `cx`.
    #[inline]
    fn refine(&mut self, s: i32, cx: usize) -> i32 {
        let lo = (s >> 7) as usize;
        let w = s & 127;
        let base = cx * 33 + lo;
        self.idx = base + (w >> 6) as usize;
        (self.t[base] as i32 * (128 - w) + self.t[base + 1] as i32 * w) >> 11
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let v = self.t[self.idx] as i32;
        let target = (bit as i32) << 16;
        self.t[self.idx] = (v + ((target - v) >> APM_RATE)).clamp(0, 65_535) as u16;
    }
}

/// Everything computed by a prediction that training needs afterwards.
#[derive(Clone, Copy, Default)]
struct Pending {
    valid: bool,
    x: [i32; N_INPUTS],
    sets: [usize; N_MIXERS],
    dots: [i32; N_MIXERS],
    probs: [i32; N_MIXERS],
    p_mix: i32,
    p_final: i32,
    /// Counter index within the current buckets (0..15).
    node: usize,
    match_live: bool,
    match_expect: u32,
    match_bucket: usize,
    match_byte_idx: usize,
    /// History-map entry consulted for each history-keeping context.
    ind_idx: [usize; N_IND],
}

/// The bit-level mixing model. See the module documentation.
pub struct MixModel {
    params: MixParams,
    /// Last eight bytes, most recent lowest.
    hist: u64,
    /// Bits of the current byte seen so far, with a leading 1.
    c0: u32,
    bitpos: u32,
    word: u64,
    /// Hash of the previous word, for the two-word context.
    prev_word: u64,

    o0: Vec<u32>,
    o1: Vec<u32>,
    /// History maps: for each context kind, bit history -> probability.
    ind: Vec<u32>,
    tables: Vec<Vec<Bucket>>,
    keys: [u64; N_HASHED],
    slots: [usize; N_HASHED],

    history: Vec<u8>,
    pos: u32,
    match_ptr: u32,
    match_len: u32,
    match_table: Vec<u32>,
    match_counters: [u32; 32],
    match_byte: Vec<u32>,

    w1: Vec<i32>,
    w2: Vec<i32>,
    apm1: Apm,
    apm2: Apm,

    pending: Pending,
}

impl MixModel {
    /// # Panics
    /// Panics if `params` is not valid; callers validate first.
    pub fn new(params: MixParams) -> Self {
        assert!(params.is_valid(), "invalid MixParams");
        let mut m = MixModel {
            params,
            hist: 0,
            c0: 1,
            bitpos: 0,
            word: 0,
            prev_word: 0,
            o0: vec![NODE_INIT; 256],
            o1: vec![CTX_INIT; 65_536],
            ind: vec![NODE_INIT; 8 * N_IND * HIST_SIZE],
            tables: (0..N_HASHED)
                .map(|_| vec![Bucket::EMPTY; 1usize << params.hash_bits])
                .collect(),
            keys: [0; N_HASHED],
            slots: [0; N_HASHED],
            history: vec![0u8; 1usize << params.match_bits],
            pos: 0,
            match_ptr: 0,
            match_len: 0,
            match_table: vec![0u32; 1usize << (params.match_bits - 2)],
            match_counters: [NODE_INIT; 32],
            match_byte: vec![NODE_INIT; 2 * 65_536],
            w1: vec![W1_INIT; N_SETS * N_INPUTS],
            w2: vec![W2_INIT; 256 * N_MIXERS],
            apm1: Apm::new(),
            apm2: Apm::new(),
            pending: Pending::default(),
        };
        m.start_byte();
        m
    }

    pub fn params(&self) -> MixParams {
        self.params
    }

    /// Return to the defined initial state.
    pub fn reset(&mut self) {
        *self = MixModel::new(self.params);
    }

    /// Probability that the next bit is 1, as a 12-bit value in `1..=4095`.
    ///
    /// Calling this, or not, never changes later predictions.
    #[inline]
    pub fn p1(&mut self) -> u32 {
        if !self.pending.valid {
            self.compute();
        }
        self.pending.p_final as u32
    }

    /// Learn from the bit that actually occurred and move on.
    #[inline]
    pub fn update(&mut self, bit: u32) {
        if !self.pending.valid {
            self.compute();
        }
        self.learn(bit);
        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        if self.bitpos == 4 {
            let bits = self.params.hash_bits as u32;
            for k in 0..N_HASHED {
                let key =
                    hash64(self.keys[k] ^ (self.c0 as u64).wrapping_mul(0xA24B_AED4_963E_E407));
                self.slots[k] = find(&mut self.tables[k], bits, key);
            }
        } else if self.bitpos == 8 {
            self.end_byte((self.c0 & 0xFF) as u8);
        }
        self.pending.valid = false;
    }

    /// Learn from a whole byte without asking for predictions.
    pub fn update_byte(&mut self, byte: u8) {
        for i in (0..8).rev() {
            self.update(((byte >> i) & 1) as u32);
        }
    }

    fn match_bucket(len: u32) -> usize {
        if len < 16 {
            len as usize
        } else {
            (12 + len.ilog2() as usize).min(31)
        }
    }

    fn compute(&mut self) {
        let flags = self.params.flags;
        let c0 = self.c0 as usize;
        let c1 = (self.hist & 0xFF) as usize;
        let mut pd = Pending {
            valid: true,
            ..Pending::default()
        };

        // Position in the nibble's decision tree: 0 for the first bit, then
        // 1..=2, 3..=6, 7..=14.
        let nb = self.bitpos & 3;
        pd.node = ((1usize << nb) | (c0 & ((1 << nb) - 1))) - 1;

        pd.x[0] = input(self.o0[c0]);
        let node1 = self.o1[(c1 << 8) | c0];
        pd.x[1] = ctx_input(node1);
        let bp = self.bitpos as usize * N_IND * HIST_SIZE;
        pd.ind_idx[0] = bp + (node1 & 0xFFF) as usize;
        pd.x[2] = stretch((self.ind[pd.ind_idx[0]] >> 20) as i32);
        for k in 0..N_HASHED {
            let node = self.tables[k][self.slots[k]].nodes[pd.node];
            pd.ind_idx[k + 1] = bp + (k + 1) * HIST_SIZE + (node & 0xFFF) as usize;
            if k == WORD_TABLE && flags & FLAG_NO_WORD != 0 {
                continue;
            }
            pd.x[3 + 2 * k] = ctx_input(node);
            pd.x[4 + 2 * k] = stretch((self.ind[pd.ind_idx[k + 1]] >> 20) as i32);
        }

        if self.match_len > 0 && flags & FLAG_NO_MATCH == 0 {
            let predicted = self.history[self.match_ptr as usize] as u32;
            // The match stays live while the bits of this byte agree.
            if (predicted | 0x100) >> (8 - self.bitpos) == self.c0 {
                pd.match_live = true;
                pd.match_expect = (predicted >> (7 - self.bitpos)) & 1;
                pd.match_bucket = Self::match_bucket(self.match_len);
                let st = stretch((self.match_counters[pd.match_bucket] >> 20) as i32);
                pd.x[IN_MATCH] = if pd.match_expect == 1 { st } else { -st };
                let long = usize::from(self.match_len >= 16);
                pd.match_byte_idx = (long << 16) | ((predicted as usize) << 8) | c0;
                pd.x[IN_MATCH_BYTE] = input(self.match_byte[pd.match_byte_idx]);
            }
        }
        pd.x[IN_BIAS] = 256;

        pd.sets = [
            SEL_BASE[0] + c0,
            SEL_BASE[1] + pd.match_bucket,
            SEL_BASE[2] + c1,
        ];
        let mut dot2 = 0i64;
        for j in 0..N_MIXERS {
            let w = &self.w1[pd.sets[j] * N_INPUTS..(pd.sets[j] + 1) * N_INPUTS];
            let mut dot = 0i64;
            for i in 0..N_INPUTS {
                dot += pd.x[i] as i64 * w[i] as i64;
            }
            pd.dots[j] = (dot >> 16).clamp(-2047, 2047) as i32;
            pd.probs[j] = squash(pd.dots[j]);
            dot2 += pd.dots[j] as i64 * self.w2[c0 * N_MIXERS + j] as i64;
        }
        pd.p_mix = squash((dot2 >> 16).clamp(-2047, 2047) as i32).clamp(1, 4095);

        pd.p_final = if flags & FLAG_NO_APM != 0 {
            pd.p_mix
        } else {
            let s = stretch(pd.p_mix) + 2048;
            let a1 = self.apm1.refine(s, (c1 << 8) | c0);
            let cx2 = (hash64((self.hist & 0xFF_FFFF) ^ ((c0 as u64) << 24)) >> 48) as usize;
            let a2 = self.apm2.refine(s, cx2);
            ((pd.p_mix + a1 + 2 * a2 + 2) >> 2).clamp(1, 4095)
        };
        self.pending = pd;
    }

    fn learn(&mut self, bit: u32) {
        let pd = self.pending;
        let flags = self.params.flags;
        let c0 = self.c0 as usize;
        let c1 = (self.hist & 0xFF) as usize;
        let target = (bit as i32) << 12;

        if flags & FLAG_FROZEN_MIXER == 0 {
            let err2 = (target - pd.p_mix) * LR_LAYER2;
            for j in 0..N_MIXERS {
                let w = &mut self.w2[c0 * N_MIXERS + j];
                *w = (*w + ((pd.dots[j] * err2 + 0x8000) >> 16)).clamp(-W_LIMIT, W_LIMIT);

                let err = (target - pd.probs[j]) * LR_LAYER1;
                let ws = &mut self.w1[pd.sets[j] * N_INPUTS..(pd.sets[j] + 1) * N_INPUTS];
                for i in 0..N_INPUTS {
                    ws[i] = (ws[i] + ((pd.x[i] * err + 0x8000) >> 16)).clamp(-W_LIMIT, W_LIMIT);
                }
            }
        }
        if flags & FLAG_NO_APM == 0 {
            self.apm1.update(bit);
            self.apm2.update(bit);
        }

        train(&mut self.o0[c0], bit, COUNT_LIMIT);
        for idx in pd.ind_idx {
            train(&mut self.ind[idx], bit, MAP_LIMIT);
        }
        ctx_train(&mut self.o1[(c1 << 8) | c0], bit);
        for k in 0..N_HASHED {
            ctx_train(&mut self.tables[k][self.slots[k]].nodes[pd.node], bit);
        }
        if pd.match_live {
            train(
                &mut self.match_counters[pd.match_bucket],
                u32::from(bit == pd.match_expect),
                COUNT_LIMIT,
            );
            train(&mut self.match_byte[pd.match_byte_idx], bit, COUNT_LIMIT);
        }
    }

    fn end_byte(&mut self, byte: u8) {
        let mask = (self.history.len() - 1) as u32;

        // Extend or drop the current match, then record this position.
        if self.match_len > 0 && self.history[self.match_ptr as usize] == byte {
            self.match_len = (self.match_len + 1).min(MATCH_MAX);
            self.match_ptr = (self.match_ptr + 1) & mask;
        } else {
            self.match_len = 0;
        }
        self.history[(self.pos & mask) as usize] = byte;
        self.pos = self.pos.wrapping_add(1);
        self.hist = (self.hist << 8) | byte as u64;

        let slot = (hash64(self.hist & ((1 << (8 * MATCH_MIN)) - 1))
            >> (64 - (self.params.match_bits as u32 - 2))) as usize;
        if self.match_len == 0 {
            let cand = self.match_table[slot];
            let dist = self.pos.wrapping_sub(cand);
            if cand != 0 && dist != 0 && dist <= mask {
                let mut len = 0u32;
                while len < MATCH_VERIFY
                    && len < cand
                    && self.history[(cand.wrapping_sub(1 + len) & mask) as usize]
                        == self.history[(self.pos.wrapping_sub(1 + len) & mask) as usize]
                {
                    len += 1;
                }
                // A short agreement is a hash collision, not a match.
                if len as u64 >= MATCH_MIN {
                    self.match_len = len;
                    self.match_ptr = cand & mask;
                }
            }
        }
        self.match_table[slot] = self.pos;

        let lower = byte.to_ascii_lowercase();
        let was = self.word;
        self.word = if lower.is_ascii_lowercase() || byte >= 0x80 {
            (self.word.wrapping_add(lower as u64 + 1)).wrapping_mul(0x2F0B_3A55_D1A9_C6E1)
        } else {
            0
        };
        if self.word == 0 && was != 0 {
            self.prev_word = was;
        }

        self.c0 = 1;
        self.bitpos = 0;
        self.start_byte();
    }

    /// Compute this byte's context keys and claim their first-nibble buckets.
    fn start_byte(&mut self) {
        let h = self.hist;
        let word_ctx = if self.word != 0 {
            self.word
        } else {
            // Between words, fall back to the last two bytes.
            (h & 0xFFFF) | (1 << 40)
        };
        self.keys = [
            hash64((h & 0xFFFF) | (2 << 56)),
            hash64((h & 0xFF_FFFF) | (3 << 56)),
            hash64((h & 0xFFFF_FFFF) | (4 << 56)),
            hash64((h & 0xFFFF_FFFF_FFFF) | (6 << 56)),
            hash64(word_ctx ^ (7 << 56)),
            hash64((h & 0xFF_FFFF_FFFF) | (5 << 56)),
            // Sparse: the two bytes before the last one.
            hash64((h & 0xFF_FF00) | (8 << 56)),
            // The current word together with the one before it.
            hash64(word_ctx.wrapping_mul(0x9E37_79B9) ^ self.prev_word ^ (9 << 56)),
            // Sparse: the last byte and the one three back.
            hash64((h & 0xFF_00FF) | (10 << 56)),
            hash64(h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (11 << 56)),
        ];
        let bits = self.params.hash_bits as u32;
        for k in 0..N_HASHED {
            self.slots[k] = find(&mut self.tables[k], bits, self.keys[k]);
        }
    }

    /// SHA-256 over the complete model state, for tests.
    pub fn state_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        let mut put = |v: &[u32]| {
            for x in v {
                h.update(x.to_le_bytes());
            }
        };
        put(&[
            self.params.hash_bits as u32,
            self.params.match_bits as u32,
            self.params.flags as u32,
            self.c0,
            self.bitpos,
            self.pos,
            self.match_ptr,
            self.match_len,
        ]);
        put(&self.o0);
        put(&self.o1);
        put(&self.ind);
        put(&self.match_table);
        put(&self.match_counters);
        put(&self.match_byte);
        for t in &self.tables {
            for b in t {
                put(&[b.tag]);
                put(&b.nodes);
            }
        }
        for w in self.w1.iter().chain(&self.w2) {
            h.update(w.to_le_bytes());
        }
        for v in self.apm1.t.iter().chain(&self.apm2.t) {
            h.update(v.to_le_bytes());
        }
        h.update(self.hist.to_le_bytes());
        h.update(self.word.to_le_bytes());
        h.update(self.prev_word.to_le_bytes());
        for k in self.keys {
            h.update(k.to_le_bytes());
        }
        for s in self.slots {
            h.update((s as u64).to_le_bytes());
        }
        h.update(&self.history);
        h.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(flags: u8) -> MixParams {
        MixParams {
            hash_bits: 10,
            match_bits: 12,
            flags,
        }
    }

    fn sample(len: usize, seed: u64) -> Vec<u8> {
        let words: [&[u8]; 6] = [
            b"alpha ",
            b"beta ",
            b"gamma ",
            b"delta\n",
            b"{\"k\": 1}",
            b"x",
        ];
        let mut s = seed;
        let mut out = Vec::new();
        while out.len() < len {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as usize;
            if r.is_multiple_of(9) {
                out.push((r >> 8) as u8);
            } else {
                out.extend_from_slice(words[r % words.len()]);
            }
        }
        out.truncate(len);
        out
    }

    /// Ideal code length in bits of `data` under the model.
    fn cost_bits(m: &mut MixModel, data: &[u8]) -> f64 {
        let mut total = 0.0;
        for &byte in data {
            for i in (0..8).rev() {
                let bit = ((byte >> i) & 1) as u32;
                let p1 = m.p1() as f64 / 4096.0;
                total -= if bit == 1 { p1 } else { 1.0 - p1 }.log2();
                m.update(bit);
            }
        }
        total
    }

    #[test]
    fn squash_and_stretch_are_consistent() {
        assert_eq!(squash(0), 2048);
        assert_eq!(squash(-3000), 0);
        assert_eq!(squash(3000), 4095);
        let mut last = -1;
        for d in -2047..=2047 {
            let p = squash(d);
            assert!((0..=4095).contains(&p) && p >= last);
            last = p;
        }
        for p in 0..4096 {
            let s = stretch(p);
            assert!((-2047..=2047).contains(&s));
            // stretch is a right inverse up to the table's resolution.
            assert!((squash(s) - p).abs() <= 16, "p={p} s={s}");
        }
    }

    #[test]
    fn probabilities_are_always_in_range() {
        for flags in 0..=KNOWN_FLAGS {
            let mut m = MixModel::new(small(flags));
            for &byte in &sample(8000, flags as u64) {
                for i in (0..8).rev() {
                    let p = m.p1();
                    assert!((1..=4095).contains(&p));
                    m.update(((byte >> i) & 1) as u32);
                }
            }
        }
    }

    #[test]
    fn two_instances_stay_identical() {
        let data = sample(20_000, 3);
        let mut a = MixModel::new(small(0));
        let mut b = MixModel::new(small(0));
        for &byte in &data {
            for i in (0..8).rev() {
                assert_eq!(a.p1(), b.p1());
                let bit = ((byte >> i) & 1) as u32;
                a.update(bit);
                b.update(bit);
            }
        }
        assert_eq!(a.state_digest(), b.state_digest());
    }

    #[test]
    fn predicting_does_not_change_state() {
        let data = sample(20_000, 4);
        let mut with = MixModel::new(small(0));
        let mut without = MixModel::new(small(0));
        for &byte in &data {
            for i in (0..8).rev() {
                with.p1();
                with.p1();
                with.update(((byte >> i) & 1) as u32);
            }
            without.update_byte(byte);
        }
        assert_eq!(with.state_digest(), without.state_digest());
    }

    #[test]
    fn reset_restores_initial_state() {
        let fresh = MixModel::new(small(0)).state_digest();
        let mut m = MixModel::new(small(0));
        for &b in &sample(5000, 5) {
            m.update_byte(b);
        }
        assert_ne!(m.state_digest(), fresh);
        m.reset();
        assert_eq!(m.state_digest(), fresh);
    }

    #[test]
    fn learns_structure() {
        let data = sample(40_000, 6);
        let bits = cost_bits(&mut MixModel::new(small(0)), &data);
        assert!(
            bits / (data.len() as f64) < 2.5,
            "{} bpb",
            bits / data.len() as f64
        );
    }

    #[test]
    fn match_model_helps_on_long_repeats() {
        // Incompressible noise, repeated: only the match model can see it.
        let mut s = 99u64;
        let noise: Vec<u8> = (0..3000)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (s >> 40) as u8
            })
            .collect();
        let data = [noise.clone(), noise.clone(), noise].concat();
        let with = cost_bits(&mut MixModel::new(small(0)), &data);
        let without = cost_bits(&mut MixModel::new(small(FLAG_NO_MATCH)), &data);
        assert!(with < without * 0.7, "with {with}, without {without}");
    }

    #[test]
    fn survives_window_wraparound() {
        // History window of 4096 bytes, far more data than that.
        let mut m = MixModel::new(small(0));
        let data = sample(60_000, 8);
        let bits = cost_bits(&mut m, &data);
        assert!(bits.is_finite());
    }
}
