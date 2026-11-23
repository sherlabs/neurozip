//! Deterministic conversion of non-negative integer weights into a strictly
//! positive frequency table summing to exactly [`TOTAL`].
//!
//! Every predictor, whatever its internals, ends by calling [`normalize`].
//! The rule is:
//!
//! 1. `W = sum(w)`. If `W == 0` every symbol gets `TOTAL / 256`.
//! 2. `scale = floor((TOTAL - 256) * 2^32 / W)`.
//! 3. `f[s] = 1 + floor(w[s] * scale / 2^32)`.
//! 4. The shortfall `TOTAL - sum(f)` (never negative, at most a few hundred)
//!    is added to the symbol with the largest weight; ties go to the lowest
//!    symbol value.
//!
//! Only integer operations are used, so the result is identical everywhere.

use crate::rangecoder::TOTAL;

/// Number of symbols in the byte alphabet.
pub const SYMBOLS: usize = 256;

/// A frequency table for one byte position.
pub type Freqs = [u16; SYMBOLS];

/// See the module documentation for the exact rule.
pub fn normalize(weights: &[u32; SYMBOLS], out: &mut Freqs) {
    let mut sum = 0u64;
    let mut best = 0usize;
    let mut best_w = 0u32;
    for (s, &w) in weights.iter().enumerate() {
        sum += w as u64;
        if w > best_w {
            best_w = w;
            best = s;
        }
    }
    if sum == 0 {
        *out = [(TOTAL / SYMBOLS as u32) as u16; SYMBOLS];
        return;
    }
    let budget = (TOTAL - SYMBOLS as u32) as u64;
    // w <= sum, so w * scale <= budget << 32 < 2^48: no overflow.
    let scale = (budget << 32) / sum;
    let mut used = 0u32;
    for (o, &w) in out.iter_mut().zip(weights.iter()) {
        let f = 1 + ((w as u64 * scale) >> 32) as u32;
        used += f;
        *o = f as u16;
    }
    // used <= 256 + budget == TOTAL, and the largest entry plus the
    // shortfall is at most TOTAL - 255, which fits in u16.
    out[best] += (TOTAL - used) as u16;
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn check(w: &[u32; SYMBOLS]) -> Freqs {
        let mut f = [0u16; SYMBOLS];
        normalize(w, &mut f);
        assert!(f.iter().all(|&x| x >= 1), "zero frequency");
        assert_eq!(f.iter().map(|&x| x as u32).sum::<u32>(), TOTAL);
        f
    }

    #[test]
    fn all_zero_is_uniform() {
        let f = check(&[0; SYMBOLS]);
        assert!(f.iter().all(|&x| x == 256));
    }

    #[test]
    fn single_symbol_gets_almost_everything() {
        let mut w = [0u32; SYMBOLS];
        w[65] = 1;
        let f = check(&w);
        assert_eq!(f[65] as u32, TOTAL - 255);
        w[65] = u32::MAX;
        let f = check(&w);
        assert_eq!(f[65] as u32, TOTAL - 255);
    }

    #[test]
    fn maximum_weights_do_not_overflow() {
        check(&[u32::MAX; SYMBOLS]);
        let mut w = [u32::MAX; SYMBOLS];
        w[0] = 0;
        check(&w);
    }

    #[test]
    fn ties_go_to_lowest_symbol() {
        let mut w = [0u32; SYMBOLS];
        w[10] = 3;
        w[200] = 3;
        w[77] = 1;
        let f = check(&w);
        assert!(f[10] > f[200]);
    }

    proptest! {
        #[test]
        fn sums_to_total_and_is_monotone(
            w in proptest::collection::vec(any::<u32>(), SYMBOLS),
            shift in 0u32..32,
        ) {
            let mut weights = [0u32; SYMBOLS];
            for (d, s) in weights.iter_mut().zip(&w) {
                *d = *s >> shift;
            }
            let f = check(&weights);
            // Except for the one symbol that receives the shortfall, order
            // of weights is preserved.
            let top = (0..SYMBOLS).max_by_key(|&s| (weights[s], std::cmp::Reverse(s))).unwrap();
            for a in 0..SYMBOLS {
                for b in 0..SYMBOLS {
                    if a != top && b != top && weights[a] >= weights[b] {
                        prop_assert!(f[a] >= f[b]);
                    }
                }
                prop_assert!(f[top] >= f[a]);
            }
        }

        #[test]
        fn sparse_weights(
            entries in proptest::collection::vec((0usize..SYMBOLS, 1u32..1_000_000), 0..12),
        ) {
            let mut weights = [0u32; SYMBOLS];
            for (s, w) in entries {
                weights[s] = w;
            }
            check(&weights);
        }
    }
}
