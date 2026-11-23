//! Reversible filter for x86 machine code (the "E8/E9" or BCJ transform).
//!
//! x86 `CALL` (opcode `E8`) and `JMP` (`E9`) instructions carry a 32-bit
//! target relative to the instruction. Calls to one function from many
//! places therefore all look different. Rewriting the target as an absolute
//! address makes them identical, which a context model predicts far better.
//! This is a long-established technique (see `docs/PRIOR_ART.md`).
//!
//! Only targets whose top byte is `00` or `FF` (a small forward or backward
//! displacement) are rewritten, and only their low three bytes change. The
//! encoder works backwards and the decoder forwards, which makes the
//! transform exactly reversible on any input, code or not (see [`encode`]).

/// A block is filtered when at least this fraction (1 / N) of its bytes
/// start a rewritable instruction.
const DENSITY_DIVISOR: usize = 256;

#[inline]
fn candidate(block: &[u8], i: usize) -> bool {
    block[i] & 0xFE == 0xE8 && (block[i + 4] == 0x00 || block[i + 4] == 0xFF)
}

/// Whether the filter should be applied to this block. A deterministic
/// function of the raw bytes, so encoder and decoder agree on stored blocks.
pub fn wanted(block: &[u8]) -> bool {
    if block.len() < 64 {
        return false;
    }
    let mut count = 0usize;
    let mut i = 0;
    while i + 4 < block.len() {
        if candidate(block, i) {
            count += 1;
            i += 5;
        } else {
            i += 1;
        }
    }
    count * DENSITY_DIVISOR >= block.len()
}

#[inline]
fn rewrite(block: &mut [u8], i: usize, offset: u64, forward: bool) {
    let at = (offset.wrapping_add(i as u64) as u32) & 0x00FF_FFFF;
    let v = block[i + 1] as u32 | (block[i + 2] as u32) << 8 | (block[i + 3] as u32) << 16;
    let v = if forward {
        v.wrapping_add(at)
    } else {
        v.wrapping_sub(at)
    } & 0x00FF_FFFF;
    block[i + 1] = v as u8;
    block[i + 2] = (v >> 8) as u8;
    block[i + 3] = (v >> 16) as u8;
}

/// Rewrite relative targets as absolute. `offset` is the block's position
/// in the whole input.
///
/// Positions are visited from the end to the start. A rewrite at `i`
/// changes bytes `i + 1 ..= i + 3`, which other positions inspect, so the
/// order matters: when [`decode`] walks forward and reaches `i`, everything
/// before `i` has been restored and everything after is still rewritten,
/// which is exactly what this function saw when it decided about `i`.
pub fn encode(block: &mut [u8], offset: u64) {
    if block.len() < 5 {
        return;
    }
    for i in (0..block.len() - 4).rev() {
        if candidate(block, i) {
            rewrite(block, i, offset, true);
        }
    }
}

/// Exact inverse of [`encode`] with the same `offset`.
pub fn decode(block: &mut [u8], offset: u64) {
    if block.len() < 5 {
        return;
    }
    for i in 0..block.len() - 4 {
        if candidate(block, i) {
            rewrite(block, i, offset, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn calls_to_one_target_become_identical() {
        // Three calls to absolute address 0x1000 from different places.
        let mut code = vec![0x90u8; 600];
        for &at in &[100usize, 300, 500] {
            let rel = (0x1000i64 - (at as i64 + 5)) as i32;
            code[at] = 0xE8;
            code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
        }
        let original = code.clone();
        encode(&mut code, 0);
        assert_eq!(code[101..104], code[301..304]);
        assert_eq!(code[101..104], code[501..504]);
        decode(&mut code, 0);
        assert_eq!(code, original);
    }

    #[test]
    fn detection() {
        assert!(!wanted(
            b"plain text with no machine code in it at all, just words and spaces"
        ));
        assert!(!wanted(&[0u8; 10]));
        let mut code = vec![0x90u8; 1000];
        for at in (0..990).step_by(50) {
            code[at] = 0xE8;
            code[at + 4] = 0x00;
        }
        assert!(wanted(&code));
    }

    proptest! {
        #[test]
        fn roundtrip_any_bytes(
            mut data in proptest::collection::vec(
                prop_oneof![Just(0xE8u8), Just(0xE9), Just(0x00), Just(0xFF), any::<u8>()], 0..600),
            offset in any::<u64>(),
        ) {
            let original = data.clone();
            encode(&mut data, offset);
            decode(&mut data, offset);
            prop_assert_eq!(data, original);
        }
    }
}
