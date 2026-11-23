//! Predictors: deterministic, online-learned models of the next byte.

pub mod ctx;
pub mod mix;

use crate::freq::Freqs;

/// A deterministic adaptive model of the next byte.
///
/// # Contract
///
/// Encoder and decoder each own a predictor and drive it identically, so
/// implementations must obey all of the following:
///
/// * **Determinism.** State and output depend only on the construction
///   parameters and the sequence of bytes passed to [`update`]. No floating
///   point, no randomness, no dependence on platform, thread count or time.
/// * **Prediction is observation only.** Calling or not calling [`predict`]
///   must not change any later prediction. This lets a decoder skip
///   prediction for blocks that were stored without range coding and still
///   stay in step with the encoder.
/// * **Strictly positive output.** [`predict`] fills a table in which every
///   entry is at least 1 and the entries sum to
///   [`TOTAL`](crate::rangecoder::TOTAL).
///
/// [`update`]: Predictor::update
/// [`predict`]: Predictor::predict
pub trait Predictor {
    /// Frequency table for the next byte.
    fn predict(&mut self, out: &mut Freqs);
    /// Learn from the byte that actually occurred.
    fn update(&mut self, byte: u8);
    /// Move the context past `byte` without learning from it.
    ///
    /// The archive format never uses this. It exists for the "adaptive
    /// learning disabled" ablation, which measures a model frozen part-way
    /// through the input.
    fn advance(&mut self, byte: u8);
    /// Return to the defined initial state.
    fn reset(&mut self);
}
