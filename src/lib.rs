//! NeuroZip: an experimental lossless compressor built from a deterministic
//! online-learned predictor and an integer range coder.
//!
//! This crate currently contains the Phase 1 foundation:
//!
//! * [`rangecoder`] — integer range encoder and decoder;
//! * [`freq`] — deterministic normalisation of model output into strictly
//!   positive frequency tables;
//! * [`model`] — the [`Predictor`](model::Predictor) contract and the
//!   adaptive order-0..N context model (the non-neural baseline);
//! * [`mod@format`] — the `.nz` container;
//! * [`codec`] — block-wise compression and decompression with per-block
//!   stored fallback, checksums and resource limits.
//!
//! ```
//! use neurozip::{compress, decompress, Limits, Options};
//!
//! let data = b"an example, an example, an example".repeat(20);
//! let archive = compress(&data, &Options::default()).unwrap();
//! assert!(archive.len() < data.len());
//! assert_eq!(decompress(&archive, &Limits::default()).unwrap(), data);
//! ```

#![forbid(unsafe_code)]

pub mod codec;
pub mod crc32;
pub mod error;
pub mod format;
pub mod freq;
pub mod model;
pub mod rangecoder;
pub mod x86;

pub use codec::{
    Ablation, Limits, Options, Stats, ablation_size, compress, compress_stream, decompress,
    decompress_stream, sha256,
};
pub use error::{Error, Result};
pub use format::ModelSpec;
pub use model::ctx::CtxParams;
pub use model::mix::MixParams;
