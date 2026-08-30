//! Runtime decompression for the embedded wasm standard-library image.
//!
//! Blob construction remains in `pyre-interpreter`'s build script; only the
//! runtime-independent decoder lives in this non-LLBC crate.

pub fn decompress_size_prepended(blob: &[u8]) -> Result<Vec<u8>, String> {
    lz4_flex::block::decompress_size_prepended(blob).map_err(|err| err.to_string())
}
