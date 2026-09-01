//! Opaque adapter for the VM-independent zlib engine in rustpython-common.
//!
//! Keep the wrapper methods in pyre-native: the interpreter's LLBC extraction
//! treats this crate as its native boundary and must not lower the compression
//! engine into the traceable graph.

use rustpython_common::compression::zlib as common;

pub const MAX_WBITS: i32 = common::MAX_WBITS;
pub const DEF_BUF_SIZE: usize = common::DEF_BUF_SIZE;

pub const Z_NO_FLUSH: i32 = common::Z_NO_FLUSH;
pub const Z_PARTIAL_FLUSH: i32 = common::Z_PARTIAL_FLUSH;
pub const Z_SYNC_FLUSH: i32 = common::Z_SYNC_FLUSH;
pub const Z_FULL_FLUSH: i32 = common::Z_FULL_FLUSH;
pub const Z_FINISH: i32 = common::Z_FINISH;

#[derive(Debug)]
pub enum InitError {
    InvalidOption,
    Zlib(String),
}

impl InitError {
    pub fn into_message(self) -> String {
        match self {
            Self::InvalidOption => "Invalid initialization option".to_owned(),
            Self::Zlib(message) => message,
        }
    }
}

impl From<common::InitError> for InitError {
    fn from(error: common::InitError) -> Self {
        match error {
            common::InitError::InvalidOption => Self::InvalidOption,
            common::InitError::Zlib(message) => Self::Zlib(message),
        }
    }
}

#[inline(never)]
pub fn compress(data: &[u8], level: i32, wbits: i32) -> Result<Vec<u8>, InitError> {
    common::compress(data, level, wbits).map_err(Into::into)
}

#[inline(never)]
pub fn decompress(data: &[u8], wbits: i32, bufsize: usize) -> Result<Vec<u8>, InitError> {
    common::decompress(data, wbits, bufsize).map_err(Into::into)
}

pub struct Compressor(common::Compressor);

impl Compressor {
    #[inline(never)]
    pub fn new(
        level: i32,
        method: i32,
        wbits: i32,
        mem_level: i32,
        strategy: i32,
        zdict: Option<&[u8]>,
    ) -> Result<Self, InitError> {
        common::Compressor::new(level, method, wbits, mem_level, strategy, zdict)
            .map(Self)
            .map_err(Into::into)
    }

    #[inline(never)]
    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>, String> {
        self.0.compress(data)
    }

    #[inline(never)]
    pub fn flush(&mut self, mode: i32) -> Result<Vec<u8>, String> {
        self.0.flush(mode)
    }

    #[inline(never)]
    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    #[inline(never)]
    pub fn copy(&mut self) -> Result<Self, String> {
        self.0.copy().map(Self)
    }
}

pub struct Decompressor(common::Decompressor);

impl Decompressor {
    #[inline(never)]
    pub fn new(wbits: i32, zdict: Option<Vec<u8>>) -> Result<Self, InitError> {
        common::Decompressor::new(wbits, zdict)
            .map(Self)
            .map_err(Into::into)
    }

    #[inline(never)]
    pub fn eof(&self) -> bool {
        self.0.eof()
    }

    #[inline(never)]
    pub fn unused_data(&self) -> &[u8] {
        self.0.unused_data()
    }

    #[inline(never)]
    pub fn unconsumed_tail(&self) -> &[u8] {
        self.0.unconsumed_tail()
    }

    #[inline(never)]
    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    #[inline(never)]
    pub fn copy(&self) -> Result<Self, String> {
        self.0.copy().map(Self)
    }

    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, String> {
        self.0.decompress(data, max_length)
    }

    #[inline(never)]
    pub fn flush(&mut self, length: usize) -> Result<Vec<u8>, String> {
        self.0.flush(length)
    }
}

#[derive(Debug)]
pub enum DecompressError {
    Zlib(String),
    Eof,
}

pub struct ZlibDecompressor(common::ZlibDecompressor);

impl ZlibDecompressor {
    #[inline(never)]
    pub fn new(wbits: i32, zdict: Option<Vec<u8>>) -> Result<Self, InitError> {
        common::ZlibDecompressor::new(wbits, zdict)
            .map(Self)
            .map_err(Into::into)
    }

    #[inline(never)]
    pub fn eof(&self) -> bool {
        self.0.eof()
    }

    #[inline(never)]
    pub fn unused_data(&self) -> &[u8] {
        self.0.unused_data()
    }

    #[inline(never)]
    pub fn needs_input(&self) -> bool {
        self.0.needs_input()
    }

    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, DecompressError> {
        self.0
            .decompress(data, max_length)
            .map_err(|err| match err {
                common::DecompressError::Zlib(err) => DecompressError::Zlib(err),
                common::DecompressError::Eof => DecompressError::Eof,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_roundtrip() {
        let data = b"shared zlib engine";
        let encoded = compress(data, -1, MAX_WBITS).unwrap();
        assert_eq!(decompress(&encoded, MAX_WBITS, DEF_BUF_SIZE).unwrap(), data);
    }
}
