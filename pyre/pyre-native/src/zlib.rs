//! zlib DEFLATE backend — deliberate duplication of RustPython's zlib
//! machinery (`stdlib/src/zlib.rs` + the shared `compression.rs` state
//! machine), stripped of the `vm`/`PyResult` layer and specialised to the
//! `flate2` zlib-rs backend.  Errors surface as plain messages the
//! interpreter maps onto `zlib.error`.  Kept outside the LLBC extraction so
//! the native codec never lowers into the traceable graph.

use flate2::{
    Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status, write::ZlibEncoder,
};
use std::io::Write;

pub const MAX_WBITS: i8 = 15;
pub const DEF_BUF_SIZE: usize = 16 * 1024;

// flush modes (libz values); mirrored as module constants on the Python side.
pub const Z_NO_FLUSH: i32 = 0;
pub const Z_PARTIAL_FLUSH: i32 = 1;
pub const Z_SYNC_FLUSH: i32 = 2;
pub const Z_FULL_FLUSH: i32 = 3;
pub const Z_FINISH: i32 = 4;

const Z_DEFAULT_COMPRESSION: i32 = -1;
const Z_NO_COMPRESSION: i32 = 0;
const Z_BEST_COMPRESSION: i32 = 9;

const CHUNKSIZE: usize = u32::MAX as usize;
const USE_AFTER_FINISH_ERR: &str = "Error -2: inconsistent stream state";

fn level_compression(level: i32) -> Option<Compression> {
    match level {
        Z_DEFAULT_COMPRESSION => Some(Compression::default()),
        Z_NO_COMPRESSION..=Z_BEST_COMPRESSION => Some(Compression::new(level as u32)),
        _ => None,
    }
}

enum InitOptions {
    Standard { header: bool, wbits: u8 },
    Gzip { wbits: u8 },
}

impl InitOptions {
    fn new(wbits: i8) -> Result<Self, String> {
        let header = wbits > 0;
        let wbits = wbits.unsigned_abs();
        match wbits {
            9..=15 => Ok(Self::Standard { header, wbits }),
            25..=31 => Ok(Self::Gzip { wbits: wbits - 16 }),
            _ => Err("Invalid initialization option".to_owned()),
        }
    }

    fn decompress(self) -> Decompress {
        match self {
            Self::Standard { header, wbits } => Decompress::new_with_window_bits(header, wbits),
            Self::Gzip { wbits } => Decompress::new_gzip(wbits),
        }
    }

    fn compress(self, level: Compression) -> Compress {
        match self {
            Self::Standard { header, wbits } => {
                Compress::new_with_window_bits(level, header, wbits)
            }
            Self::Gzip { wbits } => Compress::new_gzip(level, wbits),
        }
    }
}

// ── input chunker (compression.rs Chunker) ──────────────────────────────

struct Chunker<'a> {
    data1: &'a [u8],
    data2: &'a [u8],
}

impl<'a> Chunker<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self {
            data1: data,
            data2: &[],
        }
    }
    fn chain(data1: &'a [u8], data2: &'a [u8]) -> Self {
        if data1.is_empty() {
            Self {
                data1: data2,
                data2: &[],
            }
        } else {
            Self { data1, data2 }
        }
    }
    const fn len(&self) -> usize {
        self.data1.len() + self.data2.len()
    }
    const fn is_empty(&self) -> bool {
        self.data1.is_empty()
    }
    fn to_vec(&self) -> Vec<u8> {
        [self.data1, self.data2].concat()
    }
    fn chunk(&self) -> &'a [u8] {
        self.data1.get(..CHUNKSIZE).unwrap_or(self.data1)
    }
    fn advance(&mut self, consumed: usize) {
        self.data1 = &self.data1[consumed..];
        if self.data1.is_empty() {
            self.data1 = core::mem::take(&mut self.data2);
        }
    }
}

/// compression.rs `_decompress_chunks`, specialised to `Decompress` with an
/// optional dictionary retry (the `DecompressWithDict::maybe_set_dict` path).
fn decompress_chunks(
    data: &mut Chunker<'_>,
    d: &mut Decompress,
    zdict: Option<&[u8]>,
    bufsize: usize,
    max_length: Option<usize>,
    calc_flush: impl Fn(bool) -> FlushDecompress,
) -> Result<(Vec<u8>, bool), String> {
    if data.is_empty() {
        return Ok((Vec::new(), true));
    }
    let max_length = max_length.unwrap_or(usize::MAX);
    let mut buf = Vec::new();

    'outer: loop {
        let chunk = data.chunk();
        let flush = calc_flush(chunk.len() == data.len());
        loop {
            let additional = core::cmp::min(bufsize, max_length - buf.capacity());
            if additional == 0 {
                return Ok((buf, false));
            }
            buf.reserve_exact(additional);

            let prev_in = d.total_in();
            let res = d.decompress_vec(chunk, &mut buf, flush);
            let consumed = d.total_in() - prev_in;

            data.advance(consumed as usize);

            match res {
                Ok(status) => {
                    let stream_end = status == Status::StreamEnd;
                    if stream_end || data.is_empty() {
                        buf.shrink_to_fit();
                        return Ok((buf, stream_end));
                    } else if !chunk.is_empty() && consumed == 0 {
                        continue;
                    }
                    continue 'outer;
                }
                Err(e) => {
                    // maybe_set_dict: retry once with the stored dictionary.
                    match zdict.filter(|_| e.needs_dictionary().is_some()) {
                        Some(zd) => {
                            d.set_dictionary(zd).map_err(|e| e.to_string())?;
                            continue 'outer;
                        }
                        None => return Err(e.to_string()),
                    }
                }
            }
        }
    }
}

fn decompress_all(
    data: &[u8],
    d: &mut Decompress,
    zdict: Option<&[u8]>,
    bufsize: usize,
    max_length: Option<usize>,
    calc_flush: impl Fn(bool) -> FlushDecompress,
) -> Result<(Vec<u8>, bool), String> {
    let mut chunker = Chunker::new(data);
    decompress_chunks(&mut chunker, d, zdict, bufsize, max_length, calc_flush)
}

// ── one-shot entry points ───────────────────────────────────────────────

/// `zlib.compress(data, level, wbits)`.
#[inline(never)]
pub fn compress(data: &[u8], level: i32, wbits: i8) -> Result<Vec<u8>, String> {
    let level = level_compression(level).ok_or_else(|| "Bad compression level".to_owned())?;
    let compress = InitOptions::new(wbits)?.compress(level);
    let mut encoder = ZlibEncoder::new_with_compress(Vec::new(), compress);
    encoder.write_all(data).unwrap();
    Ok(encoder.finish().unwrap())
}

/// `zlib.decompress(data, wbits, bufsize)`.
#[inline(never)]
pub fn decompress(data: &[u8], wbits: i8, bufsize: usize) -> Result<Vec<u8>, String> {
    let mut d = InitOptions::new(wbits)?.decompress();
    let (buf, stream_end) =
        decompress_all(data, &mut d, None, bufsize, None, |_| FlushDecompress::Sync)?;
    if !stream_end {
        return Err("Error -5 while decompressing data: incomplete or truncated stream".to_owned());
    }
    Ok(buf)
}

// ── streaming compressor (zlib.Compress) ────────────────────────────────

pub struct Compressor {
    compress: Option<Compress>,
}

impl Compressor {
    #[inline(never)]
    pub fn new(level: i32, wbits: i8, zdict: Option<&[u8]>) -> Result<Self, String> {
        let level =
            level_compression(level).ok_or_else(|| "invalid initialization option".to_owned())?;
        let mut compress = InitOptions::new(wbits)?.compress(level);
        if let Some(zdict) = zdict {
            compress.set_dictionary(zdict).map_err(|e| e.to_string())?;
        }
        Ok(Self {
            compress: Some(compress),
        })
    }

    #[inline(never)]
    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>, String> {
        let compressor = self
            .compress
            .as_mut()
            .ok_or_else(|| USE_AFTER_FINISH_ERR.to_owned())?;
        let mut buf = Vec::new();
        for mut chunk in data.chunks(CHUNKSIZE) {
            while !chunk.is_empty() {
                buf.reserve(DEF_BUF_SIZE);
                let prev_in = compressor.total_in();
                compressor
                    .compress_vec(chunk, &mut buf, FlushCompress::None)
                    .map_err(|_| "error while compressing".to_owned())?;
                let consumed = (compressor.total_in() - prev_in) as usize;
                chunk = &chunk[consumed..];
            }
        }
        buf.shrink_to_fit();
        Ok(buf)
    }

    /// Returns the flushed bytes; `finished` is true once a `Z_FINISH` flush
    /// has consumed the stream (the object may no longer be used).
    #[inline(never)]
    pub fn flush(&mut self, mode: i32) -> Result<Vec<u8>, String> {
        let flush = match mode {
            Z_NO_FLUSH => return Ok(vec![]),
            Z_PARTIAL_FLUSH => FlushCompress::Partial,
            Z_SYNC_FLUSH => FlushCompress::Sync,
            Z_FULL_FLUSH => FlushCompress::Full,
            Z_FINISH => FlushCompress::Finish,
            _ => return Err("invalid mode".to_owned()),
        };
        let compressor = self
            .compress
            .as_mut()
            .ok_or_else(|| USE_AFTER_FINISH_ERR.to_owned())?;
        let mut buf = Vec::new();
        let status = loop {
            if buf.len() == buf.capacity() {
                buf.reserve(DEF_BUF_SIZE);
            }
            let status = compressor
                .compress_vec(&[], &mut buf, flush)
                .map_err(|_| "error while compressing".to_owned())?;
            if buf.len() != buf.capacity() {
                break status;
            }
        };
        if status == Status::StreamEnd {
            if mode == Z_FINISH {
                self.compress = None;
            } else {
                return Err("unexpected eof".to_owned());
            }
        }
        buf.shrink_to_fit();
        Ok(buf)
    }

    pub fn is_finished(&self) -> bool {
        self.compress.is_none()
    }
}

// ── streaming decompressor (zlib.Decompress) ────────────────────────────

pub struct Decompressor {
    decompress: Option<Decompress>,
    zdict: Option<Vec<u8>>,
    eof: bool,
    unused_data: Vec<u8>,
    unconsumed_tail: Vec<u8>,
}

impl Decompressor {
    #[inline(never)]
    pub fn new(wbits: i8, zdict: Option<Vec<u8>>) -> Result<Self, String> {
        let mut decompress = InitOptions::new(wbits)?.decompress();
        if let Some(d) = &zdict
            && wbits < 0
        {
            decompress
                .set_dictionary(d)
                .map_err(|_| "failed to set dictionary".to_owned())?;
        }
        Ok(Self {
            decompress: Some(decompress),
            zdict,
            eof: false,
            unused_data: Vec::new(),
            unconsumed_tail: Vec::new(),
        })
    }

    pub fn eof(&self) -> bool {
        self.eof
    }
    pub fn unused_data(&self) -> &[u8] {
        &self.unused_data
    }
    pub fn unconsumed_tail(&self) -> &[u8] {
        &self.unconsumed_tail
    }

    /// zlib.rs `PyDecompress::decompress_inner`.
    fn decompress_inner(
        &mut self,
        data: &[u8],
        bufsize: usize,
        max_length: Option<usize>,
        is_flush: bool,
    ) -> Result<(Result<Vec<u8>, String>, bool), String> {
        let Self {
            decompress,
            zdict,
            unused_data,
            unconsumed_tail,
            ..
        } = self;
        let Some(d) = decompress.as_mut() else {
            return Err(USE_AFTER_FINISH_ERR.to_owned());
        };

        let prev_in = d.total_in();
        let res = if is_flush {
            // ignore zdict on a flush, finish on the final chunk
            let calc_flush = |final_chunk| {
                if final_chunk {
                    FlushDecompress::Finish
                } else {
                    FlushDecompress::None
                }
            };
            decompress_all(data, d, None, bufsize, max_length, calc_flush)
        } else {
            decompress_all(data, d, zdict.as_deref(), bufsize, max_length, |_| {
                FlushDecompress::Sync
            })
        };
        let (ret, stream_end) = match res {
            Ok((buf, stream_end)) => (Ok(buf), stream_end),
            Err(err) => (Err(err), false),
        };
        let consumed = (d.total_in() - prev_in) as usize;

        // save unused input
        let unconsumed = &data[consumed..];
        if !unconsumed.is_empty() {
            if stream_end {
                unused_data.extend_from_slice(unconsumed);
            } else {
                *unconsumed_tail = unconsumed.to_vec();
            }
        } else if !unconsumed_tail.is_empty() {
            unconsumed_tail.clear();
        }

        Ok((ret, stream_end))
    }

    /// `Decompress.decompress(data, max_length)`; `max_length` of `None` is
    /// unlimited.
    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, String> {
        let (ret, stream_end) = self.decompress_inner(data, DEF_BUF_SIZE, max_length, false)?;
        self.eof |= stream_end;
        ret
    }

    /// `Decompress.flush(length)`.
    #[inline(never)]
    pub fn flush(&mut self, length: usize) -> Result<Vec<u8>, String> {
        let data = core::mem::take(&mut self.unconsumed_tail);
        let (ret, _) = self.decompress_inner(&data, length, None, true)?;
        if self.eof {
            self.decompress = None;
        }
        ret
    }
}

// ── buffered decompressor (zlib._ZlibDecompressor, DecompressState) ──────

/// Error surface of [`ZlibDecompressor::decompress`]: `Zlib` maps to
/// `zlib.error`, `Eof` to `EOFError` ("End of stream already reached").
#[derive(Debug)]
pub enum DecompressError {
    Zlib(String),
    Eof,
}

pub struct ZlibDecompressor {
    decompress: Decompress,
    zdict: Option<Vec<u8>>,
    unused_data: Vec<u8>,
    input_buffer: Vec<u8>,
    eof: bool,
    needs_input: bool,
}

impl ZlibDecompressor {
    #[inline(never)]
    pub fn new(wbits: i8, zdict: Option<Vec<u8>>) -> Result<Self, String> {
        let mut decompress = InitOptions::new(wbits)?.decompress();
        if let Some(d) = &zdict
            && wbits < 0
        {
            decompress
                .set_dictionary(d)
                .map_err(|_| "failed to set dictionary".to_owned())?;
        }
        Ok(Self {
            decompress,
            zdict,
            unused_data: Vec::new(),
            input_buffer: Vec::new(),
            eof: false,
            needs_input: true,
        })
    }

    pub fn eof(&self) -> bool {
        self.eof
    }
    pub fn unused_data(&self) -> &[u8] {
        &self.unused_data
    }
    pub fn needs_input(&self) -> bool {
        self.needs_input
    }

    /// compression.rs `DecompressState::decompress`.
    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, DecompressError> {
        if self.eof {
            return Err(DecompressError::Eof);
        }

        let Self {
            decompress,
            zdict,
            unused_data,
            input_buffer,
            eof,
            needs_input,
        } = self;

        let mut chunks = Chunker::chain(input_buffer.as_slice(), data);

        let prev_len = chunks.len();
        let (ret, stream_end) = match decompress_chunks(
            &mut chunks,
            decompress,
            zdict.as_deref(),
            DEF_BUF_SIZE,
            max_length,
            |_| FlushDecompress::Sync,
        ) {
            Ok((buf, stream_end)) => (Ok(buf), stream_end),
            Err(err) => (Err(err), false),
        };
        let consumed = prev_len - chunks.len();

        *eof |= stream_end;

        if *eof {
            *needs_input = false;
            if !chunks.is_empty() {
                *unused_data = chunks.to_vec();
            }
        } else if chunks.is_empty() {
            input_buffer.clear();
            *needs_input = true;
        } else {
            *needs_input = false;
            if let Some(n_consumed_from_data) = consumed.checked_sub(input_buffer.len()) {
                input_buffer.clear();
                input_buffer.extend_from_slice(&data[n_consumed_from_data..]);
            } else {
                input_buffer.drain(..consumed);
                input_buffer.extend_from_slice(data);
            }
        }

        ret.map_err(DecompressError::Zlib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_default() {
        let data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit";
        let c = compress(data, -1, MAX_WBITS).unwrap();
        let d = decompress(&c, MAX_WBITS, DEF_BUF_SIZE).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn roundtrip_all_levels_and_wbits() {
        let data = b"the quick brown fox jumps over the lazy dog".repeat(20);
        for level in 0..=9 {
            for &wbits in &[15i8, 31, -15] {
                let c = compress(&data, level, wbits).unwrap();
                let d = decompress(&c, wbits, DEF_BUF_SIZE).unwrap();
                assert_eq!(d, data, "level={level} wbits={wbits}");
            }
        }
    }

    #[test]
    fn bad_level_rejected() {
        assert!(compress(b"x", 10, MAX_WBITS).is_err());
        assert!(compress(b"x", -40, MAX_WBITS).is_err());
    }

    #[test]
    fn streaming_roundtrip() {
        let mut co = Compressor::new(-1, MAX_WBITS, None).unwrap();
        let mut out = co.compress(b"hello ").unwrap();
        out.extend(co.compress(b"world").unwrap());
        out.extend(co.flush(Z_FINISH).unwrap());
        assert!(co.is_finished());

        let mut do_ = Decompressor::new(MAX_WBITS, None).unwrap();
        let got = do_.decompress(&out, None).unwrap();
        assert_eq!(got, b"hello world");
        assert!(do_.eof());
    }

    #[test]
    fn buffered_decompressor_gzip_roundtrip() {
        // gzip uses _ZlibDecompressor(wbits=-MAX_WBITS) over raw deflate.
        let raw = compress(b"gzip payload contents", -1, -15).unwrap();
        let mut d = ZlibDecompressor::new(-15, None).unwrap();
        let got = d.decompress(&raw, None).unwrap();
        assert_eq!(got, b"gzip payload contents");
        assert!(d.eof());
        // decompress after eof raises Eof
        assert!(matches!(d.decompress(b"", None), Err(DecompressError::Eof)));
    }

    #[test]
    fn buffered_decompressor_incremental_needs_input() {
        let full = compress(&b"streamed content ".repeat(100), -1, MAX_WBITS).unwrap();
        let mut d = ZlibDecompressor::new(MAX_WBITS, None).unwrap();
        let mut out = Vec::new();
        for byte in full.chunks(1) {
            out.extend(d.decompress(byte, None).unwrap());
        }
        assert_eq!(out, b"streamed content ".repeat(100));
        assert!(d.eof());
    }

    #[test]
    fn streaming_max_length_unconsumed_tail() {
        let full = compress(&b"abcdefghij".repeat(50), -1, MAX_WBITS).unwrap();
        let mut d = Decompressor::new(MAX_WBITS, None).unwrap();
        let first = d.decompress(&full, Some(5)).unwrap();
        assert_eq!(first.len(), 5);
        assert!(!d.unconsumed_tail().is_empty());
        let rest = d.flush(DEF_BUF_SIZE).unwrap();
        assert_eq!([first, rest].concat(), b"abcdefghij".repeat(50));
    }
}
