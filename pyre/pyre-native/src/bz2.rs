//! bzip2 backend — PyPy: `pypy/module/bz2/interp_bz2.py`.
//!
//! Upstream drives the C libbz2 through `rffi`; the same stream API comes
//! here from the `bzip2` crate on its pure-Rust `libbz2-rs-sys` backend, the
//! companion of the `zlib-rs` port `zlib.rs` already uses.  Kept outside the
//! LLBC extraction so the codec never lowers into the traceable graph.
//!
//! `interp_bz2.py` predates the `bzerror` latch and the messages
//! `lib-python/3/test/test_bz2.py` pins, so the observable surface follows
//! `Modules/_bz2module.c` at the version `lib-python/stdlib-version.txt`
//! names; the object shape and buffer growth stay upstream's.

use bzip2::{Action, Compress, Compression, Decompress, Error, Status};

/// `interp_bz2.py:99 INITIAL_BUFFER_SIZE` and `:109 BIGCHUNK` (the 32-bit
/// `rffi.INT` arm, which every supported target takes).
const INITIAL_BUFFER_SIZE: usize = 8192;
const BIGCHUNK: usize = 512 * 1024;

/// `interp_bz2.py:179 _new_buffer_size`: keep doubling until BIGCHUNK, then
/// the buffer size is no longer increased.
const fn new_buffer_size(current_size: usize) -> usize {
    if current_size < BIGCHUNK {
        current_size + current_size
    } else {
        current_size
    }
}

/// The `bzerror` values `interp_bz2.py:158 _catch_bz2_error` separates.  The
/// interpreter module owns the exception mapping, as it owns the object
/// space this backend deliberately knows nothing about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bz2Error {
    /// `BZ_PARAM_ERROR`
    Param,
    /// `BZ_DATA_ERROR` / `BZ_DATA_ERROR_MAGIC`
    Data,
    /// `BZ_SEQUENCE_ERROR`
    Sequence,
    /// `BZ_MEM_ERROR`
    Mem,
}

impl From<Error> for Bz2Error {
    fn from(error: Error) -> Self {
        match error {
            Error::Param => Self::Param,
            Error::Data | Error::DataMagic => Self::Data,
            Error::Sequence => Self::Sequence,
        }
    }
}

// ── streaming compressor (_bz2.BZ2Compressor) ───────────────────────────

/// `interp_bz2.py:266 W_BZ2Compressor` — the compress stream plus the
/// `running` flag the object reads back as "already flushed".
pub struct Compressor {
    compress: Compress,
    flushed: bool,
}

impl Compressor {
    /// `interp_bz2.py:293 _init_bz2comp` — `BZ2_bzCompressInit(bzs,
    /// compresslevel, 0, 0)`.  `None` for a level outside 1..=9, which the
    /// caller reports as the ValueError upstream raises before init.
    #[inline(never)]
    pub fn new(compresslevel: i64) -> Option<Self> {
        let level = u32::try_from(compresslevel)
            .ok()
            .and_then(Compression::try_new)?;
        Some(Self {
            // A zero work factor selects libbz2's own default of 30.
            compress: Compress::new(level, 0),
            flushed: false,
        })
    }

    pub fn is_flushed(&self) -> bool {
        self.flushed
    }

    /// `interp_bz2.py:315 compress`.
    #[inline(never)]
    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>, Bz2Error> {
        self.run(data, Action::Run)
    }

    /// `interp_bz2.py:358 flush` — the stream ends here and the object may
    /// not be used again.
    #[inline(never)]
    pub fn flush(&mut self) -> Result<Vec<u8>, Bz2Error> {
        self.flushed = true;
        self.run(&[], Action::Finish)
    }

    /// One pass over the input with the requested action, growing the output
    /// block whenever libbz2 fills it.
    fn run(&mut self, mut input: &[u8], action: Action) -> Result<Vec<u8>, Bz2Error> {
        let mut out = Vec::new();
        let mut block = vec![0u8; INITIAL_BUFFER_SIZE];
        loop {
            // In regular compression mode, stop when input data is exhausted.
            if action == Action::Run && input.is_empty() {
                break;
            }
            let previous_in = self.compress.total_in();
            let previous_out = self.compress.total_out();
            let status = self.compress.compress(input, &mut block, action)?;
            let consumed = (self.compress.total_in() - previous_in) as usize;
            let produced = (self.compress.total_out() - previous_out) as usize;
            out.extend_from_slice(&block[..produced]);
            input = &input[consumed..];
            // In flushing mode, stop when all buffered data has been flushed.
            if action == Action::Finish && status == Status::StreamEnd {
                break;
            }
            if produced == block.len() {
                block = vec![0u8; new_buffer_size(block.len())];
            }
        }
        out.shrink_to_fit();
        Ok(out)
    }
}

// ── streaming decompressor (_bz2.BZ2Decompressor) ───────────────────────

/// `interp_bz2.py:396 W_BZ2Decompressor` — the decompress stream and the
/// unconsumed-input bookkeeping `decompress` reports back through
/// `needs_input` and `unused_data`.
pub struct Decompressor {
    decompress: Decompress,
    eof: bool,
    /// Re-entering `BZ2_bzDecompress` after a failure can write out of
    /// bounds, so the first failure latches and every later call is refused.
    failed: bool,
    needs_input: bool,
    unused_data: Vec<u8>,
    /// Input handed to `decompress` that libbz2 has not consumed yet.
    input_buffer: Vec<u8>,
}

impl Decompressor {
    /// `interp_bz2.py:429 _init_bz2decomp` — `BZ2_bzDecompressInit(bzs, 0, 0)`,
    /// i.e. the fast (non-`small`) algorithm.
    #[inline(never)]
    pub fn new() -> Self {
        Self {
            decompress: Decompress::new(false),
            eof: false,
            failed: false,
            needs_input: true,
            unused_data: Vec::new(),
            input_buffer: Vec::new(),
        }
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn needs_input(&self) -> bool {
        self.needs_input
    }

    pub fn unused_data(&self) -> &[u8] {
        &self.unused_data
    }

    /// `interp_bz2.py:491 decompress` — `max_length` of `None` is upstream's
    /// negative value, meaning unlimited output.
    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, Bz2Error> {
        // Prepend unconsumed input if necessary.
        let mut input = std::mem::take(&mut self.input_buffer);
        input.extend_from_slice(data);

        let max_length = max_length.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        let mut consumed = 0usize;
        let mut block = vec![0u8; INITIAL_BUFFER_SIZE.min(max_length)];
        loop {
            let previous_in = self.decompress.total_in();
            let previous_out = self.decompress.total_out();
            let status = self.decompress.decompress(&input[consumed..], &mut block);
            consumed += (self.decompress.total_in() - previous_in) as usize;
            let produced = (self.decompress.total_out() - previous_out) as usize;
            out.extend_from_slice(&block[..produced]);
            match status {
                Err(error) => return Err(self.fail(error.into())),
                Ok(Status::MemNeeded) => return Err(self.fail(Bz2Error::Mem)),
                Ok(Status::StreamEnd) => {
                    self.eof = true;
                    break;
                }
                Ok(_) => {}
            }
            if consumed == input.len() {
                break;
            }
            if produced == block.len() {
                // The output block is full: grow it unless `max_length` has
                // already been reached.
                if out.len() == max_length {
                    break;
                }
                block = vec![0u8; new_buffer_size(block.len()).min(max_length - out.len())];
            }
        }

        let leftover = input.len() - consumed;
        if self.eof {
            self.needs_input = false;
            if leftover > 0 {
                self.unused_data = input[consumed..].to_vec();
            }
        } else if leftover == 0 {
            self.needs_input = true;
        } else {
            self.needs_input = false;
            input.drain(..consumed);
            self.input_buffer = input;
        }
        out.shrink_to_fit();
        Ok(out)
    }

    /// Latch the first failure and drop the pending input with it, matching
    /// the `next_in = NULL` the error path leaves behind.
    fn fail(&mut self, error: Bz2Error) -> Bz2Error {
        self.failed = true;
        self.needs_input = false;
        self.input_buffer = Vec::new();
        error
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}
