//! liblzma backend — the codec half of `_lzma`.
//!
//! `Modules/_lzmamodule.c` at the version `lib-python/stdlib-version.txt`
//! names is the structure this follows; PyPy has no interp-level counterpart.
//! The engine is the `xz-core` crate, a pure-Rust port of liblzma with the
//! same entry points, so each function here stands where the C module calls
//! the library. Kept outside the LLBC extraction, as `zlib` and `bz2` are.
//!
//! The object-space half — filter dicts, the argument surface, and the
//! exception classes — belongs to the interpreter module, so nothing here
//! knows what a Python object is.

use std::ffi::c_void;
use std::mem::MaybeUninit;

use xz_core::common::alone_decoder::lzma_alone_decoder;
use xz_core::common::alone_encoder::lzma_alone_encoder;
use xz_core::common::auto_decoder::lzma_auto_decoder;
use xz_core::common::common::{lzma_code, lzma_end, lzma_get_check};
use xz_core::common::easy_encoder::lzma_easy_encoder;
use xz_core::common::filter_common::lzma_filters_free;
use xz_core::common::filter_decoder::{lzma_properties_decode, lzma_raw_decoder};
use xz_core::common::filter_encoder::{
    lzma_properties_encode, lzma_properties_size, lzma_raw_encoder,
};
use xz_core::common::stream_decoder::lzma_stream_decoder;
use xz_core::common::stream_encoder::lzma_stream_encoder;
use xz_core::lzma::lzma_encoder_presets::lzma_lzma_preset;
use xz_core::types::*;

// ── the constants `lzma_exec` publishes ─────────────────────────────────

/// Container formats.  These are the module's own enumeration, not liblzma's.
pub const FORMAT_AUTO: i32 = 0;
pub const FORMAT_XZ: i32 = 1;
pub const FORMAT_ALONE: i32 = 2;
pub const FORMAT_RAW: i32 = 3;

// A liblzma enum is `int` where the C compiler is MSVC and `unsigned` where it
// is not, so `lzma_check`, `lzma_mode` and `lzma_match_finder` change signedness
// with the target.  The module publishes its own `u32` surface and casts at each
// call into the library, keeping that variance inside this file.
pub const CHECK_NONE: u32 = LZMA_CHECK_NONE as u32;
pub const CHECK_CRC32: u32 = LZMA_CHECK_CRC32 as u32;
pub const CHECK_CRC64: u32 = LZMA_CHECK_CRC64 as u32;
pub const CHECK_SHA256: u32 = LZMA_CHECK_SHA256 as u32;
pub const CHECK_ID_MAX: u32 = LZMA_CHECK_ID_MAX as u32;
/// `_lzmamodule.c`'s own `LZMA_CHECK_UNKNOWN`, one past the last real id.
pub const CHECK_UNKNOWN: u32 = CHECK_ID_MAX + 1;

pub const FILTER_LZMA1: u64 = LZMA_FILTER_LZMA1;
pub const FILTER_LZMA2: u64 = LZMA_FILTER_LZMA2;
pub const FILTER_DELTA: u64 = LZMA_FILTER_DELTA;
pub const FILTER_X86: u64 = LZMA_FILTER_X86;
pub const FILTER_IA64: u64 = LZMA_FILTER_IA64;
pub const FILTER_ARM: u64 = LZMA_FILTER_ARM;
pub const FILTER_ARMTHUMB: u64 = LZMA_FILTER_ARMTHUMB;
pub const FILTER_SPARC: u64 = LZMA_FILTER_SPARC;
pub const FILTER_POWERPC: u64 = LZMA_FILTER_POWERPC;

pub const MF_HC3: u32 = LZMA_MF_HC3 as u32;
pub const MF_HC4: u32 = LZMA_MF_HC4 as u32;
pub const MF_BT2: u32 = LZMA_MF_BT2 as u32;
pub const MF_BT3: u32 = LZMA_MF_BT3 as u32;
pub const MF_BT4: u32 = LZMA_MF_BT4 as u32;

pub const MODE_FAST: u32 = LZMA_MODE_FAST as u32;
pub const MODE_NORMAL: u32 = LZMA_MODE_NORMAL as u32;

pub const PRESET_DEFAULT: u32 = 6;
pub const PRESET_EXTREME: u32 = LZMA_PRESET_EXTREME;

/// The longest filter chain liblzma accepts.
pub const FILTERS_MAX: usize = LZMA_FILTERS_MAX as usize;

/// The output block each coding call fills before copying it out.  Upstream
/// grows one buffer instead; the result is the same bytes either way.
const INITIAL_BUFFER_SIZE: usize = 8192;

// ── errors ──────────────────────────────────────────────────────────────

/// The failures `catch_lzma_error` separates, plus the two the filter
/// conversion raises before liblzma is reached.  The interpreter module owns
/// the exception classes and their messages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LzmaError {
    UnsupportedCheck,
    Mem,
    MemLimit,
    Format,
    Options,
    Data,
    Buf,
    Prog,
    Unrecognized(lzma_ret),
    /// `lzma_lzma_preset` rejected the preset.
    InvalidPreset(u32),
    /// A filter id no filter in the chain can carry.
    InvalidFilterId(u64),
    /// FORMAT_ALONE was given a chain that is not a single LZMA1 filter.
    AloneChainNotSingleLzma1,
}

impl LzmaError {
    /// `catch_lzma_error`: `None` for the four return values that are not
    /// failures.
    fn from_ret(ret: lzma_ret) -> Option<Self> {
        match ret {
            LZMA_OK | LZMA_GET_CHECK | LZMA_NO_CHECK | LZMA_STREAM_END => None,
            LZMA_UNSUPPORTED_CHECK => Some(Self::UnsupportedCheck),
            LZMA_MEM_ERROR => Some(Self::Mem),
            LZMA_MEMLIMIT_ERROR => Some(Self::MemLimit),
            LZMA_FORMAT_ERROR => Some(Self::Format),
            LZMA_OPTIONS_ERROR => Some(Self::Options),
            LZMA_DATA_ERROR => Some(Self::Data),
            LZMA_BUF_ERROR => Some(Self::Buf),
            LZMA_PROG_ERROR => Some(Self::Prog),
            other => Some(Self::Unrecognized(other)),
        }
    }

    fn check(ret: lzma_ret) -> Result<lzma_ret, Self> {
        match Self::from_ret(ret) {
            Some(error) => Err(error),
            None => Ok(ret),
        }
    }
}

// ── filter chains ───────────────────────────────────────────────────────

/// One filter dict, already read out of the object space.  Every option
/// `parse_filter_spec_lzma` / `_delta` / `_bcj` accepts has a slot here;
/// an option the filter does not use is simply ignored, exactly as the
/// keyword parse of a different spec would ignore it.
#[derive(Clone, Copy, Default, Debug)]
pub struct FilterSpec {
    pub id: u64,
    pub preset: Option<u32>,
    pub dict_size: Option<u32>,
    pub lc: Option<u32>,
    pub lp: Option<u32>,
    pub pb: Option<u32>,
    pub mode: Option<u32>,
    pub nice_len: Option<u32>,
    pub mf: Option<u32>,
    pub depth: Option<u32>,
    pub dist: Option<u32>,
    pub start_offset: Option<u32>,
}

/// The option struct one chain entry points at.  The chain owns these boxes
/// for as long as liblzma may read them — that is, until the coder has been
/// initialized, which is the same lifetime `free_filter_chain` gives them.
enum FilterOptions {
    Lzma(Box<lzma_options_lzma>),
    Delta(Box<lzma_options_delta>),
    Bcj(Box<lzma_options_bcj>),
}

/// `parse_filter_chain_spec`'s result: the `lzma_filter` array liblzma reads,
/// terminated by `LZMA_VLI_UNKNOWN`, plus the options it points into.
struct FilterChain {
    filters: Vec<lzma_filter>,
    owned: Vec<FilterOptions>,
}

/// `parse_filter_spec_lzma`: fill in the preset's defaults first, then
/// override with whatever the spec named.
fn lzma_options(spec: &FilterSpec) -> Result<Box<lzma_options_lzma>, LzmaError> {
    let preset = spec.preset.unwrap_or(PRESET_DEFAULT);
    let mut options = Box::new(unsafe { MaybeUninit::<lzma_options_lzma>::zeroed().assume_init() });
    if unsafe { lzma_lzma_preset(&mut *options, preset) } != 0 {
        return Err(LzmaError::InvalidPreset(preset));
    }
    if let Some(value) = spec.dict_size {
        options.dict_size = value;
    }
    if let Some(value) = spec.lc {
        options.lc = value;
    }
    if let Some(value) = spec.lp {
        options.lp = value;
    }
    if let Some(value) = spec.pb {
        options.pb = value;
    }
    if let Some(value) = spec.mode {
        options.mode = value as lzma_mode;
    }
    if let Some(value) = spec.nice_len {
        options.nice_len = value;
    }
    if let Some(value) = spec.mf {
        options.mf = value as lzma_match_finder;
    }
    if let Some(value) = spec.depth {
        options.depth = value;
    }
    Ok(options)
}

impl FilterChain {
    /// `parse_filter_chain_spec`.  The caller has already rejected a chain
    /// longer than [`FILTERS_MAX`].
    fn new(specs: &[FilterSpec]) -> Result<Self, LzmaError> {
        let mut chain = Self {
            filters: Vec::with_capacity(specs.len() + 1),
            owned: Vec::with_capacity(specs.len()),
        };
        for spec in specs {
            let options = filter_options(spec)?;
            let pointer = match &options {
                FilterOptions::Lzma(o) => &raw const **o as *mut c_void,
                FilterOptions::Delta(o) => &raw const **o as *mut c_void,
                FilterOptions::Bcj(o) => &raw const **o as *mut c_void,
            };
            chain.owned.push(options);
            chain.filters.push(lzma_filter {
                id: spec.id,
                options: pointer,
            });
        }
        chain.filters.push(lzma_filter {
            id: LZMA_VLI_UNKNOWN,
            options: std::ptr::null_mut(),
        });
        Ok(chain)
    }

    fn as_ptr(&self) -> *const lzma_filter {
        self.filters.as_ptr()
    }

    /// The single LZMA1 options `Compressor_init_alone` hands to
    /// `lzma_alone_encoder`, or `None` for any other chain.
    fn lone_lzma1_options(&self) -> Option<*const lzma_options_lzma> {
        match (self.filters.len(), self.owned.first()) {
            (2, Some(FilterOptions::Lzma(options))) if self.filters[0].id == FILTER_LZMA1 => {
                Some(&raw const **options)
            }
            _ => None,
        }
    }
}

/// `lzma_filter_converter`'s dispatch on the filter id.
fn filter_options(spec: &FilterSpec) -> Result<FilterOptions, LzmaError> {
    match spec.id {
        FILTER_LZMA1 | FILTER_LZMA2 => Ok(FilterOptions::Lzma(lzma_options(spec)?)),
        FILTER_DELTA => {
            let mut options =
                Box::new(unsafe { MaybeUninit::<lzma_options_delta>::zeroed().assume_init() });
            options.type_ = LZMA_DELTA_TYPE_BYTE;
            options.dist = spec.dist.unwrap_or(1);
            Ok(FilterOptions::Delta(options))
        }
        FILTER_X86 | FILTER_POWERPC | FILTER_IA64 | FILTER_ARM | FILTER_ARMTHUMB | FILTER_SPARC => {
            let mut options =
                Box::new(unsafe { MaybeUninit::<lzma_options_bcj>::zeroed().assume_init() });
            options.start_offset = spec.start_offset.unwrap_or(0);
            Ok(FilterOptions::Bcj(options))
        }
        id => Err(LzmaError::InvalidFilterId(id)),
    }
}

// ── the raw stream ──────────────────────────────────────────────────────

/// An `lzma_stream` whose coder is released when the owner is dropped.
///
/// The coder state lives behind `internal` and holds no pointer back to the
/// stream, so moving this struct before the first coding call is safe.
struct Stream {
    raw: lzma_stream,
}

impl Stream {
    fn new() -> Self {
        Self {
            raw: unsafe { MaybeUninit::<lzma_stream>::zeroed().assume_init() },
        }
    }

    /// One `lzma_code` call over `input[consumed..]` into `block`, reporting
    /// what it consumed and produced.
    fn code(
        &mut self,
        input: &[u8],
        consumed: &mut usize,
        block: &mut [u8],
        action: lzma_action,
    ) -> (lzma_ret, usize) {
        let tail = &input[*consumed..];
        self.raw.next_in = tail.as_ptr();
        self.raw.avail_in = tail.len();
        self.raw.next_out = block.as_mut_ptr();
        self.raw.avail_out = block.len();
        let ret = unsafe { lzma_code(&mut self.raw, action) };
        *consumed = input.len() - self.raw.avail_in;
        let produced = block.len() - self.raw.avail_out;
        // The stream must not keep pointing at a borrow that ends here.
        self.raw.next_in = std::ptr::null();
        self.raw.avail_in = 0;
        self.raw.next_out = std::ptr::null_mut();
        self.raw.avail_out = 0;
        (ret, produced)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { lzma_end(&mut self.raw) };
    }
}

/// The block to fill next, never wider than what is left of `max_length`.
fn next_block(produced_so_far: usize, max_length: usize) -> Vec<u8> {
    vec![0u8; INITIAL_BUFFER_SIZE.min(max_length - produced_so_far)]
}

// ── LZMACompressor ──────────────────────────────────────────────────────

pub struct Compressor {
    stream: Stream,
    flushed: bool,
}

impl Compressor {
    /// `Compressor_new`'s three per-format initializers.  `check` is the
    /// integrity check for FORMAT_XZ; the other formats have none.
    pub fn new(
        format: i32,
        check: u32,
        preset: u32,
        filters: Option<&[FilterSpec]>,
    ) -> Result<Self, LzmaError> {
        let mut compressor = Self {
            stream: Stream::new(),
            flushed: false,
        };
        let raw = &mut compressor.stream.raw;
        let ret = match (format, filters) {
            (FORMAT_XZ, None) => unsafe { lzma_easy_encoder(raw, preset, check as lzma_check) },
            (FORMAT_XZ, Some(specs)) => {
                let chain = FilterChain::new(specs)?;
                unsafe { lzma_stream_encoder(raw, chain.as_ptr(), check as lzma_check) }
            }
            (FORMAT_ALONE, None) => {
                let options = lzma_options(&FilterSpec {
                    preset: Some(preset),
                    ..FilterSpec::default()
                })?;
                unsafe { lzma_alone_encoder(raw, &*options) }
            }
            (FORMAT_ALONE, Some(specs)) => {
                let chain = FilterChain::new(specs)?;
                let Some(options) = chain.lone_lzma1_options() else {
                    return Err(LzmaError::AloneChainNotSingleLzma1);
                };
                unsafe { lzma_alone_encoder(raw, options) }
            }
            (_, Some(specs)) => {
                let chain = FilterChain::new(specs)?;
                unsafe { lzma_raw_encoder(raw, chain.as_ptr()) }
            }
            (_, None) => LZMA_PROG_ERROR,
        };
        LzmaError::check(ret)?;
        Ok(compressor)
    }

    pub fn is_flushed(&self) -> bool {
        self.flushed
    }

    /// `_lzma_LZMACompressor_compress_impl`.
    #[inline(never)]
    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>, LzmaError> {
        self.code(data, LZMA_RUN)
    }

    /// `_lzma_LZMACompressor_flush_impl` — the object is spent afterwards.
    #[inline(never)]
    pub fn flush(&mut self) -> Result<Vec<u8>, LzmaError> {
        self.flushed = true;
        self.code(&[], LZMA_FINISH)
    }

    /// `compress()`, the static helper both methods run through.
    fn code(&mut self, data: &[u8], action: lzma_action) -> Result<Vec<u8>, LzmaError> {
        let mut out = Vec::new();
        let mut block = vec![0u8; INITIAL_BUFFER_SIZE];
        let mut consumed = 0usize;
        loop {
            let (ret, produced) = self.stream.code(data, &mut consumed, &mut block, action);
            out.extend_from_slice(&block[..produced]);
            // No input and room left over is not a real error.
            let ret = if ret == LZMA_BUF_ERROR && data.is_empty() && produced < block.len() {
                LZMA_OK
            } else {
                ret
            };
            LzmaError::check(ret)?;
            if (action == LZMA_RUN && consumed == data.len())
                || (action == LZMA_FINISH && ret == LZMA_STREAM_END)
            {
                break;
            }
        }
        out.shrink_to_fit();
        Ok(out)
    }
}

// ── LZMADecompressor ────────────────────────────────────────────────────

pub struct Decompressor {
    stream: Stream,
    check: u32,
    eof: bool,
    needs_input: bool,
    unused_data: Vec<u8>,
    /// Input handed in earlier that the coder has not consumed yet.
    input_buffer: Vec<u8>,
}

impl Decompressor {
    /// `_lzma_LZMADecompressor_impl`.  A caller that names no memory limit
    /// passes `u64::MAX`, as the clinic default does.
    pub fn new(
        format: i32,
        memlimit: u64,
        filters: Option<&[FilterSpec]>,
    ) -> Result<Self, LzmaError> {
        // The decoder reports the stream's check id through these.
        const DECODER_FLAGS: u32 = LZMA_TELL_ANY_CHECK | LZMA_TELL_NO_CHECK;
        let mut decompressor = Self {
            stream: Stream::new(),
            check: CHECK_UNKNOWN,
            eof: false,
            needs_input: true,
            unused_data: Vec::new(),
            input_buffer: Vec::new(),
        };
        let raw = &mut decompressor.stream.raw;
        let ret = match format {
            FORMAT_AUTO => unsafe { lzma_auto_decoder(raw, memlimit, DECODER_FLAGS) },
            FORMAT_XZ => unsafe { lzma_stream_decoder(raw, memlimit, DECODER_FLAGS) },
            FORMAT_ALONE => {
                decompressor.check = CHECK_NONE;
                unsafe { lzma_alone_decoder(raw, memlimit) }
            }
            _ => {
                decompressor.check = CHECK_NONE;
                let chain = FilterChain::new(filters.unwrap_or(&[]))?;
                unsafe { lzma_raw_decoder(raw, chain.as_ptr()) }
            }
        };
        LzmaError::check(ret)?;
        Ok(decompressor)
    }

    pub fn check(&self) -> u32 {
        self.check
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    pub fn needs_input(&self) -> bool {
        self.needs_input
    }

    pub fn unused_data(&self) -> &[u8] {
        &self.unused_data
    }

    /// `decompress()` — the wrapper that prepends unconsumed input, runs the
    /// coder, and decides what `needs_input` and `unused_data` say next.
    #[inline(never)]
    pub fn decompress(
        &mut self,
        data: &[u8],
        max_length: Option<usize>,
    ) -> Result<Vec<u8>, LzmaError> {
        let pending = std::mem::take(&mut self.input_buffer);
        let mut joined;
        let input: &[u8] = if pending.is_empty() {
            data
        } else {
            joined = pending;
            joined.extend_from_slice(data);
            &joined
        };

        let (out, consumed, capped) = match self.decompress_buf(input, max_length) {
            Ok(result) => result,
            Err(error) => return Err(error),
        };

        let leftover = &input[consumed..];
        if self.eof {
            self.needs_input = false;
            if !leftover.is_empty() {
                self.unused_data = leftover.to_vec();
            }
        } else if leftover.is_empty() {
            // Output space ran out with the input exhausted: the coder may
            // still hold bytes, so the next call needs no more input.
            self.needs_input = !capped;
        } else {
            self.needs_input = false;
            self.input_buffer = leftover.to_vec();
        }
        Ok(out)
    }

    /// `decompress_buf`: at most `max_length` bytes out, reporting how much
    /// of the input that took and whether the limit is what stopped it.
    fn decompress_buf(
        &mut self,
        input: &[u8],
        max_length: Option<usize>,
    ) -> Result<(Vec<u8>, usize, bool), LzmaError> {
        let max_length = max_length.unwrap_or(usize::MAX);
        let mut out = Vec::new();
        let mut block = next_block(0, max_length);
        let mut consumed = 0usize;
        let mut capped = false;
        loop {
            let (ret, produced) = self.stream.code(input, &mut consumed, &mut block, LZMA_RUN);
            out.extend_from_slice(&block[..produced]);
            let filled = produced == block.len();
            // Nothing left to read and room left over is not a real error.
            let ret = if ret == LZMA_BUF_ERROR && consumed == input.len() && !filled {
                LZMA_OK
            } else {
                ret
            };
            LzmaError::check(ret)?;
            if ret == LZMA_GET_CHECK || ret == LZMA_NO_CHECK {
                self.check = lzma_get_check(&self.stream.raw) as u32;
            }
            if ret == LZMA_STREAM_END {
                self.eof = true;
                break;
            }
            // Output space is checked before input: the coder may still have
            // a few bytes to hand over even with nothing left to read.
            if filled {
                if out.len() == max_length {
                    capped = true;
                    break;
                }
                block = next_block(out.len(), max_length);
            } else if consumed == input.len() {
                break;
            }
        }
        out.shrink_to_fit();
        Ok((out, consumed, capped))
    }
}

// ── module-level functions ──────────────────────────────────────────────

/// `_lzma_is_check_supported_impl`.
#[inline(never)]
pub fn is_check_supported(check_id: u32) -> bool {
    xz_core::check::check::lzma_check_is_supported(check_id as lzma_check) != 0
}

/// `_lzma__encode_filter_properties_impl`.
#[inline(never)]
pub fn encode_filter_properties(spec: &FilterSpec) -> Result<Vec<u8>, LzmaError> {
    let chain = FilterChain::new(std::slice::from_ref(spec))?;
    let filter = chain.filters[0];
    let mut encoded_size: u32 = 0;
    LzmaError::check(unsafe { lzma_properties_size(&mut encoded_size, &filter) })?;
    let mut properties = vec![0u8; encoded_size as usize];
    LzmaError::check(unsafe { lzma_properties_encode(&filter, properties.as_mut_ptr()) })?;
    Ok(properties)
}

/// One decoded filter as `build_filter_spec` describes it: the id, then the
/// option fields that filter's properties carry.
pub struct DecodedFilter {
    pub id: u64,
    pub fields: Vec<(&'static str, u64)>,
}

/// `_lzma__decode_filter_properties_impl`.
#[inline(never)]
pub fn decode_filter_properties(
    filter_id: u64,
    properties: &[u8],
) -> Result<DecodedFilter, LzmaError> {
    let mut filter = lzma_filter {
        id: filter_id,
        options: std::ptr::null_mut(),
    };
    LzmaError::check(unsafe {
        lzma_properties_decode(
            &mut filter,
            std::ptr::null(),
            properties.as_ptr(),
            properties.len(),
        )
    })?;
    let decoded = unsafe { build_filter_spec(&filter) };
    // `lzma_properties_decode` allocated the options through the library's
    // own allocator, so the library frees them.
    let mut chain = [
        filter,
        lzma_filter {
            id: LZMA_VLI_UNKNOWN,
            options: std::ptr::null_mut(),
        },
    ];
    unsafe { lzma_filters_free(chain.as_mut_ptr(), std::ptr::null()) };
    decoded
}

/// `build_filter_spec`: only the fields `lzma_properties_{encode,decode}`
/// look at are reported, which is why LZMA2 names `dict_size` alone.
///
/// # Safety
/// `filter.options` must be the option struct the filter's id implies, or
/// null for a filter whose properties carry nothing.
unsafe fn build_filter_spec(filter: &lzma_filter) -> Result<DecodedFilter, LzmaError> {
    let mut fields = Vec::new();
    match filter.id {
        FILTER_LZMA1 => {
            let options = unsafe { &*(filter.options as *const lzma_options_lzma) };
            fields.push(("lc", options.lc as u64));
            fields.push(("lp", options.lp as u64));
            fields.push(("pb", options.pb as u64));
            fields.push(("dict_size", options.dict_size as u64));
        }
        FILTER_LZMA2 => {
            let options = unsafe { &*(filter.options as *const lzma_options_lzma) };
            fields.push(("dict_size", options.dict_size as u64));
        }
        FILTER_DELTA => {
            let options = unsafe { &*(filter.options as *const lzma_options_delta) };
            fields.push(("dist", options.dist as u64));
        }
        FILTER_X86 | FILTER_POWERPC | FILTER_IA64 | FILTER_ARM | FILTER_ARMTHUMB | FILTER_SPARC => {
            if !filter.options.is_null() {
                let options = unsafe { &*(filter.options as *const lzma_options_bcj) };
                fields.push(("start_offset", options.start_offset as u64));
            }
        }
        id => return Err(LzmaError::InvalidFilterId(id)),
    }
    Ok(DecodedFilter {
        id: filter.id,
        fields,
    })
}
