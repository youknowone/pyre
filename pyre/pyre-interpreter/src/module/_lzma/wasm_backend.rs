//! rustpython-common does not compile `compression::lzma` on Android or
//! wasm32.  The xz-core engine in `pyre_native` supplies the same surface
//! those targets used before the shared module existed.

use pyre_native::lzma as native;

pub const FORMAT_AUTO: i32 = native::FORMAT_AUTO;
pub const FORMAT_XZ: i32 = native::FORMAT_XZ;
pub const FORMAT_ALONE: i32 = native::FORMAT_ALONE;
pub const FORMAT_RAW: i32 = native::FORMAT_RAW;

pub const CHECK_NONE: i32 = native::CHECK_NONE as i32;
pub const CHECK_CRC32: i32 = native::CHECK_CRC32 as i32;
pub const CHECK_CRC64: i32 = native::CHECK_CRC64 as i32;
pub const CHECK_SHA256: i32 = native::CHECK_SHA256 as i32;
pub const CHECK_ID_MAX: i32 = native::CHECK_ID_MAX as i32;
pub const CHECK_UNKNOWN: i32 = native::CHECK_UNKNOWN as i32;

pub const FILTER_LZMA1: u64 = native::FILTER_LZMA1;
pub const FILTER_LZMA2: u64 = native::FILTER_LZMA2;
pub const FILTER_DELTA: u64 = native::FILTER_DELTA;
pub const FILTER_X86: u64 = native::FILTER_X86;
pub const FILTER_POWERPC: u64 = native::FILTER_POWERPC;
pub const FILTER_IA64: u64 = native::FILTER_IA64;
pub const FILTER_ARM: u64 = native::FILTER_ARM;
pub const FILTER_ARMTHUMB: u64 = native::FILTER_ARMTHUMB;
pub const FILTER_SPARC: u64 = native::FILTER_SPARC;

pub const MF_HC3: i32 = native::MF_HC3 as i32;
pub const MF_HC4: i32 = native::MF_HC4 as i32;
pub const MF_BT2: i32 = native::MF_BT2 as i32;
pub const MF_BT3: i32 = native::MF_BT3 as i32;
pub const MF_BT4: i32 = native::MF_BT4 as i32;

pub const MODE_FAST: i32 = native::MODE_FAST as i32;
pub const MODE_NORMAL: i32 = native::MODE_NORMAL as i32;

pub const PRESET_DEFAULT: u32 = native::PRESET_DEFAULT;
pub const PRESET_EXTREME: u32 = native::PRESET_EXTREME;

pub use native::FilterSpec;

#[allow(dead_code)] // same four variants as rustpython-common::compression::lzma::Error
pub enum Error {
    Memory,
    Value(String),
    Lzma(String),
    Eof,
}

fn map_error(error: native::LzmaError) -> Error {
    use native::LzmaError as E;
    match error {
        E::Mem => Error::Memory,
        E::UnsupportedCheck => Error::Lzma("Unsupported integrity check".to_owned()),
        E::MemLimit => Error::Lzma("Memory usage limit exceeded".to_owned()),
        E::Format => Error::Lzma("Input format not supported by decoder".to_owned()),
        E::Options => Error::Lzma("Invalid or unsupported options".to_owned()),
        E::Data => Error::Lzma("Corrupt input data".to_owned()),
        E::Buf => Error::Lzma("Insufficient buffer space".to_owned()),
        E::Prog => Error::Lzma("Internal error".to_owned()),
        E::Unrecognized(ret) => Error::Lzma(format!("Unrecognized error from liblzma: {ret}")),
        E::InvalidPreset(preset) => Error::Lzma(format!("Invalid compression preset: {preset}")),
        E::InvalidFilterId(id) => Error::Value(format!("Invalid filter ID: {id}")),
        E::AloneChainNotSingleLzma1 => Error::Value(
            "Invalid filter chain for FORMAT_ALONE - must be a single LZMA1 filter".to_owned(),
        ),
    }
}

pub struct Compressor(native::Compressor);

impl Compressor {
    pub fn new(
        format: i32,
        check: i32,
        preset: u32,
        filters: Option<Vec<FilterSpec>>,
    ) -> Result<Self, Error> {
        native::Compressor::new(format, check as u32, preset, filters.as_deref())
            .map(Self)
            .map_err(map_error)
    }

    pub fn compress(&mut self, data: &[u8]) -> Result<Vec<u8>, Error> {
        self.0.compress(data).map_err(map_error)
    }

    pub fn flush(&mut self) -> Result<Vec<u8>, Error> {
        self.0.flush().map_err(map_error)
    }
}

pub struct Decompressor(native::Decompressor);

impl Decompressor {
    pub fn new(
        format: i32,
        memlimit: Option<u64>,
        filters: Option<Vec<FilterSpec>>,
    ) -> Result<Self, Error> {
        native::Decompressor::new(format, memlimit.unwrap_or(u64::MAX), filters.as_deref())
            .map(Self)
            .map_err(map_error)
    }

    pub fn decompress(&mut self, data: &[u8], max_length: Option<usize>) -> Result<Vec<u8>, Error> {
        self.0.decompress(data, max_length).map_err(map_error)
    }

    pub fn check(&self) -> i32 {
        self.0.check() as i32
    }

    pub fn eof(&self) -> bool {
        self.0.eof()
    }

    pub fn unused_data(&self) -> &[u8] {
        self.0.unused_data()
    }

    pub fn needs_input(&self) -> bool {
        self.0.needs_input()
    }
}

pub fn is_check_supported(check_id: i32) -> bool {
    native::is_check_supported(check_id as u32)
}

pub fn encode_filter_properties(spec: &FilterSpec) -> Result<Vec<u8>, Error> {
    native::encode_filter_properties(spec).map_err(map_error)
}

pub fn decode_filter_properties(id: u64, properties: &[u8]) -> Result<FilterSpec, Error> {
    let decoded = native::decode_filter_properties(id, properties).map_err(map_error)?;
    let mut spec = FilterSpec {
        id: decoded.id,
        ..FilterSpec::default()
    };
    for (name, value) in decoded.fields {
        let value = Some(value as u32);
        match name {
            "lc" => spec.lc = value,
            "lp" => spec.lp = value,
            "pb" => spec.pb = value,
            "dict_size" => spec.dict_size = value,
            "dist" => spec.dist = value,
            "start_offset" => spec.start_offset = value,
            _ => {}
        }
    }
    Ok(spec)
}
