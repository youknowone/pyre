//! rustpython-common text-codec engines, with pyre error-handler glue.
//!
//! The conversion loops live in `rustpython_common::encodings`.  This
//! module supplies the `EncodeContext` / `DecodeContext` adapter and
//! routes named / registered handlers through the same UnicodeError
//! objects the rest of pyre already builds.

use std::ops::Range;

use rustpython_common::encodings::{
    self, ByteOrder, CodecContext, DecodeContext, DecodeErrorHandler, EncodeContext,
    EncodeErrorHandler, EncodeReplace, StrBuffer, StrSize,
};
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

use crate::type_methods::{
    EncodeErrorOwner, EncodeReplacement, call_registered_decode_error_handler,
    call_registered_encode_error_handler,
};

pub use encodings::unicode_escape::EscapeNote;

struct EngineStr(Wtf8Buf);

impl AsRef<Wtf8> for EngineStr {
    fn as_ref(&self) -> &Wtf8 {
        &self.0
    }
}

impl StrBuffer for EngineStr {}

pub fn encode_unicode_escape(s: &Wtf8) -> Vec<u8> {
    encodings::unicode_escape::encode_bytes(s)
}

pub fn encode_raw_unicode_escape(s: &Wtf8) -> Vec<u8> {
    encodings::raw_unicode_escape::encode_bytes(s)
}

pub fn encode_utf7(s: &Wtf8) -> Vec<u8> {
    encodings::utf7::encode_bytes(s)
}

pub fn encode_escape(data: &[u8]) -> Vec<u8> {
    encodings::escape::encode(data)
}

pub fn decode_escape(
    data: &[u8],
    errors: &str,
) -> Result<(Vec<u8>, Option<String>), crate::PyError> {
    let mode = encodings::escape::EscapeErrorMode::from_name(errors).ok_or_else(|| {
        crate::PyError::value_error(format!(
            "decoding error; unknown error handling code: {errors}"
        ))
    })?;
    match encodings::escape::decode(data, mode) {
        Ok(result) => Ok(result),
        Err(encodings::escape::EscapeDecodeError::TrailingBackslash) => {
            Err(crate::PyError::value_error("Trailing \\ in string"))
        }
        Err(encodings::escape::EscapeDecodeError::InvalidHex { position }) => Err(
            crate::PyError::value_error(format!("invalid \\x escape at position {position}")),
        ),
        Err(encodings::escape::EscapeDecodeError::UnknownHandler { name }) => {
            Err(crate::PyError::value_error(format!(
                "decoding error; unknown error handling code: {name}"
            )))
        }
    }
}

pub fn encode_utf8(
    s: &Wtf8,
    w_object: pyre_object::PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let ctx = PyreEncodeContext::new(encodings::utf8::ENCODING_NAME, s, w_object);
    encodings::utf8::encode(ctx, &PyreErrors { errors })
}

pub fn encode_ascii(
    s: &Wtf8,
    w_object: pyre_object::PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let ctx = PyreEncodeContext::new(encodings::ascii::ENCODING_NAME, s, w_object);
    encodings::ascii::encode(ctx, &PyreErrors { errors })
}

pub fn encode_latin1(
    s: &Wtf8,
    w_object: pyre_object::PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let ctx = PyreEncodeContext::new(encodings::latin_1::ENCODING_NAME, s, w_object);
    encodings::latin_1::encode(ctx, &PyreErrors { errors })
}

pub fn decode_ascii(data: Vec<u8>, errors: &str) -> Result<Wtf8Buf, crate::PyError> {
    let ctx = PyreDecodeContext::new(encodings::ascii::ENCODING_NAME, data);
    encodings::ascii::decode(ctx, &PyreErrors { errors }).map(|(text, _)| text)
}

pub fn decode_latin1(data: Vec<u8>, errors: &str) -> Result<Wtf8Buf, crate::PyError> {
    let ctx = PyreDecodeContext::new(encodings::latin_1::ENCODING_NAME, data);
    encodings::latin_1::decode(ctx, &PyreErrors { errors }).map(|(text, _)| text)
}

pub fn encode_utf16(
    s: &Wtf8,
    w_object: pyre_object::PyObjectRef,
    errors: &str,
    order: ByteOrder,
    bom: bool,
) -> Result<Vec<u8>, crate::PyError> {
    let name = match order {
        ByteOrder::Native => encodings::utf16::ENCODING_NAME,
        ByteOrder::Little => encodings::utf16::ENCODING_NAME_LE,
        ByteOrder::Big => encodings::utf16::ENCODING_NAME_BE,
    };
    let ctx = PyreEncodeContext::new(name, s, w_object);
    encodings::utf16::encode(ctx, &PyreErrors { errors }, order, bom)
}

pub fn encode_utf32(
    s: &Wtf8,
    w_object: pyre_object::PyObjectRef,
    errors: &str,
    order: ByteOrder,
    bom: bool,
) -> Result<Vec<u8>, crate::PyError> {
    let name = match order {
        ByteOrder::Native => encodings::utf32::ENCODING_NAME,
        ByteOrder::Little => encodings::utf32::ENCODING_NAME_LE,
        ByteOrder::Big => encodings::utf32::ENCODING_NAME_BE,
    };
    let ctx = PyreEncodeContext::new(name, s, w_object);
    encodings::utf32::encode(ctx, &PyreErrors { errors }, order, bom)
}

/// Decode one incremental utf-16/32 chunk.
///
/// Bare `utf-16` / `utf-32` still use native/BOM detection, but a
/// `UnicodeDecodeError.encoding` names the effective `-le` / `-be`
/// spelling the settled byte order selected.
pub fn decode_utf16_32(
    data: &[u8],
    is32: bool,
    order: ByteOrder,
    errors: &str,
    final_: bool,
    error_encoding: &str,
) -> Result<(Wtf8Buf, usize, i32), crate::PyError> {
    let ctx = PyreDecodeContext::new(error_encoding, data.to_vec());
    let handler = PyreErrors { errors };
    if is32 {
        encodings::utf32::decode(ctx, &handler, order, final_)
    } else {
        encodings::utf16::decode(ctx, &handler, order, final_)
    }
}

pub fn decode_unicode_escape(
    data: Vec<u8>,
    errors: &str,
    final_: bool,
) -> Result<(Wtf8Buf, usize, Option<EscapeNote>), crate::PyError> {
    let ctx = PyreDecodeContext::new(encodings::unicode_escape::ENCODING_NAME, data);
    encodings::unicode_escape::decode(ctx, &PyreErrors { errors }, final_)
}

pub fn decode_raw_unicode_escape(
    data: Vec<u8>,
    errors: &str,
    final_: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    let ctx = PyreDecodeContext::new(encodings::raw_unicode_escape::ENCODING_NAME, data);
    encodings::raw_unicode_escape::decode(ctx, &PyreErrors { errors }, final_)
}

pub fn decode_utf7(
    data: Vec<u8>,
    errors: &str,
    final_: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    let ctx = PyreDecodeContext::new(encodings::utf7::ENCODING_NAME, data);
    encodings::utf7::decode(ctx, &PyreErrors { errors }, final_)
}

struct PyreEncodeContext<'a> {
    encoding: &'a str,
    data: &'a Wtf8,
    pos: StrSize,
    w_object: pyre_object::PyObjectRef,
}

impl<'a> PyreEncodeContext<'a> {
    fn new(encoding: &'a str, data: &'a Wtf8, w_object: pyre_object::PyObjectRef) -> Self {
        Self {
            encoding,
            data,
            pos: StrSize::default(),
            w_object,
        }
    }

    fn char_len(&self) -> usize {
        self.data.code_points().count()
    }
}

impl CodecContext for PyreEncodeContext<'_> {
    type Error = crate::PyError;
    type StrBuf = EngineStr;
    type BytesBuf = Vec<u8>;

    fn string(&self, s: Wtf8Buf) -> Self::StrBuf {
        EngineStr(s)
    }

    fn bytes(&self, b: Vec<u8>) -> Self::BytesBuf {
        b
    }
}

impl EncodeContext for PyreEncodeContext<'_> {
    fn full_data(&self) -> &Wtf8 {
        self.data
    }

    fn data_len(&self) -> StrSize {
        StrSize {
            bytes: self.data.len(),
            chars: self.char_len(),
        }
    }

    fn remaining_data(&self) -> &Wtf8 {
        &self.data[self.pos.bytes..]
    }

    fn position(&self) -> StrSize {
        self.pos
    }

    fn restart_from(&mut self, pos: StrSize) -> Result<(), Self::Error> {
        if pos.chars > self.char_len() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::IndexError,
                format!("position {} from error handler out of bounds", pos.chars),
            ));
        }
        self.pos = pos;
        Ok(())
    }

    fn error_encoding(&self, range: Range<StrSize>, reason: Option<&str>) -> Self::Error {
        crate::typedef::unicode_encode_error(
            self.encoding,
            self.w_object,
            range.start.chars as i64,
            range.end.chars as i64,
            reason.unwrap_or("unknown encoding error"),
        )
    }
}

struct PyreDecodeContext {
    encoding: String,
    data: Vec<u8>,
    pos: usize,
}

impl PyreDecodeContext {
    fn new(encoding: &str, data: Vec<u8>) -> Self {
        Self {
            encoding: encoding.to_owned(),
            data,
            pos: 0,
        }
    }
}

impl CodecContext for PyreDecodeContext {
    type Error = crate::PyError;
    type StrBuf = EngineStr;
    type BytesBuf = Vec<u8>;

    fn string(&self, s: Wtf8Buf) -> Self::StrBuf {
        EngineStr(s)
    }

    fn bytes(&self, b: Vec<u8>) -> Self::BytesBuf {
        b
    }
}

impl DecodeContext for PyreDecodeContext {
    fn full_data(&self) -> &[u8] {
        &self.data
    }

    fn remaining_data(&self) -> &[u8] {
        &self.data[self.pos..]
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, by: usize) {
        self.pos += by;
    }

    fn restart_from(&mut self, pos: usize) -> Result<(), Self::Error> {
        if pos > self.data.len() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::IndexError,
                format!("position {pos} from error handler out of bounds"),
            ));
        }
        self.pos = pos;
        Ok(())
    }

    fn error_decoding(&self, byte_range: Range<usize>, reason: Option<&str>) -> Self::Error {
        crate::typedef::unicode_decode_error(
            &self.encoding,
            &self.data,
            byte_range.start,
            byte_range.end,
            reason.unwrap_or("unknown decoding error"),
        )
    }
}

struct PyreErrors<'a> {
    errors: &'a str,
}

impl<'a> EncodeErrorHandler<PyreEncodeContext<'a>> for PyreErrors<'_> {
    fn handle_encode_error(
        &self,
        ctx: &mut PyreEncodeContext<'a>,
        range: Range<StrSize>,
        reason: Option<&str>,
    ) -> Result<(EncodeReplace<PyreEncodeContext<'a>>, StrSize), crate::PyError> {
        match self.errors {
            "strict" => encodings::errors::Strict.handle_encode_error(ctx, range, reason),
            "ignore" => encodings::errors::Ignore.handle_encode_error(ctx, range, reason),
            "replace" => encodings::errors::Replace.handle_encode_error(ctx, range, reason),
            "xmlcharrefreplace" => {
                encodings::errors::XmlCharRefReplace.handle_encode_error(ctx, range, reason)
            }
            "backslashreplace" => {
                encodings::errors::BackslashReplace.handle_encode_error(ctx, range, reason)
            }
            "namereplace" => encodings::errors::NameReplace.handle_encode_error(ctx, range, reason),
            "surrogateescape" => {
                encodings::errors::SurrogateEscape.handle_encode_error(ctx, range, reason)
            }
            "surrogatepass" => SurrogatePass.handle_encode_error(ctx, range, reason),
            _ => {
                let (rep, newpos) = call_registered_encode_error_handler(
                    self.errors,
                    ctx.encoding,
                    ctx.w_object,
                    ctx.char_len(),
                    range.start.chars,
                    range.end.chars,
                    reason.unwrap_or("unknown encoding error"),
                    EncodeErrorOwner::UnicodeObject,
                )?;
                let restart = str_size_at_char(ctx.data, newpos);
                let replace = match rep {
                    EncodeReplacement::Str(cps) => {
                        let mut buf = Wtf8Buf::new();
                        for cp in cps {
                            buf.push(
                                CodePoint::from_u32(cp)
                                    .expect("encode handler returned a Unicode code point"),
                            );
                        }
                        EncodeReplace::Str(EngineStr(buf))
                    }
                    EncodeReplacement::Bytes(bytes) => EncodeReplace::Bytes(bytes),
                };
                Ok((replace, restart))
            }
        }
    }
}

impl DecodeErrorHandler<PyreDecodeContext> for PyreErrors<'_> {
    fn handle_decode_error(
        &self,
        ctx: &mut PyreDecodeContext,
        byte_range: Range<usize>,
        reason: Option<&str>,
    ) -> Result<(EngineStr, usize), crate::PyError> {
        match self.errors {
            "strict" => encodings::errors::Strict.handle_decode_error(ctx, byte_range, reason),
            "ignore" => encodings::errors::Ignore.handle_decode_error(ctx, byte_range, reason),
            "replace" => encodings::errors::Replace.handle_decode_error(ctx, byte_range, reason),
            "backslashreplace" => {
                encodings::errors::BackslashReplace.handle_decode_error(ctx, byte_range, reason)
            }
            "surrogateescape" => {
                encodings::errors::SurrogateEscape.handle_decode_error(ctx, byte_range, reason)
            }
            "surrogatepass" => SurrogatePass.handle_decode_error(ctx, byte_range, reason),
            "xmlcharrefreplace" | "namereplace" => {
                Err(crate::typedef::decode_error_encode_only_handler())
            }
            _ => {
                let mut replace = Wtf8Buf::new();
                let (newpos, new_bytes) = call_registered_decode_error_handler(
                    self.errors,
                    &ctx.encoding,
                    &ctx.data,
                    byte_range.start,
                    byte_range.end,
                    reason.unwrap_or("unknown decoding error"),
                    &mut replace,
                )?;
                if let Some(bytes) = new_bytes {
                    ctx.data = bytes;
                }
                Ok((EngineStr(replace), newpos))
            }
        }
    }
}

struct SurrogatePass;

impl<'a> EncodeErrorHandler<PyreEncodeContext<'a>> for SurrogatePass {
    fn handle_encode_error(
        &self,
        ctx: &mut PyreEncodeContext<'a>,
        range: Range<StrSize>,
        reason: Option<&str>,
    ) -> Result<(EncodeReplace<PyreEncodeContext<'a>>, StrSize), crate::PyError> {
        let kind = StandardEncoding::parse(ctx.encoding)
            .ok_or_else(|| ctx.error_encoding(range.clone(), reason))?;
        let err_str = &ctx.full_data()[range.start.bytes..range.end.bytes];
        let mut out = Vec::new();
        for ch in err_str.code_points() {
            let c = ch.to_u32();
            if !(0xd800..=0xdfff).contains(&c) {
                return Err(ctx.error_encoding(range, reason));
            }
            match kind {
                StandardEncoding::Utf8 => out.extend(ch.encode_wtf8(&mut [0; 4]).as_bytes()),
                StandardEncoding::Utf16Le => out.extend((c as u16).to_le_bytes()),
                StandardEncoding::Utf16Be => out.extend((c as u16).to_be_bytes()),
                StandardEncoding::Utf32Le => out.extend(c.to_le_bytes()),
                StandardEncoding::Utf32Be => out.extend(c.to_be_bytes()),
            }
        }
        Ok((EncodeReplace::Bytes(out), range.end))
    }
}

impl DecodeErrorHandler<PyreDecodeContext> for SurrogatePass {
    fn handle_decode_error(
        &self,
        ctx: &mut PyreDecodeContext,
        byte_range: Range<usize>,
        reason: Option<&str>,
    ) -> Result<(EngineStr, usize), crate::PyError> {
        let kind = StandardEncoding::parse(&ctx.encoding)
            .ok_or_else(|| ctx.error_decoding(byte_range.clone(), reason))?;
        let rest = &ctx.full_data()[byte_range.start..];
        let (value, byte_length) = match kind {
            StandardEncoding::Utf8 => {
                let unit = rest.get(..3).and_then(|chunk| {
                    let [a, b, c] = [chunk[0], chunk[1], chunk[2]];
                    ((a & 0xf0) == 0xe0 && (b & 0xc0) == 0x80 && (c & 0xc0) == 0x80).then_some(
                        (u32::from(a & 0x0f) << 12)
                            + (u32::from(b & 0x3f) << 6)
                            + u32::from(c & 0x3f),
                    )
                });
                (unit, 3)
            }
            StandardEncoding::Utf16Le => (
                rest.get(..2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as u32),
                2,
            ),
            StandardEncoding::Utf16Be => (
                rest.get(..2)
                    .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]) as u32),
                2,
            ),
            StandardEncoding::Utf32Le => (
                rest.get(..4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                4,
            ),
            StandardEncoding::Utf32Be => (
                rest.get(..4)
                    .map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                4,
            ),
        };
        let c = value
            .and_then(CodePoint::from_u32)
            .filter(|c| matches!(c.to_u32(), 0xd800..=0xdfff))
            .ok_or_else(|| ctx.error_decoding(byte_range.clone(), reason))?;
        let mut out = Wtf8Buf::new();
        out.push(c);
        Ok((EngineStr(out), byte_range.start + byte_length))
    }
}

enum StandardEncoding {
    Utf8,
    Utf16Le,
    Utf16Be,
    Utf32Le,
    Utf32Be,
}

impl StandardEncoding {
    fn parse(name: &str) -> Option<Self> {
        let compact: String = name
            .chars()
            .filter(|c| !matches!(c, '-' | '_' | ' '))
            .flat_map(char::to_lowercase)
            .collect();
        match compact.as_str() {
            "utf8" | "cputf8" => Some(Self::Utf8),
            "utf16le" => Some(Self::Utf16Le),
            "utf16be" => Some(Self::Utf16Be),
            "utf16" if cfg!(target_endian = "little") => Some(Self::Utf16Le),
            "utf16" => Some(Self::Utf16Be),
            "utf32le" => Some(Self::Utf32Le),
            "utf32be" => Some(Self::Utf32Be),
            "utf32" if cfg!(target_endian = "little") => Some(Self::Utf32Le),
            "utf32" => Some(Self::Utf32Be),
            _ => None,
        }
    }
}

fn str_size_at_char(data: &Wtf8, chars: usize) -> StrSize {
    match data.code_point_indices().nth(chars) {
        Some((bytes, _)) => StrSize { bytes, chars },
        None => StrSize {
            bytes: data.len(),
            chars,
        },
    }
}
