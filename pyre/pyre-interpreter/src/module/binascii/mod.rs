//! binascii module — PyPy: `pypy/module/binascii/`.
//!
//! Pure-Rust base64 / hex / crc32 conversions, enough for `base64.py`,
//! `email`, `quopri`, and the hashlib/zip helpers to import and run.

use pyre_object::*;

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Accept a str (ASCII) or any bytes-like and surface the raw bytes.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if is_str(obj) {
            Ok(w_str_get_value(obj).as_bytes().to_vec())
        } else if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj).to_vec())
        } else {
            Err(crate::PyError::type_error(
                "argument should be bytes, buffer or ASCII string",
            ))
        }
    }
}

fn b64_encode(data: &[u8], newline: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4 + 1);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[(n >> 18 & 0x3f) as usize]);
        out.push(B64[(n >> 12 & 0x3f) as usize]);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6 & 0x3f) as usize]
        } else {
            b'='
        });
        out.push(if chunk.len() > 2 {
            B64[(n & 0x3f) as usize]
        } else {
            b'='
        });
    }
    if newline {
        out.push(b'\n');
    }
    out
}

fn b64_decode(data: &[u8]) -> Result<Vec<u8>, crate::PyError> {
    // Reverse lookup; ignore whitespace, stop at padding.
    let mut rev = [255u8; 256];
    for (i, &c) in B64.iter().enumerate() {
        rev[c as usize] = i as u8;
    }
    let mut bits = 0u32;
    let mut nbits = 0u32;
    let mut out = Vec::new();
    for &c in data {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        let v = rev[c as usize];
        if v == 255 {
            // Skip non-alphabet bytes (lenient, matching default mode).
            continue;
        }
        bits = (bits << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    Ok(out)
}

fn hex_encode(data: &[u8], sep: Option<(u8, usize)>) -> Vec<u8> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(data.len() * 2);
    for (i, &b) in data.iter().enumerate() {
        if let Some((s, per)) = sep {
            if per > 0 && i != 0 && i % per == 0 {
                out.push(s);
            }
        }
        out.push(HEX[(b >> 4) as usize]);
        out.push(HEX[(b & 0xf) as usize]);
    }
    out
}

fn hex_decode(data: &[u8]) -> Result<Vec<u8>, crate::PyError> {
    let stripped: Vec<u8> = data
        .iter()
        .copied()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if stripped.len() % 2 != 0 {
        return Err(crate::PyError::value_error("Odd-length string"));
    }
    let mut out = Vec::with_capacity(stripped.len() / 2);
    let nibble = |c: u8| -> Result<u8, crate::PyError> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(crate::PyError::value_error("Non-hexadecimal digit found")),
        }
    };
    for pair in stripped.chunks(2) {
        out.push((nibble(pair[0])? << 4) | nibble(pair[1])?);
    }
    Ok(out)
}

fn crc32_compute(buf: &[u8], start: u32) -> u32 {
    let mut crc = !start;
    for &b in buf {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

crate::py_module! {
    "binascii",
    exceptions: {
        // binascii.c — Error subclasses ValueError; Incomplete subclasses
        // Exception (NULL base).
        "Error" => crate::builtins::lookup_exc_class("ValueError").expect("ValueError installed"),
        "Incomplete" => crate::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    functions: {
        "b2a_base64" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let newline = crate::builtins::kwarg_get(kwargs, "newline")
                .or_else(|| pos.get(1).copied())
                .map(|o| crate::baseobjspace::is_true(o).unwrap_or(true))
                .unwrap_or(true);
            Ok(w_bytes_from_bytes(&b64_encode(&data, newline)))
        },
        "a2b_base64" / * = |args| {
            let (pos, _kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            Ok(w_bytes_from_bytes(&b64_decode(&data)?))
        },
        "b2a_hex" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let sep = crate::builtins::kwarg_get(kwargs, "sep")
                .or_else(|| pos.get(1).copied())
                .and_then(|o| as_bytes(o).ok())
                .and_then(|b| b.first().copied());
            let per = crate::builtins::kwarg_get(kwargs, "bytes_per_sep")
                .or_else(|| pos.get(2).copied())
                .map(|o| unsafe { w_int_get_value(o) } as usize)
                .unwrap_or(1);
            Ok(w_bytes_from_bytes(&hex_encode(&data, sep.map(|s| (s, per)))))
        },
        "hexlify" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let sep = crate::builtins::kwarg_get(kwargs, "sep")
                .or_else(|| pos.get(1).copied())
                .and_then(|o| as_bytes(o).ok())
                .and_then(|b| b.first().copied());
            let per = crate::builtins::kwarg_get(kwargs, "bytes_per_sep")
                .or_else(|| pos.get(2).copied())
                .map(|o| unsafe { w_int_get_value(o) } as usize)
                .unwrap_or(1);
            Ok(w_bytes_from_bytes(&hex_encode(&data, sep.map(|s| (s, per)))))
        },
        "a2b_hex" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            Ok(w_bytes_from_bytes(&hex_decode(&data)?))
        },
        "unhexlify" / 1 = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            Ok(w_bytes_from_bytes(&hex_decode(&data)?))
        },
        "crc32" / * = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let start = args.get(1).map(|&o| unsafe { w_int_get_value(o) } as u32).unwrap_or(0);
            Ok(w_int_new(crc32_compute(&data, start) as i64))
        },
    },
}
