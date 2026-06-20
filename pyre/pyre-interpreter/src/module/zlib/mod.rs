//! zlib module — PyPy: `pypy/module/zlib/`.
//!
//! Pure-Rust CRC-32 / Adler-32 checksums plus the module constants and the
//! `error` type, enough for `gzip` / `zipfile` / `tarfile` to import. Actual
//! DEFLATE compression is not implemented (no zlib backend is linked), so
//! the `compress` / `decompress` family raises at call time.

use pyre_object::*;

fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj).to_vec())
        } else if is_str(obj) {
            Ok(w_str_get_value(obj).as_bytes().to_vec())
        } else {
            Err(crate::PyError::type_error(
                "a bytes-like object is required",
            ))
        }
    }
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

fn adler32_compute(buf: &[u8], start: u32) -> u32 {
    let mut a = start & 0xffff;
    let mut b = (start >> 16) & 0xffff;
    for &x in buf {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

crate::py_module! {
    "zlib",
    interpleveldefs: {
        "ZLIB_VERSION" => w_str_new("1.3.1"),
        "ZLIB_RUNTIME_VERSION" => w_str_new("1.3.1"),
        // Referenced as an attribute by gzip; a placeholder type keeps
        // `import gzip` working (its use raises only when constructed).
        "_ZlibDecompressor" => crate::typedef::w_object(),
    },
    int_constants: {
        "DEFLATED" => 8,
        "MAX_WBITS" => 15,
        "DEF_MEM_LEVEL" => 8,
        "DEF_BUF_SIZE" => 16384,
        "Z_DEFAULT_COMPRESSION" => -1,
        "Z_NO_COMPRESSION" => 0,
        "Z_BEST_SPEED" => 1,
        "Z_BEST_COMPRESSION" => 9,
        "Z_DEFAULT_STRATEGY" => 0,
        "Z_FILTERED" => 1,
        "Z_HUFFMAN_ONLY" => 2,
        "Z_RLE" => 3,
        "Z_FIXED" => 4,
        "Z_NO_FLUSH" => 0,
        "Z_PARTIAL_FLUSH" => 1,
        "Z_SYNC_FLUSH" => 2,
        "Z_FULL_FLUSH" => 3,
        "Z_FINISH" => 4,
        "Z_BLOCK" => 5,
        "Z_TREES" => 6,
    },
    exceptions: {
        "error" => crate::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    functions: {
        "crc32" / * = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let start = args.get(1).map(|&o| unsafe { w_int_get_value(o) } as u32).unwrap_or(0);
            Ok(w_int_new(crc32_compute(&data, start) as i64))
        },
        "adler32" / * = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let start = args.get(1).map(|&o| unsafe { w_int_get_value(o) } as u32).unwrap_or(1);
            Ok(w_int_new(adler32_compute(&data, start) as i64))
        },
        "compress" / * = |_| Err(crate::PyError::not_implemented("zlib compression is unavailable")),
        "decompress" / * = |_| Err(crate::PyError::not_implemented("zlib decompression is unavailable")),
        "compressobj" / * = |_| Err(crate::PyError::not_implemented("zlib compression is unavailable")),
        "decompressobj" / * = |_| Err(crate::PyError::not_implemented("zlib decompression is unavailable")),
    },
}
