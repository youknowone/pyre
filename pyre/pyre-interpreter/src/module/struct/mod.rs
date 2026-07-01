//! _struct module — PyPy: `pypy/module/struct/`.
//!
//! Implements just enough of `pack` / `unpack` / `calcsize` /
//! `_clearcache` plus the `error` type alias to let `struct.py` load,
//! and the `W_Struct` class (`interp_struct.py:213 W_Struct`) whose
//! `pack` / `unpack` promote the format string by value before each
//! call (`jit.promote_string(self.format)`).  Each packer handles the
//! format codes pyre actually uses during import (`<q`, `<d`, etc.).

use pyre_object::*;

fn parse_format(fmt: &str) -> (char, Vec<char>) {
    let chars = fmt.chars();
    let first = chars.clone().next().unwrap_or('@');
    let (endian, rest) = if matches!(first, '<' | '>' | '!' | '=' | '@') {
        (first, chars.skip(1).collect::<String>())
    } else {
        ('@', fmt.to_string())
    };
    (
        endian,
        rest.chars().filter(|c| !c.is_ascii_whitespace()).collect(),
    )
}

fn code_size(c: char) -> usize {
    match c {
        'b' | 'B' | 'c' | '?' | 'x' => 1,
        'h' | 'H' | 'e' => 2,
        'i' | 'I' | 'l' | 'L' | 'f' => 4,
        'q' | 'Q' | 'd' | 'n' | 'N' => 8,
        _ => 0,
    }
}

/// Accept str or bytes-like format spec (PyPy `calcsize` /
/// `_clearcache` parity) and surface as `String`.
fn format_to_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if is_str(obj) {
            // A lone surrogate is never a valid format character; read via
            // WTF-8 and degrade through the codec's `backslashreplace`
            // handler rather than panicking in `w_str_get_value`.
            let w = w_str_get_wtf8(obj).to_wtf8_buf();
            if let Ok(s) = w.as_str() {
                return Ok(s.to_string());
            }
            let s_obj = w_str_from_wtf8(w);
            let bytes = crate::type_methods::encode_object(s_obj, "utf-8", "backslashreplace")?;
            Ok(String::from_utf8(bytes).unwrap_or_default())
        } else if bytesobject::is_bytes_like(obj) {
            Ok(String::from_utf8_lossy(bytesobject::bytes_like_data(obj)).into_owned())
        } else {
            Err(crate::PyError::type_error("format must be str or bytes"))
        }
    }
}

/// Low 64 bits (two's-complement) of a Python int, whether it arrives as a
/// small `W_IntObject` or a `W_LongObject`.  Narrower codes truncate this via
/// `as u8`/`as u16`/`as u32`, which keeps the correct low bytes for both
/// signed and unsigned formats.  Unsigned `Q`/`N` above `i64::MAX` reach here
/// as a `W_LongObject`, so a plain `w_int_get_value` would be wrong.
unsafe fn int_pack_bits(arg: PyObjectRef) -> u64 {
    use num_traits::ToPrimitive;
    unsafe {
        if is_long(arg) {
            let big = longobject::w_long_get_value(arg);
            big.to_u64()
                .or_else(|| big.to_i64().map(|v| v as u64))
                .unwrap_or(0)
        } else {
            w_int_get_value(arg) as u64
        }
    }
}

fn pack_into(out: &mut Vec<u8>, code: char, little: bool, arg: PyObjectRef) {
    match code {
        'b' | 'B' => {
            out.push(unsafe { int_pack_bits(arg) } as u8);
        }
        'h' | 'H' => {
            let v = unsafe { int_pack_bits(arg) } as u16;
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'i' | 'I' | 'l' | 'L' => {
            let v = unsafe { int_pack_bits(arg) } as u32;
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'q' | 'Q' | 'n' | 'N' => {
            let v = unsafe { int_pack_bits(arg) };
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'f' => {
            let v = unsafe {
                if is_float(arg) {
                    w_float_get_value(arg) as f32
                } else {
                    w_int_get_value(arg) as f32
                }
            };
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'd' => {
            let v = unsafe {
                if is_float(arg) {
                    w_float_get_value(arg)
                } else {
                    w_int_get_value(arg) as f64
                }
            };
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        _ => {}
    }
}

fn unpack_one(buf: &[u8], pos: &mut usize, code: char, little: bool) -> Option<PyObjectRef> {
    macro_rules! take {
        ($n:expr) => {
            if *pos + $n > buf.len() {
                return None;
            } else {
                let slice = &buf[*pos..*pos + $n];
                *pos += $n;
                slice
            }
        };
    }
    // Endian-aware fixed-width read: `<T>::from_le_bytes` / `from_be_bytes`.
    macro_rules! rd {
        ($ty:ty, $n:expr) => {{
            let b: [u8; $n] = take!($n).try_into().unwrap();
            if little {
                <$ty>::from_le_bytes(b)
            } else {
                <$ty>::from_be_bytes(b)
            }
        }};
    }
    // Uppercase codes are unsigned, lowercase signed
    // (`interp_array.py unpack_value` — the `b'B'`/`b'H'`/`b'I'`/`b'Q'`
    // rows box unsigned values, boxing `Q`/`N` above `i64::MAX` into a
    // `W_LongObject`).
    match code {
        'b' => Some(w_int_new(take!(1)[0] as i8 as i64)),
        'B' => Some(w_int_new(take!(1)[0] as i64)),
        'h' => Some(w_int_new(rd!(i16, 2) as i64)),
        'H' => Some(w_int_new(rd!(u16, 2) as i64)),
        'i' | 'l' => Some(w_int_new(rd!(i32, 4) as i64)),
        'I' | 'L' => Some(w_int_new(rd!(u32, 4) as i64)),
        'q' | 'n' => Some(w_int_new(rd!(i64, 8))),
        'Q' | 'N' => {
            let v = rd!(u64, 8);
            if v <= i64::MAX as u64 {
                Some(w_int_new(v as i64))
            } else {
                Some(pyre_object::longobject::w_long_new(
                    malachite_bigint::BigInt::from(v),
                ))
            }
        }
        'f' => Some(w_float_new(rd!(f32, 4) as f64)),
        'd' => Some(w_float_new(rd!(f64, 8))),
        _ => None,
    }
}

/// `interp_struct.py:71 do_pack` — pack `values` according to `format`.
fn do_pack(format: &str, values: &[PyObjectRef]) -> PyObjectRef {
    let (endian, codes) = parse_format(format);
    let little = matches!(endian, '<' | '=' | '@');
    let mut out = Vec::new();
    for (i, code) in codes.iter().enumerate() {
        let arg = values.get(i).copied().unwrap_or(w_none());
        pack_into(&mut out, *code, little, arg);
    }
    w_bytes_from_bytes(&out)
}

/// `interp_struct.py:139 do_unpack` — unpack `buf` according to `format`.
fn do_unpack(format: &str, buf: &[u8]) -> PyObjectRef {
    let (endian, codes) = parse_format(format);
    let little = matches!(endian, '<' | '=' | '@');
    let mut out = Vec::new();
    let mut pos = 0usize;
    for code in codes {
        match unpack_one(buf, &mut pos, code, little) {
            Some(v) => out.push(v),
            None => break,
        }
    }
    w_tuple_new(out)
}

/// `interp_struct.py:213 W_Struct` — a compiled struct object holding its
/// format string and precomputed size.
#[crate::pyre_class("_struct.Struct")]
pub struct W_Struct {
    /// Format string object (`text_or_bytes_w` of the constructor arg),
    /// promoted by value before each pack/unpack.  `_immutable_fields_ =
    /// ["format", "size"]`.
    format: PyObjectRef,
    size: i64,
}

#[crate::pyre_methods(doc = "Struct(fmt) --> compiled struct object")]
impl W_Struct {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Struct::allocate(W_Struct {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            format: w_str_new(""),
            size: -1,
        })
    }

    /// `interp_struct.py:222 descr__init__` — store the (normalized)
    /// format string and its `_calcsize`.
    fn __init__(&mut self, w_format: PyObjectRef) -> Result<(), crate::PyError> {
        let format = format_to_string(w_format)?;
        let (_, codes) = parse_format(&format);
        self.size = codes.iter().copied().map(code_size).sum::<usize>() as i64;
        self.format = w_str_new(&format);
        Ok(())
    }

    #[getter]
    fn format(&self) -> PyObjectRef {
        self.format
    }

    #[getter]
    fn size(&self) -> i64 {
        self.size
    }

    /// `interp_struct.py:227 descr_pack` —
    /// `do_pack(space, jit.promote_string(self.format), args_w)`.
    /// The whole-args-slice ABI hands `args[0]` = self; the packed values
    /// are `args[1..]`.
    fn pack(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        Ok(do_pack(fmt, &args[1..]))
    }

    /// `interp_struct.py:234 descr_unpack` —
    /// `do_unpack(space, jit.promote_string(self.format), w_str)`.
    fn unpack(&self, w_str: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        let buf = unsafe {
            if bytesobject::is_bytes_like(w_str) {
                bytesobject::bytes_like_data(w_str)
            } else {
                return Err(crate::PyError::type_error(
                    "a bytes-like object is required",
                ));
            }
        };
        Ok(do_unpack(fmt, buf))
    }
}

crate::py_module! {
    "_struct",
    interpleveldefs: {
        "error" => crate::typedef::w_object(),
        "Struct" => type_object(),
    },
    inline_functions: {
        fn _clearcache() {}
        fn calcsize(fmt_obj: PyObjectRef) -> Result<i64, crate::PyError> {
            let fmt = format_to_string(fmt_obj)?;
            let (_, codes) = parse_format(&fmt);
            Ok(codes.iter().copied().map(code_size).sum::<usize>() as i64)
        }
        fn unpack(fmt: &str, buf: &[u8]) -> PyObjectRef {
            do_unpack(fmt, buf)
        }
    },
    functions: {
        // `pack(fmt, *args)` — variadic positional after fmt; route
        // through the args slice for now (typed varargs not supported
        // by inline_functions arity inference).
        "pack" / * = |args| {
            if args.is_empty() {
                return Ok(w_bytes_from_bytes(&[]));
            }
            let fmt = unsafe {
                if !is_str(args[0]) {
                    return Err(crate::PyError::type_error("pack: format must be str"));
                }
                w_str_get_value(args[0])
            };
            Ok(do_pack(fmt, &args[1..]))
        },
        "unpack_from" / * = |_| Ok(w_tuple_new(vec![])),
        "iter_unpack" / 2 = |_| Ok(w_list_new(vec![])),
    },
}
