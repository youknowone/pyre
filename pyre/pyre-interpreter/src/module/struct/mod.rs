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
            Ok(w_str_get_value(obj).to_string())
        } else if bytesobject::is_bytes_like(obj) {
            Ok(String::from_utf8_lossy(bytesobject::bytes_like_data(obj)).into_owned())
        } else {
            Err(crate::PyError::type_error("format must be str or bytes"))
        }
    }
}

fn pack_into(out: &mut Vec<u8>, code: char, little: bool, arg: PyObjectRef) {
    match code {
        'b' | 'B' => {
            out.push(unsafe { w_int_get_value(arg) } as i8 as u8);
        }
        'h' | 'H' => {
            let v = unsafe { w_int_get_value(arg) } as i16;
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'i' | 'I' | 'l' | 'L' => {
            let v = unsafe { w_int_get_value(arg) } as i32;
            out.extend_from_slice(&if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        'q' | 'Q' | 'n' | 'N' => {
            let v = unsafe { w_int_get_value(arg) };
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
    match code {
        'b' | 'B' => {
            let b = take!(1);
            Some(w_int_new(b[0] as i8 as i64))
        }
        'h' | 'H' => {
            let b: [u8; 2] = take!(2).try_into().unwrap();
            let v = if little {
                i16::from_le_bytes(b)
            } else {
                i16::from_be_bytes(b)
            };
            Some(w_int_new(v as i64))
        }
        'i' | 'I' | 'l' | 'L' => {
            let b: [u8; 4] = take!(4).try_into().unwrap();
            let v = if little {
                i32::from_le_bytes(b)
            } else {
                i32::from_be_bytes(b)
            };
            Some(w_int_new(v as i64))
        }
        'q' | 'Q' | 'n' | 'N' => {
            let b: [u8; 8] = take!(8).try_into().unwrap();
            let v = if little {
                i64::from_le_bytes(b)
            } else {
                i64::from_be_bytes(b)
            };
            Some(w_int_new(v))
        }
        'f' => {
            let b: [u8; 4] = take!(4).try_into().unwrap();
            let v = if little {
                f32::from_le_bytes(b)
            } else {
                f32::from_be_bytes(b)
            };
            Some(w_float_new(v as f64))
        }
        'd' => {
            let b: [u8; 8] = take!(8).try_into().unwrap();
            let v = if little {
                f64::from_le_bytes(b)
            } else {
                f64::from_be_bytes(b)
            };
            Some(w_float_new(v))
        }
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
