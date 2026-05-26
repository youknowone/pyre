//! _struct module — PyPy: `pypy/module/struct/`.
//!
//! Implements just enough of `pack` / `unpack` / `calcsize` /
//! `_clearcache` plus the `error` type alias to let `struct.py` load.
//! Each packer handles the format codes pyre actually uses during
//! import (`<q`, `<d`, etc.).

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

crate::py_module! {
    "_struct",
    interpleveldefs: {
        "error" => crate::typedef::w_object(),
    },
    inline_functions: {
        fn _clearcache() {}
        fn calcsize(fmt_obj: PyObjectRef) -> Result<i64, crate::PyError> {
            let fmt = format_to_string(fmt_obj)?;
            let (_, codes) = parse_format(&fmt);
            Ok(codes.iter().copied().map(code_size).sum::<usize>() as i64)
        }
        fn unpack(fmt: &str, buf: &[u8]) -> PyObjectRef {
            let (endian, codes) = parse_format(fmt);
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
                w_str_get_value(args[0]).to_string()
            };
            let (endian, codes) = parse_format(&fmt);
            let little = matches!(endian, '<' | '=' | '@');
            let mut out = Vec::new();
            for (i, code) in codes.iter().enumerate() {
                let arg = args.get(i + 1).copied().unwrap_or(w_none());
                pack_into(&mut out, *code, little, arg);
            }
            Ok(w_bytes_from_bytes(&out))
        },
        "unpack_from" / * = |_| Ok(w_tuple_new(vec![])),
        "iter_unpack" / 2 = |_| Ok(w_list_new(vec![])),
        // `struct.Struct(fmt)` — minimal instance with `format` attribute.
        "Struct" / 1 = |args| {
            let fmt = args.first().copied().unwrap_or(w_str_new(""));
            let obj = w_instance_new(crate::typedef::w_object());
            let _ = crate::baseobjspace::setattr(obj, "format", fmt);
            Ok(obj)
        },
    },
}
