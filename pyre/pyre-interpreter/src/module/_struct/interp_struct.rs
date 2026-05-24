//! _struct implementation — PyPy: pypy/module/struct/interp_struct.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// `_struct` C-extension stub — PyPy: pypy/module/struct/interp_struct.py.
///
/// Implements just enough to let `struct.py` load: `pack`, `unpack`,
/// `calcsize`, `_clearcache`, and the `error` type. Each packer handles
/// the format codes pyre actually uses during import (`<q`, `<d`, etc.).
pub fn register_module(ns: &mut DictStorage) {
    fn parse_format(fmt: &str) -> (char, Vec<char>) {
        // Returns (byte_order, codes).
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
            'h' | 'H' => 2,
            'i' | 'I' | 'l' | 'L' | 'f' => 4,
            'q' | 'Q' | 'd' | 'n' | 'N' => 8,
            'e' => 2,
            _ => 0,
        }
    }
    crate::dict_storage_store(
        ns,
        "_clearcache",
        crate::make_builtin_function_with_arity("_clearcache", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "calcsize",
        crate::make_builtin_function_with_arity(
            "calcsize",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let fmt = unsafe {
                    if pyre_object::is_str(args[0]) {
                        pyre_object::w_str_get_value(args[0]).to_string()
                    } else if pyre_object::bytesobject::is_bytes_like(args[0]) {
                        let data = pyre_object::bytesobject::bytes_like_data(args[0]);
                        String::from_utf8_lossy(data).into_owned()
                    } else {
                        return Err(crate::PyError::type_error("calcsize: format must be str"));
                    }
                };
                let (_, codes) = parse_format(&fmt);
                let total: usize = codes.iter().copied().map(code_size).sum();
                Ok(pyre_object::w_int_new(total as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pack",
        crate::make_builtin_function("pack", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_bytes_from_bytes(&[]));
            }
            let fmt = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Err(crate::PyError::type_error("pack: format must be str"));
                }
            };
            let (endian, codes) = parse_format(&fmt);
            let little = matches!(endian, '<' | '=' | '@');
            let mut out = Vec::new();
            for (i, code) in codes.iter().enumerate() {
                let arg = args.get(i + 1).copied().unwrap_or(pyre_object::w_none());
                match *code {
                    'b' | 'B' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i8;
                        out.push(v as u8);
                    }
                    'h' | 'H' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i16;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'i' | 'I' | 'l' | 'L' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i32;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'q' | 'Q' | 'n' | 'N' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'f' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg) as f32
                            } else {
                                pyre_object::w_int_get_value(arg) as f32
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'd' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg)
                            } else {
                                pyre_object::w_int_get_value(arg) as f64
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    _ => {}
                }
            }
            Ok(pyre_object::w_bytes_from_bytes(&out))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unpack",
        crate::make_builtin_function_with_arity(
            "unpack",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("unpack requires (fmt, buffer)"));
                }
                let fmt = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                let buf = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "unpack: buffer must be bytes-like",
                        ));
                    }
                };
                let (endian, codes) = parse_format(&fmt);
                let little = matches!(endian, '<' | '=' | '@');
                let mut out = Vec::new();
                let mut pos = 0usize;
                for code in codes {
                    match code {
                        'b' | 'B' => {
                            if pos >= buf.len() {
                                break;
                            }
                            out.push(pyre_object::w_int_new(buf[pos] as i8 as i64));
                            pos += 1;
                        }
                        'h' | 'H' => {
                            if pos + 2 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1]];
                            let v = if little {
                                i16::from_le_bytes(chunk)
                            } else {
                                i16::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 2;
                        }
                        'i' | 'I' | 'l' | 'L' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                i32::from_le_bytes(chunk)
                            } else {
                                i32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 4;
                        }
                        'q' | 'Q' | 'n' | 'N' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                i64::from_le_bytes(chunk)
                            } else {
                                i64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v));
                            pos += 8;
                        }
                        'f' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                f32::from_le_bytes(chunk)
                            } else {
                                f32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v as f64));
                            pos += 4;
                        }
                        'd' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                f64::from_le_bytes(chunk)
                            } else {
                                f64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v));
                            pos += 8;
                        }
                        _ => {}
                    }
                }
                Ok(pyre_object::w_tuple_new(out))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "unpack_from",
        crate::make_builtin_function("unpack_from", |_| Ok(pyre_object::w_tuple_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "iter_unpack",
        crate::make_builtin_function_with_arity(
            "iter_unpack",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // Struct class — minimal constructor returning instance with format
    // attribute. Used by struct.Struct(fmt).pack/unpack.
    crate::dict_storage_store(
        ns,
        "Struct",
        crate::make_builtin_function_with_arity(
            "Struct",
            |args| {
                let fmt = args.first().copied().unwrap_or(pyre_object::w_str_new(""));
                let obj = pyre_object::w_instance_new(crate::typedef::w_object());
                let _ = crate::baseobjspace::setattr(obj, "format", fmt);
                Ok(obj)
            },
            1,
        ),
    );
}
