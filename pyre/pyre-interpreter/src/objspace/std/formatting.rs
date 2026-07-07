//! pypy/objspace/std/formatting.py — printf-style string formatting.
#![allow(non_camel_case_types, non_snake_case)]

use malachite_bigint::BigInt;
use rustpython_common::cformat::{
    CConversionFlags, CFormatConversion, CFormatPart, CFormatPrecision, CFormatQuantity,
    CFormatSpec, CFormatSpecKeyed, CFormatType, CFormatWtf8, CNumberType,
};

use crate::objspace::descroperation::{int_value, is_int_like};
use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// `str % args` — printf-style string formatting.
///
/// The format string is parsed by `rustpython_common::cformat` into a
/// sequence of literal / conversion-spec parts; pyre supplies the value
/// glue (fetching arguments from the tuple / mapping and coercing each
/// `W_Root` to the number/float/str the spec's formatter consumes).
///
/// Argument dispatch mirrors CPython's `getnextarg`: a non-tuple right
/// operand provides a single positional value, a keyed spec (`%(k)s`)
/// looks the value up in the operand as a mapping and consumes one
/// positional slot if any remains, and surplus positional values are an
/// error only when the operand is not itself a mapping.
pub(crate) unsafe fn str_format_percent(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    let fmt_str = w_str_get_wtf8(fmt);
    let format = CFormatWtf8::parse_from_wtf8(fmt_str)
        .map_err(|err| PyError::value_error(err.to_string()))?;

    // `unicodeobject.c PyUnicode_Format` — the operand is usable as a
    // mapping (for `%(key)s` lookups) when it exposes `__getitem__` and is
    // neither a tuple nor a str. A tuple supplies positional values in
    // order; any other operand is the single positional value.
    let args_is_tuple = is_tuple(args);
    let dict = if !args_is_tuple && !is_str(args) && has_getitem(args) {
        Some(args)
    } else {
        None
    };
    let positional: Vec<PyObjectRef> = if args_is_tuple {
        let n = w_tuple_len(args);
        (0..n)
            .filter_map(|i| w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };
    let mut pos = positional.into_iter().peekable();

    let mut result = Wtf8Buf::new();
    let mut saw_specifier = false;

    for (idx, part) in format {
        match part {
            CFormatPart::Literal(literal) => result.push_wtf8(&literal),
            CFormatPart::Spec(CFormatSpecKeyed {
                mapping_key,
                mut spec,
            }) => {
                saw_specifier = true;
                let value = if let Some(key) = mapping_key {
                    let Some(dict) = dict else {
                        return Err(PyError::type_error("format requires a mapping"));
                    };
                    let w_value = crate::baseobjspace::getitem(dict, w_str_from_wtf8(key))?;
                    // A keyed spec still consumes a positional slot when one
                    // is available (`%(k)s %s` leaves nothing for the `%s`).
                    let _ = pos.next();
                    w_value
                } else {
                    update_quantity_from_tuple(
                        &mut pos,
                        &mut spec.min_field_width,
                        &mut spec.flags,
                    )?;
                    update_precision_from_tuple(&mut pos, &mut spec.precision)?;
                    let Some(v) = pos.next() else {
                        return Err(PyError::type_error(
                            "not enough arguments for format string",
                        ));
                    };
                    v
                };
                result.push_wtf8(&spec_format_string(&spec, value, idx)?);
            }
        }
    }

    // `checkconsumed` — surplus positional values are converted to an error
    // only when the operand is not a mapping. With no specifiers at all, an
    // empty tuple / a mapping is allowed but any other non-empty operand is
    // surplus.
    let surplus = if saw_specifier {
        pos.peek().is_some()
    } else {
        !(args_is_tuple && w_tuple_len(args) == 0)
    };
    if dict.is_none() && surplus {
        return Err(PyError::type_error(
            "not all arguments converted during string formatting",
        ));
    }

    Ok(w_str_from_wtf8(result))
}

/// True when `obj`'s type carries `__getitem__` (`PyMapping_Check`), so a
/// `%(key)s` spec can index it.
unsafe fn has_getitem(obj: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj) {
        Some(tp) => crate::baseobjspace::lookup_in_type(tp, "__getitem__").is_some(),
        None => false,
    }
}

/// Apply a parsed spec to one argument, producing the formatted fragment.
/// `formatting.py fmt_s / fmt_d / fmt_f / ...` — the per-conversion value
/// coercion and formatting.
unsafe fn spec_format_string(
    spec: &CFormatSpec,
    obj: PyObjectRef,
    idx: usize,
) -> Result<Wtf8Buf, PyError> {
    match &spec.format_type {
        CFormatType::String(conversion) => {
            let result = match conversion {
                CFormatConversion::Str => crate::py_str_wtf8(obj)?,
                CFormatConversion::Repr => crate::py_repr_wtf8(obj)?,
                CFormatConversion::Ascii => Wtf8Buf::from_string(crate::builtins::py_ascii(obj)?),
                // `%b` is a bytes-only conversion; the unicode formatter
                // rejects it (`fmt_b` → `unknown_fmtchar`). `idx` is the
                // position of the `%`, the message reports the `b`.
                CFormatConversion::Bytes => {
                    return Err(PyError::value_error(format!(
                        "unsupported format character 'b' (0x62) at index {}",
                        idx + 1
                    )));
                }
            };
            Ok(spec.format_string(result))
        }
        CFormatType::Number(number_type) => {
            let value = match number_type {
                CNumberType::DecimalD | CNumberType::DecimalI | CNumberType::DecimalU => {
                    number_arg_decimal(spec, obj)?
                }
                _ => number_arg_integer(spec, obj)?,
            };
            Ok(Wtf8Buf::from_string(spec.format_number(&value)))
        }
        CFormatType::Float(_) => {
            let value = crate::baseobjspace::float_w(obj)?;
            Ok(Wtf8Buf::from_string(spec.format_float(value)))
        }
        CFormatType::Character(_) => Ok(spec.format_char(char_arg(obj)?)),
    }
}

/// BigInt from an `int` / `bool` / `long`.
unsafe fn arg_to_bigint(obj: PyObjectRef) -> BigInt {
    if is_bool(obj) {
        BigInt::from(w_bool_get_value(obj) as i64)
    } else if is_int(obj) {
        BigInt::from(w_int_get_value(obj))
    } else {
        w_long_get_value(obj).clone()
    }
}

/// `fmt_d / fmt_i / fmt_u` argument coercion — `%d`/`%i`/`%u` accept any
/// integer, a float (truncated), or an object with `__index__` / `__int__`.
unsafe fn number_arg_decimal(spec: &CFormatSpec, obj: PyObjectRef) -> Result<BigInt, PyError> {
    if is_int_like(obj) || is_long(obj) {
        return Ok(arg_to_bigint(obj));
    }
    if is_float(obj) {
        let pyint = crate::typedef::float_to_pyint(
            floatobject::w_float_get_value(obj),
            crate::typedef::FloatToIntMode::Trunc,
        )?;
        return Ok(arg_to_bigint(pyint));
    }
    if has_dunder(obj, "__index__") {
        return Ok(crate::builtins::obj_to_bigint(
            crate::baseobjspace::space_index(obj)?,
        ));
    }
    if let Some(method) = crate::baseobjspace::lookup(obj, "__int__") {
        let r = crate::builtins::call_and_check(method, &[obj])?;
        if is_int_like(r) || is_long(r) {
            return Ok(arg_to_bigint(r));
        }
    }
    Err(number_type_error(spec, obj, "a real number is required"))
}

/// `fmt_x / fmt_X / fmt_o` argument coercion — the radix conversions accept
/// an integer or an `__index__` object, but not a float.
unsafe fn number_arg_integer(spec: &CFormatSpec, obj: PyObjectRef) -> Result<BigInt, PyError> {
    if is_int_like(obj) || is_long(obj) {
        return Ok(arg_to_bigint(obj));
    }
    if has_dunder(obj, "__index__") {
        return Ok(crate::builtins::obj_to_bigint(
            crate::baseobjspace::space_index(obj)?,
        ));
    }
    Err(number_type_error(spec, obj, "an integer is required"))
}

/// `%{c} format: {what}, not {type}` for a non-numeric argument.
unsafe fn number_type_error(spec: &CFormatSpec, obj: PyObjectRef, what: &str) -> PyError {
    PyError::type_error(format!(
        "%{} format: {what}, not {}",
        spec.format_type.to_char(),
        crate::baseobjspace::object_functionstr_type_name(obj),
    ))
}

/// `fmt_c` argument coercion — a single-character str, or an integer /
/// `__index__` in `range(0x110000)`.
unsafe fn char_arg(obj: PyObjectRef) -> Result<CodePoint, PyError> {
    if is_str(obj) {
        let s = w_str_get_wtf8(obj);
        let mut cps = s.code_points();
        if let Some(cp) = cps.next() {
            if cps.next().is_none() {
                return Ok(cp);
            }
        }
        let n = s.code_points().count();
        return Err(PyError::type_error(format!(
            "%c requires an int or a unicode character, not a string of length {n}"
        )));
    }
    let value = if is_int_like(obj) || is_long(obj) {
        arg_to_bigint(obj)
    } else if has_dunder(obj, "__index__") {
        crate::builtins::obj_to_bigint(crate::baseobjspace::space_index(obj)?)
    } else {
        return Err(PyError::type_error(format!(
            "%c requires an int or a unicode character, not {}",
            crate::baseobjspace::object_functionstr_type_name(obj),
        )));
    };
    let overflow = || {
        PyError::new(
            PyErrorKind::OverflowError,
            "%c arg not in range(0x110000)".to_string(),
        )
    };
    let n = u32::try_from(&value).map_err(|_| overflow())?;
    CodePoint::from_u32(n).ok_or_else(overflow)
}

/// True when `obj`'s type carries `name` above `object`'s default.
unsafe fn has_dunder(obj: PyObjectRef, name: &str) -> bool {
    match crate::typedef::r#type(obj) {
        Some(tp) => crate::baseobjspace::lookup_in_type(tp, name).is_some(),
        None => false,
    }
}

/// `peel_num` — a `*` field width reads its value (and, when negative, the
/// left-align flag) from the next positional argument.
unsafe fn update_quantity_from_tuple(
    pos: &mut std::iter::Peekable<std::vec::IntoIter<PyObjectRef>>,
    quantity: &mut Option<CFormatQuantity>,
    flags: &mut CConversionFlags,
) -> Result<(), PyError> {
    if !matches!(quantity, Some(CFormatQuantity::FromValuesTuple)) {
        return Ok(());
    }
    let v = star_int(pos.next())?;
    if v < 0 {
        flags.insert(CConversionFlags::LEFT_ADJUST);
    }
    *quantity = Some(CFormatQuantity::Amount(v.unsigned_abs() as usize));
    Ok(())
}

/// `peel_num` — a `*` precision reads its value from the next positional
/// argument (a negative precision is treated as absent).
unsafe fn update_precision_from_tuple(
    pos: &mut std::iter::Peekable<std::vec::IntoIter<PyObjectRef>>,
    precision: &mut Option<CFormatPrecision>,
) -> Result<(), PyError> {
    if !matches!(
        precision,
        Some(CFormatPrecision::Quantity(CFormatQuantity::FromValuesTuple))
    ) {
        return Ok(());
    }
    let v = star_int(pos.next())?;
    *precision = Some(CFormatPrecision::Quantity(CFormatQuantity::Amount(
        v.max(0) as usize,
    )));
    Ok(())
}

/// The `*` argument must be an int; consume it, matching `nextinputvalue`.
unsafe fn star_int(arg: Option<PyObjectRef>) -> Result<i64, PyError> {
    let Some(arg) = arg else {
        return Err(PyError::type_error(
            "not enough arguments for format string",
        ));
    };
    if !is_int_like(arg) {
        return Err(PyError::type_error("* wants int"));
    }
    Ok(int_value(arg))
}
