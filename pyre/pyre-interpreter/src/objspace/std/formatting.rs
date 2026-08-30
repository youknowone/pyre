//! pypy/objspace/std/formatting.py — printf-style string formatting.
#![allow(non_camel_case_types, non_snake_case)]

use majit_rlib::rbigint::RBigInt as BigInt;
use num_traits::ToPrimitive;
use rustpython_common::cformat::{
    CCharacterType, CConversionFlags, CFormatBytes, CFormatConversion, CFormatError,
    CFormatErrorType, CFormatPart, CFormatPrecision, CFormatQuantity, CFormatSpec,
    CFormatSpecKeyed, CFormatType, CFormatWtf8, CNumberType,
};

use crate::objspace::descroperation::{int_value, is_int_like};
use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

#[derive(Clone, Copy)]
enum DeferredPercentError {
    Unsupported(CFormatError),
    Incomplete(CFormatError),
    IncompleteMappingKey(CFormatError),
    Quantity(CFormatError),
}

impl DeferredPercentError {
    fn unsupported(self) -> Option<CFormatError> {
        match self {
            Self::Unsupported(error) => Some(error),
            _ => None,
        }
    }

    fn before_conversion(self) -> Option<CFormatError> {
        match self {
            Self::Incomplete(error) | Self::Quantity(error) => Some(error),
            _ => None,
        }
    }

    fn after_parts(self, is_mapping: bool) -> Result<(), PyError> {
        let error = match self {
            Self::Unsupported(_) => {
                unreachable!("the recovered unsupported spec raises after operand acquisition")
            }
            Self::Incomplete(_) | Self::Quantity(_) => {
                unreachable!("the recovered spec raises before conversion acquisition")
            }
            Self::IncompleteMappingKey(_) if !is_mapping => {
                return Err(PyError::type_error("format requires a mapping"));
            }
            Self::IncompleteMappingKey(error) => error,
        };
        Err(PyError::value_error(error.to_string()))
    }
}

fn is_format_char(codepoint: CodePoint, expected: char) -> bool {
    codepoint.to_u32() == expected as u32
}

fn wtf8_prefix(fmt: &Wtf8, codepoints: usize) -> Wtf8Buf {
    let mut prefix = Wtf8Buf::new();
    for codepoint in fmt.code_points().take(codepoints) {
        prefix.push(codepoint);
    }
    prefix
}

fn wtf8_acquisition_prefix(
    fmt: &Wtf8,
    error: CFormatError,
    search_end: usize,
    include_precision_star: bool,
) -> Wtf8Buf {
    let codepoints: Vec<_> = fmt.code_points().collect();
    let spec_start = (0..search_end)
        .filter(|&index| is_format_char(codepoints[index], '%'))
        .filter(|&index| {
            let prefix = wtf8_prefix(fmt, index + 1);
            matches!(
                CFormatWtf8::parse_from_wtf8(&prefix),
                Err(CFormatError {
                    typ: CFormatErrorType::IncompleteFormat,
                    index: error_index,
                }) if error_index == index + 1
            )
        })
        .next_back()
        .expect("a width or precision error belongs to a conversion spec");

    let mut recovered = Wtf8Buf::new();
    recovered.extend(codepoints[..=spec_start].iter().copied());
    let mut cursor = spec_start + 1;
    if codepoints
        .get(cursor)
        .is_some_and(|&c| is_format_char(c, '('))
    {
        let mapping_start = cursor;
        let mut nesting = 1;
        cursor += 1;
        while nesting != 0 {
            let codepoint = codepoints[cursor];
            cursor += 1;
            if is_format_char(codepoint, '(') {
                nesting += 1;
            } else if is_format_char(codepoint, ')') {
                nesting -= 1;
            }
        }
        recovered.extend(codepoints[mapping_start..cursor].iter().copied());
    }
    while codepoints.get(cursor).is_some_and(|&c| {
        ['#', '0', '-', ' ', '+']
            .into_iter()
            .any(|flag| is_format_char(c, flag))
    }) {
        cursor += 1;
    }
    if !matches!(error.typ, CFormatErrorType::WidthTooBig)
        && codepoints
            .get(cursor)
            .is_some_and(|&c| is_format_char(c, '*'))
    {
        recovered.push_char('*');
        cursor += 1;
    } else {
        while codepoints
            .get(cursor)
            .is_some_and(|c| c.to_u32() >= '0' as u32 && c.to_u32() <= '9' as u32)
        {
            cursor += 1;
        }
    }
    if include_precision_star
        && codepoints
            .get(cursor)
            .is_some_and(|&c| is_format_char(c, '.'))
        && codepoints
            .get(cursor + 1)
            .is_some_and(|&c| is_format_char(c, '*'))
    {
        recovered.push_char('.');
        recovered.push_char('*');
    }
    recovered.push_char('s');
    recovered
}

fn bytes_acquisition_prefix(
    fmt: &[u8],
    error: CFormatError,
    search_end: usize,
    include_precision_star: bool,
) -> Vec<u8> {
    let spec_start = (0..search_end)
        .filter(|&index| fmt[index] == b'%')
        .filter(|&index| {
            matches!(
                CFormatBytes::parse_from_bytes(&fmt[..=index]),
                Err(CFormatError {
                    typ: CFormatErrorType::IncompleteFormat,
                    index: error_index,
                }) if error_index == index + 1
            )
        })
        .next_back()
        .expect("a width or precision error belongs to a conversion spec");

    let mut recovered = fmt[..=spec_start].to_vec();
    let mut cursor = spec_start + 1;
    if fmt.get(cursor) == Some(&b'(') {
        let mapping_start = cursor;
        let mut nesting = 1;
        cursor += 1;
        while nesting != 0 {
            match fmt[cursor] {
                b'(' => nesting += 1,
                b')' => nesting -= 1,
                _ => {}
            }
            cursor += 1;
        }
        recovered.extend_from_slice(&fmt[mapping_start..cursor]);
    }
    while fmt.get(cursor).is_some_and(|c| b"#0- +".contains(c)) {
        cursor += 1;
    }
    if !matches!(error.typ, CFormatErrorType::WidthTooBig) && fmt.get(cursor) == Some(&b'*') {
        recovered.push(b'*');
        cursor += 1;
    } else {
        while fmt.get(cursor).is_some_and(u8::is_ascii_digit) {
            cursor += 1;
        }
    }
    if include_precision_star
        && fmt.get(cursor) == Some(&b'.')
        && fmt.get(cursor + 1) == Some(&b'*')
    {
        recovered.extend_from_slice(b".*");
    }
    recovered.push(b's');
    recovered
}

/// Parse a unicode percent format, retaining the first deferred parser error.
///
/// PyPy's `StringFormatter.format` parses one spec at a time: `parse_fmt`
/// performs mapping lookup and consumes `*` operands, then the loop validates
/// the conversion character. CPython 3.14 additionally consumes the conversion
/// operand before reporting an unsupported character. The shared RustPython
/// parser instead validates the entire format eagerly, which used to report
/// the `ValueError` before either upstream's operand-side effects occurred.
///
/// Replace only that unsupported character with `s` and stop there. The caller
/// can execute every preceding spec and the recovered spec's operand-acquisition
/// path, then surface the saved 3.14 error without formatting the operand.
/// An incomplete mapping key retains only the complete prefix. Other incomplete
/// specs and oversized quantities retain a synthetic current spec through the
/// mapping lookup and the `*` operands which precede their error stage. This is
/// the `parse_fmt` acquisition order without asking for the absent conversion
/// operand.
fn parse_wtf8_incremental(
    fmt: &Wtf8,
) -> Result<(CFormatWtf8, Option<DeferredPercentError>), PyError> {
    match CFormatWtf8::parse_from_wtf8(fmt) {
        Ok(format) => Ok((format, None)),
        Err(error) => {
            if matches!(
                error.typ,
                CFormatErrorType::WidthTooBig | CFormatErrorType::PrecisionTooBig
            ) {
                let recovered = CFormatWtf8::parse_from_wtf8(&wtf8_acquisition_prefix(
                    fmt,
                    error,
                    error.index,
                    false,
                ))
                .expect("the acquisition prefix of a deferred quantity error must parse");
                return Ok((recovered, Some(DeferredPercentError::Quantity(error))));
            }
            if matches!(error.typ, CFormatErrorType::IncompleteFormat) {
                let recovered = CFormatWtf8::parse_from_wtf8(&wtf8_acquisition_prefix(
                    fmt,
                    error,
                    fmt.code_points().count(),
                    true,
                ))
                .expect("the acquisition prefix of an incomplete format must parse");
                return Ok((recovered, Some(DeferredPercentError::Incomplete(error))));
            }
            let (prefix_len, replacement, deferred) = match error.typ {
                CFormatErrorType::UnsupportedFormatChar(_) => {
                    (error.index, true, DeferredPercentError::Unsupported(error))
                }
                CFormatErrorType::UnmatchedKeyParentheses => (
                    error
                        .index
                        .checked_sub(1)
                        .expect("an incomplete mapping key follows its percent sign"),
                    false,
                    DeferredPercentError::IncompleteMappingKey(error),
                ),
                _ => return Err(PyError::value_error(error.to_string())),
            };
            let mut prefix = wtf8_prefix(fmt, prefix_len);
            if replacement {
                prefix.push_char('s');
            }
            let recovered = CFormatWtf8::parse_from_wtf8(&prefix)
                .expect("the complete prefix of a deferred percent-format error must parse");
            Ok((recovered, Some(deferred)))
        }
    }
}

/// Bytes counterpart of [`parse_wtf8_incremental`].
fn parse_bytes_incremental(
    fmt: &[u8],
) -> Result<(CFormatBytes, Option<DeferredPercentError>), PyError> {
    match CFormatBytes::parse_from_bytes(fmt) {
        Ok(format) => Ok((format, None)),
        Err(error) => {
            if matches!(
                error.typ,
                CFormatErrorType::WidthTooBig | CFormatErrorType::PrecisionTooBig
            ) {
                let recovered = CFormatBytes::parse_from_bytes(&bytes_acquisition_prefix(
                    fmt,
                    error,
                    error.index,
                    false,
                ))
                .expect("the acquisition prefix of a deferred quantity error must parse");
                return Ok((recovered, Some(DeferredPercentError::Quantity(error))));
            }
            if matches!(error.typ, CFormatErrorType::IncompleteFormat) {
                let recovered = CFormatBytes::parse_from_bytes(&bytes_acquisition_prefix(
                    fmt,
                    error,
                    fmt.len(),
                    true,
                ))
                .expect("the acquisition prefix of an incomplete format must parse");
                return Ok((recovered, Some(DeferredPercentError::Incomplete(error))));
            }
            let (prefix_len, replacement, deferred) = match error.typ {
                CFormatErrorType::UnsupportedFormatChar(_) => {
                    (error.index, true, DeferredPercentError::Unsupported(error))
                }
                CFormatErrorType::UnmatchedKeyParentheses => (
                    error
                        .index
                        .checked_sub(1)
                        .expect("an incomplete mapping key follows its percent sign"),
                    false,
                    DeferredPercentError::IncompleteMappingKey(error),
                ),
                _ => return Err(PyError::value_error(error.to_string())),
            };
            let mut prefix = fmt[..prefix_len].to_vec();
            if replacement {
                prefix.push(b's');
            }
            let recovered = CFormatBytes::parse_from_bytes(&prefix)
                .expect("the complete prefix of a deferred percent-format error must parse");
            Ok((recovered, Some(deferred)))
        }
    }
}

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
    // The conversions below run Python, and `BINARY_OP %` popped both operands
    // before dispatching here, so neither the format string nor the operand
    // is rooted by the frame any more.  A str is immobile and read directly
    // once pinned; `args` may be a dict and is read back from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let fmt = pyre_object::gc_roots::pin_root(fmt);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let args = pyre_object::gc_roots::pin_root(args);
    let fmt_str = w_str_get_wtf8(fmt);
    let (format, deferred_error) = parse_wtf8_incremental(fmt_str)?;

    // `unicodeobject.c PyUnicode_Format` — the operand is usable as a
    // mapping (for `%(key)s` lookups) when it exposes `__getitem__` and is
    // neither a tuple nor a str. A tuple supplies positional values in
    // order; any other operand is the single positional value.
    let args_is_tuple = is_tuple(args);
    let dict = if !args_is_tuple && !is_str(args) && has_getitem(args) {
        Some(args_slot)
    } else {
        None
    };
    let items: Vec<PyObjectRef> = if args_is_tuple {
        let n = w_tuple_len(args);
        (0..n)
            .filter_map(|i| w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };
    let mut pos = OperandColumn {
        base: pyre_object::gc_roots::pin_roots(&items),
        len: items.len(),
        cursor: 0,
    };

    let mut result = Wtf8Buf::new();
    let mut saw_specifier = false;

    let mut parts = format.into_iter().peekable();
    while let Some((_, part)) = parts.next() {
        match part {
            CFormatPart::Literal(literal) => result.push_wtf8(&literal),
            CFormatPart::Spec(CFormatSpecKeyed {
                mapping_key,
                mut spec,
            }) => {
                saw_specifier = true;
                let current_deferred = deferred_error.filter(|_| parts.peek().is_none());
                let value = if let Some(key) = mapping_key {
                    let Some(dict_slot) = dict else {
                        return Err(PyError::type_error("format requires a mapping"));
                    };
                    let w_value = crate::baseobjspace::getitem(
                        pyre_object::gc_roots::shadow_stack_get(dict_slot),
                        w_str_from_wtf8(key),
                    )?;
                    // A keyed spec still consumes a positional slot when one
                    // is available (`%(k)s %s` leaves nothing for the `%s`).
                    let _ = pos.next();
                    let w_value = pyre_object::gc_roots::pin_root(w_value);
                    mapping_star_operands(
                        &mut spec,
                        w_value,
                        current_deferred
                            .and_then(DeferredPercentError::before_conversion)
                            .is_none(),
                    )?;
                    if let Some(error) =
                        current_deferred.and_then(DeferredPercentError::before_conversion)
                    {
                        return Err(PyError::value_error(error.to_string()));
                    }
                    w_value
                } else {
                    update_quantity_from_tuple(
                        &mut pos,
                        &mut spec.min_field_width,
                        &mut spec.flags,
                    )?;
                    update_precision_from_tuple(&mut pos, &mut spec.precision)?;
                    if let Some(error) =
                        current_deferred.and_then(DeferredPercentError::before_conversion)
                    {
                        return Err(PyError::value_error(error.to_string()));
                    }
                    let Some(v) = pos.next() else {
                        return Err(PyError::type_error(
                            "not enough arguments for format string",
                        ));
                    };
                    v
                };
                if let Some(error) = current_deferred.and_then(DeferredPercentError::unsupported) {
                    return Err(PyError::value_error(error.to_string()));
                }
                result.push_wtf8(&spec_format_string(&spec, value)?);
            }
        }
    }

    if let Some(error) = deferred_error {
        error.after_parts(dict.is_some())?;
    }

    // `checkconsumed` — surplus positional values are converted to an error
    // only when the operand is not a mapping. With no specifiers at all, an
    // empty tuple / a mapping is allowed but any other non-empty operand is
    // surplus.
    let surplus = if saw_specifier {
        pos.has_next()
    } else {
        !(args_is_tuple && w_tuple_len(pyre_object::gc_roots::shadow_stack_get(args_slot)) == 0)
    };
    if dict.is_none() && surplus {
        return Err(PyError::type_error(
            "not all arguments converted during string formatting",
        ));
    }

    // `str % args` printf formatting is a dominant dynamic-churn producer.
    Ok(w_str_from_wtf8_managed(result))
}

pub(crate) unsafe fn bytes_format_percent(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    // CPython 3.14 bytearray_mod_lock_held increments `ob_exports` before
    // `_PyBytes_FormatEx` observes the receiver and decrements it after every
    // success/error return.  Re-entrant formatting callbacks may overwrite
    // bytes, but cannot resize and invalidate the live pointer/length pair.
    let receiver_exported = pyre_object::bytearrayobject::is_bytearray(fmt);
    if receiver_exported {
        pyre_object::bytearrayobject::w_bytearray_exports_incref(fmt);
    }
    let formatted = bytes_format_percent_inner(fmt, args);
    if receiver_exported {
        pyre_object::bytearrayobject::w_bytearray_exports_decref(fmt);
    }
    formatted
}

unsafe fn bytes_format_percent_inner(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    // As `str_format_percent`: the conversions run Python and `BINARY_OP %`
    // popped both operands, so nothing else roots them.  `fmt` is pinned for
    // the borrow below — a bytes-like object never moves, but an unreachable
    // one is still swept, and the slice points into its payload.
    let _roots = pyre_object::gc_roots::push_roots();
    let fmt = pyre_object::gc_roots::pin_root(fmt);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    let args = pyre_object::gc_roots::pin_root(args);
    let fmt_bytes = pyre_object::bytesobject::bytes_like_data(fmt);
    let (format, deferred_error) = parse_bytes_incremental(fmt_bytes)?;
    let (num_specifiers, mapping_required) = format
        .check_specifiers()
        .ok_or_else(|| PyError::type_error("format requires a mapping"))?;
    let mut parts = format.into_iter().peekable();
    let is_mapping = bytes_format_is_mapping(args);
    let mut result = Vec::new();

    if num_specifiers == 0 && deferred_error.is_none() {
        if !is_mapping && !bytes_format_empty_tuple(args) {
            return Err(PyError::type_error(
                "not all arguments converted during bytes formatting",
            ));
        }
        for (_, part) in parts {
            match part {
                CFormatPart::Literal(literal) => result.extend(literal),
                CFormatPart::Spec(_) => unreachable!(),
            }
        }
        return Ok(bytes_format_result(fmt, &result));
    }

    if mapping_required {
        if !is_mapping {
            return Err(PyError::type_error("format requires a mapping"));
        }
        while let Some((_, part)) = parts.next() {
            match part {
                CFormatPart::Literal(literal) => result.extend(literal),
                CFormatPart::Spec(CFormatSpecKeyed {
                    mapping_key,
                    mut spec,
                }) => {
                    let current_deferred = deferred_error.filter(|_| parts.peek().is_none());
                    let key = mapping_key.expect("mapping spec carries a key");
                    let value = crate::baseobjspace::getitem(
                        pyre_object::gc_roots::shadow_stack_get(args_slot),
                        pyre_object::w_bytes_from_bytes(&key),
                    )?;
                    let value = pyre_object::gc_roots::pin_root(value);
                    mapping_star_operands(
                        &mut spec,
                        value,
                        current_deferred
                            .and_then(DeferredPercentError::before_conversion)
                            .is_none(),
                    )?;
                    if let Some(error) =
                        current_deferred.and_then(DeferredPercentError::before_conversion)
                    {
                        return Err(PyError::value_error(error.to_string()));
                    }
                    if let Some(error) =
                        current_deferred.and_then(DeferredPercentError::unsupported)
                    {
                        return Err(PyError::value_error(error.to_string()));
                    }
                    result.extend(spec_format_bytes(&spec, value)?);
                }
            }
        }
        if let Some(error) = deferred_error {
            error.after_parts(is_mapping)?;
        }
        return Ok(bytes_format_result(fmt, &result));
    }

    let items: Vec<PyObjectRef> = if pyre_object::is_tuple(args) {
        let n = pyre_object::w_tuple_len(args);
        (0..n)
            .filter_map(|i| pyre_object::w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };
    let mut pos = OperandColumn {
        base: pyre_object::gc_roots::pin_roots(&items),
        len: items.len(),
        cursor: 0,
    };

    while let Some((_, part)) = parts.next() {
        match part {
            CFormatPart::Literal(literal) => result.extend(literal),
            CFormatPart::Spec(CFormatSpecKeyed { mut spec, .. }) => {
                let current_deferred = deferred_error.filter(|_| parts.peek().is_none());
                update_quantity_from_tuple(&mut pos, &mut spec.min_field_width, &mut spec.flags)?;
                update_precision_from_tuple(&mut pos, &mut spec.precision)?;
                if let Some(error) =
                    current_deferred.and_then(DeferredPercentError::before_conversion)
                {
                    return Err(PyError::value_error(error.to_string()));
                }
                let Some(value) = pos.next() else {
                    return Err(PyError::type_error(
                        "not enough arguments for format string",
                    ));
                };
                if let Some(error) = current_deferred.and_then(DeferredPercentError::unsupported) {
                    return Err(PyError::value_error(error.to_string()));
                }
                result.extend(spec_format_bytes(&spec, value)?);
            }
        }
    }

    if let Some(error) = deferred_error {
        error.after_parts(is_mapping)?;
    }

    if pos.has_next() {
        Err(PyError::type_error(
            "not all arguments converted during bytes formatting",
        ))
    } else {
        Ok(bytes_format_result(fmt, &result))
    }
}

unsafe fn bytes_format_result(fmt: PyObjectRef, data: &[u8]) -> PyObjectRef {
    if pyre_object::bytearrayobject::is_bytearray(fmt) {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(data)
    } else {
        pyre_object::bytesobject::w_bytes_from_bytes(data)
    }
}

unsafe fn bytes_format_empty_tuple(obj: PyObjectRef) -> bool {
    pyre_object::is_tuple(obj) && pyre_object::w_tuple_len(obj) == 0
}

unsafe fn bytes_format_is_mapping(obj: PyObjectRef) -> bool {
    !pyre_object::is_tuple(obj)
        && !pyre_object::is_str(obj)
        && !pyre_object::bytesobject::is_bytes_like(obj)
        && has_dunder(obj, "__getitem__")
}

/// `CFormatSpec::format_number` over RPython rbigint.
fn cformat_rbigint(spec: &CFormatSpec, num: &BigInt) -> Result<String, PyError> {
    let CFormatType::Number(number_type) = spec.format_type else {
        unreachable!()
    };
    let (radix, upper, prefix) = match number_type {
        CNumberType::DecimalD | CNumberType::DecimalI | CNumberType::DecimalU => (10, false, ""),
        CNumberType::Octal => (
            8,
            false,
            if spec.flags.contains(CConversionFlags::ALTERNATE_FORM) {
                "0o"
            } else {
                ""
            },
        ),
        CNumberType::HexLower => (
            16,
            false,
            if spec.flags.contains(CConversionFlags::ALTERNATE_FORM) {
                "0x"
            } else {
                ""
            },
        ),
        CNumberType::HexUpper => (
            16,
            true,
            if spec.flags.contains(CConversionFlags::ALTERNATE_FORM) {
                "0X"
            } else {
                ""
            },
        ),
    };
    let digits = match radix {
        8 => majit_rlib::rbigint::BASE8,
        10 => majit_rlib::rbigint::BASE10,
        16 => majit_rlib::rbigint::BASE16,
        _ => unreachable!("percent integer formats use radix 8, 10, or 16"),
    };
    let negative = num.int_lt(0);
    // Only the decimal conversion is quadratic, so only it carries the
    // `sys.set_int_max_str_digits` limit; `%o`/`%x`/`%X` are exempt.
    let maxdigits = if radix == 10 {
        crate::module::sys::state::int_max_str_digits()
    } else {
        0
    };
    let mut magnitude =
        num.format(digits, "", "", maxdigits as i64)
            .map_err(|error| match error {
                majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
                majit_rlib::rbigint::RBigIntError::MaxStrDigits => {
                    crate::builtins::int_max_str_digits_error(maxdigits)
                }
                _ => unreachable!("validated radix formatting returned an unrelated error"),
            })?;
    if negative {
        let Some(unsigned) = magnitude.strip_prefix('-') else {
            return Err(PyError::system_error(
                "rbigint formatting omitted the negative sign",
            ));
        };
        magnitude = unsigned.to_owned();
    }
    if upper {
        magnitude.make_ascii_uppercase();
    }
    if let Some(CFormatPrecision::Quantity(CFormatQuantity::Amount(precision))) = spec.precision
        && magnitude.len() < precision
    {
        magnitude = format!("{}{magnitude}", "0".repeat(precision - magnitude.len()));
    }
    let sign = if negative {
        "-"
    } else {
        spec.flags.sign_string()
    };
    let signed_prefix = format!("{sign}{prefix}");
    let width = match spec.min_field_width {
        Some(CFormatQuantity::Amount(width)) => width,
        _ => 0,
    };
    if spec.flags.contains(CConversionFlags::ZERO_PAD) {
        let fill = if spec.flags.contains(CConversionFlags::LEFT_ADJUST) {
            ' '
        } else {
            '0'
        };
        let needed = width.saturating_sub(signed_prefix.len() + magnitude.len());
        if spec.flags.contains(CConversionFlags::LEFT_ADJUST) {
            Ok(format!(
                "{signed_prefix}{magnitude}{}",
                fill.to_string().repeat(needed)
            ))
        } else {
            Ok(format!(
                "{signed_prefix}{}{magnitude}",
                fill.to_string().repeat(needed)
            ))
        }
    } else {
        let body = format!("{signed_prefix}{magnitude}");
        let needed = width.saturating_sub(body.len());
        if spec.flags.contains(CConversionFlags::LEFT_ADJUST) {
            Ok(format!("{body}{}", " ".repeat(needed)))
        } else {
            Ok(format!("{}{body}", " ".repeat(needed)))
        }
    }
}

/// Reserve a spec's padding run before the formatter materialises it.
///
/// `min_field_width` reaches here straight from Python — `'%*s' % (n, x)`
/// takes it from the argument tuple — and the fill run is built eagerly, so
/// an unsatisfiable width has to unwind rather than abort the process. The
/// fill character is always ASCII, so `width` bounds the run in bytes.
fn check_min_field_width(spec: &CFormatSpec) -> Result<(), PyError> {
    if let Some(CFormatQuantity::Amount(width)) = spec.min_field_width {
        crate::builtins::try_vec_with_capacity::<u8>(width)?;
    }
    Ok(())
}

unsafe fn spec_format_bytes(spec: &CFormatSpec, obj: PyObjectRef) -> Result<Vec<u8>, PyError> {
    check_min_field_width(spec)?;
    match &spec.format_type {
        CFormatType::String(CFormatConversion::Repr | CFormatConversion::Ascii) => {
            Ok(spec.format_bytes(crate::builtins::py_ascii(obj)?.as_bytes()))
        }
        // `format_obj` reads `%b` and `%s` as the same conversion for bytes.
        CFormatType::Bytes | CFormatType::String(CFormatConversion::Str) => {
            if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
                return Ok(spec.format_bytes(pyre_object::bytesobject::bytes_like_data(src)));
            }
            let Some(method) = crate::baseobjspace::lookup(obj, "__bytes__") else {
                return Err(PyError::type_error(format!(
                    "%b requires a bytes-like object, or an object that implements __bytes__, not '{}'",
                    crate::baseobjspace::object_functionstr_type_name(obj)
                )));
            };
            // bytesobject.py `invoke_bytes_method`:
            // `space.get_and_call_function(w_bytes_method, w_source)`.
            let w_type =
                crate::typedef::r#type(obj).map_or(pyre_object::PY_NULL, |w_type| w_type.as_ptr());
            let bytes = crate::baseobjspace::get_and_call_function(method, obj, w_type, &[])?;
            if !pyre_object::is_bytes(bytes) {
                return Err(PyError::type_error(format!(
                    "__bytes__ returned non-bytes (type {})",
                    crate::baseobjspace::object_functionstr_type_name(bytes)
                )));
            }
            Ok(spec.format_bytes(pyre_object::bytesobject::bytes_like_data(bytes)))
        }
        CFormatType::Number(number_type) => {
            let value = match number_type {
                CNumberType::DecimalD | CNumberType::DecimalI | CNumberType::DecimalU => {
                    number_arg_decimal(spec, obj)?
                }
                _ => number_arg_integer(spec, obj)?,
            };
            Ok(cformat_rbigint(spec, &value)?.into_bytes())
        }
        CFormatType::Float(_) => {
            let value = crate::baseobjspace::float_w(obj).map_err(|e| {
                if e.kind == PyErrorKind::TypeError {
                    PyError::type_error(format!(
                        "float argument required, not {}",
                        crate::baseobjspace::object_functionstr_type_name(obj)
                    ))
                } else {
                    e
                }
            })?;
            Ok(spec.format_float(value).into_bytes())
        }
        CFormatType::Character(CCharacterType::Character) => {
            Ok(spec.format_char(bytes_char_arg(obj)?))
        }
    }
}

unsafe fn bytes_char_arg(obj: PyObjectRef) -> Result<u8, PyError> {
    if pyre_object::bytesobject::is_bytes(obj) || pyre_object::bytearrayobject::is_bytearray(obj) {
        let data = pyre_object::bytesobject::bytes_like_data(obj);
        if data.len() == 1 {
            return Ok(data[0]);
        }
        let kind = if pyre_object::bytesobject::is_bytes(obj) {
            "bytes"
        } else {
            "bytearray"
        };
        return Err(PyError::type_error(format!(
            "%c requires an integer in range(256) or a single byte, not a {kind} object of length {}",
            data.len()
        )));
    }
    let value = if pyre_object::pyobject::is_int_or_long(obj) {
        arg_to_bigint(obj)
    } else if has_dunder(obj, "__index__") {
        crate::builtins::obj_to_bigint(crate::baseobjspace::space_index(obj)?)
    } else {
        let type_name = match crate::typedef::r#type(obj) {
            Some(w_type) => crate::baseobjspace::type_fully_qualified_name(w_type.as_ptr()),
            None => crate::baseobjspace::object_functionstr_type_name(obj),
        };
        return Err(PyError::type_error(format!(
            "%c requires an integer in range(256) or a single byte, not {}",
            type_name
        )));
    };
    let overflow = || PyError::new(PyErrorKind::OverflowError, "%c arg not in range(256)");
    if pyre_object::jit_bigint_to_i64_fits(&value) == 0 {
        return Err(overflow());
    }
    let n = pyre_object::jit_bigint_to_i64_value(&value);
    if !(0..=255).contains(&n) {
        return Err(overflow());
    }
    Ok(n as u8)
}

/// True when `obj`'s type carries `__getitem__` (`PyMapping_Check`), so a
/// `%(key)s` spec can index it.
unsafe fn has_getitem(obj: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj) {
        Some(tp) => crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__getitem__").is_some(),
        None => false,
    }
}

/// Apply a parsed spec to one argument, producing the formatted fragment.
/// `formatting.py fmt_s / fmt_d / fmt_f / ...` — the per-conversion value
/// coercion and formatting.
unsafe fn spec_format_string(spec: &CFormatSpec, obj: PyObjectRef) -> Result<Wtf8Buf, PyError> {
    check_min_field_width(spec)?;
    match &spec.format_type {
        CFormatType::String(conversion) => {
            let result = match conversion {
                CFormatConversion::Str => crate::py_str_wtf8(obj)?,
                CFormatConversion::Repr => crate::py_repr_wtf8(obj)?,
                CFormatConversion::Ascii => Wtf8Buf::from_string(crate::builtins::py_ascii(obj)?),
            };
            Ok(spec.format_string(result))
        }
        // `%b` is a bytes-only conversion; a `CFormatContext::Str` parse
        // rejects the `b` as an unsupported format character, so the unicode
        // formatter never sees a spec carrying it.
        CFormatType::Bytes => unreachable!("`%b` does not parse in a str format string"),
        CFormatType::Number(number_type) => {
            let value = match number_type {
                CNumberType::DecimalD | CNumberType::DecimalI | CNumberType::DecimalU => {
                    number_arg_decimal(spec, obj)?
                }
                _ => number_arg_integer(spec, obj)?,
            };
            Ok(Wtf8Buf::from_string(cformat_rbigint(spec, &value)?))
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
        // bigint_w returns the immutable payload reference, not a copy.
        w_long_get_value(obj).translated_alias()
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
    if let Some(method) = crate::baseobjspace::lookup(obj, "__int__") {
        let r = crate::builtins::call_and_check(method, &[obj])?;
        if is_int_like(r) || is_long(r) {
            return Ok(arg_to_bigint(r));
        }
    }
    if has_dunder(obj, "__index__") {
        // `format_num_helper`: a TypeError from the numeric decoder (a non-int
        // `__index__` return included) is reported as the operand-type error,
        // naming the original argument, not the coerced result.
        return match crate::baseobjspace::space_index(obj) {
            Ok(w) => Ok(crate::builtins::obj_to_bigint(w)),
            Err(e) if e.kind == crate::PyErrorKind::TypeError => {
                Err(number_type_error(spec, obj, "a real number is required"))
            }
            Err(e) => Err(e),
        };
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
        // `format_num_helper` (maybe_index): a TypeError from `space.index`
        // is reported as the operand-type error naming the original argument.
        return match crate::baseobjspace::space_index(obj) {
            Ok(w) => Ok(crate::builtins::obj_to_bigint(w)),
            Err(e) if e.kind == crate::PyErrorKind::TypeError => {
                Err(number_type_error(spec, obj, "an integer is required"))
            }
            Err(e) => Err(e),
        };
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
        if let Some(cp) = cps.next()
            && cps.next().is_none()
        {
            return Ok(cp);
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
        let tn = match crate::typedef::r#type(obj) {
            Some(w_type) => crate::baseobjspace::type_fully_qualified_name(w_type.as_ptr()),
            None => crate::baseobjspace::object_functionstr_type_name(obj),
        };
        return Err(PyError::type_error(format!(
            "%c requires an int or a unicode character, not {tn}"
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
        Some(tp) => crate::baseobjspace::lookup_in_type(tp.as_ptr(), name).is_some(),
        None => false,
    }
}

/// `peel_num` — a `*` field width reads its value (and, when negative, the
/// left-align flag) from the next positional argument.
/// The positional operands, addressed by shadow slot rather than held in a
/// `Vec`.
///
/// Every conversion runs Python — `__str__`, `__repr__`, `__format__`, and a
/// `*` width's `__index__` — so each one is a collection point.  A plain `Vec`
/// of operands is not something the collector rewrites, so an entry still
/// waiting its turn would be formatted at a pre-move address once it is a list
/// or a dict.  The column is pinned once by the caller and each operand is
/// read back from its slot when its turn comes.
struct OperandColumn {
    base: usize,
    len: usize,
    cursor: usize,
}

impl OperandColumn {
    fn next(&mut self) -> Option<PyObjectRef> {
        if self.cursor >= self.len {
            return None;
        }
        let operand = pyre_object::gc_roots::shadow_stack_get(self.base + self.cursor);
        self.cursor += 1;
        Some(operand)
    }

    /// Whether any operand is still unconsumed — the `checkconsumed` test.
    fn has_next(&self) -> bool {
        self.cursor < self.len
    }
}

unsafe fn update_quantity_from_tuple(
    pos: &mut OperandColumn,
    quantity: &mut Option<CFormatQuantity>,
    flags: &mut CConversionFlags,
) -> Result<(), PyError> {
    if !matches!(quantity, Some(CFormatQuantity::FromValuesTuple)) {
        return Ok(());
    }
    let v = star_int(pos.next(), StarField::Width)?;
    if v < 0 {
        flags.insert(CConversionFlags::LEFT_ADJUST);
    }
    *quantity = Some(CFormatQuantity::Amount(v.unsigned_abs() as usize));
    Ok(())
}

/// `peel_num` — a `*` precision reads its value from the next positional
/// argument (a negative precision is treated as absent).
unsafe fn update_precision_from_tuple(
    pos: &mut OperandColumn,
    precision: &mut Option<CFormatPrecision>,
) -> Result<(), PyError> {
    if !matches!(
        precision,
        Some(CFormatPrecision::Quantity(CFormatQuantity::FromValuesTuple))
    ) {
        return Ok(());
    }
    let v = star_int(pos.next(), StarField::Precision)?;
    *precision = Some(CFormatPrecision::Quantity(CFormatQuantity::Amount(
        v.max(0) as usize,
    )));
    Ok(())
}

/// Consume `*` fields on a keyed conversion in CPython 3.14 order.
///
/// PyPy `StringFormatter.parse_fmt` obtains `w_value` from
/// `getmappingvalue`, then `peel_num` asks `nextinputvalue` for each star.
/// CPython 3.14's `PyUnicode_Format` observably uses the mapped value as that
/// first star operand: `'%(x)*s' % {'x': 'a'}` raises `* wants int`, while an
/// integer mapped value is consumed and the conversion then raises `not enough
/// arguments for format string`. Keep PyPy's lookup-then-star control-flow,
/// with the 3.14 operand source at this spec-deviation site.
unsafe fn mapping_star_operands(
    spec: &mut CFormatSpec,
    mapped_value: PyObjectRef,
    conversion_follows: bool,
) -> Result<(), PyError> {
    let has_width_star = matches!(spec.min_field_width, Some(CFormatQuantity::FromValuesTuple));
    let has_precision_star = matches!(
        spec.precision,
        Some(CFormatPrecision::Quantity(CFormatQuantity::FromValuesTuple))
    );
    if !has_width_star && !has_precision_star {
        return Ok(());
    }

    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(mapped_value);
    let mut mapped = OperandColumn {
        base,
        len: 1,
        cursor: 0,
    };
    update_quantity_from_tuple(&mut mapped, &mut spec.min_field_width, &mut spec.flags)?;
    update_precision_from_tuple(&mut mapped, &mut spec.precision)?;

    if !conversion_follows {
        return Ok(());
    }

    // At least one star consumed the sole mapped value. The conversion still
    // requires its own operand, exactly like BaseStringFormatter.format's
    // `nextinputvalue` after conversion-character validation.
    Err(PyError::type_error(
        "not enough arguments for format string",
    ))
}

#[derive(Clone, Copy)]
enum StarField {
    Width,
    Precision,
}

/// The `*` argument must be an int; consume it, matching `nextinputvalue`.
unsafe fn star_int(arg: Option<PyObjectRef>, field: StarField) -> Result<i64, PyError> {
    let Some(arg) = arg else {
        return Err(PyError::type_error(
            "not enough arguments for format string",
        ));
    };
    if !pyre_object::pyobject::is_int_or_long(arg) {
        return Err(PyError::type_error("* wants int"));
    }
    let big = crate::builtins::obj_to_bigint(arg);
    match field {
        StarField::Width => {
            if pyre_object::jit_bigint_to_i64_fits(&big) != 0 {
                Ok(pyre_object::jit_bigint_to_i64_value(&big))
            } else {
                Err(PyError::overflow_error(
                    "Python int too large to convert to C ssize_t",
                ))
            }
        }
        StarField::Precision => {
            if pyre_object::jit_bigint_to_i64_fits(&big) != 0 {
                let value = pyre_object::jit_bigint_to_i64_value(&big);
                if i32::try_from(value).is_ok() {
                    return Ok(value);
                }
            }
            Err(PyError::overflow_error(
                "Python int too large to convert to C int",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_error(error: PyError, kind: PyErrorKind, message: &str) {
        assert_eq!(error.kind, kind);
        assert_eq!(error.message_text(), message);
    }

    #[test]
    fn percent_unknown_conversion_observes_python314_operand_order() {
        crate::typedef::init_typeobjects();

        // PyPy `StringFormatter.format` validates the conversion before its
        // final `nextinputvalue`; CPython 3.14 `PyUnicode_Format` first proves
        // that operand exists. This is the observable 3.14-spec departure.
        let error =
            unsafe { str_format_percent(w_str_new("%z"), w_tuple_new(Vec::new())).unwrap_err() };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(w_str_new("%z"), w_tuple_new(vec![w_int_new(1)])).unwrap_err()
        };
        assert_error(
            error,
            PyErrorKind::ValueError,
            "unsupported format character 'z' (0x7a) at index 1",
        );

        let error = unsafe {
            bytes_format_percent(w_bytes_from_bytes(b"%*z"), w_tuple_new(vec![w_int_new(1)]))
                .unwrap_err()
        };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_str_new("not an int")) };
        let error = unsafe { str_format_percent(w_str_new("%(x)*s"), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::TypeError, "* wants int");

        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_int_new(2)) };
        let error = unsafe { str_format_percent(w_str_new("%(x)*s"), mapping).unwrap_err() };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );
    }

    #[test]
    fn percent_incomplete_syntax_runs_the_complete_prefix_first() {
        crate::typedef::init_typeobjects();

        let error =
            unsafe { str_format_percent(w_str_new("%s %"), w_tuple_new(Vec::new())).unwrap_err() };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(w_str_new("%s %"), w_tuple_new(vec![w_str_new("done")])).unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");

        let error = unsafe {
            str_format_percent(w_str_new("%(x"), w_tuple_new(vec![w_int_new(1)])).unwrap_err()
        };
        assert_error(error, PyErrorKind::TypeError, "format requires a mapping");

        let mapping = w_dict_new();
        let error = unsafe { str_format_percent(w_str_new("%(x"), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::ValueError, "incomplete format key");

        let error = unsafe {
            bytes_format_percent(
                w_bytes_from_bytes(b"%s %"),
                w_tuple_new(vec![w_bytes_from_bytes(b"done")]),
            )
            .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");

        let error = unsafe {
            bytes_format_percent(w_bytes_from_bytes(b"%(x"), w_tuple_new(vec![w_int_new(1)]))
                .unwrap_err()
        };
        assert_error(error, PyErrorKind::TypeError, "format requires a mapping");

        let mapping = w_dict_new();
        let error =
            unsafe { bytes_format_percent(w_bytes_from_bytes(b"%(x"), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::ValueError, "incomplete format key");

        let error = unsafe {
            str_format_percent(w_str_new("%s %10"), w_tuple_new(Vec::new())).unwrap_err()
        };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(w_str_new("%s %10"), w_tuple_new(vec![w_str_new("done")]))
                .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");

        let error =
            unsafe { str_format_percent(w_str_new("%*"), w_tuple_new(Vec::new())).unwrap_err() };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(w_str_new("%*"), w_tuple_new(vec![w_int_new(2)])).unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");

        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_str_new("not an int")) };
        let error = unsafe { str_format_percent(w_str_new("%(x).*"), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::TypeError, "* wants int");

        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_int_new(2)) };
        let error = unsafe { str_format_percent(w_str_new("%(x).*"), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");

        let error = unsafe {
            bytes_format_percent(w_bytes_from_bytes(b"%.*"), w_tuple_new(vec![w_int_new(2)]))
                .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "incomplete format");
    }

    #[test]
    fn percent_oversized_quantity_runs_pre_conversion_acquisition() {
        crate::typedef::init_typeobjects();

        let huge = "9".repeat(100);
        let width_after_string = format!("%s %{huge}s");
        let error = unsafe {
            str_format_percent(w_str_new(&width_after_string), w_tuple_new(Vec::new())).unwrap_err()
        };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(
                w_str_new(&width_after_string),
                w_tuple_new(vec![w_str_new("done")]),
            )
            .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "width too big");

        let precision_after_star = format!("%*.{huge}s");
        let error = unsafe {
            str_format_percent(w_str_new(&precision_after_star), w_tuple_new(Vec::new()))
                .unwrap_err()
        };
        assert_error(
            error,
            PyErrorKind::TypeError,
            "not enough arguments for format string",
        );

        let error = unsafe {
            str_format_percent(
                w_str_new(&precision_after_star),
                w_tuple_new(vec![w_int_new(2)]),
            )
            .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "precision too big");

        let mapping_precision = format!("%(x)*.{huge}s");
        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_str_new("not an int")) };
        let error =
            unsafe { str_format_percent(w_str_new(&mapping_precision), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::TypeError, "* wants int");

        let mapping = w_dict_new();
        unsafe { w_dict_setitem_str_no_proxy(mapping, "x", w_int_new(2)) };
        let error =
            unsafe { str_format_percent(w_str_new(&mapping_precision), mapping).unwrap_err() };
        assert_error(error, PyErrorKind::ValueError, "precision too big");

        let bytes_width = format!("%s %{huge}s").into_bytes();
        let error = unsafe {
            bytes_format_percent(
                w_bytes_from_bytes(&bytes_width),
                w_tuple_new(vec![w_bytes_from_bytes(b"done")]),
            )
            .unwrap_err()
        };
        assert_error(error, PyErrorKind::ValueError, "width too big");
    }
}
