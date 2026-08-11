//! CPython-compatible `_json` accelerator.
//!
//! The storage shape follows `json/scanner.py` and PyPy's
//! `_pypyjson.interp_encoder.W_Encoder`: scanner callbacks and encoder state
//! are fields on their owning Python objects.  In particular, there is no TLS
//! or process side-table for semantic state.

mod machinery;

use pyre_object::{PyObjectRef, gc_roots};
use rustpython_wtf8::Wtf8;

use crate::error::{PyError, PyResult};

type PyErrorRootSlots = [Option<usize>; 3];

enum PyResultRootSlots {
    Value(usize),
    Error(PyErrorRootSlots),
}

fn pin_pyerror_payload(err: &PyError) -> PyErrorRootSlots {
    let mut slots = [None; 3];
    for (slot, value) in
        slots
            .iter_mut()
            .zip([err.exc_object, err.w_name_context, err.w_obj_context])
    {
        if !value.is_null() {
            *slot = Some(gc_roots::shadow_stack_len());
            gc_roots::pin_root(value);
        }
    }
    slots
}

fn reload_pyerror_payload(mut err: PyError, slots: PyErrorRootSlots) -> PyError {
    if let Some(slot) = slots[0] {
        err.exc_object = gc_roots::shadow_stack_get(slot);
    }
    if let Some(slot) = slots[1] {
        err.w_name_context = gc_roots::shadow_stack_get(slot);
    }
    if let Some(slot) = slots[2] {
        err.w_obj_context = gc_roots::shadow_stack_get(slot);
    }
    err
}

fn pin_pyresult_payload(result: &PyResult) -> PyResultRootSlots {
    match result {
        Ok(value) => {
            let slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(*value);
            PyResultRootSlots::Value(slot)
        }
        Err(err) => PyResultRootSlots::Error(pin_pyerror_payload(err)),
    }
}

fn reload_pyresult_payload(result: PyResult, slots: PyResultRootSlots) -> PyResult {
    match (result, slots) {
        (Ok(_), PyResultRootSlots::Value(slot)) => Ok(gc_roots::shadow_stack_get(slot)),
        (Err(err), PyResultRootSlots::Error(slots)) => Err(reload_pyerror_payload(err, slots)),
        _ => unreachable!("root slots match the result variant"),
    }
}

fn require_string(obj: PyObjectRef) -> Result<&'static Wtf8, PyError> {
    if !unsafe { pyre_object::is_str(obj) } {
        return Err(PyError::type_error("first argument must be a string"));
    }
    Ok(unsafe { pyre_object::w_str_get_wtf8(obj) })
}

fn index_i64(obj: PyObjectRef) -> Result<i64, PyError> {
    let indexed = crate::baseobjspace::space_index(obj)?;
    crate::baseobjspace::int_w(indexed)
}

fn json_decode_error(msg: String, doc: PyObjectRef, pos: usize) -> PyError {
    let Some(module) = crate::importing::get_sys_module("json.decoder") else {
        return PyError::value_error(format!("{msg}: line 1 column 1 (char {pos})"));
    };
    let Ok(class) = crate::baseobjspace::getattr_str(module, "JSONDecodeError") else {
        return PyError::value_error(format!("{msg}: line 1 column 1 (char {pos})"));
    };
    let args = [
        pyre_object::w_str_new(&msg),
        doc,
        pyre_object::w_int_new(pos as i64),
    ];
    match crate::call::call_function_impl_result(class, &args) {
        Ok(exc) => unsafe { PyError::from_exc_object(exc) },
        Err(err) => err,
    }
}

fn encode_basestring_impl(obj: PyObjectRef, ascii_only: bool) -> PyResult {
    let value = require_string(obj)?;
    Ok(pyre_object::w_str_from_wtf8_managed(
        machinery::encode_string(value, ascii_only),
    ))
}

fn scanstring_impl(doc: PyObjectRef, end: i64, strict_obj: PyObjectRef) -> PyResult {
    let value = require_string(doc)?;
    if end < 0 {
        return Err(PyError::value_error("end is out of bounds"));
    }
    let strict = crate::baseobjspace::is_true(strict_obj)?;
    let start = end as usize;
    let byte_start = unsafe { pyre_object::w_str_index_to_byte(doc, start) };
    let rest = value.get(byte_start..).ok_or_else(|| {
        json_decode_error("Unterminated string starting at".to_owned(), doc, start)
    })?;
    match machinery::scan_string(rest, start, strict) {
        Ok((decoded, next, _)) => {
            let _roots = gc_roots::push_roots();
            let decoded_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(decoded));
            let next_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(pyre_object::w_int_new(next as i64));
            Ok(pyre_object::w_tuple_new(vec![
                gc_roots::shadow_stack_get(decoded_slot),
                gc_roots::shadow_stack_get(next_slot),
            ]))
        }
        Err(err) => Err(json_decode_error(err.msg, doc, err.pos)),
    }
}

fn stop_iteration(index: i64) -> PyError {
    let class = crate::builtins::lookup_exc_class("StopIteration")
        .expect("StopIteration installed before _json");
    match crate::call::call_function_impl_result(class, &[pyre_object::w_int_new(index)]) {
        Ok(exc) => unsafe { PyError::from_exc_object(exc) },
        Err(err) => err,
    }
}

#[inline]
fn skip_json_whitespace(
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> (usize, usize) {
    let bytes = unsafe { pyre_object::w_str_get_wtf8(doc) }.as_bytes();
    while matches!(bytes.get(byte_index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        char_index += 1;
        byte_index += 1;
    }
    (char_index, byte_index)
}

fn scanner_decode_error(msg: &str, doc: PyObjectRef, pos: usize) -> PyError {
    json_decode_error(msg.to_owned(), doc, pos)
}

/// CPython `scan_once_unicode`: scanner state owns only the six callbacks and
/// strict flag; the memo dict is born per public call and threaded through the
/// recursive object/array descent.
fn scanner_scan_once(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    char_index: usize,
    byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        gc_roots::pin_root(value);
    }
    let doc = gc_roots::shadow_stack_get(slot + 2);
    let bytes = unsafe { pyre_object::w_str_get_wtf8(doc) }.as_bytes();
    let Some(&first) = bytes.get(byte_index) else {
        return Err(stop_iteration(char_index as i64));
    };

    match first {
        b'"' => {
            let strict = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
                .expect("Scanner payload")
                .strict;
            let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
            let (decoded, next_char, bytes_used) =
                machinery::scan_string(&value[byte_index + 1..], char_index + 1, strict)
                    .map_err(|err| json_decode_error(err.msg, doc, err.pos))?;
            Ok((
                pyre_object::w_str_from_wtf8(decoded),
                next_char,
                byte_index + 1 + bytes_used,
            ))
        }
        b'{' => {
            crate::stack_check::stack_check().map_err(|_| {
                PyError::recursion_error(
                    "maximum recursion depth exceeded while decoding a JSON object from a string",
                )
            })?;
            scanner_parse_object(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                doc,
                char_index + 1,
                byte_index + 1,
            )
        }
        b'[' => {
            crate::stack_check::stack_check().map_err(|_| {
                PyError::recursion_error(
                    "maximum recursion depth exceeded while decoding a JSON array from a string",
                )
            })?;
            scanner_parse_array(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                doc,
                char_index + 1,
                byte_index + 1,
            )
        }
        _ => scan_scalar(
            gc_roots::shadow_stack_get(slot),
            doc,
            char_index,
            byte_index,
        ),
    }
}

fn scan_scalar(
    self_obj: PyObjectRef,
    doc: PyObjectRef,
    index: usize,
    byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
    let rest = &value.as_bytes()[byte_index..];
    let simple = if rest.starts_with(b"null") {
        Some((pyre_object::w_none(), 4))
    } else if rest.starts_with(b"true") {
        Some((pyre_object::w_bool_from(true), 4))
    } else if rest.starts_with(b"false") {
        Some((pyre_object::w_bool_from(false), 5))
    } else {
        None
    };
    if let Some((obj, len)) = simple {
        return Ok((obj, index + len, byte_index + len));
    }

    let mut end = 0usize;
    if rest.get(end) == Some(&b'-') {
        end += 1;
    }
    match rest.get(end) {
        Some(b'0') => end += 1,
        Some(b'1'..=b'9') => {
            end += 1;
            while matches!(rest.get(end), Some(b'0'..=b'9')) {
                end += 1;
            }
        }
        _ => end = 0,
    }
    let mut is_float = false;
    if end != 0 && rest.get(end) == Some(&b'.') && matches!(rest.get(end + 1), Some(b'0'..=b'9')) {
        is_float = true;
        end += 2;
        while matches!(rest.get(end), Some(b'0'..=b'9')) {
            end += 1;
        }
    }
    if end != 0 && matches!(rest.get(end), Some(b'e' | b'E')) {
        let mut exponent = end + 1;
        if matches!(rest.get(exponent), Some(b'+' | b'-')) {
            exponent += 1;
        }
        if matches!(rest.get(exponent), Some(b'0'..=b'9')) {
            is_float = true;
            exponent += 1;
            while matches!(rest.get(exponent), Some(b'0'..=b'9')) {
                exponent += 1;
            }
            end = exponent;
        }
    }
    if end != 0 {
        let parser = crate::baseobjspace::getattr_str(
            self_obj,
            if is_float { "parse_float" } else { "parse_int" },
        )?;
        let number = unsafe { core::str::from_utf8_unchecked(&rest[..end]) };
        let parsed =
            crate::call::call_function_impl_result(parser, &[pyre_object::w_str_new(number)])?;
        return Ok((parsed, index + end, byte_index + end));
    }

    for (token, len) in [("NaN", 3usize), ("Infinity", 8), ("-Infinity", 9)] {
        if rest.starts_with(token.as_bytes()) {
            let parser = crate::baseobjspace::getattr_str(self_obj, "parse_constant")?;
            let parsed =
                crate::call::call_function_impl_result(parser, &[pyre_object::w_str_new(token)])?;
            return Ok((parsed, index + len, byte_index + len));
        }
    }
    Err(stop_iteration(index as i64))
}

fn scanner_parse_object(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        gc_roots::pin_root(value);
    }
    let pairs_hook = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
        .expect("Scanner payload")
        .object_pairs_hook;
    let has_pairs_hook = !unsafe { pyre_object::is_none(pairs_hook) };
    let result = if has_pairs_hook {
        pyre_object::w_list_new_empty()
    } else {
        pyre_object::w_dict_new()
    };
    gc_roots::pin_root(result);

    (char_index, byte_index) =
        skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
    let doc = gc_roots::shadow_stack_get(slot + 2);
    if unsafe { pyre_object::w_str_get_wtf8(doc) }
        .as_bytes()
        .get(byte_index)
        == Some(&b'}')
    {
        char_index += 1;
        byte_index += 1;
    } else {
        loop {
            let doc = gc_roots::shadow_stack_get(slot + 2);
            if unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
                != Some(&b'"')
            {
                return Err(scanner_decode_error(
                    "Expecting property name enclosed in double quotes",
                    doc,
                    char_index,
                ));
            }
            let strict = W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
                .expect("Scanner payload")
                .strict;
            let value = unsafe { pyre_object::w_str_get_wtf8(doc) };
            let (decoded, next_char, bytes_used) =
                machinery::scan_string(&value[byte_index + 1..], char_index + 1, strict)
                    .map_err(|err| json_decode_error(err.msg, doc, err.pos))?;
            char_index = next_char;
            byte_index += 1 + bytes_used;

            let iteration_roots = gc_roots::push_roots();
            let item_slot = gc_roots::shadow_stack_len();
            let candidate = pyre_object::w_str_from_wtf8(decoded);
            gc_roots::pin_root(candidate);
            let memo = gc_roots::shadow_stack_get(slot + 1);
            let key =
                match crate::baseobjspace::finditem(memo, gc_roots::shadow_stack_get(item_slot))? {
                    Some(key) => key,
                    None => {
                        crate::baseobjspace::setitem(
                            memo,
                            gc_roots::shadow_stack_get(item_slot),
                            gc_roots::shadow_stack_get(item_slot),
                        )?;
                        gc_roots::shadow_stack_get(item_slot)
                    }
                };
            gc_roots::pin_root(key);

            (char_index, byte_index) =
                skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
            let doc = gc_roots::shadow_stack_get(slot + 2);
            if unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
                != Some(&b':')
            {
                return Err(scanner_decode_error(
                    "Expecting ':' delimiter",
                    doc,
                    char_index,
                ));
            }
            char_index += 1;
            byte_index += 1;
            (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);

            let (value, next_char, next_byte) = scanner_scan_once(
                gc_roots::shadow_stack_get(slot),
                gc_roots::shadow_stack_get(slot + 1),
                gc_roots::shadow_stack_get(slot + 2),
                char_index,
                byte_index,
            )
            .map_err(|err| {
                if err.kind == crate::PyErrorKind::StopIteration {
                    scanner_decode_error(
                        "Expecting value",
                        gc_roots::shadow_stack_get(slot + 2),
                        char_index,
                    )
                } else {
                    err
                }
            })?;
            gc_roots::pin_root(value);
            char_index = next_char;
            byte_index = next_byte;

            if has_pairs_hook {
                let pair = pyre_object::w_tuple_new(vec![
                    gc_roots::shadow_stack_get(item_slot + 1),
                    gc_roots::shadow_stack_get(item_slot + 2),
                ]);
                gc_roots::pin_root(pair);
                unsafe {
                    pyre_object::w_list_append(
                        gc_roots::shadow_stack_get(slot + 3),
                        gc_roots::shadow_stack_get(item_slot + 3),
                    );
                }
            } else {
                crate::baseobjspace::setitem(
                    gc_roots::shadow_stack_get(slot + 3),
                    gc_roots::shadow_stack_get(item_slot + 1),
                    gc_roots::shadow_stack_get(item_slot + 2),
                )?;
            }
            drop(iteration_roots);

            (char_index, byte_index) =
                skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
            let doc = gc_roots::shadow_stack_get(slot + 2);
            match unsafe { pyre_object::w_str_get_wtf8(doc) }
                .as_bytes()
                .get(byte_index)
            {
                Some(b'}') => {
                    char_index += 1;
                    byte_index += 1;
                    break;
                }
                Some(b',') => {
                    let comma = char_index;
                    char_index += 1;
                    byte_index += 1;
                    (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);
                    if unsafe { pyre_object::w_str_get_wtf8(doc) }
                        .as_bytes()
                        .get(byte_index)
                        == Some(&b'}')
                    {
                        return Err(scanner_decode_error(
                            "Illegal trailing comma before end of object",
                            doc,
                            comma,
                        ));
                    }
                }
                _ => {
                    return Err(scanner_decode_error(
                        "Expecting ',' delimiter",
                        doc,
                        char_index,
                    ));
                }
            }
        }
    }

    let result = gc_roots::shadow_stack_get(slot + 3);
    let hook = if has_pairs_hook {
        W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
            .expect("Scanner payload")
            .object_pairs_hook
    } else {
        W_Scanner::from_obj(gc_roots::shadow_stack_get(slot))
            .expect("Scanner payload")
            .object_hook
    };
    if unsafe { pyre_object::is_none(hook) } {
        Ok((result, char_index, byte_index))
    } else {
        let hooked = crate::call::call_function_impl_result(hook, &[result])?;
        Ok((hooked, char_index, byte_index))
    }
}

fn scanner_parse_array(
    self_obj: PyObjectRef,
    memo: PyObjectRef,
    doc: PyObjectRef,
    mut char_index: usize,
    mut byte_index: usize,
) -> Result<(PyObjectRef, usize, usize), PyError> {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, memo, doc] {
        gc_roots::pin_root(value);
    }
    let result = pyre_object::w_list_new_empty();
    gc_roots::pin_root(result);
    (char_index, byte_index) =
        skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
    let doc = gc_roots::shadow_stack_get(slot + 2);
    if unsafe { pyre_object::w_str_get_wtf8(doc) }
        .as_bytes()
        .get(byte_index)
        == Some(&b']')
    {
        return Ok((
            gc_roots::shadow_stack_get(slot + 3),
            char_index + 1,
            byte_index + 1,
        ));
    }

    loop {
        let item_roots = gc_roots::push_roots();
        let (value, next_char, next_byte) = scanner_scan_once(
            gc_roots::shadow_stack_get(slot),
            gc_roots::shadow_stack_get(slot + 1),
            gc_roots::shadow_stack_get(slot + 2),
            char_index,
            byte_index,
        )
        .map_err(|err| {
            if err.kind == crate::PyErrorKind::StopIteration {
                scanner_decode_error(
                    "Expecting value",
                    gc_roots::shadow_stack_get(slot + 2),
                    char_index,
                )
            } else {
                err
            }
        })?;
        gc_roots::pin_root(value);
        unsafe {
            pyre_object::w_list_append(
                gc_roots::shadow_stack_get(slot + 3),
                gc_roots::shadow_stack_get(gc_roots::shadow_stack_len() - 1),
            );
        }
        drop(item_roots);
        char_index = next_char;
        byte_index = next_byte;
        (char_index, byte_index) =
            skip_json_whitespace(gc_roots::shadow_stack_get(slot + 2), char_index, byte_index);
        let doc = gc_roots::shadow_stack_get(slot + 2);
        match unsafe { pyre_object::w_str_get_wtf8(doc) }
            .as_bytes()
            .get(byte_index)
        {
            Some(b']') => {
                char_index += 1;
                byte_index += 1;
                break;
            }
            Some(b',') => {
                let comma = char_index;
                char_index += 1;
                byte_index += 1;
                (char_index, byte_index) = skip_json_whitespace(doc, char_index, byte_index);
                if unsafe { pyre_object::w_str_get_wtf8(doc) }
                    .as_bytes()
                    .get(byte_index)
                    == Some(&b']')
                {
                    return Err(scanner_decode_error(
                        "Illegal trailing comma before end of array",
                        doc,
                        comma,
                    ));
                }
            }
            _ => {
                return Err(scanner_decode_error(
                    "Expecting ',' delimiter",
                    doc,
                    char_index,
                ));
            }
        }
    }
    Ok((gc_roots::shadow_stack_get(slot + 3), char_index, byte_index))
}

fn scanner_call_impl(self_obj: PyObjectRef, doc: PyObjectRef, index: i64) -> PyResult {
    require_string(doc)?;
    if index < 0 {
        return Err(PyError::value_error("idx cannot be negative"));
    }
    let char_index = index as usize;
    let byte_index = unsafe { pyre_object::w_str_index_to_byte(doc, char_index) };
    if byte_index >= unsafe { pyre_object::w_str_get_wtf8(doc) }.len() {
        return Err(stop_iteration(index));
    }
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [self_obj, doc] {
        gc_roots::pin_root(value);
    }
    // CPython `scanner_call`: one exact dict per public invocation, shared by
    // the whole recursive descent and discarded with the call.
    let memo = pyre_object::w_dict_new();
    gc_roots::pin_root(memo);
    let (value, next, _) = scanner_scan_once(
        gc_roots::shadow_stack_get(slot),
        gc_roots::shadow_stack_get(slot + 2),
        gc_roots::shadow_stack_get(slot + 1),
        char_index,
        byte_index,
    )?;
    gc_roots::pin_root(value);
    Ok(pyre_object::w_tuple_new(vec![
        gc_roots::shadow_stack_get(slot + 3),
        pyre_object::w_int_new(next as i64),
    ]))
}

#[crate::pyre_class("_json.Scanner")]
pub struct W_Scanner {
    strict: bool,
    parse_float: PyObjectRef,
    parse_int: PyObjectRef,
    parse_constant: PyObjectRef,
    object_hook: PyObjectRef,
    object_pairs_hook: PyObjectRef,
}

mod scanner_class {
    use super::*;

    #[crate::pyre_methods]
    impl W_Scanner {
        fn __call__(&mut self, doc: PyObjectRef, index: i64) -> Result<PyObjectRef, PyError> {
            scanner_call_impl(self as *mut Self as PyObjectRef, doc, index)
        }

        #[getter]
        fn strict(&self) -> PyObjectRef {
            pyre_object::w_bool_from(self.strict)
        }
        #[getter]
        fn parse_float(&self) -> PyObjectRef {
            self.parse_float
        }
        #[getter]
        fn parse_int(&self) -> PyObjectRef {
            self.parse_int
        }
        #[getter]
        fn parse_constant(&self) -> PyObjectRef {
            self.parse_constant
        }
        #[getter]
        fn object_hook(&self) -> PyObjectRef {
            self.object_hook
        }
        #[getter]
        fn object_pairs_hook(&self) -> PyObjectRef {
            self.object_pairs_hook
        }
    }
}

#[crate::pyre_class("_json.Encoder")]
pub struct W_Encoder {
    markers: PyObjectRef,
    default: PyObjectRef,
    encoder: PyObjectRef,
    indent: PyObjectRef,
    key_separator: PyObjectRef,
    item_separator: PyObjectRef,
    sort_keys: PyObjectRef,
    skipkeys: PyObjectRef,
    allow_nan: PyObjectRef,
    fast_mode: i64,
    depth: i64,
}

mod encoder_class {
    use super::*;

    #[crate::pyre_methods]
    impl W_Encoder {
        fn __call__(
            &mut self,
            obj: PyObjectRef,
            level: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let level = index_i64(level)?;
            encoder_call_impl(self as *mut Self as PyObjectRef, obj, level)
        }

        #[getter]
        fn markers(&self) -> PyObjectRef {
            self.markers
        }
        #[getter]
        fn default(&self) -> PyObjectRef {
            self.default
        }
        #[getter]
        fn encoder(&self) -> PyObjectRef {
            self.encoder
        }
        #[getter]
        fn indent(&self) -> PyObjectRef {
            self.indent
        }
        #[getter]
        fn key_separator(&self) -> PyObjectRef {
            self.key_separator
        }
        #[getter]
        fn item_separator(&self) -> PyObjectRef {
            self.item_separator
        }
        #[getter]
        fn sort_keys(&self) -> PyObjectRef {
            self.sort_keys
        }
        #[getter]
        fn skipkeys(&self) -> PyObjectRef {
            self.skipkeys
        }
        #[getter]
        fn allow_nan(&self) -> PyObjectRef {
            self.allow_nan
        }
    }
}

fn is_instance(obj: PyObjectRef, ty: &pyre_object::PyType) -> bool {
    unsafe { crate::baseobjspace::isinstance_w(obj, crate::typedef::gettypeobject(ty)) }
}

fn encoder_attr(self_obj: PyObjectRef, name: &str) -> PyResult {
    crate::baseobjspace::getattr_str(self_obj, name)
}

fn append_python_string(out: &mut rustpython_wtf8::Wtf8Buf, obj: PyObjectRef) -> PyResult {
    let value = require_string(obj)?;
    out.push_wtf8(value);
    Ok(pyre_object::w_none())
}

fn encode_string_field(
    self_obj: PyObjectRef,
    out: &mut rustpython_wtf8::Wtf8Buf,
    obj: PyObjectRef,
) -> PyResult {
    let mode = W_Encoder::from_obj(self_obj)
        .expect("Encoder payload")
        .fast_mode;
    if mode == 0 || mode == 1 {
        out.push_wtf8(&machinery::encode_string(require_string(obj)?, mode == 0));
        return Ok(pyre_object::w_none());
    }
    let encoder = encoder_attr(self_obj, "encoder")?;
    let encoded = crate::call::call_function_impl_result(encoder, &[obj])?;
    append_python_string(out, encoded)
}

fn base_repr(obj: PyObjectRef, ty: &pyre_object::PyType) -> PyResult {
    let w_type = crate::typedef::gettypeobject(ty);
    let repr = crate::baseobjspace::getattr_str(w_type, "__repr__")?;
    crate::call::call_function_impl_result(repr, &[obj])
}

fn encode_float(
    self_obj: PyObjectRef,
    out: &mut rustpython_wtf8::Wtf8Buf,
    obj: PyObjectRef,
) -> PyResult {
    let value = unsafe { pyre_object::w_float_get_value(obj) };
    if value.is_finite() {
        return append_python_string(out, base_repr(obj, &pyre_object::FLOAT_TYPE)?);
    }
    let text = if value.is_nan() {
        "NaN"
    } else if value.is_sign_positive() {
        "Infinity"
    } else {
        "-Infinity"
    };
    if !crate::baseobjspace::is_true(encoder_attr(self_obj, "allow_nan")?)? {
        let repr = base_repr(obj, &pyre_object::FLOAT_TYPE)?;
        let repr = crate::baseobjspace::text_w(repr)?.to_owned();
        return Err(PyError::value_error(format!(
            "Out of range float values are not JSON compliant: {repr}"
        )));
    }
    out.push_str(text);
    Ok(pyre_object::w_none())
}

/// Attach the PEP 678 context notes emitted by Python 3.14's
/// `json.encoder._make_iterencode`.  The original exception remains
/// authoritative if a pathological `add_note` override itself fails.
fn add_json_note(mut err: PyError, note: impl Into<rustpython_wtf8::Wtf8Buf>) -> PyError {
    let _roots = gc_roots::push_roots();
    let exc = err.to_exc_object();
    let exc_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(exc);
    // The note quotes a key the caller supplied, which may hold a lone
    // surrogate, so it is carried as the WTF-8 it is.
    let note = pyre_object::w_str_from_wtf8_managed(note.into());
    let note_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(note);
    if let Ok(add_note) =
        crate::baseobjspace::getattr_str(gc_roots::shadow_stack_get(exc_slot), "add_note")
    {
        let _ = crate::call::call_function_impl_result(
            add_note,
            &[gc_roots::shadow_stack_get(note_slot)],
        );
    }
    err.exc_object = gc_roots::shadow_stack_get(exc_slot);
    err
}

fn short_type_name(obj: PyObjectRef) -> String {
    crate::baseobjspace::object_functionstr_type_name(obj)
}

fn encode_child(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    // RPython inserts a stack check on this recursive `_encode` edge;
    // `encoder_listencode_obj` brackets the equivalent recursion with
    // `Py_EnterRecursiveCall`.
    crate::stack_check::stack_check()?;
    let encoder = W_Encoder::from_obj(self_obj).expect("Encoder payload");
    let depth = encoder.depth;
    if depth >= crate::stack_check::get_recursion_limit() as i64 {
        return Err(PyError::recursion_error(
            "maximum recursion depth exceeded while encoding a JSON object",
        ));
    }
    encoder.depth = depth + 1;
    let result = encode_value(self_obj, obj, level);
    W_Encoder::from_obj(self_obj)
        .expect("Encoder payload")
        .depth = depth;
    result
}

fn marker_key(obj: PyObjectRef) -> PyObjectRef {
    crate::function::immutable_unique_id(obj)
        .unwrap_or_else(|| pyre_object::w_int_new(obj as usize as i64))
}

fn with_marker<F>(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    encode: F,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError>
where
    F: FnOnce(PyObjectRef, PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, PyError>,
{
    let markers = encoder_attr(self_obj, "markers")?;
    if unsafe { pyre_object::is_none(markers) } {
        return encode(self_obj, obj);
    }
    // PyPy W_Encoder stores the `space.id(w_obj)` key in the caller-provided
    // markers dict.  Keep that Python key rooted across callbacks and reuse it
    // for deletion; no parallel Rust identity table is introduced.
    let _roots = gc_roots::push_roots();
    let markers_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(markers);
    let self_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(self_obj);
    let obj_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(obj);
    let key = marker_key(gc_roots::shadow_stack_get(obj_slot));
    let key_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(key);
    if crate::baseobjspace::contains(
        gc_roots::shadow_stack_get(markers_slot),
        gc_roots::shadow_stack_get(key_slot),
    )? {
        return Err(PyError::value_error("Circular reference detected"));
    }
    crate::baseobjspace::setitem(
        gc_roots::shadow_stack_get(markers_slot),
        gc_roots::shadow_stack_get(key_slot),
        gc_roots::shadow_stack_get(obj_slot),
    )?;
    let result = encode(
        gc_roots::shadow_stack_get(self_slot),
        gc_roots::shadow_stack_get(obj_slot),
    );
    let delete = crate::baseobjspace::delitem(
        gc_roots::shadow_stack_get(markers_slot),
        gc_roots::shadow_stack_get(key_slot),
    );
    match (result, delete) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn encode_value(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    unsafe {
        if pyre_object::is_none(obj) {
            out.push_str("null");
        } else if pyre_object::is_bool(obj) {
            out.push_str(if pyre_object::w_bool_get_value(obj) {
                "true"
            } else {
                "false"
            });
        } else if is_instance(obj, &pyre_object::STR_TYPE) {
            encode_string_field(self_obj, &mut out, obj)?;
        } else if is_instance(obj, &pyre_object::INT_TYPE) {
            append_python_string(&mut out, base_repr(obj, &pyre_object::INT_TYPE)?)?;
        } else if is_instance(obj, &pyre_object::FLOAT_TYPE) {
            encode_float(self_obj, &mut out, obj)?;
        } else if is_instance(obj, &pyre_object::LIST_TYPE)
            || is_instance(obj, &pyre_object::TUPLE_TYPE)
        {
            return with_marker(self_obj, obj, |self_obj, obj| {
                encode_sequence(self_obj, obj, level)
            });
        } else if is_instance(obj, &pyre_object::DICT_TYPE) {
            return with_marker(self_obj, obj, |self_obj, obj| {
                encode_dict(self_obj, obj, level)
            });
        } else {
            return with_marker(self_obj, obj, |self_obj, obj| {
                let default = encoder_attr(self_obj, "default")?;
                // `_default` exceptions propagate bare; errors while
                // encoding its returned object gain context for the source.
                let converted = crate::call::call_function_impl_result(default, &[obj])?;
                encode_child(self_obj, converted, level).map_err(|err| {
                    add_json_note(
                        err,
                        format!("when serializing {} object", short_type_name(obj)),
                    )
                })
            });
        }
    }
    Ok(out)
}

fn indent_value(self_obj: PyObjectRef) -> Result<Option<&'static Wtf8>, PyError> {
    let indent = encoder_attr(self_obj, "indent")?;
    if unsafe { pyre_object::is_none(indent) } {
        Ok(None)
    } else {
        Ok(Some(require_string(indent)?))
    }
}

fn append_indent(out: &mut rustpython_wtf8::Wtf8Buf, indent: &Wtf8, level: i64) {
    out.push_char('\n');
    for _ in 0..level.max(0) {
        out.push_wtf8(indent);
    }
}

fn encode_sequence(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    let _roots = gc_roots::push_roots();
    let iter = crate::baseobjspace::iter(obj)?;
    let iter_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(iter);
    let separator_obj = encoder_attr(self_obj, "item_separator")?;
    let separator = require_string(separator_obj)?.to_wtf8_buf();
    let indent = indent_value(self_obj)?.map(Wtf8::to_wtf8_buf);
    let child_level = level.max(0) + 1;
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_char('[');
    let mut first = true;
    let mut item_index = 0usize;
    loop {
        let item = match crate::baseobjspace::next(gc_roots::shadow_stack_get(iter_slot)) {
            Ok(item) => item,
            Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
            Err(err) => return Err(err),
        };
        if first {
            first = false;
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        } else {
            out.push_wtf8(&separator);
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        }
        let encoded = encode_child(self_obj, item, child_level).map_err(|err| {
            add_json_note(
                err,
                format!(
                    "when serializing {} item {item_index}",
                    short_type_name(obj)
                ),
            )
        })?;
        out.push_wtf8(&encoded);
        item_index += 1;
    }
    if !first && let Some(indent) = &indent {
        append_indent(&mut out, indent, level.max(0));
    }
    out.push_char(']');
    Ok(out)
}

fn coerce_key(self_obj: PyObjectRef, key: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if is_instance(key, &pyre_object::STR_TYPE) {
            return Ok(Some(key));
        }
        if pyre_object::is_bool(key) {
            return Ok(Some(pyre_object::w_str_new(
                if pyre_object::w_bool_get_value(key) {
                    "true"
                } else {
                    "false"
                },
            )));
        }
        if pyre_object::is_none(key) {
            return Ok(Some(pyre_object::w_str_new("null")));
        }
        if is_instance(key, &pyre_object::INT_TYPE) {
            return Ok(Some(base_repr(key, &pyre_object::INT_TYPE)?));
        }
        if is_instance(key, &pyre_object::FLOAT_TYPE) {
            let mut out = rustpython_wtf8::Wtf8Buf::new();
            encode_float(self_obj, &mut out, key)?;
            return Ok(Some(pyre_object::w_str_from_wtf8(out)));
        }
    }
    if crate::baseobjspace::is_true(encoder_attr(self_obj, "skipkeys")?)? {
        Ok(None)
    } else {
        Err(PyError::type_error(format!(
            "keys must be str, int, float, bool or None, not {}",
            crate::baseobjspace::object_functionstr_type_name(key)
        )))
    }
}

fn encode_dict(
    self_obj: PyObjectRef,
    obj: PyObjectRef,
    level: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    // CPython's encoder iterates `items()`.  Keeping the returned Python
    // iterable as the owner makes mutations from re-entrant key encoders
    // visible and avoids holding raw pointers into a mutable list.
    let items = crate::call::call_function_impl_result(
        crate::baseobjspace::getattr_str(obj, "items")?,
        &[],
    )?;
    let items = if crate::baseobjspace::is_true(encoder_attr(self_obj, "sort_keys")?)? {
        let builtins = crate::importing::get_sys_module("builtins")
            .ok_or_else(|| PyError::runtime_error("builtins module is unavailable"))?;
        let sorted = crate::baseobjspace::getattr_str(builtins, "sorted")?;
        crate::call::call_function_impl_result(sorted, &[items])?
    } else {
        items
    };
    let _roots = gc_roots::push_roots();
    let iter = crate::baseobjspace::iter(items)?;
    let iter_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(iter);
    let item_separator = require_string(encoder_attr(self_obj, "item_separator")?)?.to_wtf8_buf();
    let key_separator = require_string(encoder_attr(self_obj, "key_separator")?)?.to_wtf8_buf();
    let indent = indent_value(self_obj)?.map(Wtf8::to_wtf8_buf);
    let child_level = level.max(0) + 1;
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_char('{');
    let mut first = true;
    loop {
        let pair = match crate::baseobjspace::next(gc_roots::shadow_stack_get(iter_slot)) {
            Ok(pair) => pair,
            Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
            Err(err) => return Err(err),
        };
        let pair_items = crate::builtins::collect_iterable(pair)?;
        if pair_items.len() != 2 {
            return Err(PyError::value_error(
                "dictionary update sequence element has length other than 2",
            ));
        }
        // `collect_iterable` hands back a plain `Vec`, and both `coerce_key`
        // and the key encoding below run arbitrary Python — a custom encoder
        // can drop the iterable a dict subclass's `items()` returned and move
        // this pair.  Root both halves before the first of those calls.
        let _pair_roots = gc_roots::push_roots();
        let pair_slot = gc_roots::shadow_stack_len();
        gc_roots::pin_root(pair_items[0]);
        gc_roots::pin_root(pair_items[1]);
        let Some(key) = coerce_key(self_obj, gc_roots::shadow_stack_get(pair_slot))? else {
            continue;
        };
        gc_roots::pin_root(key);
        if first {
            first = false;
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        } else {
            out.push_wtf8(&item_separator);
            if let Some(indent) = &indent {
                append_indent(&mut out, indent, child_level);
            }
        }
        encode_string_field(
            self_obj,
            &mut out,
            gc_roots::shadow_stack_get(pair_slot + 2),
        )?;
        out.push_wtf8(&key_separator);
        let encoded = encode_child(
            self_obj,
            gc_roots::shadow_stack_get(pair_slot + 1),
            child_level,
        )
        .map_err(|err| {
            let key_repr =
                unsafe { crate::display::py_repr_wtf8(gc_roots::shadow_stack_get(pair_slot + 2)) }
                    .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<?>".to_owned()));
            add_json_note(
                err,
                crate::display::wtf8_format!(
                    format!("when serializing {} item ", short_type_name(obj)),
                    key_repr
                ),
            )
        })?;
        out.push_wtf8(&encoded);
    }
    if !first && let Some(indent) = &indent {
        append_indent(&mut out, indent, level.max(0));
    }
    out.push_char('}');
    Ok(out)
}

fn encoder_call_impl(self_obj: PyObjectRef, obj: PyObjectRef, level: i64) -> PyResult {
    let encoded = encode_value(self_obj, obj, level.max(0))?;
    let _roots = gc_roots::push_roots();
    let encoded_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(encoded));
    Ok(pyre_object::w_list_new(vec![gc_roots::shadow_stack_get(
        encoded_slot,
    )]))
}

#[allow(clippy::too_many_arguments)]
fn make_encoder_impl(
    markers: PyObjectRef,
    default: PyObjectRef,
    encoder: PyObjectRef,
    indent: PyObjectRef,
    key_separator: PyObjectRef,
    item_separator: PyObjectRef,
    sort_keys: PyObjectRef,
    skipkeys: PyObjectRef,
    allow_nan: PyObjectRef,
) -> PyResult {
    if !unsafe { pyre_object::is_none(markers) } && !is_instance(markers, &pyre_object::DICT_TYPE) {
        return Err(PyError::type_error(format!(
            "make_encoder() argument 1 must be dict or None, not {}",
            crate::baseobjspace::object_functionstr_type_name(markers)
        )));
    }
    require_string(key_separator)?;
    require_string(item_separator)?;
    if !unsafe { pyre_object::is_none(indent) } {
        require_string(indent)?;
    }
    let sort_keys = pyre_object::w_bool_from(crate::baseobjspace::is_true(sort_keys)?);
    let skipkeys = pyre_object::w_bool_from(crate::baseobjspace::is_true(skipkeys)?);
    let allow_nan = pyre_object::w_bool_from(crate::baseobjspace::is_true(allow_nan)?);

    let module = crate::importing::get_sys_module("_json")
        .ok_or_else(|| PyError::runtime_error("_json module is unavailable"))?;
    let ascii = crate::baseobjspace::getattr_str(module, "encode_basestring_ascii")?;
    let unicode = crate::baseobjspace::getattr_str(module, "encode_basestring")?;
    let fast_mode = if encoder == ascii {
        0
    } else if encoder == unicode {
        1
    } else {
        2
    };

    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    for value in [
        markers,
        default,
        encoder,
        indent,
        key_separator,
        item_separator,
        sort_keys,
        skipkeys,
        allow_nan,
    ] {
        gc_roots::pin_root(value);
    }
    let _ = encoder_class::type_object();
    Ok(W_Encoder::allocate_stable(W_Encoder {
        ob: pyre_object::PyObject::default(),
        markers: gc_roots::shadow_stack_get(slot),
        default: gc_roots::shadow_stack_get(slot + 1),
        encoder: gc_roots::shadow_stack_get(slot + 2),
        indent: gc_roots::shadow_stack_get(slot + 3),
        key_separator: gc_roots::shadow_stack_get(slot + 4),
        item_separator: gc_roots::shadow_stack_get(slot + 5),
        sort_keys: gc_roots::shadow_stack_get(slot + 6),
        skipkeys: gc_roots::shadow_stack_get(slot + 7),
        allow_nan: gc_roots::shadow_stack_get(slot + 8),
        fast_mode,
        depth: 0,
    }))
}

fn make_scanner_impl(context: PyObjectRef) -> PyResult {
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(context);
    for name in [
        "strict",
        "parse_float",
        "parse_int",
        "parse_constant",
        "object_hook",
        "object_pairs_hook",
    ] {
        let value = crate::baseobjspace::getattr_str(gc_roots::shadow_stack_get(slot), name)?;
        gc_roots::pin_root(value);
    }
    // Match the C accelerator: strict is converted during construction, so
    // an overriding `__bool__` raises before the first token is scanned.
    let strict = crate::baseobjspace::is_true(gc_roots::shadow_stack_get(slot + 1))?;
    let _ = scanner_class::type_object();
    Ok(W_Scanner::allocate_stable(W_Scanner {
        ob: pyre_object::PyObject::default(),
        strict,
        parse_float: gc_roots::shadow_stack_get(slot + 2),
        parse_int: gc_roots::shadow_stack_get(slot + 3),
        parse_constant: gc_roots::shadow_stack_get(slot + 4),
        object_hook: gc_roots::shadow_stack_get(slot + 5),
        object_pairs_hook: gc_roots::shadow_stack_get(slot + 6),
    }))
}

crate::py_module! {
    "_json",
    inline_functions: {
        fn make_scanner(context: PyObjectRef) -> Result<PyObjectRef, PyError> {
            make_scanner_impl(context)
        }

        fn scanstring(
            string: PyObjectRef,
            end: PyObjectRef,
            #[default(pyre_object::w_bool_from(true))] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let end = index_i64(end)?;
            scanstring_impl(string, end, strict)
        }

        fn encode_basestring(string: PyObjectRef) -> Result<PyObjectRef, PyError> {
            encode_basestring_impl(string, false)
        }

        fn encode_basestring_ascii(string: PyObjectRef) -> Result<PyObjectRef, PyError> {
            encode_basestring_impl(string, true)
        }

        fn make_encoder(
            markers: PyObjectRef,
            default: PyObjectRef,
            encoder: PyObjectRef,
            indent: PyObjectRef,
            key_separator: PyObjectRef,
            item_separator: PyObjectRef,
            sort_keys: PyObjectRef,
            skipkeys: PyObjectRef,
            allow_nan: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            make_encoder_impl(markers, default, encoder, indent, key_separator,
                item_separator, sort_keys, skipkeys, allow_nan)
        }
    }
}
