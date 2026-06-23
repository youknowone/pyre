//! Builtin type method implementations.
//!
//! PyPy equivalents:
//!   pypy/objspace/std/listobject.py  (list methods)
//!   pypy/objspace/std/unicodeobject.py  (str methods)
//!   pypy/objspace/std/dictmultiobject.py  (dict methods)
//!   pypy/objspace/std/tupleobject.py  (tuple methods)
//!
//! Separated from space.rs to avoid bloating the hot-path compilation
//! unit. Method functions are registered into TypeDef at startup.

use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

// ── List methods ─────────────────────────────────────────────────────
// All take self (list) as first arg.

pub fn list_method_append(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "append() takes exactly one argument");
    unsafe { w_list_append(args[0], args[1]) };
    Ok(w_none())
}

pub fn list_method_extend(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    let list = args[0];
    let other = args[1];
    unsafe {
        if is_list(other) {
            let n = w_list_len(other);
            for i in 0..n {
                if let Some(item) = w_list_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else if is_tuple(other) {
            let n = w_tuple_len(other);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else {
            // listobject.py:1019 _extend_from_iterable
            let items = crate::builtins::collect_iterable(other)?;
            for item in items {
                w_list_append(list, item);
            }
        }
    }
    Ok(w_none())
}

/// PyPy: listobject.py descr_insert — list.insert(index, item)
pub fn list_method_insert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 3, "insert() takes exactly 2 arguments");
    let index = unsafe { w_int_get_value(args[1]) };
    unsafe { pyre_object::listobject::w_list_insert(args[0], index, args[2]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_pop — list.pop([index])
/// listobject.py:759-772 — empty list raises "pop from empty list",
/// otherwise out-of-range raises "pop index out of range".
pub fn list_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "pop() requires self");
    let index = if args.len() > 1 {
        unsafe { w_int_get_value(args[1]) }
    } else {
        -1
    };
    let length = unsafe { pyre_object::w_list_len(args[0]) } as i64;
    if length == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop from empty list",
        ));
    }
    match unsafe { pyre_object::listobject::w_list_pop(args[0], index) } {
        Some(v) => Ok(v),
        None => Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop index out of range",
        )),
    }
}

/// PyPy: listobject.py descr_clear — list.clear()
pub fn list_method_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    unsafe { pyre_object::listobject::w_list_clear(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_copy — list.copy()
pub fn list_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let list = args[0];
    unsafe {
        let n = w_list_len(list);
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
        Ok(w_list_new(items))
    }
}

/// PyPy: listobject.py descr_reverse — list.reverse()
pub fn list_method_reverse(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    unsafe { pyre_object::listobject::w_list_reverse(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_sort — list.sort()
pub fn list_method_sort(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let list = args[0];
    // listobject.py `descr_sort` shares `rpython/rlib/listsort.py`'s comparison
    // machinery (general `space.lt`, `key=`, `reverse=`) with `sorted()`.
    // The flat builtin ABI hands us `[self, kwargs?]`, exactly the shape
    // `sorted()` expects, so produce the sorted sequence through the same
    // path and copy it back into the list in place.
    let sorted_list = crate::builtins::builtin_sorted(args)?;
    unsafe {
        let n = w_list_len(sorted_list);
        let items: Vec<PyObjectRef> = (0..n)
            .filter_map(|i| w_list_getitem(sorted_list, i as i64))
            .collect();
        pyre_object::listobject::w_list_clear(list);
        for item in items {
            w_list_append(list, item);
        }
    }
    Ok(w_none())
}

/// listobject.py:795 `descr_index` — list.index(value[, start[, stop]]).
pub fn list_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "index() takes at least 1 argument");
    let list = args[0];
    let value = args[1];
    // listobject.py:799 unwrap_spec defaults: w_start=0 / w_stop=sys.maxint.
    // listobject.py:803 unwrap_start_stop handles negative normalization,
    // __index__ coercion and TypeError for non-index arguments.
    let size = unsafe { pyre_object::w_list_len(list) } as i64;
    let w_start = if args.len() >= 3 {
        args[2]
    } else {
        w_int_new(0)
    };
    let w_stop = if args.len() >= 4 {
        args[3]
    } else {
        w_int_new(i64::MAX)
    };
    let (start, stop) = crate::sliceobject::unwrap_start_stop(size, w_start, w_stop)?;
    match crate::listobject::w_list_find_or_count(list, value, start, stop, false)? {
        crate::listobject::FindOrCountResult::Index(i) => Ok(w_int_new(i)),
        crate::listobject::FindOrCountResult::NotFound => Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "list.index(x): x not in list".to_string(),
        )),
        crate::listobject::FindOrCountResult::Count(_) => {
            unreachable!("find_or_count with count=false never returns Count")
        }
    }
}

/// listobject.py:744 `descr_count` — list.count(value)
pub fn list_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "count() takes exactly 1 argument");
    let list = args[0];
    let value = args[1];
    match crate::listobject::w_list_find_or_count(list, value, 0, i64::MAX, true)? {
        crate::listobject::FindOrCountResult::Count(n) => Ok(w_int_new(n)),
        crate::listobject::FindOrCountResult::NotFound => Ok(w_int_new(0)),
        crate::listobject::FindOrCountResult::Index(_) => {
            unreachable!("find_or_count with count=true never returns Index")
        }
    }
}

/// listobject.py:782 `descr_remove` — list.remove(value).
pub fn list_method_remove(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "remove() takes exactly 1 argument");
    crate::listobject::w_list_remove(args[0], args[1])?;
    Ok(w_none())
}

// ── String methods ───────────────────────────────────────────────────

pub fn str_method_join(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let iterable = args[1];
    let items: Vec<PyObjectRef> = unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            (0..n)
                .filter_map(|i| w_list_getitem(iterable, i as i64))
                .collect()
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            (0..n)
                .filter_map(|i| w_tuple_getitem(iterable, i as i64))
                .collect()
        } else {
            crate::builtins::collect_iterable(iterable)?
        }
    };
    // pypy/objspace/std/unicodeobject.py:856-872 descr_join — each
    // element must be a str; otherwise TypeError("sequence item N:
    // expected str instance, <T> found"). Silently dropping non-str
    // items lost the error and produced an empty join.
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    for (i, item) in items.iter().enumerate() {
        if unsafe { !is_str(*item) } {
            return Err(crate::PyError::type_error(format!(
                "sequence item {i}: expected str instance, {} found",
                unsafe { (*(*(*item)).ob_type).name }
            )));
        }
        if i > 0 {
            out.push_wtf8(sep);
        }
        out.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(*item) });
    }
    Ok(pyre_object::w_str_from_wtf8(out))
}

/// `str.split` / `str.rsplit` take `sep` and `maxsplit` positionally or by
/// keyword.  Builtin kwargs arrive as a trailing `__pyre_kw__` dict, so
/// resolve each argument from its positional slot (after the receiver),
/// falling back to the matching keyword.
fn resolve_split_args(
    args: &[PyObjectRef],
    fn_name: &str,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["sep", "maxsplit"], fn_name)?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "sep", pos.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "maxsplit", pos.get(2).is_some())?;
    let sep = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"))
        .unwrap_or(pyre_object::PY_NULL);
    let maxsplit = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "maxsplit"))
        .unwrap_or(pyre_object::PY_NULL);
    Ok((sep, maxsplit))
}

/// Box a code-point slice into a str object.
fn cps_to_str(cps: &[CodePoint]) -> PyObjectRef {
    let mut buf = Wtf8Buf::with_capacity(cps.len());
    for &cp in cps {
        buf.push(cp);
    }
    w_str_from_wtf8(buf)
}

/// A lone surrogate is not whitespace.
fn cp_is_whitespace(cp: CodePoint) -> bool {
    match cp.to_char() {
        Some(c) => c.is_whitespace(),
        None => false,
    }
}

/// `str.split()` with no separator: split on runs of whitespace,
/// dropping leading/trailing runs.  When `maxsplit >= 0`, after that
/// many splits the rest (leading whitespace stripped) is one tail token.
fn wtf8_split_whitespace(s: &Wtf8, maxsplit: i64) -> Vec<PyObjectRef> {
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut out: Vec<PyObjectRef> = Vec::new();
    let mut i = 0usize;
    loop {
        if maxsplit >= 0 && out.len() as i64 >= maxsplit {
            break;
        }
        while i < cps.len() && cp_is_whitespace(cps[i]) {
            i += 1;
        }
        if i == cps.len() {
            break;
        }
        let start = i;
        while i < cps.len() && !cp_is_whitespace(cps[i]) {
            i += 1;
        }
        out.push(cps_to_str(&cps[start..i]));
    }
    while i < cps.len() && cp_is_whitespace(cps[i]) {
        i += 1;
    }
    if i < cps.len() {
        out.push(cps_to_str(&cps[i..]));
    }
    out
}

/// `str.rsplit()` with no separator: like `wtf8_split_whitespace` but
/// scanning from the right, so the tail token is the leading remainder.
fn wtf8_rsplit_whitespace(s: &Wtf8, maxsplit: i64) -> Vec<PyObjectRef> {
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut tokens: Vec<PyObjectRef> = Vec::new();
    let mut i = cps.len();
    loop {
        if maxsplit >= 0 && tokens.len() as i64 >= maxsplit {
            break;
        }
        while i > 0 && cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        let end = i;
        while i > 0 && !cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        tokens.push(cps_to_str(&cps[i..end]));
    }
    tokens.reverse();
    let mut prefix_end = i;
    while prefix_end > 0 && cp_is_whitespace(cps[prefix_end - 1]) {
        prefix_end -= 1;
    }
    if prefix_end > 0 {
        let mut out = vec![cps_to_str(&cps[..prefix_end])];
        out.extend(tokens);
        out
    } else {
        tokens
    }
}

pub fn str_method_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let (sep_arg, maxsplit_arg) = resolve_split_args(args, "split")?;
    let sep = parse_split_sep(sep_arg)?;
    // `unicodeobject.py:972 @unwrap_spec(maxsplit=int) descr_split` —
    // `space.int_w(w_maxsplit)` routes through `__index__`, so any
    // int-like object (subclass, numpy int, etc.) is accepted.
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    let parts: Vec<PyObjectRef> = match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            if maxsplit < 0 {
                s.split(sep)
                    .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                    .collect()
            } else {
                s.splitn((maxsplit as usize) + 1, sep)
                    .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                    .collect()
            }
        }
        None => wtf8_split_whitespace(s, maxsplit),
    };
    Ok(w_list_new(parts))
}

/// `pypy/objspace/std/unicodeobject.py:992-994 W_UnicodeObject
/// .convert_arg_to_w_unicode` parity — `sep` must be `None` or a
/// `str`; anything else surfaces a TypeError at the same call site
/// where PyPy's `space.unicode_w` would.
fn parse_split_sep(value: PyObjectRef) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if value.is_null() || unsafe { is_none(value) } {
        return Ok(None);
    }
    if unsafe { is_str(value) } {
        return Ok(Some(unsafe { w_str_get_wtf8(value) }.to_wtf8_buf()));
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(value) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "must be str or None, not {tp_name}"
    )))
}

/// `unicodeobject.py:972 @unwrap_spec(maxsplit=int)` parity —
/// `space.int_w(w_maxsplit)` routes through `__index__`, so any
/// int-like object is accepted; an absent maxsplit defaults to -1
/// (unlimited).  An explicit `None` is not int-like and has no
/// `__index__`, so it raises `TypeError` like any other non-integer.
fn parse_split_maxsplit(value: PyObjectRef) -> Result<i64, crate::PyError> {
    if value.is_null() {
        return Ok(-1);
    }
    crate::builtins::space_index_w(value)
}

/// `pypy/objspace/std/unicodeobject.py:993-1024 W_UnicodeObject
/// .descr_rsplit`.  Mirrors `split` semantics in reverse — when
/// `maxsplit` is positive, only the rightmost `maxsplit` separators
/// participate.  Argument validation follows the same
/// `@unwrap_spec(maxsplit=int)` + `convert_arg_to_w_unicode` shape
/// as `descr_split`.
pub fn str_method_rsplit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let (sep_arg, maxsplit_arg) = resolve_split_args(args, "rsplit")?;
    let sep = parse_split_sep(sep_arg)?;
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    let parts: Vec<PyObjectRef> = match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path —
            // mirrors the forward `split` rejection.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            let mut out: Vec<&Wtf8> = if maxsplit < 0 {
                s.rsplit(sep).collect()
            } else {
                s.rsplitn((maxsplit as usize) + 1, sep).collect()
            };
            out.reverse();
            out.into_iter()
                .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                .collect()
        }
        None => wtf8_rsplit_whitespace(s, maxsplit),
    };
    Ok(w_list_new(parts))
}

/// `pypy/objspace/std/unicodeobject.py:767-770 W_UnicodeObject.descr_casefold`:
///
/// ```python
/// def descr_casefold(self, space):
///     value = self._value
///     return space.newutf8(unicode_casefold(value), -1)
/// ```
///
/// PyPy delegates to `rpython.rlib.runicode.unicode_casefold` which
/// applies the full Unicode `CaseFolding.txt` mapping (status C +
/// status F: ß → ss, ﬁ → fi, İ → i + combining dot,
/// Lithuanian Į → i̇ǫ, the Greek sigma, etc.).  Pyre routes through
/// the `caseless` crate's `default_case_fold_str` which is itself
/// generated from `CaseFolding.txt` status-C+F entries, so the
/// surface matches `unicode_casefold` for every Unicode code point.
pub fn str_method_casefold(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    // Fast path: no lone surrogate, fold the whole &str at once.
    if let Ok(valid) = s.as_str() {
        return Ok(w_str_new(&caseless::default_case_fold_str(valid)));
    }
    // Surrogate path: fold each scalar code point, pass surrogates through.
    let out = wtf8_map_chars(s, |c, out| {
        let mut buf = [0u8; 4];
        out.push_str(&caseless::default_case_fold_str(c.encode_utf8(&mut buf)));
    });
    Ok(w_str_from_wtf8(out))
}

#[allow(dead_code)]
fn casefold_basic(s: &str) -> String {
    // Multi-char expansion casefolds the Unicode `Final_Sigma` /
    // `Lt`/`Lu` cases pyre's interpreter test corpus actually
    // touches.  Any code point outside this whitelist falls back
    // to Rust's `char::to_lowercase` which matches CPython for the
    // overwhelming majority of letters.
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'ß' => out.push_str("ss"),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' => out.push_str("st"),
            'ﬆ' => out.push_str("st"),
            'İ' => out.push_str("i\u{0307}"),
            _ => {
                for lower in ch.to_lowercase() {
                    out.push(lower);
                }
            }
        }
    }
    out
}

/// `pypy/objspace/std/unicodeobject.py:429-430 W_UnicodeObject
/// .descr_format_map` parity:
///
/// ```python
/// def descr_format_map(self, space, w_mapping):
///     return newformat.format_method(space, self, None, w_mapping, True)
/// ```
///
/// PyPy passes the mapping straight through to `format_method`; each
/// `{name}` field is then resolved by `space.getitem(mapping, w_key)`
/// at format-time, so the mapping is consulted *lazily*.  This
/// matters for mappings with side-effecting `__getitem__`, a
/// `__missing__` hook, or no `keys()` — pre-materialising via
/// `keys()` (the previous implementation) breaks `defaultdict`,
/// custom `Mapping` subclasses, and any object that only implements
/// `__getitem__`.
pub fn str_method_format_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let fmt = args[0];
    let mapping = args[1];
    str_method_format_core(fmt, &[], None, Some(mapping))
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject._strip` —
/// `s.strip([chars])`.  When `chars` is missing or None, defaults to
/// ASCII whitespace (the `str::trim` set).  When provided, removes
/// any character contained in `chars` from each end (NOT a substring
/// match — `'aabaa'.strip('a') == 'b'`).
fn strip_chars(s: &Wtf8, chars: Option<&Wtf8>, left: bool, right: bool) -> Wtf8Buf {
    let chars_set: Option<Vec<CodePoint>> = chars.map(|c| c.code_points().collect());
    let mut current: &Wtf8 = s;
    if left {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_start_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_start(),
        };
    }
    if right {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_end_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_end(),
        };
    }
    current.to_wtf8_buf()
}

/// `pypy/objspace/std/unicodeobject.py:1464-1473 W_UnicodeObject
/// ._strip` — extract the optional `chars` argument as a `&str`,
/// raising TypeError on non-str non-None arguments rather than
/// silently falling through to the whitespace default.
fn extract_strip_chars(arg: PyObjectRef, fn_name: &str) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if arg.is_null() || unsafe { pyre_object::is_none(arg) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_str(arg) } {
        return Ok(Some(
            unsafe { pyre_object::w_str_get_wtf8(arg) }.to_wtf8_buf(),
        ));
    }
    Err(crate::PyError::type_error(format!(
        "{fn_name} arg must be None or str"
    )))
}

pub fn str_method_strip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "strip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        true,
        true,
    )))
}

pub fn str_method_lstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "lstrip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        true,
        false,
    )))
}

pub fn str_method_rstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "rstrip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        false,
        true,
    )))
}

/// `unicodeobject.py descr_startswith` — accepts either a single str
/// prefix or a tuple of str prefixes (CPython parity).
/// unicodeobject.py:848 descr_startswith(self, prefix, start=0, end=sys.maxsize)
pub fn str_method_startswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let slice = str_slice_args(s, args);
    str_prefix_match(slice, args[1], "startswith", true).map(w_bool_from)
}

pub fn str_method_endswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let slice = str_slice_args(s, args);
    str_prefix_match(slice, args[1], "endswith", false).map(w_bool_from)
}

fn str_slice_args<'a>(s: &'a Wtf8, args: &[pyre_object::PyObjectRef]) -> &'a Wtf8 {
    let char_len = s.code_points().count() as i64;
    let start = if args.len() >= 3 {
        let v = unsafe { pyre_object::w_int_get_value(args[2]) };
        if v < 0 {
            (char_len + v).max(0) as usize
        } else {
            (v as usize).min(char_len as usize)
        }
    } else {
        0
    };
    let end = if args.len() >= 4 {
        let v = unsafe { pyre_object::w_int_get_value(args[3]) };
        if v < 0 {
            (char_len + v).max(0) as usize
        } else {
            (v as usize).min(char_len as usize)
        }
    } else {
        char_len as usize
    };
    let empty = unsafe { Wtf8::from_bytes_unchecked(&[]) };
    if start > end {
        return empty;
    }
    let bytes = s.as_bytes();
    let byte_start = s
        .code_point_indices()
        .nth(start)
        .map_or(bytes.len(), |(i, _)| i);
    let byte_end = s
        .code_point_indices()
        .nth(end)
        .map_or(bytes.len(), |(i, _)| i);
    unsafe { Wtf8::from_bytes_unchecked(&bytes[byte_start..byte_end]) }
}

fn str_prefix_match(
    s: &Wtf8,
    needle: PyObjectRef,
    method: &str,
    start: bool,
) -> Result<bool, crate::PyError> {
    let h = s.as_bytes();
    // WTF-8 is self-synchronizing, so a byte-level prefix/suffix match
    // coincides with a code-point-level one.
    let test = |p: &Wtf8| {
        let p = p.as_bytes();
        if start {
            h.starts_with(p)
        } else {
            h.ends_with(p)
        }
    };
    if unsafe { pyre_object::is_str(needle) } {
        let p = unsafe { pyre_object::w_str_get_wtf8(needle) };
        return Ok(test(p));
    }
    if unsafe { pyre_object::is_tuple(needle) } {
        let n = unsafe { pyre_object::w_tuple_len(needle) };
        for i in 0..n as i64 {
            let item =
                unsafe { pyre_object::w_tuple_getitem(needle, i) }.expect("index is in range");
            if !unsafe { pyre_object::is_str(item) } {
                return Err(crate::PyError::type_error(format!(
                    "tuple for {method} must only contain str, not {}",
                    unsafe { (*(*item).ob_type).name }
                )));
            }
            let p = unsafe { pyre_object::w_str_get_wtf8(item) };
            if test(p) {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    Err(crate::PyError::type_error(format!(
        "{method} first arg must be str or a tuple of str, not {}",
        unsafe { (*(*needle).ob_type).name }
    )))
}

pub fn str_method_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 3);
    // pypy/objspace/std/unicodeobject.py:1132-1148 descr_replace —
    // both `old` and `new` must be str / W_UnicodeObject; otherwise
    // TypeError("replace() argument N must be str, not ...").
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 1 must be str, not {}",
            unsafe { (*(*args[1]).ob_type).name }
        )));
    }
    if !unsafe { pyre_object::is_str(args[2]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 2 must be str, not {}",
            unsafe { (*(*args[2]).ob_type).name }
        )));
    }
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let old = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
    let new = unsafe { pyre_object::w_str_get_wtf8(args[2]) };
    // `unicodeobject.py descr_replace` — optional count argument; a
    // negative count means "no limit" (matches CPython); 0 leaves the
    // string untouched.
    let maxcount = match args.get(3) {
        Some(&w_count) if unsafe { pyre_object::is_int(w_count) } => unsafe {
            pyre_object::w_int_get_value(w_count)
        },
        _ => -1,
    };
    Ok(w_str_from_wtf8(wtf8_replace(s, old, new, maxcount)))
}

/// WTF-8 window for the optional `start` / `end` search args: resolve them
/// (PyPy slice semantics via `unwrap_start_stop`) into a byte-offset window
/// `(byte_start, byte_end)` into the WTF-8 backing, indexing by code point so
/// a surrogate-bearing string does not panic in `w_str_get_value`.  Returns
/// `None` when the codepoint window is empty because `start` is past the end
/// or past `end` (the search-miss case shared by count).
fn wtf8_idx_window(
    s: &Wtf8,
    args: &[PyObjectRef],
) -> Result<Option<(usize, usize)>, crate::PyError> {
    let cp_len = s.code_points().count() as i64;
    let w_start = if args.len() >= 3 { args[2] } else { w_none() };
    let w_end = if args.len() >= 4 { args[3] } else { w_none() };
    let (start, end) = crate::sliceobject::unwrap_start_stop(cp_len, w_start, w_end)?;
    if start > cp_len {
        return Ok(None);
    }
    let end = end.min(cp_len);
    if start > end {
        return Ok(None);
    }
    let byte_start = s
        .code_point_indices()
        .nth(start as usize)
        .map_or(s.len(), |(i, _)| i);
    let byte_end = s
        .code_point_indices()
        .nth(end as usize)
        .map_or(s.len(), |(i, _)| i);
    Ok(Some((byte_start, byte_end)))
}

pub fn str_method_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    Ok(w_int_new(str_unwrap_and_search(args, true)?.unwrap_or(-1)))
}

pub fn str_method_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    Ok(w_int_new(str_unwrap_and_search(args, false)?.unwrap_or(-1)))
}

pub fn str_method_upper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let out = wtf8_map_chars(s, |c, out| {
        for u in c.to_uppercase() {
            out.push_char(u);
        }
    });
    Ok(w_str_from_wtf8(out))
}

pub fn str_method_lower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let out = wtf8_map_chars(s, |c, out| {
        for l in c.to_lowercase() {
            out.push_char(l);
        }
    });
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_format
/// Requires format spec parser — correct for no-arg case only.
/// `str.format(*args)` — PyPy: unicodeobject.py descr_format → newformat.py
pub fn str_method_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    // `pypy/objspace/std/newformat.py Formatter.format` —
    // positional args are slots 1.. of the receiver; keyword args
    // (`{name}` lookups) live in the trailing CALL_KW dict.
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(&args[1..]);
    str_method_format_core(args[0], positional, kwargs_dict, None)
}

/// Shared core for `str.format` (`{name}` looks up the trailing
/// CALL_KW dict) and `str.format_map` (`{name}` looks up the
/// mapping via `space.getitem(mapping, w_key)`).  PyPy folds both
/// into one `newformat.format_method(space, fmt, args_w, w_kwds, ...)`
/// entry point per `unicodeobject.py:422-430`; pyre splits the
/// keyword-source into "dict snapshot" vs "lazy mapping" so the
/// mapping path stays line-by-line lazy (no pre-materialisation).
fn str_method_format_core(
    fmt_obj: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let fmt = unsafe { pyre_object::w_str_get_value(fmt_obj) };
    let mut auto_idx = 0usize;
    // `newformat.py` auto_numbering_state — `None` = ANS_INIT, `Some(true)`
    // = ANS_AUTO (empty `{}` fields), `Some(false)` = ANS_MANUAL (numbered
    // `{0}` fields).  Mixing the two raises ValueError.
    let mut numbering: Option<bool> = None;
    let rendered = format_render(
        fmt,
        positional,
        kwargs_dict,
        mapping,
        &mut auto_idx,
        &mut numbering,
        2,
    )?;
    Ok(pyre_object::w_str_from_wtf8(rendered))
}

/// `newformat.py Formatter.format` rendering pass.  Renders the
/// template `fmt`, threading the auto-/manual-numbering state through
/// the recursive evaluation of nested `{...}` format specs so that
/// `"{:{}}".format(42, ">5")` consumes positional args 0 then 1.
/// `depth` bounds that recursion (the markup recursion limit is 2).
fn format_render(
    fmt: &str,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
    auto_idx: &mut usize,
    numbering: &mut Option<bool>,
    depth: u32,
) -> Result<Wtf8Buf, crate::PyError> {
    let lookup_kwarg = |name: &str| -> Result<Option<PyObjectRef>, crate::PyError> {
        if let Some(m) = mapping {
            // `newformat.format_method(... w_mapping, True)` resolves
            // `{name}` via `space.getitem(mapping, w_key)` per
            // `newformat.py:Template.get_value`; KeyError propagates
            // to the caller (no silent default).
            let w_key = pyre_object::w_str_new(name);
            return crate::baseobjspace::getitem(m, w_key).map(Some);
        }
        if let Some(dict) = kwargs_dict {
            let v = unsafe { pyre_object::w_dict_lookup(dict, pyre_object::w_str_new(name)) };
            return Ok(v);
        }
        Ok(None)
    };

    let mut result = Wtf8Buf::new();
    let bytes = fmt.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                result.push_char('{');
                i += 2;
                continue;
            }
            i += 1;
            let field_start = i;
            // Track brace depth so a nested `{...}` inside the format
            // spec is captured whole rather than ending the field at the
            // spec's first `}`.
            let mut brace_depth = 1;
            while i < bytes.len() {
                if bytes[i] == b'{' {
                    brace_depth += 1;
                } else if bytes[i] == b'}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        break;
                    }
                }
                i += 1;
            }
            let field_text = &fmt[field_start..i];
            if i < bytes.len() {
                i += 1;
            }

            // `newformat.py:_parse_field` — split the replacement field
            // into the argument name, an optional `!conversion`, and the
            // trailing `:spec`.  A `:` or `!` inside `[...]` is part of an
            // item key, not a separator, so brackets are skipped over.
            let fb = field_text.as_bytes();
            let mut p = 0;
            let mut name_end = fb.len();
            let mut conversion: Option<char> = None;
            let mut spec = String::new();
            while p < fb.len() {
                let c = fb[p];
                if c == b':' || c == b'!' {
                    name_end = p;
                    if c == b'!' {
                        p += 1;
                        if p < fb.len() {
                            conversion = Some(fb[p] as char);
                            p += 1;
                        }
                        if p < fb.len() && fb[p] == b':' {
                            p += 1;
                        }
                    } else {
                        p += 1;
                    }
                    spec = field_text.get(p..).unwrap_or("").to_string();
                    break;
                } else if c == b'[' {
                    while p + 1 < fb.len() && fb[p + 1] != b']' {
                        p += 1;
                    }
                }
                p += 1;
            }
            let name = &field_text[..name_end];

            // `newformat.py:_get_argument` — the argument name is the
            // prefix up to the first `[` or `.`; the remainder is a chain
            // of attribute / item lookups resolved by
            // `resolve_format_lookups`.
            let nb = name.as_bytes();
            let mut k = 0;
            while k < nb.len() && nb[k] != b'[' && nb[k] != b'.' {
                k += 1;
            }
            let base = &name[..k];
            let rest = &name[k..];

            // pypy/objspace/std/newformat.py:1066-1071 Template.get_arg:
            // missing positional → IndexError "Replacement index N out of
            // range for positional args tuple"; missing keyword →
            // KeyError(name).
            let w_arg = if base.is_empty() {
                if let Some(false) = *numbering {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        "cannot switch from manual field specification to automatic \
                         field numbering"
                            .to_string(),
                    ));
                }
                *numbering = Some(true);
                let idx = *auto_idx;
                *auto_idx += 1;
                match positional.get(idx).copied() {
                    Some(v) => v,
                    None => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::IndexError,
                            format!(
                                "Replacement index {idx} out of range for positional args tuple"
                            ),
                        ));
                    }
                }
            } else if let Ok(idx) = base.parse::<usize>() {
                if let Some(true) = *numbering {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        "cannot switch from automatic field numbering to manual \
                         field specification"
                            .to_string(),
                    ));
                }
                *numbering = Some(false);
                match positional.get(idx).copied() {
                    Some(v) => v,
                    None => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::IndexError,
                            format!(
                                "Replacement index {idx} out of range for positional args tuple"
                            ),
                        ));
                    }
                }
            } else {
                match lookup_kwarg(base)? {
                    Some(v) => v,
                    None => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::KeyError,
                            format!("'{base}'"),
                        ));
                    }
                }
            };
            let val = resolve_format_lookups(w_arg, rest)?;

            // `newformat.py:_convert_field` — `!s`/`!r`/`!a` apply
            // str / repr / ascii before the format spec.
            let converted = match conversion {
                None => val,
                // `!s` is `str(self)`, preserved in WTF-8 so a lone
                // surrogate (a str, or an exception with a str argument)
                // passes through unchanged.
                Some('s') => pyre_object::w_str_from_wtf8(unsafe { crate::py_str_wtf8(val)? }),
                Some('r') => pyre_object::w_str_new(&unsafe { crate::py_repr(val)? }),
                Some('a') => pyre_object::w_str_new(&crate::builtins::py_ascii(val)?),
                Some(c) => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        format!("Unknown conversion specifier {c}"),
                    ));
                }
            };
            // A spec containing `{` is itself a template: render it
            // (sharing the numbering state) before applying it.  A
            // rendered spec is expected to be valid text (format specs
            // do not carry surrogates).
            let resolved_spec: String = if spec.bytes().any(|b| b == b'{') {
                if depth == 0 {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        "Max string recursion exceeded".to_string(),
                    ));
                }
                let nested = format_render(
                    &spec,
                    positional,
                    kwargs_dict,
                    mapping,
                    auto_idx,
                    numbering,
                    depth - 1,
                )?;
                match nested.as_str() {
                    Ok(v) => v.to_string(),
                    Err(_) => String::new(),
                }
            } else {
                spec
            };
            let formatted = format_value_dispatch(converted, &resolved_spec)?;
            result.push_wtf8(&formatted);
        } else if bytes[i] == b'}' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
            result.push_char('}');
            i += 2;
        } else if bytes[i] == b'}' {
            // A lone `}` (not `}}`) is emitted literally.
            result.push_char('}');
            i += 1;
        } else {
            // Literal run up to the next brace; `{` / `}` are single
            // ASCII bytes that never occur inside a multi-byte
            // sequence, so the run is itself valid text and copies
            // whole (a byte-at-a-time `as char` would mojibake
            // non-ASCII literals).
            let start = i;
            while i < bytes.len() && bytes[i] != b'{' && bytes[i] != b'}' {
                i += 1;
            }
            result.push_str(&fmt[start..i]);
        }
    }
    Ok(result)
}

/// `newformat.py:_resolve_lookups` — walk the `.attr` / `[element]`
/// suffix of a replacement-field name, resolving each step against
/// `w_obj` via `getattr` / `getitem`.  A bracketed element made only
/// of decimal digits is an integer index; anything else is a string
/// key (`_parse_int` returns -1 for non-numeric elements).
fn resolve_format_lookups(w_obj: PyObjectRef, rest: &str) -> Result<PyObjectRef, crate::PyError> {
    let rb = rest.as_bytes();
    let mut w_obj = w_obj;
    let mut i = 0;
    while i < rb.len() {
        let c = rb[i];
        if c == b'.' {
            i += 1;
            let start = i;
            while i < rb.len() && rb[i] != b'[' && rb[i] != b'.' {
                i += 1;
            }
            if start == i {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::ValueError,
                    "Empty attribute in format string".to_string(),
                ));
            }
            w_obj = crate::baseobjspace::getattr_str(w_obj, &rest[start..i])?;
        } else if c == b'[' {
            i += 1;
            let start = i;
            let mut got_bracket = false;
            while i < rb.len() {
                if rb[i] == b']' {
                    got_bracket = true;
                    break;
                }
                i += 1;
            }
            if got_bracket {
                let elem = &rest[start..i];
                i += 1;
                let numeric = elem.len() > 0 && elem.bytes().all(|b| b.is_ascii_digit());
                let w_item = if numeric {
                    match elem.parse::<i64>() {
                        Ok(idx) => pyre_object::w_int_new(idx),
                        Err(_) => pyre_object::w_str_new(elem),
                    }
                } else {
                    pyre_object::w_str_new(elem)
                };
                w_obj = crate::baseobjspace::getitem(w_obj, w_item)?;
            } else {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::ValueError,
                    "Missing ']' in format string".to_string(),
                ));
            }
        } else {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "Only '[' and '.' may follow ']' in format string".to_string(),
            ));
        }
    }
    Ok(w_obj)
}

/// Mini Python format-spec parser — `pypy/objspace/std/newformat.py
/// _parse_spec`.  Recognises the subset pyre exercises today:
/// `[fill][align][sign][#][0][width][grouping][.precision][type]`.
/// `alt_form` (the `#` flag) is now stored on the parsed spec so int /
/// float formatters can apply the base prefix or trailing-zero
/// preservation per PyPy `newformat.py:454-468`.  `grouping` holds the
/// thousands separator (`,` or `_`) parsed at `newformat.py:501-512`.
struct ParsedSpec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    alt_form: bool,
    zero_pad: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: char,
}

fn parse_spec(spec: &str) -> ParsedSpec {
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let n = chars.len();
    let mut fill = ' ';
    let mut align: Option<char> = None;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = chars[0];
        align = Some(chars[1]);
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = Some(chars[0]);
        i = 1;
    }
    let mut sign: Option<char> = None;
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        sign = Some(chars[i]);
        i += 1;
    }
    let mut alt_form = false;
    if i < n && chars[i] == '#' {
        alt_form = true;
        i += 1;
    }
    let mut zero_pad = false;
    if i < n && chars[i] == '0' {
        zero_pad = true;
        // `pypy/objspace/std/newformat.py:454-460` — the `0` flag
        // implies `fill='0', align='='` when no explicit fill /
        // align were provided.  Without this, `f"{-7:=05d}"`
        // would left-fill with spaces instead of zeros.
        if align.is_none() {
            align = Some('=');
        }
        if fill == ' ' {
            fill = '0';
        }
        i += 1;
    }
    let mut width = 0usize;
    while i < n && chars[i].is_ascii_digit() {
        width = width * 10 + (chars[i] as u8 - b'0') as usize;
        i += 1;
    }
    // `pypy/objspace/std/newformat.py:501-512` — the thousands
    // separator (`,` or `_`) sits between width and `.precision`.
    let mut grouping: Option<char> = None;
    if i < n && matches!(chars[i], ',' | '_') {
        grouping = Some(chars[i]);
        i += 1;
    }
    let mut precision: Option<usize> = None;
    if i < n && chars[i] == '.' {
        i += 1;
        let mut p = 0usize;
        while i < n && chars[i].is_ascii_digit() {
            p = p * 10 + (chars[i] as u8 - b'0') as usize;
            i += 1;
        }
        precision = Some(p);
    }
    let ty = if i < n { chars[i] } else { '\0' };
    ParsedSpec {
        fill,
        align,
        sign,
        alt_form,
        zero_pad,
        width,
        grouping,
        precision,
        ty,
    }
}

/// `pypy/objspace/std/newformat.py:740 _group_digits` — insert the
/// thousands separator `sep` into a run of plain digits every
/// `interval` positions counted from the right (3 for decimal
/// presentations, 4 for `_` on binary / octal / hex).
fn group_digits(digits: &str, sep: char, interval: usize) -> String {
    let chars: Vec<char> = digits.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(len + len / interval);
    for (idx, c) in chars.iter().enumerate() {
        if idx > 0 && (len - idx) % interval == 0 {
            out.push(sep);
        }
        out.push(*c);
    }
    out
}

/// Group only the leading integer run of a numeric body (the digits
/// before any `.`, exponent, or `%`), leaving the fractional / suffix
/// portion untouched.  Float groupings always use interval 3.
fn group_integer_prefix(body: &str, sep: char) -> String {
    let int_len = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if int_len <= 3 {
        return body.to_string();
    }
    let (int_part, rest) = body.split_at(int_len);
    format!("{}{}", group_digits(int_part, sep, 3), rest)
}

fn pad_to_width(body: String, fill: char, align: char, width: usize) -> String {
    if body.chars().count() >= width {
        return body;
    }
    let need = width - body.chars().count();
    match align {
        '<' => {
            let mut s = body;
            for _ in 0..need {
                s.push(fill);
            }
            s
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            let mut s = String::with_capacity(width);
            for _ in 0..left {
                s.push(fill);
            }
            s.push_str(&body);
            for _ in 0..right {
                s.push(fill);
            }
            s
        }
        // `pypy/objspace/std/newformat.py:454-468` — `=` alignment
        // splits the leading sign / base prefix from the digit body
        // and inserts fill BETWEEN them, so `f"{-7:=05d}"` renders
        // as `"-0007"` (sign then zeros then digits) instead of
        // `"000-7"`.  Numeric bodies the int / float paths emit
        // start with at most one sign char (`-`/`+`/space) followed
        // by an optional base prefix (`0x`/`0X`/`0o`/`0b`).
        '=' => {
            let mut chars = body.chars().peekable();
            let mut prefix = String::new();
            if let Some(&c) = chars.peek() {
                if c == '-' || c == '+' || c == ' ' {
                    prefix.push(c);
                    chars.next();
                }
            }
            // Optional alt-form base prefix: 0x / 0X / 0o / 0b.
            let rest_so_far: String = chars.clone().collect();
            if rest_so_far.len() >= 2 && rest_so_far.as_bytes()[0] == b'0' {
                let next = rest_so_far.as_bytes()[1];
                if matches!(next, b'x' | b'X' | b'o' | b'b') {
                    prefix.push('0');
                    prefix.push(next as char);
                    chars.next();
                    chars.next();
                }
            }
            let digits: String = chars.collect();
            let mut s = String::with_capacity(width);
            s.push_str(&prefix);
            for _ in 0..need {
                s.push(fill);
            }
            s.push_str(&digits);
            s
        }
        _ => {
            // Default `>` (right-align) for any unknown sigil.
            let mut s = String::with_capacity(width);
            for _ in 0..need {
                s.push(fill);
            }
            s.push_str(&body);
            s
        }
    }
}

/// Public entry point for the f-string `FormatWithSpec` opcode in
/// `eval.rs::format_with_spec`. Forwards to the same parser used by
/// `str.format` so both surfaces share the spec semantics.
pub fn format_with_spec_public(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    format_with_spec(val, spec)
}

/// Bind the type-level `__format__` descriptor `meth` to `val` and call it
/// with `spec_obj`, requiring a `str` result.  Dispatching the looked-up
/// `meth` (rather than a fresh instance lookup) keeps `__format__` a
/// type-level special method — an instance-dict `__format__` is ignored —
/// and binds a `@staticmethod` / `@classmethod` / other descriptor override
/// through the descriptor protocol.  The spec is passed through untouched so
/// the type's own `__format__` runs its validation.
pub(crate) fn call_format_dispatch(
    val: PyObjectRef,
    meth: PyObjectRef,
    spec_obj: PyObjectRef,
) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let w_type = crate::typedef::r#type(val).unwrap_or(pyre_object::PY_NULL);
        let result = crate::baseobjspace::get_and_call_function(meth, val, w_type, &[spec_obj])?;
        if !pyre_object::is_str(result) {
            return Err(crate::PyError::type_error(format!(
                "__format__ must return a str, not {}",
                arg_type_name(result)
            )));
        }
        Ok(pyre_object::w_str_get_wtf8(result).to_wtf8_buf())
    }
}

/// `PyObject_Format` — when `val` is a class instance whose type defines
/// `__format__`, dispatch to it (the result must be a `str`); otherwise
/// apply the shared builtin spec parser, with an empty spec collapsing to
/// `str(value)`.  Shared by `format()`, the `FormatSimple`/`FormatWithSpec`
/// f-string opcodes, and `str.format` field formatting.
pub fn format_value_dispatch(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    // A class instance always dispatches to its `__format__` (its own
    // override or the inherited `object.__format__`).  A builtin subclass
    // dispatches whenever it overrides `__format__` with anything other than
    // the inherited builtin default — a `def`, `@staticmethod`,
    // `@classmethod`, or any non-`BUILTIN_FUNCTION_TYPE` descriptor; the
    // builtin default takes the fast path below, which formats the
    // underlying value directly.  `__format__` is resolved on the type (not
    // the instance) so an instance-dict attribute does not shadow it.
    if let Some(meth) = unsafe { crate::baseobjspace::lookup(val, "__format__") } {
        if unsafe { is_instance(val) }
            || !unsafe { py_type_check(meth, &crate::function::BUILTIN_FUNCTION_TYPE) }
        {
            let spec_obj = pyre_object::w_str_new(spec);
            return call_format_dispatch(val, meth, spec_obj);
        }
    }
    if spec.is_empty() {
        // Empty spec collapses to `str(value)`, preserved in WTF-8 so a
        // str — or an exception whose single argument is a str — keeps
        // its lone surrogates.
        Ok(unsafe { crate::py_str_wtf8(val)? })
    } else {
        Ok(format_with_spec_public(val, spec)?)
    }
}

/// The type name of `obj` for a TypeError message — the `w_class` name
/// for instances, else the storage type name.
pub(crate) fn arg_type_name(obj: PyObjectRef) -> String {
    if obj.is_null() {
        return "object".to_string();
    }
    unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => w_type_get_name(tp).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    }
}

/// Read a format spec's stored string value. The spec must be a `str`
/// (or subclass); its `__str__` is not consulted, so a raising override
/// does not leak out of formatting.  `arg_desc` names the argument in the
/// `TypeError` raised for a non-`str` spec (`format()` reports `format()
/// argument 2`, a type's `__format__` reports `__format__() argument`).
pub(crate) fn read_format_spec(
    spec_obj: PyObjectRef,
    arg_desc: &str,
) -> Result<String, crate::PyError> {
    if !spec_obj.is_null() && unsafe { is_str(spec_obj) } {
        return Ok(unsafe { w_str_get_value(spec_obj) }.to_string());
    }
    Err(crate::PyError::type_error(format!(
        "{arg_desc} must be str, not {}",
        arg_type_name(spec_obj)
    )))
}

/// `int/float/str/bool.__format__(self, format_spec)` — formats `self`
/// through the shared spec parser without re-dispatching to an instance
/// `__format__` (which `format_value_dispatch` would do for subclasses,
/// risking recursion).  An empty spec collapses to `str(self)`.
pub fn builtin_value_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let spec = if args.len() > 1 {
        read_format_spec(args[1], "__format__() argument")?
    } else {
        String::new()
    };
    if spec.is_empty() {
        // `str(self)` — a str self passes through as WTF-8.
        if unsafe { pyre_object::is_str(args[0]) } {
            return Ok(pyre_object::w_str_from_wtf8(
                unsafe { pyre_object::w_str_get_wtf8(args[0]) }.to_wtf8_buf(),
            ));
        }
        return Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0])? }));
    }
    Ok(pyre_object::w_str_from_wtf8(format_with_spec_public(
        args[0], &spec,
    )?))
}

fn format_with_spec(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec);
    unsafe {
        if pyre_object::is_int(val) || pyre_object::is_bool(val) {
            let v = if pyre_object::is_bool(val) {
                pyre_object::w_bool_get_value(val) as i64
            } else {
                pyre_object::w_int_get_value(val)
            };
            // `newformat.py:1018-1026` — the `c` presentation type
            // formats the integer as the single Unicode character
            // `chr(v)`, then pads as text (default left align).
            if p.ty == 'c' {
                let body = u32::try_from(v)
                    .ok()
                    .and_then(char::from_u32)
                    .map_or_else(|| format!("{v}"), |c| c.to_string());
                // `c` keeps the integer default alignment (right).
                let align = p.align.unwrap_or('>');
                return Ok(Wtf8Buf::from_string(pad_to_width(
                    body, p.fill, align, p.width,
                )));
            }
            // Float-style spec on int: coerce to f64 (matches CPython
            // `int.__format__('.3f')` behaviour).  `%` is a float-only
            // presentation type, so route ints through it too.
            if matches!(p.ty, 'f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%') {
                return Ok(Wtf8Buf::from_string(format_float(v as f64, &p)));
            }
            return Ok(Wtf8Buf::from_string(format_int(v, &p)));
        }
        if pyre_object::is_float(val) {
            let v = pyre_object::floatobject::w_float_get_value(val);
            return Ok(Wtf8Buf::from_string(format_float(v, &p)));
        }
        if pyre_object::is_str(val) {
            // Read the WTF-8 view so a lone-surrogate body formats and
            // pads by code point instead of panicking.
            let full = pyre_object::w_str_get_wtf8(val);
            let body = if let Some(prec) = p.precision {
                let mut t = Wtf8Buf::new();
                let mut n = 0usize;
                for cp in full.code_points() {
                    if n >= prec {
                        break;
                    }
                    t.push(cp);
                    n += 1;
                }
                t
            } else {
                full.to_wtf8_buf()
            };
            let align = p.align.unwrap_or('<');
            return Ok(pad_wtf8(&body, p.fill, align, p.width));
        }
        Ok(Wtf8Buf::from_string(pad_to_width(
            crate::py_str(val)?,
            p.fill,
            p.align.unwrap_or('<'),
            p.width,
        )))
    }
}

/// Pad a WTF-8 string body to `width` code points with `fill`,
/// honouring `<` / `^` / `>` alignment.  String bodies never use the
/// numeric `=` alignment, so any non-`<`/`^` alignment right-aligns.
fn pad_wtf8(body: &Wtf8, fill: char, align: char, width: usize) -> Wtf8Buf {
    let body_len = body.code_points().count();
    if body_len >= width {
        return body.to_wtf8_buf();
    }
    let need = width - body_len;
    let fill_cp = CodePoint::from_char(fill);
    let mut out = Wtf8Buf::with_capacity(body.len() + need * 4);
    match align {
        '<' => {
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, need);
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            push_cp_repeated(&mut out, fill_cp, left);
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, right);
        }
        _ => {
            push_cp_repeated(&mut out, fill_cp, need);
            out.push_wtf8(body);
        }
    }
    out
}

fn format_int(v: i64, p: &ParsedSpec) -> String {
    let abs = v.unsigned_abs();
    let digits = match p.ty {
        'x' => format!("{abs:x}"),
        'X' => format!("{abs:X}"),
        'o' => format!("{abs:o}"),
        'b' => format!("{abs:b}"),
        _ => format!("{abs}"),
    };
    // `newformat.py:646-651` — `,` always groups by 3; `_` groups by 4
    // for the bit presentations (b/o/x/X) and by 3 otherwise.
    let digits = if let Some(sep) = p.grouping {
        let interval = if sep == '_' && matches!(p.ty, 'x' | 'X' | 'o' | 'b') {
            4
        } else {
            3
        };
        group_digits(&digits, sep, interval)
    } else {
        digits
    };
    let sign_char = if v < 0 {
        "-"
    } else {
        match p.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    // `pypy/objspace/std/newformat.py:454-460` — alt-form `#`
    // prepends the matching base prefix for x/X/o/b; ignored for
    // d / decimal.
    let alt_prefix = if p.alt_form {
        match p.ty {
            'x' => "0x",
            'X' => "0X",
            'o' => "0o",
            'b' => "0b",
            _ => "",
        }
    } else {
        ""
    };
    let body = format!("{sign_char}{alt_prefix}{digits}");
    // `pypy/objspace/std/newformat.py:454-468` — when zero_pad is
    // set, parse_spec already promoted `align = '='` and `fill =
    // '0'` so `pad_to_width` performs the sign-aware insertion.
    let align = p.align.unwrap_or('>');
    pad_to_width(body, p.fill, align, p.width)
}

fn format_float(v: f64, p: &ParsedSpec) -> String {
    let prec = p.precision.unwrap_or(6);
    // Always format on `v.abs()` so the sign is reattached exactly
    // once below.  Rust's `{:e}` / `{:E}` include the sign already,
    // which previously duplicated `-` for negative values; using
    // `abs()` consistently fixes that.
    let abs = v.abs();
    let body = match p.ty {
        // `pypy/objspace/std/newformat.py` `format_e_g_complex`
        // mirrors C printf — the exponent has an explicit sign and
        // is zero-padded to two digits ("e+02", "E-04").  Rust's
        // `{:e}` emits a sign-less, minimal exponent ("e2"), so
        // route through `normalise_exponent` here.
        'e' => crate::baseobjspace::normalise_exponent(&format!("{abs:.prec$e}"), false),
        'E' => crate::baseobjspace::normalise_exponent(&format!("{abs:.prec$E}"), true),
        'f' | 'F' => format!("{:.*}", prec, abs),
        // `newformat.py` percent type: scale by 100, format fixed, suffix `%`.
        '%' => format!("{:.*}%", prec, abs * 100.0),
        // `g`/`G` always format `general`: default precision 6, trailing
        // zeros trimmed unless alt-form keeps them.  `n` matches `g` but
        // takes its digit grouping from the locale (none in the C locale).
        'g' | 'G' | 'n' => crate::baseobjspace::format_g_like(abs, prec, p.ty == 'G', p.alt_form),
        '\0' => {
            // No presentation type formats like `repr()` — the shortest
            // round-trip string, which keeps the trailing `.0` for whole
            // values.  An explicit precision (or `#`) switches to `g`.
            if p.precision.is_some() || p.alt_form {
                crate::baseobjspace::format_g_like(abs, prec, false, p.alt_form)
            } else {
                crate::display::format_float_repr(abs)
            }
        }
        _ => format!("{}", abs),
    };
    let sign_char = if v.is_sign_negative() && !v.is_nan() {
        "-"
    } else {
        match p.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    // `pypy/objspace/std/newformat.py:464-468` — alt-form `#` keeps
    // the trailing dot (and trailing zeros) for floats so the
    // requested precision is visible even when the value is whole.
    // Cheapest expression: append `.` if missing.
    let body = if p.alt_form
        && !body.contains('.')
        && matches!(p.ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | '\0')
    {
        format!("{body}.")
    } else {
        body
    };
    // `newformat.py:646-648` — float groupings always use interval 3,
    // applied to the integer run only (the fractional / exponent / `%`
    // suffix is left untouched).
    let body = match p.grouping {
        Some(sep) => group_integer_prefix(&body, sep),
        None => body,
    };
    let body = format!("{sign_char}{body}");
    // Same `=`/`'0'` promotion as `format_int`; pad_to_width does
    // the sign-aware insertion.
    let align = p.align.unwrap_or('>');
    pad_to_width(body, p.fill, align, p.width)
}

/// runicode.py:333 unicode_encode_utf_8 + interp_codecs.py
/// surrogatepass / surrogateescape encode branches.  The WTF-8 backing
/// already stores a lone surrogate as its three-byte sequence, so the
/// surrogate-free common case is a direct byte copy; surrogate code points
/// are routed to the named error handler.  `w_object` is the str being
/// encoded, threaded through so a strict failure can build a structured
/// UnicodeEncodeError carrying it.
fn encode_utf8_with_errors(
    w_object: PyObjectRef,
    err_mode: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let s: &Wtf8 = unsafe { w_str_get_wtf8(w_object) };
    // utf8_encode_utf_8 fast path: no surrogates → already valid UTF-8.
    if let Ok(valid) = s.as_str() {
        return Ok(valid.as_bytes().to_vec());
    }
    let mut out = Vec::with_capacity(s.len());
    let mut buf = [0u8; 4];
    for (index, cp) in s.code_points().enumerate() {
        if let Some(c) = cp.to_char() {
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        let code = cp.to_u32();
        match err_mode {
            // surrogatepass_errors encode branch (interp_codecs.py:455-458):
            // emit the three-byte sequence for the surrogate code point.
            "surrogatepass" => {
                out.push(0xE0 | (code >> 12) as u8);
                out.push(0x80 | ((code >> 6) & 0x3f) as u8);
                out.push(0x80 | (code & 0x3f) as u8);
            }
            // surrogateescape_errors encode branch (interp_codecs.py:528-534):
            // a 0xDC80..0xDCFF surrogate maps back to the byte code-0xDC00;
            // any other surrogate fails.
            "surrogateescape" => {
                if (0xDC80..=0xDCFF).contains(&code) {
                    out.push((code - 0xDC00) as u8);
                } else {
                    return Err(crate::typedef::unicode_encode_error(
                        "utf-8",
                        w_object,
                        index,
                        index + 1,
                        "surrogates not allowed",
                    ));
                }
            }
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    "utf-8",
                    w_object,
                    index,
                    index + 1,
                    "surrogates not allowed",
                ));
            }
            "ignore" => {}
            "replace" => out.push(b'?'),
            "backslashreplace" => out.extend_from_slice(format!("\\u{code:04x}").as_bytes()),
            "xmlcharrefreplace" => out.extend_from_slice(format!("&#{code};").as_bytes()),
            _ => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::LookupError,
                    format!("unknown error handler name '{err_mode}'"),
                ));
            }
        }
    }
    Ok(out)
}

/// PyPy: unicodeobject.py descr_encode → encode_object.
/// For the common 'utf-8' / 'ascii' fast paths, returns the UTF-8 bytes
/// of the string. Other codecs fall through to a best-effort UTF-8 encoding.
pub fn str_method_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    // `encoding` and `errors` arrive positionally or by keyword; builtin
    // kwargs are packed in a trailing `__pyre_kw__` dict.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // `get_encoding_and_errors` unwraps both arguments through
    // `space.text_w`; a present non-string value raises
    // `TypeError("expected str, got X object")` (baseobjspace.py
    // `_typed_unwrap_error`).  An absent argument keeps the default.
    let str_arg = |obj: Option<PyObjectRef>, default: &str| -> Result<String, crate::PyError> {
        match obj {
            None => Ok(default.to_string()),
            Some(o) if o.is_null() => Ok(default.to_string()),
            Some(o) if unsafe { pyre_object::is_str(o) } => {
                Ok(unsafe { w_str_get_value(o) }.to_string())
            }
            Some(o) => {
                let tname = unsafe { (*(*o).ob_type).name };
                Err(crate::PyError::type_error(format!(
                    "expected str, got {tname} object"
                )))
            }
        }
    };
    // `encode(encoding=None, errors=None)` — both positional-or-keyword;
    // the gateway rejects unknown keywords and a value given both ways.
    crate::builtins::kwarg_reject_unknown(kwargs, &["encoding", "errors"], "encode")?;
    let dual =
        |name: &str, p: Option<PyObjectRef>| -> Result<Option<PyObjectRef>, crate::PyError> {
            let kw = crate::builtins::kwarg_get(kwargs, name);
            if p.is_some() && kw.is_some() {
                return Err(crate::PyError::type_error(format!(
                    "got multiple values for argument '{name}'"
                )));
            }
            Ok(p.or(kw))
        };
    let encoding = str_arg(dual("encoding", pos.get(1).copied())?, "utf-8")?;
    let errors = str_arg(dual("errors", pos.get(2).copied())?, "strict")?;
    Ok(pyre_object::w_bytes_from_bytes(&encode_object(
        args[0], &encoding, &errors,
    )?))
}

/// `unicodeobject.py W_UnicodeObject.descr_encode` → `encode_object`.
/// Encodes a str (`w_object`) to bytes with the named codec and error
/// handler.  Shared by `str.encode`, `bytes(str, …)` and
/// `bytearray(str, …)` so all three honour the same codec set and error
/// handlers.  The whole path reads the surrogate-aware WTF-8 view, so a
/// lone surrogate is routed to the error handler rather than crashing.
pub fn encode_object(
    w_object: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let enc_lower = encoding.to_ascii_lowercase().replace('_', "-");
    if matches!(enc_lower.as_str(), "utf-8" | "utf8" | "u8") {
        return encode_utf8_with_errors(w_object, errors);
    }
    let s = unsafe { w_str_get_wtf8(w_object) };
    match enc_lower.as_str() {
        "ascii" | "us-ascii" | "646" => encode_narrow(
            s,
            w_object,
            "ascii",
            0x7f,
            "ordinal not in range(128)",
            errors,
        ),
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => encode_narrow(
            s,
            w_object,
            "latin-1",
            0xff,
            "ordinal not in range(256)",
            errors,
        ),
        "raw-unicode-escape" => Ok(encode_raw_unicode_escape(s)),
        _ => match encode_utf16_32(s, &enc_lower, w_object, errors) {
            Some(out) => out,
            None => Err(crate::PyError::new(
                crate::PyErrorKind::LookupError,
                format!("unknown encoding: {encoding}"),
            )),
        },
    }
}

/// `unicodeobject.c:_PyUnicode_EncodeRawUnicodeEscape` — code points
/// below 0x100 map to a single Latin-1 byte; 0x100..0x10000 become the
/// 6-byte `\uXXXX` form; everything larger becomes `\UXXXXXXXX`.  Unlike
/// `unicode-escape`, the backslash and control characters are not
/// escaped.
pub fn encode_raw_unicode_escape(s: &Wtf8) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for cp in s.code_points() {
        let v = cp.to_u32();
        if v < 0x100 {
            out.push(v as u8);
        } else if v < 0x10000 {
            out.extend_from_slice(format!("\\u{v:04x}").as_bytes());
        } else {
            out.extend_from_slice(format!("\\U{v:08x}").as_bytes());
        }
    }
    out
}

/// `unicodeobject.c:_PyUnicode_DecodeRawUnicodeEscape` — the inverse of
/// [`encode_raw_unicode_escape`].  A backslash starts a `\uXXXX` /
/// `\UXXXXXXXX` escape; any other byte (including a lone backslash or a
/// malformed escape) is taken as a Latin-1 code point.
pub fn decode_raw_unicode_escape(data: &[u8]) -> Result<Wtf8Buf, crate::PyError> {
    let mut out = Wtf8Buf::new();
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        if b != b'\\' {
            out.push_char(b as char);
            i += 1;
            continue;
        }
        // Count the run of backslashes; only an odd run can introduce a
        // `\u`/`\U` escape (an even run is literal escaped backslashes,
        // but raw-unicode-escape does not collapse them — each `\` is a
        // literal byte 0x5c).  The escape applies when `\` is followed by
        // `u` or `U` with enough hex digits.
        let kind = data.get(i + 1).copied();
        let want = match kind {
            Some(b'u') => 4usize,
            Some(b'U') => 8usize,
            _ => 0,
        };
        if want != 0 && i + 2 + want <= data.len() {
            let hex = &data[i + 2..i + 2 + want];
            if let Ok(hs) = std::str::from_utf8(hex) {
                if let Ok(v) = u32::from_str_radix(hs, 16) {
                    if let Some(c) = CodePoint::from_u32(v) {
                        out.push(c);
                        i += 2 + want;
                        continue;
                    }
                }
            }
        }
        // Not a valid escape — emit the backslash literally as Latin-1.
        out.push_char(b as char);
        i += 1;
    }
    Ok(out)
}

fn encode_narrow(
    s: &Wtf8,
    source: PyObjectRef,
    enc_name: &str,
    max_cp: u32,
    range_msg: &str,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let cps: Vec<u32> = s.code_points().map(|c| c.to_u32()).collect();
    let mut out: Vec<u8> = Vec::with_capacity(cps.len());
    let mut i = 0usize;
    while i < cps.len() {
        if cps[i] <= max_cp {
            out.push(cps[i] as u8);
            i += 1;
            continue;
        }
        // `surrogateescape` rescues only a 0xDC80..0xDCFF code point, mapping
        // it back to the byte `code-0xDC00` (interp_codecs.py:528-534); any
        // other unencodable code point still raises, so it is handled one at
        // a time rather than over the maximal run.
        if errors == "surrogateescape" && (0xDC80..=0xDCFF).contains(&cps[i]) {
            out.push((cps[i] - 0xDC00) as u8);
            i += 1;
            continue;
        }
        // Maximal run of consecutive unencodable code points — `strict`
        // reports the whole span as one error, like CPython.  A
        // `surrogateescape`-rescuable code point ends the run.
        let start = i;
        let mut end = i;
        while end < cps.len()
            && cps[end] > max_cp
            && !(errors == "surrogateescape" && (0xDC80..=0xDCFF).contains(&cps[end]))
        {
            end += 1;
        }
        match errors {
            // `surrogateescape` reached here only for an unencodable code
            // point outside the rescue range, so it raises like `strict`.
            // `surrogatepass` only rescues surrogates for utf-8/16/32, so a
            // narrow codec re-raises the original UnicodeEncodeError.
            "strict" | "surrogateescape" | "surrogatepass" => {
                return Err(crate::typedef::unicode_encode_error(
                    enc_name, source, start, end, range_msg,
                ));
            }
            "ignore" => {}
            "replace" => out.resize(out.len() + (end - start), b'?'),
            "backslashreplace" => {
                for &cp in &cps[start..end] {
                    let esc = if cp <= 0xff {
                        format!("\\x{cp:02x}")
                    } else if cp <= 0xffff {
                        format!("\\u{cp:04x}")
                    } else {
                        format!("\\U{cp:08x}")
                    };
                    out.extend_from_slice(esc.as_bytes());
                }
            }
            "xmlcharrefreplace" => {
                for &cp in &cps[start..end] {
                    out.extend_from_slice(format!("&#{cp};").as_bytes());
                }
            }
            _ => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::LookupError,
                    format!("unknown error handler name '{errors}'"),
                ));
            }
        }
        i = end;
    }
    Ok(out)
}

/// Collapse a normalized encoding name to its separator-free form so
/// that `utf-16-le`, `utf16le` and `utf_16_le` all compare equal.
fn compact_codec_name(lower: &str) -> String {
    lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect()
}

/// Append a 16-bit code unit in the requested byte order.
fn push_unit16(out: &mut Vec<u8>, unit: u16, big_endian: bool) {
    out.extend_from_slice(&if big_endian {
        unit.to_be_bytes()
    } else {
        unit.to_le_bytes()
    });
}

/// Append a 32-bit code unit in the requested byte order.
fn push_unit32(out: &mut Vec<u8>, unit: u32, big_endian: bool) {
    out.extend_from_slice(&if big_endian {
        unit.to_be_bytes()
    } else {
        unit.to_le_bytes()
    });
}

/// Emit a non-surrogate scalar value: utf-32 writes one 32-bit unit,
/// utf-16 writes one BMP unit or a surrogate pair for astral planes.
fn emit_scalar(out: &mut Vec<u8>, cp: u32, is32: bool, big_endian: bool) {
    if is32 {
        push_unit32(out, cp, big_endian);
    } else if cp <= 0xFFFF {
        push_unit16(out, cp as u16, big_endian);
    } else {
        let v = cp - 0x10000;
        push_unit16(out, 0xD800 | (v >> 10) as u16, big_endian);
        push_unit16(out, 0xDC00 | (v & 0x3FF) as u16, big_endian);
    }
}

/// utf-16 / utf-32 encode for the `lower`-normalized codec name, or
/// `None` if `lower` names neither.  The bare `utf-16` / `utf-32` forms
/// emit a little-endian BOM; the `-le` / `-be` forms omit it.  A lone
/// surrogate is routed through `errors` (`surrogatepass` emits its raw
/// code unit; `strict` raises) rather than crashing.
pub fn encode_utf16_32(
    s: &Wtf8,
    lower: &str,
    w_object: PyObjectRef,
    errors: &str,
) -> Option<Result<Vec<u8>, crate::PyError>> {
    // `codec` is the canonical name reported in a UnicodeEncodeError, so
    // a `-le` / `-be` spelling keeps its suffix while `utf16` normalizes
    // to `utf-16`.
    let (is32, big_endian, bom, codec) = match compact_codec_name(lower).as_str() {
        "utf16" | "u16" => (false, false, true, "utf-16"),
        "utf16le" => (false, false, false, "utf-16-le"),
        "utf16be" => (false, true, false, "utf-16-be"),
        "utf32" | "u32" => (true, false, true, "utf-32"),
        "utf32le" => (true, false, false, "utf-32-le"),
        "utf32be" => (true, true, false, "utf-32-be"),
        _ => return None,
    };
    Some(encode_utf16_32_impl(
        s, is32, big_endian, bom, codec, w_object, errors,
    ))
}

fn encode_utf16_32_impl(
    s: &Wtf8,
    is32: bool,
    big_endian: bool,
    bom: bool,
    codec: &str,
    w_object: PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let mut out = Vec::new();
    if bom {
        emit_scalar(&mut out, 0xFEFF, is32, big_endian);
    }
    for (index, cp) in s.code_points().enumerate() {
        let code = cp.to_u32();
        if !(0xD800..=0xDFFF).contains(&code) {
            emit_scalar(&mut out, code, is32, big_endian);
            continue;
        }
        // Lone surrogate — only the utf-8/16/32 surrogatepass branch may
        // emit it, as a raw code unit (interp_codecs.py surrogatepass).
        match errors {
            "surrogatepass" => {
                if is32 {
                    push_unit32(&mut out, code, big_endian);
                } else {
                    push_unit16(&mut out, code as u16, big_endian);
                }
            }
            // surrogateescape rescues a 0xDC80..0xDCFF surrogate to the byte
            // code-0xDC00; any other surrogate still raises.
            "surrogateescape" if (0xDC80..=0xDCFF).contains(&code) => {
                out.push((code - 0xDC00) as u8);
            }
            "ignore" => {}
            "replace" => emit_scalar(&mut out, '?' as u32, is32, big_endian),
            "backslashreplace" => {
                for b in format!("\\u{code:04x}").bytes() {
                    emit_scalar(&mut out, b as u32, is32, big_endian);
                }
            }
            "xmlcharrefreplace" => {
                for b in format!("&#{code};").bytes() {
                    emit_scalar(&mut out, b as u32, is32, big_endian);
                }
            }
            "strict" | "surrogateescape" => {
                return Err(crate::typedef::unicode_encode_error(
                    codec,
                    w_object,
                    index,
                    index + 1,
                    "surrogates not allowed",
                ));
            }
            _ => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::LookupError,
                    format!("unknown error handler name '{errors}'"),
                ));
            }
        }
    }
    Ok(out)
}

/// utf-16 / utf-32 decode for the `lower`-normalized codec name, or
/// `None` if `lower` names neither.  The bare `utf-16` / `utf-32` forms
/// consume a leading BOM to choose endianness (defaulting to
/// little-endian); the `-le` / `-be` forms are fixed.  A lone surrogate
/// is routed through `err_mode` (`surrogatepass` keeps it as a code
/// point; `strict` raises), so the result is a `Wtf8Buf`.
pub fn decode_utf16_32(
    data: &[u8],
    lower: &str,
    err_mode: &str,
) -> Option<Result<Wtf8Buf, crate::PyError>> {
    // `codec` is the canonical name reported in a UnicodeDecodeError.
    let (is32, fixed_be, codec) = match compact_codec_name(lower).as_str() {
        "utf16" | "u16" => (false, None, "utf-16"),
        "utf16le" => (false, Some(false), "utf-16-le"),
        "utf16be" => (false, Some(true), "utf-16-be"),
        "utf32" | "u32" => (true, None, "utf-32"),
        "utf32le" => (true, Some(false), "utf-32-le"),
        "utf32be" => (true, Some(true), "utf-32-be"),
        _ => return None,
    };
    Some(if is32 {
        decode_utf32_impl(data, fixed_be, codec, err_mode)
    } else {
        decode_utf16_impl(data, fixed_be, codec, err_mode)
    })
}

/// Resolve endianness and the body start offset: a fixed `-le`/`-be`
/// codec ignores any BOM, while the bare form consumes a leading BOM and
/// otherwise defaults to little-endian.
fn resolve_bom(data: &[u8], is32: bool, fixed_be: Option<bool>) -> (bool, usize) {
    match fixed_be {
        Some(be) => (be, 0),
        None if is32 && data.starts_with(&[0xFF, 0xFE, 0x00, 0x00]) => (false, 4),
        None if is32 && data.starts_with(&[0x00, 0x00, 0xFE, 0xFF]) => (true, 4),
        None if !is32 && data.starts_with(&[0xFF, 0xFE]) => (false, 2),
        None if !is32 && data.starts_with(&[0xFE, 0xFF]) => (true, 2),
        None => (false, 0),
    }
}

/// Read one `unit`-byte (2 or 4) code unit at `pos` in the given order.
fn read_code_unit(data: &[u8], pos: usize, unit: usize, big_endian: bool) -> u32 {
    if unit == 2 {
        let arr = [data[pos], data[pos + 1]];
        if big_endian {
            u16::from_be_bytes(arr) as u32
        } else {
            u16::from_le_bytes(arr) as u32
        }
    } else {
        let arr = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
        if big_endian {
            u32::from_be_bytes(arr)
        } else {
            u32::from_le_bytes(arr)
        }
    }
}

/// Decode error-handler dispatch for utf-16 / utf-32 (interp_codecs.py
/// surrogatepass/surrogateescape branches plus the generic handlers).
/// Appends the replacement to `out` and returns the byte position to
/// resume decoding at.  `unit` is 2 for utf-16, 4 for utf-32.
fn utf16_32_decode_error(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    big_endian: bool,
    unit: usize,
    out: &mut Wtf8Buf,
) -> Result<usize, crate::PyError> {
    match err_mode {
        "strict" => Err(crate::typedef::unicode_decode_error(
            codec, data, start, end, reason,
        )),
        "ignore" => Ok(end),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok(end)
        }
        "backslashreplace" => {
            for &b in &data[start..end.min(data.len())] {
                out.push_str(&format!("\\x{b:02x}"));
            }
            Ok(end)
        }
        // surrogatepass: reconstruct one surrogate from the unit at
        // `start` and keep it; a non-surrogate value re-raises
        // (interp_codecs.py:476-510).
        "surrogatepass" => {
            if start + unit <= data.len() {
                let ch = read_code_unit(data, start, unit, big_endian);
                if (0xD800..=0xDFFF).contains(&ch) {
                    out.push(CodePoint::from_u32(ch).unwrap());
                    return Ok(start + unit);
                }
            }
            Err(crate::typedef::unicode_decode_error(
                codec, data, start, end, reason,
            ))
        }
        // surrogateescape: escape each >=128 byte as 0xdc00+byte, up to 4
        // bytes or the first ASCII byte (interp_codecs.py:536-555).
        "surrogateescape" => {
            let mut consumed = 0usize;
            while consumed < 4 && start + consumed < end {
                let b = data[start + consumed];
                if b < 128 {
                    break;
                }
                out.push(CodePoint::from_u32(0xDC00 + b as u32).unwrap());
                consumed += 1;
            }
            if consumed == 0 {
                return Err(crate::typedef::unicode_decode_error(
                    codec, data, start, end, reason,
                ));
            }
            Ok(start + consumed)
        }
        _ => Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )),
    }
}

/// `unicodehelper.py str_decode_utf_16_helper` (runicode.py:517).
fn decode_utf16_impl(
    data: &[u8],
    fixed_be: Option<bool>,
    codec: &str,
    err_mode: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let (big_endian, mut pos) = resolve_bom(data, false, fixed_be);
    let len = data.len();
    let mut out = Wtf8Buf::with_capacity(len / 2);
    while pos < len {
        if len - pos < 2 {
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos,
                len,
                "truncated data",
                big_endian,
                2,
                &mut out,
            )?;
            if len - pos < 2 {
                break;
            }
            continue;
        }
        let ch = read_code_unit(data, pos, 2, big_endian);
        pos += 2;
        if !(0xD800..=0xDFFF).contains(&ch) {
            out.push(CodePoint::from_u32(ch).unwrap());
            continue;
        } else if ch >= 0xDC00 {
            // unexpected lone low surrogate
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos - 2,
                pos,
                "illegal encoding",
                big_endian,
                2,
                &mut out,
            )?;
            continue;
        }
        // high surrogate: a low surrogate must follow
        if len - pos < 2 {
            pos -= 2;
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos,
                len,
                "unexpected end of data",
                big_endian,
                2,
                &mut out,
            )?;
        } else {
            let ch2 = read_code_unit(data, pos, 2, big_endian);
            pos += 2;
            if (0xDC00..=0xDFFF).contains(&ch2) {
                let c = (((ch & 0x3FF) << 10) | (ch2 & 0x3FF)) + 0x10000;
                out.push(CodePoint::from_u32(c).unwrap());
            } else {
                pos = utf16_32_decode_error(
                    err_mode,
                    codec,
                    data,
                    pos - 4,
                    pos - 2,
                    "illegal UTF-16 surrogate",
                    big_endian,
                    2,
                    &mut out,
                )?;
            }
        }
    }
    Ok(out)
}

/// `unicodehelper.py str_decode_utf_32_helper` (runicode.py:762).  The
/// public codec rejects surrogates (`allow_surrogates=False`), so a
/// surrogate code point is routed through the error handler.
fn decode_utf32_impl(
    data: &[u8],
    fixed_be: Option<bool>,
    codec: &str,
    err_mode: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let (big_endian, mut pos) = resolve_bom(data, true, fixed_be);
    let len = data.len();
    let mut out = Wtf8Buf::with_capacity(len / 4);
    while pos < len {
        if len - pos < 4 {
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos,
                len,
                "truncated data",
                big_endian,
                4,
                &mut out,
            )?;
            if len - pos < 4 {
                break;
            }
            continue;
        }
        let ch = read_code_unit(data, pos, 4, big_endian);
        if (0xD800..=0xDFFF).contains(&ch) {
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos,
                pos + 4,
                "code point in surrogate code point range(0xd800, 0xe000)",
                big_endian,
                4,
                &mut out,
            )?;
            continue;
        } else if ch >= 0x110000 {
            pos = utf16_32_decode_error(
                err_mode,
                codec,
                data,
                pos,
                len,
                "code point not in range(0x110000)",
                big_endian,
                4,
                &mut out,
            )?;
            continue;
        }
        out.push(CodePoint::from_u32(ch).unwrap());
        pos += 4;
    }
    Ok(out)
}

/// Map each scalar code point of `s` through `f`, appending to a
/// `Wtf8Buf`; a lone surrogate passes through unchanged.  Used by the
/// case-mapping methods, which leave surrogates untouched.
fn wtf8_map_chars(s: &Wtf8, f: impl Fn(char, &mut Wtf8Buf)) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(s.len());
    for cp in s.code_points() {
        match cp.to_char() {
            Some(c) => f(c, &mut out),
            None => out.push(cp),
        }
    }
    out
}

pub fn str_method_isdigit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    // A lone surrogate satisfies no character class, so a non-UTF-8
    // backing is never all-digit (and the empty string is false too).
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isdecimal(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c.is_numeric()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isnumeric(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_numeric()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_istitle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let mut cased = false;
    let mut prev_cased = false;
    for cp in s.code_points() {
        // A lone surrogate is uncased, so it resets `prev_cased` like
        // any other non-cased code point (the `None` arm / `else`).
        match cp.to_char() {
            Some(c) => {
                if c.is_uppercase() {
                    if prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else if c.is_lowercase() {
                    if !prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else {
                    prev_cased = false;
                }
            }
            None => prev_cased = false,
        }
    }
    Ok(w_bool_from(cased))
}

pub fn str_method_isalpha(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_alphabetic()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isidentifier
pub fn str_method_isidentifier(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    // An identifier cannot contain a lone surrogate, so a non-UTF-8
    // backing is never an identifier.
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => is_identifier(v),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Check if a string is a valid Python identifier.
/// Python 3 identifiers: ID_Start ID_Continue*
/// Simplified: accepts ASCII identifiers + Unicode letters/digits.
fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject.descr_zfill`.
/// Pads with leading zeros up to `width`; when the string starts with
/// a sign character (`+`/`-`), the sign stays at the front and zeros
/// fill between it and the digits (`'-42'.zfill(5) == '-0042'`).
pub fn str_method_zfill(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) }.max(0) as usize;
    let len = s.code_points().count();
    if len >= width {
        return Ok(args[0]);
    }
    let need = width - len;
    let mut cps = s.code_points();
    let mut out = Wtf8Buf::with_capacity(width);
    let first = cps.clone().next();
    if first == Some(CodePoint::from_char('+')) || first == Some(CodePoint::from_char('-')) {
        out.push(first.unwrap());
        cps.next();
    }
    for _ in 0..need {
        out.push_char('0');
    }
    for cp in cps {
        out.push(cp);
    }
    Ok(w_str_from_wtf8(out))
}

/// Number of non-overlapping occurrences of `needle` in `haystack`,
/// scanning over the WTF-8 bytes. The encoding is self-synchronizing,
/// so a byte-window match starts on a code-point boundary and the count
/// equals the code-point-level count. An empty needle matches at every
/// code-point boundary (len+1 positions), as in `str.count`.
fn wtf8_count(haystack: &Wtf8, needle: &Wtf8) -> usize {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return haystack.code_points().count() + 1;
    }
    let mut count = 0;
    let mut i = 0;
    while i + n.len() <= h.len() {
        if &h[i..i + n.len()] == n {
            count += 1;
            i += n.len();
        } else {
            i += 1;
        }
    }
    count
}

/// First byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. An empty needle matches at `lo`.
fn wtf8_find_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(lo);
    }
    if needle.len() > hi {
        return None;
    }
    (lo..=hi - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Last byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. An empty needle matches at `hi`.
fn wtf8_rfind_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(hi);
    }
    if needle.len() > hi {
        return None;
    }
    (lo..=hi - needle.len())
        .rev()
        .find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// PyPy `_unwrap_and_search` (unicodeobject.py:1288-1317) — the shared
/// path for find/rfind/index/rindex. `start`/`end` (args[2]/args[3])
/// are codepoint indices: `unwrap_start_stop` adds the length to a
/// negative value and lower-clamps to 0. The search runs over the
/// WTF-8 bytes inside that window and the byte offset is converted back
/// to a codepoint index. Returns None when not found.
/// `unicodeobject.py:1288 _unwrap_and_search` — the shared path for
/// find/rfind/index/rindex. `start`/`end` (args[2]/args[3]) flow through
/// `unwrap_start_stop`, so `None` / omitted arguments default, any
/// `__index__`-bearing object is accepted, and a `TypeError` propagates.
/// The search runs over the WTF-8 bytes inside the codepoint window and
/// the matching byte offset is mapped back to a codepoint index. Returns
/// `Ok(None)` when not found.
fn str_unwrap_and_search(
    args: &[PyObjectRef],
    forward: bool,
) -> Result<Option<i64>, crate::PyError> {
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let sub = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    let h = s.as_bytes();
    // Byte offset of each codepoint, with the trailing length appended,
    // so `cp_offsets[i]` is `_index_to_byte(i)` and a byte offset maps
    // back via its position.
    let mut cp_offsets: Vec<usize> = s.code_point_indices().map(|(i, _)| i).collect();
    cp_offsets.push(h.len());
    let length = (cp_offsets.len() - 1) as i64;

    let w_start = if args.len() >= 3 { args[2] } else { w_none() };
    let w_end = if args.len() >= 4 { args[3] } else { w_none() };
    let (start, end) = crate::sliceobject::unwrap_start_stop(length, w_start, w_end)?;

    let start_index = if start == 0 {
        0
    } else if start > length {
        return Ok(None);
    } else {
        cp_offsets[start as usize]
    };
    let end_index = if end >= length {
        h.len()
    } else {
        cp_offsets[end as usize]
    };
    if start_index > end_index {
        return Ok(None);
    }

    let res_index = if forward {
        wtf8_find_bounded(h, sub, start_index, end_index)
    } else {
        wtf8_rfind_bounded(h, sub, start_index, end_index)
    };
    Ok(res_index.and_then(|ri| cp_offsets.iter().position(|&o| o == ri).map(|i| i as i64)))
}

/// PyPy: unicodeobject.py descr_count
pub fn str_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    // Operands read as WTF-8 so lone surrogates do not panic; the optional
    // start / end arguments bound the count window over the code points.
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let sub = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
    let Some((byte_start, byte_end)) = wtf8_idx_window(s, args)? else {
        return Ok(w_int_new(0));
    };
    let window = rustpython_wtf8::Wtf8::from_bytes(&s.as_bytes()[byte_start..byte_end])
        .expect("code-point boundary slice is valid WTF-8");
    Ok(w_int_new(wtf8_count(window, sub) as i64))
}

/// PyPy: unicodeobject.py descr_index
/// `unicodeobject.py:1006-1010 _descr_index` — missing substring raises
/// "substring not found" (ValueError).
pub fn str_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    match str_unwrap_and_search(args, true)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// `unicodeobject.py descr_rindex` — like rfind, but raises ValueError
/// when the substring is absent.
/// unicodeobject.py:572 descr_rindex
pub fn str_method_rindex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    match str_unwrap_and_search(args, false)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// PyPy: unicodeobject.py descr_title
pub fn str_method_title(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let mut result = Wtf8Buf::with_capacity(s.len());
    let mut prev_is_sep = true;
    for cp in s.code_points() {
        match cp.to_char() {
            Some(c) => {
                if prev_is_sep {
                    for u in c.to_uppercase() {
                        result.push_char(u);
                    }
                } else {
                    for l in c.to_lowercase() {
                        result.push_char(l);
                    }
                }
                prev_is_sep = !c.is_alphanumeric();
            }
            // A lone surrogate is not alphanumeric — it starts a new word.
            None => {
                result.push(cp);
                prev_is_sep = true;
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

/// PyPy: unicodeobject.py descr_capitalize
pub fn str_method_capitalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let mut result = Wtf8Buf::with_capacity(s.len());
    let mut cps = s.code_points();
    if let Some(first) = cps.next() {
        match first.to_char() {
            Some(c) => {
                for u in c.to_uppercase() {
                    result.push_char(u);
                }
            }
            None => result.push(first),
        }
        for cp in cps {
            match cp.to_char() {
                Some(c) => {
                    for l in c.to_lowercase() {
                        result.push_char(l);
                    }
                }
                None => result.push(cp),
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

/// PyPy: unicodeobject.py descr_swapcase
pub fn str_method_swapcase(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let out = wtf8_map_chars(s, |c, out| {
        if c.is_uppercase() {
            for l in c.to_lowercase() {
                out.push_char(l);
            }
        } else {
            for u in c.to_uppercase() {
                out.push_char(u);
            }
        }
    });
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_center
/// Resolve the fillchar arg for `center`/`ljust`/`rjust`. Defaults to
/// `' '` when missing; PyPy raises TypeError when the fill string is
/// not exactly one character long (unicodeobject.py:1191-1194
/// _convert_fillchar parity).
fn pad_fillchar(args: &[PyObjectRef], method: &str) -> Result<CodePoint, crate::PyError> {
    if args.len() <= 2 {
        return Ok(CodePoint::from_char(' '));
    }
    if !unsafe { pyre_object::is_str(args[2]) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 2 must be a single character"
        )));
    }
    let raw = unsafe { w_str_get_wtf8(args[2]) };
    let mut iter = raw.code_points();
    let first = iter.next();
    if first.is_none() || iter.next().is_some() {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 2 must be a single character"
        )));
    }
    Ok(first.unwrap())
}

/// Append `cp` to `out`, `n` times.
fn push_cp_repeated(out: &mut Wtf8Buf, cp: CodePoint, n: usize) {
    for _ in 0..n {
        out.push(cp);
    }
}

pub fn str_method_center(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) }.max(0) as usize;
    let fillchar = pad_fillchar(args, "center")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(args[0]);
    }
    // unicodeobject.py:1098 d = (width - len) ; lpad = d//2 + (d & width & 1)
    let d = width - s_len;
    let left = d / 2 + (d & width & 1);
    let right = d - left;
    let mut out = Wtf8Buf::with_capacity(s.len() + (left + right) * 4);
    push_cp_repeated(&mut out, fillchar, left);
    out.push_wtf8(s);
    push_cp_repeated(&mut out, fillchar, right);
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_ljust
pub fn str_method_ljust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) }.max(0) as usize;
    let fillchar = pad_fillchar(args, "ljust")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(args[0]);
    }
    let mut out = Wtf8Buf::with_capacity(s.len() + (width - s_len) * 4);
    out.push_wtf8(s);
    push_cp_repeated(&mut out, fillchar, width - s_len);
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_rjust
pub fn str_method_rjust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) }.max(0) as usize;
    let fillchar = pad_fillchar(args, "rjust")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(args[0]);
    }
    let mut out = Wtf8Buf::with_capacity(s.len() + (width - s_len) * 4);
    push_cp_repeated(&mut out, fillchar, width - s_len);
    out.push_wtf8(s);
    Ok(w_str_from_wtf8(out))
}

/// `pypy/objspace/std/unicodeobject.py descr_isprintable` —
///
/// ```python
/// def descr_isprintable(self, space):
///     for ch in self._utf8:
///         if not unicodedb.isprintable(ord(ch)):
///             return space.w_False
///     return space.w_True
/// ```
///
/// Empty string returns True per CPython.  Non-printable categories
/// per Unicode standard: Cc/Cf/Cs/Co/Cn/Zl/Zp + Zs other than
/// U+0020.  Rust stdlib's `char::is_control()` only covers Cc; full
/// parity requires a Unicode category table.  Convergence path:
/// import `unicode_general_category` (e.g. via the `unicode-general-category`
/// crate) and check `!matches!(cat, Cc|Cf|Cs|Co|Cn|Zl|Zp)` + `Zs == ' '`.
/// For now: approximate via `!c.is_control() && c != ' ' || c == ' '`
/// which catches the common ASCII-only cases.
pub fn str_method_isprintable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    // Empty returns True (vacuous); a lone surrogate is not printable.
    let result = match s.as_str() {
        Ok(v) => v.chars().all(|c| {
            // Cc (control) — Rust stdlib catches this.
            // Zl / Zp — single chars U+2028 / U+2029 are non-printable.
            // Zs other than space — narrow no-break U+202F, etc., are
            // non-printable, but plain space ' ' is.
            !c.is_control() && c != '\u{2028}' && c != '\u{2029}'
        }),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isspace
pub fn str_method_isspace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_whitespace()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// True iff `s` has at least one cased (alphabetic) code point and
/// every cased code point matches the requested case.  Lone
/// surrogates are uncased and ignored, so `'ABC\udcff'.isupper()` is
/// still True — unlike the character-class predicates, a surrogate
/// does not force a false result here.
fn wtf8_cased_all(s: &Wtf8, want_upper: bool) -> bool {
    let mut has_cased = false;
    let mut all_match = true;
    for cp in s.code_points() {
        if let Some(c) = cp.to_char() {
            if c.is_alphabetic() {
                has_cased = true;
                let ok = if want_upper {
                    c.is_uppercase()
                } else {
                    c.is_lowercase()
                };
                if !ok {
                    all_match = false;
                }
            }
        }
    }
    has_cased && all_match
}

/// PyPy: unicodeobject.py descr_isupper
pub fn str_method_isupper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, true)))
}

/// PyPy: unicodeobject.py descr_islower
pub fn str_method_islower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, false)))
}

/// PyPy: unicodeobject.py descr_isalnum
pub fn str_method_isalnum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(|c| c.is_alphanumeric()),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isascii
pub fn str_method_isascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => v.is_ascii(),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Builds an owned str from a byte sub-slice of a WTF-8 string. The
/// slice must start and end on code-point boundaries, which holds for
/// the partition cuts below (the separator aligns on boundaries).
fn wtf8_slice_str(bytes: &[u8]) -> PyObjectRef {
    let part = unsafe { Wtf8::from_bytes_unchecked(bytes) };
    w_str_from_wtf8(part.to_wtf8_buf())
}

/// Replaces up to `maxcount` occurrences of `sub` with `by` over the
/// WTF-8 bytes (rstring.py:220-309 `replace_count` isutf8 path). A
/// negative `maxcount` means no limit; an empty `sub` inserts `by` at
/// every code-point boundary, including the ends.
fn wtf8_replace(input: &Wtf8, sub: &Wtf8, by: &Wtf8, maxcount: i64) -> Wtf8Buf {
    if maxcount == 0 {
        return input.to_wtf8_buf();
    }
    let inp = input.as_bytes();
    let sub_b = sub.as_bytes();
    let mut out = Wtf8Buf::new();
    let mut start = 0usize;
    let mut maxcount = maxcount;
    if sub_b.is_empty() {
        let mut indices = input.code_point_indices().map(|(i, _)| i);
        // Skip the leading boundary at 0; it is handled by the first
        // `by` insertion before each code point.
        indices.next();
        loop {
            out.push_wtf8(by);
            maxcount -= 1;
            if start == inp.len() || maxcount == 0 {
                break;
            }
            let next = indices.next().unwrap_or(inp.len());
            out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
            start = next;
        }
    } else {
        while maxcount != 0 {
            match wtf8_find_bounded(inp, sub_b, start, inp.len()) {
                Some(next) => {
                    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
                    out.push_wtf8(by);
                    start = next + sub_b.len();
                    maxcount -= 1;
                }
                None => break,
            }
        }
    }
    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..]) });
    out
}

/// PyPy: unicodeobject.py descr_partition
pub fn str_method_partition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) }.as_bytes();
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    match wtf8_find_bounded(s, sep, 0, s.len()) {
        Some(i) => Ok(w_tuple_new(vec![
            wtf8_slice_str(&s[..i]),
            args[1],
            wtf8_slice_str(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![args[0], w_str_new(""), w_str_new("")])),
    }
}

/// PyPy: unicodeobject.py descr_rpartition
pub fn str_method_rpartition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) }.as_bytes();
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    match wtf8_rfind_bounded(s, sep, 0, s.len()) {
        Some(i) => Ok(w_tuple_new(vec![
            wtf8_slice_str(&s[..i]),
            args[1],
            wtf8_slice_str(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![w_str_new(""), w_str_new(""), args[0]])),
    }
}

/// PyPy: unicodeobject.py descr_splitlines.
/// Walks `\n`, `\r`, and `\r\n` boundaries explicitly so that
/// `keepends=True` retains the terminator on each emitted line and a
/// trailing `\n` does NOT produce an extra empty entry — matching
/// `'a\nb\n'.splitlines() == ['a', 'b']`.
pub fn str_method_splitlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // `\n` / `\r` are single bytes that cannot occur inside a multi-byte
    // WTF-8 sequence, so the line boundaries are found by walking bytes;
    // each emitted slice cuts on a code-point boundary.
    let bytes = unsafe { w_str_get_wtf8(pos[0]) }.as_bytes();
    // keepends is positional-or-keyword.
    let keepends = crate::builtins::kwarg_get(kwargs, "keepends")
        .or_else(|| pos.get(1).copied())
        .map(crate::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let mut parts: Vec<PyObjectRef> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' || bytes[i] == b'\r' {
            let mut term_end = i + 1;
            if bytes[i] == b'\r' && term_end < bytes.len() && bytes[term_end] == b'\n' {
                term_end += 1;
            }
            let end = if keepends { term_end } else { i };
            parts.push(wtf8_slice_str(&bytes[start..end]));
            start = term_end;
            i = term_end;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        parts.push(wtf8_slice_str(&bytes[start..]));
    }
    Ok(w_list_new(parts))
}

/// PyPy: unicodeobject.py descr_removeprefix (Python 3.9+)
pub fn str_method_removeprefix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removeprefix() takes exactly one argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    let s = unsafe { w_str_get_wtf8(pos[0]) };
    let prefix = unsafe { w_str_get_wtf8(pos[1]) };
    match s.strip_prefix(prefix) {
        Some(rest) => Ok(w_str_from_wtf8(rest.to_wtf8_buf())),
        None => Ok(pos[0]),
    }
}

/// PyPy: unicodeobject.py descr_removesuffix (Python 3.9+)
pub fn str_method_removesuffix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removesuffix() takes exactly one argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    let s = unsafe { w_str_get_wtf8(pos[0]) };
    let suffix = unsafe { w_str_get_wtf8(pos[1]) };
    match s.strip_suffix(suffix) {
        Some(rest) => Ok(w_str_from_wtf8(rest.to_wtf8_buf())),
        None => Ok(pos[0]),
    }
}

/// PyPy: unicodeobject.py descr_expandtabs
pub fn str_method_expandtabs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let tabsize = if args.len() > 1 {
        unsafe { w_int_get_value(args[1]) }
    } else {
        8
    };
    // Tabs advance to the next multiple of `tabsize` measured from the
    // start of the current line (the column resets on `\n` / `\r`); a
    // non-positive `tabsize` drops tabs entirely.
    let mut result = Wtf8Buf::with_capacity(s.len());
    let mut col: i64 = 0;
    for cp in s.code_points() {
        match cp.to_char() {
            Some('\t') => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col += incr;
                    for _ in 0..incr {
                        result.push_char(' ');
                    }
                }
            }
            Some('\n') | Some('\r') => {
                result.push(cp);
                col = 0;
            }
            _ => {
                result.push(cp);
                col += 1;
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

/// PyPy: unicodeobject.py descr_translate
///
/// str.translate(table) — table is a dict mapping ordinals (int) to
/// ordinals (int), strings (str), or None (delete).
pub fn str_method_translate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "translate() takes exactly one argument");
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let table = args[1];
    let mut result = Wtf8Buf::with_capacity(s.len());
    unsafe {
        for cp in s.code_points() {
            let key = w_int_new(cp.to_u32() as i64);
            if let Some(val) = w_dict_lookup(table, key) {
                if is_none(val) {
                    // None → delete character
                } else if is_int(val) {
                    if let Some(c) = CodePoint::from_u32(w_int_get_value(val) as u32) {
                        result.push(c);
                    }
                } else if is_str(val) {
                    result.push_wtf8(w_str_get_wtf8(val));
                } else {
                    result.push(cp);
                }
            } else {
                result.push(cp);
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

// ── Dict methods ─────────────────────────────────────────────────────

/// Resolve the actual backing W_DictObject for either a plain dict or
/// a dict subclass instance (which stores data in `__dict_data__`).
///
/// PyPy: W_DictMultiObject subclass instances ARE dicts, so no indirection
/// is needed. In pyre, dict subclass instances are W_ObjectObject with a
/// backing dict stored as an attribute.
pub fn resolve_dict_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_dict(obj) {
            return obj;
        }
        // `pypy/objspace/std/dictproxyobject.py:75-82 keys_w/values_w/
        // items_w` forward through `space.call_method(self.w_mapping,
        // ...)` — the mapping is unwrapped before any dict-method
        // dispatch.  Surface the same shape here so
        // `dict_method_{keys,values,items,get,copy,update,...}` work
        // on `type.__dict__` without per-method proxy plumbing.
        if pyre_object::is_dict_proxy(obj) {
            let inner = pyre_object::w_dict_proxy_get_mapping(obj);
            if !inner.is_null() && pyre_object::is_dict(inner) {
                return inner;
            }
        }
        if is_instance(obj) {
            if let Ok(backing) = crate::baseobjspace::getattr_str(obj, "__dict_data__") {
                if is_dict(backing) {
                    return backing;
                }
            }
        }
    }
    pyre_object::PY_NULL
}

fn dict_lookup_checked(
    dict: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        pyre_object::dictmultiobject::w_dict_lookup_checked(dict, key)
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())
    }
}

pub(crate) fn dict_store_checked(
    dict: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        pyre_object::dictmultiobject::w_dict_store_checked(dict, key, value)
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())
    }
}

pub fn dict_method_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if dict.is_null() {
        return Ok(default);
    }
    Ok(dict_lookup_checked(dict, key)?.unwrap_or(default))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_keys` parity — returns
/// a live `dict_keys` view bound to the source dict, not a snapshot
/// list.  The view's iter / len / contains semantics dispatch back
/// through the source dict (see baseobjspace getattr arm) so
/// mutations on the dict are visible through the view, matching
/// `W_DictViewKeysObject`'s behaviour.
pub fn dict_method_keys(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        // Type-erased fallback: the receiver isn't a dict, surface
        // an empty view rather than fabricating a foreign-shaped
        // list (the view's source-dict slot tolerates PY_NULL via
        // the read-side guards).
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Keys,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Keys,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_values` parity — same
/// shape as `descr_keys`, kind tag `Values`.
pub fn dict_method_values(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Values,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Values,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_items` parity — same
/// shape as `descr_keys`, kind tag `Items`.
pub fn dict_method_items(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Items,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Items,
    ))
}

/// Materialise a dict_keys / values / items view's current snapshot
/// as a list of items.  Mirrors the view iteration bodies on
/// `W_DictViewKeysObject` / values / items — pyre's `repr` /
/// `len` / `compare` / set-op paths call this to produce the
/// kind-appropriate list eagerly.
///
/// `__iter__` no longer routes through this helper: it allocates a
/// live `W_BaseDictMultiIterObject` that walks the source dict's entries
/// directly and trips on the dictversion counter, raising
/// `RuntimeError("dictionary changed size during iteration")` when
/// the source mutates mid-iteration.
pub fn dict_view_snapshot(view: PyObjectRef) -> Vec<PyObjectRef> {
    let kind = unsafe { pyre_object::dictmultiobject::w_dict_view_get_kind(view) };
    let dict = unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
    if dict.is_null() {
        return Vec::new();
    }
    let items = unsafe { pyre_object::w_dict_items(dict) };
    match kind {
        pyre_object::dictmultiobject::DictViewKind::Keys => {
            items.into_iter().map(|(k, _)| k).collect()
        }
        pyre_object::dictmultiobject::DictViewKind::Values => {
            items.into_iter().map(|(_, v)| v).collect()
        }
        pyre_object::dictmultiobject::DictViewKind::Items => items
            .into_iter()
            .map(|(k, v)| w_tuple_new(vec![k, v]))
            .collect(),
    }
}

/// `pypy/objspace/std/dictmultiobject.py:585-587 descr_copy` —
/// `return w_dict.copy()` which delegates to `strategy.copy(w_dict)`
/// (`:1152 AbstractTypedStrategy.copy`).  Typed strategies preserve
/// their backing shape by cloning the typed storage box and wrapping
/// it in a fresh W_DictObject with the same strategy.  Used by
/// `dict.copy()` and (via `resolve_dict_backing` proxy unwrap) by
/// `mappingproxy.copy()` (`dictproxyobject.py:84 copy_w`).
pub fn dict_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_dict_new());
    }
    let src = resolve_dict_backing(args[0]);
    if src.is_null() {
        return Ok(pyre_object::w_dict_new());
    }
    unsafe { Ok(pyre_object::dictmultiobject::w_dict_copy(src)) }
}

/// PyPy: dictmultiobject.py descr_update — dict.update([other], **kwargs).
///
/// CPython 3.x signature accepts a single optional positional that is
/// either a mapping (uses keys()) or an iterable of (key, value) pairs,
/// followed by arbitrary kwargs that are merged on top.  The trailing
/// `dictmultiobject.py:1378-1398 update1` — merge `w_data` into
/// `w_dict`.  Shared by `dict.__init__` and `dict.update`.
pub(crate) fn dict_update1(w_dict: PyObjectRef, w_data: PyObjectRef) -> Result<(), crate::PyError> {
    let dict = resolve_dict_backing(w_dict);
    if dict.is_null() {
        return Ok(());
    }
    let other_raw = resolve_dict_backing(w_data);
    unsafe {
        let fast_path_eligible = other_raw.is_null() == false
            && pyre_object::is_dict(other_raw)
            && dict_subclass_uses_default_iter(w_data);
        if fast_path_eligible {
            // `dictmultiobject.py:1401-1406 update1_dict_dict`
            let dst_is_empty = pyre_object::dictmultiobject::w_dict_is_regular_empty_no_proxy(dict);
            let src_proxy_free = pyre_object::w_dict_get_dict_storage_proxy(other_raw).is_null();
            if dst_is_empty && src_proxy_free {
                let w_copy = pyre_object::dictmultiobject::w_dict_copy(other_raw);
                pyre_object::dictmultiobject::w_dict_adopt_regular_copy_for_empty_update(
                    dict, w_copy,
                );
            } else {
                for (k, v) in pyre_object::w_dict_items(other_raw) {
                    dict_store_checked(dict, k, v)?;
                }
            }
        } else {
            // `dictmultiobject.py:1388-1398 update1`
            let w_keys_method = match crate::baseobjspace::getattr_str(w_data, "keys") {
                Ok(value) => Some(value),
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
                Err(e) => return Err(e),
            };
            if let Some(w_method) = w_keys_method {
                // `dictmultiobject.py:1421-1424 update1_keys`
                let w_keys_view = crate::call::call_function_impl_result(w_method, &[])?;
                let keys = crate::builtins::collect_iterable(w_keys_view)?;
                for k in keys {
                    let v = crate::baseobjspace::getitem(w_data, k)?;
                    dict_store_checked(dict, k, v)?;
                }
            } else {
                // `dictmultiobject.py:1410-1418 update1_pairs`
                let pairs = crate::builtins::collect_iterable(w_data)?;
                for (idx, pair) in pairs.into_iter().enumerate() {
                    let entries = crate::builtins::collect_iterable(pair)?;
                    if entries.len() != 2 {
                        return Err(crate::PyError::value_error(format!(
                            "dictionary update sequence element #{idx} has length {}; 2 is required",
                            entries.len()
                        )));
                    }
                    dict_store_checked(dict, entries[0], entries[1])?;
                }
            }
        }
    }
    Ok(())
}

/// `dictmultiobject.py:1430-1443 init_or_update` — shared by `dict.__init__`
/// and `dict.update`; `name` selects the error-message verb (`"dict"` vs
/// `"update"`). Stores resolve the subclass backing before writing, so a dict
/// subclass instance is updated through its backing dict rather than its own
/// (uninitialised) strategy slot.
///
/// `__pyre_kw__`-marked dict is the kwargs vehicle pyre's CALL_KW
/// emits for builtin callees (`call.rs:727-744`).
pub fn dict_init_or_update(
    args: &[PyObjectRef],
    name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "dict init_or_update needs the receiver");
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at most 1 argument, got {}",
            positional.len() - 1
        )));
    }
    let dict = positional[0];
    if let Some(other) = positional.get(1).copied() {
        dict_update1(dict, other)?;
    }
    let backing = resolve_dict_backing(dict);
    if backing.is_null() {
        // A dict subclass declared with `__slots__` has no attribute storage
        // for its item backing (pyre keeps a dict subclass's items in an
        // instance attribute), so there is nowhere to merge into. Full
        // slotted-dict-subclass support needs intrinsic dict backing.
        return Ok(w_none());
    }
    if let Some(kwargs) = kwargs_dict {
        unsafe {
            for (k, v) in pyre_object::w_dict_items(kwargs) {
                if pyre_object::is_str(k)
                    && pyre_object::w_str_get_wtf8(k).as_str() == Ok("__pyre_kw__")
                {
                    continue;
                }
                dict_store_checked(backing, k, v)?;
            }
        }
    }
    dict_sync_dict_storage_proxy(backing);
    Ok(w_none())
}

/// `dictmultiobject.py:137-139 descr_update` → `init_or_update`; the verb in
/// the arity error is `update`.
pub fn dict_method_update(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "dict.update() needs the receiver");
    dict_init_or_update(args, "update")
}

/// `dictmultiobject.py:1380-1386 update1` —
///
/// ```python
/// if (isinstance(w_data, W_DictMultiObject) and
///         space.is_w(
///             space.findattr(space.type(w_data), w_st_iter),
///             space.findattr(space.w_dict, w_st_iter))):
///     update1_dict_dict(space, w_dict, w_data)
/// ```
///
/// Returns True when `other` is either a real `dict` (no subclass)
/// or a `dict` subclass that hasn't shadowed `__iter__`.  Pyre's
/// `is_dict` already established the W_DictMultiObject side; this
/// helper performs the `findattr` identity check to keep
/// `__iter__`-overriding subclasses on the slow `keys()` path.
fn dict_subclass_uses_default_iter(other: PyObjectRef) -> bool {
    let Some(other_type) = crate::typedef::r#type(other) else {
        return false;
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
    if dict_type.is_null() {
        // No registered dict typeobject yet (init order quirk) —
        // degrade to "fast path" (treat as plain dict) to preserve
        // current behaviour.
        return true;
    }
    // Real `dict` type — no subclass at all, so __iter__ is by
    // definition unshadowed.
    if std::ptr::eq(other_type as *const _, dict_type as *const _) {
        return true;
    }
    let other_iter = unsafe { crate::baseobjspace::lookup_in_type(other_type, "__iter__") };
    let dict_iter = unsafe { crate::baseobjspace::lookup_in_type(dict_type, "__iter__") };
    match (other_iter, dict_iter) {
        (Some(a), Some(b)) => std::ptr::eq(a, b),
        _ => false,
    }
}

/// If dict has a dict_storage_proxy (i.e. it was returned by `globals()`),
/// sync all str-keyed entries back to the DictStorage.
///
/// For `W_ModuleDictObject` every str-keyed write already fires
/// `maybe_sync_dict_storage_store` through `w_module_dict_setitem_str`
/// (`dictmultiobject.rs:362-398`), so the redundant walk would only
/// duplicate work — and would mis-cast the module-dict layout to the
/// regular `W_DictObject` shape (different field offsets for
/// `entries` / `dstorage`).  Early-return for module dicts.
fn dict_sync_dict_storage_proxy(dict: PyObjectRef) {
    unsafe {
        if dict.is_null() {
            return;
        }
        if pyre_object::dictmultiobject::is_module_dict(dict) {
            return;
        }
        let ns_ptr = pyre_object::w_dict_get_dict_storage_proxy(dict);
        if ns_ptr.is_null() {
            return;
        }
        let ns = &mut *(ns_ptr as *mut crate::DictStorage);
        let entries = pyre_object::dictmultiobject::w_dict_object_storage(dict);
        for (k, v) in entries {
            if pyre_object::is_str(k.obj) {
                let name = pyre_object::w_str_get_value(k.obj);
                crate::dict_storage_store(ns, name, *v);
            }
        }
    }
}

/// `dictmultiobject.py:246-255 descr_pop` →
/// `strategy.pop(self, w_key, w_default)` — single-operation pop
/// via strategy dispatch (one hash).
pub fn dict_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "dict.pop() takes at least 1 argument");
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied();
    if !dict.is_null() {
        unsafe {
            match pyre_object::dictmultiobject::w_dict_pop_checked(dict, key) {
                Ok(Some(val)) => return Ok(val),
                Ok(None) => {}
                Err(_) => return Err(crate::baseobjspace::take_pending_hash_error()),
            }
        }
    }
    default.ok_or_else(|| crate::PyError::key_error_with_key(key))
}

/// `pypy/objspace/std/dictmultiobject.py:1395 W_DictMultiObject.descr_popitem`:
///
/// ```python
/// def descr_popitem(self, space):
///     try:
///         w_key, w_value = self.popitem()
///     except KeyError:
///         raise oefmt(space.w_KeyError, "dictionary is empty")
///     return space.newtuple([w_key, w_value])
/// ```
///
/// In Python 3.7+ `popitem` is LIFO (returns the most recently
/// inserted pair); pyre's `w_dict_items` preserves insertion order
/// so popping the last entry matches the spec.
pub fn dict_method_popitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
    }
    unsafe {
        if pyre_object::w_dict_len(dict) == 0 {
            return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
        }
        let items = pyre_object::w_dict_items(dict);
        let (k, v) = items
            .last()
            .copied()
            .ok_or_else(|| crate::PyError::key_error("popitem(): dictionary is empty"))?;
        pyre_object::dictmultiobject::w_dict_delitem(dict, k);
        Ok(pyre_object::w_tuple_new(vec![k, v]))
    }
}

/// `dictmultiobject.py:267-269 descr_setdefault` →
/// `self.setdefault(w_key, w_default)` — delegates to
/// `strategy.setdefault` as a single atomic operation (one hash).
pub fn dict_method_setdefault(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if !dict.is_null() {
        unsafe {
            return pyre_object::dictmultiobject::w_dict_setdefault_checked(dict, key, default)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error());
        }
    }
    Ok(default)
}

#[cfg(test)]
mod dict_method_tests {
    use super::*;

    use crate::test_hooks::install_hash_hook;

    fn assert_type_error<T: std::fmt::Debug>(result: Result<T, crate::PyError>) {
        let err = result.expect_err("operation should reject unhashable dict key");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    #[test]
    fn dict_get_rejects_unhashable_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_get(&[dict, key]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setitem_rejects_unhashable_key_without_inserting() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(crate::baseobjspace::setitem(dict, key, w_int_new(1)));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setdefault_rejects_unhashable_key() {
        // `dictmultiobject.py:749-753 EmptyDictStrategy.setdefault`:
        //   self.switch_to_correct_strategy(w_dict, w_key)
        //   w_dict.setitem(w_key, w_default)
        // `w_dict.setitem` hashes the key via the object strategy's
        // `space.hash_w`, so an unhashable key raises TypeError before
        // anything is stored — the dict stays empty.
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_setdefault(&[dict, key, w_int_new(1)]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_pop_empty_returns_default_without_hashing_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let default = w_int_new(42);

        let result = dict_method_pop(&[dict, key, default]).expect("default should be returned");
        assert_eq!(result, default);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_pop_empty_without_default_raises_keyerror_not_typeerror() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        let err = dict_method_pop(&[dict, key]).expect_err("missing key should raise KeyError");
        assert_eq!(err.kind, crate::PyErrorKind::KeyError);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_update_pairs_rejects_unhashable_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let pair = w_tuple_new(vec![key, w_int_new(1)]);
        let pairs = w_list_new(vec![pair]);

        assert_type_error(dict_method_update(&[dict, pairs]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }
}

// ── Tuple methods ────────────────────────────────────────────────────

/// PyPy: tupleobject.py descr_index — tuple.index(value)
pub fn tuple_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "index() takes at least 1 argument");
    let tup = args[0];
    let value = args[1];
    unsafe {
        let n = w_tuple_len(tup);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(tup, i as i64) {
                if std::ptr::eq(item, value) {
                    return Ok(w_int_new(i as i64));
                }
                if is_int(item) && is_int(value) && w_int_get_value(item) == w_int_get_value(value)
                {
                    return Ok(w_int_new(i as i64));
                }
            }
        }
    }
    Err(crate::PyError::value_error(
        "tuple.index(x): x not in tuple",
    ))
}

/// PyPy: tupleobject.py descr_count — tuple.count(value)
pub fn tuple_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "count() takes exactly 1 argument");
    let tup = args[0];
    let value = args[1];
    let mut count: i64 = 0;
    unsafe {
        let n = w_tuple_len(tup);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(tup, i as i64) {
                if std::ptr::eq(item, value)
                    || (is_int(item)
                        && is_int(value)
                        && w_int_get_value(item) == w_int_get_value(value))
                {
                    count += 1;
                }
            }
        }
    }
    Ok(w_int_new(count))
}
