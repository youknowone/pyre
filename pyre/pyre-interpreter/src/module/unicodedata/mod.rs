//! unicodedata module — PyPy: `pypy/module/unicodedata/`.
//!
//! Real implementation backed by the runtime-independent
//! `rustpython-unicode` crate.  The module-level functions read the latest
//! bundled database; `unicodedata.ucd_3_2_0` reads the Unicode 3.2.0 view
//! (used by `stringprep`).  `name` / `lookup` / `normalize` / `is_normalized`
//! are version-independent, matching the crate, so the 3.2.0 instance shares
//! those callables with the module.
//!
//! Signatures and error types/messages follow CPython 3.14.
//!
//! Data-version caveat: the pinned crate bundles a newer Unicode release
//! (`unidata_version` reports 17.0.0) than CPython 3.14 (16.0.0), so results
//! differ on code points that release newly assigns (they read as unassigned
//! under 16.0.0), and on two assigned-character cases the crate handles
//! differently: `decomposition` of Hangul LVT syllables yields the two-part
//! `<LV> <T>` form rather than the fully expanded jamo, and `numeric` of
//! vulgar-fraction code points carries f32-rounded values.  These are
//! upstream-crate properties, to reconcile when the pin is revisited.

use pyre_object::*;
use rustpython_unicode::{self as ucd_core, NormalizeForm};
use rustpython_wtf8::{CodePoint, Wtf8};

use crate::{PyError, PyErrorKind};

pub(crate) fn character_name(ch: char) -> Option<String> {
    ucd_core::character_name(ch)
}

type PyResult = Result<PyObjectRef, PyError>;

/// Latest bundled database view — the module-level functions.
const MODERN: ucd_core::Ucd = ucd_core::Ucd::new(true);
/// Unicode 3.2.0 view — `unicodedata.ucd_3_2_0`.
const LEGACY: ucd_core::Ucd = ucd_core::Ucd::new(false);

/// Extract the single-code-point argument of a character function.
///
/// `argno` is `Some(1)` for functions that also accept an optional default
/// (the argument clinic numbers those "argument 1"); `None` for single-argument
/// functions.  A non-string argument and a string not exactly one code point
/// long both raise `TypeError`, matching the clinic's converter messages.
fn extract_char(func: &str, argno: Option<u32>, obj: PyObjectRef) -> Result<CodePoint, PyError> {
    let argword = match argno {
        Some(n) => format!("argument {n}"),
        None => "argument".to_string(),
    };
    if !unsafe { is_str(obj) } {
        let ty = crate::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() {argword} must be a unicode character, not {ty}"
        )));
    }
    let s = unsafe { w_str_get_wtf8(obj) };
    let mut cps = s.code_points();
    if let Some(cp) = cps.next()
        && cps.next().is_none()
    {
        return Ok(cp);
    }
    let n = s.code_points().count();
    Err(PyError::type_error(format!(
        "{func}(): {argword} must be a unicode character, not a string of length {n}"
    )))
}

/// Single required character argument (`category`, `bidirectional`, …).
fn one_char(func: &str, args: &[PyObjectRef]) -> Result<CodePoint, PyError> {
    if args.len() != 1 {
        return Err(PyError::type_error(format!(
            "unicodedata.{func}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    extract_char(func, None, args[0])
}

/// Character argument plus an optional default (`digit`, `decimal`, `numeric`,
/// `name`).
fn char_and_default(
    func: &str,
    args: &[PyObjectRef],
) -> Result<(CodePoint, Option<PyObjectRef>), PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(format!(
            "{func} expected at least 1 argument, got 0"
        )));
    }
    if args.len() > 2 {
        return Err(PyError::type_error(format!(
            "{func} expected at most 2 arguments, got {}",
            args.len()
        )));
    }
    let cp = extract_char(func, Some(1), args[0])?;
    Ok((cp, args.get(1).copied()))
}

// Version-sensitive queries: the module-level `fn` uses `MODERN`, the
// `*_old` twin (bound onto `ucd_3_2_0`) uses `LEGACY`.

fn category_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new(db.category(one_char("category", args)?)))
}
fn category(args: &[PyObjectRef]) -> PyResult {
    category_impl(&MODERN, args)
}
fn category_old(args: &[PyObjectRef]) -> PyResult {
    category_impl(&LEGACY, args)
}

fn bidirectional_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new(
        db.bidirectional(one_char("bidirectional", args)?),
    ))
}
fn bidirectional(args: &[PyObjectRef]) -> PyResult {
    bidirectional_impl(&MODERN, args)
}
fn bidirectional_old(args: &[PyObjectRef]) -> PyResult {
    bidirectional_impl(&LEGACY, args)
}

fn east_asian_width_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new(
        db.east_asian_width(one_char("east_asian_width", args)?),
    ))
}
fn east_asian_width(args: &[PyObjectRef]) -> PyResult {
    east_asian_width_impl(&MODERN, args)
}
fn east_asian_width_old(args: &[PyObjectRef]) -> PyResult {
    east_asian_width_impl(&LEGACY, args)
}

fn combining_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(db.combining(one_char("combining", args)?) as i64))
}
fn combining(args: &[PyObjectRef]) -> PyResult {
    combining_impl(&MODERN, args)
}
fn combining_old(args: &[PyObjectRef]) -> PyResult {
    combining_impl(&LEGACY, args)
}

fn mirrored_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(db.mirrored(one_char("mirrored", args)?) as i64))
}
fn mirrored(args: &[PyObjectRef]) -> PyResult {
    mirrored_impl(&MODERN, args)
}
fn mirrored_old(args: &[PyObjectRef]) -> PyResult {
    mirrored_impl(&LEGACY, args)
}

fn decomposition_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new(
        &db.decomposition(one_char("decomposition", args)?),
    ))
}
fn decomposition(args: &[PyObjectRef]) -> PyResult {
    decomposition_impl(&MODERN, args)
}
fn decomposition_old(args: &[PyObjectRef]) -> PyResult {
    decomposition_impl(&LEGACY, args)
}

fn digit_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("digit", args)?;
    match db.digit(cp) {
        Some(v) => Ok(w_int_new(v as i64)),
        None => default.ok_or_else(|| PyError::value_error("not a digit")),
    }
}
fn digit(args: &[PyObjectRef]) -> PyResult {
    digit_impl(&MODERN, args)
}
fn digit_old(args: &[PyObjectRef]) -> PyResult {
    digit_impl(&LEGACY, args)
}

fn decimal_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("decimal", args)?;
    match db.decimal(cp) {
        Some(v) => Ok(w_int_new(v as i64)),
        None => default.ok_or_else(|| PyError::value_error("not a decimal")),
    }
}
fn decimal(args: &[PyObjectRef]) -> PyResult {
    decimal_impl(&MODERN, args)
}
fn decimal_old(args: &[PyObjectRef]) -> PyResult {
    decimal_impl(&LEGACY, args)
}

fn numeric_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("numeric", args)?;
    match db.numeric(cp) {
        Some(v) => Ok(w_float_new(v)),
        None => default.ok_or_else(|| PyError::value_error("not a numeric character")),
    }
}
fn numeric(args: &[PyObjectRef]) -> PyResult {
    numeric_impl(&MODERN, args)
}
fn numeric_old(args: &[PyObjectRef]) -> PyResult {
    numeric_impl(&LEGACY, args)
}

// Version-independent queries: `name` / `lookup` / `normalize` /
// `is_normalized` do not depend on the database version, so the 3.2.0 instance
// binds the same callables.

fn name(args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("name", args)?;
    if let Some(name) = cp.to_char().and_then(ucd_core::character_name) {
        return Ok(w_str_new(&name));
    }
    default.ok_or_else(|| PyError::value_error("no such name"))
}

fn lookup(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(PyError::type_error(format!(
            "lookup expected 1 argument, got {}",
            args.len()
        )));
    }
    let obj = args[0];
    if !unsafe { is_str(obj) } {
        let ty = crate::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "lookup() argument must be str, not {ty}"
        )));
    }
    let name = unsafe { w_str_get_wtf8(obj) };
    if let Ok(name) = name.as_str()
        && let Some(ch) = ucd_core::lookup_character(name)
    {
        let mut buf = String::with_capacity(ch.len_utf8());
        buf.push(ch);
        return Ok(w_str_new(&buf));
    }
    Err(PyError::key_error(format!(
        "undefined character name '{}'",
        name.to_string_lossy()
    )))
}

/// Parse the normalization-form argument (`NFC`/`NFKC`/`NFD`/`NFKD`).
fn normalize_form(func: &str, obj: PyObjectRef) -> Result<NormalizeForm, PyError> {
    if !unsafe { is_str(obj) } {
        let ty = crate::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() argument 1 must be str, not {ty}"
        )));
    }
    let s = unsafe { w_str_get_wtf8(obj) }.as_str().unwrap_or_default();
    s.parse::<NormalizeForm>()
        .map_err(|()| PyError::value_error("invalid normalization form"))
}

/// Borrow the string argument to be normalized (`unistr`).
fn normalize_text(func: &str, obj: PyObjectRef) -> Result<&'static Wtf8, PyError> {
    if !unsafe { is_str(obj) } {
        let ty = crate::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() argument 2 must be str, not {ty}"
        )));
    }
    Ok(unsafe { w_str_get_wtf8(obj) })
}

fn normalize(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "normalize expected 2 arguments, got {}",
            args.len()
        )));
    }
    let form = normalize_form("normalize", args[0])?;
    let text = normalize_text("normalize", args[1])?;
    Ok(w_str_from_wtf8(ucd_core::normalize(form, text)))
}

fn is_normalized(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "is_normalized expected 2 arguments, got {}",
            args.len()
        )));
    }
    let form = normalize_form("is_normalized", args[0])?;
    let text = normalize_text("is_normalized", args[1])?;
    Ok(w_bool_from(ucd_core::is_normalized(form, text)))
}

crate::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new(&ucd_core::unicode_version()),
    },
    functions: {
        "category"         / * = category,
        "bidirectional"    / * = bidirectional,
        "east_asian_width" / * = east_asian_width,
        "combining"        / * = combining,
        "mirrored"         / * = mirrored,
        "decomposition"    / * = decomposition,
        "digit"            / * = digit,
        "decimal"          / * = decimal,
        "numeric"          / * = numeric,
        "name"             / * = name,
        "lookup"           / * = lookup,
        "normalize"        / * = normalize,
        "is_normalized"    / * = is_normalized,
    },
    extra_init: |ns| {
        // `unicodedata.ucd_3_2_0` — a `UCD` instance pinned to the Unicode
        // 3.2.0 database (used by `stringprep`).  Version-sensitive queries
        // bind their `*_old` twins (which read `Ucd::new(false)`);
        // name/lookup/normalize/is_normalized share the module callables.
        // Functions live in the instance __dict__, so attribute access
        // returns them unbound — `ucd_3_2_0.category(ch)` dispatches with the
        // single `ch` argument, exactly like `category(ch)`.
        let ucd_ty = crate::typedef::make_builtin_type("UCD", |_| {});
        unsafe { typeobject::w_type_set_hasdict(ucd_ty, true) };
        let ucd = w_instance_new(ucd_ty);
        let d = crate::baseobjspace::getdict(ucd);
        let bind = |name: &'static str, func: crate::gateway::BuiltinCodeFn| {
            let f = crate::gateway::make_module_builtin_function(name, func);
            unsafe { w_dict_setitem_str(d, name, f) };
        };
        bind("category", category_old);
        bind("bidirectional", bidirectional_old);
        bind("east_asian_width", east_asian_width_old);
        bind("combining", combining_old);
        bind("mirrored", mirrored_old);
        bind("decomposition", decomposition_old);
        bind("digit", digit_old);
        bind("decimal", decimal_old);
        bind("numeric", numeric_old);
        bind("name", name);
        bind("lookup", lookup);
        bind("normalize", normalize);
        bind("is_normalized", is_normalized);
        unsafe { w_dict_setitem_str(d, "unidata_version", w_str_new("3.2.0")) };
        crate::module_ns_store(ns, "ucd_3_2_0", ucd);
    },
}
