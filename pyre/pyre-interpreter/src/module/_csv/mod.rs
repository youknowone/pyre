//! `_csv` — interp-level CSV reader/writer accelerator for the `csv`
//! stdlib module.
//!
//! Port of `pypy/module/_csv/` (`interp_csv.py` W_Dialect + `_build_dialect`,
//! `interp_reader.py` W_Reader state machine, `interp_writer.py` W_Writer).
//! The dialect-format validation messages and the `QUOTE_STRINGS` /
//! `QUOTE_NOTNULL` quoting styles target CPython 3.14, which PyPy (3.11-era)
//! predates.
//!
//! `csv.py` does `from _csv import (Error, writer, reader, register_dialect,
//! unregister_dialect, get_dialect, list_dialects, field_size_limit,
//! QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE, QUOTE_STRINGS,
//! QUOTE_NOTNULL)` and `from _csv import Dialect`, so every one of those names
//! is exported here.

use pyre_object::PyObjectRef;
use pyre_object::gc_roots;

use crate::PyError;

// `interp_csv.py` quoting styles, extended with the 3.14 additions.
const QUOTE_MINIMAL: i64 = 0;
const QUOTE_ALL: i64 = 1;
const QUOTE_NONNUMERIC: i64 = 2;
const QUOTE_NONE: i64 = 3;
const QUOTE_STRINGS: i64 = 4;
const QUOTE_NOTNULL: i64 = 5;

// `interp_reader.py` parser states.
const START_RECORD: u8 = 0;
const START_FIELD: u8 = 1;
const ESCAPED_CHAR: u8 = 2;
const IN_FIELD: u8 = 3;
const IN_QUOTED_FIELD: u8 = 4;
const ESCAPE_IN_QUOTED_FIELD: u8 = 5;
const QUOTE_IN_QUOTED_FIELD: u8 = 6;
const EAT_CRNL: u8 = 7;
const AFTER_ESCAPED_CRNL: u8 = 8;

// `interp_reader.py FieldLimit.limit` — process-global max parsed field
// size; a plain `i64` so it needs no GC root.
static FIELD_LIMIT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(128 * 1024);

/// Resolved dialect in the parser/serializer's internal form: code points
/// for the single-character options (with `None` standing in for the
/// `NOT_SET` sentinel), and plain values for the rest.
struct DialectConfig {
    delimiter: u32,
    doublequote: bool,
    escapechar: Option<u32>,
    lineterminator: String,
    quotechar: Option<u32>,
    quoting: i64,
    skipinitialspace: bool,
    strict: bool,
}

/// Build a `PyError` whose raised object is an instance of `_csv.Error`
/// (registered by the `exceptions:` block), with `msg` as the single
/// argument — `interp_csv.py W_Reader.error` / `W_Writer.error`.
fn csv_error(msg: String) -> PyError {
    let mut err = PyError::runtime_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("_csv.Error") {
        let args = [cls, pyre_object::w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

// ── dialect format parsing (`interp_csv.py` `_get_*` + `_build_dialect`) ──

fn codepoint_kind(can_be_none: bool) -> &'static str {
    if can_be_none {
        "a unicode character or None"
    } else {
        "a unicode character"
    }
}

/// `_set_char` / `_set_char_or_none` — resolve a single-character option.
/// `default` applies when the slot is absent (`PY_NULL`); a Python `None`
/// maps to `NOT_SET` (`None`) when `can_be_none`, else raises.
fn get_codepoint(
    w_src: PyObjectRef,
    default: Option<u32>,
    name: &str,
    can_be_none: bool,
) -> Result<Option<u32>, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    if unsafe { pyre_object::is_none(w_src) } {
        if can_be_none {
            return Ok(None);
        }
        return Err(PyError::type_error(format!(
            "\"{name}\" must be {}, not {}",
            codepoint_kind(can_be_none),
            unsafe { crate::baseobjspace::getfulltypename(w_src) },
        )));
    }
    if !unsafe { pyre_object::is_str(w_src) } {
        return Err(PyError::type_error(format!(
            "\"{name}\" must be {}, not {}",
            codepoint_kind(can_be_none),
            unsafe { crate::baseobjspace::getfulltypename(w_src) },
        )));
    }
    let s = unsafe { pyre_object::w_str_get_value(w_src) };
    let mut chars = s.chars();
    if let Some(c) = chars.next() {
        if chars.next().is_none() {
            return Ok(Some(c as u32));
        }
    }
    Err(PyError::type_error(format!(
        "\"{name}\" must be {}, not a string of length {}",
        codepoint_kind(can_be_none),
        s.chars().count(),
    )))
}

/// `_get_bool` — `None`/absent → default, else truthiness.
fn get_bool(w_src: PyObjectRef, default: bool) -> Result<bool, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    crate::baseobjspace::is_true(w_src)
}

/// `_get_int` — absent → default; a non-int (including Python `None`)
/// raises `TypeError`.
fn get_int(w_src: PyObjectRef, default: i64, name: &str) -> Result<i64, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    if !unsafe { pyre_object::is_int(w_src) } {
        return Err(PyError::type_error(format!(
            "\"{name}\" must be an integer"
        )));
    }
    Ok(unsafe { pyre_object::w_int_get_value(w_src) })
}

/// `_get_str` — absent → default; a non-str raises `TypeError`.
fn get_str(w_src: PyObjectRef, default: &str, name: &str) -> Result<String, PyError> {
    if w_src.is_null() {
        return Ok(default.to_string());
    }
    if !unsafe { pyre_object::is_str(w_src) } {
        return Err(PyError::type_error(format!(
            "\"{name}\" must be a string, not {}",
            unsafe { crate::baseobjspace::getfulltypename(w_src) },
        )));
    }
    Ok(unsafe { pyre_object::w_str_get_value(w_src) }.to_string())
}

/// `dialect_check_char` / `dialect_check_chars` — the cross-field
/// constraints `dialect_init` applies after each option is parsed:
/// `delimiter` / `quotechar` / `escapechar` may not be `\r` / `\n`, may not
/// be a space when `skipinitialspace` (except `delimiter`), must be pairwise
/// distinct, and may not appear in `lineterminator`. Each violation is a
/// `ValueError`.
fn validate_dialect(cfg: &DialectConfig) -> Result<(), PyError> {
    let check_char = |name: &str, c: u32, allow_space: bool| -> Result<(), PyError> {
        if c == '\r' as u32
            || c == '\n' as u32
            || (c == ' ' as u32 && cfg.skipinitialspace && !allow_space)
        {
            return Err(PyError::value_error(format!("bad {name} value")));
        }
        Ok(())
    };
    check_char("delimiter", cfg.delimiter, true)?;
    if let Some(e) = cfg.escapechar {
        check_char("escapechar", e, false)?;
    }
    if let Some(q) = cfg.quotechar {
        check_char("quotechar", q, false)?;
    }
    let pairs = [
        (
            "delimiter",
            "escapechar",
            Some(cfg.delimiter),
            cfg.escapechar,
        ),
        ("delimiter", "quotechar", Some(cfg.delimiter), cfg.quotechar),
        ("escapechar", "quotechar", cfg.escapechar, cfg.quotechar),
    ];
    for (n1, n2, a, b) in pairs {
        if let (Some(x), Some(y)) = (a, b) {
            if x == y {
                return Err(PyError::value_error(format!("bad {n1} or {n2} value")));
            }
        }
    }
    for c in cfg.lineterminator.chars() {
        let cp = c as u32;
        if cp == cfg.delimiter || Some(cp) == cfg.quotechar || Some(cp) == cfg.escapechar {
            return Err(PyError::value_error(
                "bad dialect value: a special character is also in the lineterminator".to_string(),
            ));
        }
    }
    Ok(())
}

fn valid_quoting(q: i64) -> bool {
    (QUOTE_MINIMAL..=QUOTE_NOTNULL).contains(&q)
}

/// `_fetch` — `space.findattr`; a missing attribute (AttributeError) is the
/// "not provided" marker (`PY_NULL`), other errors propagate.
fn fetch(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, PyError> {
    match crate::baseobjspace::getattr_str(obj, name) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(pyre_object::PY_NULL),
        Err(e) => Err(e),
    }
}

fn is_csv_dialect(obj: PyObjectRef) -> Result<bool, PyError> {
    crate::baseobjspace::isinstance(obj, dialect_class::type_object())
}

enum BuildOutcome {
    Existing(PyObjectRef),
    Config(DialectConfig),
}

/// `interp_csv.py _build_dialect`. Each `w_*` is `PY_NULL` when the option
/// was not supplied; a string `w_dialect` is resolved through the registry,
/// and an unmodified `W_Dialect` short-circuits.
#[allow(clippy::too_many_arguments)]
fn build_dialect_config(
    w_dialect: PyObjectRef,
    mut w_delimiter: PyObjectRef,
    mut w_doublequote: PyObjectRef,
    mut w_escapechar: PyObjectRef,
    mut w_lineterminator: PyObjectRef,
    mut w_quotechar: PyObjectRef,
    mut w_quoting: PyObjectRef,
    mut w_skipinitialspace: PyObjectRef,
    mut w_strict: PyObjectRef,
) -> Result<BuildOutcome, PyError> {
    if !w_dialect.is_null() {
        let mut w_dialect = w_dialect;
        if unsafe { pyre_object::is_str(w_dialect) } {
            w_dialect = lookup_registered_dialect(w_dialect)?;
        }
        if is_csv_dialect(w_dialect)?
            && w_delimiter.is_null()
            && w_doublequote.is_null()
            && w_escapechar.is_null()
            && w_lineterminator.is_null()
            && w_quotechar.is_null()
            && w_quoting.is_null()
            && w_skipinitialspace.is_null()
            && w_strict.is_null()
        {
            return Ok(BuildOutcome::Existing(w_dialect));
        }
        if w_delimiter.is_null() {
            w_delimiter = fetch(w_dialect, "delimiter")?;
        }
        if w_doublequote.is_null() {
            w_doublequote = fetch(w_dialect, "doublequote")?;
        }
        if w_escapechar.is_null() {
            w_escapechar = fetch(w_dialect, "escapechar")?;
        }
        if w_lineterminator.is_null() {
            w_lineterminator = fetch(w_dialect, "lineterminator")?;
        }
        if w_quotechar.is_null() {
            w_quotechar = fetch(w_dialect, "quotechar")?;
        }
        if w_quoting.is_null() {
            w_quoting = fetch(w_dialect, "quoting")?;
        }
        if w_skipinitialspace.is_null() {
            w_skipinitialspace = fetch(w_dialect, "skipinitialspace")?;
        }
        if w_strict.is_null() {
            w_strict = fetch(w_dialect, "strict")?;
        }
    }

    let delimiter = get_codepoint(w_delimiter, Some(',' as u32), "delimiter", false)?;
    let doublequote = get_bool(w_doublequote, true)?;
    let escapechar = get_codepoint(w_escapechar, None, "escapechar", true)?;
    let lineterminator = get_str(w_lineterminator, "\r\n", "lineterminator")?;
    let mut quoting = get_int(w_quoting, QUOTE_MINIMAL, "quoting")?;
    if !valid_quoting(quoting) {
        return Err(PyError::type_error("bad \"quoting\" value"));
    }
    // `quotechar=None` with no explicit `quoting` forces `QUOTE_NONE`.
    if !w_quotechar.is_null() && unsafe { pyre_object::is_none(w_quotechar) } && w_quoting.is_null()
    {
        quoting = QUOTE_NONE;
    }
    let quotechar = get_codepoint(w_quotechar, Some('"' as u32), "quotechar", true)?;
    let skipinitialspace = get_bool(w_skipinitialspace, false)?;
    let strict = get_bool(w_strict, false)?;

    let delimiter = delimiter
        .ok_or_else(|| PyError::type_error("\"delimiter\" must be a 1-character string"))?;
    if quoting != QUOTE_NONE && quotechar.is_none() {
        return Err(PyError::type_error(
            "quotechar must be set if quoting enabled",
        ));
    }

    let cfg = DialectConfig {
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        skipinitialspace,
        strict,
    };
    validate_dialect(&cfg)?;
    Ok(BuildOutcome::Config(cfg))
}

fn char_obj(cp: u32) -> PyObjectRef {
    let s: String = char::from_u32(cp)
        .map(|c| c.to_string())
        .unwrap_or_default();
    pyre_object::w_str_new(&s)
}

fn opt_char_obj(cp: Option<u32>) -> PyObjectRef {
    match cp {
        Some(c) => char_obj(c),
        None => pyre_object::w_none(),
    }
}

/// Materialise a `DialectConfig` as a `_csv.Dialect` instance. The canonical
/// values are kept in private slots (`_csv_*`); the public `delimiter` /
/// `quotechar` / ... names are read-only GetSetProperties (see
/// `dialect_class`) that surface them — matching the read-only
/// `interp_attrproperty` / GetSetProperty layout of `W_Dialect.typedef`.
fn config_to_dialect(cfg: &DialectConfig) -> Result<PyObjectRef, PyError> {
    let d = pyre_object::w_instance_new(dialect_class::type_object());
    let _roots = gc_roots::push_roots();
    let slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(d);
    let set = |name: &str, val: PyObjectRef| -> Result<(), PyError> {
        let d = gc_roots::shadow_stack_get(slot);
        crate::baseobjspace::setattr_str(d, name, val)?;
        Ok(())
    };
    set("_csv_delimiter", char_obj(cfg.delimiter))?;
    set(
        "_csv_doublequote",
        pyre_object::w_bool_from(cfg.doublequote),
    )?;
    set("_csv_escapechar", opt_char_obj(cfg.escapechar))?;
    set(
        "_csv_lineterminator",
        pyre_object::w_str_new(&cfg.lineterminator),
    )?;
    set("_csv_quotechar", opt_char_obj(cfg.quotechar))?;
    set("_csv_quoting", pyre_object::w_int_new(cfg.quoting))?;
    set(
        "_csv_skipinitialspace",
        pyre_object::w_bool_from(cfg.skipinitialspace),
    )?;
    set("_csv_strict", pyre_object::w_bool_from(cfg.strict))?;
    Ok(gc_roots::shadow_stack_get(slot))
}

fn read_char_field(d: PyObjectRef, name: &str) -> Result<Option<u32>, PyError> {
    let v = crate::baseobjspace::getattr_str(d, name)?;
    if unsafe { pyre_object::is_none(v) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_str(v) } {
        let mut chars = unsafe { pyre_object::w_str_get_value(v) }.chars();
        if let Some(c) = chars.next() {
            if chars.next().is_none() {
                return Ok(Some(c as u32));
            }
        }
    }
    Ok(None)
}

/// Recover the internal `DialectConfig` from a `_csv.Dialect` instance's
/// private slots — used by the reader/writer hot paths.
fn derive_config(d: PyObjectRef) -> Result<DialectConfig, PyError> {
    let delimiter = read_char_field(d, "_csv_delimiter")?.unwrap_or(',' as u32);
    let quotechar = read_char_field(d, "_csv_quotechar")?;
    let escapechar = read_char_field(d, "_csv_escapechar")?;
    let doublequote =
        crate::baseobjspace::is_true(crate::baseobjspace::getattr_str(d, "_csv_doublequote")?)?;
    let skipinitialspace = crate::baseobjspace::is_true(crate::baseobjspace::getattr_str(
        d,
        "_csv_skipinitialspace",
    )?)?;
    let strict = crate::baseobjspace::is_true(crate::baseobjspace::getattr_str(d, "_csv_strict")?)?;
    let quoting = {
        let v = crate::baseobjspace::getattr_str(d, "_csv_quoting")?;
        if unsafe { pyre_object::is_int(v) } {
            unsafe { pyre_object::w_int_get_value(v) }
        } else {
            QUOTE_MINIMAL
        }
    };
    let lineterminator = {
        let v = crate::baseobjspace::getattr_str(d, "_csv_lineterminator")?;
        if unsafe { pyre_object::is_str(v) } {
            unsafe { pyre_object::w_str_get_value(v) }.to_string()
        } else {
            "\r\n".to_string()
        }
    };
    Ok(DialectConfig {
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        skipinitialspace,
        strict,
    })
}

// ── registry (`app_csv.py`) ──

/// The module-global `_dialects` mapping, fetched through `sys.modules`
/// (GC-rooted there) rather than a thread-local raw pointer.
fn csv_dialects() -> Result<PyObjectRef, PyError> {
    let m = crate::importing::get_sys_module("_csv")
        .ok_or_else(|| PyError::runtime_error("_csv module not initialized"))?;
    crate::baseobjspace::getattr_str(m, "_dialects")
}

fn lookup_registered_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let dialects = csv_dialects()?;
    match unsafe { pyre_object::dictmultiobject::w_dict_lookup(dialects, name) } {
        Some(d) => Ok(d),
        None => Err(csv_error("unknown dialect".to_string())),
    }
}

// ── `_csv.Dialect` type ──

mod dialect_class {
    use super::*;

    // Each public dialect attribute is a read-only GetSetProperty reading the
    // corresponding `_csv_*` private slot (`interp_csv.py` GetSetProperty /
    // interp_attrproperty); a plain instance attribute would be writable and
    // deletable, which the type forbids.
    // GetSetProperty fget callbacks receive `(descriptor_self, w_obj)`, so the
    // dialect instance is at `args[1]`.
    macro_rules! dialect_getter {
        ($fn:ident, $slot:literal) => {
            fn $fn(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
                crate::baseobjspace::getattr_str(args[1], $slot)
            }
        };
    }
    dialect_getter!(get_delimiter, "_csv_delimiter");
    dialect_getter!(get_doublequote, "_csv_doublequote");
    dialect_getter!(get_escapechar, "_csv_escapechar");
    dialect_getter!(get_lineterminator, "_csv_lineterminator");
    dialect_getter!(get_quotechar, "_csv_quotechar");
    dialect_getter!(get_quoting, "_csv_quoting");
    dialect_getter!(get_skipinitialspace, "_csv_skipinitialspace");
    dialect_getter!(get_strict, "_csv_strict");

    /// `W_Dialect___new__` — build the dialect from the optional template +
    /// format options; with no template a default (excel-like) dialect.
    fn dialect_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        let template = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let outcome = build_dialect_config(
            template,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )?;
        match outcome {
            BuildOutcome::Existing(d) => Ok(d),
            BuildOutcome::Config(cfg) => config_to_dialect(&cfg),
        }
    }

    pub fn type_object() -> PyObjectRef {
        // Process-global immortal type object (see `make_builtin_type`).
        static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_csv.Dialect", |ns| {
                crate::dict_storage_store(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(dialect_new),
                );
                // `dialect_new` does all the work; a no-op `__init__`
                // keeps the template argument from reaching
                // `object.__init__`.
                crate::dict_storage_store(
                    ns,
                    "__init__",
                    crate::make_builtin_function("__init__", |_| Ok(pyre_object::w_none())),
                );
                // `W_Dialect.reduce_ex_w` — dialects are not picklable
                // (and so not copyable).
                crate::dict_storage_store(
                    ns,
                    "__reduce_ex__",
                    crate::make_builtin_function("__reduce_ex__", |_| {
                        Err(PyError::type_error("can't pickle _csv.Dialect objects"))
                    }),
                );
                crate::dict_storage_store(
                    ns,
                    "__reduce__",
                    crate::make_builtin_function("__reduce__", |_| {
                        Err(PyError::type_error("can't pickle _csv.Dialect objects"))
                    }),
                );
                for (name, getter) in [
                    (
                        "delimiter",
                        get_delimiter as fn(&[PyObjectRef]) -> Result<PyObjectRef, PyError>,
                    ),
                    ("doublequote", get_doublequote),
                    ("escapechar", get_escapechar),
                    ("lineterminator", get_lineterminator),
                    ("quotechar", get_quotechar),
                    ("quoting", get_quoting),
                    ("skipinitialspace", get_skipinitialspace),
                    ("strict", get_strict),
                ] {
                    crate::dict_storage_store(
                        ns,
                        name,
                        crate::typedef::make_getset_descriptor_named(
                            crate::make_builtin_function(name, getter),
                            name,
                        ),
                    );
                }
            });
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp as usize
        }) as PyObjectRef
    }
}

// ── `_csv.reader` ──

fn add_char(
    field: &mut String,
    field_len: &mut usize,
    c: char,
    limit: i64,
    line_num: i64,
) -> Result<(), PyError> {
    if *field_len as i64 >= limit {
        return Err(csv_error(format!(
            "line {line_num}: field larger than field limit"
        )));
    }
    field.push(c);
    *field_len += 1;
    Ok(())
}

/// `parse_save_field` — record the finished field's text, whether it was
/// unquoted, and its code-point length; the final value (str / float / None)
/// is computed once the record is complete (`parse_save_field`'s quoting
/// conversions).
fn save_field(
    fields: &mut Vec<(String, bool, usize)>,
    field: &mut String,
    field_len: &mut usize,
    unquoted: &mut bool,
) {
    let len = *field_len;
    fields.push((std::mem::take(field), *unquoted, len));
    *unquoted = true;
    *field_len = 0;
}

fn to_float(w_str: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let float_type = crate::typedef::gettypefor(&pyre_object::FLOAT_TYPE)
        .ok_or_else(|| PyError::runtime_error("float type unavailable"))?;
    crate::call::call_function_impl_result(float_type, &[w_str])
}

/// `W_Reader.next_w` — parse the next CSV record from the underlying line
/// iterator. Re-entering the reader from its own line iterator (gh-145105) is
/// rejected with a `_csv.Error`.
fn reader_next_impl(self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let reading = crate::baseobjspace::getattr_str(self_obj, "_reading")
        .ok()
        .map(|v| crate::baseobjspace::is_true(v).unwrap_or(false))
        .unwrap_or(false);
    if reading {
        return Err(csv_error("reader is already iterating".to_string()));
    }
    let _roots = gc_roots::push_roots();
    let self_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(self_obj);
    crate::baseobjspace::setattr_str(
        gc_roots::shadow_stack_get(self_slot),
        "_reading",
        pyre_object::w_bool_from(true),
    )?;
    let result = reader_next_inner(gc_roots::shadow_stack_get(self_slot));
    // FINALLY reset of the re-entrancy guard. The inner `result` must be
    // returned/propagated unchanged, so a failure of this reset write is
    // deliberately ignored rather than masking the inner exception.
    if let Err(_e) = crate::baseobjspace::setattr_str(
        gc_roots::shadow_stack_get(self_slot),
        "_reading",
        pyre_object::w_bool_from(false),
    ) {}
    result
}

fn reader_next_inner(self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let dialect_obj = crate::baseobjspace::getattr_str(self_obj, "dialect")?;
    let cfg = derive_config(dialect_obj)?;
    let limit = FIELD_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
    let mut line_num = {
        let v = crate::baseobjspace::getattr_str(self_obj, "line_num")?;
        if unsafe { pyre_object::is_int(v) } {
            unsafe { pyre_object::w_int_get_value(v) }
        } else {
            0
        }
    };

    let _roots = gc_roots::push_roots();
    let iter_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(crate::baseobjspace::getattr_str(self_obj, "_iterator")?);
    let self_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(self_obj);

    let mut fields: Vec<(String, bool, usize)> = Vec::new();
    let mut field = String::new();
    let mut field_len: usize = 0;
    let mut field_unquoted = true;
    let mut state = START_RECORD;

    'lines: loop {
        let w_iter = gc_roots::shadow_stack_get(iter_slot);
        let line = match crate::baseobjspace::next(w_iter) {
            Ok(l) => l,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                if state != START_RECORD
                    && state != EAT_CRNL
                    && (field_len > 0 || state == IN_QUOTED_FIELD)
                {
                    if cfg.strict {
                        return Err(csv_error(format!(
                            "line {line_num}: unexpected end of data"
                        )));
                    }
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    break 'lines;
                }
                return Err(PyError::stop_iteration());
            }
            Err(e) => return Err(e),
        };
        line_num += 1;
        if unsafe { pyre_object::bytesobject::is_bytes(line) } {
            return Err(csv_error(format!(
                "line {line_num}: iterator should return strings, not bytes (the file should be opened in text mode)"
            )));
        }
        if !unsafe { pyre_object::is_str(line) } {
            return Err(csv_error(format!(
                "line {line_num}: iterator should return strings, not {} (the file should be opened in text mode)",
                unsafe { crate::baseobjspace::getfulltypename(line) },
            )));
        }
        let s = unsafe { pyre_object::w_str_get_value(line) };
        for c in s.chars() {
            let cp = c as u32;
            let is_nl = cp == 10 || cp == 13;

            if state == START_RECORD {
                if is_nl {
                    state = EAT_CRNL;
                    continue;
                }
                state = START_FIELD;
            }

            if state == START_FIELD {
                if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if Some(cp) == cfg.quotechar && cfg.quoting != QUOTE_NONE {
                    field_unquoted = false;
                    state = IN_QUOTED_FIELD;
                } else if Some(cp) == cfg.escapechar {
                    state = ESCAPED_CHAR;
                } else if cp == 32 && cfg.skipinitialspace {
                    // ignore leading space
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_FIELD;
                }
            } else if state == ESCAPED_CHAR {
                add_char(&mut field, &mut field_len, c, limit, line_num)?;
                state = if is_nl { AFTER_ESCAPED_CRNL } else { IN_FIELD };
            } else if state == IN_FIELD || state == AFTER_ESCAPED_CRNL {
                if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if Some(cp) == cfg.escapechar {
                    state = ESCAPED_CHAR;
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = START_FIELD;
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                }
            } else if state == IN_QUOTED_FIELD {
                if Some(cp) == cfg.escapechar {
                    state = ESCAPE_IN_QUOTED_FIELD;
                } else if Some(cp) == cfg.quotechar && cfg.quoting != QUOTE_NONE {
                    state = if cfg.doublequote {
                        QUOTE_IN_QUOTED_FIELD
                    } else {
                        IN_FIELD
                    };
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                }
            } else if state == ESCAPE_IN_QUOTED_FIELD {
                add_char(&mut field, &mut field_len, c, limit, line_num)?;
                state = IN_QUOTED_FIELD;
            } else if state == QUOTE_IN_QUOTED_FIELD {
                if cfg.quoting != QUOTE_NONE && Some(cp) == cfg.quotechar {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_QUOTED_FIELD;
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = START_FIELD;
                } else if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if !cfg.strict {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_FIELD;
                } else {
                    let dc = char::from_u32(cfg.delimiter).unwrap_or('?');
                    let qc = cfg.quotechar.and_then(char::from_u32).unwrap_or('?');
                    return Err(csv_error(format!(
                        "line {line_num}: '{dc}' expected after '{qc}'"
                    )));
                }
            } else if state == EAT_CRNL {
                if !is_nl {
                    return Err(csv_error(format!(
                        "line {line_num}: new-line character seen in unquoted field - do you need to open the file with newline=''?"
                    )));
                }
            }
        }

        match state {
            s if s == IN_FIELD || s == QUOTE_IN_QUOTED_FIELD => {
                save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                break 'lines;
            }
            s if s == ESCAPED_CHAR => {
                add_char(&mut field, &mut field_len, '\n', limit, line_num)?;
                state = IN_FIELD;
            }
            s if s == IN_QUOTED_FIELD => {}
            s if s == ESCAPE_IN_QUOTED_FIELD => {
                add_char(&mut field, &mut field_len, '\n', limit, line_num)?;
                state = IN_QUOTED_FIELD;
            }
            s if s == START_FIELD => {
                save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                break 'lines;
            }
            s if s == AFTER_ESCAPED_CRNL => {}
            _ => break 'lines,
        }
    }

    let self_obj = gc_roots::shadow_stack_get(self_slot);
    crate::baseobjspace::setattr_str(self_obj, "line_num", pyre_object::w_int_new(line_num))?;

    let result = pyre_object::listobject::w_list_new(Vec::new());
    let result_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(result);
    for (s, unquoted, len) in fields {
        // `parse_save_field` quoting conversions: an empty unquoted field is
        // `None` under QUOTE_NOTNULL / QUOTE_STRINGS; a non-empty unquoted
        // field is coerced to `float` under QUOTE_NONNUMERIC / QUOTE_STRINGS.
        let w = if unquoted
            && len == 0
            && (cfg.quoting == QUOTE_NOTNULL || cfg.quoting == QUOTE_STRINGS)
        {
            pyre_object::w_none()
        } else {
            let ws = pyre_object::w_str_new(&s);
            if unquoted
                && len != 0
                && (cfg.quoting == QUOTE_NONNUMERIC || cfg.quoting == QUOTE_STRINGS)
            {
                to_float(ws)?
            } else {
                ws
            }
        };
        let result = gc_roots::shadow_stack_get(result_slot);
        unsafe { pyre_object::listobject::w_list_append(result, w) };
    }
    Ok(gc_roots::shadow_stack_get(result_slot))
}

mod reader_class {
    use super::*;

    crate::py_class! {
        "_csv.reader",
        methods: {
            fn __iter__(self_obj: PyObjectRef) -> PyObjectRef {
                self_obj
            }
            fn __next__(self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
                reader_next_impl(self_obj)
            }
        }
    }
}

// ── `_csv.writer` ──

fn special_chars(cfg: &DialectConfig) -> Vec<u32> {
    let mut s = vec![cfg.delimiter, 13, 10];
    for c in cfg.lineterminator.chars() {
        s.push(c as u32);
    }
    if let Some(e) = cfg.escapechar {
        s.push(e);
    }
    if let Some(q) = cfg.quotechar {
        s.push(q);
    }
    s
}

/// `W_Writer.writerow` — serialize one record.
fn writer_writerow_impl(
    self_obj: PyObjectRef,
    w_fields: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let dialect_obj = crate::baseobjspace::getattr_str(self_obj, "dialect")?;
    let cfg = derive_config(dialect_obj)?;
    let w_filewrite = crate::baseobjspace::getattr_str(self_obj, "_write")?;

    let row = match crate::builtins::collect_iterable(w_fields) {
        Ok(r) => r,
        Err(e) if e.kind == crate::PyErrorKind::TypeError => {
            let r = unsafe { crate::display::py_repr(w_fields) }.unwrap_or_default();
            return Err(csv_error(format!("iterable expected, not {r}")));
        }
        Err(e) => return Err(e),
    };

    let special = special_chars(&cfg);
    let quote_char = cfg.quotechar.and_then(char::from_u32).unwrap_or('"');
    let delim_char = char::from_u32(cfg.delimiter).unwrap_or(',');
    let n = row.len();
    let mut rec = String::new();

    for (i, &w_field) in row.iter().enumerate() {
        let field = if unsafe { pyre_object::is_none(w_field) } {
            String::new()
        } else if unsafe { pyre_object::is_float(w_field) } {
            unsafe { crate::display::py_repr(w_field) }?
        } else {
            unsafe { crate::display::py_str(w_field) }?
        };

        let mut quoted = match cfg.quoting {
            QUOTE_NONNUMERIC => crate::baseobjspace::float_w(w_field).is_err(),
            QUOTE_ALL => true,
            QUOTE_MINIMAL => {
                let mut q = false;
                for c in field.chars() {
                    let cp = c as u32;
                    if !special.contains(&cp) {
                        continue;
                    }
                    if Some(cp) == cfg.escapechar {
                        continue;
                    }
                    if Some(cp) != cfg.quotechar || cfg.doublequote {
                        q = true;
                        break;
                    }
                }
                q
            }
            QUOTE_STRINGS => unsafe { pyre_object::is_str(w_field) },
            QUOTE_NOTNULL => !unsafe { pyre_object::is_none(w_field) },
            _ => false,
        };

        // An empty field can only be represented by quoting it. The quoting
        // styles that never quote a field that is not already quoted
        // (QUOTE_NONE and — for a non-quotable value — QUOTE_STRINGS /
        // QUOTE_NOTNULL) raise instead of silently dropping it.
        let cannot_force_quote = cfg.quoting == QUOTE_NONE
            || cfg.quoting == QUOTE_STRINGS
            || cfg.quoting == QUOTE_NOTNULL;
        if field.is_empty() {
            if cfg.delimiter == ' ' as u32 && cfg.skipinitialspace && !quoted {
                if cannot_force_quote {
                    return Err(csv_error(
                        "empty field must be quoted if delimiter is a space and skipinitialspace is true"
                            .to_string(),
                    ));
                }
                quoted = true;
            }
            if n == 1 && !quoted {
                if cannot_force_quote {
                    return Err(csv_error(
                        "single empty field record must be quoted".to_string(),
                    ));
                }
                quoted = true;
            }
        }

        if i > 0 {
            rec.push(delim_char);
        }
        if quoted {
            rec.push(quote_char);
        }

        for c in field.chars() {
            let cp = c as u32;
            if special.contains(&cp) {
                let want_escape = if cfg.quoting == QUOTE_NONE {
                    true
                } else {
                    let mut we = false;
                    if Some(cp) == cfg.quotechar {
                        if cfg.doublequote {
                            rec.push(quote_char);
                        } else {
                            we = true;
                        }
                    }
                    if Some(cp) == cfg.escapechar {
                        we = true;
                    }
                    we
                };
                if want_escape {
                    match cfg.escapechar.and_then(char::from_u32) {
                        Some(e) => rec.push(e),
                        None => {
                            return Err(csv_error(
                                "need to escape, but no escapechar set".to_string(),
                            ));
                        }
                    }
                }
            }
            rec.push(c);
        }

        if quoted {
            rec.push(quote_char);
        }
    }

    rec.push_str(&cfg.lineterminator);
    crate::call::call_function_impl_result(w_filewrite, &[pyre_object::w_str_new(&rec)])
}

/// `W_Writer.writerows` — serialize a sequence of records.
fn writer_writerows_impl(
    self_obj: PyObjectRef,
    w_seqseq: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let it = crate::baseobjspace::iter(w_seqseq)?;
    let _roots = gc_roots::push_roots();
    let it_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(it);
    let self_slot = gc_roots::shadow_stack_len();
    gc_roots::pin_root(self_obj);
    loop {
        let it = gc_roots::shadow_stack_get(it_slot);
        let row = match crate::baseobjspace::next(it) {
            Ok(r) => r,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        };
        writer_writerow_impl(gc_roots::shadow_stack_get(self_slot), row)?;
    }
    Ok(pyre_object::w_none())
}

mod writer_class {
    use super::*;

    crate::py_class! {
        "_csv.writer",
        methods: {
            fn writerow(self_obj: PyObjectRef, row: PyObjectRef) -> Result<PyObjectRef, PyError> {
                writer_writerow_impl(self_obj, row)
            }
            fn writerows(self_obj: PyObjectRef, rows: PyObjectRef) -> Result<PyObjectRef, PyError> {
                writer_writerows_impl(self_obj, rows)
            }
        }
    }
}

/// Resolve the dialect object for a reader/writer constructor.
#[allow(clippy::too_many_arguments)]
fn resolve_dialect(
    w_dialect: PyObjectRef,
    w_delimiter: PyObjectRef,
    w_doublequote: PyObjectRef,
    w_escapechar: PyObjectRef,
    w_lineterminator: PyObjectRef,
    w_quotechar: PyObjectRef,
    w_quoting: PyObjectRef,
    w_skipinitialspace: PyObjectRef,
    w_strict: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let outcome = build_dialect_config(
        w_dialect,
        w_delimiter,
        w_doublequote,
        w_escapechar,
        w_lineterminator,
        w_quotechar,
        w_quoting,
        w_skipinitialspace,
        w_strict,
    )?;
    match outcome {
        BuildOutcome::Existing(d) => Ok(d),
        BuildOutcome::Config(cfg) => config_to_dialect(&cfg),
    }
}

/// `app_csv.list_dialects` — the registered dialect names; takes no args.
fn list_dialects_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if !args.is_empty() {
        return Err(PyError::type_error(format!(
            "list_dialects() takes no arguments ({} given)",
            args.len()
        )));
    }
    let dialects = csv_dialects()?;
    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(dialects) };
    Ok(pyre_object::listobject::w_list_new(
        items.into_iter().map(|(k, _)| k).collect(),
    ))
}

/// `csv_field_size_limit` — return the current limit, and set it when an
/// integer argument is supplied.
fn field_size_limit_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() > 1 {
        return Err(PyError::type_error(format!(
            "field_size_limit() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let old = FIELD_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
    if let Some(&v) = args.first() {
        if !unsafe { pyre_object::is_int(v) } {
            return Err(PyError::type_error("limit must be an integer"));
        }
        FIELD_LIMIT.store(
            unsafe { pyre_object::w_int_get_value(v) },
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Ok(pyre_object::w_int_new(old))
}

crate::py_module! {
    "_csv",
    interpleveldefs: {
        "Dialect" => dialect_class::type_object(),
        "__version__" => pyre_object::w_str_new("1.0"),
    },
    int_constants: {
        "QUOTE_MINIMAL" => QUOTE_MINIMAL,
        "QUOTE_ALL" => QUOTE_ALL,
        "QUOTE_NONNUMERIC" => QUOTE_NONNUMERIC,
        "QUOTE_NONE" => QUOTE_NONE,
        "QUOTE_STRINGS" => QUOTE_STRINGS,
        "QUOTE_NOTNULL" => QUOTE_NOTNULL,
    },
    exceptions: {
        "Error" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _csv init"),
    },
    inline_functions: {
        // `csv_reader` — build the reader over an iterable of lines.
        fn reader(
            iterable: PyObjectRef,
            #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let w_iter = crate::baseobjspace::iter(iterable)?;
            let dialect_obj = resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            )?;
            let r = pyre_object::w_instance_new(reader_class::type_object());
            let _roots = gc_roots::push_roots();
            let slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(r);
            gc_roots::pin_root(dialect_obj);
            gc_roots::pin_root(w_iter);
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "dialect", gc_roots::shadow_stack_get(slot + 1))?;
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_iterator", gc_roots::shadow_stack_get(slot + 2))?;
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "line_num", pyre_object::w_int_new(0))?;
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_reading", pyre_object::w_bool_from(false))?;
            Ok(gc_roots::shadow_stack_get(slot))
        }

        // `csv_writer` — build the writer over a file-like object's `write`.
        fn writer(
            fileobj: PyObjectRef,
            #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let dialect_obj = resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            )?;
            // A missing `write` attribute is a TypeError ("argument 1 must
            // have a write method"); a `write` whose access itself raises
            // (e.g. a property) propagates that error unchanged.
            let w_write = match crate::baseobjspace::getattr_str(fileobj, "write") {
                Ok(w) => w,
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                    return Err(PyError::type_error("argument 1 must have a write method"));
                }
                Err(e) => return Err(e),
            };
            let w = pyre_object::w_instance_new(writer_class::type_object());
            let _roots = gc_roots::push_roots();
            let slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(w);
            gc_roots::pin_root(dialect_obj);
            gc_roots::pin_root(w_write);
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "dialect", gc_roots::shadow_stack_get(slot + 1))?;
            crate::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_write", gc_roots::shadow_stack_get(slot + 2))?;
            Ok(gc_roots::shadow_stack_get(slot))
        }

        // `app_csv.register_dialect` — validate + register under `name`.
        fn register_dialect(
            name: PyObjectRef,
            #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            if !unsafe { pyre_object::is_str(name) } {
                return Err(PyError::type_error("dialect name must be a string"));
            }
            let dialect_obj = resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            )?;
            let dialects = csv_dialects()?;
            unsafe { pyre_object::dictmultiobject::w_dict_store(dialects, name, dialect_obj) };
            Ok(pyre_object::w_none())
        }

        // `app_csv.unregister_dialect`.
        fn unregister_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
            let dialects = csv_dialects()?;
            if unsafe { pyre_object::dictmultiobject::w_dict_delitem(dialects, name) } {
                Ok(pyre_object::w_none())
            } else {
                Err(csv_error("unknown dialect".to_string()))
            }
        }

        // `app_csv.get_dialect`.
        fn get_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
            lookup_registered_dialect(name)
        }
    },
    functions: {
        // `list_dialects` / `field_size_limit` are varargs so the
        // "takes no arguments" / "at most 1 argument" guards can be enforced
        // (the flat builtin ABI does not reject extra positionals on its own).
        "list_dialects" / * = list_dialects_fn,
        "field_size_limit" / * = field_size_limit_fn,
    },
    extra_init: |ns| {
        // `app_csv._dialects = {}` — the registry mapping, kept in the module
        // namespace so it is reachable for GC.
        crate::dict_storage_store(ns, "_dialects", pyre_object::w_dict_new());
    },
}
