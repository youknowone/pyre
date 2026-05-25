use std::fmt;

use pyre_object::pyobject::{
    BOOL_TYPE, ELLIPSIS_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE, LONG_TYPE, MODULE_TYPE,
    NONE_TYPE, PyObjectRef, PyType, STR_TYPE, TYPE_TYPE,
};

use crate::{
    BUILTIN_CODE_TYPE, BUILTIN_FUNCTION_TYPE, FUNCTION_TYPE, builtin_code_name, function_get_name,
};

/// Try to call a dunder method (__repr__, __str__, etc.) on an instance.
///
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, name), w_obj)`
/// Uses the unified `call_function` instead of a dedicated callback.
fn try_call_dunder(obj: PyObjectRef, name: &str) -> Option<String> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return None;
        }
        let method = crate::baseobjspace::lookup(obj, name)?;
        if method.is_null() {
            return None;
        }
        let result = crate::call_function(method, &[obj]);
        if result.is_null() {
            return None;
        }
        if pyre_object::is_str(result) {
            return Some(pyre_object::w_str_get_value(result).to_string());
        }
    }
    None
}

/// `pypy/objspace/std/floatobject.py W_FloatObject.descr_repr` parity.
/// CPython prints lowercase `nan` / `inf`, uses scientific notation for
/// magnitudes outside `[1e-4, 1e17)` (approximately), and otherwise
/// uses positional form with at most 17 significant digits.  Pyre's
/// approximation:
///   - integral floats in the positional band → `"<n>.0"`
///   - magnitude < 1e-4 or >= 1e16 → `"{:e}"` with explicit sign
///   - otherwise → Rust's `Display` (`{}`)
fn format_float_repr(val: f64) -> String {
    if val.is_nan() {
        return "nan".to_string();
    }
    if val.is_infinite() {
        return if val < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    let abs = val.abs();
    if val == 0.0 {
        return if val.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    if abs >= 1e16 || abs < 1e-4 {
        // Build `{m}e[+|-]NN` in CPython's float_repr style:
        // exponent is signed, two-digit minimum.  Rust's `{:e}` emits
        // unsigned exponent with no padding (`1e100`); rewrite to match
        // CPython's `1e+100` / `1.5e-10`.
        let raw = format!("{val:e}");
        if let Some(epos) = raw.find('e') {
            let (mantissa, exp) = raw.split_at(epos);
            let exp = &exp[1..]; // drop 'e'
            let (sign, mag) = if let Some(rest) = exp.strip_prefix('-') {
                ("-", rest)
            } else if let Some(rest) = exp.strip_prefix('+') {
                ("+", rest)
            } else {
                ("+", exp)
            };
            let mag_padded = if mag.len() < 2 {
                format!("0{mag}")
            } else {
                mag.to_string()
            };
            return format!("{mantissa}e{sign}{mag_padded}");
        }
        return raw;
    }
    if val.fract() == 0.0 {
        return format!("{val:.1}");
    }
    format!("{val}")
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject._descr_repr`
/// parity — pick the outer quote (prefer single, switch to double iff
/// the string contains a single but no double), then escape backslash,
/// the matching quote, common whitespace, and control characters.
/// Non-control codepoints pass through verbatim.
fn format_str_repr(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Format a PyObjectRef for debug display.
///
/// # Safety
/// `obj` must be a valid pointer to a known Python object type.
pub unsafe fn py_repr(obj: PyObjectRef) -> String {
    let obj = crate::baseobjspace::unwrap_cell(obj);
    if obj.is_null() {
        return "NULL".to_string();
    }
    unsafe {
        let tp = (*obj).ob_type;
        if std::ptr::eq(tp, &INT_TYPE as *const PyType) {
            let int_obj = obj as *const pyre_object::intobject::W_IntObject;
            format!("{}", (*int_obj).intval)
        } else if std::ptr::eq(tp, &FLOAT_TYPE as *const PyType) {
            let float_obj = obj as *const pyre_object::floatobject::W_FloatObject;
            let val = (*float_obj).floatval;
            format_float_repr(val)
        } else if std::ptr::eq(tp, &LONG_TYPE as *const PyType) {
            let long_obj = obj as *const pyre_object::longobject::W_LongObject;
            format!("{}", &*(*long_obj).value)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const pyre_object::boolobject::W_BoolObject;
            if (*bool_obj).boolval {
                "True".to_string()
            } else {
                "False".to_string()
            }
        } else if std::ptr::eq(tp, &pyre_object::pyobject::LIST_TYPE as *const PyType) {
            let n = pyre_object::w_list_len(obj);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_list_getitem(obj, i as i64) {
                    parts.push(py_repr(item));
                }
            }
            format!("[{}]", parts.join(", "))
        } else if pyre_object::is_tuple(obj) {
            // `pyre_object::is_tuple` covers `TUPLE_TYPE` plus the
            // arity-2 specialisations (`SPECIALISED_TUPLE_{II,FF,OO}_TYPE`,
            // `pypy/objspace/std/specialisedtupleobject.py:161-167`).
            // Without this union dispatch the specialised variants
            // (returned by `w_tuple_new(items)` whenever `items.len() == 2`)
            // would fall through to the generic `<{name} object at ...>`
            // fallback — visible as `<tuple object at 0x...>` on
            // `print(e.args)` for two-arg exception constructors.
            //
            // structseq instances (`_structseq.py:43-87 structseqtype`)
            // are tuple subclasses with `w_class` pointing at a custom
            // type that installs its own `__repr__`.  Route them
            // through the subclass dunder before the generic tuple
            // formatting so `repr(pwd_entry)` prints
            // `'pwd.struct_passwd(pw_name=..., ...)'` instead of the
            // bare tuple form.  Plain `tuple()` keeps the fast path
            // because its `w_class` is the canonical tuple type.
            let w_class = (*obj).w_class;
            let tuple_class = crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
            if !w_class.is_null() && !std::ptr::eq(w_class, tuple_class) {
                // try_call_dunder gates on is_instance(obj); structseq
                // instances are tuple subclasses with ob_type ==
                // TUPLE_TYPE, so reach for the subclass __repr__
                // directly via lookup(MRO).
                if let Some(method) = crate::baseobjspace::lookup(obj, "__repr__") {
                    if !method.is_null() {
                        let r = crate::call_function(method, &[obj]);
                        if !r.is_null() && pyre_object::is_str(r) {
                            return pyre_object::w_str_get_value(r).to_string();
                        }
                    }
                }
            }
            let n = pyre_object::w_tuple_len(obj);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(obj, i as i64) {
                    parts.push(py_repr(item));
                }
            }
            if n == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        } else if unsafe { pyre_object::is_dict(obj) } {
            // `pypy/objspace/std/dictmultiobject.py:130-150 descr_repr`
            // iterates `self.iteritems()`, which dispatches to the
            // strategy on both `W_DictObject` and `W_ModuleDictObject`.
            // `w_dict_items` already routes through `is_module_dict`,
            // so reach for the unified surface instead of casting
            // through the W_DictObject layout.
            let entries = pyre_object::w_dict_items(obj);
            let mut parts = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                parts.push(format!("{}: {}", py_repr(k), py_repr(v)));
            }
            format!("{{{}}}", parts.join(", "))
        } else if pyre_object::is_bytes_like(obj) {
            // `pypy/objspace/std/bytesobject.py W_BytesObject.descr_repr`
            // and `bytearrayobject.py W_BytearrayObject.descr_repr` —
            // ASCII-printable bytes pass through, control bytes use
            // `\xNN`, single quotes get backslash-escaped (or the outer
            // quote flips to double when both quote kinds appear, but
            // pyre keeps the simpler single-quote form for now).
            // bytearray wraps the bytes literal: `bytearray(b'...')`.
            let data = pyre_object::bytes_like_data(obj);
            let mut body = String::with_capacity(data.len() + 4);
            body.push_str("b'");
            for &b in data {
                match b {
                    b'\\' => body.push_str("\\\\"),
                    b'\'' => body.push_str("\\'"),
                    b'\n' => body.push_str("\\n"),
                    b'\r' => body.push_str("\\r"),
                    b'\t' => body.push_str("\\t"),
                    0x20..=0x7e => body.push(b as char),
                    _ => body.push_str(&format!("\\x{b:02x}")),
                }
            }
            body.push('\'');
            if pyre_object::bytearrayobject::is_bytearray(obj) {
                format!("bytearray({body})")
            } else {
                body
            }
        } else if pyre_object::is_set_or_frozenset(obj) {
            // `pypy/objspace/std/setobject.py W_BaseSetObject.descr_repr`
            // → `'%s({%s})' % (typename, items_repr_joined)` for
            // frozenset and `'{%s}' % items_repr_joined` for set.  Empty
            // set keeps the `set()` constructor form.
            let items = pyre_object::w_set_items(obj);
            let parts: Vec<String> = items.iter().map(|&v| py_repr(v)).collect();
            let is_frozen = pyre_object::is_frozenset(obj);
            if items.is_empty() {
                if is_frozen {
                    "frozenset()".to_string()
                } else {
                    "set()".to_string()
                }
            } else if is_frozen {
                format!("frozenset({{{}}})", parts.join(", "))
            } else {
                format!("{{{}}}", parts.join(", "))
            }
        } else if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            format_str_repr(pyre_object::w_str_get_value(obj))
        } else if std::ptr::eq(tp, &NONE_TYPE as *const PyType) {
            "None".to_string()
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::NOTIMPLEMENTED_TYPE as *const PyType,
        ) {
            "NotImplemented".to_string()
        } else if std::ptr::eq(tp, &ELLIPSIS_TYPE as *const PyType) {
            "Ellipsis".to_string()
        } else if std::ptr::eq(tp, &BUILTIN_CODE_TYPE as *const PyType) {
            // Raw BuiltinCode objects (Code-level, not normally user-visible)
            let name = builtin_code_name(obj);
            format!("<code {name}>")
        } else if std::ptr::eq(tp, &BUILTIN_FUNCTION_TYPE as *const PyType) {
            // function.py:721 BuiltinFunction.descr_function_repr
            let name = function_get_name(obj);
            format!("<built-in function {name}>")
        } else if std::ptr::eq(tp, &FUNCTION_TYPE as *const PyType) {
            // function.py:268-269 Function.descr_function_repr —
            // FunctionWithFixedCode inherits this and reports as <function>.
            let name = function_get_name(obj);
            format!("<function {name}>")
        } else if unsafe { pyre_object::is_exception(obj) } {
            // `pypy/module/exceptions/interp_exceptions.py:135-147
            // W_BaseException.descr_repr` →
            //   lgt = len(self.args_w)
            //   if lgt == 0: args_repr = "()"
            //   elif lgt == 1: args_repr = "(" + repr(args_w[0]) + ")"
            //   else: args_repr = repr(space.newtuple(args_w))
            //   clsname = self.getclass(space).getname(space)
            //   return clsname + args_repr
            // Note: the 1-arg branch has no trailing comma (line 140-142
            // emits `"(" + utf8 + ")"`).  The multi-arg branch's inner
            // commas come from `repr(tuple)` which never adds a trailing
            // comma either; pyre joins with ", " inside the outer parens
            // to mirror that exactly.
            //
            // Pull the registered class name from `r#type(obj).__name__`
            // (preserves user subclasses like `class MyErr(Exception)`)
            // and read `args_w` from the typed `W_ExceptionObject.args_w`
            // slot — `exc_constructor!` (`builtins.rs`) stamps the tuple
            // there directly so `e.args` identity is preserved across
            // reads.  Falls back to the `message` slot for exceptions
            // produced outside the constructor path (`gateway.rs` raise
            // sites that bypass `exc_constructor!`).
            let class_name = if let Some(cls) = crate::typedef::r#type(obj) {
                pyre_object::w_type_get_name(cls).to_string()
            } else {
                pyre_object::excobject::exc_kind_name(pyre_object::w_exception_get_kind(obj))
                    .to_string()
            };
            let args_obj = unsafe { pyre_object::excobject::w_exception_get_args(obj) };
            let inner = if !args_obj.is_null() && pyre_object::is_tuple(args_obj) {
                let n = pyre_object::w_tuple_len(args_obj);
                if n == 0 {
                    String::new()
                } else if n == 1 {
                    let item = pyre_object::w_tuple_getitem(args_obj, 0).unwrap_or(args_obj);
                    py_repr(item)
                } else {
                    let mut parts = Vec::with_capacity(n);
                    for i in 0..n {
                        if let Some(item) = pyre_object::w_tuple_getitem(args_obj, i as i64) {
                            parts.push(py_repr(item));
                        }
                    }
                    parts.join(", ")
                }
            } else {
                let msg = pyre_object::excobject::w_exception_get_message(obj);
                if msg.is_empty() {
                    String::new()
                } else {
                    format!("'{msg}'")
                }
            };
            format!("{class_name}({inner})")
        } else if std::ptr::eq(tp, &TYPE_TYPE as *const PyType) {
            let name = pyre_object::w_type_get_name(obj);
            format!("<class '{name}'>")
        } else if std::ptr::eq(tp, &pyre_object::UNION_TYPE as *const PyType) {
            // PyPy: UnionType.__repr__ → " | ".join([_repr_item(x) for x in self.__args__])
            let args = pyre_object::w_union_get_args(obj);
            let n = pyre_object::w_tuple_len(args);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(args, i as i64) {
                    if pyre_object::is_none(item) {
                        parts.push("None".to_string());
                    } else if pyre_object::is_type(item) {
                        parts.push(pyre_object::w_type_get_name(item).to_string());
                    } else {
                        parts.push(py_repr(item));
                    }
                }
            }
            parts.join(" | ")
        } else if std::ptr::eq(tp, &MODULE_TYPE as *const PyType) {
            let name = pyre_object::w_module_get_name(obj);
            format!("<module '{name}'>")
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::MAPPING_PROXY_TYPE as *const PyType,
        ) {
            // `pypy/objspace/std/dictproxyobject.py:47 descr_repr` →
            // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`.
            let inner = pyre_object::w_dict_proxy_get_mapping(obj);
            format!("mappingproxy({})", py_repr(inner))
        } else if std::ptr::eq(
            tp,
            &pyre_object::dictviewobject::DICT_KEYS_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictviewobject::DICT_VALUES_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictviewobject::DICT_ITEMS_TYPE as *const PyType,
        ) {
            // `pypy/objspace/std/dictmultiobject.py
            // W_DictMultiViewKeysObject.descr_repr` →
            // `"dict_keys([k1, k2, ...])"` (and the same shape for
            // values / items).  Pyre snapshots the source dict via
            // `dict_view_snapshot` so the rendered list matches what
            // the iter dispatch would produce.
            let kind = pyre_object::dictviewobject::w_dict_view_get_kind(obj);
            let label = match kind {
                pyre_object::dictviewobject::DictViewKind::Keys => "dict_keys",
                pyre_object::dictviewobject::DictViewKind::Values => "dict_values",
                pyre_object::dictviewobject::DictViewKind::Items => "dict_items",
            };
            let snapshot = crate::type_methods::dict_view_snapshot(obj);
            let parts: Vec<String> = snapshot.iter().map(|&item| py_repr(item)).collect();
            format!("{label}([{}])", parts.join(", "))
        } else if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            // Try __repr__ first, then __str__
            if let Some(s) = try_call_dunder(obj, "__repr__") {
                return s;
            }
            if let Some(s) = try_call_dunder(obj, "__str__") {
                return s;
            }
            let w_type = pyre_object::w_instance_get_type(obj);
            let name = pyre_object::w_type_get_name(w_type);
            format!("<{name} object at {obj:?}>")
        } else {
            format!("<{} object at {:?}>", (*tp).name, obj)
        }
    }
}

/// Format for str() — tries __str__ first, then __repr__.
pub unsafe fn py_str(obj: PyObjectRef) -> String {
    unsafe {
        let obj = crate::baseobjspace::unwrap_cell(obj);
        if obj.is_null() {
            return "NULL".to_string();
        }
        let tp = (*obj).ob_type;
        // For strings, return the value directly (no quotes).
        if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            return pyre_object::w_str_get_value(obj).to_string();
        }
        if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            if let Some(s) = try_call_dunder(obj, "__str__") {
                return s;
            }
            if let Some(s) = try_call_dunder(obj, "__repr__") {
                return s;
            }
        }
        // `pypy/module/exceptions/interp_exceptions.py:126-133
        // W_BaseException.descr_str`:
        //
        // ```python
        // def descr_str(self, space):
        //     lgt = len(self.args_w)
        //     if lgt == 0:
        //         return space.newtext('')
        //     elif lgt == 1:
        //         return space.str(self.args_w[0])
        //     else:
        //         return space.str(space.newtuple(self.args_w))
        // ```
        //
        // PyPy reads `self.args_w` on every call so `e.args = (...)`
        // mutations are reflected by subsequent `str(e)` reads.  Pyre
        // previously returned the constructor-time `message` snapshot,
        // which split repr/str apart after the user mutated args.
        if unsafe { pyre_object::is_exception(obj) } {
            // `pypy/module/exceptions/interp_exceptions.py:447-459`
            // `W_UnicodeTranslateError.descr_str`,
            // `:1061-1071` `W_UnicodeDecodeError.descr_str`,
            // `:1175-1191` `W_UnicodeEncodeError.descr_str` — each
            // typedef registers `__str__ = interp2app(descr_str)`,
            // overriding the inherited `W_BaseException.descr_str`.
            // Dispatched on `ExcKind` because Pyre flattens the three
            // PyPy subclasses into the single `W_ExceptionObject`
            // struct.
            let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
            match kind {
                pyre_object::excobject::ExcKind::UnicodeTranslateError => {
                    return unicode_translate_error_str(obj);
                }
                pyre_object::excobject::ExcKind::UnicodeDecodeError => {
                    return unicode_decode_error_str(obj);
                }
                pyre_object::excobject::ExcKind::UnicodeEncodeError => {
                    return unicode_encode_error_str(obj);
                }
                _ => {}
            }
            let args = pyre_object::excobject::w_exception_get_args(obj);
            if args.is_null() {
                return String::new();
            }
            if !pyre_object::is_tuple(args) {
                return py_str(args);
            }
            let n: usize = pyre_object::w_tuple_len(args);
            if n == 0 {
                return String::new();
            }
            if n == 1 {
                let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
                return py_str(first);
            }
            return py_str(args);
        }
        py_repr(obj)
    }
}

/// Format an `int` `%d` position slot from a `W_ExceptionObject`
/// typed Unicode*Error position field.  `descr_init`'s typecheck
/// admits `int` (including subclasses), so a successfully-initialised
/// instance always yields a number here.  After a writer-driven
/// mutation through `readwrite_attrproperty_w`, however, the slot may
/// hold any object — PyPy's appexec-driven `"%d" % w_start` raises
/// `TypeError` on non-int values.  Pyre's `py_str` cannot propagate
/// `PyError` from inside `descr_str`, so the closest behavior is
/// Python's `"%s" % value` (str-coerced) for the failure case: that
/// keeps the original value visible in the formatted message instead
/// of silently substituting `0`.  `Ok(i64)` carries a numeric value
/// (used for `end - 1` arithmetic and the `end == start + 1` shape
/// check); `Err(String)` carries the pre-formatted str-coerced
/// fallback for direct interpolation into the message.
unsafe fn unicode_err_int_slot(stored: PyObjectRef) -> Result<i64, String> {
    unsafe {
        if stored.is_null() || pyre_object::is_none(stored) {
            // Never set / explicit None — PyPy class-default `w_start
            // = None`.  `"%d" % None` raises, but in pyre py_str
            // cannot raise; surface "None" so the bad state is at
            // least visible.
            return Err("None".to_string());
        }
        // `int_w` walks the __int__/__index__ protocol, so int
        // subclasses with stored intval (`class MyInt(int): pass`,
        // `True`/`False`) and any object implementing __index__ all
        // resolve to the numeric value — matching PyPy's
        // `"%d" % value` semantics.
        if let Ok(v) = crate::baseobjspace::int_w(stored) {
            return Ok(v);
        }
        Err(py_str(stored))
    }
}

/// Format an `str` `%s` slot (encoding / reason) from a typed
/// Unicode*Error field.  Mirrors Python's `"%s" % value` which calls
/// `str(value)` on non-str inputs (Python format-string `%s`
/// semantics).  `descr_init`'s `isinstance_str_w` check rejects
/// non-str at construction time; this helper covers the
/// post-construction mutation case (`e.encoding = 42`,
/// `e.reason = None`, etc.) the way PyPy would via `%s`-coerce.
unsafe fn unicode_err_str_slot(stored: PyObjectRef) -> String {
    unsafe {
        if stored.is_null() {
            return String::new();
        }
        if pyre_object::is_str(stored) {
            return pyre_object::w_str_get_value(stored).to_string();
        }
        py_str(stored)
    }
}

/// Single-char `%d`-slot formatter: takes the `(Ok|Err)` from
/// `unicode_err_int_slot` and renders either the `int` or the
/// str-coerced fallback verbatim.
fn unicode_err_int_repr(slot: &Result<i64, String>) -> String {
    match slot {
        Ok(v) => v.to_string(),
        Err(s) => s.clone(),
    }
}

/// `end - 1` for the plural message: matches PyPy's `self.end - 1`.
/// On an int slot, arithmetic; on the str-coerced fallback, the
/// value is embedded verbatim so the message still reflects what the
/// user actually stored.
fn unicode_err_end_minus_one_repr(slot: &Result<i64, String>) -> String {
    match slot {
        Ok(v) => (v - 1).to_string(),
        Err(s) => s.clone(),
    }
}

/// `pypy/module/exceptions/interp_exceptions.py:447-459
/// W_UnicodeTranslateError.descr_str`:
///
/// ```python
/// if self.object is None:
///     return ""
/// if self.end == self.start + 1:
///     badchar = ord(self.object[self.start])
///     if badchar <= 0xff:
///         return "can't translate character '\\x%02x' in position %d: %s"
///     ...
/// return "can't translate characters in position %d-%d: %s"
/// ```
///
/// PyPy's `self.object is None` covers both the never-set state
/// (class-default `w_object = None`) and a writer-driven
/// `e.object = None` mutation through `readwrite_attrproperty_w`.
/// Both shapes resolve to `space.w_None`; pyre stores `PY_NULL` for
/// the never-set case and the runtime `w_none()` singleton for an
/// explicit `None` assignment.  Treat either as the unset signal so
/// `str(e)` mirrors PyPy after `e.object = None`.
///
/// PyPy's appexec format raises `TypeError` on non-int `start`/`end`
/// and surfaces `IndexError` if `self.object[self.start]` is OOR.
/// Pyre's `py_str` cannot propagate `PyError`, so non-int slots are
/// rendered via `"%s"`-style str-coercion (`unicode_err_int_slot`)
/// and an OOR / non-str `w_object` keeps the single-character format
/// shape with a `<?>` placeholder for the indexed character — never
/// silently degrading to the plural-range message when the shape
/// `end == start + 1` says single-char.
unsafe fn unicode_translate_error_str(obj: PyObjectRef) -> String {
    unsafe {
        let w_object = pyre_object::excobject::w_exception_get_object(obj);
        if w_object.is_null() || pyre_object::is_none(w_object) {
            return String::new();
        }
        let start_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_start(obj));
        let end_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_end(obj));
        let reason = unicode_err_str_slot(pyre_object::excobject::w_exception_get_reason(obj));
        let start_repr = unicode_err_int_repr(&start_slot);
        // Shape predicate `self.end == self.start + 1` — true iff both
        // slots are int AND `end == start + 1`.  Any non-int slot
        // makes PyPy's `==` False (different types), so render as the
        // plural shape with str-coerced position values.
        let single_char = matches!((&start_slot, &end_slot), (Ok(s), Ok(e)) if *e == *s + 1);
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let badchar_repr = if pyre_object::is_str(w_object) {
                let text = pyre_object::w_str_get_value(w_object);
                let chars: Vec<char> = text.chars().collect();
                usize::try_from(start)
                    .ok()
                    .and_then(|i| chars.get(i).copied())
                    .map(|ch| {
                        let badchar = ch as u32;
                        if badchar <= 0xff {
                            format!("'\\x{:02x}'", badchar)
                        } else if badchar <= 0xffff {
                            format!("'\\u{:04x}'", badchar)
                        } else {
                            format!("'\\U{:08x}'", badchar)
                        }
                    })
            } else {
                None
            };
            return format!(
                "can't translate character {} in position {}: {}",
                badchar_repr.unwrap_or_else(|| "<?>".to_string()),
                start_repr,
                reason
            );
        }
        format!(
            "can't translate characters in position {}-{}: {}",
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
            reason
        )
    }
}

/// `pypy/module/exceptions/interp_exceptions.py:1061-1071
/// W_UnicodeDecodeError.descr_str`:
///
/// ```python
/// if self.object is None: return ""
/// if self.end == self.start + 1:
///     return "'%s' codec can't decode byte 0x%02x in position %d: %s"%(
///         self.encoding, self.object[self.start], self.start, self.reason)
/// return "'%s' codec can't decode bytes in position %d-%d: %s" % (
///     self.encoding, self.start, self.end - 1, self.reason)
/// ```
///
/// PyPy's appexec lets `%d` raise on non-int `start`/`end` and
/// `self.object[self.start]` raise on out-of-range / non-subscriptable
/// objects.  Pyre's `py_str` cannot propagate `PyError`, so non-int
/// slots fall back to `"%s"`-style str-coercion and an OOR /
/// non-bytes-like `w_object` keeps the single-byte format shape with
/// `0x??` for the byte position — the shape never silently degrades
/// to the plural-range message when `end == start + 1`.
unsafe fn unicode_decode_error_str(obj: PyObjectRef) -> String {
    unsafe {
        let w_object = pyre_object::excobject::w_exception_get_object(obj);
        if w_object.is_null() || pyre_object::is_none(w_object) {
            return String::new();
        }
        let encoding = unicode_err_str_slot(pyre_object::excobject::w_exception_get_encoding(obj));
        let start_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_start(obj));
        let end_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_end(obj));
        let reason = unicode_err_str_slot(pyre_object::excobject::w_exception_get_reason(obj));
        let start_repr = unicode_err_int_repr(&start_slot);
        let single_char = matches!((&start_slot, &end_slot), (Ok(s), Ok(e)) if *e == *s + 1);
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let byte_repr = if pyre_object::is_bytes_like(w_object) {
                let data = pyre_object::bytes_like_data(w_object);
                usize::try_from(start)
                    .ok()
                    .and_then(|i| data.get(i).copied())
                    .map(|byte| format!("0x{:02x}", byte))
            } else {
                None
            };
            return format!(
                "'{}' codec can't decode byte {} in position {}: {}",
                encoding,
                byte_repr.unwrap_or_else(|| "0x??".to_string()),
                start_repr,
                reason
            );
        }
        format!(
            "'{}' codec can't decode bytes in position {}-{}: {}",
            encoding,
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
            reason
        )
    }
}

/// `pypy/module/exceptions/interp_exceptions.py:1175-1191
/// W_UnicodeEncodeError.descr_str` — same single/range split as
/// `W_UnicodeTranslateError` but prefixed with the encoding name.
/// Non-int / non-str / OOR mutations match the parity rules in
/// [`unicode_translate_error_str`] / [`unicode_decode_error_str`].
unsafe fn unicode_encode_error_str(obj: PyObjectRef) -> String {
    unsafe {
        let w_object = pyre_object::excobject::w_exception_get_object(obj);
        if w_object.is_null() || pyre_object::is_none(w_object) {
            return String::new();
        }
        let encoding = unicode_err_str_slot(pyre_object::excobject::w_exception_get_encoding(obj));
        let start_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_start(obj));
        let end_slot = unicode_err_int_slot(pyre_object::excobject::w_exception_get_end(obj));
        let reason = unicode_err_str_slot(pyre_object::excobject::w_exception_get_reason(obj));
        let start_repr = unicode_err_int_repr(&start_slot);
        let single_char = matches!((&start_slot, &end_slot), (Ok(s), Ok(e)) if *e == *s + 1);
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let badchar_repr = if pyre_object::is_str(w_object) {
                let text = pyre_object::w_str_get_value(w_object);
                let chars: Vec<char> = text.chars().collect();
                usize::try_from(start)
                    .ok()
                    .and_then(|i| chars.get(i).copied())
                    .map(|ch| {
                        let badchar = ch as u32;
                        if badchar <= 0xff {
                            format!("'\\x{:02x}'", badchar)
                        } else if badchar <= 0xffff {
                            format!("'\\u{:04x}'", badchar)
                        } else {
                            format!("'\\U{:08x}'", badchar)
                        }
                    })
            } else {
                None
            };
            return format!(
                "'{}' codec can't encode character {} in position {}: {}",
                encoding,
                badchar_repr.unwrap_or_else(|| "<?>".to_string()),
                start_repr,
                reason
            );
        }
        format!(
            "'{}' codec can't encode characters in position {}-{}: {}",
            encoding,
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
            reason
        )
    }
}

/// Display wrapper for PyObjectRef.
pub struct PyDisplay(pub PyObjectRef);

impl fmt::Display for PyDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            write!(f, "NULL")
        } else {
            write!(f, "{}", unsafe { py_str(self.0) })
        }
    }
}
