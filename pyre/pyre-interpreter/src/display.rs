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
        } else if std::ptr::eq(tp, &pyre_object::pyobject::DICT_TYPE as *const PyType) {
            let d = &*(obj as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let mut parts = Vec::with_capacity(entries.len());
            for &(k, v) in entries {
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
            let str_obj = obj as *const pyre_object::strobject::W_StrObject;
            format!("'{}'", &*(*str_obj).value)
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
