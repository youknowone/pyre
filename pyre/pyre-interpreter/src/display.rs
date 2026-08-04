use std::fmt;

use pyre_object::pyobject::{
    BOOL_TYPE, ELLIPSIS_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE, LONG_TYPE, MODULE_TYPE,
    NONE_TYPE, PyObjectRef, PyType, STR_TYPE, TYPE_TYPE,
};
use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::{
    BUILTIN_CODE_TYPE, BUILTIN_FUNCTION_TYPE, FUNCTION_TYPE, METHOD_DESCRIPTOR_TYPE,
    builtin_code_name, function_get_name, function_get_qualname,
};

/// Try to call a dunder method (__repr__, __str__, etc.) on an instance,
/// returning the raw result object when it is a `str`.
pub(crate) unsafe fn try_call_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return Ok(None);
        }
        let Some(method) = crate::baseobjspace::lookup(obj, name) else {
            return Ok(None);
        };
        if method.is_null() {
            return Ok(None);
        }
        // A raising `__repr__`/`__str__` propagates; a non-string return is a
        // TypeError (`object.c slot_tp_repr` / `slot_tp_str`).
        let w_type = crate::typedef::r#type(obj).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
        let result = crate::baseobjspace::get_and_call_function(method, obj, w_type, &[])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

pub(crate) unsafe fn try_call_dunder_obj_above_object(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return Ok(None);
        }
        let Some(w_type) = crate::typedef::r#type(obj) else {
            return Ok(None);
        };
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_type.as_ptr(), name)
        else {
            return Ok(None);
        };
        if method.is_null() || std::ptr::eq(src, crate::typedef::w_object()) {
            return Ok(None);
        }
        let result = crate::baseobjspace::get_and_call_function(method, obj, w_type.as_ptr(), &[])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

/// WTF-8 carrying variant of [`try_call_dunder`]: dispatches `__str__` /
/// `__repr__` on an instance and preserves a surrogate-bearing result
/// instead of folding it through a `&str` (which would panic).
unsafe fn try_call_dunder_wtf8(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        Ok(try_call_dunder_obj(obj, name)?
            .map(|result| pyre_object::w_str_get_wtf8(result).to_wtf8_buf()))
    }
}

/// `TypeError: __repr__ returned non-string (type X)` for a dunder whose
/// override returned a non-`str` (CPython 3.14 `slot_tp_repr`).
unsafe fn dunder_returned_non_string(name: &str, result: PyObjectRef) -> crate::PyError {
    let type_name = match unsafe { crate::typedef::r#type(result) } {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string(),
        None => "object".to_string(),
    };
    crate::PyError::type_error(format!("{name} returned non-string (type {type_name})"))
}

/// `floatobject.py W_FloatObject.descr_repr` — the shortest decimal string
/// that round-trips to `val` (lowercase `nan`/`inf`, signed two-digit
/// exponents, `.0` on integral values). Delegates to the shortest-repr
/// formatter in `rustpython_literal::float`.
pub(crate) fn format_float_repr(val: f64) -> String {
    rustpython_literal::float::to_string(val)
}

/// `repr`-style string escaping: pick the outer quote (prefer `'`, switch
/// to `"` when the value contains `'` but not `"`), escape the backslash,
/// the chosen quote, whitespace and non-printable code points, and render
/// lone surrogates as `\uXXXX`. Delegates to the shared escape engine in
/// `rustpython_literal::escape`.
pub(crate) fn format_wtf8_repr(s: &Wtf8) -> String {
    use rustpython_literal::escape::{Quote, UnicodeEscape};
    let escape = UnicodeEscape::with_preferred_quote(s, Quote::Single);
    let mut out = String::new();
    escape.str_repr().write(&mut out).unwrap();
    out
}

/// `bytearrayobject.py W_BytearrayObject.descr_repr` — `bytearray(b'...')`.
/// The outer quote prefers `'`, flipping to `"` only when the data holds a
/// `'` but no `"`; the body always backslash-escapes `'` and `\` and never
/// escapes `"`, so a `'` survives as `\'` even under a `"` outer quote.
pub(crate) fn bytearray_repr_string(data: &[u8], class_name: &str) -> String {
    let has_single = data.contains(&b'\'');
    let has_double = data.contains(&b'"');
    let quote = if has_single && !has_double {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(data.len() + 14);
    out.push_str(class_name);
    out.push_str("(b");
    out.push(quote as char);
    for &c in data {
        match c {
            b'\'' | b'\\' => {
                out.push('\\');
                out.push(c as char);
            }
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(c as char),
            _ => out.push_str(&format!("\\x{c:02x}")),
        }
    }
    out.push(quote as char);
    out.push(')');
    out
}

thread_local! {
    /// Object pointers currently mid-`py_repr` on this thread.  Guards the
    /// recursive container branches against unbounded recursion on a
    /// reference cycle (a list holding itself, a dict valued by itself).
    /// Mirrors the per-thread reprlist behind `Py_ReprEnter`/`Py_ReprLeave`.
    static REPR_ACTIVE: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record `obj` as mid-repr on this thread, or report `false` when it already
/// is (`Py_ReprEnter`).  Writes the runtime-mutable `REPR_ACTIVE` thread-local,
/// not a build-time constant, so the JIT residualises the call instead of
/// tracing into it (`@dont_look_inside`, the `eval::set_in_flight_exception`
/// shape), the [`repr_leave`] twin.
#[majit_macros::dont_look_inside]
pub(crate) fn repr_enter(obj: PyObjectRef) -> bool {
    let key = obj as usize;
    REPR_ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        if active.contains(&key) {
            false
        } else {
            active.push(key);
            true
        }
    })
}

/// Drop `obj` from the mid-repr set (`Py_ReprLeave`) — see [`repr_enter`].
#[majit_macros::dont_look_inside]
pub(crate) fn repr_leave(obj: PyObjectRef) {
    let key = obj as usize;
    REPR_ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        if let Some(pos) = active.iter().rposition(|&k| k == key) {
            active.remove(pos);
        }
    });
}

/// RAII cycle guard.  `enter` returns `None` when `obj` is already being
/// repr'd on this thread — the caller emits the `...` placeholder — and
/// otherwise records `obj`, removing it again when the guard drops.
pub(crate) struct ReprGuard(PyObjectRef);

impl ReprGuard {
    pub(crate) fn enter(obj: PyObjectRef) -> Option<ReprGuard> {
        repr_enter(obj).then_some(ReprGuard(obj))
    }
}

impl Drop for ReprGuard {
    fn drop(&mut self) {
        repr_leave(self.0);
    }
}

/// `dictmultiobject.py:130-150 descr_repr` — `{k: v, ...}`.  Iterates
/// `w_dict_items` (which routes through `is_module_dict`), guarded against
/// self-recursion.  Shared by the `py_repr` dict fast path and the dict
/// type's `__repr__` method (so dict-subclass instances and `super().
/// __repr__()` format their backing the same way).
///
/// # Safety
/// `obj` must be a real `W_DictObject` (caller resolves any subclass
/// backing via `resolve_dict_backing` first).
pub unsafe fn dict_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("{...}".to_string()));
    };
    let entries = pyre_object::w_dict_items(obj);
    let mut out = Wtf8Buf::new();
    out.push_str("{");
    for (i, (k, v)) in entries.into_iter().enumerate() {
        // `dictmultiobject.py:388` joins the pairs by position, so a key or
        // value whose `__repr__` answers `""` still gets its separator.
        if i != 0 {
            out.push_str(", ");
        }
        out.push_wtf8(&py_repr_wtf8(k)?);
        out.push_str(": ");
        out.push_wtf8(&py_repr_wtf8(v)?);
    }
    out.push_str("}");
    Ok(out)
}

/// `listobject.py W_ListObject.descr_repr`, factored out so the TypeDef slot
/// and the native `py_repr` fast path use the same recursion guard and item
/// walk. Calling the base descriptor on a subclass must not redispatch its
/// overriding `__repr__`.
pub unsafe fn list_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("[...]".to_string()));
    };
    let n = pyre_object::w_list_len(obj);
    let mut out = Wtf8Buf::new();
    out.push_str("[");
    for i in 0..n {
        // `listobject.py:217-221` stops rather than skipping when an item is
        // gone, since an item's `__repr__` may have shortened the list.
        let Some(item) = pyre_object::w_list_getitem(obj, i as i64) else {
            break;
        };
        // The separator goes by position, not by how much has been written:
        // an item whose `__repr__` answers `""` still takes a slot.
        if i != 0 {
            out.push_str(", ");
        }
        out.push_wtf8(&py_repr_wtf8(item)?);
    }
    out.push_str("]");
    Ok(out)
}

/// `tupleobject.py W_AbstractTupleObject.descr_repr`. This is the base slot
/// body, so it formats tuple storage directly even when invoked through
/// `super().__repr__()` on a subtype.
pub unsafe fn tuple_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("(...)".to_string()));
    };
    let n = pyre_object::w_tuple_len(obj);
    let mut out = Wtf8Buf::new();
    out.push_str("(");
    for i in 0..n {
        if let Some(item) = pyre_object::w_tuple_getitem(obj, i as i64) {
            // `tupleobject.py:114` joins by position, so an item whose
            // `__repr__` answers `""` still gets its separator.
            if i != 0 {
                out.push_str(", ");
            }
            out.push_wtf8(&py_repr_wtf8(item)?);
        }
    }
    if n == 1 {
        out.push_str(",");
    }
    out.push_str(")");
    Ok(out)
}

/// Format a PyObjectRef for debug display.
///
/// # Safety
/// `obj` must be a valid pointer to a known Python object type.
/// Format an `int`/`long`/`float`/`bool` storage object with its builtin
/// `repr` (which equals its `str` for these types).  Returns `None` for
/// any other storage type.  Shared by `py_repr`'s leaf path and `py_str`'s
/// fallback so a builtin leaf subclass that overrides only `__repr__`
/// still `str()`s via the inherited builtin `tp_str`.
unsafe fn builtin_leaf_repr_string(
    obj: PyObjectRef,
    tp: *const PyType,
) -> Result<Option<String>, crate::PyError> {
    unsafe {
        Ok(if std::ptr::eq(tp, &INT_TYPE as *const PyType) {
            // A machine int is at most 19 digits, below the 640 floor
            // `sys.set_int_max_str_digits` accepts, so it never trips the
            // conversion-length limit the `long` arm has to check.
            Some(format!("{}", pyre_object::intobject::w_int_get_value(obj)))
        } else if std::ptr::eq(tp, &FLOAT_TYPE as *const PyType) {
            let float_obj = obj as *const pyre_object::floatobject::W_FloatObject;
            Some(format_float_repr((*float_obj).floatval))
        } else if std::ptr::eq(tp, &pyre_object::COMPLEX_TYPE as *const PyType) {
            Some(crate::typedef::complex_repr_string(
                pyre_object::w_complex_get_real(obj),
                pyre_object::w_complex_get_imag(obj),
            ))
        } else if std::ptr::eq(tp, &LONG_TYPE as *const PyType) {
            // `long_to_decimal_string` enforces the
            // `sys.set_int_max_str_digits` conversion-length limit, so the
            // guard belongs to the conversion itself rather than to the
            // `__repr__` descriptor that is only one of its callers.
            Some(crate::builtins::int_to_decimal_string(obj)?)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const pyre_object::boolobject::W_BoolObject;
            Some(
                if (*bool_obj).intval != 0 {
                    "True"
                } else {
                    "False"
                }
                .to_string(),
            )
        } else {
            None
        })
    }
}

/// Dispatch a user-defined `__repr__`/`__str__` override for a builtin leaf
/// subclass instance.  `int`/`float`/`str`/... keep `ob_type` at the
/// canonical storage type and carry the Python class in `w_class`, so the
/// `ob_type`-keyed formatters ignore a subclass override.  Returns `Some`
/// only when the dunder resolves above `object` (whose inherited default
/// must fall through to the builtin formatting instead of re-entering).
/// `builtin_subclass_dunder` returning the raw `str` result object so a
/// WTF-8-preserving caller (`py_str_wtf8`) can read a lone-surrogate result
/// via `w_str_get_wtf8` instead of the panicking `w_str_get_value`.
pub(crate) unsafe fn builtin_subclass_dunder_obj(
    obj: PyObjectRef,
    tp: *const PyType,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let is_leaf = std::ptr::eq(tp, &INT_TYPE as *const PyType)
            || std::ptr::eq(tp, &LONG_TYPE as *const PyType)
            || std::ptr::eq(tp, &FLOAT_TYPE as *const PyType)
            || std::ptr::eq(tp, &pyre_object::COMPLEX_TYPE as *const PyType)
            || std::ptr::eq(tp, &BOOL_TYPE as *const PyType)
            || std::ptr::eq(tp, &STR_TYPE as *const PyType)
            || std::ptr::eq(tp, &pyre_object::LIST_TYPE as *const PyType)
            || pyre_object::is_tuple(obj)
            || std::ptr::eq(
                tp,
                &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const PyType,
            );
        if !is_leaf {
            return Ok(None);
        }
        // An exact builtin is not a subclass instance: its `w_class` is the
        // canonical type object, so the MRO walk below would resolve the
        // builtin's own dunder and re-dispatch it through a full call rather
        // than the native formatting this function exists to defer to.
        if pyre_object::is_exact_builtin_instance(obj) {
            return Ok(None);
        }
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        // Only a subclass can redirect the dunder: it keeps the builtin
        // `ob_type` and retags `w_class` (`typedef::subclass_to_tag`), which
        // is exactly what `is_exact_builtin_instance` tests. An exact
        // instance resolves the dunder to the builtin the caller is about to
        // run natively, and the builtin types are immutable, so the two MRO
        // walks and the descriptor call below can only reproduce it.
        //
        // `long` is the one leaf where the descriptor does more than the leaf
        // formatter: `longobject.py descr_repr` also enforces
        // `sys.set_int_max_str_digits`, and that check sits in the descriptor
        // rather than in the conversion, so an exact `long` keeps going
        // through it. A machine `int` cannot reach any settable limit — 19
        // digits against a floor of 640.
        if !std::ptr::eq(tp, &LONG_TYPE as *const PyType)
            && pyre_object::is_exact_builtin_instance(obj)
        {
            return Ok(None);
        }
        let Some((src, found)) = crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        // `object`'s inherited default is not a leaf override — fall through
        // so the builtin formatting runs (and `object.__repr__` does not
        // re-enter through this path). An explicit
        // `Subclass.__str__ = object.__str__` is different: its owner is the
        // subclass and the descriptor must run, allowing object.__str__ to
        // delegate to the subclass's __repr__.
        let w_object = crate::typedef::w_object();
        if std::ptr::eq(src, w_object) {
            return Ok(None);
        }
        // A raising override propagates; a non-string return is a TypeError.
        let r = crate::builtins::call_and_check(found, &[obj])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// Resolve a class object's special method on its metaclass.
///
/// PyPy: `space.lookup(w_obj, name)` uses `space.type(w_obj)`, so a class
/// receives an EnumType/user-metaclass override before the native `type`
/// implementation. The builtin `type`/`object` definitions are terminals and
/// deliberately left to the native formatting path to avoid re-entry.
pub(crate) unsafe fn type_metaclass_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_type(obj) {
            return Ok(None);
        }
        let Some(metaclass) = crate::typedef::r#type(obj) else {
            return Ok(None);
        };
        if !pyre_object::is_type(metaclass.as_ptr()) {
            return Ok(None);
        }
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(metaclass.as_ptr(), name)
        else {
            return Ok(None);
        };
        if std::ptr::eq(src, crate::typedef::w_type())
            || std::ptr::eq(src, crate::typedef::w_object())
        {
            return Ok(None);
        }
        let result = crate::builtins::call_and_check(method, &[obj])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

/// Dispatch a user-defined `__str__` / `__repr__` override on an
/// exception subclass.  The builtin `descr_str` / `descr_repr` are
/// handled natively in `py_str` / `py_repr`, but a Python subclass
/// (`class E(Exception): def __str__(self): ...`) installs its own
/// method that must win, the same way `str(e)` dispatches it in PyPy.
/// Returns `None` only when `__str__`/`__repr__` resolves to the builtin
/// `BaseException` / `object` registration (no override). A raising override
/// propagates, and a non-string result raises `TypeError`.
/// `exc_user_dunder` variant returning the raw `str` result object so a
/// WTF-8-preserving caller (`exception_descr_str_wtf8`) can read the
/// lone-surrogate-carrying bytes directly. A raising override propagates and
/// a non-string result raises `TypeError`, matching descroperation.py's
/// `space.str` / `space.repr` result check.
pub(crate) unsafe fn exc_user_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        // `object`'s and `BaseException`'s registrations are the two the
        // native formatting stands in for, and so are the `descr_str`
        // builtins the exception classes install on top of them — calling
        // any of those back from here would recurse.  A builtin that the
        // native path does *not* implement, such as
        // `BaseExceptionGroup.__str__`, still has to be dispatched.
        if method.is_null() || std::ptr::eq(src, crate::typedef::w_object()) {
            return Ok(None);
        }
        if crate::builtins::is_native_exception_dunder(method) {
            return Ok(None);
        }
        if let Some(base) = crate::builtins::lookup_exc_class("BaseException") {
            if std::ptr::eq(src, base) {
                return Ok(None);
            }
        }
        let r = crate::builtins::call_and_check(method, &[obj])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// Dispatch a user-defined `__repr__` / `__str__` override on a
/// `types.ModuleType` subclass. A module instance keeps `ob_type` at the
/// canonical module type and carries its Python class in `w_class`, so the
/// `ob_type`-keyed formatting in `py_repr` / `py_str` would otherwise bypass a
/// subclass override. Returns `None` when the method resolves to the base
/// `module` registration or `object` (no override); a raising override
/// propagates and a non-string result raises `TypeError`.
unsafe fn module_user_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        let module_class = crate::typedef::gettypeobject(&MODULE_TYPE);
        if std::ptr::eq(w_class, module_class) {
            return Ok(None);
        }
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        if method.is_null()
            || std::ptr::eq(src, module_class)
            || std::ptr::eq(src, crate::typedef::w_object())
        {
            return Ok(None);
        }
        let r = crate::builtins::call_and_check(method, &[obj])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// `space.repr` — the whole type dispatch, answering the encoded bytes.
///
/// `listobject.py:206-225 _listrepr_inner` assembles a container's repr in a
/// `rutf8.Utf8StringBuilder` from each item's `space.utf8_len_w(space.repr(...))`,
/// so a lone surrogate an item wrote survives being nested. A `Wtf8Buf` is the
/// buffer that can hold the same thing here; a Rust `String` cannot, which is
/// why [`py_repr`] is a lossy view of this rather than the other way round.
pub unsafe fn py_repr_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    // A tagged immediate must be formatted before `ob_type` touches it as a
    // pointer; `repr` of a plain `int` is its
    // decimal value. Gated on `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Ok(Wtf8Buf::from_string(format!(
            "{}",
            pyre_object::tagged_int::untag_int(obj)
        )));
    }
    // The recursive container branches below (dict/list/tuple/set/deque/
    // slice/range) re-enter `py_repr_wtf8` on each element in native Rust with
    // no Python frame push, so a deeply nested structure blows the C stack
    // before any frame-level check fires. Guard the stack here so
    // `repr(deeply_nested)` raises RecursionError instead of overflowing.
    crate::stack_check::stack_check()?;
    if obj.is_null() {
        return Ok(Wtf8Buf::from_string("NULL".to_string()));
    }
    unsafe {
        let tp = (*obj).ob_type;
        // A builtin leaf subclass keeps `ob_type` at the canonical storage
        // type but carries the Python class in `w_class`; dispatch its
        // `__repr__` override before the `ob_type`-keyed formatting below.
        if let Some(r) = builtin_subclass_dunder_obj(obj, tp, "__repr__")? {
            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
        }
        // A class object is an instance of its metaclass.  PyPy's
        // `space.repr` therefore resolves `__repr__` on that metaclass before
        // `type`'s native `<class ...>` representation.  This is observable
        // for EnumType and any user metaclass defining `__repr__`.
        if let Some(result) = type_metaclass_dunder_obj(obj, "__repr__")? {
            return Ok(pyre_object::w_str_get_wtf8(result).to_wtf8_buf());
        }
        let formatted = if let Some(s) = builtin_leaf_repr_string(obj, tp)? {
            s
        } else if pyre_object::interp_array::is_array(obj) {
            return crate::module::array::array_repr_wtf8(obj);
        } else if std::ptr::eq(tp, &pyre_object::pyobject::LIST_TYPE as *const PyType) {
            return list_repr(obj);
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
                // structseq instances are tuple subclasses with ob_type ==
                // TUPLE_TYPE, so reach for a subclass __repr__ via the MRO.
                // `tuple` itself installs no `__repr__` dict entry (it is
                // handled natively below), so a plain tuple subclass
                // resolves `__repr__` to `object` — fall through to the
                // tuple formatting in that case rather than printing the
                // generic `<object at ...>`.
                if let Some((src, method)) =
                    crate::baseobjspace::lookup_where_with_method_cache(w_class, "__repr__")
                {
                    if !std::ptr::eq(src, crate::typedef::w_object()) && !method.is_null() {
                        // A raising override propagates; a non-string return is
                        // a TypeError like every other `__repr__` override.
                        let r = crate::builtins::call_and_check(method, &[obj])?;
                        if pyre_object::is_str(r) {
                            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
                        }
                        return Err(dunder_returned_non_string("__repr__", r));
                    }
                }
            }
            return tuple_repr(obj);
        } else if unsafe { pyre_object::is_dict(obj) } {
            return unsafe { dict_repr(obj) };
        } else if pyre_object::sliceobject::is_slice(obj) {
            // `pypy/objspace/std/sliceobject.py descr_repr` —
            // `slice(%r, %r, %r)`.
            let mut out = Wtf8Buf::new();
            out.push_str("slice(");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_start(
                obj,
            ))?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_stop(
                obj,
            ))?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_step(
                obj,
            ))?);
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::is_bytes_like(obj) {
            // `bytesobject.py W_BytesObject.descr_repr` — ASCII-printable
            // bytes pass through, control/high bytes use `\xNN`, and the
            // outer quote prefers `'`, flipping to `"` when the data holds a
            // `'` but no `"`.
            let data = pyre_object::bytes_like_data(obj);
            if pyre_object::bytearrayobject::is_bytearray(obj) {
                // `bytearrayobject.py W_BytearrayObject.descr_repr` differs
                // from the bytes form: it chooses the same outer quote but
                // always backslash-escapes an inner `'` (never `"`), so the
                // shared bytes escaper cannot express it.
                bytearray_repr_string(data, "bytearray")
            } else {
                let escape = rustpython_literal::escape::AsciiEscape::new_repr(data);
                let mut body = String::new();
                escape.bytes_repr().write(&mut body).unwrap();
                body
            }
        } else if pyre_object::is_set_or_frozenset(obj) {
            // `pypy/objspace/std/setobject.py W_BaseSetObject.descr_repr`
            // → `'%s({%s})' % (typename, items_repr_joined)` for
            // frozenset and `'{%s}' % items_repr_joined` for set.  Empty
            // set keeps the `set()` constructor form.
            let is_frozen = pyre_object::is_frozenset(obj);
            let is_exact_set = pyre_object::is_exact_type(obj, &pyre_object::setobject::SET_TYPE);
            let class_name = crate::typedef::r#type(obj)
                .map(|w_type| pyre_object::w_type_get_name(w_type.as_ptr()))
                .unwrap_or(if is_frozen { "frozenset" } else { "set" });
            let Some(_guard) = ReprGuard::enter(obj) else {
                return Ok(Wtf8Buf::from_string(format!("{class_name}(...)")));
            };
            let items = pyre_object::w_set_items(obj);
            let mut out = Wtf8Buf::new();
            if items.is_empty() {
                out.push_str(class_name);
                out.push_str("()");
                return Ok(out);
            }
            if !is_exact_set {
                out.push_str(class_name);
                out.push_str("(");
            }
            out.push_str("{");
            for (i, &item) in items.iter().enumerate() {
                if i != 0 {
                    out.push_str(", ");
                }
                out.push_wtf8(&py_repr_wtf8(item)?);
            }
            out.push_str("}");
            if !is_exact_set {
                out.push_str(")");
            }
            return Ok(out);
        } else if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            format_wtf8_repr(pyre_object::w_str_get_wtf8(obj))
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
        } else if std::ptr::eq(tp, &crate::function::SLOT_WRAPPER_TYPE as *const PyType) {
            let name = function_get_name(obj);
            let owner = crate::function::fget_func_objclass(obj)?;
            let owner_name = pyre_object::w_type_get_name(owner);
            format!("<slot wrapper '{name}' of '{owner_name}' objects>")
        } else if std::ptr::eq(
            tp,
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const PyType,
        ) {
            let name = function_get_name(obj);
            let owner = crate::function::fget_func_objclass(obj)?;
            let owner_name = pyre_object::w_type_get_name(owner);
            format!("<method '{name}' of '{owner_name}' objects>")
        } else if std::ptr::eq(tp, &BUILTIN_FUNCTION_TYPE as *const PyType) {
            // function.py:721 BuiltinFunction.descr_function_repr.  Same text
            // the `__repr__` this type registers in `typedef.rs` produces;
            // this native arm is the one `repr()` actually reaches.
            let name = function_get_name(obj);
            let w_self = crate::function::function_get_self_or_none(obj);
            crate::function::builtin_function_repr_text(name, w_self)
        } else if std::ptr::eq(tp, &FUNCTION_TYPE as *const PyType) {
            // function.py:283 Function.descr_function_repr —
            // `self.getrepr(space, 'function %s' % self.qualname)`, and
            // `baseobjspace.py:115 getrepr` appends ` at 0x<addr>`.  Exact
            // builtin values take this fast path instead of dispatching
            // through the `__repr__` the type registers in `typedef.rs`, so it
            // must produce the same address-bearing text.
            let name = function_get_qualname(obj);
            format!("<function {name} at {obj:p}>")
        } else if unsafe { pyre_object::is_exception(obj) } {
            // A user subclass that overrides `__repr__` shadows the builtin
            // `W_BaseException.descr_repr`; dispatch it before the native
            // formatting below.
            if let Some(r) = exc_user_dunder_obj(obj, "__repr__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            }
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
            // and read `args_w` from the typed `W_BaseException.args_w`
            // slot — `exc_constructor!` (`builtins.rs`) stamps the tuple
            // there directly so `e.args` identity is preserved across
            // reads.  Falls back to the `message` slot for exceptions
            // produced outside the constructor path (`gateway.rs` raise
            // sites that bypass `exc_constructor!`).
            let class_name = if let Some(cls) = crate::typedef::r#type(obj) {
                pyre_object::w_type_get_name(cls.as_ptr()).to_string()
            } else {
                pyre_object::interp_exceptions::exc_kind_name(pyre_object::w_exception_get_kind(
                    obj,
                ))
                .to_string()
            };
            let args_obj = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
            let mut inner = Wtf8Buf::new();
            if !args_obj.is_null() && pyre_object::is_tuple(args_obj) {
                let n = pyre_object::w_tuple_len(args_obj);
                if n == 1 {
                    let item = pyre_object::w_tuple_getitem(args_obj, 0).unwrap_or(args_obj);
                    inner.push_wtf8(&py_repr_wtf8(item)?);
                } else {
                    for i in 0..n {
                        if let Some(item) = pyre_object::w_tuple_getitem(args_obj, i as i64) {
                            // `interp_exceptions.py:135-147` spells the args
                            // with `repr(tuple(args))`, which separates by
                            // position — an argument whose `__repr__` answers
                            // `""` still takes a slot.
                            if i != 0 {
                                inner.push_str(", ");
                            }
                            inner.push_wtf8(&py_repr_wtf8(item)?);
                        }
                    }
                }
            }
            let mut out = Wtf8Buf::new();
            out.push_str(&class_name);
            out.push_str("(");
            out.push_wtf8(&inner);
            out.push_str(")");
            return Ok(out);
        } else if std::ptr::eq(tp, &TYPE_TYPE as *const PyType) {
            let name = crate::baseobjspace::type_repr_qualified_name(obj);
            format!("<class '{name}'>")
        } else if std::ptr::eq(tp, &pyre_object::UNION_TYPE as *const PyType) {
            // PyPy: UnionType.__repr__ → " | ".join([_repr_item(x) for x in self.__args__])
            let args = pyre_object::w_union_get_args(obj);
            let n = pyre_object::w_tuple_len(args);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(args, i as i64) {
                    // `_repr_item_union` (`_pypy_generic_alias.py:141`) —
                    // `type(None)` renders as `None`; a bare `None` may
                    // still reach here from direct construction paths.
                    if pyre_object::is_none(item)
                        || std::ptr::eq(
                            item,
                            crate::typedef::gettypeobject(&pyre_object::NONE_TYPE),
                        )
                    {
                        parts.push("None".to_string());
                    } else {
                        parts.push(crate::_pypy_generic_alias::repr_item(item)?);
                    }
                }
            }
            parts.join(" | ")
        } else if std::ptr::eq(tp, &pyre_object::GENERIC_ALIAS_TYPE as *const PyType) {
            // GenericAlias.__repr__ (`_pypy_generic_alias.py:57`).
            return Ok(Wtf8Buf::from_string(crate::_pypy_generic_alias::repr(obj)?));
        } else if std::ptr::eq(tp, &MODULE_TYPE as *const PyType) {
            // A `types.ModuleType` subclass carries its class in `w_class`; a
            // subclass `__repr__` override wins over the native module
            // formatting.
            if let Some(r) = module_user_dunder_obj(obj, "__repr__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            } else {
                crate::typedef::module_repr_string(obj)?
            }
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::MAPPING_PROXY_TYPE as *const PyType,
        ) {
            // `pypy/objspace/std/dictproxyobject.py:47 descr_repr` →
            // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`.
            let inner = pyre_object::w_dict_proxy_get_mapping(obj);
            let mut out = Wtf8Buf::new();
            out.push_str("mappingproxy(");
            out.push_wtf8(&py_repr_wtf8(inner)?);
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::typedef::is_getset_property(obj) {
            // CPython 3.14 `PyGetSetDescr_Type.tp_repr`.
            crate::typedef::getset_descriptor_repr(obj)
        } else if pyre_object::is_member(obj) {
            // CPython 3.14 `PyMemberDescr_Type.tp_repr = member_repr`.
            // Member descriptors are native-layout objects with no `w_class`,
            // so the generic builtin-dunder fallback below cannot discover
            // their registered __repr__ method.
            crate::typedef::member_descriptor_repr(obj)
        } else if std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_KEYS_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_VALUES_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_ITEMS_TYPE as *const PyType,
        ) {
            // `dictmultiobject.py viewrepr`: the view itself participates in
            // the shared identity recursion set and emits `...` on re-entry.
            // This is distinct from the owning dict's `{...}` placeholder: a
            // dict may contain one of its own values/items views.
            let Some(_guard) = ReprGuard::enter(obj) else {
                return Ok(Wtf8Buf::from_string("...".to_string()));
            };
            let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj);
            let label = match kind {
                pyre_object::dictmultiobject::DictViewKind::Keys => "dict_keys",
                pyre_object::dictmultiobject::DictViewKind::Values => "dict_values",
                pyre_object::dictmultiobject::DictViewKind::Items => "dict_items",
            };
            let snapshot = crate::type_methods::dict_view_snapshot(obj);
            let mut out = Wtf8Buf::new();
            out.push_str(label);
            out.push_str("([");
            for (i, &item) in snapshot.iter().enumerate() {
                if i != 0 {
                    out.push_str(", ");
                }
                out.push_wtf8(&py_repr_wtf8(item)?);
            }
            out.push_str("])");
            return Ok(out);
        } else if pyre_object::is_w_range(obj) {
            // `functional.py W_Range.descr_repr` —
            // `range(start, stop)`, with the step appended only when
            // it is not 1.  Bounds may be bignum, so render each wrapped
            // int rather than a machine word.
            let (start, stop, step) = pyre_object::w_range_fields(obj);
            let step_is_one =
                pyre_object::range_obj_to_bigint(step) == pyre_object::rbigint::RBigInt::from(1);
            let mut out = Wtf8Buf::new();
            out.push_str("range(");
            out.push_wtf8(&py_repr_wtf8(start)?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(stop)?);
            if !step_is_one {
                out.push_str(", ");
                out.push_wtf8(&py_repr_wtf8(step)?);
            }
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::interp_sre::is_sre_pattern(obj) {
            // `pypy/module/_sre/interp_sre.py:153 W_SRE_Pattern.repr_w`.
            crate::module::_sre::interp_sre::sre_pattern_repr_str(obj)?
        } else if pyre_object::interp_sre::is_sre_match(obj) {
            // `pypy/module/_sre/interp_sre.py:684 W_SRE_Match.repr_w`.
            crate::module::_sre::interp_sre::sre_match_repr_str(obj)?
        } else if pyre_object::memoryview::is_w_memoryview(obj) {
            // `memoryobject.py descr_repr` — `<memory at 0x...>`, or
            // `<released memory at 0x...>` once the view is released.
            let label = if pyre_object::memoryview::w_memoryview_released(obj) {
                "released memory"
            } else {
                "memory"
            };
            format!("<{label} at {obj:?}>")
        } else if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            // Try __repr__ first, then __str__
            if let Some(w) = try_call_dunder_wtf8(obj, "__repr__")? {
                return Ok(w);
            }
            if let Some(w) = try_call_dunder_wtf8(obj, "__str__")? {
                return Ok(w);
            }
            let name = crate::baseobjspace::getfulltypename(obj);
            format!("<{name} object at {obj:?}>")
        } else {
            // A builtin type carrying its own `__repr__` dict entry (e.g.
            // `_struct.Struct`) — dispatch it before the generic
            // `<name object at 0x...>` fallback.  Mirrors the tuple-subclass
            // path above.
            let w_class = (*obj).w_class;
            if !w_class.is_null() {
                if let Some((src, method)) =
                    crate::baseobjspace::lookup_where_with_method_cache(w_class, "__repr__")
                {
                    if !std::ptr::eq(src, crate::typedef::w_object()) && !method.is_null() {
                        let r = crate::builtins::call_and_check(method, &[obj])?;
                        if pyre_object::is_str(r) {
                            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
                        }
                        return Err(dunder_returned_non_string("__repr__", r));
                    }
                }
            }
            let name = crate::baseobjspace::getfulltypename(obj);
            format!("<{name} object at {obj:?}>")
        };
        Ok(Wtf8Buf::from_string(formatted))
    }
}

pub unsafe fn py_repr(obj: PyObjectRef) -> Result<String, crate::PyError> {
    Ok(unsafe { py_repr_wtf8(obj) }?.to_string_lossy().into_owned())
}

/// Format for str() — tries __str__ first, then __repr__.
pub unsafe fn py_str_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        // `str` of a tagged `int` immediate is its decimal value; format
        // it before `ob_type` deref. Gated on
        // `CAN_BE_TAGGED` (default false).
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
            return Ok(Wtf8Buf::from_string(format!(
                "{}",
                pyre_object::tagged_int::untag_int(obj)
            )));
        }
        if obj.is_null() {
            return Ok(Wtf8Buf::from_string("NULL".to_string()));
        }
        let tp = (*obj).ob_type;
        // For strings, return the value directly (no quotes).
        if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            if let Some(r) = builtin_subclass_dunder_obj(obj, tp, "__str__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            }
            return Ok(pyre_object::w_str_get_wtf8(obj).to_wtf8_buf());
        }
        if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            if let Some(w) = try_call_dunder_wtf8(obj, "__str__")? {
                return Ok(w);
            }
            if let Some(w) = try_call_dunder_wtf8(obj, "__repr__")? {
                return Ok(w);
            }
        }
        if unsafe { pyre_object::is_exception(obj) } {
            if let Some(w) = exception_descr_str_wtf8(obj)? {
                return Ok(w);
            }
            if let Some(w) = exception_kind_str_wtf8(obj)? {
                return Ok(w);
            }
            // A user subclass that overrides `__str__` shadows the builtin
            // `W_BaseException.descr_str`; dispatch it before the generic
            // args formatting below.  The kind arms above already handled
            // the Unicode / OSError / KeyError `__str__` overrides, so a
            // non-overridden exception here resolves `__str__` to the
            // BaseException builtin and falls through unchanged.
            return base_exception_str_wtf8(obj);
        }
        // `int`/`float`/... define no `tp_str`, so `str()` falls back to
        // `repr()` (a `__str__` override wins, otherwise the `__repr__`
        // override or builtin formatting from `py_repr`).  `str` itself
        // has its own `tp_str` and is handled by the `STR_TYPE` branch
        // above, so this fallthrough never reaches a bare-`str` subclass.
        if let Some(r) = builtin_subclass_dunder_obj(obj, tp, "__str__")? {
            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
        }
        // A `types.ModuleType` subclass `__str__` override wins; without one,
        // `str` falls back to `__repr__` through `py_repr`.
        if pyre_object::is_module(obj) {
            if let Some(r) = module_user_dunder_obj(obj, "__str__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            }
        }
        py_repr_wtf8(obj)
    }
}

pub unsafe fn py_str(obj: PyObjectRef) -> Result<String, crate::PyError> {
    Ok(unsafe { py_str_wtf8(obj) }?.to_string_lossy().into_owned())
}

/// `pypy/module/exceptions/interp_exceptions.py:126-133
/// W_BaseException.descr_str`:
///
/// ```python
/// def descr_str(self, space):
///     lgt = len(self.args_w)
///     if lgt == 0:
///         return space.newtext('')
///     elif lgt == 1:
///         return space.str(self.args_w[0])
///     else:
///         return space.str(space.newtuple(self.args_w))
/// ```
///
/// PyPy reads `self.args_w` on every call so `e.args = (...)` mutations are
/// reflected by subsequent `str(e)` reads.
///
/// # Safety
/// `obj` must be a live `W_BaseException`.
pub(crate) unsafe fn base_exception_str(obj: PyObjectRef) -> Result<String, crate::PyError> {
    Ok(unsafe { base_exception_str_wtf8(obj) }?
        .to_string_lossy()
        .into_owned())
}

pub(crate) unsafe fn base_exception_str_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
        if args.is_null() {
            return Ok(Wtf8Buf::new());
        }
        if !pyre_object::is_tuple(args) {
            return py_str_wtf8(args);
        }
        let n: usize = pyre_object::w_tuple_len(args);
        if n == 0 {
            return Ok(Wtf8Buf::new());
        }
        if n == 1 {
            let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
            return py_str_wtf8(first);
        }
        py_str_wtf8(args)
    }
}

/// The `descr_str` overrides the builtin exception classes register on top of
/// `W_BaseException.descr_str`, dispatched on the instance's `ExcKind` because
/// pyre flattens PyPy's subclasses into the single `W_BaseException` struct.
/// `None` means the instance's class inherits the base `descr_str`.
///
/// # Safety
/// `obj` must be a live `W_BaseException`.
pub(crate) unsafe fn exception_kind_str(
    obj: PyObjectRef,
) -> Result<Option<String>, crate::PyError> {
    Ok(unsafe { exception_kind_str_wtf8(obj) }?.map(|w| w.to_string_lossy().into_owned()))
}

pub(crate) unsafe fn exception_kind_str_wtf8(
    obj: PyObjectRef,
) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        // `pypy/module/exceptions/interp_exceptions.py:447-459`
        // `W_UnicodeTranslateError.descr_str`,
        // `:1061-1071` `W_UnicodeDecodeError.descr_str`,
        // `:1175-1191` `W_UnicodeEncodeError.descr_str` — each
        // typedef registers `__str__ = interp2app(descr_str)`,
        // overriding the inherited `W_BaseException.descr_str`.
        // Dispatched on `ExcKind` because Pyre flattens the three
        // PyPy subclasses into the single `W_BaseException`
        // struct.
        let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
        match kind {
            pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError => {
                return unicode_translate_error_str(obj).map(Some);
            }
            pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError => {
                return unicode_decode_error_str(obj).map(Some);
            }
            pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError => {
                return unicode_encode_error_str(obj).map(Some);
            }
            // `interp_exceptions.py:540-548 W_KeyError.descr_str` —
            // a single-argument KeyError stringifies as `repr(args[0])`
            // so `str(KeyError('k'))` is `"'k'"`; with any other arg
            // count it falls back to `W_BaseException.descr_str` below.
            pyre_object::interp_exceptions::ExcKind::KeyError => {
                let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
                if !args.is_null()
                    && pyre_object::is_tuple(args)
                    && pyre_object::w_tuple_len(args) == 1
                {
                    let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
                    return Ok(Some(py_repr_wtf8(first)?));
                }
            }
            // `interp_exceptions.py:667-703 W_OSError.descr_str` reads
            // the `errno`/`strerror`/`filename`/`filename2` slots:
            // the 2-argument form renders as `"[Errno N] strerror"`,
            // extended with `": 'filename'"` and `" -> 'filename2'"`
            // when those are present.  `_init_error` drops filename
            // from `args`, so prefer the slot and fall back to the
            // positional arg (same 2..=5 gate as the getters) for the
            // internal-constructor path that leaves the slots `PY_NULL`.
            // Both errno and strerror absent falls back to
            // `W_BaseException.descr_str` below.
            pyre_object::interp_exceptions::ExcKind::OSError
            | pyre_object::interp_exceptions::ExcKind::FileNotFoundError => {
                let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
                let n = if !args.is_null() && pyre_object::is_tuple(args) {
                    pyre_object::w_tuple_len(args)
                } else {
                    0
                };
                let slot_or_arg = |slot: pyre_object::PyObjectRef,
                                   idx: usize|
                 -> Option<pyre_object::PyObjectRef> {
                    if !slot.is_null() {
                        return Some(slot);
                    }
                    if (2..=5).contains(&n) && idx < n {
                        unsafe { pyre_object::w_tuple_getitem(args, idx as i64) }
                    } else {
                        None
                    }
                };
                let w_errno = slot_or_arg(
                    pyre_object::interp_exceptions::w_exception_get_errno(obj),
                    0,
                );
                let w_strerror = slot_or_arg(
                    pyre_object::interp_exceptions::w_exception_get_strerror(obj),
                    1,
                );
                if let (Some(w_errno), Some(w_strerror)) = (w_errno, w_strerror) {
                    let errno = py_str_wtf8(w_errno)?;
                    let strerror = py_str_wtf8(w_strerror)?;
                    let mut out = Wtf8Buf::new();
                    out.push_str("[Errno ");
                    out.push_wtf8(&errno);
                    out.push_str("] ");
                    out.push_wtf8(&strerror);
                    let w_filename = slot_or_arg(
                        pyre_object::interp_exceptions::w_exception_get_filename(obj),
                        2,
                    )
                    .filter(|&f| !pyre_object::is_none(f));
                    if let Some(fname) = w_filename {
                        let w_filename2 = slot_or_arg(
                            pyre_object::interp_exceptions::w_exception_get_filename2(obj),
                            4,
                        )
                        .filter(|&f| !pyre_object::is_none(f));
                        if let Some(fname2) = w_filename2 {
                            out.push_str(": ");
                            out.push_wtf8(&py_repr_wtf8(fname)?);
                            out.push_str(" -> ");
                            out.push_wtf8(&py_repr_wtf8(fname2)?);
                            return Ok(Some(out));
                        }
                        out.push_str(": ");
                        out.push_wtf8(&py_repr_wtf8(fname)?);
                        return Ok(Some(out));
                    }
                    return Ok(Some(out));
                }
            }
            // `interp_exceptions.py:859-883 W_SyntaxError.descr_str` —
            // a non-str `msg` stringifies plainly; otherwise the message
            // is suffixed with the `basename(filename)` and `line N` /
            // `lines N-M` derived from the location attributes.  The
            // WTF-8 path already implements this; reuse it and drop any
            // lone surrogates for the plain-`String` caller.
            pyre_object::interp_exceptions::ExcKind::SyntaxError => {
                if let Some(w) = exception_descr_str_wtf8(obj)? {
                    return Ok(Some(w));
                }
            }
            _ => {}
        }
        Ok(None)
    }
}

/// `str(obj)` for diagnostic display (traceback headers / messages written to
/// stderr): like [`py_str`], but a lone surrogate is backslash-escaped
/// (`\udcXX`, the `backslashreplace` handler stderr uses) and a raising
/// `__str__` degrades to a placeholder, so rendering a diagnostic never panics.
///
/// # Safety
/// `obj` must be a valid object.
pub unsafe fn py_str_display(obj: PyObjectRef) -> String {
    unsafe {
        let w = match py_str_wtf8(obj) {
            Ok(w) => w,
            Err(_) => return "<unprintable>".to_string(),
        };
        if let Ok(s) = w.as_str() {
            return s.to_owned();
        }
        let s_obj = pyre_object::w_str_from_wtf8(w);
        crate::type_methods::encode_object(s_obj, "utf-8", "backslashreplace")
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .unwrap_or_else(|| "<unprintable>".to_string())
    }
}

/// The WTF-8 carrying subset of `W_BaseException.descr_str`: a base
/// exception whose `args_w` is a single `str` stringifies to that str
/// verbatim (`interp_exceptions.py:131 space.str(self.args_w[0])`).
/// Returns `None` for every other shape — no args, multiple args, a
/// non-`str` arg, or the Unicode/`KeyError` kinds whose `descr_str`
/// overrides are ASCII-only — letting `py_str_wtf8` fall back to
/// `py_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
unsafe fn exception_descr_str_wtf8(obj: PyObjectRef) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        // A user subclass that overrides `__str__` shadows the builtin
        // `W_BaseException.descr_str`; dispatch it (preserving WTF-8)
        // before the single-`str`-arg fast path below, matching `py_str`.
        if let Some(r) = exc_user_dunder_obj(obj, "__str__")? {
            return Ok(Some(pyre_object::w_str_get_wtf8(r).to_wtf8_buf()));
        }
        let kind = pyre_object::w_exception_get_kind(obj);
        if matches!(
            kind,
            pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                | pyre_object::interp_exceptions::ExcKind::KeyError
        ) {
            return Ok(None);
        }
        // `interp_exceptions.py:859 W_SyntaxError.descr_str` — format
        // `msg (filename, line lineno)`, falling back through the
        // filename-only, lineno-only, and bare-msg shapes. Shared by the
        // IndentationError / TabError subclasses (same `ExcKind`).
        if kind == pyre_object::interp_exceptions::ExcKind::SyntaxError {
            let w_msg = crate::baseobjspace::syntax_error_attr(obj, "msg");
            // `type(self.msg) is not str` → `return str(self.msg)`.
            if w_msg.is_null() || !pyre_object::pyobject::is_exact_type(w_msg, &STR_TYPE) {
                return Ok(Some(py_str_wtf8(w_msg)?));
            }
            let compose = |extra: Option<Wtf8Buf>| -> Wtf8Buf {
                let mut out = pyre_object::w_str_get_wtf8(w_msg).to_wtf8_buf();
                if let Some(inner) = extra {
                    out.push_str(" (");
                    out.push_wtf8(&inner);
                    out.push_str(")");
                }
                out
            };
            // `line %d` / `lines %d-%d` when `end_lineno > lineno`.
            let w_lineno = crate::baseobjspace::syntax_error_attr(obj, "lineno");
            let lineno_str: Option<Wtf8Buf> =
                if pyre_object::pyobject::is_exact_type(w_lineno, &INT_TYPE) {
                    let lineno = crate::baseobjspace::int_w(w_lineno).unwrap_or(0);
                    let w_end = crate::baseobjspace::syntax_error_attr(obj, "end_lineno");
                    let end = if pyre_object::pyobject::is_exact_type(w_end, &INT_TYPE) {
                        crate::baseobjspace::int_w(w_end).ok()
                    } else {
                        None
                    };
                    Some(match end {
                        Some(end) if end > lineno => {
                            Wtf8Buf::from_string(format!("lines {lineno}-{end}"))
                        }
                        _ => Wtf8Buf::from_string(format!("line {lineno}")),
                    })
                } else {
                    None
                };
            // `have_filename` → `my_basename(self.filename)`.
            // `interp_exceptions.py:875` substitutes `"???"` for a falsy
            // filename; 3.14 only tests `PyUnicode_Check`, so an *empty*
            // filename basenames to the empty string.
            let w_filename = crate::baseobjspace::syntax_error_attr(obj, "filename");
            if pyre_object::pyobject::is_exact_type(w_filename, &STR_TYPE) {
                let fbuf = pyre_object::w_str_get_wtf8(w_filename).to_wtf8_buf();
                let start = fbuf
                    .as_bytes()
                    .iter()
                    .rposition(|&b| b == b'/')
                    .map_or(0, |i| i + 1);
                let mut inner = fbuf[start..].to_wtf8_buf();
                if let Some(l) = lineno_str {
                    inner.push_str(", ");
                    inner.push_wtf8(&l);
                }
                return Ok(Some(compose(Some(inner))));
            }
            return Ok(Some(compose(lineno_str)));
        }
        let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
        if args.is_null() || !pyre_object::is_tuple(args) {
            return Ok(None);
        }
        if pyre_object::w_tuple_len(args) != 1 {
            return Ok(None);
        }
        let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
        // A tagged `int` immediate is never a `str`; skip the `ob_type` deref
        // (which would read the immediate as a pointer) and fall back to
        // `py_str`, which formats the tagged value directly.
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(first) {
            return Ok(None);
        }
        if first.is_null() || !std::ptr::eq((*first).ob_type, &STR_TYPE as *const PyType) {
            return Ok(None);
        }
        Ok(Some(pyre_object::w_str_get_wtf8(first).to_wtf8_buf()))
    }
}

/// Format an `int` `%d` position slot from a `W_BaseException`
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
        // `descr_str` deliberately str-coerces rather than raising; a raising
        // `__str__` on the mutated slot degrades to empty here.
        Err(py_str(stored).unwrap_or_default())
    }
}

/// Format an `str` `%s` slot (encoding / reason) from a typed
/// Unicode*Error field.  Mirrors Python's `"%s" % value` which calls
/// `str(value)` on non-str inputs (Python format-string `%s`
/// semantics).  `descr_init`'s `isinstance_str_w` check rejects
/// non-str at construction time; this helper covers the
/// post-construction mutation case (`e.encoding = 42`,
/// `e.reason = None`, etc.) the way PyPy would via `%s`-coerce.
unsafe fn unicode_err_str_slot(stored: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        if stored.is_null() {
            return Ok(Wtf8Buf::new());
        }
        if pyre_object::is_exact_type(stored, &pyre_object::STR_TYPE) {
            return Ok(pyre_object::w_str_get_wtf8(stored).to_wtf8_buf());
        }
        // `%s` propagates an exception raised by the value's `__str__`.
        py_str_wtf8(stored)
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
unsafe fn unicode_translate_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if initial.is_null() || pyre_object::is_none(initial) {
            return Ok(Wtf8Buf::new());
        }
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        // Formatting `reason` can run arbitrary Python and mutate the
        // exception.  CPython 3.14 rereads `object` before indexing it.
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if w_object.is_null() || !pyre_object::is_str(w_object) {
            return Err(crate::PyError::type_error(
                "UnicodeError 'object' attribute must be str",
            ));
        }
        let start_repr = unicode_err_int_repr(&start_slot);
        // Shape predicate `self.end == self.start + 1` — true iff both
        // slots are int AND `end == start + 1`.  Any non-int slot
        // makes PyPy's `==` False (different types), so render as the
        // plural shape with str-coerced position values.
        let single_char = matches!((&start_slot, &end_slot), (Ok(s), Ok(e)) if *e == *s + 1);
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let badchar_repr = if pyre_object::is_str(w_object) {
                // Read the offending code point through the surrogate-aware
                // WTF-8 view: the bad character is frequently a lone surrogate
                // (utf-8 strict encode), which `w_str_get_value` cannot hold.
                let code_points: Vec<u32> = pyre_object::w_str_get_wtf8(w_object)
                    .code_points()
                    .map(|c| c.to_u32())
                    .collect();
                usize::try_from(start)
                    .ok()
                    .and_then(|i| code_points.get(i).copied())
                    .map(|badchar| {
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
            let mut out = Wtf8Buf::from_string(format!(
                "can't translate character {} in position {}: ",
                badchar_repr.unwrap_or_else(|| "<?>".to_string()),
                start_repr,
            ));
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = Wtf8Buf::from_string(format!(
            "can't translate characters in position {}-{}: ",
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
        ));
        out.push_wtf8(&reason);
        Ok(out)
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
unsafe fn unicode_decode_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if initial.is_null() || pyre_object::is_none(initial) {
            return Ok(Wtf8Buf::new());
        }
        let encoding =
            unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_encoding(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ))?;
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if w_object.is_null() || !pyre_object::is_bytes_like(w_object) {
            return Err(crate::PyError::type_error(
                "UnicodeError 'object' attribute must be bytes",
            ));
        }
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
            let mut out = Wtf8Buf::new();
            out.push_str("'");
            out.push_wtf8(&encoding);
            out.push_str(&format!(
                "' codec can't decode byte {} in position {}: ",
                byte_repr.unwrap_or_else(|| "0x??".to_string()),
                start_repr,
            ));
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = Wtf8Buf::new();
        out.push_str("'");
        out.push_wtf8(&encoding);
        out.push_str(&format!(
            "' codec can't decode bytes in position {}-{}: ",
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
        ));
        out.push_wtf8(&reason);
        Ok(out)
    }
}

/// `pypy/module/exceptions/interp_exceptions.py:1175-1191
/// W_UnicodeEncodeError.descr_str` — same single/range split as
/// `W_UnicodeTranslateError` but prefixed with the encoding name.
/// Non-int / non-str / OOR mutations match the parity rules in
/// [`unicode_translate_error_str`] / [`unicode_decode_error_str`].
unsafe fn unicode_encode_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if initial.is_null() || pyre_object::is_none(initial) {
            return Ok(Wtf8Buf::new());
        }
        let encoding =
            unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_encoding(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ))?;
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        if w_object.is_null() || !pyre_object::is_str(w_object) {
            return Err(crate::PyError::type_error(
                "UnicodeError 'object' attribute must be str",
            ));
        }
        let start_repr = unicode_err_int_repr(&start_slot);
        let single_char = matches!((&start_slot, &end_slot), (Ok(s), Ok(e)) if *e == *s + 1);
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let badchar_repr = if pyre_object::is_str(w_object) {
                // Read the offending code point through the surrogate-aware
                // WTF-8 view: the bad character is frequently a lone surrogate
                // (utf-8 strict encode), which `w_str_get_value` cannot hold.
                let code_points: Vec<u32> = pyre_object::w_str_get_wtf8(w_object)
                    .code_points()
                    .map(|c| c.to_u32())
                    .collect();
                usize::try_from(start)
                    .ok()
                    .and_then(|i| code_points.get(i).copied())
                    .map(|badchar| {
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
            let mut out = Wtf8Buf::new();
            out.push_str("'");
            out.push_wtf8(&encoding);
            out.push_str(&format!(
                "' codec can't encode character {} in position {}: ",
                badchar_repr.unwrap_or_else(|| "<?>".to_string()),
                start_repr,
            ));
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = Wtf8Buf::new();
        out.push_str("'");
        out.push_wtf8(&encoding);
        out.push_str(&format!(
            "' codec can't encode characters in position {}-{}: ",
            start_repr,
            unicode_err_end_minus_one_repr(&end_slot),
        ));
        out.push_wtf8(&reason);
        Ok(out)
    }
}

/// Display wrapper for PyObjectRef.
pub struct PyDisplay(pub PyObjectRef);

impl fmt::Display for PyDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            write!(f, "NULL")
        } else {
            // `Display` cannot surface a `PyError`; a raising `__str__` in a
            // diagnostic output context degrades to a placeholder rather than
            // propagating (the user-facing `print()`/`str()` paths thread the
            // error through `py_str`).
            let s =
                unsafe { py_str(self.0) }.unwrap_or_else(|_| "<exception in __str__>".to_string());
            write!(f, "{s}")
        }
    }
}
