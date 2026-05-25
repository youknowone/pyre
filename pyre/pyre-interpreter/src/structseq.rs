//! structseq factory — `lib_pypy/_structseq.py` parity port.
//!
//! Each call to [`make_struct_seq`] produces a new tuple subclass with
//! named-field GetSetProperty descriptors and a side table mapping
//! `(class pointer, field name) → tuple index`.  Modules like `pwd`,
//! `grp`, `resource`, `posix` materialise their result types this way
//! so `obj.st_mode`, `pw.pw_uid`, `r.ru_utime` resolve to a real tuple
//! element instead of being string placeholders.
//!
//! PyPy reference:
//!
//! * `lib_pypy/_structseq.py:9-37 structseqfield` — per-field descriptor
//!   exposing `__get__` that returns `obj[self.index]` (positional) or
//!   `obj.__dict__[self.__name__]` (extra).  Pyre matches the positional
//!   half via [`structseq_field_get`] reading the W_GetSetProperty's
//!   `name` slot and dispatching through `STRUCTSEQ_REGISTRY`.
//! * `lib_pypy/_structseq.py:43-87 structseqtype` — metaclass.  Pyre
//!   replaces the metaclass machinery with a direct
//!   `make_builtin_type_with_base(name, init, tuple_type)` call inside
//!   [`make_struct_seq`].
//! * `lib_pypy/_structseq.py:95-144 structseq_new` — the
//!   `cls(sequence[, dict])` constructor.  Pyre implements the
//!   positional-only subset (the `extra_fields` dict-overlay arm is not
//!   yet wired up; pwd / grp / resource don't use it).
//! * `lib_pypy/_structseq.py:156-163 structseq_repr` — `"name(f0=v0,
//!   f1=v1, ...)"` rendering.

use indexmap::IndexMap;
use std::cell::RefCell;

use pyre_object::PyObjectRef;

use crate::PyError;

/// `lib_pypy/_structseq.py:43-87` — metaclass-installed class-level
/// metadata.  Pyre stores the name + positional field list keyed by the
/// subclass W_TypeObject pointer so the generic field getter resolves
/// indices without a per-field closure.
struct StructSeqDescr {
    name: String,
    /// Field names in positional order.  Names starting with `_` are
    /// CPython's "unnamed" placeholders (`_structseq.py:67-69`).
    fields: Vec<String>,
}

thread_local! {
    /// `class_ptr → StructSeqDescr`.  Pyre keys by the subclass type
    /// pointer because the GetSetProperty descriptor only carries a
    /// `name` slot (`getsetproperty.rs:174`), not the owning class.
    static STRUCTSEQ_REGISTRY: RefCell<IndexMap<usize, StructSeqDescr>> =
        RefCell::new(IndexMap::new());
}

/// `lib_pypy/_structseq.py:31-37 structseqfield.__get__` —
/// resolves the descriptor's name to a positional index via the
/// per-type registry and returns `obj[index]`.
///
/// args[0] = descriptor (`W_GetSetProperty`), args[1] = receiver.
fn structseq_field_get(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() < 2 {
        return Err(PyError::type_error(
            "structseq field getter missing receiver",
        ));
    }
    let desc = args[0];
    let inst = args[1];
    let name_obj = unsafe { pyre_object::getsetproperty::w_getset_get_name(desc) };
    if name_obj.is_null() || !unsafe { pyre_object::is_str(name_obj) } {
        return Err(PyError::type_error(
            "structseq field descriptor has no name",
        ));
    }
    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let cls = unsafe { (*inst).w_class };
    let idx = STRUCTSEQ_REGISTRY.with(|r| -> Option<usize> {
        let map = r.borrow();
        let entry = map.get(&(cls as usize))?;
        entry.fields.iter().position(|n| n == &name)
    });
    let Some(idx) = idx else {
        return Err(PyError::attribute_error(format!(
            "structseq object has no field {name}"
        )));
    };
    let item = unsafe { pyre_object::w_tuple_getitem(inst, idx as i64) }
        .ok_or_else(|| PyError::index_error("structseq field out of range"))?;
    Ok(item)
}

/// `lib_pypy/_structseq.py:156-163 structseq_repr`.
fn structseq_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let inst = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if inst.is_null() {
        return Err(PyError::type_error("structseq __repr__ missing self"));
    }
    let cls = unsafe { (*inst).w_class };
    let (name, fields) = STRUCTSEQ_REGISTRY.with(|r| -> (String, Vec<String>) {
        let map = r.borrow();
        map.get(&(cls as usize))
            .map(|d| (d.name.clone(), d.fields.clone()))
            .unwrap_or_default()
    });
    let n = unsafe { pyre_object::w_tuple_len(inst) };
    let mut parts: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let item = unsafe { pyre_object::w_tuple_getitem(inst, i as i64) }
            .unwrap_or(pyre_object::w_none());
        let fname = fields.get(i).cloned().unwrap_or_else(|| format!("?{i}"));
        let r_str = unsafe { crate::py_repr(item) };
        parts.push(format!("{fname}={r_str}"));
    }
    Ok(pyre_object::w_str_new(&format!(
        "{name}({})",
        parts.join(", ")
    )))
}

/// `lib_pypy/_structseq.py:95-144 structseq_new` — the
/// `cls(sequence)` constructor.  Pyre supports the all-positional
/// arity-exact case (`len(sequence) == n_sequence_fields`); the
/// `extra_fields` overlay arm raises TypeError until pwd/grp/resource
/// actually need it.
fn structseq_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() < 2 {
        return Err(PyError::type_error("structseq() requires class + sequence"));
    }
    let cls = args[0];
    let sequence = args[1];
    let n_seq = read_class_int(cls, "n_sequence_fields").unwrap_or(0) as usize;
    let n_fields = read_class_int(cls, "n_fields").unwrap_or(n_seq as i64) as usize;
    let items = crate::builtins::collect_iterable(sequence)?;
    if items.len() < n_seq {
        return Err(PyError::type_error(format!(
            "expected a sequence with {} {} items. has {}",
            if n_seq < n_fields {
                "at least"
            } else {
                "exactly"
            },
            n_seq,
            items.len()
        )));
    }
    if items.len() > n_fields {
        return Err(PyError::type_error(format!(
            "expected a sequence with {} {} items. has {}",
            if n_seq < n_fields {
                "at most"
            } else {
                "exactly"
            },
            n_fields,
            items.len()
        )));
    }
    let trimmed: Vec<PyObjectRef> = items.into_iter().take(n_seq).collect();
    Ok(new_instance(cls, trimmed))
}

fn read_class_int(cls: PyObjectRef, attr: &str) -> Option<i64> {
    let v = crate::baseobjspace::getattr(cls, attr).ok()?;
    if unsafe { pyre_object::is_int(v) } {
        Some(unsafe { pyre_object::w_int_get_value(v) })
    } else {
        None
    }
}

/// Allocate a structseq instance directly from a Rust-side value
/// vector — host modules use this when they already have all the
/// positional fields materialised and do not need the iteration /
/// arity-check work `structseq_descr_new` does for app-level callers.
pub fn new_instance(cls: PyObjectRef, items: Vec<PyObjectRef>) -> PyObjectRef {
    let obj = pyre_object::w_tuple_new_array_backed(items);
    unsafe {
        (*obj).w_class = cls;
    }
    obj
}

/// `lib_pypy/_structseq.py:43-87 structseqtype.__new__` —
/// build a new tuple subclass with the supplied positional field names.
/// The returned type is the value module callers stash so future
/// allocations route through [`new_instance`].
pub fn make_struct_seq(name: &'static str, field_names: &[&'static str]) -> PyObjectRef {
    let n_sequence_fields = field_names.len();
    let n_unnamed_fields = field_names.iter().filter(|n| n.starts_with('_')).count();
    let owned_names: Vec<String> = field_names.iter().map(|s| s.to_string()).collect();
    let owned_names_for_init = owned_names.clone();

    let tuple_type = crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);

    let cls = crate::typedef::make_builtin_type_with_base(
        name,
        move |ns| {
            // `_structseq.py:79-80` — `__new__` / `__reduce__` /
            // `__setattr__` / `__repr__` / `__str__` are wired by the
            // metaclass.  Pyre installs `__new__` + `__repr__` only;
            // `__reduce__` and `__setattr__` are TBD when pickle and
            // user-side mutation are exercised by the test suite.
            crate::dict_storage_store(
                ns,
                "__new__",
                crate::make_builtin_function("__new__", structseq_descr_new),
            );
            crate::dict_storage_store(
                ns,
                "__repr__",
                crate::make_builtin_function_with_arity("__repr__", structseq_repr, 1),
            );
            crate::dict_storage_store(
                ns,
                "__str__",
                crate::make_builtin_function_with_arity("__str__", structseq_repr, 1),
            );

            crate::dict_storage_store(
                ns,
                "n_sequence_fields",
                pyre_object::w_int_new(n_sequence_fields as i64),
            );
            crate::dict_storage_store(
                ns,
                "n_fields",
                pyre_object::w_int_new(n_sequence_fields as i64),
            );
            crate::dict_storage_store(
                ns,
                "n_unnamed_fields",
                pyre_object::w_int_new(n_unnamed_fields as i64),
            );

            // Per-field GetSetProperty descriptors.  `_structseq.py:31-37`
            // implements `structseqfield.__get__` — pyre fans out to the
            // generic `structseq_field_get` keyed by descriptor name.
            for fname in &owned_names_for_init {
                let getter = crate::make_builtin_function_with_arity(
                    "structseq_field_get",
                    structseq_field_get,
                    2,
                );
                let desc = crate::typedef::make_getset_descriptor_named(getter, fname.as_str());
                crate::dict_storage_store(ns, fname.as_str(), desc);
            }

            // `_structseq.py:85-86` — `__match_args__` excludes
            // unnamed (leading-`_`) fields.
            let match_args: Vec<PyObjectRef> = owned_names_for_init
                .iter()
                .filter(|n| !n.starts_with('_'))
                .map(|n| pyre_object::w_str_new(n.as_str()))
                .collect();
            crate::dict_storage_store(ns, "__match_args__", pyre_object::w_tuple_new(match_args));
        },
        tuple_type,
    );

    STRUCTSEQ_REGISTRY.with(|r| {
        r.borrow_mut().insert(
            cls as usize,
            StructSeqDescr {
                name: name.to_string(),
                fields: owned_names,
            },
        );
    });

    cls
}
