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
//!   `cls(sequence[, dict])` constructor, including the surplus-positional
//!   / dict / `None`-default fill of the named-only extra fields and the
//!   single-field scalar-wrap path.
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
    /// Named-only fields stored in the instance `__dict__` rather than the
    /// tuple body (`_structseq.py:31-37` — the `obj.__dict__[name]` arm).
    /// `os.stat_result` uses these for the float `st_atime`/`st_mtime`/
    /// `st_ctime` (which shadow the integer sequence slots 7..10) and the
    /// `st_*_ns` / `st_blksize` / `st_blocks` / `st_rdev` extras.  A name
    /// present here takes priority over a same-named positional slot, so a
    /// data-descriptor read resolves to the extra value while `obj[i]`
    /// still returns the sequence integer.
    extra_fields: Vec<String>,
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
    // `_structseq.py:31` — `structseqfield.__get__` returns the descriptor
    // itself for class-level access (`if obj is None: return self`).
    if inst.is_null() || unsafe { pyre_object::pyobject::is_none(inst) } {
        return Ok(desc);
    }
    let name_obj = unsafe { pyre_object::getsetproperty::w_getset_get_name(desc) };
    if name_obj.is_null() || !unsafe { pyre_object::is_str(name_obj) } {
        return Err(PyError::type_error(
            "structseq field descriptor has no name",
        ));
    }
    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let cls = unsafe { (*inst).w_class };

    enum Resolved {
        Extra,
        Positional(usize),
        Missing,
    }
    // `_structseq.py:31-37` — an extra (dict-backed) field shadows a
    // same-named positional slot, so resolve those first.
    let resolved = STRUCTSEQ_REGISTRY.with(|r| {
        let map = r.borrow();
        let Some(entry) = map.get(&(cls as usize)) else {
            return Resolved::Missing;
        };
        if entry.extra_fields.iter().any(|n| n == &name) {
            Resolved::Extra
        } else if let Some(idx) = entry.fields.iter().position(|n| n == &name) {
            Resolved::Positional(idx)
        } else {
            Resolved::Missing
        }
    });
    match resolved {
        Resolved::Extra => {
            let w_dict = crate::baseobjspace::getdict(inst);
            if !w_dict.is_null() {
                if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, &name) } {
                    return Ok(v);
                }
            }
            Err(PyError::attribute_error(format!(
                "structseq object has no field {name}"
            )))
        }
        Resolved::Positional(idx) => {
            let item = unsafe { pyre_object::w_tuple_getitem(inst, idx as i64) }
                .ok_or_else(|| PyError::index_error("structseq field out of range"))?;
            Ok(item)
        }
        Resolved::Missing => Err(PyError::attribute_error(format!(
            "structseq object has no field {name}"
        ))),
    }
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

/// `lib_pypy/_structseq.py structseq_reduce` — `return type(self),
/// (tuple(self), self.__dict__)`.  The reconstruction call routes back
/// through [`structseq_descr_new`] (`cls(sequence, dict)`), with the
/// instance `__dict__` supplying the named-only extra fields.
fn structseq_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let inst = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if inst.is_null() {
        return Err(PyError::type_error("structseq __reduce__ missing self"));
    }
    let cls = unsafe { (*inst).w_class };
    // `tuple(self)` — the positional body as a plain tuple.
    let n = unsafe { pyre_object::w_tuple_len(inst) };
    let mut items: Vec<PyObjectRef> = Vec::with_capacity(n);
    for i in 0..n {
        items.push(
            unsafe { pyre_object::w_tuple_getitem(inst, i as i64) }
                .unwrap_or_else(pyre_object::w_none),
        );
    }
    let body_tuple = pyre_object::w_tuple_new(items);
    // `self.__dict__` carries the named-only extras for reconstruction.
    let w_dict = crate::baseobjspace::getdict(inst);
    let dict = if w_dict.is_null() {
        pyre_object::w_dict_new()
    } else {
        w_dict
    };
    let inner = pyre_object::w_tuple_new(vec![body_tuple, dict]);
    Ok(pyre_object::w_tuple_new(vec![cls, inner]))
}

/// `lib_pypy/_structseq.py structseq_setattr` — structseq instances are
/// read-only.  Setting a known field raises `"readonly attribute"`;
/// setting any other name raises the standard missing-attribute error.
fn structseq_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() < 3 {
        return Err(PyError::type_error(
            "structseq __setattr__ requires name and value",
        ));
    }
    let inst = args[0];
    let attr_obj = args[1];
    if !unsafe { pyre_object::is_str(attr_obj) } {
        return Err(PyError::type_error("attribute name must be string"));
    }
    let attr = unsafe { pyre_object::w_str_get_value(attr_obj) };
    let cls = unsafe { (*inst).w_class };
    // `attr not in type(self).__dict__` — own-dict membership, not MRO.
    let in_type_dict = {
        let dict_ptr = unsafe { pyre_object::typeobject::w_type_get_dict_ptr(cls) }
            as *const crate::DictStorage;
        if dict_ptr.is_null() {
            false
        } else {
            crate::dict_storage_get(unsafe { &*dict_ptr }, &attr).is_some()
        }
    };
    if !in_type_dict {
        let cls_name = unsafe { pyre_object::w_type_get_name(cls) };
        return Err(PyError::attribute_error(format!(
            "'{cls_name}' object has no attribute '{attr}'"
        )));
    }
    Err(PyError::attribute_error("readonly attribute"))
}

/// `lib_pypy/_structseq.py:95-144 structseq_new` — the `cls(sequence[,
/// dict])` constructor.  The first `n_sequence_fields` items fill the
/// tuple body; any surplus positional items, then the optional dict, then
/// `None` defaults, fill the named-only extra fields.
fn structseq_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() < 2 {
        return Err(PyError::type_error("structseq() requires class + sequence"));
    }
    let cls = args[0];
    let n_seq = read_class_int(cls, "n_sequence_fields").unwrap_or(0) as usize;
    let n_fields = read_class_int(cls, "n_fields").unwrap_or(n_seq as i64) as usize;
    let (name, extra_names) = STRUCTSEQ_REGISTRY.with(|r| {
        let map = r.borrow();
        map.get(&(cls as usize))
            .map(|d| (d.name.clone(), d.extra_fields.clone()))
            .unwrap_or_else(|| ("structseq".to_string(), Vec::new()))
    });

    // `_structseq.py:95-101` — the optional second arg is a dict supplying
    // values for the named-only extra fields.
    if args.len() > 3 {
        return Err(PyError::type_error(format!(
            "{name}() takes at most 2 arguments ({} given)",
            args.len() - 1
        )));
    }
    let dict_arg = args.get(2).copied();
    if let Some(d) = dict_arg {
        if !unsafe { pyre_object::is_dict(d) } {
            return Err(PyError::type_error(format!(
                "{name} takes a dict as second arg, if any"
            )));
        }
    }

    // `_structseq.py:102-107` — a 1-field structseq wraps its scalar arg;
    // otherwise the arg is iterated into the field values.
    let mut items = if n_seq == 1 {
        vec![args[1]]
    } else {
        crate::builtins::collect_iterable(args[1])?
    };
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

    // `_structseq.py:115-143` — first `n_seq` items form the tuple body;
    // surplus items fill leading extras, then the dict, then `None`.
    let surplus = items.len() - n_seq;
    let surplus_vals: Vec<PyObjectRef> = items.split_off(n_seq);
    let body = items;
    let mut extras: Vec<(&str, PyObjectRef)> = Vec::with_capacity(extra_names.len());
    for (i, ename) in extra_names.iter().enumerate() {
        let in_dict = dict_arg
            .is_some_and(|d| unsafe { pyre_object::w_dict_getitem_str(d, ename).is_some() });
        let value = if i < surplus {
            if in_dict {
                return Err(PyError::type_error(format!(
                    "duplicate value for '{ename}'"
                )));
            }
            surplus_vals[i]
        } else if let Some(d) = dict_arg {
            unsafe { pyre_object::w_dict_getitem_str(d, ename) }.unwrap_or_else(pyre_object::w_none)
        } else {
            pyre_object::w_none()
        };
        extras.push((ename.as_str(), value));
    }

    // `app_posix.py:71-80 stat_result.__init__` — a tuple-constructed
    // stat_result leaves the float `st_atime`/`st_mtime`/`st_ctime` extras
    // as None; fall back to the integer timestamps at body slots 7..9.
    if name == "os.stat_result" && body.len() > 9 {
        for (slot, ename) in [(7usize, "st_atime"), (8, "st_mtime"), (9, "st_ctime")] {
            if let Some(entry) = extras.iter_mut().find(|(n, _)| *n == ename) {
                if unsafe { pyre_object::is_none(entry.1) } {
                    entry.1 = body[slot];
                }
            }
        }
    }

    Ok(new_instance_with_extra(cls, body, extras))
}

fn read_class_int(cls: PyObjectRef, attr: &str) -> Option<i64> {
    let v = crate::baseobjspace::getattr_str(cls, attr).ok()?;
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
    new_instance_with_extra(cls, items, Vec::new())
}

/// Allocate a structseq instance carrying both the positional tuple body
/// (`items`) and named-only extras (`extras`).  The extras are written
/// into the instance `__dict__` so the per-field getter can resolve them
/// (`_structseq.py:31-37`); the owning type must have been built with a
/// matching `extra_fields` list via [`make_struct_seq_with_extra`] (which
/// sets `hasdict`).  `os.stat_result` uses this for the float time fields
/// and the `st_*_ns` extras.
pub fn new_instance_with_extra(
    cls: PyObjectRef,
    items: Vec<PyObjectRef>,
    extras: Vec<(&str, PyObjectRef)>,
) -> PyObjectRef {
    let obj = pyre_object::w_tuple_new_array_backed(items);
    unsafe {
        (*obj).w_class = cls;
    }
    if !extras.is_empty() {
        let w_dict = pyre_object::w_dict_new();
        for (k, v) in extras {
            unsafe { pyre_object::w_dict_setitem_str(w_dict, k, v) };
        }
        let _ = crate::baseobjspace::setdict(obj, w_dict);
    }
    obj
}

/// `lib_pypy/_structseq.py:43-87 structseqtype.__new__` —
/// build a new tuple subclass with the supplied positional field names.
/// The returned type is the value module callers stash so future
/// allocations route through [`new_instance`].
pub fn make_struct_seq(name: &'static str, field_names: &[&'static str]) -> PyObjectRef {
    make_struct_seq_impl(name, field_names, &[])
}

/// Like [`make_struct_seq`] but adds named-only fields beyond the tuple
/// sequence (`_structseq.py:31-37` extra-field arm).  `extra_field_names`
/// resolve through the instance `__dict__`, shadowing any same-named
/// positional slot, and the type is marked `hasdict` so [`new_instance_with_extra`]
/// can store them.  `os.stat_result` is the canonical user.
pub fn make_struct_seq_with_extra(
    name: &'static str,
    field_names: &[&'static str],
    extra_field_names: &[&'static str],
) -> PyObjectRef {
    make_struct_seq_impl(name, field_names, extra_field_names)
}

fn make_struct_seq_impl(
    name: &'static str,
    field_names: &[&'static str],
    extra_field_names: &[&'static str],
) -> PyObjectRef {
    let n_sequence_fields = field_names.len();
    let n_unnamed_fields = field_names.iter().filter(|n| n.starts_with('_')).count();
    let owned_names: Vec<String> = field_names.iter().map(|s| s.to_string()).collect();
    let owned_extra: Vec<String> = extra_field_names.iter().map(|s| s.to_string()).collect();
    let has_extra = !owned_extra.is_empty();

    // Descriptor set = sequence names ∪ extra names (sequence order first,
    // then extra-only).  A name in both gets a single descriptor; the
    // getter routes it to the extra (dict) value.
    let mut descriptor_names: Vec<String> = owned_names.clone();
    for e in &owned_extra {
        if !descriptor_names.contains(e) {
            descriptor_names.push(e.clone());
        }
    }
    let n_fields = descriptor_names.len();

    let owned_names_for_init = owned_names.clone();

    let tuple_type = crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);

    let cls = crate::typedef::make_builtin_type_with_base(
        name,
        move |ns| {
            // `_structseq.py:79-80` — `__new__` / `__reduce__` /
            // `__setattr__` / `__repr__` / `__str__` are wired by the
            // metaclass.
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
                "__reduce__",
                crate::make_builtin_function_with_arity("__reduce__", structseq_reduce, 1),
            );
            crate::dict_storage_store(
                ns,
                "__setattr__",
                crate::make_builtin_function_with_arity("__setattr__", structseq_setattr, 3),
            );

            crate::dict_storage_store(
                ns,
                "n_sequence_fields",
                pyre_object::w_int_new(n_sequence_fields as i64),
            );
            crate::dict_storage_store(ns, "n_fields", pyre_object::w_int_new(n_fields as i64));
            crate::dict_storage_store(
                ns,
                "n_unnamed_fields",
                pyre_object::w_int_new(n_unnamed_fields as i64),
            );

            // Per-field GetSetProperty descriptors.  `_structseq.py:31-37`
            // implements `structseqfield.__get__` — pyre fans out to the
            // generic `structseq_field_get` keyed by descriptor name.
            for fname in &descriptor_names {
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

    // Extra fields live in the instance `__dict__`, so the type must
    // advertise `hasdict` for `setdict`/`getdict` to route through the
    // instance-dict side table.
    if has_extra {
        unsafe { pyre_object::typeobject::w_type_set_hasdict(cls, true) };
    }

    STRUCTSEQ_REGISTRY.with(|r| {
        r.borrow_mut().insert(
            cls as usize,
            StructSeqDescr {
                name: name.to_string(),
                fields: owned_names,
                extra_fields: owned_extra,
            },
        );
    });

    cls
}
