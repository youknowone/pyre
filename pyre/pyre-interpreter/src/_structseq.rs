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
//!   half via [`structseq_field_get`] reading the GetSetProperty's
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
use std::sync::{Mutex, OnceLock};

use pyre_object::PyObjectRef;

use crate::PyError;

/// `lib_pypy/_structseq.py:43-87` — metaclass-installed class-level
/// metadata.  Pyre stores the name + positional field list keyed by the
/// subclass W_TypeObject pointer so the generic field getter resolves
/// indices without a per-field closure.
struct StructSeqDescr {
    name: String,
    /// Field names in positional order.  Names starting with `_` are
    /// unnamed placeholders (`_structseq.py:67-69`).
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

/// `class_ptr → StructSeqDescr`.  Pyre keys by the subclass type
/// pointer because the GetSetProperty descriptor only carries a
/// `name` slot (`typedef.rs:174`), not the owning class.
static STRUCTSEQ_REGISTRY: OnceLock<Mutex<IndexMap<usize, StructSeqDescr>>> = OnceLock::new();

fn structseq_registry() -> &'static Mutex<IndexMap<usize, StructSeqDescr>> {
    STRUCTSEQ_REGISTRY.get_or_init(|| Mutex::new(IndexMap::new()))
}

/// Whether `obj` is one of the heap types created by `structseqtype`.
/// Structseq types are unacceptable as bases but, unlike most types with that
/// flag, their constructor accepts the `sequence=` and `dict=` keywords.
pub(crate) fn is_structseq_type(obj: PyObjectRef) -> bool {
    structseq_registry()
        .lock()
        .unwrap()
        .contains_key(&(obj as usize))
}

/// `lib_pypy/_structseq.py:31-37 structseqfield.__get__` —
/// resolves the descriptor's name to a positional index via the
/// per-type registry and returns `obj[index]`.
///
/// args[0] = descriptor (`GetSetProperty`), args[1] = receiver.
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
    let name_obj = unsafe { pyre_object::typedef::w_getset_get_name(desc) };
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
    let resolved = {
        let map = structseq_registry().lock().unwrap();
        let Some(entry) = map.get(&(cls as usize)) else {
            return Err(PyError::attribute_error(format!(
                "structseq object has no field {name}"
            )));
        };
        if entry.extra_fields.iter().any(|n| n == &name) {
            Resolved::Extra
        } else if let Some(idx) = entry.fields.iter().position(|n| n == &name) {
            Resolved::Positional(idx)
        } else {
            Resolved::Missing
        }
    };
    match resolved {
        Resolved::Extra => {
            let w_dict = crate::baseobjspace::getdict_native(inst);
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
    let (name, fields) = {
        let map = structseq_registry().lock().unwrap();
        map.get(&(cls as usize))
            .map(|d| (d.name.clone(), d.fields.clone()))
            .unwrap_or_default()
    };
    let n = unsafe { pyre_object::w_tuple_len(inst) };
    let mut parts: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let item = unsafe { pyre_object::w_tuple_getitem(inst, i as i64) }
            .unwrap_or(pyre_object::w_none());
        let fname = fields.get(i).cloned().unwrap_or_else(|| format!("?{i}"));
        let r_str = unsafe { crate::py_repr(item)? };
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
    let w_dict = crate::baseobjspace::getdict_native(inst);
    let dict = if w_dict.is_null() {
        pyre_object::w_dict_new()
    } else {
        w_dict
    };
    let inner = pyre_object::w_tuple_new(vec![body_tuple, dict]);
    Ok(pyre_object::w_tuple_new(vec![cls, inner]))
}

/// CPython 3.14 `structseq___replace__` — copy the positional body and
/// named-only fields, overlay keyword changes, and return the same structseq
/// type.  Types with unnamed positional fields cannot map every tuple slot
/// back to a keyword and therefore reject replacement altogether.
fn structseq_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&inst) = positional.first() else {
        return Err(PyError::type_error(
            "__replace__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() != 1 {
        return Err(PyError::type_error(
            "__replace__() takes no positional arguments",
        ));
    }
    let cls = unsafe { (*inst).w_class };
    let (name, fields, extra_fields) = {
        let map = structseq_registry().lock().unwrap();
        let Some(descr) = map.get(&(cls as usize)) else {
            return Err(PyError::type_error(
                "__replace__() requires a structseq instance",
            ));
        };
        (
            descr.name.clone(),
            descr.fields.clone(),
            descr.extra_fields.clone(),
        )
    };
    if fields.iter().any(|field| field.starts_with('_')) {
        return Err(PyError::type_error(format!(
            "__replace__() is not supported for {name} because it has unnamed field(s)"
        )));
    }

    let changes: Vec<(String, PyObjectRef)> = kwargs
        .map(|dict| unsafe { pyre_object::w_dict_items(dict) })
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(key, value)| {
            if unsafe { pyre_object::is_str(key) }
                && unsafe { pyre_object::w_str_get_value(key) } == "__pyre_kw__"
            {
                None
            } else if unsafe { pyre_object::is_str(key) } {
                Some((
                    unsafe { pyre_object::w_str_get_value(key) }.to_string(),
                    value,
                ))
            } else {
                // Python call syntax guarantees string keyword names.  Keep a
                // defensive non-string marker without invoking user `repr`
                // while the copied structseq fields are held in raw locals.
                Some(("<non-string>".to_string(), value))
            }
        })
        .collect();
    let unexpected: Vec<String> = changes
        .iter()
        .filter(|(key, _)| !fields.contains(key) && !extra_fields.contains(key))
        .map(|(key, _)| format!("'{key}'"))
        .collect();
    if !unexpected.is_empty() {
        return Err(PyError::type_error(format!(
            "Got unexpected field name(s): [{}]",
            unexpected.join(", ")
        )));
    }

    let body: Vec<PyObjectRef> = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            changes
                .iter()
                .find(|(key, _)| key == field)
                .map(|(_, value)| *value)
                .or_else(|| unsafe { pyre_object::w_tuple_getitem(inst, index as i64) })
                .unwrap_or_else(pyre_object::w_none)
        })
        .collect();
    let source_dict = crate::baseobjspace::getdict_native(inst);
    let extras: Vec<(&str, PyObjectRef)> = extra_fields
        .iter()
        .map(|field| {
            let value = changes
                .iter()
                .find(|(key, _)| key == field)
                .map(|(_, value)| *value)
                .or_else(|| {
                    (!source_dict.is_null())
                        .then(|| unsafe { pyre_object::w_dict_getitem_str(source_dict, field) })
                        .flatten()
                })
                .unwrap_or_else(pyre_object::w_none);
            (field.as_str(), value)
        })
        .collect();
    Ok(new_instance_with_extra(cls, body, extras))
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
    let in_type_dict = crate::type_dict_contains(cls, &attr);
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
    if args.len() < 2 || args[1].is_null() {
        return Err(PyError::type_error("structseq() requires class + sequence"));
    }
    let cls = args[0];
    let n_seq = read_class_int(cls, "n_sequence_fields").unwrap_or(0) as usize;
    let n_fields = read_class_int(cls, "n_fields").unwrap_or(n_seq as i64) as usize;
    let (name, extra_names) = {
        let map = structseq_registry().lock().unwrap();
        map.get(&(cls as usize))
            .map(|d| (d.name.clone(), d.extra_fields.clone()))
            .unwrap_or_else(|| ("structseq".to_string(), Vec::new()))
    };

    // `_structseq.py:95-101` — the optional second arg is a dict supplying
    // values for the named-only extra fields.
    if args.len() > 3 {
        return Err(PyError::type_error(format!(
            "{name}() takes at most 2 arguments ({} given)",
            args.len() - 1
        )));
    }
    // Signature binding leaves an omitted optional argument as PY_NULL.  An
    // explicit None is different and is rejected by both PyPy's
    // `isinstance(dict, builtin_dict)` and CPython 3.14's `PyDict_Check`.
    let dict_arg = args.get(2).copied().filter(|d| !d.is_null());
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

    // CPython 3.14 consumes only named-only fields that have not already
    // been supplied by surplus sequence items.  Any remaining key is either
    // a duplicate positional value or an unknown field; both use the shared
    // structseq diagnostic.  PyPy's older app-level constructor only noticed
    // duplicates among extra fields, so the 3.14 rule wins here.
    if let Some(d) = dict_arg {
        let allowed = &extra_names[surplus..];
        let has_unexpected = unsafe { pyre_object::w_dict_items(d) }
            .into_iter()
            .any(|(key, _)| {
                if !unsafe { pyre_object::is_str(key) } {
                    return true;
                }
                let key = unsafe { pyre_object::w_str_get_value(key) };
                !allowed.iter().any(|name| name == &key)
            });
        if has_unexpected {
            return Err(PyError::type_error(
                "got duplicate or unexpected field name(s)",
            ));
        }
    }

    let mut extras: Vec<(&str, PyObjectRef)> = Vec::with_capacity(extra_names.len());
    for (i, ename) in extra_names.iter().enumerate() {
        let in_dict = dict_arg
            .is_some_and(|d| unsafe { pyre_object::w_dict_getitem_str(d, ename).is_some() });
        let value = if i < surplus {
            if in_dict {
                return Err(PyError::type_error(
                    "got duplicate or unexpected field name(s)",
                ));
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
    // RPython keeps constructor arguments live as GC references.  Mirror that
    // shape explicitly across the tuple/dict allocations instead of relying
    // on raw Rust Vec entries surviving a moving collection.
    let _roots = pyre_object::gc_roots::push_roots();
    // Publish the class, every item and every extra as one batch: the
    // forwarding query inside a first `pin_root` can park behind another
    // thread's collection, which would leave the values still held only in
    // these Rust vectors naming pre-move addresses.
    let mut roots = Vec::with_capacity(1 + items.len() + extras.len());
    roots.push(cls);
    roots.extend_from_slice(&items);
    roots.extend(extras.iter().map(|&(_, value)| value));
    let cls_slot = pyre_object::gc_roots::pin_roots(&roots);
    let items_slot = cls_slot + 1;
    let extras_slot = items_slot + items.len();
    let rooted_items = (0..items.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(items_slot + index))
        .collect();
    let obj = pyre_object::w_tuple_new_array_backed(rooted_items);
    pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    unsafe {
        (*pyre_object::gc_roots::shadow_stack_get(obj_slot)).w_class =
            pyre_object::gc_roots::shadow_stack_get(cls_slot);
    }
    if !extras.is_empty() {
        let w_dict = pyre_object::w_dict_new();
        pyre_object::gc_roots::pin_root(w_dict);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        for (index, (key, _)) in extras.iter().enumerate() {
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    key,
                    pyre_object::gc_roots::shadow_stack_get(extras_slot + index),
                )
            };
        }
        crate::baseobjspace::setdict(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        )
        .expect(
            "structseq extras: setdict on a fresh hasdict tuple subclass with a fresh dict cannot fail",
        );
    }
    pyre_object::gc_roots::shadow_stack_get(obj_slot)
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

    let cls = make_heap_structseq_type(
        name,
        move |ns| {
            // `_structseq.py:79-80` — `__new__` / `__reduce__` /
            // `__setattr__` / `__repr__` / `__str__` are wired by the
            // metaclass.
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr_with_signature(
                        structseq_descr_new,
                        crate::gateway::Signature::new(
                            vec!["cls", "sequence", "dict"],
                            None,
                            None,
                            0,
                            1,
                        ),
                    ),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__repr__",
                    crate::make_builtin_function_with_arity("__repr__", structseq_repr, 1),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__str__",
                    crate::make_builtin_function_with_arity("__str__", structseq_repr, 1),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__reduce__",
                    crate::make_builtin_function_with_arity("__reduce__", structseq_reduce, 1),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__replace__",
                    crate::make_builtin_function("__replace__", structseq_replace),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__setattr__",
                    crate::make_builtin_function_with_arity("__setattr__", structseq_setattr, 3),
                )
            };

            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "n_sequence_fields",
                    pyre_object::w_int_new(n_sequence_fields as i64),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "n_fields",
                    pyre_object::w_int_new(n_fields as i64),
                )
            };
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "n_unnamed_fields",
                    pyre_object::w_int_new(n_unnamed_fields as i64),
                )
            };

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
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        fname.as_str(),
                        desc,
                    )
                };
            }

            // `_structseq.py:85-86` — `__match_args__` excludes
            // unnamed (leading-`_`) fields.
            let match_args: Vec<PyObjectRef> = owned_names_for_init
                .iter()
                .filter(|n| !n.starts_with('_'))
                .map(|n| pyre_object::w_str_new(n.as_str()))
                .collect();
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__match_args__",
                    pyre_object::w_tuple_new(match_args),
                )
            };
        },
        tuple_type,
    );

    // Extra fields live in the instance `__dict__`, so the type must
    // advertise `hasdict` for `setdict`/`getdict` to route through the
    // instance-dict side table.
    if has_extra {
        unsafe { pyre_object::typeobject::w_type_set_hasdict(cls, true) };
    }

    {
        structseq_registry().lock().unwrap().insert(
            cls as usize,
            StructSeqDescr {
                name: name.to_string(),
                fields: owned_names,
                extra_fields: owned_extra,
            },
        );
    }

    cls
}

/// `lib_pypy/_structseq.py:43-87 structseqtype.__new__` creates an ordinary
/// heap type through `type.__new__`, even though its instances use tuple
/// storage.  Keep that ownership shape: structseq classes have mutable class
/// dictionaries (and can therefore participate in cycles with instances),
/// while remaining unacceptable as base classes like CPython 3.14 structseqs.
fn make_heap_structseq_type(
    full_name: &str,
    init: impl FnOnce(PyObjectRef),
    base: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let ns = pyre_object::w_dict_new();
    pyre_object::gc_roots::pin_root(ns);
    init(ns);

    let (module, short_name) = full_name
        .rsplit_once('.')
        .map_or((None, full_name), |(module, name)| (Some(module), name));
    if let Some(module) = module {
        let w_module = pyre_object::w_str_new(module);
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "__module__",
                w_module,
            )
        };
    }

    let bases = pyre_object::w_tuple_new(vec![base]);
    pyre_object::gc_roots::pin_root(bases);
    let bases_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let cls = pyre_object::w_type_new(
        short_name,
        pyre_object::gc_roots::shadow_stack_get(bases_slot),
        pyre_object::gc_roots::shadow_stack_get(ns_slot) as *mut u8,
    );
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let cls = pyre_object::gc_roots::shadow_stack_get(cls_slot);
    unsafe {
        let parent_layout = pyre_object::w_type_get_layout_ptr(base);
        pyre_object::w_type_set_layout(cls, parent_layout);
        pyre_object::w_type_set_hasdict(cls, pyre_object::w_type_get_hasdict(base));
        pyre_object::w_type_set_weakrefable(cls, pyre_object::w_type_get_weakrefable(base));
        // CPython's PyStructSequence types set no acceptable-base flag.
        pyre_object::w_type_set_acceptable_as_base_class(cls, false);

        let base_mro = pyre_object::w_type_get_mro(base);
        let mut mro = vec![cls];
        if !base_mro.is_null() {
            mro.extend_from_slice((*base_mro).as_slice());
        } else {
            mro.push(base);
        }
        pyre_object::w_type_set_mro(cls, mro);
        crate::typedef::stamp_new_descr_self(pyre_object::gc_roots::shadow_stack_get(ns_slot), cls);
        pyre_object::typeobject::w_type_ready(cls);
    }
    pyre_object::gc_roots::shadow_stack_get(cls_slot)
}
