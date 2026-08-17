//! _abc module — PyPy: `pypy/module/_abc/`.
//!
//! ABCMeta backing for `abc.py`.  `_abc_instancecheck` /
//! `_abc_subclasscheck` walk `__mro__` for direct inheritance and the
//! per-class `_abc_registry` list populated by `_abc_register` for
//! virtual subclasses.  Mirrors `pypy/module/_abc/app_abc.py`'s
//! `_abc_register` / `_abc_subclasscheck` flow, including its
//! positive/negative caches: without them every check that is not a direct
//! `__mro__` hit re-runs the subclass hook, the registry walk and the
//! `__subclasses__` walk, all recursively, on every single call.

use pyre_object::*;
use std::sync::atomic::{AtomicU64, Ordering};

// `abc_invalidation_counter` (`app_abc.py:47`): bumped by every successful
// `_abc_register` — and by nothing else — and read by `get_cache_token`.  A
// negative cache recorded before a bump no longer describes the registry, so
// `_abc_negative_cache_version` is compared against this on every check.
static INVALIDATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `ref(cls)` as `SimpleWeakSet` spells it (`app_abc.py:20`).  This is the
/// interpreter-level `weakref.ref` object, not [`weakref::w_weakref_new`]'s
/// bare GC struct: the caches are ordinary sets, so an entry has to be a real
/// object with a type — one whose `__hash__` and `__eq__` go by referent, which
/// is what lets a probe find an entry recorded earlier.
///
/// `get_or_make_weakref` returns the one weakref a class already has, so only
/// the first probe of a given class allocates.
///
/// `None` for a class that cannot be weak-referenced at all; the caller reads
/// that as "not cacheable" rather than raising, since the answer to the
/// subclass question does not depend on whether it can be remembered.
fn class_weakref(cls: PyObjectRef) -> Option<PyObjectRef> {
    use crate::module::_weakref::interp__weakref as wr;
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    // Both calls below allocate, and an allocation can claim root slots of its
    // own — so every slot index comes back from the `publish` that made it
    // rather than from arithmetic on an earlier one, and every argument is read
    // out of its slot rather than from a local copy.
    let type_slot = roots.publish(&[wr::weakref_type()]);
    let lifeline = wr::getlifeline(roots.get(cls_slot)).ok()?;
    let lifeline_slot = roots.publish(&[lifeline]);
    Some(wr::get_or_make_weakref(
        roots.get(lifeline_slot),
        roots.get(type_slot),
        roots.get(cls_slot),
    ))
}

/// `app_abc.py:33-38 SimpleWeakSet.__contains__` — `ref(item) in self.data`, a
/// set whose members are weakrefs, probed with a weakref to the same class.
///
/// A missing or non-set attribute reads as "not cached" rather than raising:
/// `_abc_init` installs both caches, but an ABC built before this module (a
/// pickled class, a hand-rolled `ABCMeta` subclass that skips `_abc_init`)
/// has neither, and such a class must still answer subclass checks.
///
/// Takes the class and the attribute name rather than the set itself, so that
/// the set is read only after the probe exists: making the probe allocates, and
/// a reference read across an allocation names where the object used to be.
///
/// `wr in self.data` goes through the membership protocol rather than
/// [`w_set_contains`]: a weakref hashes by running interpreter-level code
/// (`_weakref.ref.__hash__` hashes the referent and memoises the result), and
/// the raw set primitives document that a caller whose element hashes that way
/// owes them a digest taken while the operands are still rooted.
fn weak_cache_contains(
    cls: PyObjectRef,
    name: &str,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let Some(probe) = class_weakref(item) else {
        return Ok(false);
    };
    let probe_slot = roots.publish(&[probe]);
    let cache = cache_attr(roots.get(cls_slot), name);
    if cache.is_null() || !unsafe { is_set(cache) } {
        return Ok(false);
    }
    crate::baseobjspace::contains(cache, roots.get(probe_slot))
}

/// `app_abc.py:39-40 SimpleWeakSet.add` — `self.data.add(ref(item))`.  Silently
/// declines a class with no cache slot, for the same reason
/// [`weak_cache_contains`] reads one as a miss, and calls the set's own `add`
/// for the same reason it uses the membership protocol.
///
/// Upstream's `add` passes `ref()` a callback that discards the entry once the
/// referent dies; this does not, so a checked class that is later collected
/// leaves its spent weakref behind as a member.  Such an entry answers no
/// probe — a weakref compares by referent and this one has none — and it keeps
/// no class alive, so what it costs is the weakref itself.
fn weak_cache_add(cls: PyObjectRef, name: &str, item: PyObjectRef) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let Some(entry) = class_weakref(item) else {
        return Ok(());
    };
    let entry_slot = roots.publish(&[entry]);
    let cache = cache_attr(roots.get(cls_slot), name);
    if cache.is_null() || !unsafe { is_set(cache) } {
        return Ok(());
    }
    let add = crate::baseobjspace::getattr_str(cache, "add")?;
    let add_slot = roots.publish(&[add]);
    crate::call::call_function_impl_result(roots.get(add_slot), &[roots.get(entry_slot)])?;
    Ok(())
}

/// The named cache attribute of `cls`, or null when it has none.  Read fresh
/// at every use: the walks between two reads run arbitrary Python, which can
/// rebind the attribute and can move the set.
fn cache_attr(cls: PyObjectRef, name: &str) -> PyObjectRef {
    match crate::baseobjspace::getattr_str(cls, name) {
        Ok(cache) => cache,
        Err(_) => std::ptr::null_mut(),
    }
}

/// The registry generation `cls`'s negative cache was recorded against.  A
/// class with no version attribute, or one holding something other than an
/// `int`, reports generation 0, which is below every counter value a
/// registration produces — so its negative cache is discarded rather than
/// trusted.
fn negative_cache_version(cls: PyObjectRef) -> u64 {
    let version = cache_attr(cls, "_abc_negative_cache_version");
    if version.is_null() || !unsafe { is_int(version) } {
        return 0;
    }
    unsafe { w_int_get_value(version) }.max(0) as u64
}

// `_py_abc.ABCMeta.__new__` (`_py_abc.py:48`) gives every ABC its OWN
// `_abc_registry`. Create it here as a per-class list so the registry is not
// inherited: without an own entry `register`/`subclass_of` would resolve
// `_abc_registry` up the MRO and share one base class's list across every
// descendant ABC (e.g. Complex/Real/Rational/Integral all collapsing to a
// single registry).
fn abc_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        let fresh = w_list_new(vec![]);
        crate::baseobjspace::setattr_str(cls, "_abc_registry", fresh)?;
        // `app_abc.py:75-77` — the caches are per-class for the same reason the
        // registry is: resolved up the MRO, one base's caches would answer for
        // every descendant ABC, and a hit on `Rational` would satisfy
        // `Integral`.  Each value is built before the call that stores it, so
        // that no allocation happens between reading `cls` and using it.
        let cache = w_set_new();
        crate::baseobjspace::setattr_str(cls, "_abc_cache", cache)?;
        let negative_cache = w_set_new();
        crate::baseobjspace::setattr_str(cls, "_abc_negative_cache", negative_cache)?;
        let version = w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64);
        crate::baseobjspace::setattr_str(cls, "_abc_negative_cache_version", version)?;
        let mut abstract_names = Vec::new();
        let bases = unsafe { w_type_get_bases(cls) };
        if !bases.is_null() && unsafe { is_tuple(bases) } {
            for i in 0..unsafe { w_tuple_len(bases) } {
                let Some(base) = (unsafe { w_tuple_getitem(bases, i as i64) }) else {
                    continue;
                };
                // app_abc.py:67-71 — `getattr(..., set())` defaults only a
                // missing attribute.  A descriptor failure is observable and
                // any iterable, not just a set/frozenset, supplies names.
                let names = match crate::baseobjspace::getattr_str(base, "__abstractmethods__") {
                    Ok(names) => names,
                    Err(err) if err.kind == crate::PyErrorKind::AttributeError => w_set_new(),
                    Err(err) => return Err(err),
                };
                for name in crate::builtins::collect_iterable(names)? {
                    // `_py_abc.py:69` — object-level getattr validates that
                    // every supplied abstract-method name is a string, then
                    // lets descriptors and metaclass attributes provide an
                    // implementation.  Only a missing attribute defaults.
                    let value = match crate::baseobjspace::getattr(cls, name) {
                        Ok(value) => value,
                        Err(err) if err.kind == crate::PyErrorKind::AttributeError => w_none(),
                        Err(err) => return Err(err),
                    };
                    if crate::baseobjspace::isabstractmethod_w(value)? {
                        abstract_names.push(name);
                    }
                }
            }
        }
        let namespace = unsafe { w_type_get_dict_ptr(cls) as PyObjectRef };
        if !namespace.is_null() {
            for (name, value) in unsafe { w_dict_items(namespace) } {
                if crate::baseobjspace::isabstractmethod_w(value)? {
                    abstract_names.push(name);
                }
            }
        }
        let methods = w_frozenset_from_items(&abstract_names);
        crate::baseobjspace::setattr_str(cls, "__abstractmethods__", methods)?;
    }
    Ok(w_none())
}

// `_abc_register` (`_abcmodule.c:_abc__abc_register_impl`) —
// `cls._abc_registry.add(subclass)`.  Pyre stores the registry as a list
// attribute (no WeakSet).
fn register(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "_abc_register() requires (cls, subclass)",
        ));
    }
    let cls = args[0];
    let subclass = args[1];
    // `subclass` must be a class (`PyType_Check`).  Pyre's stdlib stubs
    // register callable non-type shells to ABCs — `_contextvars.Context` is a
    // builtin function here, yet `contextvars` runs `Mapping.register(Context)`
    // at import.  `__subclasscheck__` already tolerates such members by
    // skipping non-type registry entries (see `subclass_of`), so the
    // `PyObject_IsSubclass` guards below (which reject non-type args) only run
    // for real types; a callable stub falls straight through to the append,
    // and only a genuine non-class value (`register(42)`) is rejected.
    if unsafe { is_type(subclass) } {
        // Already a subclass (`PyObject_IsSubclass(subclass, cls) > 0`) —
        // nothing to register.  This also dedups: a previously registered
        // `subclass` resolves through `__subclasscheck__`'s registry walk.
        if crate::baseobjspace::issubclass(subclass, cls)? {
            return Ok(subclass);
        }
        // Registering `subclass` would also make `cls` its subclass.
        if crate::baseobjspace::issubclass(cls, subclass)? {
            return Err(crate::PyError::runtime_error(
                "Refusing to create an inheritance cycle",
            ));
        }
    } else if !crate::baseobjspace::callable_w(subclass) {
        return Err(crate::PyError::type_error("Can only register classes"));
    }
    let registry = match crate::baseobjspace::getattr_str(cls, "_abc_registry") {
        Ok(r) if !unsafe { is_none(r) } => r,
        _ => {
            let fresh = w_list_new(vec![]);
            crate::baseobjspace::setattr_str(cls, "_abc_registry", fresh)?;
            fresh
        }
    };
    unsafe {
        w_list_append(registry, subclass);
    }
    // `app_abc.py:100-101` — invalidate every negative cache.  A class this
    // registration now makes a subclass may already be recorded as a non-match
    // somewhere, and only the counter can reach those entries: they live on
    // arbitrary other ABCs, not on `cls`.
    INVALIDATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    // `app_abc.py:102-105` — an ABC that carries a structural-match marker
    // hands it to the registered class and its descendants
    // (`_internal_set_collection_flag_recursive`).  Registering with
    // `Mapping` / `Sequence` is what makes `case {...}` / `case [...]` accept
    // a class that inherits from neither, so the marker has to travel with
    // the registration, not only with `__abc_tpflags__` at class creation.
    let flag = unsafe { typeobject::w_type_get_flag_map_or_seq(cls) };
    if flag != b'?' && unsafe { is_type(subclass) } {
        set_collection_flag_recursive(subclass, flag);
    }
    Ok(subclass)
}

// `interp_abc.py:15-20 set_collection_flag_recursive` — stamp the marker on
// `w_type` and every class already deriving from it.
fn set_collection_flag_recursive(w_type: PyObjectRef, flag: u8) {
    unsafe {
        // A non-heap type's marker is fixed at registration
        // (`objspace.py:104-108` marks exactly dict / dictproxy / list /
        // tuple), and `Py_TPFLAGS_IMMUTABLETYPE` stops the recursion there.
        // `_collections_abc` runs `Sequence.register(str)` and
        // `ByteString.register(bytes)`, so without this stop `str` / `bytes` /
        // `bytearray` would start matching `case [...]` — the one thing a
        // sequence pattern must never accept.
        //
        // A class already carrying the marker passed it to its descendants at
        // creation (`inherit_flag_map_or_seq`), so that subtree is done.
        if !typeobject::w_type_is_heaptype(w_type)
            || typeobject::w_type_get_flag_map_or_seq(w_type) == flag
        {
            return;
        }
        typeobject::w_type_set_flag_map_or_seq(w_type, flag);
        // `only_real_subclasses` is False for every walk but
        // `descr___subclasses__` (typeobject.py:677-680).
        for child in typeobject::w_type_get_subclasses(w_type, false) {
            set_collection_flag_recursive(child, flag);
        }
    }
}

// `_py_abc.ABCMeta.__subclasscheck__` (`_py_abc.py:108-147`): the caches
// first, then the subclass hook, then a direct `__mro__` test, then the
// recursive registry and subclass walks.  `issubclass` re-dispatches through
// `__subclasscheck__` so a registered or descendant ABC applies its own hook
// in turn — which is also why the caches are load-bearing rather than a
// refinement: an uncached miss re-runs all three walks at every level of that
// recursion, so one `isinstance` against a deep ABC costs a walk of the whole
// ABC graph.
fn subclass_of(cls: PyObjectRef, subclass: PyObjectRef) -> Result<bool, crate::PyError> {
    // _py_abc.py:110-111 — `if not isinstance(subclass, type): raise
    // TypeError('issubclass() arg 1 must be a class')`.  The `__mro__`/registry
    // walks below dereference `subclass` as a type, so a non-type argument
    // (`issubclass({}, ABC)`) must be rejected up front, not read as garbage.
    if !unsafe { is_type(subclass) } {
        return Err(crate::PyError::type_error(
            "issubclass() arg 1 must be a class",
        ));
    }

    // The hook, registry walk, and `__subclasses__` walk below can all run
    // arbitrary Python.  Keep the two arguments live and reload them after
    // every such call, matching the translated livevars carried by PyPy's
    // `ABCMeta.__subclasscheck__`.
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.base();
    roots.pin_root(cls);
    let subclass_slot = cls_slot + 1;
    roots.pin_root(subclass);

    // `app_abc.py:130-131` — a positive hit is final: nothing invalidates it,
    // since `register` can only ever add subclasses.
    if weak_cache_contains(roots.get(cls_slot), "_abc_cache", roots.get(subclass_slot))? {
        return Ok(true);
    }
    // `app_abc.py:132-138` — a negative hit only holds for the registry it was
    // recorded against, so a bumped counter discards the whole cache rather
    // than trusting any entry in it.
    let counter = INVALIDATION_COUNTER.load(Ordering::Relaxed);
    if negative_cache_version(roots.get(cls_slot)) < counter {
        let fresh = w_set_new();
        crate::baseobjspace::setattr_str(roots.get(cls_slot), "_abc_negative_cache", fresh)?;
        let version = w_int_new(counter as i64);
        crate::baseobjspace::setattr_str(
            roots.get(cls_slot),
            "_abc_negative_cache_version",
            version,
        )?;
    } else if weak_cache_contains(
        roots.get(cls_slot),
        "_abc_negative_cache",
        roots.get(subclass_slot),
    )? {
        return Ok(false);
    }

    // Every arm below concludes the same question, so each one only decides the
    // verdict; the single recording site after the block puts it in the matching
    // cache (`app_abc.py:144-163`, which records at each of its own arms).
    let verdict = 'decide: {
        // _py_abc.py:122-130 — `ok = cls.__subclasshook__(subclass)`.
        let hook = crate::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasshook__")?;
        if !hook.is_null() {
            let hook_roots = pyre_object::gc_roots::push_roots();
            let hook_slot = hook_roots.base();
            hook_roots.pin_root(hook);
            let ok = crate::call::call_function_impl_result(
                hook_roots.get(hook_slot),
                &[roots.get(subclass_slot)],
            )?;
            if !unsafe { is_not_implemented(ok) } {
                break 'decide crate::baseobjspace::is_true(ok)?;
            }
        }
        // _py_abc.py:131-134 — direct subclass via `__mro__`.
        unsafe {
            let mro_ptr = w_type_get_mro(roots.get(subclass_slot));
            if !mro_ptr.is_null() {
                for &t in (*mro_ptr).as_slice() {
                    if std::ptr::eq(t, roots.get(cls_slot)) {
                        break 'decide true;
                    }
                }
            }
        }
        // _py_abc.py:135-139 — subclass of a registered class (recursive).
        if let Ok(registry) = crate::baseobjspace::getattr_str(roots.get(cls_slot), "_abc_registry")
            && !registry.is_null()
            && unsafe { is_list(registry) }
        {
            let registry_roots = pyre_object::gc_roots::push_roots();
            let registry_slot = registry_roots.base();
            registry_roots.pin_root(registry);
            let n = unsafe { w_list_len(registry_roots.get(registry_slot)) };
            for i in 0..n {
                if let Some(rcls) =
                    unsafe { w_list_getitem(registry_roots.get(registry_slot), i as i64) }
                {
                    // A registered entry that is not a class cannot be a base
                    // class, so it can never make `subclass` a subclass — skip
                    // it rather than letting `issubclass` raise.  `range` is
                    // registered to `Sequence` but is a builtin function in
                    // pyre, so without this guard a single bad entry aborts the
                    // whole recursive check.
                    if !unsafe { is_type(rcls) } {
                        continue;
                    }
                    let item_roots = pyre_object::gc_roots::push_roots();
                    let rcls_slot = item_roots.base();
                    item_roots.pin_root(rcls);
                    if crate::baseobjspace::issubclass(
                        roots.get(subclass_slot),
                        item_roots.get(rcls_slot),
                    )? {
                        break 'decide true;
                    }
                }
            }
        }
        // _py_abc.py:140-144 — `for scls in cls.__subclasses__():`.  This must go
        // through normal attribute lookup, call, and iteration.  Reading the
        // internal type subclass vector directly hides user overrides and their
        // TypeError/custom exceptions, which are observable ABCMeta semantics.
        let subclasses_method =
            crate::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasses__")?;
        let walk_roots = pyre_object::gc_roots::push_roots();
        let method_slot = walk_roots.base();
        walk_roots.pin_root(subclasses_method);
        let subclasses = crate::call::call_function_impl_result(walk_roots.get(method_slot), &[])?;
        let subclasses_slot = method_slot + 1;
        walk_roots.pin_root(subclasses);
        let iterator = crate::baseobjspace::iter(walk_roots.get(subclasses_slot))?;
        let iterator_slot = subclasses_slot + 1;
        walk_roots.pin_root(iterator);
        loop {
            let scls = match crate::baseobjspace::next(walk_roots.get(iterator_slot)) {
                Ok(scls) => scls,
                Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
                Err(err) => return Err(err),
            };
            let item_roots = pyre_object::gc_roots::push_roots();
            let scls_slot = item_roots.base();
            item_roots.pin_root(scls);
            if crate::baseobjspace::issubclass(roots.get(subclass_slot), item_roots.get(scls_slot))?
            {
                break 'decide true;
            }
        }
        false
    };

    // `app_abc.py:144-163` records at each of its own arms; one site here covers
    // all of them.
    let recorded = if verdict {
        "_abc_cache"
    } else {
        "_abc_negative_cache"
    };
    weak_cache_add(roots.get(cls_slot), recorded, roots.get(subclass_slot))?;
    Ok(verdict)
}

fn instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let cls = args[0];
    let instance = args[1];
    if unsafe { crate::baseobjspace::isinstance_w(instance, cls) } {
        return Ok(w_bool_from(true));
    }
    // `type(instance)` — the instance's real class.  User-defined instances
    // carry the generic layout marker in `ob_type` and the real class in
    // `w_class`, so reading `ob_type` directly would resolve to `object`;
    // `r#type` returns the class for both builtin and user instances.
    let subclass = crate::typedef::r#type(instance).map_or(std::ptr::null_mut(), |p| p.as_ptr());
    if subclass.is_null() {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(cls, subclass)?))
}

fn subclasscheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(subclass_of(args[0], args[1])?))
}

/// `_abc._reset_registry(cls)`: clear only this ABC's virtual-subclass
/// registry.  ABCMeta exposes it as `_abc_registry_clear`; a no-op leaks
/// registrations between independent users and test cases.
/// The invalidation counter stays put: only `_abc_register` advances it, so
/// an outstanding `get_cache_token` survives a registry reset.
fn reset_registry(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        crate::baseobjspace::setattr_str(cls, "_abc_registry", w_list_new(vec![]))?;
    }
    Ok(w_none())
}

/// `_abc._reset_caches(cls)` (`app_abc.py:188-191`): empty both of this ABC's
/// caches, leaving the registry and the invalidation counter untouched — a
/// cleared cache is answered by re-running the walks, which is not a change of
/// answer, so no token needs to expire.
///
/// Cleared in place rather than rebound, so anything already holding the set
/// sees the clear.
fn reset_caches(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        for name in ["_abc_cache", "_abc_negative_cache"] {
            let cache = cache_attr(cls, name);
            if !cache.is_null() && unsafe { is_set(cache) } {
                unsafe { w_set_clear(cache) };
            }
        }
    }
    Ok(w_none())
}

crate::py_module! {
    "_abc",
    functions: {
        "get_cache_token"     / 0 = |_| Ok(w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64)),
        "_abc_init"           / 1 = abc_init,
        "_abc_register"       / 2 = register,
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = |_| Ok(w_tuple_new(vec![])),
        "_reset_registry"     / 1 = reset_registry,
        "_reset_caches"       / 1 = reset_caches,
    },
}
