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

// `abc_invalidation_counter` (`app_abc.py`): bumped by every successful
// `_abc_register` — and by nothing else — and read by `get_cache_token`.  A
// negative cache recorded before a bump no longer describes the registry, so
// `_abc_negative_cache_version` is compared against this on every check.
static INVALIDATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The app-level `SimpleWeakSet` (`app_abc.py`), stashed at module init
/// the way `weakref_type` stashes its own.  The registry and both caches are
/// instances of it, so the collection this module installs is the one
/// `_get_dump` describes and a collected member drops itself.
static SIMPLE_WEAK_SET_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn simple_weak_set_type() -> PyObjectRef {
    *SIMPLE_WEAK_SET_TYPE
        .get()
        .expect("_abc.SimpleWeakSet must be installed at module init") as PyObjectRef
}

/// `SimpleWeakSet()` — the empty collection `_abc_init` installs and the
/// invalidation in `subclass_of` rebinds to.
fn new_simple_weak_set() -> Result<PyObjectRef, crate::PyError> {
    crate::call::call_function_impl_result(simple_weak_set_type(), &[])
}

/// `app_abc.py SimpleWeakSet.__contains__` through the membership
/// protocol, which is where the weakref probe and the `TypeError` fallback
/// live.
///
fn weak_cache_contains(
    cls: PyObjectRef,
    name: &str,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let item_slot = roots.publish(&[item]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    crate::baseobjspace::contains(roots.get(cache_slot), roots.get(item_slot))
}

/// `app_abc.py SimpleWeakSet.add` — `self.data.add(ref(item, self._remove))`.
/// Called rather than open-coded so the entry carries the callback that
/// discards it once the referent dies; a bare `ref` would leave a spent one
/// behind for every class the check ever saw.
///
/// Every item reaching here is a
/// class — `register` and `subclass_of` each reject a non-type argument — and a
/// class is weak-referenceable, so `add` needs no test of its own, which is
/// also why `app_abc.py add` has none.
fn weak_cache_add(cls: PyObjectRef, name: &str, item: PyObjectRef) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let item_slot = roots.publish(&[item]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    let add = crate::baseobjspace::getattr_str(roots.get(cache_slot), "add")?;
    let add_slot = roots.publish(&[add]);
    crate::call::call_function_impl_result(roots.get(add_slot), &[roots.get(item_slot)])?;
    Ok(())
}

/// `SimpleWeakSet.clear` (`app_abc.py`) on the named collection, in
/// place, so anything already holding it sees the clear.
fn weak_cache_clear(cls: PyObjectRef, name: &str) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let cache = cache_attr(roots.get(cls_slot), name)?;
    let cache_slot = roots.publish(&[cache]);
    let clear = crate::baseobjspace::getattr_str(roots.get(cache_slot), "clear")?;
    let clear_slot = roots.publish(&[clear]);
    crate::call::call_function_impl_result(roots.get(clear_slot), &[])?;
    Ok(())
}

/// The named collection attribute of `cls`, or null when it has none.  Read
/// fresh at every use: the walks between two reads run arbitrary Python, which
/// can rebind the attribute and can move the object.
///
/// `app_abc.py:110` reads the slot as a plain attribute, so a class that never
/// ran `_abc_init` raises `AttributeError` out of the check rather than being
/// answered as a class with an empty cache.  A metaclass hook that raises
/// something else propagates unchanged.
fn cache_attr(cls: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(cls, name)
}

/// Compare `cls._abc_negative_cache_version` against the invalidation counter
/// the way `_abc_subclasscheck` (`<`) and `_abc_instancecheck` (`==`) do.
///
/// The stored value is passed to the comparison as it stands rather than
/// normalised to an integer first: a rebound non-int raises from the compare,
/// and a value with its own `__lt__` / `__eq__` stays observable.
fn negative_cache_version_compare(
    cls: PyObjectRef,
    op: crate::objspace::descroperation::CompareOp,
) -> Result<bool, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let version = cache_attr(roots.get(cls_slot), "_abc_negative_cache_version")?;
    let version_slot = roots.publish(&[version]);
    let counter_slot = roots.publish(&[w_int_new(
        INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64
    )]);
    crate::baseobjspace::is_true(crate::objspace::descroperation::compare(
        roots.get(version_slot),
        roots.get(counter_slot),
        op,
    )?)
}

/// `app_abc.py _abc_init` — install the three collections and the
/// negative-cache generation, then compute `__abstractmethods__`.
fn abc_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        // `app_abc.py:74-77` — registry and both caches are per-class for the
        // same reason: resolved up the MRO, one base's would answer for every
        // descendant ABC, and a hit on `Rational` would satisfy `Integral`.
        // Each value is built before the call that stores it, so that no
        // allocation happens between reading `cls` and using it.
        for name in ["_abc_registry", "_abc_cache", "_abc_negative_cache"] {
            let fresh = new_simple_weak_set()?;
            crate::baseobjspace::setattr_str(cls, name, fresh)?;
        }
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
    // `_abc_register`: `if not isinstance(subclass, type): raise TypeError(
    // "Can only register classes")`.  Everything downstream reads a registry
    // entry as a class, so the rejection is what makes that sound.
    if !unsafe { is_type(subclass) } {
        return Err(crate::PyError::type_error("Can only register classes"));
    }
    // Already a subclass (`issubclass(subclass, cls)`) — nothing to register.
    // This also dedups: a previously registered `subclass` resolves through
    // `__subclasscheck__`'s registry walk.
    if crate::baseobjspace::issubclass(subclass, cls)? {
        return Ok(subclass);
    }
    // Registering `subclass` would also make `cls` its subclass.  Tested after
    // the arm above, so `X.register(X)` stays a no-op rather than a cycle.
    if crate::baseobjspace::issubclass(cls, subclass)? {
        return Err(crate::PyError::runtime_error(
            "Refusing to create an inheritance cycle",
        ));
    }
    // `app_abc.py cls._abc_registry.add(subclass)`.
    weak_cache_add(cls, "_abc_registry", subclass)?;
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
    // `_abc_register` reads the flags under `if isinstance(cls, type):`.
    if unsafe { is_type(cls) } {
        let flag = unsafe { typeobject::w_type_get_flag_map_or_seq(cls) };
        if flag != b'?' {
            set_collection_flag_recursive(subclass, flag);
        }
    }
    Ok(subclass)
}

// `interp_abc.py set_collection_flag_recursive` — stamp the marker on
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
        // `descr___subclasses__` (typeobject.py).
        for child in typeobject::w_type_get_subclasses(w_type, false) {
            set_collection_flag_recursive(child, flag);
        }
    }
}

// `_py_abc.ABCMeta.__subclasscheck__` (`_py_abc.py`): the caches
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
    if negative_cache_version_compare(
        roots.get(cls_slot),
        crate::objspace::descroperation::CompareOp::Lt,
    )? {
        let fresh = new_simple_weak_set()?;
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
        // _py_abc.py — `ok = cls.__subclasshook__(subclass)`.
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
        // _py_abc.py — direct subclass via `__mro__`.
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
        // `app_abc.py for rcls in cls._abc_registry:` — subclass of a
        // registered class (recursive).  `SimpleWeakSet.__iter__` copies the
        // set before yielding and skips entries whose referent is gone, so the
        // walk survives a collection that fires the discard callback partway
        // through it.
        let registry = cache_attr(roots.get(cls_slot), "_abc_registry")?;
        {
            let registry_roots = pyre_object::gc_roots::push_roots();
            let registry_slot = registry_roots.base();
            registry_roots.pin_root(registry);
            let iterator = crate::baseobjspace::iter(registry_roots.get(registry_slot))?;
            let iterator_slot = registry_slot + 1;
            registry_roots.pin_root(iterator);
            loop {
                let rcls = match crate::baseobjspace::next(registry_roots.get(iterator_slot)) {
                    Ok(rcls) => rcls,
                    Err(err) if err.matches_stop_iteration() => break,
                    Err(err) => return Err(err),
                };
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
        // _py_abc.py — `for scls in cls.__subclasses__():`.  This must go
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
                Err(err) if err.matches_stop_iteration() => break,
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

/// `_abc_instancecheck` (`app_abc.py`).
///
/// The two classes are asked separately because they can differ: `__class__`
/// is an ordinary attribute an object may answer with something other than its
/// real type, and a proxy that does so is meant to pass the check for what it
/// claims to be.  Reading only `type(instance)` would ignore the claim;
/// reading only `__class__` would let it deny the real one.  A positive cache
/// hit on the claimed class is taken before the real type is even read, which
/// is the whole point of inlining the cache check here rather than leaving it
/// to `__subclasscheck__`.
fn instancecheck(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_bool_from(false));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[args[0]]);
    let instance_slot = roots.publish(&[args[1]]);

    // `app_abc.py subclass = instance.__class__`.
    let subclass = crate::baseobjspace::getattr_str(roots.get(instance_slot), "__class__")?;
    let subclass_slot = roots.publish(&[subclass]);
    if weak_cache_contains(roots.get(cls_slot), "_abc_cache", roots.get(subclass_slot))? {
        return Ok(w_bool_from(true));
    }

    // `app_abc.py subtype = type(instance)` — the instance's real class.
    // User-defined instances carry the generic layout marker in `ob_type` and
    // the real class in `w_class`, so reading `ob_type` directly would resolve
    // to `object`; `r#type` returns the class for both builtin and user
    // instances.
    let subtype = crate::typedef::r#type(roots.get(instance_slot))
        .map_or(std::ptr::null_mut(), |p| p.as_ptr());
    if subtype.is_null() {
        return Ok(w_bool_from(false));
    }
    let subtype_slot = roots.publish(&[subtype]);

    if std::ptr::eq(roots.get(subtype_slot), roots.get(subclass_slot)) {
        // `app_abc.py:115-117` — one class, so the negative cache can answer.
        // The version test is `==`, not `<`: a cache recorded against a
        // *later* counter than the one read here cannot describe this
        // registry either.
        if negative_cache_version_compare(
            roots.get(cls_slot),
            crate::objspace::descroperation::CompareOp::Eq,
        )? && weak_cache_contains(
            roots.get(cls_slot),
            "_abc_negative_cache",
            roots.get(subclass_slot),
        )? {
            return Ok(w_bool_from(false));
        }
        return Ok(w_bool_from(subclasscheck_of(
            roots.get(cls_slot),
            roots.get(subclass_slot),
        )?));
    }
    // `app_abc.py any(cls.__subclasscheck__(c) for c in (subclass, subtype))`.
    for slot in [subclass_slot, subtype_slot] {
        if subclasscheck_of(roots.get(cls_slot), roots.get(slot))? {
            return Ok(w_bool_from(true));
        }
    }
    Ok(w_bool_from(false))
}

/// `cls.__subclasscheck__(subclass)` through attribute lookup, the way
/// `app_abc.py:118` and `:121` spell it, so an `ABCMeta` subclass that
/// overrides the hook is the one that answers.
fn subclasscheck_of(cls: PyObjectRef, subclass: PyObjectRef) -> Result<bool, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    let subclass_slot = roots.publish(&[subclass]);
    let check = crate::baseobjspace::getattr_str(roots.get(cls_slot), "__subclasscheck__")?;
    let check_slot = roots.publish(&[check]);
    let result =
        crate::call::call_function_impl_result(roots.get(check_slot), &[roots.get(subclass_slot)])?;
    let result_slot = roots.publish(&[result]);
    crate::baseobjspace::is_true(roots.get(result_slot))
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
        weak_cache_clear(cls, "_abc_registry")?;
    }
    Ok(w_none())
}

/// `_abc._reset_caches(cls)` (`app_abc.py`): empty both of this ABC's
/// caches, leaving the registry and the invalidation counter untouched — a
/// cleared cache is answered by re-running the walks, which is not a change of
/// answer, so no token needs to expire.
///
/// Cleared in place rather than rebound, so anything already holding the set
/// sees the clear.
fn reset_caches(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if let Some(&cls) = args.first() {
        for name in ["_abc_cache", "_abc_negative_cache"] {
            weak_cache_clear(cls, name)?;
        }
    }
    Ok(w_none())
}

/// `_abc._get_dump(cls)` (`app_abc.py`): shallow copies of the
/// registry, both caches, and the negative-cache version.  The three sets are
/// the collections' own `data`, which is why they are `SimpleWeakSet`s and not
/// bare sets — `ABC._dump_registry` prints what this returns.
fn get_dump(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&cls) = args.first() else {
        return Ok(w_tuple_new(vec![]));
    };
    let roots = pyre_object::gc_roots::push_roots();
    let cls_slot = roots.publish(&[cls]);
    // `app_abc.py _get_dump` builds the four-tuple as a single expression, so
    // every `.data` it reads stays a live variable that the later reads reload.
    // Keep the slots rather than the raw pointers: `data` is whatever
    // `cache.data` answers, so it can be a list or a dict — the two kinds a
    // minor collection moves — and each further `cache_attr` / `getattr_str`
    // runs the descriptor protocol.
    let mut data_slots = Vec::with_capacity(3);
    for name in ["_abc_registry", "_abc_cache", "_abc_negative_cache"] {
        let cache = cache_attr(roots.get(cls_slot), name)?;
        let cache_slot = roots.publish(&[cache]);
        let data = crate::baseobjspace::getattr_str(roots.get(cache_slot), "data")?;
        data_slots.push(roots.publish(&[data]));
    }
    // `_get_dump` reports the generation attribute itself, not a number derived
    // from it.  The last read that can run Python; take it before the reloads
    // below so they answer with final addresses.
    let version = cache_attr(roots.get(cls_slot), "_abc_negative_cache_version")?;
    let mut items: Vec<PyObjectRef> = data_slots.iter().map(|&slot| roots.get(slot)).collect();
    items.push(version);
    Ok(w_tuple_new(items))
}

crate::py_module! {
    "_abc",
    functions: {
        "get_cache_token"     / 0 = |_| Ok(w_int_new(INVALIDATION_COUNTER.load(Ordering::Relaxed) as i64)),
        "_abc_init"           / 1 = abc_init,
        "_abc_register"       / 2 = register,
        "_abc_instancecheck"  / 2 = instancecheck,
        "_abc_subclasscheck"  / 2 = subclasscheck,
        "_get_dump"           / 1 = get_dump,
        "_reset_registry"     / 1 = reset_registry,
        "_reset_caches"       / 1 = reset_caches,
    },
    extra_init: |ns| {
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_abc.py"),
            "app_abc.py",
            &["SimpleWeakSet"],
            &[],
        );
        let simple_weak_set = crate::module_ns_get(ns, "SimpleWeakSet")
            .expect("_abc.SimpleWeakSet must be installed by appleveldefs");
        let _ = SIMPLE_WEAK_SET_TYPE.set(simple_weak_set as usize);
    },
}
