//! Pickle reduce protocol for `object` — PyPy: `pypy/objspace/std/objectobject.py`.
//!
//! The app-level helpers `reduce_1` / `reduce_2` / `get_slotvalues` /
//! `slotnames` (objectobject.py:23-84) are bundled in
//! `reduce_protocol_app.py` and resolved lazily through
//! `appleveldef_install` into a rooted GC module dict.  The three
//! handles the interp-level code calls (`reduce_1`, `reduce_2`,
//! `get_slotvalues`) keep that namespace as their `__globals__`, so
//! `get_slotvalues` can still reach its sibling `slotnames`.
//!
//! The interp-level descriptors `descr_reduce` / `descr_reduce_ex` /
//! `object_getstate_default` / `getnewargs` mirror objectobject.py's
//! `descr__reduce__` / `descr__reduce_ex__` / `object_getstate_default`
//! / `_getnewargs`.

use pyre_object::PyObjectRef;

use crate::PyResult;
use crate::error::PyError;

/// Resolved app-level handles, indexed `[reduce_1, reduce_2,
/// get_slotvalues]`.  PyPy resolves these once via `app.interphook`
/// (objectobject.py:88-90); pyre resolves them on first use into a
/// thread-local cache.
const REDUCE_1: usize = 0;
const REDUCE_2: usize = 1;
const GET_SLOTVALUES: usize = 2;

static HANDLES: std::sync::Mutex<[usize; 3]> = std::sync::Mutex::new([0; 3]);

/// Resolve (and cache) the three app-level handles.
///
/// Executes `reduce_protocol_app.py` into its own fresh module globals and
/// stores the selected handles in process-global GC root slots.  The functions
/// retain the execution namespace as their `__globals__`, which keeps
/// `slotnames` reachable from `get_slotvalues` even though only three names
/// are surfaced.
fn handle(which: usize) -> PyObjectRef {
    let cached = HANDLES.lock().unwrap()[which];
    if cached != 0 {
        return cached as PyObjectRef;
    }

    // Do not hold HANDLES while executing Python: applevel installation can
    // collect, and the root walker below must be able to lock the slots.  The
    // shadow root keeps the namespace/functions alive until the finished
    // array is published.  A racing initializer may duplicate this work, but
    // publication remains atomic under the mutex and either complete set is
    // equivalent.
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("reduce_protocol: no execution context");
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
    pyre_object::gc_roots::pin_root(w_app_globals);
    crate::importing::appleveldef_install(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        include_str!("reduce_protocol_app.py"),
        "reduce_protocol_app.py",
        &["reduce_1", "reduce_2", "get_slotvalues"],
    );
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
    let get = |name: &str| {
        unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) }
            .unwrap_or_else(|| panic!("reduce_protocol: `{name}` not bound"))
    };
    let initialized = [
        get("reduce_1") as usize,
        get("reduce_2") as usize,
        get("get_slotvalues") as usize,
    ];
    let mut handles = HANDLES.lock().unwrap();
    if handles[which] == 0 {
        *handles = initialized;
    }
    handles[which] as PyObjectRef
}

/// RPython's GC transform treats the app-level interphook cache as a normal
/// object graph.  Pyre stores the equivalent shared handles outside the GC,
/// so expose each slot to the global root walker and write relocated pointers
/// back in place.
pub(crate) fn walk_handle_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut handles = HANDLES.lock().unwrap();
    for slot in handles.iter_mut().filter(|slot| **slot != 0) {
        unsafe { visitor(&mut *(slot as *mut usize as *mut majit_ir::GcRef)) };
    }
}

/// `%T`-style class name of `w_obj` for error messages.
fn typename(w_obj: PyObjectRef) -> String {
    match crate::typedef::r#type(w_obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp) }.to_string(),
        None => "object".to_string(),
    }
}

/// objectobject.py:53 `get_slotvalues(obj)` — app-level handle.
pub fn get_slotvalues(w_obj: PyObjectRef) -> PyResult {
    crate::call::call_function_impl_result(handle(GET_SLOTVALUES), &[w_obj])
}

/// objectobject.py:245 `object_getstate_default(space, w_obj, required)`.
///
/// `required` is always 0 from both callers (`descr__getstate__` and the
/// proto>=2 path), so the variable-sized raise is unreachable and pyre
/// has no `variable_sized` layout notion to gate it on — the structure
/// is ported without that dead arm.
pub fn object_getstate_default(w_obj: PyObjectRef) -> PyResult {
    let w_objdict = crate::baseobjspace::findattr(w_obj, "__dict__");
    let mut w_ret = match w_objdict {
        Some(d) if crate::baseobjspace::len_w(d)? > 0 => {
            // Copy `__dict__`. For a dict subclass, drop the internal
            // `__dict_data__` key it keeps its mapping payload under: that
            // payload is reconstructed through `dictitems`, not the instance
            // state, so an attribute-less dict subclass yields `None` here
            // (matching the empty instance `__dict__` of a built-in subclass).
            // A non-dict instance keeps a user attribute of that name.
            let is_dict_inst = {
                let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
                unsafe { crate::baseobjspace::isinstance_w(w_obj, w_dict_type) }
            };
            let w_copy = pyre_object::w_dict_new();
            let mut count = 0usize;
            for (k, v) in unsafe { pyre_object::w_dict_items(d) } {
                if is_dict_inst
                    && unsafe { pyre_object::is_str(k) }
                    && unsafe { pyre_object::w_str_get_value(k) } == "__dict_data__"
                {
                    continue;
                }
                unsafe { pyre_object::w_dict_store(w_copy, k, v) };
                count += 1;
            }
            if count > 0 {
                w_copy
            } else {
                pyre_object::w_none()
            }
        }
        _ => pyre_object::w_none(),
    };
    let w_slots = get_slotvalues(w_obj)?;
    if !unsafe { pyre_object::is_none(w_slots) } {
        w_ret = pyre_object::w_tuple_new(vec![w_ret, w_slots]);
    }
    Ok(w_ret)
}

/// objectobject.py:201 `_getnewargs(space, w_obj)` — returns
/// `(hasargs, w_args, w_kwargs)`.
pub fn getnewargs(w_obj: PyObjectRef) -> Result<(bool, PyObjectRef, PyObjectRef), PyError> {
    let w_descr = unsafe { crate::baseobjspace::lookup(w_obj, "__getnewargs_ex__") };
    let hasargs;
    let w_args;
    let w_kwargs;
    if let Some(w_descr) = w_descr {
        let w_result = crate::call::call_function_impl_result(w_descr, &[w_obj])?;
        if !unsafe { pyre_object::is_tuple(w_result) } {
            return Err(PyError::type_error(format!(
                "__getnewargs_ex__ should return a tuple, not '{}'",
                typename(w_result)
            )));
        }
        let n = unsafe { pyre_object::w_tuple_len(w_result) };
        if n != 2 {
            return Err(PyError::value_error(format!(
                "__getnewargs_ex__ should return a tuple of length 2, not {n}"
            )));
        }
        let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(w_result) };
        let wa = items[0];
        let wk = items[1];
        if !unsafe { pyre_object::is_tuple(wa) } {
            return Err(PyError::type_error(format!(
                "first item of the tuple returned by __getnewargs_ex__ must be a tuple, not '{}'",
                typename(wa)
            )));
        }
        if !unsafe { pyre_object::is_dict(wk) } {
            return Err(PyError::type_error(format!(
                "second item of the tuple returned by __getnewargs_ex__ must be a dict, not '{}'",
                typename(wk)
            )));
        }
        hasargs = true;
        w_args = wa;
        w_kwargs = wk;
    } else {
        let w_descr = unsafe { crate::baseobjspace::lookup(w_obj, "__getnewargs__") };
        if let Some(w_descr) = w_descr {
            let wa = crate::call::call_function_impl_result(w_descr, &[w_obj])?;
            if !unsafe { pyre_object::is_tuple(wa) } {
                return Err(PyError::type_error(format!(
                    "__getnewargs__ should return a tuple, not '{}'",
                    typename(wa)
                )));
            }
            hasargs = true;
            w_args = wa;
        } else {
            hasargs = false;
            w_args = pyre_object::w_tuple_new(vec![]);
        }
        w_kwargs = pyre_object::w_none();
    }
    Ok((hasargs, w_args, w_kwargs))
}

/// objectobject.py:240 `descr__reduce__(space, w_obj)` — `reduce_1(obj, 0)`.
pub fn descr_reduce(w_obj: PyObjectRef) -> PyResult {
    reduce_1(w_obj, 0)
}

/// setobject.c `set_reduce` — `(type(self), (list(self),), state)` where
/// `state` is `_PyObject_GetState(self)` (`object_getstate_default`).  Both
/// `set` and `frozenset` expose this as `__reduce__`, so a subclass with a
/// payload round-trips through `copy`/`pickle` instead of collapsing to the
/// base type.
pub fn set_reduce(w_obj: PyObjectRef) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_type = crate::typedef::r#type(w_obj)
        .ok_or_else(|| PyError::type_error("cannot determine type for __reduce__"))?;
    pyre_object::gc_roots::pin_root(w_type);
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // PySequence_List(self): no GC between reading the items and `w_list_new`
    // (which re-pins them).
    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    };
    let w_list = pyre_object::listobject::w_list_new(items);
    let w_args = pyre_object::w_tuple_new(vec![w_list]);
    pyre_object::gc_roots::pin_root(w_args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_state = object_getstate_default(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;

    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        pyre_object::gc_roots::shadow_stack_get(args_slot),
        w_state,
    ]))
}

/// objectobject.py:23 `reduce_1(obj, proto)` — app-level handle.
fn reduce_1(w_obj: PyObjectRef, proto: i64) -> PyResult {
    crate::call::call_function_impl_result(
        handle(REDUCE_1),
        &[w_obj, pyre_object::w_int_new(proto)],
    )
}

/// objectobject.py:27 `reduce_2(obj, proto, args, kwargs)` — app-level handle.
fn reduce_2(
    w_obj: PyObjectRef,
    proto: i64,
    w_args: PyObjectRef,
    w_kwargs: PyObjectRef,
) -> PyResult {
    crate::call::call_function_impl_result(
        handle(REDUCE_2),
        &[w_obj, pyre_object::w_int_new(proto), w_args, w_kwargs],
    )
}

/// objectobject.py:260 `descr__reduce_ex__(space, w_obj, proto)`.
pub fn descr_reduce_ex(w_obj: PyObjectRef, proto: i64) -> PyResult {
    // Honour a user `__reduce__` override:
    // `type(obj).__reduce__ is not object.__reduce__`.
    let w_reduce = crate::baseobjspace::findattr(w_obj, "__reduce__");
    if let Some(w_reduce) = w_reduce {
        let w_type = crate::typedef::r#type(w_obj)
            .ok_or_else(|| PyError::type_error("cannot determine type for __reduce_ex__"))?;
        let w_cls_reduce = crate::baseobjspace::getattr_str(w_type, "__reduce__")?;
        let w_obj_reduce =
            crate::baseobjspace::getattr_str(crate::typedef::w_object(), "__reduce__")?;
        let mut override_ = !crate::baseobjspace::is_w(w_cls_reduce, w_obj_reduce);
        // Built-in types (range, the iterators) expose `__reduce__`
        // through instance dispatch rather than the type MRO, so the
        // type-level comparison above sees `object.__reduce__` and
        // misses the override.  Compare the bound instance method's
        // `__func__` against `object.__reduce__` to catch it: a genuine
        // override binds a different function.
        if !override_ {
            let w_inst_func = if unsafe { pyre_object::function::is_method(w_reduce) } {
                unsafe { pyre_object::function::w_method_get_func(w_reduce) }
            } else {
                w_reduce
            };
            override_ = !crate::baseobjspace::is_w(w_inst_func, w_obj_reduce);
        }
        if override_ {
            return crate::call::call_function_impl_result(w_reduce, &[]);
        }
    }
    if proto >= 2 {
        let (hasargs, w_args, w_kwargs) = getnewargs(w_obj)?;
        // objectobject.py:276 / `_PyObject_GetState(required)`: a type whose
        // instances carry C-level state that `__dict__`/`__slots__` cannot
        // reconstruct, and that supplies no `__getnewargs__`, cannot be
        // rebuilt via `__newobj__`.  `reduce_newobj` gates this on
        // `tp_basicsize` exceeding the object+dict+weakref+slots baseline;
        // pyre has no basicsize notion, so it recognises the one such builtin
        // layout
        // that reaches object-reduce — `module` (native name + dict
        // payload).  Matches the proto < 2 `copyreg._reduce_ex` refusal.
        if !hasargs && unsafe { pyre_object::is_module(w_obj) } {
            return Err(PyError::type_error(format!(
                "cannot pickle '{}' object",
                typename(w_obj)
            )));
        }
        return reduce_2(w_obj, proto, w_args, w_kwargs);
    }
    reduce_1(w_obj, proto)
}
