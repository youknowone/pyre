//! pypy/module/_weakref/interp__weakref.py
//!
//! Structural port of `WeakrefLifeline`, `W_WeakrefBase`, `W_Weakref`,
//! `W_AbstractProxy`, `W_Proxy`, `W_CallableProxy`, plus the module-level
//! `getlifeline`, `descr__new__weakref`, `proxy`, `force`, etc.
//!
//! pyre has no GC so the underlying weak reference is kept as a strong
//! pointer; the class hierarchy and method names mirror PyPy so that
//! `__weakref__` and `_weakref.{ref,proxy,...}` go through the same
//! code paths as the original.

#![allow(non_camel_case_types, non_snake_case)]

use crate::{PyError, PyNamespace, make_builtin_function, namespace_store};
use pyre_object::*;

use std::sync::OnceLock;

thread_local! {
    static WEAKREF_LIFELINE_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static WEAKREF_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static PROXY_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    static CALLABLE_PROXY_TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
}

// ── Instance attribute names ──────────────────────────────────────────
//
// PyPy stores fields directly on the W_Root subclass. pyre stores them
// as instance attributes on the corresponding hasdict type so that the
// JIT and getattr/setattr paths see the same shape they would for any
// user instance. The names mirror PyPy field names exactly.

const ATTR_CACHED_WEAKREF: &str = "cached_weakref";
const ATTR_CACHED_PROXY: &str = "cached_proxy";
const ATTR_W_OBJ_WEAK: &str = "w_obj_weak";
const ATTR_W_CALLABLE: &str = "w_callable";
const ATTR_W_HASH: &str = "w_hash";

fn read_attr(obj: PyObjectRef, name: &str) -> PyObjectRef {
    crate::baseobjspace::getattr(obj, name)
        .ok()
        .filter(|v| !v.is_null() && unsafe { !pyre_object::is_none(*v) })
        .unwrap_or(PY_NULL)
}

fn write_attr(obj: PyObjectRef, name: &str, value: PyObjectRef) {
    let _ = crate::baseobjspace::setattr(obj, name, value);
}

// ── Type registration ─────────────────────────────────────────────────

fn weakref_lifeline_type() -> PyObjectRef {
    WEAKREF_LIFELINE_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("WeakrefLifeline", |_| {});
            unsafe { pyre_object::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:270-280 W_Weakref.typedef
///
/// ```python
/// W_Weakref.typedef = TypeDef("weakref",
///     __doc__ = """A weak reference to an object 'obj'.  A 'callback' can be given,
/// which is called with 'obj' as an argument when it is about to be finalized.""",
///     __new__ = interp2app(descr__new__weakref),
///     __init__ = interp2app(W_Weakref.descr__init__weakref),
///     __eq__ = interp2app(W_Weakref.descr__eq__),
///     __ne__ = interp2app(W_Weakref.descr__ne__),
///     __hash__ = interp2app(W_Weakref.descr_hash),
///     __call__ = interp2app(W_Weakref.descr_call),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
/// )
/// ```
fn init_weakref_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__weakref_typecall),
    );
    namespace_store(
        ns,
        "__init__",
        make_builtin_function("__init__", descr__init__weakref),
    );
    namespace_store(ns, "__eq__", make_builtin_function("__eq__", descr__eq__));
    namespace_store(ns, "__ne__", make_builtin_function("__ne__", descr__ne__));
    namespace_store(
        ns,
        "__hash__",
        make_builtin_function("__hash__", descr_hash),
    );
    namespace_store(
        ns,
        "__call__",
        make_builtin_function("__call__", descr_call),
    );
    namespace_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", descr__repr__),
    );
}

pub fn weakref_type() -> PyObjectRef {
    WEAKREF_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("weakref", init_weakref_type);
            unsafe { pyre_object::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:405-410 W_Proxy.typedef
///
/// ```python
/// W_Proxy.typedef = TypeDef("weakproxy",
///     __new__ = interp2app(descr__new__proxy),
///     __hash__ = interp2app(W_Proxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     **proxy_typedef_dict)
/// W_Proxy.typedef.acceptable_as_base_class = False
/// ```
fn init_proxy_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__proxy),
    );
    namespace_store(
        ns,
        "__hash__",
        make_builtin_function("__hash__", proxy_descr__hash__),
    );
    namespace_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", descr__repr__),
    );
}

pub fn proxy_type() -> PyObjectRef {
    PROXY_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("weakproxy", init_proxy_type);
            unsafe {
                pyre_object::w_type_set_hasdict(tp, true);
                pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    })
}

/// pypy/module/_weakref/interp__weakref.py:412-418 W_CallableProxy.typedef
///
/// ```python
/// W_CallableProxy.typedef = TypeDef("weakcallableproxy",
///     __new__ = interp2app(descr__new__callableproxy),
///     __hash__ = interp2app(W_CallableProxy.descr__hash__),
///     __repr__ = interp2app(W_WeakrefBase.descr__repr__),
///     __call__ = interp2app(W_CallableProxy.descr__call__),
///     **callable_proxy_typedef_dict)
/// W_CallableProxy.typedef.acceptable_as_base_class = False
/// ```
fn init_callable_proxy_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", descr__new__callableproxy),
    );
    namespace_store(
        ns,
        "__hash__",
        make_builtin_function("__hash__", proxy_descr__hash__),
    );
    namespace_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", descr__repr__),
    );
    namespace_store(
        ns,
        "__call__",
        make_builtin_function("__call__", callable_proxy_descr__call__),
    );
}

pub fn callable_proxy_type() -> PyObjectRef {
    CALLABLE_PROXY_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            let tp =
                crate::typedef::make_builtin_type("weakcallableproxy", init_callable_proxy_type);
            unsafe {
                pyre_object::w_type_set_hasdict(tp, true);
                pyre_object::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    })
}

// ── WeakrefLifeline ───────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:19-153
//
// class WeakrefLifeline(W_Root):
//     cached_weakref  = None
//     cached_proxy    = None
//     other_refs_weak = None
//     has_callbacks   = False
//
// pyre stores cached_weakref/cached_proxy as instance attributes on a
// W_InstanceObject of type WeakrefLifeline so the field access goes
// through the same setattr/getattr path as any user instance.

/// pypy/module/_weakref/interp__weakref.py:27-28 WeakrefLifeline.__init__
///
/// ```python
/// def __init__(self, space):
///     self.space = space
/// ```
pub fn weakref_lifeline_new() -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(weakref_lifeline_type());
    write_attr(obj, ATTR_CACHED_WEAKREF, pyre_object::w_none());
    write_attr(obj, ATTR_CACHED_PROXY, pyre_object::w_none());
    obj
}

/// pypy/module/_weakref/interp__weakref.py:60-77 get_or_make_weakref
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_weakref(self, w_subtype, w_obj):
///     space = self.space
///     w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///     #
///     if space.is_w(w_weakreftype, w_subtype):
///         if self.cached_weakref is not None:
///             w_cached = self.cached_weakref()
///             if w_cached is not None:
///                 return w_cached
///         w_ref = W_Weakref(space, w_obj, None)
///         self.cached_weakref = weakref.ref(w_ref)
///     else:
///         # subclass: cannot cache
///         w_ref = space.allocate_instance(W_Weakref, w_subtype)
///         W_Weakref.__init__(w_ref, space, w_obj, None)
///         self.append_wref_to(w_ref)
///     return w_ref
/// ```
pub fn get_or_make_weakref(
    self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
) -> PyObjectRef {
    let w_weakreftype = weakref_type();
    if w_subtype.is_null() || std::ptr::eq(w_weakreftype, w_subtype) {
        let cached = read_attr(self_lifeline, ATTR_CACHED_WEAKREF);
        if !cached.is_null() {
            return cached;
        }
        let w_ref = W_Weakref_new(w_subtype, w_obj, PY_NULL);
        write_attr(self_lifeline, ATTR_CACHED_WEAKREF, w_ref);
        w_ref
    } else {
        // subclass: cannot cache
        W_Weakref_new(w_subtype, w_obj, PY_NULL)
    }
}

/// pypy/module/_weakref/interp__weakref.py:79-91 get_or_make_proxy
///
/// ```python
/// @jit.dont_look_inside
/// def get_or_make_proxy(self, w_obj):
///     space = self.space
///     if self.cached_proxy is not None:
///         w_cached = self.cached_proxy()
///         if w_cached is not None:
///             return w_cached
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, None)
///     else:
///         w_proxy = W_Proxy(space, w_obj, None)
///     self.cached_proxy = weakref.ref(w_proxy)
///     return w_proxy
/// ```
pub fn get_or_make_proxy(self_lifeline: PyObjectRef, w_obj: PyObjectRef) -> PyObjectRef {
    let cached = read_attr(self_lifeline, ATTR_CACHED_PROXY);
    if !cached.is_null() {
        return cached;
    }
    let w_proxy = if is_callable(w_obj) {
        W_CallableProxy_new(w_obj, PY_NULL)
    } else {
        W_Proxy_new(w_obj, PY_NULL)
    };
    write_attr(self_lifeline, ATTR_CACHED_PROXY, w_proxy);
    w_proxy
}

/// pypy/module/_weakref/interp__weakref.py:93-104 get_any_weakref
///
/// ```python
/// def get_any_weakref(self, space):
///     if self.cached_weakref is not None:
///         w_ref = self.cached_weakref()
///         if w_ref is not None:
///             return w_ref
///     if self.other_refs_weak is not None:
///         w_weakreftype = space.gettypeobject(W_Weakref.typedef)
///         for wref in self.other_refs_weak.items():
///             w_ref = wref()
///             if (w_ref is not None and space.isinstance_w(w_ref, w_weakreftype)):
///                 return w_ref
///     return space.w_None
/// ```
pub fn get_any_weakref(self_lifeline: PyObjectRef) -> PyObjectRef {
    let cached = read_attr(self_lifeline, ATTR_CACHED_WEAKREF);
    if !cached.is_null() {
        return cached;
    }
    pyre_object::w_none()
}

/// pypy/module/_weakref/interp__weakref.py:111-118 make_weakref_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_weakref_with_callback(self, w_subtype, w_obj, w_callable):
///     space = self.space
///     w_ref = space.allocate_instance(W_Weakref, w_subtype)
///     W_Weakref.__init__(w_ref, space, w_obj, w_callable)
///     self.append_wref_to(w_ref)
///     self.enable_callbacks()
///     return w_ref
/// ```
pub fn make_weakref_with_callback(
    _self_lifeline: PyObjectRef,
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    W_Weakref_new(w_subtype, w_obj, w_callable)
}

/// pypy/module/_weakref/interp__weakref.py:120-129 make_proxy_with_callback
///
/// ```python
/// @jit.dont_look_inside
/// def make_proxy_with_callback(self, w_obj, w_callable):
///     space = self.space
///     if space.is_true(space.callable(w_obj)):
///         w_proxy = W_CallableProxy(space, w_obj, w_callable)
///     else:
///         w_proxy = W_Proxy(space, w_obj, w_callable)
///     self.append_wref_to(w_proxy)
///     self.enable_callbacks()
///     return w_proxy
/// ```
pub fn make_proxy_with_callback(
    _self_lifeline: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    if is_callable(w_obj) {
        W_CallableProxy_new(w_obj, w_callable)
    } else {
        W_Proxy_new(w_obj, w_callable)
    }
}

// ── W_WeakrefBase / W_Weakref ─────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:158-205
//
// class W_WeakrefBase(W_Root):
//     def __init__(self, space, w_obj, w_callable):
//         assert w_callable is not space.w_None
//         self.space = space
//         self.w_obj_weak = weakref.ref(w_obj)
//         self.w_callable = w_callable
//
// class W_Weakref(W_WeakrefBase):
//     def __init__(self, space, w_obj, w_callable):
//         W_WeakrefBase.__init__(self, space, w_obj, w_callable)
//         self.w_hash = None

#[allow(non_snake_case)]
pub fn W_Weakref_new(
    w_subtype: PyObjectRef,
    w_obj: PyObjectRef,
    w_callable: PyObjectRef,
) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    // typedef.py:519 generic_new_descr → space.allocate_instance(W_Type, w_subtype)
    let actual_type = if w_subtype.is_null() {
        weakref_type()
    } else {
        w_subtype
    };
    let obj = w_instance_new(actual_type);
    // W_WeakrefBase.__init__
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    // W_Weakref.__init__: self.w_hash = None
    write_attr(obj, ATTR_W_HASH, pyre_object::w_none());
    obj
}

#[allow(non_snake_case)]
pub fn W_Proxy_new(w_obj: PyObjectRef, w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(proxy_type());
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    obj
}

#[allow(non_snake_case)]
pub fn W_CallableProxy_new(w_obj: PyObjectRef, w_callable: PyObjectRef) -> PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(callable_proxy_type());
    write_attr(obj, ATTR_W_OBJ_WEAK, w_obj);
    if !w_callable.is_null() && !unsafe { pyre_object::is_none(w_callable) } {
        write_attr(obj, ATTR_W_CALLABLE, w_callable);
    } else {
        write_attr(obj, ATTR_W_CALLABLE, pyre_object::w_none());
    }
    obj
}

/// pypy/module/_weakref/interp__weakref.py:168-171 dereference
///
/// ```python
/// @jit.dont_look_inside
/// def dereference(self):
///     w_obj = self.w_obj_weak()
///     return w_obj
/// ```
pub fn dereference(w_ref: PyObjectRef) -> PyObjectRef {
    read_attr(w_ref, ATTR_W_OBJ_WEAK)
}

/// pypy/module/_weakref/interp__weakref.py:179-190 W_WeakrefBase.descr__repr__
///
/// ```python
/// def descr__repr__(self, space):
///     w_obj = self.dereference()
///     if w_obj is None:
///         state = '; dead'
///     else:
///         typename = space.type(w_obj).getname(space)
///         objname = w_obj.getname(space)
///         if objname and objname != '?':
///             state = "; to '%s' (%s)" % (typename, objname)
///         else:
///             state = "; to '%s'" % (typename,)
///     return self.getrepr(space, self.typedef.name, state)
/// ```
pub fn descr__repr__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = dereference(w_self);
    let type_name = unsafe {
        match crate::typedef::r#type(w_self) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "weakref".to_string(),
        }
    };
    let state = if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        "; dead".to_string()
    } else {
        let objtype_name = unsafe {
            match crate::typedef::r#type(w_obj) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => "object".to_string(),
            }
        };
        format!("; to '{}'", objtype_name)
    };
    let addr = w_self as usize;
    Ok(pyre_object::w_str_new(&format!(
        "<{} at 0x{:x}{}>",
        type_name, addr, state
    )))
}

/// pypy/module/_weakref/interp__weakref.py:207-214 descr__init__weakref
///
/// ```python
/// def descr__init__weakref(self, space, w_obj, w_callable=None,
///                          __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError,
///                     "__init__ expected at most 2 arguments")
///     if __args__.keywords:
///         raise oefmt(space.w_TypeError,
///                     "ref() does not take keyword arguments")
/// ```
pub fn descr__init__weakref(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // args[0] = self, args[1] = w_obj, args[2] = w_callable (optional)
    if args.len() > 3 {
        return Err(PyError::type_error(
            "__init__ expected at most 2 arguments".to_string(),
        ));
    }
    Ok(pyre_object::w_none())
}

/// pypy/module/_weakref/interp__weakref.py:216-223 descr_hash
///
/// ```python
/// def descr_hash(self):
///     if self.w_hash is not None:
///         return self.w_hash
///     w_obj = self.dereference()
///     if w_obj is None:
///         raise oefmt(self.space.w_TypeError, "weak object has gone away")
///     self.w_hash = self.space.hash(w_obj)
///     return self.w_hash
/// ```
pub fn descr_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let cached = read_attr(w_self, ATTR_W_HASH);
    if !cached.is_null() {
        return Ok(cached);
    }
    let w_obj = dereference(w_self);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::type_error("weak object has gone away".to_string()));
    }
    // pyre's hash: identity-based for non-int/non-str. Mirrors space.hash(w_obj).
    let h = pyre_object::w_int_new(w_obj as i64);
    write_attr(w_self, ATTR_W_HASH, h);
    Ok(h)
}

/// pypy/module/_weakref/interp__weakref.py:225-229 descr_call
///
/// ```python
/// def descr_call(self):
///     w_obj = self.dereference()
///     if w_obj is None:
///         return self.space.w_None
///     return w_obj
/// ```
pub fn descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = dereference(w_self);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Ok(pyre_object::w_none());
    }
    Ok(w_obj)
}

/// pypy/module/_weakref/interp__weakref.py:231-244 W_Weakref.compare
///
/// ```python
/// def compare(self, space, w_ref2, invert):
///     if not isinstance(w_ref2, W_Weakref):
///         return space.w_NotImplemented
///     ref1 = self
///     ref2 = w_ref2
///     w_obj1 = ref1.dereference()
///     w_obj2 = ref2.dereference()
///     if w_obj1 is None or w_obj2 is None:
///         w_res = space.is_(ref1, ref2)
///     else:
///         w_res = space.eq(w_obj1, w_obj2)
///     if invert:
///         w_res = space.not_(w_res)
///     return w_res
/// ```
fn compare(w_self: PyObjectRef, w_ref2: PyObjectRef, invert: bool) -> Result<PyObjectRef, PyError> {
    if !is_w_weakref(w_ref2) {
        return Ok(pyre_object::w_not_implemented());
    }
    let w_obj1 = dereference(w_self);
    let w_obj2 = dereference(w_ref2);
    let w_res = if w_obj1.is_null()
        || unsafe { pyre_object::is_none(w_obj1) }
        || w_obj2.is_null()
        || unsafe { pyre_object::is_none(w_obj2) }
    {
        crate::baseobjspace::is_(w_self, w_ref2)
    } else {
        crate::baseobjspace::compare(w_obj1, w_obj2, crate::baseobjspace::CompareOp::Eq)?
    };
    if invert {
        Ok(crate::baseobjspace::not_(w_res))
    } else {
        Ok(w_res)
    }
}

/// pypy/module/_weakref/interp__weakref.py:246-247 descr__eq__
pub fn descr__eq__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], false)
}

/// pypy/module/_weakref/interp__weakref.py:249-250 descr__ne__
pub fn descr__ne__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    compare(args[0], args[1], true)
}

fn is_w_weakref(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    match crate::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp, weakref_type()),
        None => false,
    }
}

fn is_callable(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe {
        if crate::is_function(obj)
            || pyre_object::is_method(obj)
            || pyre_object::is_type(obj)
            || pyre_object::is_staticmethod(obj)
            || pyre_object::is_classmethod(obj)
        {
            return true;
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            return crate::baseobjspace::lookup_in_type(w_type, "__call__").is_some();
        }
    }
    false
}

// ── module-level helpers ──────────────────────────────────────────────

/// pypy/module/_weakref/interp__weakref.py:252-257 getlifeline
///
/// ```python
/// def getlifeline(space, w_obj):
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         lifeline = WeakrefLifeline(space)
///         w_obj.setweakref(space, lifeline)
///     return lifeline
/// ```
pub fn getlifeline(w_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if let Some(lifeline) = crate::baseobjspace::getweakref(w_obj) {
        return Ok(lifeline);
    }
    let lifeline = weakref_lifeline_new();
    crate::baseobjspace::setweakref(w_obj, lifeline)?;
    Ok(lifeline)
}

/// pypy/module/_weakref/interp__weakref.py:260-268 descr__new__weakref
///
/// ```python
/// def descr__new__weakref(space, w_subtype, w_obj, w_callable=None,
///                         __args__=None):
///     if __args__.arguments_w:
///         raise oefmt(space.w_TypeError, "__new__ expected at most 2 arguments")
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_weakref(w_subtype, w_obj)
///     else:
///         return lifeline.make_weakref_with_callback(w_subtype, w_obj, w_callable)
/// ```
pub fn descr__new__weakref(
    w_subtype: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    if args.len() > 2 {
        return Err(PyError::type_error(
            "__new__ expected at most 2 arguments".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let lifeline = getlifeline(w_obj)?;
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_weakref(lifeline, w_subtype, w_obj))
    } else {
        Ok(make_weakref_with_callback(
            lifeline, w_subtype, w_obj, w_callable,
        ))
    }
}

/// Type-call entry: pyre passes `(cls, *args)` so we strip cls and route
/// through descr__new__weakref with the user-supplied subtype.
fn descr__new__weakref_typecall(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "ref() takes at least 1 argument".to_string(),
        ));
    }
    descr__new__weakref(args[0], &args[1..])
}

/// pypy/module/_weakref/interp__weakref.py:283-295 getweakrefcount
///
/// ```python
/// def getweakrefcount(space, w_obj):
///     """Return the number of weak references to 'obj'."""
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         return space.newint(0)
///     else:
///         result = lifeline.traverse(_weakref_count, 0)
///         return space.newint(result)
/// ```
pub fn getweakrefcount(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_int_new(0));
    }
    let w_obj = args[0];
    let count = match crate::baseobjspace::getweakref(w_obj) {
        None => 0,
        Some(lifeline) => {
            if !read_attr(lifeline, ATTR_CACHED_WEAKREF).is_null() {
                1
            } else {
                0
            }
        }
    };
    Ok(pyre_object::w_int_new(count))
}

/// pypy/module/_weakref/interp__weakref.py:297-309 getweakrefs
///
/// ```python
/// def getweakrefs(space, w_obj):
///     """Return a list of all weak reference objects that point to 'obj'."""
///     result = []
///     lifeline = w_obj.getweakref()
///     if lifeline is not None:
///         lifeline.traverse(_get_weakrefs, result)
///     return space.newlist(result)
/// ```
pub fn getweakrefs(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_list_new(vec![]));
    }
    let w_obj = args[0];
    let mut result = Vec::new();
    if let Some(lifeline) = crate::baseobjspace::getweakref(w_obj) {
        let cached = read_attr(lifeline, ATTR_CACHED_WEAKREF);
        if !cached.is_null() {
            result.push(cached);
        }
    }
    Ok(pyre_object::w_list_new(result))
}

// ── Proxy ─────────────────────────────────────────────────────────────
//
// pypy/module/_weakref/interp__weakref.py:311-417

/// pypy/module/_weakref/interp__weakref.py:318-319 W_Proxy.descr__hash__
///
/// ```python
/// def descr__hash__(self, space):
///     raise oefmt(space.w_TypeError, "unhashable type")
/// ```
pub fn proxy_descr__hash__(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error("unhashable type".to_string()))
}

/// pypy/module/_weakref/interp__weakref.py:322-324 W_CallableProxy.descr__call__
///
/// ```python
/// def descr__call__(self, space, __args__):
///     w_obj = force(space, self)
///     return space.call_args(w_obj, __args__)
/// ```
pub fn callable_proxy_descr__call__(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_obj = force(w_self)?;
    crate::call::call_function_impl_result(w_obj, &args[1..])
}

/// pypy/module/_weakref/interp__weakref.py:329-337 proxy
///
/// ```python
/// def proxy(space, w_obj, w_callable=None):
///     """Create a proxy object that weakly references 'obj'.
/// 'callback', if given, is called with the proxy as an argument when 'obj'
/// is about to be finalized."""
///     lifeline = getlifeline(space, w_obj)
///     if space.is_none(w_callable):
///         return lifeline.get_or_make_proxy(w_obj)
///     else:
///         return lifeline.make_proxy_with_callback(w_obj, w_callable)
/// ```
pub fn proxy(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.is_empty() {
        return Err(PyError::type_error(
            "proxy() takes at least 1 argument".to_string(),
        ));
    }
    let w_obj = args[0];
    let w_callable = if args.len() >= 2 {
        args[1]
    } else {
        pyre_object::w_none()
    };
    let lifeline = getlifeline(w_obj)?;
    if w_callable.is_null() || unsafe { pyre_object::is_none(w_callable) } {
        Ok(get_or_make_proxy(lifeline, w_obj))
    } else {
        Ok(make_proxy_with_callback(lifeline, w_obj, w_callable))
    }
}

/// pypy/module/_weakref/interp__weakref.py:339-340 descr__new__proxy
///
/// ```python
/// def descr__new__proxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError, "cannot create 'weakproxy' instances")
/// ```
pub fn descr__new__proxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakproxy' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py:342-344 descr__new__callableproxy
///
/// ```python
/// def descr__new__callableproxy(space, w_subtype, w_obj, w_callable=None):
///     raise oefmt(space.w_TypeError,
///                 "cannot create 'weakcallableproxy' instances")
/// ```
pub fn descr__new__callableproxy(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Err(PyError::type_error(
        "cannot create 'weakcallableproxy' instances".to_string(),
    ))
}

/// pypy/module/_weakref/interp__weakref.py:347-354 force
///
/// ```python
/// def force(space, proxy):
///     if not isinstance(proxy, W_AbstractProxy):
///         return proxy
///     w_obj = proxy.dereference()
///     if w_obj is None:
///         raise oefmt(space.w_ReferenceError,
///                     "weakly referenced object no longer exists")
///     return w_obj
/// ```
pub fn force(proxy: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if !is_w_abstract_proxy(proxy) {
        return Ok(proxy);
    }
    let w_obj = dereference(proxy);
    if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
        return Err(PyError::reference_error(
            "weakly referenced object no longer exists".to_string(),
        ));
    }
    Ok(w_obj)
}

fn is_w_abstract_proxy(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    match crate::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp, proxy_type()) || std::ptr::eq(tp, callable_proxy_type()),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pypy/module/_weakref/interp__weakref.py:347-354 force —
    /// dead proxy must raise ReferenceError, not RuntimeError.
    #[test]
    fn test_force_dead_proxy_raises_reference_error() {
        crate::typedef::init_typeobjects();
        let proxy = W_Proxy_new(pyre_object::w_none(), PY_NULL);
        // Simulate a dead referent by setting the weak slot back to None.
        write_attr(proxy, ATTR_W_OBJ_WEAK, pyre_object::w_none());
        let err = force(proxy).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ReferenceError);
        assert_eq!(err.message, "weakly referenced object no longer exists");
    }
}
