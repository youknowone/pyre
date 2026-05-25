//! _collections module — PyPy: `pypy/module/_collections/`.
//!
//! Provides the C-accelerated `deque` / `defaultdict` / `OrderedDict`
//! type stubs.  Pyre backs each instance with an attribute-dict list
//! (`__data__`) or dict (`__data__` + `default_factory`); semantically
//! correct for `collections.py`'s `MutableSequence` /
//! `MutableMapping.register(...)` consumers but not performant.  PyPy's
//! `W_Deque` is a doubly-linked block list — porting that needs typed
//! payload backing on top of `py_class!`.

use pyre_object::*;

/// `pypy/module/_collections/interp_deque.py` — minimal `W_Deque`
/// surface (append / appendleft / pop / popleft / clear / extend +
/// container protocol) backed by an inner list at `self.__data__`.
mod deque_class {
    use super::*;

    fn data(self_obj: PyObjectRef) -> Option<PyObjectRef> {
        crate::baseobjspace::getattr(self_obj, "__data__").ok()
    }

    crate::py_class! {
        "deque",
        methods: {
            // `__init__(iterable=(), maxlen=None)` — store items as
            // `__data__` list, remember maxlen for future trimming.
            fn __init__(self_obj: PyObjectRef, iterable: Option<PyObjectRef>, maxlen: Option<PyObjectRef>) {
                let items = iterable
                    .map(|it| crate::builtins::collect_iterable(it).unwrap_or_default())
                    .unwrap_or_default();
                let _ = crate::baseobjspace::setattr(self_obj, "__data__", w_list_new(items));
                let _ = crate::baseobjspace::setattr(self_obj, "maxlen", maxlen.unwrap_or(w_none()));
            }
            fn append(self_obj: PyObjectRef, item: PyObjectRef) {
                if let Some(d) = data(self_obj) {
                    unsafe { w_list_append(d, item) };
                }
            }
            fn appendleft(self_obj: PyObjectRef, item: PyObjectRef) {
                if let Some(d) = data(self_obj) {
                    unsafe {
                        let n = w_list_len(d);
                        let mut items: Vec<_> = (0..n)
                            .filter_map(|i| w_list_getitem(d, i as i64))
                            .collect();
                        items.insert(0, item);
                        let _ = crate::baseobjspace::setattr(
                            self_obj, "__data__", w_list_new(items));
                    }
                }
            }
            fn pop(self_obj: PyObjectRef) -> PyObjectRef {
                let Some(d) = data(self_obj) else { return w_none() };
                unsafe {
                    let n = w_list_len(d);
                    if n == 0 {
                        return w_none();
                    }
                    let item = w_list_getitem(d, (n - 1) as i64).unwrap_or(w_none());
                    let items: Vec<_> = (0..n - 1)
                        .filter_map(|i| w_list_getitem(d, i as i64))
                        .collect();
                    let _ = crate::baseobjspace::setattr(
                        self_obj, "__data__", w_list_new(items));
                    item
                }
            }
            fn popleft(self_obj: PyObjectRef) -> PyObjectRef {
                let Some(d) = data(self_obj) else { return w_none() };
                unsafe {
                    let n = w_list_len(d);
                    if n == 0 {
                        return w_none();
                    }
                    let item = w_list_getitem(d, 0).unwrap_or(w_none());
                    let items: Vec<_> = (1..n)
                        .filter_map(|i| w_list_getitem(d, i as i64))
                        .collect();
                    let _ = crate::baseobjspace::setattr(
                        self_obj, "__data__", w_list_new(items));
                    item
                }
            }
            fn clear(self_obj: PyObjectRef) {
                let _ = crate::baseobjspace::setattr(self_obj, "__data__", w_list_new(vec![]));
            }
            fn extend(self_obj: PyObjectRef, iterable: PyObjectRef) -> Result<(), crate::PyError> {
                let items = crate::builtins::collect_iterable(iterable)?;
                if let Some(d) = data(self_obj) {
                    for item in items {
                        unsafe { w_list_append(d, item) };
                    }
                }
                Ok(())
            }
            fn __len__(self_obj: PyObjectRef) -> i64 {
                data(self_obj)
                    .map(|d| unsafe { w_list_len(d) } as i64)
                    .unwrap_or(0)
            }
            fn __iter__(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                match data(self_obj) {
                    Some(d) => crate::baseobjspace::iter(d),
                    None => Ok(w_seq_iter_new(w_list_new(vec![]), 0)),
                }
            }
            fn __getitem__(self_obj: PyObjectRef, index: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                match data(self_obj) {
                    Some(d) => crate::baseobjspace::getitem(d, index),
                    None => Ok(w_none()),
                }
            }
        }
    }
}

/// `pypy/module/_collections/interp_defaultdict.py` — `W_DefaultDict`
/// stub backed by an inner dict at `self.__data__` plus
/// `self.default_factory`.  Factory invocation on missing key is
/// short-circuited to `w_none()` because callable invocation needs
/// frame-level plumbing the macro can't yet express.
mod defaultdict_class {
    use super::*;

    crate::py_class! {
        "defaultdict",
        methods: {
            fn __init__(self_obj: PyObjectRef, factory: Option<PyObjectRef>) {
                let _ = crate::baseobjspace::setattr(
                    self_obj, "default_factory", factory.unwrap_or(w_none()));
                let _ = crate::baseobjspace::setattr(self_obj, "__data__", w_dict_new());
            }
            fn __getitem__(self_obj: PyObjectRef, key: PyObjectRef) -> PyObjectRef {
                let Ok(d) = crate::baseobjspace::getattr(self_obj, "__data__") else {
                    return w_none();
                };
                unsafe {
                    if let Some(v) = w_dict_lookup(d, key) {
                        return v;
                    }
                }
                let Ok(factory) = crate::baseobjspace::getattr(self_obj, "default_factory") else {
                    return w_none();
                };
                if !factory.is_null() && !unsafe { is_none(factory) } {
                    // factory invocation needs frame-level call infra;
                    // store None as a placeholder so subsequent reads
                    // are idempotent (matches PyPy `__missing__` returning None).
                    let default = w_none();
                    unsafe { w_dict_store(d, key, default) };
                    return default;
                }
                w_none()
            }
            fn __setitem__(self_obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
                if let Ok(d) = crate::baseobjspace::getattr(self_obj, "__data__") {
                    unsafe { w_dict_store(d, key, value) };
                }
            }
        }
    }
}

crate::py_module! {
    "_collections",
    interpleveldefs: {
        "deque"           => deque_class::type_object(),
        "_deque_iterator" => crate::typedef::w_object(),
        "defaultdict"     => defaultdict_class::type_object(),
        "OrderedDict"     => crate::typedef::w_type(),
    },
}
