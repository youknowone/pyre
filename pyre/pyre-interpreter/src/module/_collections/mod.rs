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

/// `pypy/module/_collections/interp_deque.py` — `W_Deque` surface
/// (append / appendleft / pop / popleft / clear / extend / extendleft /
/// rotate / count / remove / reverse / index / copy + container and repr
/// protocols, with `maxlen` bounding) backed by an inner list at
/// `self.__data__`.  The bound is kept in the private `self.__maxlen__`
/// slot and surfaced read-only via the `maxlen` property.
mod deque_class {
    use super::*;

    fn data(self_obj: PyObjectRef) -> Option<PyObjectRef> {
        crate::baseobjspace::getattr_str(self_obj, "__data__").ok()
    }

    /// Snapshot the backing list into a `Vec`.
    fn snapshot(self_obj: PyObjectRef) -> Vec<PyObjectRef> {
        match data(self_obj) {
            Some(d) => unsafe {
                (0..w_list_len(d))
                    .filter_map(|i| w_list_getitem(d, i as i64))
                    .collect()
            },
            None => Vec::new(),
        }
    }

    /// Replace the backing list with `items`.
    fn store(self_obj: PyObjectRef, items: Vec<PyObjectRef>) {
        let _ = crate::baseobjspace::setattr_str(self_obj, "__data__", w_list_new(items));
    }

    /// `self.maxlen`: `None` (unbounded) or a non-negative bound.  The
    /// bound is validated non-negative at construction, so the stored
    /// value is read back directly.
    fn maxlen_bound(self_obj: PyObjectRef) -> Option<usize> {
        let w = crate::baseobjspace::getattr_str(self_obj, "__maxlen__").ok()?;
        if w.is_null() || unsafe { is_none(w) } {
            None
        } else {
            Some(unsafe { w_int_get_value(w) } as usize)
        }
    }

    /// `W_Deque.append` + `trimleft`: drop from the left once over the bound.
    fn do_append(self_obj: PyObjectRef, item: PyObjectRef) {
        let Some(d) = data(self_obj) else { return };
        unsafe { w_list_append(d, item) };
        if let Some(m) = maxlen_bound(self_obj) {
            let mut items = snapshot(self_obj);
            if items.len() > m {
                items.drain(0..items.len() - m);
                store(self_obj, items);
            }
        }
    }

    /// `space.decode_index4` index-only case — reject a slice with a
    /// `TypeError`, route any `__index__`-able object through
    /// `getindex_w`, apply the negative-index wrap and the in-range
    /// check, and return the resolved element position.
    fn deque_index(index: PyObjectRef, len: i64) -> Result<usize, crate::PyError> {
        if unsafe { pyre_object::is_slice(index) } {
            return Err(crate::PyError::type_error("deque[:] is not supported"));
        }
        let mut idx = crate::builtins::getindex_w(index)?;
        if idx < 0 {
            idx += len;
        }
        if idx < 0 || idx >= len {
            return Err(crate::PyError::index_error("index out of range"));
        }
        Ok(idx as usize)
    }

    /// `W_Deque.compare` — element-wise (`compare_by_iteration`) ordering
    /// against another deque, ignoring `maxlen`; `NotImplemented` when the
    /// other operand is not a deque.  Delegates to list comparison over
    /// snapshots of both backings.
    fn deque_compare(
        self_obj: PyObjectRef,
        other: PyObjectRef,
        op: crate::baseobjspace::CompareOp,
    ) -> Result<PyObjectRef, crate::PyError> {
        if !crate::baseobjspace::isinstance(other, type_object())? {
            return Ok(pyre_object::w_not_implemented());
        }
        let la = w_list_new(snapshot(self_obj));
        let lb = w_list_new(snapshot(other));
        crate::baseobjspace::compare(la, lb, op)
    }

    /// `W_Deque.mul` — repeat the elements `num` times, then re-bound by
    /// `maxlen` by routing through the constructor (which trims).
    fn deque_repeat(self_obj: PyObjectRef, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        if !unsafe { is_int(n) } {
            return Ok(pyre_object::w_not_implemented());
        }
        let num = unsafe { w_int_get_value(n) }.max(0);
        let base = snapshot(self_obj);
        let mut items = Vec::with_capacity(base.len().saturating_mul(num as usize));
        for _ in 0..num {
            items.extend_from_slice(&base);
        }
        let ty = unsafe { w_instance_get_type(self_obj) };
        let list = w_list_new(items);
        match crate::baseobjspace::getattr_str(self_obj, "__maxlen__") {
            Ok(m) if !(m.is_null() || unsafe { is_none(m) }) => {
                crate::call::call_function_impl_result(ty, &[list, m])
            }
            _ => crate::call::call_function_impl_result(ty, &[list]),
        }
    }

    /// `W_Deque.appendleft` + `trimright`: drop from the right once over the bound.
    fn do_appendleft(self_obj: PyObjectRef, item: PyObjectRef) {
        let mut items = snapshot(self_obj);
        items.insert(0, item);
        if let Some(m) = maxlen_bound(self_obj) {
            items.truncate(m);
        }
        store(self_obj, items);
    }

    crate::py_class! {
        "deque",
        methods: {
            // `init(iterable=None, maxlen=None)` — remember maxlen, then
            // extend so the bound is enforced while filling.
            fn __init__(self_obj: PyObjectRef, iterable: Option<PyObjectRef>, maxlen: Option<PyObjectRef>) -> Result<(), crate::PyError> {
                store(self_obj, vec![]);
                // `gateway_nonnegint_w(w_maxlen)` — None is unbounded; a
                // non-integer is a TypeError and a negative bound is a
                // ValueError, both raised here at construction rather than
                // silently clamped when the bound is later read.
                let w_maxlen = match maxlen {
                    Some(m) if !unsafe { is_none(m) } => {
                        if !unsafe { is_int(m) } {
                            return Err(crate::PyError::type_error(
                                "an integer is required"));
                        }
                        if unsafe { w_int_get_value(m) } < 0 {
                            return Err(crate::PyError::value_error(
                                "maxlen must be non-negative"));
                        }
                        m
                    }
                    _ => w_none(),
                };
                let _ = crate::baseobjspace::setattr_str(self_obj, "__maxlen__", w_maxlen);
                if let Some(it) = iterable {
                    for item in crate::builtins::collect_iterable(it)? {
                        do_append(self_obj, item);
                    }
                }
                Ok(())
            }
            fn append(self_obj: PyObjectRef, item: PyObjectRef) {
                do_append(self_obj, item);
            }
            fn appendleft(self_obj: PyObjectRef, item: PyObjectRef) {
                do_appendleft(self_obj, item);
            }
            fn pop(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `W_Deque.pop` — empty raises IndexError.
                let mut items = snapshot(self_obj);
                let item = items.pop().ok_or_else(||
                    crate::PyError::index_error("pop from an empty deque"))?;
                store(self_obj, items);
                Ok(item)
            }
            fn popleft(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `W_Deque.popleft` — empty raises IndexError.
                let mut items = snapshot(self_obj);
                if items.is_empty() {
                    return Err(crate::PyError::index_error("pop from an empty deque"));
                }
                let item = items.remove(0);
                store(self_obj, items);
                Ok(item)
            }
            fn clear(self_obj: PyObjectRef) {
                store(self_obj, vec![]);
            }
            fn extend(self_obj: PyObjectRef, iterable: PyObjectRef) -> Result<(), crate::PyError> {
                for item in crate::builtins::collect_iterable(iterable)? {
                    do_append(self_obj, item);
                }
                Ok(())
            }
            fn extendleft(self_obj: PyObjectRef, iterable: PyObjectRef) -> Result<(), crate::PyError> {
                // Each element is appended on the left, so the result is
                // the reverse of `iterable`.
                for item in crate::builtins::collect_iterable(iterable)? {
                    do_appendleft(self_obj, item);
                }
                Ok(())
            }
            fn count(self_obj: PyObjectRef, x: PyObjectRef) -> i64 {
                snapshot(self_obj)
                    .into_iter()
                    .filter(|&it| crate::baseobjspace::eq_w(it, x))
                    .count() as i64
            }
            fn remove(self_obj: PyObjectRef, x: PyObjectRef) -> Result<(), crate::PyError> {
                let mut items = snapshot(self_obj);
                match items.iter().position(|&it| crate::baseobjspace::eq_w(it, x)) {
                    Some(pos) => {
                        items.remove(pos);
                        store(self_obj, items);
                        Ok(())
                    }
                    None => Err(crate::PyError::value_error(
                        "deque.remove(x): x not in deque")),
                }
            }
            fn __contains__(self_obj: PyObjectRef, x: PyObjectRef) -> bool {
                snapshot(self_obj)
                    .into_iter()
                    .any(|it| crate::baseobjspace::eq_w(it, x))
            }
            fn reverse(self_obj: PyObjectRef) {
                let mut items = snapshot(self_obj);
                items.reverse();
                store(self_obj, items);
            }
            fn rotate(self_obj: PyObjectRef, n: Option<PyObjectRef>) -> Result<(), crate::PyError> {
                // Rotate right by n (negative rotates left).  The count goes
                // through `__index__` so non-integers raise `TypeError`.
                let n = match n {
                    Some(v) => crate::builtins::getindex_w(v)?,
                    None => 1,
                };
                let mut items = snapshot(self_obj);
                let len = items.len() as i64;
                if len <= 1 {
                    return Ok(());
                }
                let shift = ((n % len) + len) % len;
                if shift != 0 {
                    items.rotate_right(shift as usize);
                    store(self_obj, items);
                }
                Ok(())
            }
            fn index(self_obj: PyObjectRef, x: PyObjectRef, start: Option<PyObjectRef>, stop: Option<PyObjectRef>) -> Result<i64, crate::PyError> {
                let items = snapshot(self_obj);
                let len = items.len() as i64;
                let clamp = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
                let start = clamp(
                    start.map(|v| crate::builtins::getindex_w(v)).transpose()?.unwrap_or(0),
                );
                let stop = clamp(
                    stop.map(|v| crate::builtins::getindex_w(v)).transpose()?.unwrap_or(len),
                );
                let mut i = start;
                while i < stop {
                    if crate::baseobjspace::eq_w(items[i as usize], x) {
                        return Ok(i);
                    }
                    i += 1;
                }
                Err(crate::PyError::value_error(
                    format!("{} is not in deque", unsafe { crate::py_repr(x)? })))
            }
            fn copy(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `type(self)(self)` or `type(self)(self, maxlen)`.
                let ty = unsafe { w_instance_get_type(self_obj) };
                let list = w_list_new(snapshot(self_obj));
                match crate::baseobjspace::getattr_str(self_obj, "__maxlen__") {
                    Ok(m) if !(m.is_null() || unsafe { is_none(m) }) =>
                        crate::call::call_function_impl_result(ty, &[list, m]),
                    _ => crate::call::call_function_impl_result(ty, &[list]),
                }
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
                let items = snapshot(self_obj);
                let idx = deque_index(index, items.len() as i64)?;
                Ok(items[idx])
            }
            fn __setitem__(self_obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
                let mut items = snapshot(self_obj);
                let idx = deque_index(index, items.len() as i64)?;
                items[idx] = value;
                store(self_obj, items);
                Ok(())
            }
            fn __delitem__(self_obj: PyObjectRef, index: PyObjectRef) -> Result<(), crate::PyError> {
                let mut items = snapshot(self_obj);
                let idx = deque_index(index, items.len() as i64)?;
                items.remove(idx);
                store(self_obj, items);
                Ok(())
            }
            fn __eq__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Eq)
            }
            fn __ne__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ne)
            }
            fn __lt__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Lt)
            }
            fn __le__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Le)
            }
            fn __gt__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Gt)
            }
            fn __ge__(self_obj: PyObjectRef, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ge)
            }
            fn __mul__(self_obj: PyObjectRef, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_repeat(self_obj, n)
            }
            fn __rmul__(self_obj: PyObjectRef, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                deque_repeat(self_obj, n)
            }
            fn __imul__(self_obj: PyObjectRef, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `W_Deque.imul` — empty or *1 is self; *<=0 clears; else
                // repeat in place, trimmed by maxlen.
                if !unsafe { is_int(n) } {
                    return Ok(pyre_object::w_not_implemented());
                }
                let num = unsafe { w_int_get_value(n) };
                let base = snapshot(self_obj);
                if base.is_empty() || num == 1 {
                    return Ok(self_obj);
                }
                if num <= 0 {
                    store(self_obj, vec![]);
                    return Ok(self_obj);
                }
                let mut items = Vec::with_capacity(base.len().saturating_mul(num as usize));
                for _ in 0..num {
                    items.extend_from_slice(&base);
                }
                if let Some(m) = maxlen_bound(self_obj) {
                    if items.len() > m {
                        items.drain(0..items.len() - m);
                    }
                }
                store(self_obj, items);
                Ok(self_obj)
            }
            fn __repr__(self_obj: PyObjectRef) -> Result<String, crate::PyError> {
                // `dequerepr` — a deque reachable from its own items renders
                // the inner reference as `[...]` instead of recursing.
                let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
                    return Ok("[...]".to_string());
                };
                let name = unsafe { w_type_get_name(w_instance_get_type(self_obj)) };
                let listrepr = snapshot(self_obj)
                    .into_iter()
                    .map(|it| unsafe { crate::py_repr(it) })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                Ok(match maxlen_bound(self_obj) {
                    Some(m) => format!("{name}([{listrepr}], maxlen={m})"),
                    None => format!("{name}([{listrepr}])"),
                })
            }
        },
        properties: {
            // GetSetProperty fget is invoked as `fget(descriptor, instance)`.
            fn maxlen(_descr: PyObjectRef, self_obj: PyObjectRef) -> PyObjectRef {
                crate::baseobjspace::getattr_str(self_obj, "__maxlen__")
                    .unwrap_or_else(|_| w_none())
            }
        }
    }
}

/// `pypy/module/_collections/interp_defaultdict.py` — `W_DefaultDict`
/// stub backed by an inner dict at `self.__data__` plus
/// `self.default_factory`.  A missing key invokes `default_factory` and
/// stores the result (`W_DefaultDict.missing`); a missing key with no
/// factory raises `KeyError(key)`.  Still a stub vs upstream: this type
/// subclasses `object`, not `dict`, so `isinstance(d, dict)` is False, and
/// `__missing__`/`__repr__`/`copy`/`__reduce__` are absent.
mod defaultdict_class {
    use super::*;

    crate::py_class! {
        "defaultdict",
        methods: {
            fn __init__(self_obj: PyObjectRef, factory: Option<PyObjectRef>) {
                let _ = crate::baseobjspace::setattr_str(
                    self_obj, "default_factory", factory.unwrap_or(w_none()));
                let _ = crate::baseobjspace::setattr_str(self_obj, "__data__", w_dict_new());
            }
            fn __getitem__(self_obj: PyObjectRef, key: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `interp_defaultdict.py W_DefaultDict.missing` — present key
                // returns stored value; missing key + no factory raises
                // KeyError(key); missing key + factory invokes the factory
                // and stores the result.
                let d = crate::baseobjspace::getattr_str(self_obj, "__data__")
                    .map_err(|_| crate::PyError::key_error_with_key(key))?;
                unsafe {
                    if let Some(v) = w_dict_lookup(d, key) {
                        return Ok(v);
                    }
                }
                let factory = crate::baseobjspace::getattr_str(self_obj, "default_factory")
                    .unwrap_or_else(|_| w_none());
                if factory.is_null() || unsafe { is_none(factory) } {
                    return Err(crate::PyError::key_error_with_key(key));
                }
                let value = crate::call::call_function_impl_result(factory, &[])?;
                unsafe { w_dict_store(d, key, value) };
                Ok(value)
            }
            fn __setitem__(self_obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
                if let Ok(d) = crate::baseobjspace::getattr_str(self_obj, "__data__") {
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
        // `OrderedDict` is a dict subclass; alias to the dict type
        // object so `isinstance(d, OrderedDict)` matches dict instances.
        "OrderedDict"     => crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE),
    },
}
