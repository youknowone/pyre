//! _collections module — PyPy: `pypy/module/_collections/`.
//!
//! Provides the C-accelerated `deque` / `defaultdict` / `OrderedDict`
//! types.  `deque` is the interp-level `py_class!` stub below, backed by
//! an attribute-dict list (`__data__`); semantically correct for
//! `collections.py`'s `MutableSequence` consumers but not performant
//! (PyPy's `W_Deque` is a doubly-linked block list — porting that needs
//! typed payload backing on top of `py_class!`).  `defaultdict` is the
//! app-level `dict` subclass in `app_defaultdict.py`, mirroring PyPy's
//! `app_defaultdict.py` (neither runtime can subclass the app-level
//! `dict` from interp-level).

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
        modified(self_obj);
    }

    /// `interp_deque.py:107 modified` — lightweight iteration lock.  Any
    /// mutation invalidates the outstanding lock so an in-progress
    /// `count` / `index` / `remove` / `__contains__` / comparison detects
    /// it.  PyPy invalidates a `Lock` object identity; pyre realizes the
    /// same as a monotonic `__state__` counter (as CPython does via
    /// `deque->state`): every mutation bumps it, and `checklock` raises when
    /// the snapshot no longer matches.
    fn lock_state(self_obj: PyObjectRef) -> i64 {
        match crate::baseobjspace::getattr_str(self_obj, "__state__") {
            Ok(w) if !w.is_null() && unsafe { is_int(w) } => unsafe { w_int_get_value(w) },
            _ => 0,
        }
    }

    fn modified(self_obj: PyObjectRef) {
        let next = lock_state(self_obj).wrapping_add(1);
        let _ = crate::baseobjspace::setattr_str(self_obj, "__state__", w_int_new(next));
    }

    /// `interp_deque.py:110 getlock` — snapshot the current lock token.
    fn getlock(self_obj: PyObjectRef) -> i64 {
        lock_state(self_obj)
    }

    /// `interp_deque.py:115 checklock` — raise `RuntimeError` if the deque
    /// was mutated since `lock` was taken.
    fn checklock(self_obj: PyObjectRef, lock: i64) -> Result<(), crate::PyError> {
        if lock_state(self_obj) != lock {
            return Err(crate::PyError::runtime_error(
                "deque mutated during iteration",
            ));
        }
        Ok(())
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
        modified(self_obj);
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
        // `compare_by_iteration` (baseobjspace.py:1491) walks both deques
        // through their iterators; each `W_DequeIter.next` checks the lock
        // before yielding (interp_deque.py:650-652), so a mutation is detected
        // up to — but not past — the element that decides the result.  pyre
        // snapshots, so check each deque's lock before consuming its element
        // and stop as soon as the result is determined.
        use crate::baseobjspace::CompareOp;
        let lock_a = getlock(self_obj);
        let lock_b = getlock(other);
        let snap_a = snapshot(self_obj);
        let snap_b = snapshot(other);
        let mut i = 0usize;
        loop {
            // next(w_it1): lock-check precedes the element.
            checklock(self_obj, lock_a)?;
            let x1 = snap_a.get(i).copied();
            // next(w_it2): lock-check precedes the element.
            checklock(other, lock_b)?;
            let x2 = snap_b.get(i).copied();
            match (x1, x2) {
                (Some(a), Some(b)) => {
                    if !crate::baseobjspace::eq_w(a, b)? {
                        // First differing pair decides the result; no further
                        // `next`, so no further lock check.
                        return match op {
                            CompareOp::Eq => Ok(pyre_object::w_bool_from(false)),
                            CompareOp::Ne => Ok(pyre_object::w_bool_from(true)),
                            _ => crate::baseobjspace::compare(a, b, op),
                        };
                    }
                }
                // One or both deques exhausted — decide by length.
                _ => {
                    let res = match op {
                        CompareOp::Eq => x1.is_none() && x2.is_none(),
                        CompareOp::Ne => !(x1.is_none() && x2.is_none()),
                        CompareOp::Lt => x2.is_some(),
                        CompareOp::Le => x1.is_none(),
                        CompareOp::Gt => x1.is_some(),
                        CompareOp::Ge => x2.is_none(),
                    };
                    return Ok(pyre_object::w_bool_from(res));
                }
            }
            i += 1;
        }
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
        // Dotted name so the builtin `__module__` resolves to `collections`
        // (the name dot-split fallback), matching CPython's
        // `collections.deque`; `__name__` / `__qualname__` strip to `deque`.
        "collections.deque",
        methods: {
            // `init(iterable=None, maxlen=None)` — remember maxlen, then
            // extend so the bound is enforced while filling.
            fn __init__(self_obj: PyObjectRef, #[default(None)] iterable: Option<PyObjectRef>, #[default(None)] maxlen: Option<PyObjectRef>) -> Result<(), crate::PyError> {
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
            fn count(self_obj: PyObjectRef, x: PyObjectRef) -> Result<i64, crate::PyError> {
                let lock = getlock(self_obj);
                let mut n = 0i64;
                for it in snapshot(self_obj) {
                    let equal = crate::baseobjspace::eq_w(it, x)?;
                    checklock(self_obj, lock)?;
                    if equal {
                        n += 1;
                    }
                }
                Ok(n)
            }
            fn remove(self_obj: PyObjectRef, x: PyObjectRef) -> Result<(), crate::PyError> {
                let mut items = snapshot(self_obj);
                let lock = getlock(self_obj);
                let mut pos = None;
                for (i, &it) in items.iter().enumerate() {
                    let equal = crate::baseobjspace::eq_w(it, x)?;
                    checklock(self_obj, lock)?;
                    if equal {
                        pos = Some(i);
                        break;
                    }
                }
                match pos {
                    Some(pos) => {
                        items.remove(pos);
                        store(self_obj, items);
                        Ok(())
                    }
                    None => Err(crate::PyError::value_error(
                        "deque.remove(x): x not in deque")),
                }
            }
            fn __contains__(self_obj: PyObjectRef, x: PyObjectRef) -> Result<bool, crate::PyError> {
                let lock = getlock(self_obj);
                for it in snapshot(self_obj) {
                    let equal = crate::baseobjspace::eq_w(it, x)?;
                    checklock(self_obj, lock)?;
                    if equal {
                        return Ok(true);
                    }
                }
                Ok(false)
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
                // `space.iter(self)` takes the lock before `unwrap_start_stop`
                // (interp_deque.py:393-396), so a `__index__` on start/stop
                // that mutates the deque is caught by the first `checklock`.
                let lock = getlock(self_obj);
                let clamp = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
                let start = clamp(
                    start.map(|v| crate::builtins::getindex_w(v)).transpose()?.unwrap_or(0),
                );
                let stop = clamp(
                    stop.map(|v| crate::builtins::getindex_w(v)).transpose()?.unwrap_or(len),
                );
                let upper = stop.min(len);
                let mut i = 0i64;
                while i < upper {
                    // `space.next(w_iter)` checks the lock before each element.
                    checklock(self_obj, lock)?;
                    if i >= start {
                        if crate::baseobjspace::eq_w(items[i as usize], x)? {
                            // Match returns immediately, before the post-match
                            // `checklock` (interp_deque.py:402-403).
                            return Ok(i);
                        }
                        // interp_deque.py:406 — re-check after a non-match.
                        checklock(self_obj, lock)?;
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
            fn __reduce__(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
                // `_collectionsmodule.c deque_reduce` —
                // `(type(self), args, state, iter(self))`.  `args` is
                // `((), maxlen)` when the deque is bounded so the bound
                // survives the round-trip, else `()`.  `state` is the generic
                // instance state; the items ride the listitems iterator (the
                // 4th element).
                //
                // The instance payload (`__data__`/`__maxlen__`/`__state__`)
                // lives in the attribute side-dict (this type is hasdict so the
                // attribute backing has somewhere to go), and a subclass keeps
                // user attributes in that same dict with no separate `__dict__`
                // descriptor exposed, so `object_getstate_default` reports
                // `None` here and a deque subclass round-trips its type, items
                // and bound but not extra instance attributes. Preserving those
                // needs the typed-payload deque backing noted in the module
                // header (the payload would then leave the attribute dict free
                // to be a real instance `__dict__`).
                let ty = unsafe { w_instance_get_type(self_obj) };
                let w_maxlen = crate::baseobjspace::getattr_str(self_obj, "__maxlen__")
                    .unwrap_or_else(|_| w_none());
                let args = if w_maxlen.is_null() || unsafe { is_none(w_maxlen) } {
                    w_tuple_new(vec![])
                } else {
                    w_tuple_new(vec![w_tuple_new(vec![]), w_maxlen])
                };
                let state = crate::reduce_protocol::object_getstate_default(self_obj)?;
                let items = crate::baseobjspace::iter(w_list_new(snapshot(self_obj)))?;
                Ok(w_tuple_new(vec![ty, args, state, items]))
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
                // The repr uses the short class name, so strip any dotted
                // module prefix from the builtin tp_name (`collections.deque`
                // → `deque`); a user subclass name has no dot.
                let full = unsafe { w_type_get_name(w_instance_get_type(self_obj)) };
                let name = full.rsplit('.').next().unwrap_or(full);
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

crate::py_module! {
    "_collections",
    interpleveldefs: {
        "deque"           => deque_class::type_object(),
        "_deque_iterator" => crate::typedef::w_object(),
        // `OrderedDict` is a dict subclass; alias to the dict type
        // object so `isinstance(d, OrderedDict)` matches dict instances.
        "OrderedDict"     => crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE),
    },
    // `defaultdict` is an app-level `dict` subclass — see the module
    // header and `app_defaultdict.py` (PyPy `app_defaultdict.defaultdict`).
    appleveldefs: {
        "app_defaultdict.py" => ["defaultdict"],
    },
}
