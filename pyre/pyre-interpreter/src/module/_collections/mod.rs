//! _collections module — PyPy: `pypy/module/_collections/`.
//!
//! Provides the C-accelerated `deque` / `defaultdict` / `OrderedDict`
//! types.  `deque` is an interp-level typed-payload `#[pyre_class]`, backed
//! by a list payload field; semantically correct for
//! `collections.py`'s `MutableSequence` consumers but not performant
//! (PyPy's `W_Deque` is a doubly-linked block list — porting that needs
//! a separate algorithm migration).  `defaultdict` is the
//! app-level `dict` subclass in `app_defaultdict.py`, mirroring PyPy's
//! `app_defaultdict.py` (neither runtime can subclass the app-level
//! `dict` from interp-level).

use pyre_object::*;

/// `pypy/module/_collections/interp_deque.py` — `W_Deque` surface
/// (append / appendleft / pop / popleft / clear / extend / extendleft /
/// rotate / count / remove / reverse / index / copy + container and repr
/// protocols, with `maxlen` bounding) backed by an inner list at
/// `self.data`.  The bound is kept in `self.maxlen` and surfaced read-only
/// via the `maxlen` property.
#[crate::pyre_class("collections.deque")]
pub struct W_Deque {
    /// Backing list object (a real `list`); always a valid list after `__new__`.
    data: PyObjectRef,
    /// Bound: `-1` = unbounded, `>= 0` = the maxlen. Validated non-negative at `__init__`.
    maxlen: i64,
    /// Lightweight iteration-lock counter (`interp_deque.py` `state`); bumped on every mutation.
    state: i64,
}

fn data(self_obj: PyObjectRef) -> PyObjectRef {
    W_Deque::from_obj(self_obj)
        .map(|d| d.data)
        .unwrap_or_else(w_none)
}

/// Snapshot the backing list into a `Vec`.
fn snapshot(self_obj: PyObjectRef) -> Vec<PyObjectRef> {
    let d = data(self_obj);
    unsafe {
        (0..w_list_len(d))
            .filter_map(|i| w_list_getitem(d, i as i64))
            .collect()
    }
}

/// Replace the backing list with `items`.
fn store(self_obj: PyObjectRef, items: Vec<PyObjectRef>) {
    W_Deque::from_obj(self_obj).unwrap().data = w_list_new(items);
    modified(self_obj);
}

/// `interp_deque.py modified` — lightweight iteration lock.  Any
/// mutation invalidates the outstanding lock so an in-progress
/// `count` / `index` / `remove` / `__contains__` / comparison detects
/// it.  PyPy invalidates a `Lock` object identity; pyre realizes the
/// same as a monotonic `state` counter: every mutation bumps it, and
/// `checklock` raises when the snapshot no longer matches.
fn lock_state(self_obj: PyObjectRef) -> i64 {
    // Volatile read: a `checklock` re-read must observe a `state` bump made
    // by a re-entrant mutation from an intervening user callback (`eq_w`).
    // A plain field read is cached across that callback because the payload
    // escaped and is mutated through a non-receiver pointer, which the
    // `&self`/`&mut self` receiver's aliasing guarantee forbids the compiler
    // from expecting.
    let base = self_obj as *const W_Deque;
    if base.is_null() {
        return 0;
    }
    unsafe { std::ptr::read_volatile(std::ptr::addr_of!((*base).state)) }
}

fn modified(self_obj: PyObjectRef) {
    // Volatile bump paired with `lock_state`'s volatile read (see there).
    let base = self_obj as *mut W_Deque;
    if base.is_null() {
        return;
    }
    unsafe {
        let p = std::ptr::addr_of_mut!((*base).state);
        std::ptr::write_volatile(p, std::ptr::read_volatile(p).wrapping_add(1));
    }
}

/// `interp_deque.py getlock` — snapshot the current lock token.
fn getlock(self_obj: PyObjectRef) -> i64 {
    lock_state(self_obj)
}

/// `interp_deque.py checklock` — raise `RuntimeError` if the deque
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
    let m = W_Deque::from_obj(self_obj)?.maxlen;
    if m < 0 { None } else { Some(m as usize) }
}

fn maxlen_obj(self_obj: PyObjectRef) -> PyObjectRef {
    match W_Deque::from_obj(self_obj) {
        Some(d) if d.maxlen >= 0 => w_int_new(d.maxlen),
        _ => w_none(),
    }
}

/// `W_Deque.append` + `trimleft`: drop from the left once over the bound.
fn do_append(self_obj: PyObjectRef, item: PyObjectRef) {
    let d = data(self_obj);
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
    // `compare_by_iteration` walks both deques
    // through their iterators; each `W_DequeIter.next` checks the lock
    // before yielding, so a mutation is detected
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
    let m = maxlen_obj(self_obj);
    if unsafe { is_none(m) } {
        crate::call::call_function_impl_result(ty, &[list])
    } else {
        crate::call::call_function_impl_result(ty, &[list, m])
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

#[crate::pyre_methods(weakrefable, unhashable)]
impl W_Deque {
    // `deque_new` allocates an empty unbounded payload; the construction
    // arguments (`iterable`, `maxlen`) are consumed by `__init__`, so they
    // are accepted and ignored here — the type-call protocol forwards them
    // (and any keywords) to `__new__` as well.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        // Construction arguments are consumed/validated by `__init__`; accept
        // (and ignore) any positional or keyword args here via the whole-slice
        // catch-all so a subclass with its own `__init__` keyword parameters
        // does not trip an unknown-keyword error in `__new__`.
        let _ = _args;
        // Stable (non-moving) allocation: the mutators re-derive `self`
        // from a raw `PyObjectRef` across list-allocating calls (`store`),
        // so the payload must not relocate under them.
        W_Deque::allocate_stable(W_Deque {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            data: w_list_new(vec![]),
            maxlen: -1,
            state: 0,
        })
    }

    // `init(iterable=None, maxlen=None)` — remember maxlen, then
    // extend so the bound is enforced while filling.
    fn __init__(
        &mut self,
        #[default(None)] iterable: Option<PyObjectRef>,
        #[default(None)] maxlen: Option<PyObjectRef>,
    ) -> Result<(), crate::PyError> {
        self.data = w_list_new(vec![]);
        self.state = 0;
        // gateway_nonnegint_w(w_maxlen): None unbounded; non-int → TypeError; negative → ValueError.
        self.maxlen = match maxlen {
            Some(m) if !unsafe { is_none(m) } => {
                if !unsafe { is_int(m) } {
                    return Err(crate::PyError::type_error("an integer is required"));
                }
                let v = unsafe { w_int_get_value(m) };
                if v < 0 {
                    return Err(crate::PyError::value_error("maxlen must be non-negative"));
                }
                v
            }
            _ => -1,
        };
        let self_obj = self as *mut W_Deque as PyObjectRef;
        if let Some(it) = iterable {
            for item in crate::builtins::collect_iterable(it)? {
                do_append(self_obj, item);
            }
        }
        Ok(())
    }
    fn append(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        do_append(self_obj, item);
    }
    fn appendleft(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        do_appendleft(self_obj, item);
    }
    fn pop(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // `W_Deque.pop` — empty raises IndexError.
        let mut items = snapshot(self_obj);
        let item = items
            .pop()
            .ok_or_else(|| crate::PyError::index_error("pop from an empty deque"))?;
        store(self_obj, items);
        Ok(item)
    }
    fn popleft(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // `W_Deque.popleft` — empty raises IndexError.
        let mut items = snapshot(self_obj);
        if items.is_empty() {
            return Err(crate::PyError::index_error("pop from an empty deque"));
        }
        let item = items.remove(0);
        store(self_obj, items);
        Ok(item)
    }
    fn clear(&mut self) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        store(self_obj, vec![]);
    }
    fn extend(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        for item in crate::builtins::collect_iterable(iterable)? {
            do_append(self_obj, item);
        }
        Ok(())
    }
    fn extendleft(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // Each element is appended on the left, so the result is
        // the reverse of `iterable`.
        for item in crate::builtins::collect_iterable(iterable)? {
            do_appendleft(self_obj, item);
        }
        Ok(())
    }
    fn count(&self, x: PyObjectRef) -> Result<i64, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
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
    fn remove(&mut self, x: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
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
                "deque.remove(x): x not in deque",
            )),
        }
    }
    fn __contains__(&self, x: PyObjectRef) -> Result<bool, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
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
    fn reverse(&mut self) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let mut items = snapshot(self_obj);
        items.reverse();
        store(self_obj, items);
    }
    fn rotate(&mut self, n: Option<PyObjectRef>) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
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
    fn index(
        &self,
        x: PyObjectRef,
        start: Option<PyObjectRef>,
        stop: Option<PyObjectRef>,
    ) -> Result<i64, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        let items = snapshot(self_obj);
        let len = items.len() as i64;
        // `space.iter(self)` takes the lock before `unwrap_start_stop`,
        // so a `__index__` on start/stop that mutates the deque is caught
        // by the first `checklock`.
        let lock = getlock(self_obj);
        let clamp = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
        let start = clamp(
            start
                .map(|v| crate::builtins::getindex_w(v))
                .transpose()?
                .unwrap_or(0),
        );
        let stop = clamp(
            stop.map(|v| crate::builtins::getindex_w(v))
                .transpose()?
                .unwrap_or(len),
        );
        let upper = stop.min(len);
        let mut i = 0i64;
        while i < upper {
            // `space.next(w_iter)` checks the lock before each element.
            checklock(self_obj, lock)?;
            if i >= start {
                if crate::baseobjspace::eq_w(items[i as usize], x)? {
                    // Match returns immediately, before the post-match
                    // `checklock`.
                    return Ok(i);
                }
                // Re-check the lock after a non-match.
                checklock(self_obj, lock)?;
            }
            i += 1;
        }
        Err(crate::PyError::value_error(format!(
            "{} is not in deque",
            unsafe { crate::py_repr(x)? }
        )))
    }
    fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `type(self)(self)` or `type(self)(self, maxlen)`.
        let ty = unsafe { w_instance_get_type(self_obj) };
        let list = w_list_new(snapshot(self_obj));
        let m = maxlen_obj(self_obj);
        if unsafe { is_none(m) } {
            crate::call::call_function_impl_result(ty, &[list])
        } else {
            crate::call::call_function_impl_result(ty, &[list, m])
        }
    }
    fn __reduce__(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `_collectionsmodule.c deque_reduce` —
        // `(type(self), args, state, iter(self))`.  `args` is
        // `((), maxlen)` when the deque is bounded so the bound
        // survives the round-trip, else `()`.  `state` is the generic
        // instance state; the items ride the listitems iterator (the
        // 4th element).
        //
        // The instance payload lives in typed fields, leaving a deque
        // subclass's `__dict__` free to round-trip its instance
        // attributes through `object_getstate_default`.
        let ty = unsafe { w_instance_get_type(self_obj) };
        let m = maxlen_obj(self_obj);
        let args = if unsafe { is_none(m) } {
            w_tuple_new(vec![])
        } else {
            w_tuple_new(vec![w_tuple_new(vec![]), m])
        };
        let state = crate::reduce_protocol::object_getstate_default(self_obj)?;
        let items = crate::baseobjspace::iter(w_list_new(snapshot(self_obj)))?;
        Ok(w_tuple_new(vec![ty, args, state, items]))
    }
    fn __len__(&self) -> i64 {
        let self_obj = self as *const W_Deque as PyObjectRef;
        (unsafe { w_list_len(data(self_obj)) }) as i64
    }
    fn __iter__(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        crate::baseobjspace::iter(data(self_obj))
    }
    fn __getitem__(&self, index: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        let items = snapshot(self_obj);
        let idx = deque_index(index, items.len() as i64)?;
        Ok(items[idx])
    }
    fn __setitem__(
        &mut self,
        index: PyObjectRef,
        value: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let mut items = snapshot(self_obj);
        let idx = deque_index(index, items.len() as i64)?;
        items[idx] = value;
        store(self_obj, items);
        Ok(())
    }
    fn __delitem__(&mut self, index: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let mut items = snapshot(self_obj);
        let idx = deque_index(index, items.len() as i64)?;
        items.remove(idx);
        store(self_obj, items);
        Ok(())
    }
    fn __eq__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Eq)
    }
    fn __ne__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ne)
    }
    fn __lt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Lt)
    }
    fn __le__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Le)
    }
    fn __gt__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Gt)
    }
    fn __ge__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_compare(self_obj, other, crate::baseobjspace::CompareOp::Ge)
    }
    fn __mul__(&self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_repeat(self_obj, n)
    }
    fn __rmul__(&self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_repeat(self_obj, n)
    }
    fn __imul__(&mut self, n: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
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
    fn __repr__(&self) -> Result<String, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
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

    #[getter]
    fn maxlen(&self) -> PyObjectRef {
        let self_obj = self as *const W_Deque as PyObjectRef;
        maxlen_obj(self_obj)
    }
}

crate::py_module! {
    "_collections",
    interpleveldefs: {
        "deque"           => type_object(),
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
