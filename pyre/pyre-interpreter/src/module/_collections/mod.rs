//! _collections module — PyPy: `pypy/module/_collections/`.
//!
//! Provides the C-accelerated `deque` / `defaultdict` / `OrderedDict`
//! types.  `deque` is an interp-level typed-payload `#[pyre_class]` backed
//! by the same doubly-linked list of fixed-size blocks as PyPy's `W_Deque`.
//! `defaultdict` and `OrderedDict` are
//! app-level `dict` subclasses in `app_defaultdict.py` / `app_odict.py`,
//! mirroring PyPy (neither runtime can subclass the app-level `dict` from
//! interp-level).

use pyre_object::*;
use std::cell::UnsafeCell;
use std::sync::LazyLock;

const BLOCKLEN: i64 = 62;
const CENTER: i64 = (BLOCKLEN - 1) / 2;

/// `interp_deque.py Block`: links and element storage are GC-visible object
/// references, exactly like the RPython object's three attributes.
pub mod deque_block {
    use pyre_object::*;

    #[crate::pyre_class("_collections._deque_block")]
    pub struct W_DequeBlock {
        pub leftlink: PyObjectRef,
        pub rightlink: PyObjectRef,
        pub data: PyObjectRef,
    }

    pub fn new(leftlink: PyObjectRef, rightlink: PyObjectRef) -> PyObjectRef {
        // `interp_deque.py:14 Block.data = [None] * BLOCKLEN` is annotated by
        // RPython as a list of `W_Root` references.  It must therefore keep
        // ObjectListStrategy even when every live deque entry is an int or a
        // float; a typed list strategy would unbox and later rebox the entry,
        // breaking deque's shallow-copy identity semantics.
        let data = w_list_new_object(vec![w_none(); super::BLOCKLEN as usize]);
        W_DequeBlock::allocate_stable(W_DequeBlock {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            leftlink,
            rightlink,
            data,
        })
    }
}

use deque_block::W_DequeBlock;

/// `pypy/module/_collections/interp_deque.py` — `W_Deque` surface
/// (append / appendleft / pop / popleft / clear / extend / extendleft /
/// rotate / count / remove / reverse / index / copy + container and repr
/// protocols, with `maxlen` bounding).
#[crate::pyre_class("collections.deque")]
pub struct W_Deque {
    leftblock: PyObjectRef,
    rightblock: PyObjectRef,
    leftindex: i64,
    rightindex: i64,
    len: i64,
    /// Bound: `-1` = unbounded, `>= 0` = the maxlen. Validated non-negative at `__init__`.
    maxlen: i64,
    /// Lightweight iteration-lock counter (`interp_deque.py` `state`); bumped on every mutation.
    state: i64,
}

// PyPy's deque block/endpoint transitions execute atomically under its GIL.
// CPython 3.14's free-threaded deque preserves the same boundary with object
// critical sections.  Pyre keeps the semantic state on W_Deque and uses the
// same narrow address-striped reentrant synchronization adaptation as its
// list/dict/set ports.  The stripes are reset after fork because a vanished
// thread may have held one in the parent.
struct ForkDequeLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkDequeLock {}

impl ForkDequeLock {
    fn new() -> Self {
        Self(UnsafeCell::new(parking_lot::ReentrantMutex::new(())))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

static DEQUE_LOCKS: LazyLock<Vec<ForkDequeLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkDequeLock::new()).collect());

type DequeGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

#[majit_macros::dont_look_inside]
unsafe fn w_deque_lock(obj: PyObjectRef) -> DequeGuard {
    let lock = DEQUE_LOCKS[(obj as usize >> 4) & (DEQUE_LOCKS.len() - 1)].get();
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = majit_gc::gc_sync::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

pub fn deque_locks_after_fork_child() {
    for lock in DEQUE_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

fn block(block: PyObjectRef) -> &'static mut W_DequeBlock {
    W_DequeBlock::from_obj(block).expect("deque block")
}

fn block_get(block_obj: PyObjectRef, index: i64) -> PyObjectRef {
    unsafe { w_list_getitem(block(block_obj).data, index) }.expect("deque block index")
}

fn block_set(block_obj: PyObjectRef, index: i64, value: PyObjectRef) {
    let stored = unsafe { w_list_setitem(block(block_obj).data, index, value) };
    debug_assert!(stored);
}

fn deque_len(self_obj: PyObjectRef) -> i64 {
    W_Deque::from_obj(self_obj).map(|d| d.len).unwrap_or(0)
}

/// Snapshot the backing list into a `Vec`.
///
/// The walk reads `leftblock`/`leftindex`/`len` and then follows `rightlink`,
/// so it has to exclude the chain replacement in `store`: a walk interleaved
/// with one reads endpoints belonging to two different chains and can reach a
/// `PY_NULL` link. The stripe is reentrant, so the `store`/`append_right`
/// nesting below costs nothing.
fn snapshot(self_obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let Some(deque) = W_Deque::from_obj(self_obj) else {
        return vec![];
    };
    let mut result = Vec::with_capacity(deque.len as usize);
    let mut block_obj = deque.leftblock;
    let mut index = deque.leftindex;
    for _ in 0..deque.len {
        result.push(block_get(block_obj, index));
        index += 1;
        if index >= BLOCKLEN {
            block_obj = block(block_obj).rightlink;
            index = 0;
        }
    }
    result
}

/// Replace the block chain with `items`.
///
/// Held across the whole replacement, not per `append_right`, so no other
/// thread observes the intermediate empty chain.
fn store(self_obj: PyObjectRef, items: Vec<PyObjectRef>) {
    let _roots = pyre_object::gc_roots::push_roots();
    // `append_right` allocates, so any of the appends below can collect and
    // move the entries `items` still names. They are livevars for the whole
    // replacement, not just `self_obj`. Publish both slices before the single
    // normalize: `pin_root` would normalize `self_obj` while the entries are
    // not yet on the stack, and that query is itself a safepoint.
    let root_base = pyre_object::gc_roots::publish_roots(&[self_obj]);
    let items_base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(root_base, 1 + items.len());
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    // Reentrant: the two callees take the same stripe.
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    clear_blocks(self_obj);
    for i in 0..items.len() {
        let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
        let item = pyre_object::gc_roots::shadow_stack_get(items_base + i);
        append_right(self_obj, item);
    }
}

fn clear_blocks(self_obj: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let new_block = deque_block::new(PY_NULL, PY_NULL);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.leftblock = new_block;
    deque.rightblock = new_block;
    deque.leftindex = CENTER + 1;
    deque.rightindex = CENTER;
    deque.len = 0;
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

// The two deque iterators are `.allocate()`-immortal `#[pyre_class]` wrappers
// holding a traced `deque` edge the marker never scans; re-export so `build_gc`
// can register their offsets with the immortal-root walker.
pub use deque_iter::W_DequeIter;
pub use deque_rev_iter::W_DequeRevIter;

// `W_DequeIter`: the iterator owns the deque, current position, number
// of remaining entries, and a snapshot of the deque mutation lock.
pub mod deque_iter {
    use super::{BLOCKLEN, W_Deque, block, block_get, checklock, getlock};
    use pyre_object::*;

    #[crate::pyre_class("_collections._deque_iterator")]
    pub struct W_DequeIter {
        deque: PyObjectRef,
        block: PyObjectRef,
        index: i64,
        counter: i64,
        lock: i64,
    }

    fn constructor_args(args: &[PyObjectRef]) -> Result<(PyObjectRef, i64), crate::PyError> {
        let (positional, _kwargs) = crate::builtins::split_builtin_kwargs(args);
        // `__new__`'s descriptor ABI leaves the requested class at the front
        // of the varargs slice (the typed `_cls` slot is the descriptor
        // receiver).  PyPy's W_DequeIter__new__ starts parsing after it.
        let positional = positional.get(1..).unwrap_or(&[]);
        if positional.is_empty() || positional.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "_deque_iterator expected 1 or 2 arguments, got {}",
                positional.len()
            )));
        }
        let deque = positional[0];
        if W_Deque::from_obj(deque).is_none() {
            let name = unsafe { pyre_object::type_name_of(deque) };
            return Err(crate::PyError::type_error(format!(
                "must be collections.deque, not {name}"
            )));
        }
        let index = match positional.get(1) {
            Some(&value) => crate::builtins::space_index_w(value)?,
            None => 0,
        };
        Ok((deque, index))
    }

    #[crate::pyre_methods]
    impl W_DequeIter {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let (deque, requested) = constructor_args(args)?;
            Ok(make(deque, requested))
        }

        fn __iter__(&self) -> PyObjectRef {
            self as *const W_DequeIter as PyObjectRef
        }

        fn __length_hint__(&self) -> i64 {
            self.counter.max(0)
        }

        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            if let Err(error) = checklock(self.deque, self.lock) {
                self.counter = 0;
                return Err(error);
            }
            if self.counter <= 0 {
                return Err(crate::PyError::stop_iteration());
            }
            self.counter -= 1;
            let item = block_get(self.block, self.index);
            self.index += 1;
            if self.index >= BLOCKLEN {
                self.block = block(self.block).rightlink;
                self.index = 0;
            }
            Ok(item)
        }

        fn __reduce__(&self) -> PyObjectRef {
            let self_obj = self as *const W_DequeIter as PyObjectRef;
            let ty = unsafe { w_instance_get_type(self_obj) };
            let consumed = W_Deque::from_obj(self.deque)
                .map(|deque| deque.len - self.counter)
                .unwrap_or(0);
            w_tuple_new(vec![ty, w_tuple_new(vec![self.deque, w_int_new(consumed)])])
        }
    }

    pub fn make(deque: PyObjectRef, requested: i64) -> PyObjectRef {
        let _ = type_object();
        let (len, mut block_obj, mut index) = W_Deque::from_obj(deque)
            .map(|deque| (deque.len, deque.leftblock, deque.leftindex))
            .unwrap_or((0, PY_NULL, 0));
        let requested_index = requested.max(0);
        let mut remaining = requested_index;
        while remaining > 0 && remaining < len {
            let available = BLOCKLEN - index;
            if remaining < available {
                index += remaining;
                remaining = 0;
            } else {
                remaining -= available;
                block_obj = block(block_obj).rightlink;
                index = 0;
            }
        }
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(deque);
        W_DequeIter::allocate_stable(W_DequeIter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            deque,
            block: block_obj,
            index,
            counter: len - requested_index,
            lock: getlock(deque),
        })
    }

    pub fn public_type() -> PyObjectRef {
        let ty = type_object();
        // `interp_deque.py`: W_DequeIter.typedef explicitly sets
        // acceptable_as_base_class=False.
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
        ty
    }
}

// PyPy `W_DequeRevIter`, with the same state shape and reverse indexing into
// pyre's list-backed deque payload.
pub mod deque_rev_iter {
    use super::{BLOCKLEN, W_Deque, block, block_get, checklock, getlock};
    use pyre_object::*;

    #[crate::pyre_class("_collections._deque_reverse_iterator")]
    pub struct W_DequeRevIter {
        deque: PyObjectRef,
        block: PyObjectRef,
        index: i64,
        counter: i64,
        lock: i64,
    }

    #[crate::pyre_methods]
    impl W_DequeRevIter {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let (positional, _kwargs) = crate::builtins::split_builtin_kwargs(args);
            let positional = positional.get(1..).unwrap_or(&[]);
            if positional.is_empty() || positional.len() > 2 {
                return Err(crate::PyError::type_error(format!(
                    "_deque_reverse_iterator expected 1 or 2 arguments, got {}",
                    positional.len()
                )));
            }
            let deque = positional[0];
            if W_Deque::from_obj(deque).is_none() {
                let name = unsafe { pyre_object::type_name_of(deque) };
                return Err(crate::PyError::type_error(format!(
                    "must be collections.deque, not {name}"
                )));
            }
            let requested = match positional.get(1) {
                Some(&value) => crate::builtins::space_index_w(value)?,
                None => 0,
            };
            Ok(make(deque, requested))
        }

        fn __iter__(&self) -> PyObjectRef {
            self as *const W_DequeRevIter as PyObjectRef
        }

        fn __length_hint__(&self) -> i64 {
            self.counter.max(0)
        }

        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            if let Err(error) = checklock(self.deque, self.lock) {
                self.counter = 0;
                return Err(error);
            }
            if self.counter <= 0 {
                return Err(crate::PyError::stop_iteration());
            }
            self.counter -= 1;
            let item = block_get(self.block, self.index);
            self.index -= 1;
            if self.index < 0 {
                self.block = block(self.block).leftlink;
                self.index = BLOCKLEN - 1;
            }
            Ok(item)
        }

        fn __reduce__(&self) -> PyObjectRef {
            let self_obj = self as *const W_DequeRevIter as PyObjectRef;
            let ty = unsafe { w_instance_get_type(self_obj) };
            let consumed = W_Deque::from_obj(self.deque)
                .map(|deque| deque.len - self.counter)
                .unwrap_or(0);
            w_tuple_new(vec![ty, w_tuple_new(vec![self.deque, w_int_new(consumed)])])
        }
    }

    pub fn make(deque: PyObjectRef, requested: i64) -> PyObjectRef {
        let _ = type_object();
        let (len, mut block_obj, mut index) = W_Deque::from_obj(deque)
            .map(|deque| (deque.len, deque.rightblock, deque.rightindex))
            .unwrap_or((0, PY_NULL, 0));
        let offset = requested.max(0);
        let mut remaining = offset;
        while remaining > 0 && remaining < len {
            let available = index + 1;
            if remaining < available {
                index -= remaining;
                remaining = 0;
            } else {
                remaining -= available;
                block_obj = block(block_obj).leftlink;
                index = BLOCKLEN - 1;
            }
        }
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(deque);
        W_DequeRevIter::allocate_stable(W_DequeRevIter {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            deque,
            block: block_obj,
            index,
            counter: len - offset,
            lock: getlock(deque),
        })
    }

    pub fn public_type() -> PyObjectRef {
        let ty = type_object();
        // `interp_deque.py`: W_DequeRevIter.typedef explicitly sets
        // acceptable_as_base_class=False.
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ty, false) };
        ty
    }
}

/// `W_Deque.append` + `trimleft`, ported from `interp_deque.py`.
fn append_right(self_obj: PyObjectRef, item: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(item);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let mut ri = W_Deque::from_obj(self_obj).expect("deque").rightindex + 1;
    if ri >= BLOCKLEN {
        let old_right = W_Deque::from_obj(self_obj).expect("deque").rightblock;
        let new_right = deque_block::new(old_right, PY_NULL);
        block(old_right).rightlink = new_right;
        W_Deque::from_obj(self_obj).expect("deque").rightblock = new_right;
        ri = 0;
    }
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.rightindex = ri;
    block_set(
        deque.rightblock,
        ri,
        pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1),
    );
    deque.len += 1;
    if maxlen_bound(self_obj).is_some_and(|maxlen| deque.len > maxlen as i64) {
        let _ = pop_left(self_obj).expect("trimleft on non-empty deque");
    }
    modified(self_obj);
}

fn append_left(self_obj: PyObjectRef, item: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(item);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let item = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let mut li = W_Deque::from_obj(self_obj).expect("deque").leftindex - 1;
    if li < 0 {
        let old_left = W_Deque::from_obj(self_obj).expect("deque").leftblock;
        let new_left = deque_block::new(PY_NULL, old_left);
        block(old_left).leftlink = new_left;
        W_Deque::from_obj(self_obj).expect("deque").leftblock = new_left;
        li = BLOCKLEN - 1;
    }
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    deque.leftindex = li;
    block_set(
        deque.leftblock,
        li,
        pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1),
    );
    deque.len += 1;
    if maxlen_bound(self_obj).is_some_and(|maxlen| deque.len > maxlen as i64) {
        let _ = pop_right(self_obj).expect("trimright on non-empty deque");
    }
    modified(self_obj);
}

fn pop_right(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if deque.len == 0 {
        return Err(crate::PyError::index_error("pop from an empty deque"));
    }
    deque.len -= 1;
    let item = block_get(deque.rightblock, deque.rightindex);
    block_set(deque.rightblock, deque.rightindex, w_none());
    let mut ri = deque.rightindex - 1;
    if ri < 0 {
        if deque.len == 0 {
            deque.leftindex = CENTER + 1;
            ri = CENTER;
        } else {
            let old_right = deque.rightblock;
            let new_right = block(old_right).leftlink;
            deque.rightblock = new_right;
            block(new_right).rightlink = PY_NULL;
            ri = BLOCKLEN - 1;
        }
    }
    deque.rightindex = ri;
    modified(self_obj);
    Ok(item)
}

fn pop_left(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let _deque_guard = unsafe { w_deque_lock(self_obj) };
    let self_obj = pyre_object::gc_roots::shadow_stack_get(root_base);
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if deque.len == 0 {
        return Err(crate::PyError::index_error("pop from an empty deque"));
    }
    deque.len -= 1;
    let item = block_get(deque.leftblock, deque.leftindex);
    block_set(deque.leftblock, deque.leftindex, w_none());
    let mut li = deque.leftindex + 1;
    if li >= BLOCKLEN {
        if deque.len == 0 {
            li = CENTER + 1;
            deque.rightindex = CENTER;
        } else {
            let old_left = deque.leftblock;
            let new_left = block(old_left).rightlink;
            deque.leftblock = new_left;
            block(new_left).leftlink = PY_NULL;
            li = 0;
        }
    }
    deque.leftindex = li;
    modified(self_obj);
    Ok(item)
}

/// `space.decode_index4` index-only case — reject a slice with a
/// `TypeError`, route any `__index__`-able object through
/// `getindex_w`, apply the negative-index wrap and the in-range
/// check, and return the resolved element position.
fn deque_index(index: PyObjectRef, len_of: impl FnOnce() -> i64) -> Result<usize, crate::PyError> {
    if unsafe { pyre_object::is_slice(index) } {
        return Err(crate::PyError::type_error("deque[:] is not supported"));
    }
    let mut idx = crate::builtins::getindex_w(index)?;
    // `decode_index4(w_index, self)` bounds the converted index against the
    // deque as it stands *after* `__index__` ran, so a re-entrant clear or
    // shrink is reported as IndexError instead of addressing stale storage.
    let len = len_of();
    if idx < 0 {
        idx += len;
    }
    if idx < 0 || idx >= len {
        return Err(crate::PyError::index_error("index out of range"));
    }
    Ok(idx as usize)
}

/// Re-bound a converted index against the deque as it stands *under* the
/// stripe guard.
///
/// `deque_index` bounds against the length it read while `__index__` was still
/// running, so by the time the guard is held another thread may have shortened
/// the deque. `locate` would then walk past the end of the chain and
/// `Vec::remove` would panic instead of raising, so every indexed accessor
/// re-checks here before it addresses storage.
fn check_index_under_guard(self_obj: PyObjectRef, idx: i64) -> Result<(), crate::PyError> {
    if idx < 0 || idx >= deque_len(self_obj) {
        return Err(crate::PyError::index_error("index out of range"));
    }
    Ok(())
}

/// `W_Deque.locate`: choose the nearer endpoint and walk whole blocks.
fn locate(self_obj: PyObjectRef, mut index: i64) -> (PyObjectRef, i64) {
    let deque = W_Deque::from_obj(self_obj).expect("deque");
    if index < (deque.len >> 1) {
        index += deque.leftindex;
        let mut block_obj = deque.leftblock;
        while index >= BLOCKLEN {
            block_obj = block(block_obj).rightlink;
            index -= BLOCKLEN;
        }
        (block_obj, index)
    } else {
        index = index - deque.len + 1 + deque.rightindex;
        let mut block_obj = deque.rightblock;
        while index < 0 {
            block_obj = block(block_obj).leftlink;
            index += BLOCKLEN;
        }
        (block_obj, index)
    }
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
        let initial_block = deque_block::new(PY_NULL, PY_NULL);
        // Stable (non-moving) allocation: mutators re-derive `self` across
        // block allocations, just as the translated RPython object retains
        // identity while its block chain grows.
        W_Deque::allocate_stable(W_Deque {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            leftblock: initial_block,
            rightblock: initial_block,
            leftindex: CENTER + 1,
            rightindex: CENTER,
            len: 0,
            maxlen: -1,
            state: 0,
        })
    }

    // interp_deque.py:619-620 — exact classmethod binding preserves a deque
    // subclass as the GenericAlias origin.
    #[classmethod]
    fn __class_getitem__(
        cls: PyObjectRef,
        item: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        crate::_pypy_generic_alias::generic_alias_class_getitem(&[cls, item])
    }

    // `init(iterable=None, maxlen=None)` — remember maxlen, then
    // extend so the bound is enforced while filling.
    fn __init__(
        &mut self,
        #[default(None)] iterable: Option<PyObjectRef>,
        #[default(None)] maxlen: Option<PyObjectRef>,
    ) -> Result<(), crate::PyError> {
        // gateway_nonnegint_w(w_maxlen): None unbounded; non-int → TypeError; negative → ValueError.
        // Validated before the clear so a bad maxlen leaves the deque untouched.
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
        // Reinitialize by clearing, but only when non-empty (`init`:
        // `if self.len > 0: self.clear()`). `store` bumps the iteration lock
        // (`modified`) so an outstanding scan of an existing deque detects the
        // reset; an already-empty deque leaves `state` untouched, so re-init
        // while iterating an empty deque yields StopIteration, not a mutation
        // RuntimeError.
        if deque_len(self_obj) > 0 {
            clear_blocks(self_obj);
        }
        if let Some(it) = iterable {
            for item in crate::builtins::collect_iterable(it)? {
                append_right(self_obj, item);
            }
        }
        Ok(())
    }
    fn append(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        append_right(self_obj, item);
    }
    fn appendleft(&mut self, item: PyObjectRef) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        append_left(self_obj, item);
    }
    fn pop(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        pop_right(self_obj)
    }
    fn popleft(&mut self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        pop_left(self_obj)
    }
    fn clear(&mut self) {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        clear_blocks(self_obj);
    }
    fn extend(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        for item in crate::builtins::collect_iterable(iterable)? {
            append_right(self_obj, item);
        }
        Ok(())
    }
    fn extendleft(&mut self, iterable: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // Each element is appended on the left, so the result is
        // the reverse of `iterable`.
        for item in crate::builtins::collect_iterable(iterable)? {
            append_left(self_obj, item);
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
            // CPython 3.14's deque.remove reports re-entrant mutation as
            // IndexError (PyPy's older block-list port reports RuntimeError).
            // The comparison may have cleared or otherwise resized the live
            // deque, so do not continue against the stale snapshot.
            if lock_state(self_obj) != lock {
                return Err(crate::PyError::index_error(
                    "deque mutated during iteration",
                ));
            }
            if equal {
                pos = Some(i);
                break;
            }
        }
        match pos {
            Some(pos) => {
                // The guard cannot span the loop above: `eq_w` runs arbitrary
                // Python, and holding a stripe across it would let user code
                // reach a second deque and invert the acquisition order. It
                // does cover the write-back, and the epoch is re-read under it
                // so a mutation that landed after the last comparison is
                // reported rather than silently overwritten by the snapshot.
                let _deque_guard = unsafe { w_deque_lock(self_obj) };
                if lock_state(self_obj) != lock {
                    return Err(crate::PyError::index_error(
                        "deque mutated during iteration",
                    ));
                }
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
        // `interp_deque.py reverse`: swap entries from both endpoints without
        // invalidating iterators. No Python runs inside the swap loop, so the
        // stripe covers the whole transition.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let len = deque_len(self_obj);
        for index in 0..(len >> 1) {
            let (left_block, left_index) = locate(self_obj, index);
            let (right_block, right_index) = locate(self_obj, len - index - 1);
            let left = block_get(left_block, left_index);
            let right = block_get(right_block, right_index);
            block_set(left_block, left_index, right);
            block_set(right_block, right_index, left);
        }
    }
    fn rotate(&mut self, n: Option<PyObjectRef>) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // Rotate right by n (negative rotates left).  The count goes
        // through `__index__` so non-integers raise `TypeError`.
        let n = match n {
            Some(v) => crate::builtins::getindex_w(v)?,
            None => 1,
        };
        // Read-modify-write: `snapshot` and `store` are individually atomic,
        // and the stripe held across both is what stops a concurrent mutation
        // in between from being overwritten by the stale snapshot. No Python
        // runs from here on.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
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
    fn insert(&mut self, i: PyObjectRef, x: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        // `W_Deque.insert(i, x)` — the index goes through `__index__` (converted
        // before the deque is inspected), then a bounded deque that is already
        // full raises before the element is placed.
        let index = crate::builtins::getindex_w(i)?;
        // See `rotate`: the snapshot/store pair is the read-modify-write that
        // the stripe has to cover as one.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        let mut items = snapshot(self_obj);
        if maxlen_bound(self_obj).is_some_and(|m| items.len() >= m) {
            return Err(crate::PyError::index_error(
                "deque already at its maximum size",
            ));
        }
        let len = items.len() as i64;
        // list.insert clamping: a negative index counts from the end and an
        // out-of-range index clamps to the near edge.
        let pos = if index < 0 {
            (index + len).max(0)
        } else {
            index.min(len)
        } as usize;
        items.insert(pos, x);
        store(self_obj, items);
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
                .map(crate::builtins::getindex_w)
                .transpose()?
                .unwrap_or(0),
        );
        let stop = clamp(
            stop.map(crate::builtins::getindex_w)
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
        Err(crate::PyError::value_error(crate::display::wtf8_format!(
            unsafe { crate::display::py_repr_wtf8(x)? },
            " is not in deque"
        )))
    }
    fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `type(self)(self)` or `type(self)(self, maxlen)`.
        let ty = unsafe { w_instance_get_type(self_obj) };
        let m = maxlen_obj(self_obj);
        if unsafe { is_none(m) } {
            crate::call::call_function_impl_result(ty, &[self_obj])
        } else {
            crate::call::call_function_impl_result(ty, &[self_obj, m])
        }
    }
    fn __copy__(&self) -> Result<PyObjectRef, crate::PyError> {
        // interp_deque.py typedef: `__copy__ = interp2app(W_Deque.copy)`.
        self.copy()
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
        // interp_deque.py W_Deque.reduce calls `space.iter(self)`, preserving
        // subclass __iter__ overrides (including an override that raises).
        let items = crate::baseobjspace::iter(self_obj)?;
        Ok(w_tuple_new(vec![ty, args, state, items]))
    }
    fn __len__(&self) -> i64 {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_len(self_obj)
    }
    fn __iter__(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        Ok(deque_iter::make(self_obj, 0))
    }
    fn __reversed__(&self) -> PyObjectRef {
        let self_obj = self as *const W_Deque as PyObjectRef;
        deque_rev_iter::make(self_obj, 0)
    }
    fn __getitem__(&self, index: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        let idx = deque_index(index, || deque_len(self_obj))? as i64;
        // `locate` walks the chain and the caller then reads through the block
        // it returns, so the pair has to exclude a concurrent `store`. The
        // index conversion runs Python and stays outside.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx)?;
        let (block_obj, block_index) = locate(self_obj, idx);
        Ok(block_get(block_obj, block_index))
    }
    fn __setitem__(
        &mut self,
        index: PyObjectRef,
        value: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let idx = deque_index(index, || deque_len(self_obj))? as i64;
        // interp_deque.py W_Deque.setitem replaces the block entry in place
        // and deliberately does not call `modified()`: deque iterators remain
        // valid and observe subsequently assigned elements (including after
        // pickle reconstruction).
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx)?;
        let (block_obj, block_index) = locate(self_obj, idx);
        block_set(block_obj, block_index, value);
        Ok(())
    }
    fn __delitem__(&mut self, index: PyObjectRef) -> Result<(), crate::PyError> {
        let self_obj = self as *mut W_Deque as PyObjectRef;
        let idx = deque_index(index, || deque_len(self_obj))?;
        // See `rotate`: the snapshot/store pair is the read-modify-write that
        // the stripe has to cover as one.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
        check_index_under_guard(self_obj, idx as i64)?;
        let mut items = snapshot(self_obj);
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
    fn __add__(&self, other: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        if W_Deque::from_obj(other).is_none() {
            return Err(crate::PyError::type_error(format!(
                "can only concatenate deque (not \"{}\") to deque",
                unsafe { pyre_object::type_name_of(other) },
            )));
        }

        // CPython 3.14 deque_concat copies the left operand, preserving its
        // concrete subtype, then extends that copy with the right deque.
        let copy = self.copy()?;
        if W_Deque::from_obj(copy).is_none() {
            return Err(crate::PyError::type_error(
                "deque.__add__ returned a non-deque",
            ));
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(copy);
        for item in snapshot(other) {
            append_right(pyre_object::gc_roots::shadow_stack_get(root_base), item);
        }
        Ok(pyre_object::gc_roots::shadow_stack_get(root_base))
    }
    fn __iadd__(&mut self, iterable: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        // interp_deque.py W_Deque.iadd: extend in place and return self.
        let self_obj = self as *mut W_Deque as PyObjectRef;
        self.extend(iterable)?;
        Ok(self_obj)
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
        // Nothing between the snapshot and the write-back runs Python, so the
        // whole read-modify-write fits under one stripe — without it a
        // concurrent `append` lands between them and is dropped by `store`.
        let _deque_guard = unsafe { w_deque_lock(self_obj) };
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
        if let Some(m) = maxlen_bound(self_obj)
            && items.len() > m
        {
            items.drain(0..items.len() - m);
        }
        store(self_obj, items);
        Ok(self_obj)
    }
    fn __repr__(&self) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = self as *const W_Deque as PyObjectRef;
        // `dequerepr` — a deque reachable from its own items renders
        // the inner reference as `[...]` instead of recursing.
        let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
            return Ok(pyre_object::w_str_new("[...]"));
        };
        // The repr uses the short class name, so strip any dotted
        // module prefix from the builtin tp_name (`collections.deque`
        // → `deque`); a user subclass name has no dot.
        let full = unsafe { w_type_get_name(w_instance_get_type(self_obj)) };
        let name = full.rsplit('.').next().unwrap_or(full);
        let mut out = rustpython_wtf8::Wtf8Buf::new();
        out.push_str(name);
        out.push_str("([");
        for (i, item) in snapshot(self_obj).into_iter().enumerate() {
            if i != 0 {
                out.push_str(", ");
            }
            out.push_wtf8(&unsafe { crate::py_repr_wtf8(item)? });
        }
        match maxlen_bound(self_obj) {
            Some(m) => out.push_str(&format!("], maxlen={m})")),
            None => out.push_str("])"),
        }
        Ok(pyre_object::w_str_from_wtf8(out))
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
        "deque"                   => type_object(),
        "_deque_iterator"         => deque_iter::public_type(),
        "_deque_reverse_iterator" => deque_rev_iter::public_type(),
    },
    // `defaultdict` and `OrderedDict` are app-level `dict` subclasses —
    // see the module header and `app_defaultdict.py` / `app_odict.py`
    // (PyPy `app_defaultdict.defaultdict` / `app_odict.OrderedDict`).
    appleveldefs: {
        "app_defaultdict.py" => ["defaultdict"],
        "app_odict.py" => ["OrderedDict"],
    },
}
