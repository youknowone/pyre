//! `_queue` accelerator module.

use pyre_object::*;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[crate::pyre_class("_queue.SimpleQueue")]
#[derive(Default)]
pub struct W_SimpleQueue {
    pub map: *const u8,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    queue: Mutex<VecDeque<PyObjectRef>>,
    not_empty: Condvar,
}

const _: () = assert!(
    std::mem::offset_of!(W_SimpleQueue, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_SimpleQueue must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_SimpleQueue, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_SimpleQueue must keep W_ObjectObject's storage offset"
);

fn queue_lock<'a>(
    mutex: &'a Mutex<VecDeque<PyObjectRef>>,
) -> MutexGuard<'a, VecDeque<PyObjectRef>> {
    if let Ok(guard) = mutex.try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    drop(blocked);
    guard
}

/// `_queue_SimpleQueue_get_impl` reads `timeout` only on the blocking path:
/// `block=False` is answered from the queue immediately, so the argument is
/// neither converted nor range-checked there.
///
/// `_PyTime_FromSecondsObject` runs before the sign check, so a value it cannot
/// represent as a nanosecond timestamp is refused rather than turned into a
/// wait: an infinity would otherwise block forever and a NaN would poll once.
/// The conversion is the one `_thread.lock.acquire` already performs
/// (`module/thread/mod.rs parse_acquire_args`).
fn parse_timeout(block: bool, timeout: PyObjectRef) -> Result<Option<f64>, crate::PyError> {
    if !block {
        return Ok(None);
    }
    if timeout.is_null() || unsafe { pyre_object::is_none(timeout) } {
        return Ok(None);
    }
    let seconds = crate::baseobjspace::float_w(timeout)?;
    if seconds.is_nan() {
        return Err(crate::PyError::value_error(
            "Invalid value NaN (not a number)",
        ));
    }
    // `rarithmetic.ovfcheck_float_to_longlong` bounds, as in `parse_acquire_args`.
    const NS_MIN: f64 = -9223372036854776832.0;
    const NS_MAX: f64 = 9223372036854775296.0;
    if !(NS_MIN..NS_MAX).contains(&(seconds * 1e9).ceil()) {
        return Err(crate::PyError::overflow_error(
            "timestamp out of range for platform time_t",
        ));
    }
    if seconds < 0.0 {
        return Err(crate::PyError::value_error(
            "'timeout' must be a non-negative number",
        ));
    }
    Ok(Some(seconds))
}

/// `parse_timeout` accepted only a finite, non-negative number of seconds, so
/// the deadline is always representable and `None` means "wait forever".
fn deadline_from_timeout(timeout: Option<f64>) -> Option<Instant> {
    timeout.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds))
}

fn empty_error() -> crate::PyError {
    let mut err = crate::PyError::runtime_error("");
    if let Some(cls) = crate::builtins::lookup_exc_class("_queue.Empty")
        && let Ok(exc) = crate::builtins::exc_exception_new(&[cls])
    {
        err.exc_object = exc;
    }
    err
}

fn simplequeue_put(queue: &W_SimpleQueue, item: PyObjectRef) -> PyObjectRef {
    // Taking the lock can drop the GIL and rooting is itself a collection
    // point, so `item` is read back out of its shadow-stack slot rather than
    // from the argument, which a move would have left stale.
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(item);
    let mut guard = queue_lock(&queue.queue);
    guard.push_back(pyre_object::gc_roots::shadow_stack_get(base));
    drop(guard);
    queue.not_empty.notify_one();
    w_none()
}

fn simplequeue_get(
    queue: &W_SimpleQueue,
    block: bool,
    timeout: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let timeout = parse_timeout(block, timeout)?;
    let mut guard = queue_lock(&queue.queue);
    if !block {
        return guard.pop_front().ok_or_else(empty_error);
    }
    let deadline = deadline_from_timeout(timeout);
    loop {
        if let Some(item) = guard.pop_front() {
            return Ok(item);
        }
        let blocked = crate::module::thread::before_external_block();
        if let Some(deadline) = deadline {
            let now = Instant::now();
            if now >= deadline {
                drop(blocked);
                return Err(empty_error());
            }
            let (next_guard, result) = queue
                .not_empty
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard = next_guard;
            drop(blocked);
            if result.timed_out() && guard.is_empty() {
                return Err(empty_error());
            }
        } else {
            guard = queue
                .not_empty
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(blocked);
        }
    }
}

mod simplequeue_methods {
    use super::*;

    #[crate::pyre_methods(weakrefable, unhashable)]
    impl W_SimpleQueue {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(
                    "_queue.SimpleQueue() takes no arguments",
                ));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let obj = Self::allocate_stable(Self::default());
            unsafe { (*obj).w_class = cls };
            Ok(obj)
        }

        /// `block` and `timeout` are accepted and ignored: the queue is
        /// unbounded, so a put never blocks.  They are named without a leading
        /// underscore because the keyword a caller may bind is taken from the
        /// parameter's own identifier, and `put(item, block=True,
        /// timeout=None)` is the signature.
        fn put(
            &self,
            item: PyObjectRef,
            #[default(true)] block: bool,
            #[default(w_none())] timeout: PyObjectRef,
        ) -> PyObjectRef {
            let _ = (block, timeout);
            simplequeue_put(self, item)
        }

        fn put_nowait(&self, item: PyObjectRef) -> PyObjectRef {
            simplequeue_put(self, item)
        }

        fn get(
            &self,
            #[default(true)] block: bool,
            #[default(w_none())] timeout: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            simplequeue_get(self, block, timeout)
        }

        fn get_nowait(&self) -> Result<PyObjectRef, crate::PyError> {
            simplequeue_get(self, false, w_none())
        }

        fn empty(&self) -> bool {
            queue_lock(&self.queue).is_empty()
        }

        fn qsize(&self) -> i64 {
            queue_lock(&self.queue).len() as i64
        }

        #[classmethod]
        fn __class_getitem__(
            cls: PyObjectRef,
            item: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            crate::_pypy_generic_alias::generic_alias_class_getitem(&[cls, item])
        }
    }
}

/// Drop the Rust-owned queue storage.
///
/// # Safety
/// `obj` must be a GC-dead `W_SimpleQueue`.
pub unsafe fn w_simplequeue_dealloc(obj: PyObjectRef) {
    unsafe { std::ptr::drop_in_place(obj as *mut W_SimpleQueue) };
}

/// Walk the queued items owned by a `W_SimpleQueue`.
///
/// # Safety
/// `obj_addr` must point at a live `W_SimpleQueue`.
pub unsafe fn w_simplequeue_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let queue = unsafe { &mut *(obj_addr as *mut W_SimpleQueue) };
    // `get_mut` is sound because the deque is only ever mutated while holding
    // the GIL, so collection cannot overlap a mutation. A thread parked inside
    // `queue_lock` or `Condvar::wait` may hold the mutex, but it is not
    // touching the deque; it rejoins the RUNNING census before touching it
    // again.
    let items = queue
        .queue
        .get_mut()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (front, back) = items.as_mut_slices();
    for item in front.iter_mut().chain(back.iter_mut()) {
        f(item as *mut PyObjectRef as *mut majit_ir::GcRef);
    }
}

crate::py_module! {
    "_queue",
    interpleveldefs: {
        "SimpleQueue" => simplequeue_methods::type_object(),
    },
    exceptions: {
        "Empty" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _queue init"),
    },
}
