//! `pypy/interpreter/pytraceback.py:17-115 PyTraceback` line-by-line port.
//!
//! ```python
//! class PyTraceback(baseobjspace.W_Root):
//!     def __init__(self, space, frame, lasti, next, lineno=LINENO_NOT_COMPUTED):
//!         self.space = space
//!         self.frame = frame
//!         self.lasti = lasti
//!         self.next = next
//!         self.lineno = lineno
//! ```
//!
//! Pyre stores `frame` as a raw `*mut PyFrame` opaque pointer.  PyPy's
//! `frame` is a `PyFrame` W_Root; pyre's `PyFrame` lacks the
//! `PyObject` header (`pyframe.py:39` `#[repr(C)]` without `ob_header`
//! — frames are heap-allocated outside the nursery and walked by a
//! custom GC walker).  Until `PyFrame` grows a W_Root header (which is
//! itself a multi-file change), the descriptor `tb_frame` cannot
//! return a Python-visible object directly; that gap is the
//! convergence path documented at slice 3.

use pyre_object::pyobject::*;

/// `pytraceback.py:12` `LINENO_NOT_COMPUTED = -sys.maxint-1` —
/// sentinel meaning "please take the lineno from the frame and
/// `lasti`".  Pyre uses `i64::MIN` to match RPython's `-sys.maxint-1`
/// idiom (`pytraceback.py:9-12`).
pub const LINENO_NOT_COMPUTED: i64 = i64::MIN;

pub static PYTRACEBACK_TYPE: PyType = new_pytype("traceback");

/// Layout: `[ob_header | frame: *mut PyFrame | lasti: i64 | w_next:
/// PyObjectRef | lineno: i64]`.
///
/// Only `w_next` is GC-traced.  `frame` is a raw pointer to a
/// custom-walked `PyFrame` allocation; the frame walker scans the
/// traceback chain separately when wired up at slice 2.
#[repr(C)]
pub struct PyTraceback {
    pub ob_header: PyObject,
    /// `pytraceback.py:29 self.frame = frame` — opaque `*mut PyFrame`.
    /// Not a `PyObjectRef` because `PyFrame` has no `PyObject` header.
    /// **Dangling-safe note**: by the time a traceback escapes the
    /// raising frame (the common case — `except` block in a caller),
    /// the original `PyFrame` allocation has been freed.  Readers
    /// MUST NOT dereference `frame`; metadata that survives the
    /// frame's lifetime is snapshotted into the `w_code` slot below.
    pub frame: *mut crate::pyframe::PyFrame,
    /// `pytraceback.py:30 self.lasti = lasti` — bytecode index at
    /// which the exception was raised (in instruction units).
    pub lasti: i64,
    /// `pytraceback.py:31 self.next = next` — head pointer to the
    /// preceding traceback in the chain (caller-side); `PY_NULL`
    /// terminates the chain.
    pub w_next: PyObjectRef,
    /// `pytraceback.py:32 self.lineno = lineno` — either a real
    /// source line number or `LINENO_NOT_COMPUTED`, in which case
    /// `get_lineno` calls `offset2lineno` to resolve it lazily
    /// (`pytraceback.py:34-37`).
    pub lineno: i64,
    /// Snapshot of the raising frame's `pycode` — kept alive via GC
    /// tracing so `tb_frame`-less traceback consumers (e.g.
    /// `write_traceback_chain`) can still read `source_path` /
    /// `obj_name` / `qualname` after the underlying `PyFrame` has
    /// been freed.  PyPy doesn't need this because its `PyFrame` is
    /// itself a W_Root; pyre's frame isn't, so the PyCode is
    /// the smallest GC-rooted handle that preserves the parity-
    /// visible metadata.
    pub w_code: PyObjectRef,
}

pub const PYTRACEBACK_FRAME_OFFSET: usize = std::mem::offset_of!(PyTraceback, frame);
pub const PYTRACEBACK_LASTI_OFFSET: usize = std::mem::offset_of!(PyTraceback, lasti);
pub const PYTRACEBACK_W_NEXT_OFFSET: usize = std::mem::offset_of!(PyTraceback, w_next);
pub const PYTRACEBACK_LINENO_OFFSET: usize = std::mem::offset_of!(PyTraceback, lineno);
pub const PYTRACEBACK_W_CODE_OFFSET: usize = std::mem::offset_of!(PyTraceback, w_code);

/// GC type id assigned to `PyTraceback`.  Pre-registered in
/// `pyre-jit/src/eval.rs` immediately after `PyCode`
/// (`W_CODE_GC_TYPE_ID = 43`), so it takes the next slot (44); the
/// registration site pins this with a `debug_assert_eq!`.
pub const PYTRACEBACK_GC_TYPE_ID: u32 = 44;

pub const PYTRACEBACK_OBJECT_SIZE: usize = std::mem::size_of::<PyTraceback>();

/// Two `PyObjectRef`-shaped slots are GC-traced — the chained
/// `w_next` traceback link and the `w_code` snapshot kept alive so
/// `source_path` / `obj_name` survive after the raising frame is
/// freed.  `frame` is a raw `*mut PyFrame` to a custom-walked
/// structure and `lasti`/`lineno` are scalar tags.
pub const PYTRACEBACK_GC_PTR_OFFSETS: [usize; 2] =
    [PYTRACEBACK_W_NEXT_OFFSET, PYTRACEBACK_W_CODE_OFFSET];

impl pyre_object::lltype::GcType for PyTraceback {
    fn type_id() -> u32 {
        PYTRACEBACK_GC_TYPE_ID
    }
    const SIZE: usize = PYTRACEBACK_OBJECT_SIZE;
}

/// Allocate a fresh traceback.  Mirrors
/// `pytraceback.py:27-32 PyTraceback.__init__`.
pub fn w_pytraceback_new(
    frame: *mut crate::pyframe::PyFrame,
    lasti: i64,
    w_next: PyObjectRef,
    lineno: i64,
    w_code: PyObjectRef,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_next);
    pyre_object::gc_roots::pin_root(w_code);

    let value = PyTraceback {
        ob_header: PyObject {
            ob_type: &PYTRACEBACK_TYPE as *const PyType,
            w_class: get_instantiate(&PYTRACEBACK_TYPE),
        },
        frame,
        lasti,
        w_next,
        lineno,
        w_code,
    };

    // Executing frames are GC-managed non-moving oldgen blocks
    // (`PYFRAME_GC_TYPE_ID`); to keep `tb_frame` alive across the
    // traceback's lifetime the traceback itself must be a GC object
    // whose `pytraceback_object_custom_trace` forwards the `frame`
    // edge.  Allocate into oldgen (non-moving — raw `*mut PyTraceback`
    // readers and the exception `w_traceback` chain hold bare
    // pointers), mirroring `FrameBox::new` (`pyframe.rs:307`).  Before
    // the GC hook is wired (bootstrap, tests) `try_gc_alloc_stable`
    // returns `None`; fall back to the leaked `malloc_typed` block.
    let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
        PYTRACEBACK_GC_TYPE_ID,
        PYTRACEBACK_OBJECT_SIZE,
    );
    if !raw.is_null() {
        pyre_object::gc_interp::note_alloc();
        let ptr = raw as *mut PyTraceback;
        unsafe {
            std::ptr::write(ptr, value);
        }
        // The oldgen traceback references the freshly-born `w_next` /
        // `w_code` (and, once GC-owned, the frame); remember it for the
        // next minor tracer.
        pyre_object::gc_hook::try_gc_write_barrier(raw);
        return ptr as PyObjectRef;
    }

    pyre_object::lltype::malloc_typed(value) as PyObjectRef
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_pytraceback(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PYTRACEBACK_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_frame(obj: PyObjectRef) -> *mut crate::pyframe::PyFrame {
    unsafe { (*(obj as *const PyTraceback)).frame }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_lasti(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const PyTraceback)).lasti }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_set_lasti(obj: PyObjectRef, value: i64) {
    unsafe { (*(obj as *mut PyTraceback)).lasti = value }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_w_next(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const PyTraceback)).w_next }
}

/// `pytraceback.py:54-62 descr_set_next` — loop-check before writing.
/// Raises `ValueError("traceback loop detected")` when the proposed
/// `w_new_next` chain reaches `obj` itself.
///
/// # Safety
/// `obj` must point to a valid `PyTraceback`.  `w_new_next` is
/// either `PY_NULL` (chain terminator) or a valid `PyTraceback`.
pub unsafe fn w_pytraceback_set_w_next(
    obj: PyObjectRef,
    w_new_next: PyObjectRef,
) -> Result<(), ()> {
    unsafe {
        let mut curr = w_new_next;
        while !curr.is_null() && is_pytraceback(curr) {
            if std::ptr::eq(curr, obj) {
                return Err(());
            }
            curr = w_pytraceback_get_w_next(curr);
        }
        (*(obj as *mut PyTraceback)).w_next = w_new_next;
    }
    Ok(())
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_lineno_raw(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const PyTraceback)).lineno }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_set_lineno(obj: PyObjectRef, value: i64) {
    unsafe { (*(obj as *mut PyTraceback)).lineno = value }
}

/// # Safety
/// `obj` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_w_code(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const PyTraceback)).w_code }
}

/// `pytraceback.py:34-40 PyTraceback.get_lineno` /
/// `descr_get_tb_lineno`:
///
/// ```python
/// def get_lineno(self):
///     if self.lineno == LINENO_NOT_COMPUTED:
///         self.lineno = offset2lineno(self.frame.pycode, self.lasti)
///     return self.lineno
///
/// def descr_get_tb_lineno(self, space):
///     return space.newint(self.get_lineno())
/// ```
///
/// Pyre stamps the real line number at `record_application_traceback`
/// time (the frame is guaranteed live there), so this reader never
/// has to walk back through `self.frame.pycode` — which would be
/// unsafe in pyre because `PyFrame` is not a GC-traced W_Root and
/// the frame may have been freed by the time `tb_lineno` is read.
/// The `LINENO_NOT_COMPUTED` sentinel still surfaces as `-1` for
/// the edge case where the traceback was constructed without a
/// frame (e.g. unit tests).
///
/// # Safety
/// `tb` must point to a valid `PyTraceback`.
#[inline]
pub unsafe fn w_pytraceback_get_lineno(tb: PyObjectRef) -> i64 {
    unsafe {
        let raw = w_pytraceback_get_lineno_raw(tb);
        if raw == LINENO_NOT_COMPUTED { -1 } else { raw }
    }
}

/// `pytraceback.py:104-109 record_application_traceback` parity:
///
/// ```python
/// def record_application_traceback(space, operror, frame, last_instruction):
///     if frame.pycode.hidden_applevel:
///         return
///     tb = operror.get_traceback()
///     tb = PyTraceback(space, frame, last_instruction, tb)
///     operror.set_traceback(tb)
/// ```
///
/// Pyre stores the chain head on the materialised `W_BaseException`'s
/// `w_traceback` slot (the same slot
/// `interp_exceptions.rs:303 w_exception_set_traceback` writes to).  The
/// operror-side `_application_traceback: Option<PyObjectRef>` cache
/// mirrors the slot for `to_exc_object` callers that haven't allocated
/// the exception yet.
///
/// `last_instruction` is the byte-offset of the in-flight opcode
/// (`pyframe.py:72 self.last_instr`).  In RPython this is the
/// instruction-unit index; pyre stores `last_instr` in bytes for now,
/// matching `pyframe::PyFrame.last_instr` documentation
/// (`pyframe.rs:55-77`).
///
/// # Safety
/// `w_exc_object` must point to a valid `W_BaseException` (or
/// `PY_NULL`, in which case the call is a no-op).  `frame` must be a
/// valid live `PyFrame`.
pub unsafe fn record_application_traceback(
    w_exc_object: PyObjectRef,
    frame: *mut crate::pyframe::PyFrame,
    last_instruction: i64,
) {
    if w_exc_object.is_null() || frame.is_null() {
        return;
    }
    unsafe {
        // `pycode.py:111 self.hidden_applevel` — pyre's
        // `PyCode.hidden_applevel` flag (`pycode.rs:51`) skips
        // gateway / app_main bridge frames from the traceback.
        let pycode_ptr = (*frame).pycode as *const crate::pycode::PyCode;
        if !pycode_ptr.is_null() && (*pycode_ptr).hidden_applevel {
            return;
        }
        if !pyre_object::is_exception(w_exc_object) {
            return;
        }
        // `pytraceback.py:36 self.lineno = offset2lineno(self.frame
        // .pycode, self.lasti)` — pyre resolves the line number
        // eagerly here (rather than lazily in `get_lineno`) because
        // pyframe.rs's `PyFrame` is not a GC-traced W_Root, so by
        // the time `tb_lineno` is read the frame may already be
        // freed.  Stamping at construction guarantees the value
        // survives the frame's lifetime.  `frame.pycode` is the
        // `PyCode` wrapper; the inner `CodeObject` is
        // extracted via `pyframe_get_pycode`.
        //
        // The `PyCode` PyObjectRef is also captured into the
        // `w_code` slot so the traceback's source-path / function
        // name metadata stays GC-rooted after the raising frame's
        // freed — readers (e.g. `write_traceback_chain` in
        // `error.rs`) MUST go through `w_code` rather than
        // dereferencing the dangling `frame` pointer.
        let w_code = (*frame).pycode as PyObjectRef;
        let lineno = {
            let code_obj = crate::pyframe::pyframe_get_pycode(&*frame);
            if code_obj.is_null() {
                LINENO_NOT_COMPUTED
            } else {
                crate::pyframe::offset2lineno(&*code_obj, last_instruction as isize) as i64
            }
        };
        let prev_tb = pyre_object::interp_exceptions::w_exception_get_traceback(w_exc_object);
        let new_tb = w_pytraceback_new(frame, last_instruction, prev_tb, lineno, w_code);
        pyre_object::interp_exceptions::w_exception_set_traceback(w_exc_object, new_tb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytraceback_gc_type_id_matches_descr() {
        assert_eq!(PYTRACEBACK_GC_TYPE_ID, 44);
        assert_eq!(
            <PyTraceback as pyre_object::lltype::GcType>::type_id(),
            PYTRACEBACK_GC_TYPE_ID
        );
        assert_eq!(
            <PyTraceback as pyre_object::lltype::GcType>::SIZE,
            PYTRACEBACK_OBJECT_SIZE
        );
    }

    #[test]
    fn pytraceback_alloc_and_accessors() {
        let tb = w_pytraceback_new(
            std::ptr::null_mut(),
            42,
            PY_NULL,
            LINENO_NOT_COMPUTED,
            PY_NULL,
        );
        unsafe {
            assert!(is_pytraceback(tb));
            assert_eq!(w_pytraceback_get_lasti(tb), 42);
            assert!(w_pytraceback_get_w_next(tb).is_null());
            assert_eq!(w_pytraceback_get_lineno_raw(tb), LINENO_NOT_COMPUTED);
            assert!(w_pytraceback_get_frame(tb).is_null());
            assert!(w_pytraceback_get_w_code(tb).is_null());
        }
    }

    #[test]
    fn pytraceback_set_next_self_loop_rejects() {
        let tb = w_pytraceback_new(std::ptr::null_mut(), 0, PY_NULL, 0, PY_NULL);
        unsafe {
            assert!(w_pytraceback_set_w_next(tb, tb).is_err());
            assert!(w_pytraceback_get_w_next(tb).is_null());
        }
    }

    #[test]
    fn pytraceback_set_next_chain_loop_rejects() {
        // Chain: outer -> inner -> outer should be rejected.
        let outer = w_pytraceback_new(std::ptr::null_mut(), 0, PY_NULL, 0, PY_NULL);
        let inner = w_pytraceback_new(std::ptr::null_mut(), 1, outer, 0, PY_NULL);
        unsafe {
            // outer.w_next = inner — inner.w_next is outer → cycle.
            assert!(w_pytraceback_set_w_next(outer, inner).is_err());
            assert!(w_pytraceback_get_w_next(outer).is_null());
        }
    }

    #[test]
    fn pytraceback_set_next_chain_ok() {
        let inner = w_pytraceback_new(std::ptr::null_mut(), 1, PY_NULL, 0, PY_NULL);
        let outer = w_pytraceback_new(std::ptr::null_mut(), 0, PY_NULL, 0, PY_NULL);
        unsafe {
            assert!(w_pytraceback_set_w_next(outer, inner).is_ok());
            assert_eq!(w_pytraceback_get_w_next(outer), inner);
        }
    }
}
