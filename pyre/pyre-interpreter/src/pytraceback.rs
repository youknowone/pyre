//! `pypy/interpreter/pytraceback.py PyTraceback` line-by-line port.
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
//! Upstream `frame` is an ordinary traced field of an ordinary movable
//! instance: `pytraceback.py self.frame = frame` on a
//! `baseobjspace.W_Root`, pointing at `pyframe.py:52 class
//! PyFrame(W_Root)`, which declares no `_alloc_flavor_` and is never
//! pinned — `rpython/rlib/rgc.py` documents `pin` as a
//! short-lived-buffer facility that does not extend lifetime, and
//! nothing under `pypy/interpreter/` calls it.  A minor collection
//! copies the frame (`rpython/memory/gc/incminimark.py:2237`) and
//! rewrites every referring slot in place (`:2252`), because roots
//! reach the collector as slot addresses
//! (`rpython/memory/gctransform/shadowstack.py:43-46`) and compiled
//! code re-reads the frame after each collecting call
//! (`rpython/jit/backend/x86/assembler.py:1369-1377`).
//!
//! Pyre's `PyFrame` does carry the `PyObject` prefix (its `ob_header`
//! field), so `tb_frame` returns a Python-visible object.  What
//! diverges is the slot type: this struct holds a raw `*mut PyFrame`
//! rather than a `PyObjectRef`, and the edge reaches the collector
//! only through the hand-written `pytraceback_object_custom_trace`
//! hook, which forwards it as a mutable slot.
//!
//! The pointee must additionally never move.  That is not an upstream
//! requirement and nothing in this file causes it: an executing frame
//! is reached through a live Rust `&mut PyFrame` that spans every
//! allocation the running opcode performs, and RPython's translator
//! rewrites exactly those live references in their shadow-stack slots
//! (`rpython/memory/gctransform/shadowstack.py:43-46`) where Rust has
//! no equivalent pass.  `FrameBox::new` allocating non-moving stands
//! in for that rewrite, and this file's conditional frame edge and
//! `w_code` snapshot are downstream of it.
//!
//! The rule holds for frames that reach here, not for every frame: a
//! compiled trace's own inlined-callee frame is a nursery allocation,
//! and making that uniform times `fib_recursive` out of its gate, so
//! the crossing is guarded at the seams instead (see `gate-triage.md`).
//! `record_application_traceback` is one of those seams.

use pyre_object::pyobject::*;

/// `pytraceback.py` `LINENO_NOT_COMPUTED = -sys.maxint-1` —
/// sentinel meaning "please take the lineno from the frame and
/// `lasti`".
///
/// The value is `-1`, not upstream's `-sys.maxint-1`, because the sentinel is
/// app-level visible through the fourth constructor argument and `-1` is the
/// one 3.14 measures it against: `tb_lineno_get` resolves on
/// `if (lineno == -1)`, and `_PyTraceBack_FromFrame` records a node with
/// `tb_create_raw(..., -1)`.  Upstream's comment explains its own choice —
/// negative line numbers are settable there, so it took the most negative
/// value to keep a written line number from colliding with the sentinel — and
/// 3.14 answered the same worry by making `tb_lineno` read-only instead.  Pyre
/// keeps the setter, so a written `-1` re-resolves here where 3.14 would raise
/// `AttributeError`; every value 3.14 can actually produce reads the same.
pub const LINENO_NOT_COMPUTED: i64 = -1;

pub static PYTRACEBACK_TYPE: PyType = new_pytype("traceback");

/// Layout: `[ob_header | frame: *mut PyFrame | lasti: i64 | w_next:
/// PyObjectRef | lineno: i64 | w_code: PyObjectRef]`.
///
/// Every reference slot is traced, but by the hand-written
/// `pytraceback_object_custom_trace` rather than by an offsets table:
/// `TypeInfo::object_subclass_with_custom_trace` leaves
/// `gc_ptr_offsets` empty for every type that supplies a hook.
#[repr(C)]
pub struct PyTraceback {
    pub ob_header: PyObject,
    /// `pytraceback.py self.frame = frame` — a raw `*mut PyFrame`
    /// rather than a `PyObjectRef`, so it takes part in collection
    /// only through `pytraceback_object_custom_trace`.  That hook
    /// forwards the slot when the GC owns the frame, which keeps it
    /// live and would rewrite it if the frame ever moved.  Every frame
    /// that can reach here is GC-owned: `FrameBox::new` allocates from
    /// the GC whenever the hook is installed, and the one path that is
    /// not — the pre-hook bootstrap window — collects nothing, so a
    /// frame born there is never swept either way.
    pub frame: *mut crate::pyframe::PyFrame,
    /// `pytraceback.py self.lasti = lasti` — where in the bytecode the
    /// exception was raised, in the units `tb_lasti` is defined in: bytes,
    /// two per instruction.  `descr_get_tb_lasti` hands this straight out
    /// (`pytraceback.py:45-46`), and it has to be the byte form for
    /// `traceback._get_code_position`, which recovers the instruction with
    /// `tb_lasti // 2`.  Pyre's own producers count instructions, so
    /// `record_application_traceback` converts; its two readers convert back.
    pub lasti: i64,
    /// `pytraceback.py self.next = next` — head pointer to the
    /// preceding traceback in the chain (caller-side); `PY_NULL`
    /// terminates the chain.
    pub w_next: PyObjectRef,
    /// `pytraceback.py self.lineno = lineno` — either a real
    /// source line number or `LINENO_NOT_COMPUTED`, in which case
    /// `w_pytraceback_get_lineno` resolves it from `w_code` and `lasti`.
    pub lineno: i64,
    /// Snapshot of the raising frame's `pycode`, with no upstream
    /// counterpart — `pytraceback.py` reads `self.frame.pycode`
    /// directly (`:36`), because upstream's frame edge is unconditional
    /// and the frame therefore outlives the traceback.  Pyre's edge is
    /// conditional on the GC owning the frame, so consumers that must
    /// keep working otherwise (`write_traceback_chain`) read
    /// `source_path` / `obj_name` / `qualname` through this handle
    /// instead.
    ///
    /// The condition now excludes only the pre-hook bootstrap frame:
    /// the tracer snapshot, which used to be the reachable case, is
    /// GC-owned since `snapshot_for_tracing` moved to `FrameBox::new`,
    /// and a full-corpus probe on the forwarding hook saw no non-GC
    /// frame reach a traceback.  Retiring the field would still mean
    /// deleting a guard whose failure mode is handing the collector a
    /// `std::alloc` address, on evidence that is an absence rather than
    /// a proof, so it stays.
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

impl pyre_object::lltype::GcType for PyTraceback {
    fn type_id() -> u32 {
        PYTRACEBACK_GC_TYPE_ID
    }
    const SIZE: usize = PYTRACEBACK_OBJECT_SIZE;
}

/// Allocate a fresh traceback.  Mirrors
/// `pytraceback.py PyTraceback.__init__`.
pub fn w_pytraceback_new(
    frame: *mut crate::pyframe::PyFrame,
    lasti: i64,
    w_next: PyObjectRef,
    lineno: i64,
    w_code: PyObjectRef,
) -> PyObjectRef {
    // `frame` is pinned alongside the two managed fields, because the
    // allocation below can safepoint and a raw `*mut PyFrame` held only in
    // this function's locals is reachable from no root walker.  Most frames
    // are allocated non-moving (`FrameBox::new`), which is what lets raw
    // copies exist elsewhere at all — `FrameBox::deref` reads its raw field
    // while holding a forwarding-capable `owner_root` it never reads back,
    // `eval_loop` runs behind a `&mut PyFrame` across a safepoint, and the
    // blackhole keeps the virtualizable as a bare integer.  It is not every
    // frame: a compiled trace's inlined-callee frame is a nursery
    // allocation, so a minor collection triggered by this very allocation
    // recycles it and any slot built from a pre-allocation copy names freed
    // bytes for the rest of the node's life.  Upstream needs no bracket
    // here — a minor relocates the frame and rewrites the slot
    // (`incminimark.py:2237` / `:2252`).
    let roots = pyre_object::gc_roots::push_roots();
    let inputs = pyre_object::gc_roots::pin_roots(&[w_next, w_code, frame as PyObjectRef]);

    // This host-side constructor allocates the traceback itself into oldgen:
    // its Rust caller can hold the returned pointer outside a translated
    // GC-map slot before publishing it. JIT-emitted traceback nodes do not
    // have that restriction: their live refs are GC-map roots or traced
    // object fields, so their size descriptor keeps the ordinary movable
    // nursery placement used upstream. Before the GC hook is wired
    // (bootstrap, tests) `try_gc_alloc_stable` returns `None`; fall
    // back to the leaked `malloc_typed` block.
    let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
        PYTRACEBACK_GC_TYPE_ID,
        PYTRACEBACK_OBJECT_SIZE,
    );
    if !raw.is_null() {
        // The fresh block is a root before `get_instantiate` below can enter
        // an allocation of its own, matching `FrameBox::new`'s pin of its own
        // result.  The block is non-moving, so `raw` stays the address.
        let _ = pyre_object::gc_roots::pin_root(raw as PyObjectRef);
    }

    // Every input is read back through the bracket's own cell rather than
    // reused from the argument: the allocation above may have moved any of
    // them.  Before the GC hook is wired nothing moves and each read answers
    // the address it was given.
    let value = PyTraceback {
        ob_header: PyObject {
            ob_type: &PYTRACEBACK_TYPE as *const PyType,
            w_class: get_instantiate(&PYTRACEBACK_TYPE),
        },
        frame: roots.get(inputs + 2) as *mut crate::pyframe::PyFrame,
        lasti,
        w_next: roots.get(inputs),
        lineno,
        w_code: roots.get(inputs + 1),
    };

    if raw.is_null() {
        return pyre_object::lltype::malloc_typed(value) as PyObjectRef;
    }

    let ptr = raw as *mut PyTraceback;
    unsafe {
        std::ptr::write(ptr, value);
    }
    // The oldgen traceback references the freshly-born `w_next` /
    // `w_code` (and, once GC-owned, the frame); remember it for the
    // next minor tracer.
    pyre_object::gc_hook::try_gc_write_barrier(raw);
    ptr as PyObjectRef
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

/// `pytraceback.py descr_set_next` — loop-check before writing.
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
        // An older `obj` now names a possibly younger `w_new_next`; remember it
        // for the next minor tracer, as the allocation above does for the
        // fields written at birth.
        pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
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

/// `pytraceback.py PyTraceback.get_lineno` /
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
/// The resolution walks the `w_code` snapshot rather than upstream's
/// `self.frame.pycode`: `frame` is only forwarded for a frame the GC owns, so
/// it may already have been freed by the time `tb_lineno` is read, while
/// `w_code` is forwarded unconditionally and IS that `pycode`.
///
/// `None` is the `Py_RETURN_NONE` arm of `tb_lineno_get`, taken when
/// `PyCode_Addr2Line` reports no line for `tb_lasti` — here also when the node
/// carries no code object at all, which upstream cannot reach because its
/// frame edge is unconditional.
///
/// The resolved line is NOT written back to the slot.  Upstream memoizes
/// (`self.lineno = offset2lineno(...)`); `tb_lineno_get` re-reads instead, and
/// re-reading is what keeps the slot single-writer for the JIT fold
/// `walker_specialize_traceback_walk_field`, which folds the raw slot against
/// a guard that it is not the sentinel.
///
/// `record_application_traceback` still stamps the line eagerly, so a recorded
/// node reaches the first branch and never resolves here.  That timing is not
/// observable against 3.14: `tb_lasti` and `tb_lineno` are both read-only
/// there, so the only way to hand a live node a sentinel is the constructor,
/// which lands in the second branch either way.
///
/// # Safety
/// `tb` must point to a valid `PyTraceback`.
pub unsafe fn w_pytraceback_get_lineno(tb: PyObjectRef) -> Option<i64> {
    unsafe {
        let raw = w_pytraceback_get_lineno_raw(tb);
        if raw != LINENO_NOT_COMPUTED {
            return Some(raw);
        }
        let w_code = w_pytraceback_get_w_code(tb);
        if w_code.is_null() {
            return None;
        }
        let code = crate::w_code_get_ptr(w_code) as *const crate::CodeObject;
        if code.is_null() {
            return None;
        }
        crate::pycode::w_code_addr2line(&*code, w_pytraceback_get_lasti(tb)).map(|line| line as i64)
    }
}

/// The side effect shared by `error.py OperationError.get_traceback`
/// and `interp_exceptions.py descr_gettraceback`: a traceback that
/// becomes reachable by app-level code marks its frame escaped, so
/// `ExecutionContext::leave` forces that frame's vref and the frame stays
/// inspectable after the JIT frame is gone.
///
/// This has no effect once several frames are recorded on the chain — those
/// are already marked by `leave` running with `got_exception` set.  What it
/// covers is the single-frame chain: a `raise` caught in the frame that
/// raised it, where no callee ever left, whose traceback then outlives a
/// normal return.  Without the mark such a frame keeps the vable shadow of
/// `last_instr` the JIT never wrote back, and `f_lineno` reads the `def`
/// line instead of the raising line.
///
/// # Safety
/// `w_traceback` must be `PY_NULL` or point to a valid object.
pub unsafe fn mark_traceback_escaped(w_traceback: PyObjectRef) {
    unsafe {
        if w_traceback.is_null() || !is_pytraceback(w_traceback) {
            return;
        }
        let frame = w_pytraceback_get_frame(w_traceback);
        if !frame.is_null() {
            (*frame).mark_as_escaped();
        }
    }
}

/// `pytraceback.py record_application_traceback` parity:
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
/// Upstream keys the two reads off the operror, whose
/// `_application_traceback` field holds the chain beside `_w_value`.  Pyre
/// keys them off the materialised `W_BaseException`'s `w_traceback` slot
/// instead, and takes that object rather than the `PyError`: a propagating
/// Rust `PyError` is memcpy'd at every `?` and so cannot be registered as a
/// GC root, while the instance can, which is what `set_in_flight_exception`
/// below relies on.  [`PyError::get_traceback`](crate::PyError::get_traceback)
/// and [`PyError::set_traceback`](crate::PyError::set_traceback) are the
/// carrier-level spelling of the same two slot accesses, for callers that do
/// hold the error; this entry point is also the one the JIT seam reaches
/// across an ABI boundary carrying only the exception value.
///
/// `last_instruction` is the byte-offset of the in-flight opcode
/// (`pyframe.py:72 self.last_instr`).  In RPython this is the
/// instruction-unit index; pyre stores `last_instr` in bytes for now,
/// matching `pyframe::PyFrame.last_instr` documentation
/// (`pyframe.rs`).
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
        // `pycode.py self.hidden_applevel` — pyre's
        // `PyCode.hidden_applevel` flag (`pycode.rs`) skips
        // gateway / app_main bridge frames from the traceback.
        let pycode_ptr = (*frame).pycode as *const crate::pycode::PyCode;
        if !pycode_ptr.is_null() && (*pycode_ptr).hidden_applevel {
            return;
        }
        if !pyre_object::is_exception(w_exc_object) {
            return;
        }
        // Keep the exception now being propagated GC-reachable: until a frame
        // catches it, it lives only in the in-flight Rust `PyError`, so a
        // safepoint's non-moving major would otherwise sweep its old-gen
        // traceback chain (`tstate->current_exception` parity).
        crate::eval::set_in_flight_exception(w_exc_object);
        // `pytraceback.py self.lineno = offset2lineno(self.frame
        // .pycode, self.lasti)` — pyre resolves the line number eagerly
        // here rather than leaving the sentinel for the getter, so the
        // slot never holds `LINENO_NOT_COMPUTED` for a node built here.
        // `_PyTraceBack_FromFrame` records the sentinel instead and
        // `tb_lineno_get` resolves it, but a node's `tb_lasti` and
        // `tb_lineno` are both read-only there, so which of the two
        // moments does the walk is not app-level observable.
        //
        // What the eager stamp buys is the JIT fold
        // `walker_specialize_traceback_walk_field` (pyre-jit-trace): it
        // reads this slot directly and declines on the sentinel, so a
        // node that carried the sentinel would decline on every read of
        // its line.  Frame lifetime is not part of it — the `w_code`
        // slot below is forwarded unconditionally and is the same
        // `pycode` upstream reads, which is what makes the getter's
        // resolution safe at any later point.
        //
        // `frame.pycode` is the `PyCode` wrapper; the inner
        // `CodeObject` is extracted via `pyframe_get_pycode`.
        //
        // The `PyCode` PyObjectRef is also captured into the `w_code`
        // slot so the traceback's source-path / function name metadata
        // stays GC-rooted in that same case — readers (e.g.
        // `write_traceback_chain` in `error.rs`) MUST go through
        // `w_code` rather than dereferencing the `frame` pointer.
        let w_code = (*frame).pycode as PyObjectRef;
        let lineno = {
            let code_obj = crate::pyframe::pyframe_get_pycode(&*frame);
            if code_obj.is_null() {
                LINENO_NOT_COMPUTED
            } else {
                crate::pyframe::offset2lineno(&*code_obj, last_instruction as isize) as i64
            }
        };
        // `tb = operror.get_traceback()` — the read that grows the chain
        // marks the previous head's frame, matching `get_traceback`.
        let prev_tb = pyre_object::interp_exceptions::w_exception_get_traceback(w_exc_object);
        mark_traceback_escaped(prev_tb);
        // `last_instruction` counts instructions; the slot holds bytes.
        let new_tb = w_pytraceback_new(frame, last_instruction * 2, prev_tb, lineno, w_code);
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

    /// The sentinel is the value the fourth constructor argument is measured
    /// against, so it has to be the one `tb_lineno_get` tests for.
    #[test]
    fn the_sentinel_is_the_one_the_constructor_can_hand_in() {
        assert_eq!(LINENO_NOT_COMPUTED, -1);
    }

    /// A node carrying no code object cannot resolve, which is the
    /// `Py_RETURN_NONE` arm; a stamped line is handed back as it is, including
    /// the negative values `descr_set_tb_lineno` accepts.
    #[test]
    fn an_unresolvable_node_answers_none_and_a_stamped_line_answers_itself() {
        unsafe {
            let unresolvable = w_pytraceback_new(
                std::ptr::null_mut(),
                42,
                PY_NULL,
                LINENO_NOT_COMPUTED,
                PY_NULL,
            );
            assert_eq!(w_pytraceback_get_lineno(unresolvable), None);

            let stamped = w_pytraceback_new(std::ptr::null_mut(), 42, PY_NULL, 7, PY_NULL);
            assert_eq!(w_pytraceback_get_lineno(stamped), Some(7));

            // `-2` is not the sentinel, so it reads back rather than resolving.
            let negative = w_pytraceback_new(std::ptr::null_mut(), 42, PY_NULL, -2, PY_NULL);
            assert_eq!(w_pytraceback_get_lineno(negative), Some(-2));

            let zero = w_pytraceback_new(std::ptr::null_mut(), 42, PY_NULL, 0, PY_NULL);
            assert_eq!(w_pytraceback_get_lineno(zero), Some(0));
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
