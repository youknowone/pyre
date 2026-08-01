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
//! Upstream `frame` is an ordinary traced field of an ordinary movable
//! instance: `pytraceback.py:29 self.frame = frame` on a
//! `baseobjspace.W_Root`, pointing at `pyframe.py:52 class
//! PyFrame(W_Root)`, which declares no `_alloc_flavor_` and is never
//! pinned — `rpython/rlib/rgc.py:88-97` documents `pin` as a
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
//! hook, which forwards it as a mutable slot but skips a frame the GC
//! does not own (a `FrameBox::new_boxed` tracer snapshot).
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
//! and making that uniform costs `fib_recursive` 7.75x, so the
//! crossing is guarded at the seams instead (see `gate-triage.md`).
//! `record_application_traceback` is one of those seams.

use pyre_object::pyobject::*;

/// `pytraceback.py:12` `LINENO_NOT_COMPUTED = -sys.maxint-1` —
/// sentinel meaning "please take the lineno from the frame and
/// `lasti`".  Pyre uses `i64::MIN` to match RPython's `-sys.maxint-1`
/// idiom (`pytraceback.py:9-12`).
pub const LINENO_NOT_COMPUTED: i64 = i64::MIN;

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
    /// `pytraceback.py:29 self.frame = frame` — a raw `*mut PyFrame`
    /// rather than a `PyObjectRef`, so it takes part in collection
    /// only through `pytraceback_object_custom_trace`.  That hook
    /// forwards the slot when the GC owns the frame, which keeps it
    /// live and would rewrite it if the frame ever moved; it skips a
    /// frame the GC does not own — a `FrameBox::new_boxed` tracer
    /// snapshot, freed at the end of its walk — and such a pointer is
    /// left dangling and must not be dereferenced.  The `w_code`
    /// snapshot below is what readers use in that case.
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
    /// Snapshot of the raising frame's `pycode`, with no upstream
    /// counterpart — `pytraceback.py` reads `self.frame.pycode`
    /// directly (`:36`), because upstream's frame edge is unconditional
    /// and the frame therefore outlives the traceback.  Pyre's does not
    /// hold for a frame the GC does not own, so consumers that must
    /// keep working then (`write_traceback_chain`) read `source_path` /
    /// `obj_name` / `qualname` through this handle instead.  Retiring
    /// the field is blocked on making every frame that can reach a
    /// traceback GC-owned.
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

    // The rule below cannot be spelled as a `debug_assert!` here: this crate
    // is extracted to LLBC, so an assertion lands in the JIT's view of this
    // function and its probe becomes a real call on every traceback the
    // traced code builds — measurably, on the `getattr_*` gates.  It stays a
    // stated obligation.
    //
    // `frame` was copied into `value` above and is not among the roots
    // pinned here, so the allocation below — which can safepoint — must
    // not be able to move it.  Upstream has no such rule: a minor
    // collection would relocate the frame and rewrite the slot
    // (`incminimark.py:2237` / `:2252`).  Pyre cannot, because raw
    // `*mut PyFrame` copies exist that no root walker reaches —
    // `FrameBox::deref` reads its raw field while holding a
    // forwarding-capable `owner_root` it never reads back, `eval_loop`
    // runs behind a `&mut PyFrame` across a safepoint, and the
    // blackhole keeps the virtualizable as a bare integer.  Frames are
    // therefore allocated non-moving (`FrameBox::new`), and callers of
    // `record_application_traceback` owe this function a frame that
    // stays reachable across the allocation: on the `CURRENT_FRAME` /
    // `f_backref` chain, or pinned by hand.
    //
    // Allocate the traceback itself into oldgen for the same reason —
    // raw `*mut PyTraceback` readers and the exception `w_traceback`
    // chain hold bare pointers.  Before the GC hook is wired
    // (bootstrap, tests) `try_gc_alloc_stable` returns `None`; fall
    // back to the leaked `malloc_typed` block.
    let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
        PYTRACEBACK_GC_TYPE_ID,
        PYTRACEBACK_OBJECT_SIZE,
    );
    if !raw.is_null() {
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
/// time instead of resolving it here.  The upstream walk goes through
/// `self.frame.pycode`, which this reader cannot do: `frame` is only
/// forwarded for a frame the GC owns, so it may already have been
/// freed by the time `tb_lineno` is read.  The `w_code` snapshot is
/// forwarded unconditionally and IS that `pycode`, so `offset2lineno`
/// off `w_code` and `lasti` would be safe to run lazily.  What still
/// depends on the eager stamp is the JIT fold
/// `walker_specialize_traceback_walk_field` (pyre-jit-trace): it reads
/// this slot directly and declines on the sentinel, so under a lazy
/// `get_lineno` every fresh node would decline on its first read.
///
/// Two `tb_lineno` answers diverge from `get_lineno` because of it.
/// Upstream re-resolves from the CURRENT `lasti` whenever the slot
/// still holds the sentinel, so `tb.tb_lasti = N; tb.tb_lineno` reads
/// the new offset there and the originally-stamped line here; and a
/// sentinel written back through `TracebackType(..., -sys.maxsize-1)`
/// or the `tb_lineno` setter is re-resolved there and answered `-1`
/// here.  Porting back to lazy needs the fold to call a resolver
/// rather than decline.  The sentinel also still surfaces as `-1` for
/// a traceback constructed without a frame (e.g. unit tests).
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

/// The side effect shared by `error.py:359-370 OperationError.get_traceback`
/// and `interp_exceptions.py:195-200 descr_gettraceback`: a traceback that
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

/// `extern "C"` entry for the residual call the `__traceback__` attribute
/// fold emits.  Compiled code folds that read to a raw slot load, so the
/// getter's escape mark has to be issued separately; the fold pairs the
/// load with a call here.
///
/// Cannot raise, allocates nothing, and writes only the frame's status
/// byte, which no field descriptor exposes to the trace — so the call
/// carries `cannot_raise_effect_info` and invalidates no heap cache entry.
pub extern "C" fn jit_mark_traceback_escaped(w_traceback: i64) {
    unsafe { mark_traceback_escaped(w_traceback as usize as PyObjectRef) };
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
        // Keep the exception now being propagated GC-reachable: until a frame
        // catches it, it lives only in the in-flight Rust `PyError`, so a
        // safepoint's non-moving major would otherwise sweep its old-gen
        // traceback chain (`tstate->current_exception` parity).
        crate::eval::set_in_flight_exception(w_exc_object);
        // `pytraceback.py:36 self.lineno = offset2lineno(self.frame
        // .pycode, self.lasti)` — pyre resolves the line number
        // eagerly here rather than lazily in `get_lineno`, so the slot
        // never holds `LINENO_NOT_COMPUTED` for a node built here.
        //
        // Frame lifetime is NOT what blocks the lazy form: the `w_code`
        // slot below is forwarded unconditionally and is the same
        // `pycode` upstream reads, so resolving off `w_code` + `lasti`
        // would be safe at any later point.  What blocks it is the JIT
        // fold `walker_specialize_traceback_walk_field`
        // (pyre-jit-trace), which reads this slot directly and declines
        // on the sentinel; under a lazy `get_lineno` every freshly
        // recorded node carries the sentinel on its first read, so the
        // fold would have to emit a resolver call plus the memoizing
        // write-back in place of today's plain `getfield`.  That trades
        // a folded field read for a call on the `tb_lineno` walk.
        // See `w_pytraceback_get_lineno` for the two `tb_lineno`
        // answers the eager stamp diverges on.
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
