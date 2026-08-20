//! JIT-specific call infrastructure — force/bridge callbacks, callee
//! frame creation helpers, frame pool.
//!
//! Separated from pyre-interpreter/src/call.rs so pyre-interpreter stays JIT-free.

use pyre_interpreter::{locals_w, locals_w_mut};
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::Once;

/// Whether `PYRE_NBODY_DEBUG` is set, cached at first access.
///
/// `std::env::var_os` acquires a global env lock on every call. Caching
/// here matches the equivalent helpers in `majit-backend-cranelift` and
/// `majit-backend-dynasm`. These probes were added during nbody bring-up
/// and are not part of PyPy.
fn pyre_nbody_debug_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("PYRE_NBODY_DEBUG").is_some());
    *ENABLED
}

/// Whether `MAJIT_PROBE_LIVENESS` is set, cached at first access.
#[allow(dead_code)]
fn majit_probe_liveness_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_PROBE_LIVENESS").is_some());
    *ENABLED
}

/// Whether `PYRE_PROBE_BH_STARTUP=1` is set, cached at first access.
#[allow(dead_code)]
fn pyre_probe_bh_startup_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("PYRE_PROBE_BH_STARTUP").ok().as_deref() == Some("1")
    });
    *ENABLED
}

use pyre_interpreter::bytecode::{Instruction, OpArgState};
use pyre_interpreter::{
    PyResult, function_get_closure, function_get_defaults, function_get_globals_obj,
    function_get_name, is_function, register_jit_exc_raiser, register_jit_function_caller,
};
use pyre_object::intobject::w_int_get_value;
use pyre_object::intobject::w_int_new;
use pyre_object::pyobject::is_int;
use pyre_object::{PY_NULL, PyObjectRef};

use majit_ir::GcRef;
use pyre_interpreter::pyframe::PyFrame;
use pyre_jit_trace::trace::trace_bytecode;

// Force cache removed: CallAssemblerI + bridge handles recursion
// natively without memoization.

thread_local! {
    /// Stash Python exceptions from blackhole/force paths that cross
    /// FFI boundaries (compiled code → callback → exception).
    static LAST_CA_EXCEPTION: std::cell::RefCell<Option<pyre_interpreter::error::PyError>> =
        const { std::cell::RefCell::new(None) };
    /// Callee PyFrame address whose CALL_ASSEMBLER bridge walk committed
    /// and adopted its end-of-walk state (raise_continue_running_normally
    /// analogue). The CA slow path calls the bridge hook and then the
    /// blackhole hook back-to-back; the blackhole consumes this to
    /// complete the callee from the adopted state instead of re-running
    /// the guard-state resume over already-applied effects.
    static CA_WALK_ADOPTED_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    /// Callee PyFrame address whose CALL_ASSEMBLER bridge walk terminated
    /// with a kept finish-concrete stash (the walked callee ran to its
    /// return/uncaught raise concretely).  The CA bridge hook returns a
    /// bare bool and cannot carry the concrete result itself, so it sets
    /// this handshake and the back-to-back blackhole hook consumes the
    /// stash to complete the callee — the retrace-reaches-finishframe
    /// `DoneWithThisFrame`/`ExitFrameWithExceptionRef` the assembler
    /// caller catches (pyjitpl.py:1688-1698, jitexc.py) — instead of
    /// re-running the resumed region over already-applied effects.
    static CA_WALK_FINISHED_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CA_WALK_RESUME_FRAME: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CA_WALK_RESUME_DEADFRAME: std::cell::RefCell<Option<Vec<i64>>> =
        const { std::cell::RefCell::new(None) };
    static SELF_RECURSIVE_DISPATCH_CACHE: UnsafeCell<Option<(u64, Option<u64>)>> =
        const { UnsafeCell::new(None) };
}

struct FrameRoot {
    depth: usize,
}

impl FrameRoot {
    fn new(frame: &mut PyFrame) -> Self {
        Self {
            depth: majit_gc::shadow_stack::push(majit_ir::GcRef(frame as *mut PyFrame as usize)),
        }
    }

    fn frame(&mut self) -> &mut PyFrame {
        let frame = majit_gc::shadow_stack::get(self.depth).0 as *mut PyFrame;
        unsafe { &mut *frame }
    }
}

impl Drop for FrameRoot {
    fn drop(&mut self) {
        majit_gc::shadow_stack::try_pop_to(self.depth);
    }
}

/// Take stashed exception from blackhole/force FFI paths.
// dont_look_inside: reads LAST_CA_EXCEPTION TLS; bridge/force machinery.
#[majit_macros::dont_look_inside]
pub fn take_ca_exception() -> Option<pyre_interpreter::error::PyError> {
    LAST_CA_EXCEPTION.with(|c| c.borrow_mut().take())
}

/// Root the exception parked in `LAST_CA_EXCEPTION` across the call-assembler
/// FFI boundary. Compiled code runs between `set_pending_ca_exception` and
/// `take_ca_exception` — it can drive a major collection and can overwrite the
/// single in-flight-exception cell with a later raise — so the parked
/// `PyError`'s GC refs must be forwarded here or the stashed exception is swept
/// before it surfaces. Never materialises the lazy-null `exc_object`.
pub fn walk_last_ca_exception(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    LAST_CA_EXCEPTION.with(|c| {
        // SAFETY: `as_ptr` yields the `Option<PyError>` interior; this closure
        // holds the only reference for its duration and does not re-borrow the
        // cell, so no borrow-flag conflict with a walker-triggered path.
        let opt = unsafe { &mut *c.as_ptr() };
        if let Some(err) = opt.as_mut() {
            err.walk_gc_refs(visitor);
        }
    });
}

/// Park a Python exception that needs to surface across an FFI boundary
/// (callback emitted by compiled code → here → eventually picked up by
/// `take_ca_exception` in the eval loop).
pub fn set_pending_ca_exception(err: pyre_interpreter::error::PyError) {
    LAST_CA_EXCEPTION.with(|c| {
        *c.borrow_mut() = Some(err);
    });
}

// warmspot.py:449 portal result_type == REF: FINISH always boxes via
// wrapint, so the force/resume paths always receive a boxed Ref.
// FinishProtocol and normalize_direct_finish_result removed — they
// were dead code since result_type is always Type::Ref.

#[inline]
pub(crate) fn recursive_force_cache_safe(callable: PyObjectRef) -> bool {
    unsafe {
        if pyre_interpreter::is_builtin_code(
            pyre_interpreter::function_get_code(callable) as pyre_object::PyObjectRef
        ) {
            return false;
        }
        if !function_get_closure(callable).is_null() {
            return false;
        }
        let code =
            &*(pyre_interpreter::get_pycode(callable) as *const pyre_interpreter::CodeObject);
        let func_name = function_get_name(callable);
        let mut arg_state = OpArgState::default();
        let mut saw_self_reference = false;

        for code_unit in code.instructions.iter().copied() {
            let (instruction, op_arg) = arg_state.get(code_unit);
            match instruction {
                Instruction::LoadName { namei } => {
                    let idx = namei.get(op_arg) as usize;
                    if code.names[idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::LoadGlobal { namei } => {
                    let raw = namei.get(op_arg) as usize;
                    let name_idx = raw >> 1;
                    if code.names[name_idx].as_str() != func_name {
                        return false;
                    }
                    saw_self_reference = true;
                }
                Instruction::StoreName { .. }
                | Instruction::StoreGlobal { .. }
                | Instruction::LoadDeref { .. }
                | Instruction::StoreDeref { .. }
                | Instruction::DeleteDeref { .. }
                | Instruction::ImportName { .. }
                | Instruction::ImportFrom { .. }
                | Instruction::DeleteName { .. }
                | Instruction::DeleteGlobal { .. }
                | Instruction::MakeCell { .. }
                | Instruction::CopyFreeVars { .. } => return false,
                _ => {}
            }
        }

        if !saw_self_reference {
            return false;
        }
    }

    true
}

fn self_recursive_dispatch(green_key: u64) -> Option<u64> {
    SELF_RECURSIVE_DISPATCH_CACHE.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if let Some((cached_key, token_num)) = *slot {
            if cached_key == green_key && token_num.is_some() {
                return token_num;
            }
        }

        let (driver, _) = crate::eval::driver_pair();
        let token_num = driver.get_loop_token(green_key).map(|token| token.number);
        if token_num.is_some() {
            *slot = Some((green_key, token_num));
        }
        token_num
    })
}

// Force cache implementation removed — CallAssemblerI + bridge
// handles recursive dispatch natively.

// ── JitFrame layout descriptors published to the backends ─────────

fn jitframe_layout_descrs() -> majit_gc::rewrite::JitFrameDescrs {
    use majit_backend::jitframe::*;
    majit_gc::rewrite::JitFrameDescrs {
        jitframe_tid: crate::jit::descr::JITFRAME_GC_TYPE_ID,
        jitframe_fixed_size: JITFRAME_FIXED_SIZE,
        jf_frame_info_ofs: JF_FRAME_INFO_OFS,
        jf_descr_ofs: JF_DESCR_OFS,
        jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
        jf_savedata_ofs: JF_SAVEDATA_OFS,
        jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
        jf_forward_ofs: JF_FORWARD_OFS,
        jf_frame_ofs: JF_FRAME_OFS,
        // RPython llmodel.py:385-395 + rewrite.py:680-684 consume
        // unpack_arraydescr()/lendescr offsets as jitframe-base-relative
        // addresses, not offsets relative to jf_frame itself.
        jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
        jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
        sign_size: SIGN_SIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::jitframe_layout_descrs;
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JF_FRAME_OFS};

    #[test]
    fn jitframe_layout_descrs_uses_frame_relative_offsets() {
        let descrs = jitframe_layout_descrs();
        assert_eq!(descrs.jf_frame_baseitemofs, FIRST_ITEM_OFFSET);
        assert_eq!(descrs.jf_frame_lengthofs, JF_FRAME_OFS);
    }
}

#[cfg(feature = "cranelift")]
pub fn jitframe_layout_info() -> majit_backend_cranelift::JitFrameLayoutInfo {
    majit_backend_cranelift::JitFrameLayoutInfo {
        jitframe_descrs: Some(jitframe_layout_descrs()),
    }
}

#[cfg(feature = "dynasm")]
pub fn jitframe_layout_info_dynasm() -> majit_backend_dynasm::JitFrameLayoutInfo {
    majit_backend_dynasm::JitFrameLayoutInfo {
        jitframe_descrs: Some(jitframe_layout_descrs()),
    }
}

// ── Callee frame allocation ───────────────────────────────────────

thread_local! {
    /// Callee frames the JIT created and has not finished running.
    /// Their only root between `jit_create_callee_frame_*` and
    /// `jit_drop_callee_frame` is this list: compiled code holds the
    /// frame in a register / jitframe slot, and the frame sits on no
    /// `CURRENT_FRAME` chain. Walked by `walk_jit_callee_frame_roots`.
    static LIVE_CALLEE_FRAMES: UnsafeCell<Vec<*mut PyFrame>> = const { UnsafeCell::new(Vec::new()) };

    static JIT_CALLEE_FRAME_ROOT_AREA: JitCalleeFrameRootArea = JitCalleeFrameRootArea {
        live_frames: LIVE_CALLEE_FRAMES.with(|cell| cell as *const _),
    };
}

struct JitCalleeFrameRootArea {
    live_frames: *const UnsafeCell<Vec<*mut PyFrame>>,
}

/// Create a callee frame for a JIT call, exactly as the interpreter
/// creates every other frame: `baseobjspace.py:799-801 createframe` on a
/// `pyframe.py:52 class PyFrame(W_Root)`, a normal GC object whose
/// lifetime is its reachability. Nothing recycles the block, so a frame
/// the program retains past the call — a traceback node, an `f_back`
/// chain, a `sys._getframe` result — stays valid for as long as Python
/// can reach it.
fn alloc_callee_frame(
    code: *const (),
    args: &[PyObjectRef],
    w_globals: PyObjectRef,
    execution_context: *const pyre_interpreter::PyExecutionContext,
) -> *mut PyFrame {
    let ptr = pyre_interpreter::pyframe::FrameBox::new(
        PyFrame::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            PY_NULL,
            pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        ),
    )
    .into_raw();
    LIVE_CALLEE_FRAMES.with(|cell| unsafe { &mut *cell.get() }.push(ptr));
    ptr
}

/// Drop the JIT's root on a callee frame that finished running. The
/// frame itself is not freed — `executioncontext.py:91-107 leave` frees
/// nothing either; the collector reclaims it once nothing reaches it.
fn unroot_callee_frame(ptr: *mut PyFrame) {
    // This list is what makes a minor collection scan the frame's slots.
    // Once it stops, a young ref stored since the last minor — argument
    // boxing, the CALL_ASSEMBLER writeback through `jit_frame_set_slot_*` —
    // is reachable only through the old-gen locals array's own items, which
    // a minor does not walk. The frame outlives the call whenever the program
    // retained it (a traceback node, an `f_back` chain), so re-arm the
    // remembered set here, at the moment root-walking stops: the same barrier
    // `pyframe.py:52 __init__` arms at creation, for the same reason.
    unsafe {
        pyre_interpreter::pyframe::remember_frame_locals_array((*ptr).locals_cells_stack_w);
    }
    if pyre_object::gc_hook::try_gc_owns_object(ptr as *mut u8) {
        pyre_object::gc_hook::try_gc_write_barrier(ptr as *mut u8);
    }
    LIVE_CALLEE_FRAMES.with(|cell| {
        let frames = unsafe { &mut *cell.get() };
        // Calls return in LIFO order, so the match is normally the last
        // entry.
        if let Some(pos) = frames.iter().rposition(|p| *p == ptr) {
            frames.swap_remove(pos);
        }
    });
}

/// Visit the GC-ref slots of one live callee frame: every
/// `locals_cells_stack_w` item plus the ref-bearing statics — the same
/// field set `walk_pyframe_roots` (pyre-interpreter::eval) visits for
/// interpreter frames on the `CURRENT_FRAME` chain.
///
/// # Safety
/// `frame` must point at a fully-initialized live `PyFrame`.
unsafe fn visit_callee_frame_roots(frame: *mut PyFrame, visitor: &mut dyn FnMut(&mut GcRef)) {
    let frame = unsafe { &mut *frame };
    for slot in locals_w_mut!(frame).as_mut_slice() {
        visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut GcRef) });
    }
    visitor(unsafe { &mut *(&mut frame.f_generator_nowref as *mut PyObjectRef as *mut GcRef) });
    visitor(unsafe { &mut *(&mut frame.w_yielding_from as *mut PyObjectRef as *mut GcRef) });
    visitor(unsafe { &mut *(&mut frame.w_globals as *mut PyObjectRef as *mut GcRef) });
}

/// Extra GC root walker for the callee frames a JIT call is running.
/// They sit on no `CURRENT_FRAME`/`f_backref` chain while compiled code
/// runs, so `walk_pyframe_roots` reaches neither the frames themselves
/// nor their locals: a major collection would sweep the block compiled
/// code is executing on, and a young object stored into a frame slot
/// (argument boxing, back-edge CALL_ASSEMBLER writeback) would be
/// invisible to a minor collection, leaving the slot pointing at
/// evacuated nursery memory. Registered via `register_extra_root_walker`,
/// mirroring the framework.py `root_walker.walk_roots` seam the collector
/// already uses for the other host-side root sources.
pub fn walk_jit_callee_frame_roots(visitor: &mut dyn FnMut(&mut GcRef)) {
    let data = capture_jit_callee_frame_root_area();
    unsafe { walk_jit_callee_frame_roots_area(data, visitor) };
}

pub fn capture_jit_callee_frame_root_area() -> *const () {
    JIT_CALLEE_FRAME_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_jit_callee_frame_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn walk_jit_callee_frame_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut GcRef),
) {
    let area = unsafe { &*(data as *const JitCalleeFrameRootArea) };
    let live_frames = unsafe { &mut *(*area.live_frames).get() };
    for slot in live_frames.iter_mut() {
        // Mark the frame object itself — this list is its only root
        // until the call returns. The slot is handed over as a mutable
        // `GcRef` so a relocation would be written back, as upstream's
        // slot-address roots are (`shadowstack.py:43-46`); it stays
        // unwritten only because `FrameBox::new` allocates frames
        // non-moving.
        visitor(unsafe { &mut *(slot as *mut *mut PyFrame as *mut GcRef) });
        unsafe { visit_callee_frame_roots(*slot, visitor) };
    }
}

// ── JIT call callbacks ───────────────────────────────────────────

extern "C" fn jit_call_user_function_from_frame(
    frame_ptr: i64,
    callable: i64,
    args_ptr: *const i64,
    nargs: i64,
) -> i64 {
    let frame = unsafe { &*(frame_ptr as *const PyFrame) };
    let args =
        unsafe { std::slice::from_raw_parts(args_ptr as *const PyObjectRef, nargs as usize) };
    // Depth tracked by pyre_interpreter::call::PY_RECURSION_DEPTH (eval-loop entry).
    match pyre_interpreter::call::call_user_function(frame, callable as PyObjectRef, args) {
        Ok(result) => result as i64,
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: write the exception
            // to the backend's `_exception_emulator` tp/val cells. The
            // matching GUARD_NO_EXCEPTION in the trace then reads
            // pos_exception()/pos_exc_value() and fails, and resume
            // data hands control to the except block. Do NOT stash the
            // PyError through a side channel — that would let the
            // interpreter-side eval loop surface it before the guard
            // machinery sees it, bypassing try/except.
            let exc_obj = err.to_exc_object();
            if exc_obj != pyre_object::PY_NULL {
                store_jit_exception(exc_obj as i64);
            }
            0 // garbage — GUARD_NO_EXCEPTION will fire
        }
    }
}

/// llmodel.py:194-199 _store_exception: publish a materialised exception
/// object into the active backend's `_store_exception` cells so the
/// GuardNoException after the call detects it. One arm per backend; the wasm
/// backend keeps the pending exception in shared linear memory.
pub(crate) fn store_jit_exception(value: i64) {
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::jit_exc_raise(value);
    #[cfg(feature = "dynasm")]
    majit_backend_dynasm::jit_exc_raise(value);
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_raise(value);
    #[cfg(not(any(feature = "cranelift", feature = "dynasm", target_arch = "wasm32")))]
    let _ = value;
}

/// Backend bridge for pyre-interpreter's residual-call helpers, which
/// cannot reference the backend crates directly. Publishes a materialised
/// exception object into the active backend's _store_exception cells so
/// the GuardNoException after the call detects it.
extern "C" fn jit_exc_raise_shim(value: i64) {
    store_jit_exception(value);
}

/// Publish a raise from a may-force residual helper to BOTH executors.
///
/// `bh_call_fn`/`bh_call_fn_N` is bound as the may-force CALL target
/// (`cpu.rs` `call_fn`, bound `CallFlavor::MayForce` by `codewriter.rs`'s
/// `register_helper_fn_pointers`), so the same helper runs under the blackhole
/// interpreter AND inside a compiled trace.  The blackhole reads the raise from
/// `BH_LAST_EXC_VALUE`; a compiled trace's `GUARD_NO_EXCEPTION` reads it from
/// the backend `_store_exception`
/// cells (`jit_exc_raise`).  Writing only `BH_LAST_EXC_VALUE` leaves
/// `GUARD_NO_EXCEPTION` reading a stale 0, so the guard wrongly passes and the
/// helper's NULL result flows to the consumer — keep both states in sync.
///
/// Both cells this writes — `BH_LAST_EXC_VALUE` and the backend `JIT_EXC_VALUE`
/// (via `store_jit_exception`) — are GC-rooted by `walk_parked_exception_roots`
/// and by the per-mutator `PyFrameRootArea`, so this writer needs no rooting of
/// its own.
fn publish_residual_call_exception(exc_obj: i64) {
    // Every residual raise enters the two exception channels here, so this is
    // the one place that can hold the line on what they may carry — and the
    // one place where the producer is still on the stack.  Both consumers
    // (`GUARD_NO_EXCEPTION` in compiled code, `_exit_frame_with_exception` in
    // the blackhole) end up reading the value's `ExcKind` tag through a match
    // with no wildcard arm; see `exit_frame_exception_ref`.
    let obj = exc_obj as PyObjectRef;
    if !obj.is_null()
        && unsafe { pyre_object::interp_exceptions::w_exception_kind_checked(obj) }.is_none()
    {
        reject_non_exception_channel_value(obj, "publish_residual_call_exception", String::new);
    }
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj));
    store_jit_exception(exc_obj);
}

/// The caller's residual-call exception pair, taken out of its cells for the
/// span of a nested JIT re-entry by [`park_residual_call_exception`].
///
/// A cell is the only root its exception has (`walk_bh_last_exc_value` /
/// `walk_jit_exc_value`), so a value taken out of one is pinned on the shadow
/// stack for the span and read back relocated: an RPython local holding a GC
/// object is a shadow-stack root, a Rust local is not.
struct ParkedResidualException {
    /// `None` when both cells were empty — the ordinary case, since a
    /// residual call is reached with no exception in flight.
    scope: Option<pyre_object::gc_roots::RootScope>,
    save: usize,
    bh_pinned: bool,
    backend_pinned: bool,
}

/// Take the caller's `BH_LAST_EXC_VALUE` / backend `_store_exception` pair out
/// of their cells and leave both empty.
///
/// Upstream never needs this: the raise a residual call reports lives in
/// `metainterp.last_exc_value`, a field of the MetaInterp instance that owns
/// the call, and `llmodel.py:194 _store_exception` is read back by the same
/// `bh_call_*` that armed it. Pyre keeps one cell per thread, so a nested
/// execution inside the call aliases the caller's.
fn park_residual_call_exception() -> ParkedResidualException {
    let bh = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
    let backend = crate::eval::jit_exc_value_peek_backend();
    if bh == 0 && backend == 0 {
        return ParkedResidualException {
            scope: None,
            save: 0,
            bh_pinned: false,
            backend_pinned: false,
        };
    }
    let scope = pyre_object::gc_roots::push_roots();
    let save = pyre_object::gc_roots::shadow_stack_len();
    let bh_pinned = bh != 0;
    if bh_pinned {
        pyre_object::gc_roots::pin_root(bh as pyre_object::PyObjectRef);
    }
    let backend_pinned = backend != 0;
    if backend_pinned {
        pyre_object::gc_roots::pin_root(backend as pyre_object::PyObjectRef);
    }
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
    ParkedResidualException {
        scope: Some(scope),
        save,
        bh_pinned,
        backend_pinned,
    }
}

/// Restore what [`park_residual_call_exception`] took, discarding whatever the
/// nested execution left in the cells: a raise the nested callee already
/// handled is not this helper's to report, and the caller's
/// `GUARD_NO_EXCEPTION` would read it as a spurious pending exception — the
/// failure [`drain_backend_jit_exc`] names for the walker's snapshot side.
fn unpark_residual_call_exception(parked: ParkedResidualException) {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
    if parked.scope.is_none() {
        return;
    }
    let mut index = parked.save;
    if parked.bh_pinned {
        let bh = pyre_object::gc_roots::shadow_stack_get(index) as i64;
        index += 1;
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(bh));
    }
    if parked.backend_pinned {
        store_jit_exception(pyre_object::gc_roots::shadow_stack_get(index) as i64);
    }
}

/// [`park_residual_call_exception`] as an RAII scope, for a region with more
/// than one exit.
///
/// A whole tracing session needs this for the same reason a single residual
/// call does. `pyjitpl.py:2763 execute_raised` records a residual's raise into
/// `metainterp.last_exc_value` — an instance field, so upstream's dies with the
/// MetaInterp that owned the session. Pyre's stand-in is one cell per thread,
/// so a session that ends with a raise still recorded (a walk that executes
/// residuals concretely ends on the loop-exit `StopIteration`) leaves that
/// value visible to whatever runs next, and the next guard failure's blackhole
/// resume reads it as its own pending raise.
///
/// Declare the guard before anything else in the scope so it drops last, after
/// every inner [`pyre_object::gc_roots::RootScope`] has unwound and the shadow
/// stack is back to the depth [`park_residual_call_exception`] recorded.
pub(crate) struct ResidualExceptionScope {
    parked: Option<ParkedResidualException>,
    dbg: bool,
}

impl ResidualExceptionScope {
    pub(crate) fn park(dbg: bool) -> Self {
        Self {
            parked: Some(park_residual_call_exception()),
            dbg,
        }
    }
}

impl Drop for ResidualExceptionScope {
    fn drop(&mut self) {
        if self.dbg {
            let bh = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            let backend = crate::eval::jit_exc_value_peek_backend();
            if bh != 0 || backend != 0 {
                eprintln!("[residual-exc-scope] discarding bh={bh:#x} backend={backend:#x}");
            }
        }
        if let Some(parked) = self.parked.take() {
            unpark_residual_call_exception(parked);
        }
    }
}

/// Drain the backend `_store_exception` cells (`jit_exc_clear`) without
/// touching `BH_LAST_EXC_VALUE`.
///
/// `pyjitpl.py:2763 execute_raised` records a raise from a residual call
/// into `metainterp.last_exc_value` only — the backend `_store_exception`
/// cells (`llmodel.py:194` `_store_exception`) are written exclusively by
/// compiled / blackhole execution, never by metainterp tracing.  Pyre's
/// authoritative full-body walk shares the `bh_*` residual helpers with
/// the blackhole, so a raising helper publishes to BOTH cells
/// ([`publish_residual_call_exception`]).  The walker's `execute_raised`
/// analogue (`jitcode_dispatch.rs` residual-executor Err arm) keeps the
/// raise in `BH_LAST_EXC_VALUE` (= `last_exc_value`) but must drain the
/// backend cells so the recording leaves them pristine; otherwise an
/// aborted trace's snapshot-side raise leaks into the live frame's
/// re-run, where compiled `GUARD_NO_EXCEPTION` reads it as a spurious
/// pending exception.
pub(crate) fn drain_backend_jit_exc() {
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::jit_exc_clear();
    #[cfg(feature = "dynasm")]
    majit_backend_dynasm::jit_exc_clear();
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::jit_exc_clear();
}

/// Clear the two pyre carriers for a residual-call exception after a compiled
/// frame has handled it and returned normally.
///
/// RPython keeps `MetaInterp.last_exc_value` on the metainterp that owns the
/// call and `handle_possible_exception()` clears it when control reaches the
/// handler.  Pyre's residual helpers publish to both this TLS stand-in and the
/// backend `_store_exception` cells, so a successful compiled portal exit must
/// leave both empty.  A caller exception that legitimately spans a nested JIT
/// entry is removed and restored by `park_residual_call_exception`, outside
/// this boundary.
pub(crate) fn clear_residual_call_exception() {
    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    drain_backend_jit_exc();
}

/// #73 measurement gate. When `PYRE_M73_LASTINSTR_AUDIT` is set, the three
/// blackhole traceback sites compare the production inversion
/// `containing_py_pc_for_jitcode_pc_public(last_opcode_position)` (= `orgpc`, the
/// raising op's python pc) against the forward candidate `frame.last_instr + 1`
/// (the JIT-vable resume convention normalized to `orgpc`). Divergence means the
/// forward datum is stale/off-by-one and cannot yet replace the inversion; a
/// clean corpus (zero divergences) would mean the forward carry is already
/// exact. Pure telemetry — never changes production output.
fn m73_lastinstr_audit_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var_os("PYRE_M73_LASTINSTR_AUDIT").is_some())
}

fn m73_lastinstr_audit(
    site: &str,
    jitcode_index: Option<i32>,
    opcode_position: i32,
    frame_ptr: *const PyFrame,
) {
    if !m73_lastinstr_audit_enabled() || frame_ptr.is_null() {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static HITS: AtomicU64 = AtomicU64::new(0);
    static DIVERGE: AtomicU64 = AtomicU64::new(0);
    static NONE: AtomicU64 = AtomicU64::new(0);
    let inv = jitcode_index.and_then(|index| {
        pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(index, opcode_position)
    });
    let fwd = unsafe { (*frame_ptr).last_instr as i64 } + 1;
    let hits = HITS.fetch_add(1, Ordering::Relaxed) + 1;
    if hits == 1 {
        // Per-process heartbeat: confirms the site was exercised even when no
        // divergence prints (each check.py test is a fresh process).
        eprintln!("[M73-LASTINSTR] first-hit site={site}");
    }
    match inv {
        Some(iv) => {
            let iv = iv as i64;
            if iv != fwd {
                let d = DIVERGE.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[M73-LASTINSTR] diverge site={site} inv={iv} fwd={fwd} delta={} (proc hits={hits} diverge={d} none={})",
                    iv - fwd,
                    NONE.load(Ordering::Relaxed)
                );
            }
        }
        None => {
            NONE.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(crate) extern "C" fn record_caught_blackhole_traceback(
    exc_value: i64,
    frame_value: i64,
    jitcode_index: i64,
    opcode_position: i64,
) {
    let Ok(jitcode_index) = i32::try_from(jitcode_index) else {
        return;
    };
    let Ok(opcode_position) = i32::try_from(opcode_position) else {
        return;
    };
    let forwards_existing = pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
        jitcode_index,
        opcode_position,
    );
    let frame_ptr = frame_value as *mut PyFrame;
    if frame_ptr.is_null() || exc_value == 0 {
        return;
    }
    // A forwarding raise keeps the chain it arrived with only when this frame
    // attached the head. When the raise came out of an inlined callee, the
    // head names the callee and this frame still owes one node. The ownership
    // check preserves `pyopcode.py:147-148 handle_operation_error` as the
    // one-node-per-frame-per-delivery authority.
    if forwards_existing {
        let owns_head = unsafe {
            let head =
                pyre_object::interp_exceptions::w_exception_get_traceback(exc_value as PyObjectRef);
            !head.is_null()
                && pyre_interpreter::pytraceback::is_pytraceback(head)
                && pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) == frame_ptr
        };
        if owns_head {
            return;
        }
    }
    let resolved = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
        jitcode_index,
        opcode_position,
    );
    let exact =
        pyre_jit_trace::py_coord::exact_py_pc_for_jitcode_pc_public(jitcode_index, opcode_position);
    let last_instruction = exact
        .or(resolved)
        .map_or(unsafe { (*frame_ptr).last_instr as i64 }, i64::from);
    // One catch can be reached by two recorders: this hook, emitted into the
    // loop trace by the in-trace handler-entry arm, and the IR-virtual node the
    // bridge handler-entry arm builds when the exception edge deopts into a
    // bridge.  Both then run for the same delivery and the frame gets two
    // adjacent nodes, so the handler reads its own name twice.  Upstream never
    // has the pair — `pyopcode.py handle_operation_error` is the single attach
    // point — so drop the second: `record_application_traceback` prepends, and
    // this exact node already exists exactly when the chain HEAD carries this
    // frame at this instruction.  A genuine repeat (the same frame raising
    // again at the same offset) always attaches to a fresh exception, and a
    // bare reraise, which does legitimately re-record one frame, is skipped
    // above.
    unsafe {
        let head =
            pyre_object::interp_exceptions::w_exception_get_traceback(exc_value as PyObjectRef);
        if !head.is_null()
            && pyre_interpreter::pytraceback::is_pytraceback(head)
            && pyre_interpreter::pytraceback::w_pytraceback_get_frame(head) == frame_ptr
            && pyre_interpreter::pytraceback::w_pytraceback_get_lasti(head) == last_instruction
        {
            return;
        }
    }
    m73_lastinstr_audit("caught", Some(jitcode_index), opcode_position, frame_ptr);
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            exc_value as PyObjectRef,
            frame_ptr,
            last_instruction,
        );
    }
}

/// `pypy/interpreter/error.py:410-420 record_context` for an exception whose
/// handler is part of the trace, split so the trace records the store itself.
///
/// The implicit `__context__` is derived by `handle_exception_with_context`,
/// and a raise the walk routes straight to a handler in the same trace never
/// surfaces an error the interpreter dispatches — the same gap
/// [`record_inline_traceback_for_recording`] closes for the traceback node.
/// Without this, the compiled iterations of a loop raising under a live handler
/// leave `__context__` null while the interpreted ones set it.
///
/// Chains through `chain_context`, so the active exception is selected, a
/// non-exception or self-referential pair is refused, and a chain that would
/// close a cycle is broken by nulling the offending link — all exactly as the
/// interpreter does it — and then answers the slot's resulting value.  The
/// trace's own `SetfieldGc` therefore stores what is already there; `0` means
/// the slot stays null.
///
/// The cycle break writes the `__context__` of an exception other than
/// `w_exc`, which the emitted store does not describe, so the call is declared
/// `cannot_raise_effect_info()` — a heap mutator that cannot raise — rather
/// than as having no heap effect.
pub(crate) extern "C" fn resolve_exception_context(exc_value: i64) -> i64 {
    if exc_value == 0 {
        return 0;
    }
    let w_exc = exc_value as PyObjectRef;
    if !unsafe { pyre_object::is_exception(w_exc) } {
        return 0;
    }
    let existing = unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_exc) };
    if !existing.is_null() {
        return existing as i64;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_exc);
    let active = pyre_interpreter::eval::get_sys_exception();
    pyre_interpreter::error::chain_context(w_exc, active);
    unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_exc) as i64 }
}

pub(crate) extern "C" fn record_inline_traceback_for_recording(
    exc_value: i64,
    w_code_value: i64,
    w_globals_value: i64,
    jitcode_index: i64,
    opcode_position: i64,
) {
    let Ok(jitcode_index) = i32::try_from(jitcode_index) else {
        return;
    };
    let Ok(opcode_position) = i32::try_from(opcode_position) else {
        return;
    };
    // Inline frames follow the same rule as concrete blackhole frames: a raise
    // that forwards an already-propagating exception preserves the traceback
    // attached by the original raising instruction.
    if pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
        jitcode_index,
        opcode_position,
    ) {
        return;
    }
    if exc_value == 0 || w_code_value == 0 {
        return;
    }
    let Some(last_instruction) = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
        jitcode_index,
        opcode_position,
    )
    .map(i64::from) else {
        return;
    };
    let w_exc = exc_value as PyObjectRef;
    let w_code = w_code_value as PyObjectRef;
    let w_globals = w_globals_value as PyObjectRef;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_exc);
    pyre_object::gc_roots::pin_root(w_code);
    pyre_object::gc_roots::pin_root(w_globals);
    // `record_application_traceback` requires the traceback's own frame
    // identity. The recording walker cannot force the optimizer's virtual
    // locals, so materialize a traceback-only frame from the promoted callee
    // metadata instead of substituting the caller's frame.
    let Ok(mut frame) = pyre_interpreter::createframe_obj(
        w_code as *const (),
        w_globals,
        pyre_interpreter::call::getexecutioncontext(),
        None,
    ) else {
        return;
    };
    frame.last_instr = last_instruction as isize;
    // `pyopcode.py:184 handle_operation_error` marks a propagating frame
    // finished before it leaves that frame.  This object materializes that
    // already-unwound inline callee, so preserve the same lifecycle state;
    // traceback.clear_frames() must not mistake it for a live activation.
    frame.set_frame_finished_execution(true);
    let frame_ptr = frame.into_raw();
    pyre_object::gc_roots::pin_root(frame_ptr as PyObjectRef);
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            w_exc,
            frame_ptr,
            last_instruction,
        );
    }
}

/// `pytraceback.py:104-109 record_application_traceback` for a frame the resume
/// crossed and discarded — an inlined callee the exception-edge bridge unwound
/// clear out of on its way to the live frame's own handler.
///
/// `pyopcode.py:148` runs BEFORE the `:152` exception-table lookup, so a frame
/// that only propagates contributes a node exactly like one that catches.  The
/// levels named here never reach a recorder that could speak for them: the
/// route exists because the unwind discards them, and by the time the walk
/// starts they are gone.
///
/// The node's frame is fabricated from the code object, as
/// [`record_inline_traceback_for_recording`] does and for the same reason —
/// the level's own `PyFrame` stayed virtual in the compiled trace and the
/// resume never materialized it, so there is no other object to name.  Unlike
/// that case there is also no seeded frame it could be confused with: nothing
/// else in this process can reach the discarded level either.
///
/// `py_pc` is the level's resume coordinate, which is `next_instr`-style — the
/// same coordinate `exc_table_offset` converts with `saturating_sub(1)` and
/// `set_last_instr_from_next_instr` converts for the live frame.  Everything
/// below wants the instruction that RAN: the opcode decode that answers the
/// bare-reraise question, `frame.last_instr`, and `record_application_traceback`'s
/// `tb_lasti`.  Convert once, up front.
pub(crate) extern "C" fn record_discarded_level_traceback(
    exc_value: i64,
    w_code_value: i64,
    py_pc: i64,
) {
    if exc_value == 0 || w_code_value == 0 || py_pc < 0 {
        return;
    }
    let last_instruction = py_pc.saturating_sub(1);
    let w_code = w_code_value as PyObjectRef;
    let raw_code =
        unsafe { pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject };
    if raw_code.is_null() {
        return;
    }
    // Same rule the other two recorders follow: a raise that forwards an
    // already-propagating exception keeps the traceback it arrived with.
    if pyre_jit_trace::state::raise_at_py_pc_keeps_existing_traceback(
        unsafe { &*raw_code },
        last_instruction as usize,
    ) {
        return;
    }
    let w_globals = unsafe { pyre_interpreter::w_code_get_w_globals(w_code) };
    let w_exc = exc_value as PyObjectRef;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_exc);
    pyre_object::gc_roots::pin_root(w_code);
    pyre_object::gc_roots::pin_root(w_globals);
    let Ok(mut frame) = pyre_interpreter::createframe_obj(
        w_code as *const (),
        w_globals,
        pyre_interpreter::call::getexecutioncontext(),
        None,
    ) else {
        return;
    };
    frame.last_instr = last_instruction as isize;
    // This discarded level has already propagated out during the inlined
    // unwind, exactly the `pyopcode.py:184` finished-frame transition.
    frame.set_frame_finished_execution(true);
    let frame_ptr = frame.into_raw();
    pyre_object::gc_roots::pin_root(frame_ptr as PyObjectRef);
    unsafe {
        pyre_interpreter::pytraceback::record_application_traceback(
            w_exc,
            frame_ptr,
            last_instruction,
        );
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame(frame_ptr: i64) -> i64 {
    #[cfg(feature = "cranelift")]
    let _ = majit_backend_cranelift::take_pending_frame_restore();

    // `assembler_call_helper` (warmspot.py:1021-1028) resumes the callee
    // frame the rewritten CALL_ASSEMBLER passed as arg 0.
    run_frame_through_portal(frame_ptr)
}

/// warmspot.py:941-959 `ll_portal_runner` core — run the frame the JIT handed
/// in through the portal (`maybe_compile_and_run` + interpreter main loop;
/// ContinueRunningNormally re-enters the JIT via the portal,
/// warmspot.py:961-983).
///
/// The pointer is the callee `PyFrame` the trace itself built —
/// `emit_new_pyframe_inline_with_params` emits `NewWithVtable` +
/// `SetfieldGc` for `pycode` / `w_globals` / `execution_context` and the
/// locals array, and the GC rewriter stores that pointer into the callee
/// jitframe's first slot, which is what every force site reads back. It is
/// the `frame` red of `jd.portal_calldescr`, and `ll_portal_runner` forwards
/// its reds unchanged (`warmspot.py:953-954` `portal_ptr(*args)`).
///
/// Rebuilding a frame here instead would drop the callee's arguments (its
/// locals array) and its `last_instr`, and would run the interpreter on a
/// frame that dies with the C stack.
fn run_frame_through_portal(frame_ptr: i64) -> i64 {
    // `jit_drop_callee_frame` tolerates a tagged word on the same slot because
    // it only unroots; there is nothing sensible to run here for one. The
    // portal's result is a Ref whose NULL spelling means "exception stored",
    // so returning NULL for a non-frame would manufacture an exception-less
    // error rather than deopt cleanly.
    debug_assert!(
        frame_ptr != 0 && frame_ptr & 1 == 0,
        "CALL_ASSEMBLER arg 0 must be the callee PyFrame"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let result = crate::eval::portal_runner(frame);

    // warmspot.py:449 result_type=REF: always boxed Ref
    result as i64
}

/// warmspot.py:941-959 `ll_portal_runner` — the raw-address portal entry whose
/// address is stored in `jd.portal_runner_adr` (warmspot.py:1010-1012) and
/// called by the synthetic callback loop that `compile_tmp_callback`
/// (compile.py:1125-1132) builds for a not-yet-compiled recursive callee.
///
/// The call uses the `jd.portal_calldescr` ABI —
/// `JitDriverStaticData::build_portal_calldescr` lays the args out in `vars`
/// declaration order: greens `[next_instr, is_being_profiled, pycode]` then
/// reds `[frame, ec]`. The callee `frame` red already carries `pycode` /
/// `next_instr` baked in by the caller that built it, so only the frame
/// pointer is needed to resume; the greens are consumed implicitly through the
/// reconstructed frame, matching the frame-only `assembler_call_helper` entry.
#[majit_macros::jit_may_force]
pub extern "C" fn ll_portal_runner_shim(
    _next_instr: i64,
    _is_being_profiled: i64,
    _pycode: i64,
    frame_ptr: i64,
    _ec: i64,
) -> i64 {
    // The callback loop `compile_tmp_callback` builds (compile.py:1125-1132)
    // passes the callee `frame` red that `emit_new_pyframe_inline_with_params`
    // constructed — a proper `PyFrame` with `locals_cells_stack_w` already
    // populated (NewWithVtable + SetfieldGc).  Run it directly; the greens
    // (`next_instr` / `pycode`) are redundant because the frame carries them.
    run_frame_through_portal(frame_ptr)
}

/// warmspot.py:1021-1028 — assembler_call_helper.
///
/// Called when CALL_ASSEMBLER guard-fails (not a finish exit).
/// Receives a JitFrame pointer, reconstructs a PyFrame from the
/// jf_frame slots, and resumes execution in the interpreter.
///
/// This is the JitFrame-aware counterpart to `jit_force_callee_frame`
/// (which operates on PyFrame directly). When the GC rewriter wires
/// nursery JitFrame allocation, this function replaces the force path.
#[allow(dead_code)]
#[majit_macros::jit_may_force]
pub extern "C" fn assembler_call_helper(jitframe_ptr: i64, _virtualizable_ref: i64) -> i64 {
    use majit_backend::jitframe::JitFrame;

    let jf = jitframe_ptr as *mut JitFrame;

    // warmspot.py:1022 — fail_descr = cpu.get_latest_descr(deadframe)
    // compile.py:701 handle_fail: dispatches on fail_descr to either
    // _trace_and_compile_from_bridge or resume_in_blackhole.
    // Bridge compilation is driven by must_compile() in jitdriver.
    // This force path always resumes in the interpreter (blackhole).
    let _descr = unsafe { majit_backend::llmodel::get_latest_descr(jf) };

    // For now, reconstruct a PyFrame and run it in the interpreter.
    // This is the "blackhole" path — RPython resume.py parity.
    //
    // Step 1: read the raw int arg from jf_frame[0]
    let raw_local0 = unsafe { majit_backend::llmodel::get_int_value_direct(jf, 0) } as i64;

    // Step 2: create a PyFrame and run it
    // The caller_frame is in inputs[0] which was the JitFrame's first
    // virtualizable input. For now, fall back to the existing force path.
    jit_force_self_recursive_call_raw_1(jitframe_ptr, raw_local0)
}

/// RPython: FieldDescr.offset is resolved at rtyper time. In pyre, Rust struct
/// layout determines field offsets. This resolver maps (owner_type, field_name)
/// to byte offsets for BhDescr::Field resolution in the blackhole.
#[allow(dead_code)]
fn resolve_field_offset(owner: &str, field_name: &str) -> usize {
    use pyre_interpreter::pyframe::PyFrame;
    match field_name {
        "execution_context" => std::mem::offset_of!(PyFrame, execution_context),
        "code" | "pycode" => std::mem::offset_of!(PyFrame, pycode),
        "locals_cells_stack_w" => std::mem::offset_of!(PyFrame, locals_cells_stack_w),
        "valuestackdepth" => std::mem::offset_of!(PyFrame, valuestackdepth),
        "next_instr" | "f_lasti" | "last_instr" => std::mem::offset_of!(PyFrame, last_instr),
        "namespace" | "w_globals" => std::mem::offset_of!(PyFrame, w_globals),
        "vable_token" => std::mem::offset_of!(PyFrame, vable_token),
        // #171 codewriter descr-bridge (blackhole side): the dotted nested
        // `int_items.{len,block}` leaves `_handle_list_call`
        // emits resolve to offset 0 in the codewriter assembler because
        // `W_ListObject` is a runtime Rust type absent from its struct
        // layouts. Resolve them to the real `IntArray` offsets so a
        // blackholed codewriter list body addresses the typed storage, not
        // the list header (mirrors the JIT-side bridge in
        // `pyre-jit-trace/descr.rs make_descr_from_bh`).
        "int_items.len" => {
            std::mem::offset_of!(pyre_object::W_ListObject, int_items)
                + pyre_object::INT_ARRAY_LEN_OFFSET
        }
        "int_items.block" => {
            std::mem::offset_of!(pyre_object::W_ListObject, int_items)
                + pyre_object::INT_ARRAY_BLOCK_OFFSET
        }
        "float_items.len" => {
            std::mem::offset_of!(pyre_object::W_ListObject, float_items)
                + pyre_object::FLOAT_ARRAY_LEN_OFFSET
        }
        "float_items.block" => {
            std::mem::offset_of!(pyre_object::W_ListObject, float_items)
                + pyre_object::FLOAT_ARRAY_BLOCK_OFFSET
        }
        _ => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][blackhole] WARNING: unresolved field offset owner={:?} name={:?}",
                    owner, field_name
                );
            }
            0
        }
    }
}

/// blackhole.py:1095 get_portal_runner / warmspot.py portal_runner parity:
/// Callback for bhimpl_recursive_call. Receives a frame pointer, executes
/// the frame through the JIT-enabled interpreter (eval_loop_jit), and
/// returns the result. This enables JIT re-entry at recursive portal depth.
/// warmspot.py:941-959 ll_portal_runner(*args) parity.
/// Portal runner with full portal arg ABI.
///
/// blackhole.py:1113-1116: called with merged arg lists:
///   all_i = greens_i + reds_i = [next_instr, is_being_profiled]
///   all_r = greens_r + reds_r = [pycode, frame, ec]
///   all_f = greens_f + reds_f = []
///
/// warmspot.py:972-975: portalfunc_ARGS extraction order:
///   (Int, 'green_int', 0) → next_instr = all_i[0]
///   (Int, 'green_int', 1) → is_being_profiled = all_i[1]
///   (Ref, 'green_ref', 0) → pycode = all_r[0]
///   (Ref, 'red_ref', 0)   → frame = all_r[1]
///   (Ref, 'red_ref', 1)   → ec = all_r[2]
pub(crate) fn bh_portal_runner(all_i: &[i64], all_r: &[i64], _all_f: &[i64]) -> i64 {
    // warmspot.py:972-975: extract portal args from merged lists, then run
    // through the shared `bh_portal_runner_c` body.
    let next_instr = all_i.first().copied().unwrap_or(0);
    let is_being_profiled = all_i.get(1).copied().unwrap_or(0);
    let pycode = all_r.first().copied().unwrap_or(0);
    let frame_ptr = all_r.get(1).copied().unwrap_or(0);
    let ec = all_r.get(2).copied().unwrap_or(0);
    bh_portal_runner_c(next_instr, is_being_profiled, pycode, frame_ptr, ec)
}

/// C-ABI portal runner matching the `mainjitcode_calldescr` arg layout
/// (`"iirrr"`: `next_instr:i, is_being_profiled:i, pycode:r, frame:r, ec:r`).
///
/// This is the function `get_portal_runner` hands to `bh_call_r` for
/// `bhimpl_recursive_call_*` (blackhole.py:1109-1116). `bh_call_r` invokes it
/// through `collect_call_args` + `bh_call_i_dispatch`, i.e. as a raw C function
/// whose five arguments arrive in integer registers per the calldescr — so the
/// runner must take those five scalars directly. (A Rust `fn(&[i64], …)` would
/// receive slice fat pointers there instead, corrupting every argument.)
pub extern "C" fn bh_portal_runner_c(
    next_instr: i64,
    _is_being_profiled: i64,
    pycode: i64,
    frame_ptr: i64,
    ec: i64,
) -> i64 {
    let pycode = pycode as PyObjectRef;
    let frame_ptr = frame_ptr as *mut PyFrame;
    let ec = ec as *const pyre_interpreter::PyExecutionContext;

    if frame_ptr.is_null() {
        return pyre_object::PY_NULL as i64;
    }
    let frame = unsafe { &mut *frame_ptr };
    // warmspot.py:976 calls `portal_ptr(*args)`; it does not copy the green
    // `pycode` back onto the red frame.  The frame was constructed for this
    // activation and already owns its pycode / execution context (the same
    // frame-only contract as `ll_portal_runner_shim` above).  In particular,
    // a stale CRN green must never collapse this frame onto another code
    // object's identity.
    if majit_metainterp::majit_log_enabled()
        && ((!pycode.is_null() && frame.pycode != pycode as *const ())
            || (!ec.is_null() && frame.execution_context != ec))
    {
        eprintln!(
            "[blackhole-resume] CRN/frame identity mismatch: green_pycode={pycode:p} frame_pycode={:p} red_ec={ec:p} frame_ec={:p}",
            frame.pycode, frame.execution_context,
        );
    }
    frame.set_last_instr_from_next_instr(next_instr as usize);
    match crate::eval::portal_runner_result(frame) {
        Ok(result) => result as i64,
        Err(mut err) => {
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                .with(|c| c.set(err.to_exc_object() as i64));
            pyre_object::PY_NULL as i64
        }
    }
}

/// jitexc.py JitException hierarchy — structural parity with RPython.
///
/// `_run_forever` must exit via exactly one of these variants.
/// Call sites still return `BlackholeResult` and will be migrated
/// once `consume_vable_info` guarantees resume data validity.
#[allow(dead_code)] // not all variants are constructed yet
pub enum JitException {
    /// jitexc.py:53 ContinueRunningNormally(gi, gr, gf, ri, rr, rf):
    /// blackhole reached the merge point → restart the portal. The six
    /// vectors carry the green/red int/ref/float live-in arguments used
    /// by `maybe_compile_and_run` to resume execution.
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py:17 DoneWithThisFrameVoid.
    DoneWithThisFrameVoid,
    /// jitexc.py:21 DoneWithThisFrameInt(result).
    DoneWithThisFrameInt(i64),
    /// jitexc.py:29 DoneWithThisFrameRef(result).
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py:37 DoneWithThisFrameFloat(result).
    DoneWithThisFrameFloat(f64),
    /// jitexc.py:45 ExitFrameWithExceptionRef(value): blackhole ran out
    /// of caller frames while propagating a Python exception.
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
}

/// RPython jitexc.py parity: typed exception channels for blackhole result.
///
/// Each variant matches an RPython JitException subclass 1:1.
/// The value is carried in its native type — no boxing into PyObjectRef.
pub enum BlackholeResult {
    /// jitexc.py:53 ContinueRunningNormally(gi, gr, gf, ri, rr, rf)
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<PyObjectRef>,
        green_float: Vec<f64>,
        red_int: Vec<i64>,
        red_ref: Vec<PyObjectRef>,
        red_float: Vec<f64>,
    },
    /// jitexc.py:16 DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py:20 DoneWithThisFrameInt(result: Signed)
    DoneWithThisFrameInt(i64),
    /// jitexc.py:28 DoneWithThisFrameRef(result: GCREF)
    DoneWithThisFrameRef(PyObjectRef),
    /// jitexc.py:36 DoneWithThisFrameFloat(result: FLOATSTORAGE)
    DoneWithThisFrameFloat(f64),
    /// jitexc.py:44 ExitFrameWithExceptionRef(value: GCREF)
    ExitFrameWithExceptionRef(pyre_interpreter::error::PyError),
    /// pyre-only: resume couldn't run (bad resume data, BC_ABORT, etc).
    Failed,
}

impl From<JitException> for BlackholeResult {
    /// jitexc.py parity: each JitException variant maps to BlackholeResult
    /// with its payload preserved — ContinueRunningNormally carries the
    /// 6 green/red lists through.
    fn from(exc: JitException) -> Self {
        match exc {
            JitException::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            } => BlackholeResult::ContinueRunningNormally {
                green_int,
                green_ref,
                green_float,
                red_int,
                red_ref,
                red_float,
            },
            JitException::DoneWithThisFrameVoid => BlackholeResult::DoneWithThisFrameVoid,
            JitException::DoneWithThisFrameInt(v) => BlackholeResult::DoneWithThisFrameInt(v),
            JitException::DoneWithThisFrameRef(r) => BlackholeResult::DoneWithThisFrameRef(r),
            JitException::DoneWithThisFrameFloat(f) => BlackholeResult::DoneWithThisFrameFloat(f),
            JitException::ExitFrameWithExceptionRef(err) => {
                BlackholeResult::ExitFrameWithExceptionRef(err)
            }
        }
    }
}

impl BlackholeResult {
    /// warmspot.py:985-1005: convert typed DoneWithThisFrame* result to PyResult.
    ///
    /// This is the warmspot boundary where the typed JIT exception value
    /// is converted back into a Python-level result. RPython's warmspot
    /// does this implicitly via result_kind dispatch; pyre boxes here.
    pub fn to_pyresult(&self) -> Option<PyResult> {
        match self {
            BlackholeResult::DoneWithThisFrameVoid => Some(Ok(pyre_object::PY_NULL)),
            BlackholeResult::DoneWithThisFrameInt(v) => {
                Some(Ok(pyre_object::intobject::w_int_new(*v) as PyObjectRef))
            }
            BlackholeResult::DoneWithThisFrameRef(r) => Some(Ok(*r)),
            BlackholeResult::DoneWithThisFrameFloat(f) => {
                Some(Ok(pyre_object::floatobject::w_float_new(*f) as PyObjectRef))
            }
            // warmspot.py:998-1005: raise the exception. The clone is forced by
            // the `&self` borrow; the source `err` stays in the caller's
            // `bh_result`, which is dropped without ever being propagated or
            // traceback-recorded (PyError has no Drop), so this returned clone is
            // the only copy that unwinds. That single-propagating-copy invariant
            // is load-bearing: `record_application_traceback` prepends a frame
            // node unconditionally (no per-(frame,lasti) dedup), so driving a
            // second copy that shares this `exc_object` onward would double-append
            // the same node. Keep the source copy dead — do not propagate both.
            BlackholeResult::ExitFrameWithExceptionRef(err) => Some(Err(err.clone())),
            _ => None,
        }
    }
}

/// resume.py:1042 rebuild_from_numbering / read_jitcode_pos_pc output.
/// Each decoded frame section from rd_numb.
pub struct ResumedFrame {
    /// resume.py:1050 jitcode_pos → jitcodes[jitcode_pos].
    /// PyCode pointer — same level as frame.pycode / getcode(func).
    pub code: *const (),
    /// Python bytecode PC the resume data carries (from `frame.pc =
    /// orgpc` at trace time).  pyre's tracer records Python bytecode
    /// PCs because it interprets Python bytecode (not JitCode); `py_pc`
    /// is the raw, pre-adjustment value (before any Cache / ExtendedArg /
    /// NotTaken backtracking).
    pub py_pc: usize,
    /// Raw frame.pc from rd_numb (= orgpc from snapshot).
    /// Some(pc): snapshot guard — orgpc known, liveness-based filling.
    ///   pc=0 is valid (function start / loop header at bytecode 0).
    /// None: no-snapshot guard (rd_numb pc=-1), positional fallback.
    /// CHAIN virtualizable pointer (same value on every section).
    /// RPython parity: there is ONE virtualizable per jitdriver_sd for the
    /// whole blackhole chain; inner sections do not own a separate PyFrame.
    /// Carried on every `ResumedFrame` only because pyre pre-decodes the
    /// rd_numb stream into a `Vec<ResumedFrame>` instead of streaming it
    /// like RPython's `blackhole_from_resumedata` — the value MUST be
    /// identical across sections (enforced by `build_resumed_frames`).
    pub frame_ptr: *mut PyFrame,
    /// valuestackdepth extracted from vable_values (snapshot).
    pub vsd: usize,
    /// interp_jit.py:31 w_globals — namespace pointer from vable_values.
    /// virtualizable.py:126-137 write_from_resume_data_partial:
    /// ALL static fields come from resume data, not from the heap.
    pub namespace: *const (),
    /// resume.py:928-931 consume_one_section: resolved values.
    /// Structure: [live_registers...] — no scalar inputarg header.
    /// RPython parity: vable values come from snapshot, not fail_args.
    pub values: Vec<majit_ir::Value>,
}

/// resume.py:945-956 decode_ref / getvirtual_ptr parity.
///
/// Re-box optimizer-unboxed values back to PyObjectRef for the
/// blackhole's ref register file. RPython's decode_ref dispatches
/// on TAGVIRTUAL/TAGCONST/TAGBOX/TAGSMALLINT; pyre's deadframe
/// already contains typed Values, so we just box Int/Float to
/// W_IntObject/W_FloatObject.
#[allow(dead_code)]
fn materialize_virtual(val: &majit_ir::Value) -> i64 {
    use majit_ir::Value;
    match val {
        Value::Ref(r) => r.as_usize() as i64,
        Value::Int(v) => pyre_object::intobject::w_int_new(*v) as i64,
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v) as i64,
        Value::Void => 0i64,
    }
}

/// resume.py:1028 _callback_i → next_int() → write_an_int.
/// RPython trusts type discipline — no cross-type coercion.
#[allow(dead_code)]
fn materialize_virtual_int(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Int(v) => *v,
        other => panic!("materialize_virtual_int: expected Int, got {:?}", other),
    }
}

/// resume.py:1036 _callback_f → next_float() → write_a_float.
/// RPython trusts type discipline — no cross-type coercion.
#[allow(dead_code)]
fn materialize_virtual_float(val: &majit_ir::Value) -> i64 {
    match val {
        majit_ir::Value::Float(v) => v.to_bits() as i64,
        other => panic!("materialize_virtual_float: expected Float, got {:?}", other),
    }
}

/// Fused recursive call with boxed arg.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_1(
    caller_frame: i64,
    callable: i64,
    boxed_arg: i64,
) -> i64 {
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    // result_type=REF: no RawInt unbox needed — arg is already boxed Ref
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if locals_w!(caller).len() > 0
            && !locals_w!(caller)[0].is_null()
            && unsafe { is_int(locals_w!(caller)[0]) }
        {
            Some(unsafe { w_int_get_value(locals_w!(caller)[0]) })
        } else {
            None
        };
        let boxed = boxed_arg as PyObjectRef;
        let callee_arg0 = if !boxed.is_null() && unsafe { is_int(boxed) } {
            Some(unsafe { w_int_get_value(boxed) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] enter caller_arg0={:?} callee_arg0={:?}",
            caller_arg0, callee_arg0
        );
    }
    let frame_ptr = create_callee_frame_impl(caller_frame, callable, &[boxed_arg_ref]);
    let result = jit_force_callee_frame(frame_ptr);
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() {
        let caller = unsafe { &*(caller_frame as *const PyFrame) };
        let caller_arg0 = if locals_w!(caller).len() > 0
            && !locals_w!(caller)[0].is_null()
            && unsafe { is_int(locals_w!(caller)[0]) }
        {
            Some(unsafe { w_int_get_value(locals_w!(caller)[0]) })
        } else {
            None
        };
        eprintln!(
            "[jit][force-recursive-boxed] exit caller_arg0={:?}",
            caller_arg0
        );
    }
    result
}

/// Fused recursive call with RAW INT arg, boxed result.
///
/// This keeps the trace-side argument in raw-int form even before the callee
/// has stabilized on a raw-int finish protocol. It is a closer match to
/// RPython's recursive portal argument flow than boxing the argument in the
/// trace before every helper-boundary call.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_recursive_call_1(caller_frame, callable, boxed as i64)
}

/// Self-recursive single-arg boxed helper.
///
/// Keeps the boxed helper path off the generic callable redispatch and
/// blackhole fallback route. This mirrors the specialized raw helper:
/// the callee frame is created directly from the caller's code/globals.
/// RPython warmspot.py:941 portal_runner parity.
///
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_1(caller_frame: i64, boxed_arg: i64) -> i64 {
    let boxed_arg_ref = boxed_arg as PyObjectRef;
    if caller_frame == 0 {
        return boxed_arg;
    }
    // result_type=REF: arg is already boxed Ref
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed_arg_ref);
    // blackhole.py:1101-1132 bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which re-enters JIT.
    // warmspot.py:941 ll_portal_runner: maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Self-recursive single-arg helper with raw-int arg and boxed result.
///
/// Mirrors `jit_force_self_recursive_call_1`, but keeps the trace-side
/// argument unboxed so recursive helper-boundary calls do not allocate a
/// temporary `W_Int` in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_argraw_boxed_1(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    // result_type=REF: box the int arg, dispatch as boxed Ref
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    jit_force_self_recursive_call_1(caller_frame, boxed as i64)
}

/// Fully fused recursive call with RAW INT arg — no boxing in trace at all.
///
/// Eliminates ALL per-recursive-call overhead from trace:
///   Before: CallI(box) + CallMayForce(force_1, frame, callable, boxed)
///   After:  CallMayForce(force_raw_1, frame, callable, raw_int)
///
/// Boxing happens inside this function, not in the trace.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_recursive_call_raw_1(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let callable_ref = callable as PyObjectRef;

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_callee_frame_impl_1_boxed(caller_frame, callable_ref, boxed);
    // blackhole.py:1101-1116 bhimpl_recursive_call_r: a recursive call
    // from compiled assembler is `cpu.bh_call_r(portal_runner_adr, ...)`
    // — i.e. it always re-enters through the portal runner. The portal
    // runner (warmspot.py:944-953) calls `maybe_compile_and_run` and
    // then `portal_ptr(*args)`, so the JIT-vs-interpreter decision is
    // made there. There is no "try blackhole first, then fallback to
    // portal_runner" path in RPython.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    result
}

/// Specialized raw-int recursive helper for closure-free self-recursion.
///
/// Unlike `jit_force_recursive_call_raw_1`, this does not need to rediscover
/// the callee's code/globals from a function object on every call. The caller
/// frame already carries the exact recursive target:
/// - `caller.pycode` is the callee code object
/// - `caller.w_globals` is the module globals
/// - `caller.execution_context` is the shared execution context
///
/// Trace-time recursive CALL_ASSEMBLER handles the optimized path. The
/// concrete helper should mirror RPython's force_fn behavior: execute the
/// callee's own frame without JIT on that frame, but let nested portal
/// calls re-enter compiled code through the normal portal runner path.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_force_self_recursive_call_raw_1(caller_frame: i64, raw_int_arg: i64) -> i64 {
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!("[jit][force-self-recursive] enter arg={}", raw_int_arg);
    }
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let w_code = caller.pycode;
    let green_key = crate::eval::make_green_key(w_code, 0);
    let _token_num = self_recursive_dispatch(green_key);

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    let frame_ptr = create_self_recursive_callee_frame_impl_1_boxed(caller_frame, boxed);
    // blackhole.py:1110-1116 bhimpl_recursive_call_r: calls
    // cpu.bh_call_r(portal_runner_adr, ...) which invokes
    // warmspot.py:941 ll_portal_runner. portal_runner re-enters
    // the JIT through maybe_compile_and_run + portal_ptr.
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        // warmspot.py:449 result_type=REF: always boxed Ref
        crate::eval::portal_runner(frame) as i64
    };
    jit_drop_callee_frame(frame_ptr);
    if majit_metainterp::majit_log_enabled() && raw_int_arg <= 4 {
        eprintln!(
            "[jit][force-self-recursive] exit arg={} result={}",
            raw_int_arg, result
        );
    }
    result
}

/// Dynasm x86/assembler.py:347-390 `_build_stack_check_slowpath` parity.
///
/// The interpreter-level slowpath stores the RecursionError in the
/// current thread's pending JIT exception slot for non-dynasm backend
/// glue. Dynasm's prologue, however, mirrors RPython's
/// `pos_exception()` path: after this call returns non-zero, emitted
/// code transfers `majit_backend_dynasm`'s `JIT_EXC_VALUE` into
/// `jf_guard_exc` and stamps `propagate_exception_descr` into
/// `jf_descr`. Bridge the two exception slots here.
#[cfg(feature = "dynasm")]
extern "C" fn dynasm_stack_check_slowpath_for_backend(current: usize) -> u8 {
    let result = pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend(current);
    if result != 0 {
        if let Err(mut exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            majit_backend_dynasm::jit_exc_raise(exc.to_exc_object() as i64);
        }
    }
    result
}

/// Unbox a Ref (PyObjectRef to boxed int) to a raw i64 value.
/// Used by call_assembler_guard_failure's FALLBACK path when the first
/// local is a Ref type (boxed int) instead of raw Int.
fn unbox_int_for_force(raw: i64) -> i64 {
    let obj = raw as pyre_object::PyObjectRef;
    if !obj.is_null() && unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else {
        raw
    }
}

/// resume.py:763-779 VStrPlainInfo.allocate / resume.py:817-829
/// VUniPlainInfo.allocate parity — materialize a Plain string/unicode
/// virtual via the frontend backend's bh_newstr / bh_strsetitem (and
/// unicode variants). Registered into Cranelift's guard-exit recovery
/// path so `rebuild_state_after_failure` hands bridge-input refs a real
/// string pointer instead of NULL (compiler.rs).
#[allow(dead_code)]
fn materialize_str_plain_for_cranelift(is_unicode: bool, chars: &[i64]) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let length = chars.len() as i64;
    let string = if is_unicode {
        backend.bh_newunicode(length)
    } else {
        backend.bh_newstr(length)
    };
    for (i, c) in chars.iter().enumerate() {
        if is_unicode {
            backend.bh_unicodesetitem(string, i as i64, *c);
        } else {
            backend.bh_strsetitem(string, i as i64, *c);
        }
    }
    string
}

/// resume.py:1143-1188 string_concat / slice_string and the unicode
/// counterparts — materialize Concat / Slice string virtuals via
/// cpu.bh_call_r(funcptr, args_i, args_r, args_f, calldescr).
#[allow(dead_code)]
fn materialize_str_call_for_cranelift(
    _is_unicode: bool,
    func: i64,
    calldescr: &majit_ir::DescrRef,
    args_i: &[i64],
    args_r: &[i64],
) -> i64 {
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let cd = calldescr
        .as_call_descr()
        .expect("materialize_str_call: calldescr must downcast to CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
    let result = backend.bh_call_r(
        func,
        if args_i.is_empty() {
            None
        } else {
            Some(args_i)
        },
        if args_r.is_empty() {
            None
        } else {
            Some(args_r)
        },
        None,
        &bh_calldescr,
    );
    result.0 as i64
}

pub fn install_jit_call_bridge() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // warmstate.py:108-128 ll_streq / ll_strhash registration —
        // pyre's `#[jit_interp]` macro emits the canonical `*const
        // &'static str` slot ABI for STR/UNICODE greens
        // (`majit-macros::jit_interp::emit_green_repr`).  Each frontend
        // owns its own `rstr.STR` / `rstr.UNICODE` decoder; pyre
        // registers its `default_str_eq` / `default_str_hash` /
        // `default_unicode_hash` here so `equal_whatever(GreenType::Str,
        // ..)` / `hash_whatever(GreenType::Str, ..)` route to content-
        // aware comparison/hashing for every `greens=[name: str]`
        // JitCell.  Lives on the pyre side (not in metainterp's
        // `JitDriver::new`) so non-pyre frontends supply their own slot
        // ABI without inheriting pyre's via a process-global default.
        majit_ir::value::set_str_resolver(
            majit_ir::value::default_str_eq,
            majit_ir::value::default_str_hash,
        );
        majit_ir::value::set_unicode_resolver(
            majit_ir::value::default_str_eq,
            majit_ir::value::default_unicode_hash,
        );
        // Same frontend-owns-its-object-model split for the `w_class`
        // header identity `OptVirtualize` folds `new_with_vtable` reads
        // to: `SimpleSizeDescr` carries pyre's `ob_type` in its `vtable`
        // slot but knows nothing about `PyType`, so pyre registers the
        // `get_instantiate` decoder here.
        majit_ir::descr::set_w_class_obj_resolver(pyre_jit_trace::descr::w_class_obj_for_vtable);
        register_jit_function_caller(jit_call_user_function_from_frame);
        register_jit_exc_raiser(jit_exc_raise_shim);
        // compile.py:1090 `memory_error = MemoryError()` parity — give
        // the backend malloc helpers a way to set `JIT_EXC_VALUE` to
        // pyre's lazy `W_BaseException(MemoryError, "")` singleton
        // before propagating NULL on OOM.  Backend-shared (mirrors
        // RPython where the same `memory_error` instance is reachable
        // from both the x86 and aarch64 backends).
        majit_backend::register_memory_error_provider(|| {
            pyre_object::interp_exceptions::memory_error_singleton() as i64
        });
        // rpython/translator/c/src/stack.h:42-43 LL_stack_criticalcode_start
        // /stop hooks — wrap blackhole_from_resumedata,
        // handle_async_forcing, and handle_guard_failure_in_trace so
        // StackOverflow doesn't interrupt those critical sections.
        // The pyre helpers are `extern "C" fn()`; thin wrappers adapt
        // them to the Rust `fn()` signature register_criticalcode_hooks
        // expects.
        fn criticalcode_start_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_start();
        }
        fn criticalcode_stop_adapter() {
            pyre_interpreter::stack_check::pyre_stack_criticalcode_stop();
        }
        majit_metainterp::register_criticalcode_hooks(
            criticalcode_start_adapter,
            criticalcode_stop_adapter,
        );
        // rpython/rlib/rstack.py:75-90 stack_almost_full hook — lets
        // compile.py:702-703 and warmstate.py:430 query the recursion-
        // limit-driven PYRE_STACKTOOBIG budget instead of the OS thread
        // stack.
        fn stack_almost_full_adapter() -> bool {
            pyre_interpreter::stack_check::stack_almost_full()
        }
        majit_metainterp::register_stack_almost_full_hook(stack_almost_full_adapter);
        #[cfg(feature = "cranelift")]
        {
            majit_backend_cranelift::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_cranelift::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_cranelift::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_cranelift::register_jitframe_layout(jitframe_layout_info());
            majit_backend_cranelift::register_call_assembler_unbox_int(unbox_int_for_force);
            // resume.py:763-870 VStr/VUni.allocate parity — Cranelift
            // backend's materialize_virtual_recursive invokes these
            // callbacks so that bridge-input refs and call_assembler
            // blackhole inputs (compiler.rs) receive materialized string
            // pointers, not NULL.
            majit_backend_cranelift::register_materialize_str_plain(
                materialize_str_plain_for_cranelift,
            );
            majit_backend_cranelift::register_materialize_str_call(
                materialize_str_call_for_cranelift,
            );
            // rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check
            // parity. Cranelift consumes these addresses to emit the same
            // load/sub/cmp fast path as dynasm, using a stack-slot address as
            // its current-stack approximation and calling slowpath only on miss.
            majit_backend_cranelift::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                pyre_interpreter::stack_check::pyre_stack_check_slowpath_for_backend as *const ()
                    as usize,
            );
            majit_backend_cranelift::register_prologue_probe_addr(
                pyre_interpreter::stack_check::pyre_stack_check_for_jit_prologue as *const ()
                    as usize,
            );
        }
        #[cfg(feature = "dynasm")]
        {
            majit_backend_dynasm::register_call_assembler_force(jit_force_callee_frame);
            majit_backend_dynasm::register_call_assembler_bridge(jit_ca_handle_guard_failure);
            majit_backend_dynasm::register_call_assembler_blackhole(
                jit_blackhole_resume_from_guard,
            );
            majit_backend_dynasm::register_jitframe_layout(jitframe_layout_info_dynasm());
            majit_backend_dynasm::register_call_assembler_unbox_int(unbox_int_for_force);
            // rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check
            // parity. The backend inlines MOV [endaddr]; SUB rsp; CMP [lengthaddr]
            // in every JIT prologue and calls slowpath_addr on miss.
            majit_backend_dynasm::register_stack_check_addresses(
                pyre_interpreter::stack_check::pyre_stack_get_end_adr(),
                pyre_interpreter::stack_check::pyre_stack_get_length_adr(),
                dynasm_stack_check_slowpath_for_backend as *const () as usize,
            );
        }
    });
}

/// compile.py:701-716 handle_fail → resume_in_blackhole parity.
///
/// RPython: guard failure always resumes via jitcode-level blackhole
/// (blackhole_from_resumedata → _run_forever). There is no IR-level
/// blackhole in RPython.
///
/// When rd_numb is available, uses ResumeDataDirectReader for exact
/// frame decoding (resume.py:1312 parity).
/// Complete a CALL_ASSEMBLER callee whose bridge walk already carried it past
/// the guard, so the guard-state resume must not re-run the region.
///
/// `callee_frame` is the callee `PyFrame*` the guard failed in — pyre's CA
/// virtualizable layout puts it at `fail_values[0]` (the same layout
/// `jit_ca_handle_guard_failure` reads `raw_values[0]` under).
/// Returns `Some(result)` once the walk completed the callee, i.e. the
/// `DoneWithThisFrame` / `ExitFrameWithExceptionRef` the assembler caller
/// catches (pyjitpl.py:1688-1698, jitexc.py), and `None` when the caller still
/// has to resume the guard state.
///
/// Shared by both CALL_ASSEMBLER completions — the native back-to-back
/// blackhole hook below and the wasm in-guest deopt helper
/// (`wasm_ca_resume_deopt`) — so the two cannot drift apart on which walk
/// outcomes are replay-safe.
fn ca_complete_after_bridge_walk(
    ca_finished_frame: usize,
    ca_adopted_frame: usize,
    callee_frame: usize,
) -> Option<i64> {
    // The CA bridge walk ran the callee to its finishframe concretely and
    // kept the finish-concrete stash (`CA_WALK_FINISHED_FRAME` handshake set
    // by `trace_and_compile_from_bridge`); a guard-state resume would re-run
    // the resumed region over the already-applied effects.  Complete the
    // callee with the stashed concrete instead.
    if ca_finished_frame != 0 && ca_finished_frame == callee_frame {
        match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
                let result = match cv {
                    // A void return stashes `Null`, i.e. Python `None`.
                    pyre_jit_trace::state::ConcreteValue::Null => pyre_object::w_none(),
                    other => other.to_pyobj(),
                };
                return Some(result as i64);
            }
            Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
                let pyre_jit_trace::state::ConcreteValue::Ref(exc_ref) = cv else {
                    unreachable!("FinishConcrete::Raise must hold a concrete Ref")
                };
                debug_assert!(!exc_ref.is_null());
                publish_residual_call_exception(exc_ref as i64);
                return Some(0);
            }
            // The epilogue reset the stash between the hook calls; fall
            // through to the guard-state resume.
            None => {}
        }
    } else {
        // A kept stash without a matching frame must not leak into a later
        // top-level portal `fbw_finish_concrete_take`.
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
    }

    // raise_continue_running_normally parity (pyjitpl.py:3048-3091 +
    // warmspot.py:970-983): the CA bridge walk committed its end-of-walk
    // state into the live callee frame and the frame adopted it. The walk
    // already executed the resumed region's effects concretely, so a
    // guard-state resume would re-apply them over the advanced heap
    // (double-execution / stale-value return). Complete the callee from its
    // adopted state via the portal runner instead — the same
    // portal_ptr(*args) completion the ContinueRunningNormally arm of
    // handle_blackhole_result performs.
    if ca_adopted_frame != 0 && ca_adopted_frame == callee_frame {
        let frame = unsafe { &mut *(ca_adopted_frame as *mut PyFrame) };
        return match crate::eval::portal_runner_result(frame) {
            Ok(result) => Some(result as i64),
            Err(mut err) => {
                let exc_obj = err.to_exc_object();
                if exc_obj != pyre_object::PY_NULL {
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
                    store_jit_exception(exc_obj as i64);
                }
                Some(0)
            }
        };
    }

    None
}

fn jit_blackhole_resume_from_guard(
    descr_addr: usize,
    fail_values_ptr: *const i64,
    num_fail_values: usize,
    raw_deadframe_ptr: *const i64,
    num_raw_deadframe: usize,
    guard_exc: i64,
) -> Option<i64> {
    let ca_adopted_frame = CA_WALK_ADOPTED_FRAME.with(|c| c.replace(0));
    let ca_finished_frame = CA_WALK_FINISHED_FRAME.with(|c| c.replace(0));
    let ca_resume_frame = CA_WALK_RESUME_FRAME.with(|c| c.replace(0));
    let ca_resume_deadframe = CA_WALK_RESUME_DEADFRAME.with(|c| c.borrow_mut().take());

    // rstack.stack_check_slowpath → _StackOverflow parity: drain the
    // pending JIT-prologue overflow exception when the backend probe
    // tripped. The blackhole resume path is one of the three
    // boundaries the user listed (compiled entry / call_assembler /
    // blackhole resume), so we surface RecursionError here as well as
    // in eval.rs. We do this BEFORE setting up resume state so deep
    // recursion through the blackhole interpreter cannot accumulate
    // further damage.
    if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        // Stash for the eval loop to surface — same channel the
        // blackhole/force callbacks already use for cross-FFI errors.
        crate::call_jit::set_pending_ca_exception(exc);
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
        return None;
    }

    if fail_values_ptr.is_null() || num_fail_values == 0 {
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_reset();
        return None;
    }
    let fail_values_raw = unsafe { std::slice::from_raw_parts(fail_values_ptr, num_fail_values) };
    let mut fail_values_owned;
    let fail_values = if let Some(values) = ca_resume_deadframe.as_deref() {
        values
    } else if ca_resume_frame != 0 {
        fail_values_owned = fail_values_raw.to_vec();
        fail_values_owned[0] = ca_resume_frame as i64;
        fail_values_owned.as_slice()
    } else {
        fail_values_raw
    };

    if let Some(result) =
        ca_complete_after_bridge_walk(ca_finished_frame, ca_adopted_frame, fail_values[0] as usize)
    {
        return Some(result);
    }

    let raw_deadframe_raw = if let Some(values) = ca_resume_deadframe.as_deref() {
        values
    } else if !raw_deadframe_ptr.is_null() && num_raw_deadframe > 0 {
        unsafe { std::slice::from_raw_parts(raw_deadframe_ptr, num_raw_deadframe) }
    } else {
        fail_values
    };
    let mut raw_deadframe_owned;
    let raw_deadframe = if ca_resume_frame != 0 && !raw_deadframe_raw.is_empty() {
        raw_deadframe_owned = raw_deadframe_raw.to_vec();
        raw_deadframe_owned[0] = ca_resume_frame as i64;
        raw_deadframe_owned.as_slice()
    } else {
        raw_deadframe_raw
    };

    // compile.py:710-716 `resume_in_blackhole(descr, deadframe)` parity:
    // recover the failed descr from `descr_addr` (history.py:125
    // `cpu.get_latest_descr` is the C-ABI carrier) and derive
    // (trace_id, fail_index) from descr identity.  The recovery is
    // infallible for live JIT code — `Backend::fail_descr_arc_from_addr`
    // panics if the raw value is not a live `FailDescrCell` pointer,
    // matching RPython's `cpu.get_latest_descr(deadframe)`
    // (warmspot.py:1021) which has no failure mode.
    use majit_backend::Backend;
    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let descr_arc = backend.fail_descr_arc_from_addr(descr_addr);
    let descr_fd = descr_arc
        .as_fail_descr()
        .expect("fail_descr_arc_from_addr returned non-FailDescr");
    let trace_id = descr_fd.trace_id();
    let fail_index = descr_fd.fail_index_per_trace();

    // `descr_owning_jct == None` is the giveup signal: the descr's
    // `rd_loop_token.loop_token_wref()` is dead (memmgr-evicted JCT —
    // pyjitpl.py:2898 should-be-rare path). compile.giveup() raises
    // `SwitchToBlackhole(ABORT_BRIDGE)` (compile.py:27-29) and falls
    // through here.
    //
    // Note (pyre-only, Python-portal-specific):
    // pyre's resume storage is keyed by `(green_key, trace_id, fail_index)`,
    // so we MUST recover a green_key to look up the storage.  PyPy's
    // `resume_in_blackhole` uses descr identity directly (descr.rd_data),
    // so it has no such recovery problem.
    //
    // When the JCT weakref is dead we exploit pyre's CALL_ASSEMBLER
    // virtualizable layout `vable_boxes = [frame, ni, code, vsd, ns,
    // locals..., stack...]` (as `jit_ca_handle_guard_failure` reads it) —
    // `fail_values[0]` IS the callee's `PyFrame*`, so `frame.pycode` plus
    // `pc=0` reconstructs the entry green_key.  This contract is
    // Python-portal-specific and would NOT hold for a non-virtualizable
    // JIT or a portal whose first fail arg is a scalar.  Keying resume
    // storage by descr identity directly would remove the need for this
    // recovery block.
    let actual_green_key = match majit_backend::descr_owning_jct(descr_fd).map(|j| j.green_key()) {
        Some(gk) => gk,
        None if num_fail_values >= 1 => {
            let frame_ptr = fail_values[0] as *const pyre_interpreter::pyframe::PyFrame;
            if !frame_ptr.is_null() {
                let code = unsafe { (*frame_ptr).pycode };
                crate::eval::make_green_key(code, 0)
            } else {
                0
            }
        }
        None => 0,
    };

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] gk={} trace={} fail_idx={} nvals={}",
            actual_green_key, trace_id, fail_index, num_fail_values,
        );
    }

    // --- Path 1: rd_numb-based resume (resume.py:1312 exact parity) ---
    // When rd_numb is present, use ResumeDataDirectReader to decode
    // frame sections precisely, matching RPython blackhole_from_resumedata.
    //
    // compile.py:853 guard-owned `ResumeGuardDescr` storage — share the
    // pool through `Arc<ResumeStorage>` so blackhole resume reads the
    // same `rd_consts` the GC root walker updates. No owned-Vec copy.
    // resume.py parity: deadframe_types tells decode_ref() whether a TAGBOX
    // slot holds a raw int (needs boxing) or a GcRef (use as-is). Without it,
    // unboxed ints are treated as pointers → SIGSEGV. Both come out of one
    // layout resolution so the storage and the types cannot describe
    // different deadframes.
    if let Some((storage, deadframe_types)) =
        driver.get_resume_storage_with_slot_types(actual_green_key, trace_id, fail_index)
    {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[blackhole-resume] rd_numb len={} rd_consts len={} raw_deadframe len={}",
                storage.rd_numb.len(),
                storage.rd_consts().len(),
                raw_deadframe.len(),
            );
        }
        // resume.py:922 storage.rd_consts: the decoder borrows the shared
        // pool; TAGCONST Ref entries stay visible to `walk_rd_consts_refs`.
        // resume.py:924 _prepare_pendingfields(storage.rd_pendingfields):
        // deferred field writes must be replayed before consume_vref_and_vable.
        // blackhole.py:1794 `current_exc = _prepare_resume_from_failure(
        // guard_opnum, deadframe)`. The backend trampoline grabbed
        // `jf_guard_exc` off the jitframe (`cpu.grab_exc_value`,
        // llmodel.py:240) and threaded it through the C-ABI `guard_exc`
        // parameter, so a GUARD_NO_EXCEPTION / GUARD_EXCEPTION /
        // GUARD_NOT_FORCED failure inside a CALL_ASSEMBLER-entered
        // callee delivers its pending exception to the blackhole resume
        // instead of resuming the no-exception continuation with a NULL
        // result.
        // compile.py:950-958 `ResumeGuardForcedDescr.handle_fail` fishes the
        // cache `handle_async_forcing` saved; no other `handle_fail` does.
        // `fail_values[0]` IS the callee's `PyFrame*` for the Python portal
        // (the entry-green-key recovery above relies on the same contract), so
        // it names the frame a force would have attached its cache to.
        let all_virtuals = if descr_arc.is_guard_forced() {
            crate::eval::take_forced_virtuals_for_frame(
                fail_values
                    .first()
                    .map(|v| *v as *const pyre_interpreter::pyframe::PyFrame)
                    .unwrap_or(std::ptr::null()),
            )
        } else {
            None
        };
        let result = blackhole_resume_via_rd_numb(
            &storage.rd_numb,
            storage.rd_consts(),
            raw_deadframe,
            Some(&storage.rd_pendingfields),
            Some(&storage.rd_virtuals),
            Some(deadframe_types.as_slice()),
            guard_exc,
            false, // CALL_ASSEMBLER portal is jd0 (virtualizable)
            all_virtuals,
            None, // `raw_deadframe` is rooted only by the copy made inside
        );
        return handle_blackhole_result(result, actual_green_key);
    }

    // RPython compile.py:701-716 parity: every guard must have rd_numb
    // from capture_resumedata + store_final_boxes_in_guard (resume.py:397).
    // Hitting this path means a guard was compiled without snapshot data.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[blackhole-resume] no rd_numb for key={} trace={} fail={} (force_fn fallback)",
            actual_green_key, trace_id, fail_index,
        );
    }
    None
}

/// RAII guard registering each `Ref`-typed slot of the resume `deadframe`
/// copy as a GC root for the duration of `blackhole_from_resumedata`.
///
/// `deadframe` is an off-heap copy of the guard-failure values
/// (`result.values`).  The collector is moving, so a minor collection
/// triggered while `blackhole_from_resumedata` lazily materializes virtuals
/// (`getvirtual_ptr` → allocator) relocates the boxed objects and leaves the
/// copy holding from-space pointers; `resume.rs::decode_ref` then reads
/// a stale pointer for a box-sourced slot and the blackhole dereferences
/// freed memory.  Resume *constants* are already forwarded by
/// `rd_consts_root_walker_area`, but the box-sourced slots here are not.
/// Registering each `Ref` element slot makes the root walker forward it in
/// place.
///
/// Only `Ref`-typed slots are registered: `Int`/`Float` slots hold raw
/// scalars (`decode_ref` boxes those lazily via `box_int`/`box_float`), and a
/// scalar that happened to alias a managed address must never be treated as a
/// root.  `deadframe_types[idx]` is parallel to `deadframe[idx]` (resume.rs
/// `decode_ref` keys the same index), so the type gate is exact.
struct ResumeDeadframeRoots {
    slots: Vec<*mut *mut u8>,
}

impl ResumeDeadframeRoots {
    fn register_pyframe_locals_slot(value: i64, slots: &mut Vec<*mut *mut u8>) {
        let ptr = value as *mut u8;
        if ptr.is_null()
            || !pyre_object::gc_hook::try_gc_owns_object(ptr)
            || majit_gc::gc_is_nursery_object(ptr as usize)
        {
            return;
        }
        let current = pyre_object::gc_hook::try_gc_current_object_address(ptr);
        if current.is_null() || !pyre_object::gc_hook::try_gc_owns_object(current) {
            return;
        }
        let type_id = unsafe { (*majit_gc::header::header_of(current as usize)).type_id() };
        if type_id != pyre_interpreter::pyframe::PYFRAME_GC_TYPE_ID {
            return;
        }
        let frame = current as *mut PyFrame;
        let slot = unsafe { std::ptr::addr_of_mut!((*frame).locals_cells_stack_w) as *mut *mut u8 };
        if unsafe { pyre_object::gc_hook::try_gc_add_root(slot) } {
            slots.push(slot);
        }
    }

    fn register(deadframe: &mut [i64], deadframe_types: Option<&[majit_ir::Type]>) -> Self {
        let mut slots = Vec::new();
        if let Some(types) = deadframe_types {
            for (idx, ty) in types.iter().enumerate() {
                if !matches!(ty, majit_ir::Type::Ref) {
                    continue;
                }
                let Some(cell) = deadframe.get_mut(idx) else {
                    break;
                };
                // Null / `NULLREF` slots carry no object to forward.
                if *cell == 0 {
                    continue;
                }
                let slot = cell as *mut i64 as *mut *mut u8;
                if unsafe { pyre_object::gc_hook::try_gc_add_root(slot) } {
                    slots.push(slot);
                }
                Self::register_pyframe_locals_slot(*cell, &mut slots);
            }
        }
        Self { slots }
    }
}

impl Drop for ResumeDeadframeRoots {
    fn drop(&mut self) {
        for &slot in &self.slots {
            pyre_object::gc_hook::try_gc_remove_root(slot);
        }
    }
}

/// RAII guard rooting a single bare `Ref`-carrying `i64` slot across an
/// allocating region. Mirrors the dynasm CA helper's `gc_add_root` over the
/// grabbed guard exception (majit-backend-dynasm `ca_helper`): once the slot is
/// registered the collector forwards `*slot` in place if it moves the object,
/// so the bare carrier stays valid across a hook that may allocate. `register`
/// is a no-op for a null slot (non-exception guards leave `grab_exc_value` 0).
#[cfg(target_arch = "wasm32")]
struct BareRefRoot {
    slot: Option<*mut *mut u8>,
}

#[cfg(target_arch = "wasm32")]
impl BareRefRoot {
    fn register(value: &mut i64) -> Self {
        if *value == 0 {
            return Self { slot: None };
        }
        let slot = value as *mut i64 as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self {
            slot: registered.then_some(slot),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for BareRefRoot {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            pyre_object::gc_hook::try_gc_remove_root(slot);
        }
    }
}

/// `blackhole.py:1679-1682 _exit_frame_with_exception`:
///
/// ```python
/// e = cast_opaque_ptr(GCREF, e)
/// raise ExitFrameWithExceptionRef(e)
/// ```
///
/// Upstream hands the value on as an opaque GCREF and every later
/// classification runs through a real class lookup.  Pyre instead reads a
/// `#[repr(u8)]` `ExcKind` tag out of the object and feeds it to
/// `kind_from_exc`, a wildcard-free match the compiler may lower to a
/// bounds-check-free jump table — so a word that is not a `W_BaseException`
/// becomes an indirect branch to an address derived from whatever byte sits
/// at the tag offset.  `w_exception_kind_checked` restores the class check in
/// front of that read; a value that fails it names itself here instead.
fn exit_frame_exception_ref(
    exc_value: i64,
    site: &str,
    context: impl FnOnce() -> String,
) -> pyre_interpreter::PyError {
    let obj = exc_value as PyObjectRef;
    if unsafe { pyre_object::interp_exceptions::w_exception_kind_checked(obj) }.is_some() {
        return unsafe { pyre_interpreter::PyError::from_exc_object(obj) };
    }
    if obj.is_null() {
        return pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::RuntimeError,
            "blackhole exception (null exc_value)",
        );
    }
    reject_non_exception_channel_value(obj, site, context)
}

/// Report a value that reached an exception channel without being a
/// `W_BaseException`, and abort.
///
/// Never returns: the caller is about to classify the value by its `ExcKind`
/// tag, and there is no correct classification for a value that has no tag.
fn reject_non_exception_channel_value(
    obj: PyObjectRef,
    site: &str,
    context: impl FnOnce() -> String,
) -> ! {
    // Nothing has been read out of `obj` yet, and this function only runs for a
    // value already known not to be a `W_BaseException` — a small integer or an
    // address at the end of a mapping reaches here too, and every read below
    // can fault. Emit the pointer first so that a fault still leaves evidence.
    //
    // Every line goes through the interpreter's own fd-2 seam, not `eprintln!`:
    // the wasm guest has no process stderr, so an `eprintln!` here is a silent
    // sink and this reporter — whose only job is producing evidence before the
    // abort — would leave nothing but the panic string on the one backend where
    // the value is hardest to reconstruct.
    pyre_interpreter::host_seam::emit_stderr(
        format!("[jit][BUG] {site}: not a W_BaseException: obj={obj:p}\n").as_bytes(),
    );
    // Then the raw header, before touching anything reached *through* it: if
    // `ob_type` is itself garbage the name lookup below faults, and then this
    // line is all the evidence there is.
    let tag = unsafe { pyre_object::interp_exceptions::w_exception_kind_byte(obj) };
    let words = unsafe { std::slice::from_raw_parts(obj as *const u64, 4) };
    pyre_interpreter::host_seam::emit_stderr(
        format!(
            "[jit][BUG] {site}: obj={obj:p} tag_byte={tag} \
             words=[{:#018x} {:#018x} {:#018x} {:#018x}]\n",
            words[0], words[1], words[2], words[3],
        )
        .as_bytes(),
    );
    // Second line, and only now: the context may probe words of unproven
    // provenance, so the header above must already be out.
    pyre_interpreter::host_seam::emit_stderr(
        format!("[jit][BUG] {site}: context: {}\n", context()).as_bytes(),
    );
    // `words[0]` is `ob_type` and `words[1]` is `w_class`; only read through
    // either when the pointer has the shape of one.  `ob_type` names the
    // built-in layout ("object" for every instance of a Python class), so the
    // `w_class` name is the one that identifies the value.
    let type_name = if words[0] != 0 && words[0].is_multiple_of(8) {
        unsafe { pyre_object::pyobject::type_name_of(obj) }
    } else {
        "<unreadable>"
    };
    let class_name = if words[1] != 0 && words[1].is_multiple_of(8) {
        unsafe { pyre_object::w_type_get_name(words[1] as PyObjectRef).to_string() }
    } else {
        "<none>".to_string()
    };
    panic!(
        "{site}: exception channel value is not a W_BaseException \
         (obj={obj:p} tag_byte={tag} type={type_name} class={class_name})"
    );
}

/// `executioncontext.py:91-107 ExecutionContext.leave`'s frame-chain half, for a
/// frame a blackhole resumed into and has now finished.
///
/// The profile-hook half (`if self.profilefunc: self._trace(frame,
/// 'leaveframe', w_exitvalue)`) stays with the interpreter's own
/// [`pyre_interpreter::PyExecutionContext::leave`], for the reason
/// `walker_ec_leave` states: `is_being_profiled` is a portal-driver green, so a
/// trace recorded with profiling off is only ever entered with profiling off,
/// and the blackhole resuming out of it inherits that key.
fn leave_resumed_blackhole_frame(
    frame: &majit_metainterp::blackhole::BlackholeInterpreter,
    got_exception: bool,
) {
    let frame_ptr = frame.virtualizable_ptr as *mut PyFrame;
    if frame_ptr.is_null() {
        return;
    }
    let ec = unsafe { (*frame_ptr).execution_context as *mut pyre_interpreter::PyExecutionContext };
    if ec.is_null() {
        return;
    }
    let frame_vref = unsafe { (*ec).topframeref };
    if !std::ptr::eq(
        pyre_interpreter::executioncontext::vref_referent(frame_vref),
        frame_ptr,
    ) {
        return;
    }
    // executioncontext.py:91-107 `leave`: a guard-failure blackhole resumes
    // inside a frame whose `enter` already ran in compiled code. Advancing to
    // `nextblackholeinterp` therefore has to close that still-open scope. The
    // resume-data frame chain itself is not the application frame chain, and
    // releasing a BlackholeInterpreter does not restore `topframeref`.
    //
    // Nothing here forces a vref. The guard above already established that
    // `frame_vref` resolves to this frame, so `leave`'s own `frame_vref()` has
    // nothing left to materialize, and the caller is reached through
    // `vref_referent` rather than `get_f_back`: this runs once the compiled
    // frame is gone, and forcing a vref that was finished with the NULL form
    // is exactly what `force_pyframe_vref` refuses.
    unsafe {
        (*ec).topframeref = (*frame_ptr).f_backref;
        if (*frame_ptr).escaped() || got_exception {
            let f_back = pyre_interpreter::executioncontext::vref_referent((*frame_ptr).f_backref);
            if !f_back.is_null() {
                (*f_back).mark_as_escaped();
            }
        }
    }
}

/// resume.py:1312 blackhole_from_resumedata parity:
/// Decode rd_numb via ResumeDataDirectReader, build blackhole chain,
/// run _run_forever.
pub fn blackhole_resume_via_rd_numb(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    deadframe: &[i64],
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    deadframe_types: Option<&[majit_ir::Type]>,
    guard_exc: i64,
    // compile.py:990 `vinfo = self.jitdriver_sd.virtualizable_info`: a novable
    // jitdriver (jd1 `unpackiterable_driver`) has `vinfo=None`, so its resume
    // data carries no vable section. Decoding with the (jd0) vinfo would try to
    // consume a phantom vable and dereference garbage; a novable resume passes
    // `None` for both the vinfo and the per-frame virtualizable handle.
    novable: bool,
    // compile.py:956-958 — the cache `handle_async_forcing` already
    // materialized, when this resume is the GUARD_NOT_FORCED that follows a
    // force. `None` for every other guard.
    all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
    // The caller's own rooting of the `deadframe` slice it passed, when it has
    // one. It covers the same window as `deadframe_roots` below and must end at
    // the same point, so the caller hands ownership over instead of holding it
    // across this call: a caller-side scope would stay registered for the whole
    // forward run.
    caller_deadframe_roots: Option<majit_metainterp::resume::DeadFrameRefRoots>,
) -> BlackholeResult {
    // Same window as `handle_fail`, for every blackhole resume including the
    // CALL_ASSEMBLER caller: the decode below rebuilds virtuals through the
    // blackhole allocator while the grabbed exception is still only a bare
    // pointer with no deadframe root behind it.
    let _guard_exc_root = majit_metainterp::blackhole::GuardExcRoot::park(guard_exc);

    let nbody_debug = pyre_nbody_debug_enabled();
    use majit_metainterp::resume;

    // Thread-local BH pool (RPython BlackholeInterpBuilder). Each access
    // is scoped to a single call so that bh.run() (which may re-enter
    // blackhole_resume_via_rd_numb) cannot create overlapping &mut refs.
    thread_local! {
        static BH_BUILDER_RD: std::cell::UnsafeCell<majit_metainterp::blackhole::BlackholeInterpBuilder> =
            std::cell::UnsafeCell::new(pyre_jit_trace::jitcode_runtime::build_pyre_production_bh_builder());
    }
    let sync_bh_builder_control_opcodes =
        |builder: &mut majit_metainterp::blackhole::BlackholeInterpBuilder| {
            let (op_live, op_catch_exception, op_rvmprof_code) =
                pyre_jit_trace::state::blackhole_control_opcodes();
            builder.setup_cached_control_opcodes(op_live, op_catch_exception, op_rvmprof_code);
        };
    let release_bh_rd = |bh: majit_metainterp::blackhole::BlackholeInterpreter| {
        BH_BUILDER_RD.with(|cell| unsafe { (&mut *cell.get()).release_interp(bh) });
    };

    // resume.py:1339 jitcodes[jitcode_pos]: resolve jitcode_index + pc
    // through the store the frame numbered against.
    let resolve_jitcode = |jitcode_index: i32, pc: i32| -> Option<resume::ResolvedJitCode> {
        if pc < 0 {
            return None;
        }
        let op_live = pyre_jit_trace::state::blackhole_control_opcodes().0 as u8;
        // A novable jitdriver (jd1 `unpackiterable_driver`) numbers its drain
        // frames' `jitcode_index` in the build-time `jitcode_runtime` table
        // (the extracted `_unpackiterable_unknown_length` body plus its inlined
        // build-time callees), NOT the runtime `MetaInterpStaticData.jitcodes`
        // store.  The runtime store is a different index space keyed by Python
        // CodeObject, so its low indices hold unrelated jd0 PyCode jitcodes
        // whose liveness at the same pc mistypes the drain's 2 refs as ints
        // (`Const::getint on Ref`).  Resolve against the same table the frame
        // numbered against — mirroring the driver's own bridge resume
        // (`run_compiled_detailed_with_bridge_keyed`, jitdriver.rs), which
        // resolves through the flat build-time `jitcode_registry`.
        if novable {
            let canonical =
                pyre_jit_trace::jitcode_runtime::get_jitcode_by_index(jitcode_index as usize)?;
            if !canonical.can_decode_live_vars(pc as usize, op_live) {
                return None;
            }
            let core = majit_metainterp::JitCode::from_canonical((*canonical).clone());
            return Some(resume::ResolvedJitCode::new(
                std::sync::Arc::new(core),
                pc as usize,
            ));
        }
        let pyjitcode = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        if pyjitcode.has_abort_opcode() {
            return None;
        }
        // A published resume frame carries a decodable JitCode `-live-`
        // coordinate. An unrepresentable frame declines this blackhole path.
        let resolved_pc = if pyjitcode.jitcode.can_decode_live_vars(pc as usize, op_live) {
            pc as usize
        } else {
            return None;
        };
        Some(
            resume::ResolvedJitCode::new(pyjitcode.jitcode.clone(), resolved_pc)
                .with_virtualizable_stack_base(pyjitcode.metadata.stack_base),
        )
    };

    // Own the guard-failure values in host memory so the box-sourced `Ref`
    // slots can be registered as GC roots: `blackhole_from_resumedata` below
    // lazily materializes virtuals, and a minor collection during that work
    // would relocate the boxed objects out from under the off-heap copy.
    // Rooting forwards each `Ref` slot in place; `decode_ref` then reads the
    // live (to-space) pointer rather than a dangling from-space one.  The
    // `to_vec` uses the host allocator, so it cannot itself trigger a GC.
    let mut deadframe_buf: Vec<i64> = deadframe.to_vec();
    let deadframe_roots = ResumeDeadframeRoots::register(&mut deadframe_buf, deadframe_types);
    let deadframe: &[i64] = &deadframe_buf;

    // resume.py:983-991 _prepare_virtuals: convert RdVirtualInfo → VirtualInfo
    // for lazy materialization in getvirtual_ptr/getvirtual_int.
    let count = deadframe.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals.map(|rd_virts| {
        let num_virtuals = rd_virts.len();
        rd_virts
            .iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count, num_virtuals))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // resume.py:1312-1343 blackhole_from_resumedata:
    // ResumeDataDirectReader decodes rd_numb, builds BH chain.
    // compile.py:990 parity: vinfo = self.jitdriver_sd.virtualizable_info —
    // read the active driver's cached Arc instead of rebuilding a fresh
    // VirtualizableInfo, so a single VirtualizableInfo identity is shared
    // with tracing, setup_bridge_sym, and the guard-failure recovery
    // consumers. resume.py:1314 vrefinfo = metainterp_sd.virtualref_info —
    // hand the metainterp's own VRefInfo through so consume_virtualref_info
    // can decode JIT_VIRTUAL_REF handles. resume.py:1316 ginfo is currently
    // unused in pyre (no greenfield_info installed on the driver).
    let (driver, driver_vinfo) = crate::eval::driver_pair();
    let vinfo_dyn: &dyn resume::VirtualizableInfo = driver_vinfo.as_ref();
    // A novable driver's resume data has no vable section; pass no vinfo so the
    // decoder skips `consume_vable_info` entirely.
    let vinfo_arg: Option<&dyn resume::VirtualizableInfo> =
        if novable { None } else { Some(vinfo_dyn) };
    let vrefinfo_dyn: &dyn resume::VRefInfo = driver.meta_interp().virtualref_info();
    let allocator = crate::eval::PyreBlackholeAllocator;
    // resume.py:1022 `self.metainterp_sd.liveness_info` — the one pool, for
    // every driver. A novable drain frame's `-live-` markers carry offsets
    // baked at extraction into the build-time byte stream; those bytes are the
    // prefix of this buffer (`Assembler::resuming_build_time_liveness`), so a
    // baked offset and a runtime-interned one address the same space and no
    // per-driver pick is needed. Snapshot once per call so the slice outlives
    // ResumeDataDirectReader.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();
    // Scope the &mut to chain construction; the run() loop below uses
    // release_bh_rd to drop and re-acquire the borrow.
    let bh = BH_BUILDER_RD.with(|cell| unsafe {
        let builder = &mut *cell.get();
        sync_bh_builder_control_opcodes(builder);
        resume::blackhole_from_resumedata(
            builder,
            &resolve_jitcode,
            rd_numb,
            rd_consts,
            &all_liveness,
            deadframe,
            deadframe_types,        // deadframe_types: decode_ref boxes TAGBOX ints
            rd_virtuals_slice,      // rd_virtuals
            rd_guard_pendingfields, // rd_guard_pendingfields
            Some(vrefinfo_dyn),     // resume.py:1314 metainterp_sd.virtualref_info
            vinfo_arg,              // resume.py:1312 self.jitdriver_sd.virtualizable_info
            None,                   // resume.py:1316 greenfield_info unused in pyre
            None,                   // heap PyFrame identity remains the live TAGBOX
            all_virtuals,           // resume.py:1373-1374 GUARD_NOT_FORCED cache
            &allocator,
        )
    });

    let Some((mut bh, virtualizable_ptr)) = bh else {
        if nbody_debug {
            eprintln!("[nbody-debug] blackhole_resume_via_rd_numb failed: builder returned None");
        }
        return BlackholeResult::Failed;
    };

    // resume.py:1404: virtualizable_ptr was read by consume_vable_info
    // from the vable section. Set on the blackhole for vable bytecodes.
    // A novable resume runs no vable opcodes, so leave both the pointer and
    // the vinfo handle unset (null) instead of borrowing jd0's.
    if !novable {
        if virtualizable_ptr != 0 {
            bh.virtualizable_ptr = virtualizable_ptr;
        } else if !deadframe.is_empty() {
            // Fallback for guards without vable section.
            bh.virtualizable_ptr = deadframe[0];
        }
        bh.virtualizable_info = crate::eval::get_virtualizable_info();
    }
    // Last read of `deadframe`.  `blackhole_from_resumedata` has copied every
    // live value into the blackhole register banks, which the collector reaches
    // through their own root source (`walk_bh_regs`), so the off-heap copy is
    // dead from here on.  Release its roots BEFORE the forward run: `bh.run()`
    // executes the whole remainder of the resumed frame — for a module-level
    // frame that is the rest of the program — and every `Ref` cell left
    // registered pins the object it happened to hold at the guard, whether or
    // not the resumed code still uses it.  A `for v in gen(): break` leaves the
    // abandoned generator in such a cell, so it is never collected and its
    // `finally` never runs.  `blackhole.py:1782-1796 resume_in_blackhole` ends
    // `deadframe`'s live range at `_prepare_resume_from_failure`, before
    // `_run_forever`, for the same reason.
    drop(deadframe_roots);
    drop(caller_deadframe_roots);
    // resume.py:1332-1343 builds the caller chain (`nextblackholeinterp`)
    // but does not set the virtualizable-info handle on each frame.  pyre
    // stores the vinfo per-`BlackholeInterpreter` (RPython reads it from
    // the field descriptor `blackhole.py:1374 fielddescr.get_vinfo()`, a
    // global), so every caller frame that runs a vable opcode after the
    // innermost frame returns to it needs the same handle.  Propagate it
    // down the whole chain, mirroring the forward-exec inheritance
    // `self.virtualizable_info = parent.virtualizable_info` performed when
    // a frame enters a callee.
    // blackhole.py:1095-1099 get_portal_runner parity:
    //   jitdriver_sd = self.builder.metainterp_sd.jitdrivers_sd[jdindex]
    //   fnptr        = adr2int(jitdriver_sd.portal_runner_adr)
    //   calldescr    = jitdriver_sd.mainjitcode.calldescr
    let jitdrivers_sd = vec![majit_metainterp::blackhole::BhJitDriverSd {
        result_type: majit_metainterp::blackhole::BhReturnType::Ref,
        portal_runner_ptr: Some(bh_portal_runner_c),
        mainjitcode_calldescr: {
            // `get_portal_runner` returns this descr paired with
            // `bh_portal_runner_c`, and the blackhole recursive-portal resume
            // (`bhimpl_recursive_call_r` -> `bh_call_r`) verifies the merged
            // green+red argument banks against it — so it must describe the
            // portal runner, not the traced function.  `bh_portal_runner_c`
            // consumes the warmspot `portalfunc_ARGS` (`warmspot.py:972-975`):
            // greens `[next_instr:i, is_being_profiled:i, pycode:r]` + reds
            // `[frame:r, ec:r]` (`interp_jit.py:67-68`) = 2 int + 3 ref.
            // `bh.jitcode.calldescr` is the traced function's own frame-based
            // residual-call descr and does not describe the runner.
            let mut d = bh.jitcode.calldescr.clone();
            d.arg_classes = "iirrr".to_string();
            d.result_type = 'r';
            d
        },
    }];
    {
        let vinfo = bh.virtualizable_info;
        let mut current = Some(&mut bh);
        while let Some(frame) = current {
            frame.virtualizable_info = vinfo;
            frame.record_caught_exception = Some(record_caught_blackhole_traceback);
            // Upstream resolves `jitdrivers_sd[jdindex]` through
            // `self.builder.metainterp_sd` (blackhole.py:1079, :1096), so every
            // frame of a chain shares one table.  `BlackholeInterpBuilder`
            // carries that snapshot and `acquire_interp` hands it out, but this
            // path derives the entry from the innermost frame's own jitcode and
            // so cannot seed the builder before `blackhole_from_resumedata`
            // acquires the chain.  Publish it to every frame instead: any frame
            // above the bottom one reaches `bhimpl_jit_merge_point`'s
            // recursive-portal branch, which indexes the table directly.
            frame.jitdrivers_sd = jitdrivers_sd.clone();
            current = frame.nextblackholeinterp.as_deref_mut();
        }
    }

    // Portal red-arg registers (`pypy/module/pypyjit/interp_jit.py:67
    // reds = ['frame', 'ec']`) are filled per-frame by
    // `consume_one_section` from each section's `-live-` op (resume.py
    // :1381 / `_prepare_next_section` resume.py:1017). RPython's vable
    // bytecodes use that live frame red directly. Pyre's blackhole vable
    // handlers instead address `BlackholeInterpreter::virtualizable_ptr`,
    // so bind that Pyre-only cache from EACH frame's own red register.
    {
        let multi_frame = bh.nextblackholeinterp.is_some();
        let mut current = Some(&mut bh);
        let mut frame_index = 0usize;
        while let Some(frame) = current {
            if majit_metainterp::bh_debug_enabled() {
                let raw_code =
                    pyre_jit_trace::state::raw_code_for_jitcode_index(frame.jitcode.index() as i32);
                let qualname = raw_code
                    .map(|raw| unsafe { &*raw }.qualname.as_str())
                    .unwrap_or("<null>");
                let py_pc = pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                    frame.jitcode.index() as i32,
                    frame.position as i32,
                );
                eprintln!(
                    "[bh-chain] frame={frame_index} jitcode={} qualname={qualname} position={} py_pc={py_pc:?}",
                    frame.jitcode.index(),
                    frame.position,
                );
            }
            // The outer/root frame's standard virtualizable identity comes
            // from `consume_vable_info` above. Preserve it for a single-frame
            // resume. In a multi-frame snapshot that one vable section names
            // only the outer portal frame, so even the innermost blackhole
            // must instead bind its own frame red.
            if (multi_frame || frame.virtualizable_ptr == 0)
                && let Some(jitcode_index) = frame.jitcode.try_index()
            {
                let (frame_reg, _) =
                    pyre_jit_trace::state::portal_red_regs_at(jitcode_index as i32);
                if frame_reg != u16::MAX {
                    if let Some(&frame_ptr) = frame.registers_r.get(frame_reg as usize) {
                        if frame_ptr != 0 {
                            frame.virtualizable_ptr = frame_ptr;
                        }
                    }
                }
            }
            current = frame.nextblackholeinterp.as_deref_mut();
            frame_index += 1;
        }
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!("[blackhole-resume] rd_numb path, chain built, running _run_forever",);
    }

    // #326: no rollback snapshot is taken across `bh.run()`.  The blackhole
    // commits every STORE_FAST / operand push to the virtualizable heap frame
    // as it runs forward, and an abort used to restore that frame to the
    // guard snapshot so the interpreter's re-run from the guard resume PC
    // "applied each side effect exactly once".  That reasoning only covered
    // frame-local state: a `print`, a heap store or a user frame the blackhole
    // already executed has no undo, so rewinding re-applied every one of them
    // (`x = [1,2]; <hot loop>; x.append(3); class C: pass` left `[1,2,3,3]`).
    // The only abort that can reach the blackhole at runtime sits on a Python
    // opcode boundary — `abort` is gated out of blackhole dispatch by
    // `PyJitCode::has_abort_opcode`, leaving `abort_permanent` (a whole
    // unsupported opcode) and the two declined-call rejections (whose call
    // never ran) — so the frame the blackhole leaves behind is a valid resume
    // point and the interpreter continues from it.  This is
    // `convert_and_run_from_pyjitpl`'s shape (`blackhole.py:1799-1821`): each
    // frame resumes at its own current pc, never rewound.

    // blackhole.py:1794-1795 resume_in_blackhole:
    //   current_exc = _prepare_resume_from_failure(guard_opnum, deadframe)
    //   _run_forever(blackholeinterp, current_exc)
    // `_resume_mainloop` prologue (blackhole.py:1614-1618): when the guard
    // failure carries a pending exception (GUARD_NO_EXCEPTION /
    // GUARD_EXCEPTION / GUARD_NOT_FORCED → `cpu.grab_exc_value(deadframe)`),
    // hand it to the resumed frame before running any bytecode so an
    // exception guard unwinds to its `catch_exception` handler instead of
    // resuming the no-exception path. `_run_forever` re-offers the exception
    // to each caller in turn (resume_mainloop returns the unhandled exc, the
    // loop advances to nextblackholeinterp), so walk the chain here.
    if guard_exc != 0 {
        loop {
            if bh.handle_exception_in_frame(guard_exc) {
                // Handler found in this frame; `position` now points at it.
                // Fall through to the run loop to execute the handler.
                break;
            }
            // blackhole.py:1616 no handler here → propagate to the caller.
            let next = bh.nextblackholeinterp.take();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;
            let jitcode_index = bh.jitcode.try_index().map(|v| v as i32);
            let last_opcode_position = bh.last_opcode_position;
            leave_resumed_blackhole_frame(&bh, true);
            release_bh_rd(bh);
            match next {
                Some(caller) => {
                    // `_run_forever` records a node at every frame boundary it
                    // propagates through, so a frame the chain merely passes
                    // over still appears in the traceback.  Only the bottommost
                    // frame was recorded here, which drops the node of every
                    // level between the raise and the outermost frame — for an
                    // inlined callee that is the whole of its presence, since
                    // no interpreter frame ever runs for it.  The helper
                    // resolves the raise coordinate and honours a bare reraise,
                    // as it does for the sibling propagation loop.
                    if !frame_ptr.is_null() {
                        match jitcode_index {
                            Some(jitcode_index) => record_caught_blackhole_traceback(
                                guard_exc,
                                frame_ptr as i64,
                                i64::from(jitcode_index),
                                last_opcode_position as i64,
                            ),
                            None => unsafe {
                                pyre_interpreter::pytraceback::record_application_traceback(
                                    guard_exc as PyObjectRef,
                                    frame_ptr,
                                    (*frame_ptr).last_instr as i64,
                                )
                            },
                        }
                    }
                    bh = *caller;
                }
                None => {
                    // blackhole.py:1629 bottommost frame, unhandled →
                    // raise ExitFrameWithExceptionRef(exc).
                    let err = exit_frame_exception_ref(guard_exc, "guard_exc propagation", || {
                        format!(
                            "jitcode={jitcode_index:?} last_opcode_position={last_opcode_position}"
                        )
                    });
                    if !frame_ptr.is_null() {
                        match jitcode_index {
                            // Every indexed frame recorded during blackhole
                            // propagation goes through the guarded recorder;
                            // `pyopcode.py:147-148 handle_operation_error` is
                            // upstream's single traceback attach point.
                            Some(jitcode_index) => record_caught_blackhole_traceback(
                                err.exc_object as i64,
                                frame_ptr as i64,
                                i64::from(jitcode_index),
                                last_opcode_position as i64,
                            ),
                            None => {
                                m73_lastinstr_audit(
                                    "exit_guard_exc",
                                    jitcode_index,
                                    last_opcode_position as i32,
                                    frame_ptr,
                                );
                                unsafe {
                                    pyre_interpreter::pytraceback::record_application_traceback(
                                        err.exc_object,
                                        frame_ptr,
                                        (*frame_ptr).last_instr as i64,
                                    );
                                }
                            }
                        }
                    }
                    // `guard_exc` is now owned by the typed
                    // `ExitFrameWithExceptionRef` result. A can-raise
                    // residual helper published the same value to both the
                    // backend exception cell and pyre's blackhole adapter;
                    // the guard supplied the former as `guard_exc`, so the
                    // latter must be consumed at this terminal handoff.
                    // RPython has only the deadframe/exception-result
                    // transfer here, not a second TLS owner.
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
                    return BlackholeResult::ExitFrameWithExceptionRef(err);
                }
            }
        }
    }

    // blackhole.py:1752 _run_forever parity.
    loop {
        // `_resume_mainloop` can hand an unhandled exception to a suspended
        // caller by setting `got_exception` below. Propagate that state before
        // dispatching the caller: its current position is the normal post-call
        // continuation and the call result is deliberately garbage on raise.
        // Running even one opcode first consumes that garbage as a successful
        // return and replaces the real exception with a secondary TypeError.
        if !bh.got_exception
            && let Some(args) = bh.run()
        {
            // blackhole.py:1068: raise ContinueRunningNormally(*args)
            //
            // The blackhole reached a merge point with no pending exception
            // (an unhandled raise would have propagated through the exception
            // path above, not reached `run()`'s ContinueRunningNormally).  A
            // residual call that raised AND was caught in-frame
            // (`check_residual_call_exception_after` → `route_to_catch`) cleared
            // only `BH_LAST_EXC_VALUE`; the backend `_store_exception` cells
            // (`store_jit_exception` writes BOTH — see
            // `publish_residual_call_exception`) keep the consumed exception.
            // Re-entering compiled code via this ContinueRunningNormally would
            // then have its first `GUARD_NO_EXCEPTION` read the stale cell as a
            // spurious pending exception and deopt at a coordinate with no
            // handler (the loop header), escaping the original try-block.  Drain
            // the backend cells here so the re-entry starts pristine, mirroring
            // the walker's `execute_raised` drain (`drain_backend_jit_exc`).
            drain_backend_jit_exc();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;

            let mut red_ref: Vec<PyObjectRef> =
                args.red_ref.iter().map(|&v| v as PyObjectRef).collect();
            if red_ref.is_empty() {
                red_ref.push(frame_ptr as PyObjectRef);
            }
            return BlackholeResult::ContinueRunningNormally {
                green_int: args.green_int,
                green_ref: args.green_ref.iter().map(|&v| v as PyObjectRef).collect(),
                green_float: args
                    .green_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
                red_int: args.red_int,
                red_ref,
                red_float: args
                    .red_float
                    .iter()
                    .map(|&v| f64::from_bits(v as u64))
                    .collect(),
            };
        }
        if !bh.got_exception && bh.aborted {
            if nbody_debug {
                eprintln!(
                    "[nbody-debug] blackhole_resume_via_rd_numb failed: bh.aborted position={} last_opcode_position={}",
                    bh.position, bh.last_opcode_position
                );
            }
            release_bh_rd(bh);
            return BlackholeResult::Failed;
        }
        if bh.got_exception {
            // `run()` brackets `exception_last_value` with `push_bh_regs` and
            // pops that bracket on return, so from here the propagating
            // exception lives only in a Rust local — and this block
            // allocates: `record_application_traceback` builds a traceback
            // node, and so does the caller's `handle_exception_in_frame` →
            // `route_to_catch`.  Exception objects do not move
            // (`w_exception_new_empty` uses the stable old gen) but old gen is
            // mark-sweep, so an unrooted one in that window is collectable and
            // its block reusable — which is what hands
            // `exit_frame_exception_ref` a mapped object of the wrong class.
            // `blackhole.py:1752 _run_forever` keeps every interp in the chain
            // — and with it `exception_last_value` — transitively traced for
            // its whole life; root the value across the equivalent window.
            let _exc_roots = pyre_object::gc_roots::push_roots();
            let exc_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(bh.exception_last_value as PyObjectRef);
            let exc_value = bh.exception_last_value;
            let next = bh.nextblackholeinterp.take();
            let frame_ptr = bh.virtualizable_ptr as *mut PyFrame;
            let jitcode_index = bh.jitcode.try_index().map(|v| v as i32);
            let last_opcode_position = bh.last_opcode_position;
            let bh_entry_position = bh.entry_position;
            let bh_opcode_at = bh.jitcode.code.get(last_opcode_position).copied();
            let bh_jitcode_name = bh.jitcode.name().to_string();
            // For `BC_RAISE` (`raise/r`) the following byte is the register the
            // exception was read out of, so record which slot of which bank
            // produced the value.
            let bh_raise_reg = bh.jitcode.code.get(last_opcode_position + 1).copied();
            let bh_regs_r_len = bh.registers_r.len();
            // A copy, not a probe: a ref register may hold a raw non-object
            // word (a force token, an erased unboxed buffer), so anything that
            // dereferences these belongs on the failure path only.
            //
            // The copy has to be taken here because `release_bh_rd` below ends
            // the borrow before the diagnostic closure runs, but it is only
            // ever read when `exit_frame_exception_ref` rejects the value, so
            // the same screen decides whether to pay for it. A move between
            // here and the closure does not change the answer: the class of the
            // propagating value is what is being screened.
            let bh_regs_r = if unsafe {
                pyre_object::interp_exceptions::w_exception_kind_checked(exc_value as PyObjectRef)
            }
            .is_none()
            {
                bh.registers_r.clone()
            } else {
                Vec::new()
            };
            let bh_code_window: Vec<u8> = bh
                .jitcode
                .code
                .get(last_opcode_position.saturating_sub(10)..=last_opcode_position + 1)
                .unwrap_or(&[])
                .to_vec();
            let keeps_existing_traceback = bh_opcode_at
                .is_some_and(|opcode| opcode == majit_metainterp::jitcode::insns::BC_RERAISE)
                || jitcode_index.is_some_and(|index| {
                    pyre_jit_trace::state::jitcode_pc_raise_keeps_existing_traceback(
                        index,
                        last_opcode_position as i32,
                    )
                });
            // blackhole.py _run_forever propagates an unhandled exception one
            // BlackholeInterpreter frame at a time. Record each exiting
            // frame before advancing to nextblackholeinterp, matching
            // pytraceback.py record_application_traceback at every Python
            // frame boundary. A forwarding raise preserves the existing chain.
            if !keeps_existing_traceback && !frame_ptr.is_null() {
                if let Some(jitcode_index) = jitcode_index {
                    record_caught_blackhole_traceback(
                        exc_value,
                        frame_ptr as i64,
                        i64::from(jitcode_index),
                        last_opcode_position as i64,
                    );
                } else {
                    unsafe {
                        pyre_interpreter::pytraceback::record_application_traceback(
                            exc_value as PyObjectRef,
                            frame_ptr,
                            (*frame_ptr).last_instr as i64,
                        );
                    }
                }
            }
            leave_resumed_blackhole_frame(&bh, true);
            release_bh_rd(bh);
            // Re-read through the pin: the records above are allocation
            // points, so the pinned slot — not the stale local — is the live
            // value from here on.
            let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
            let Some(mut caller_bh) = next.map(|b| *b) else {
                let err =
                    exit_frame_exception_ref(exc_value, "blackhole exception propagation", || {
                        format!(
                            "jitcode={jitcode_index:?} name={bh_jitcode_name} \
                             entry_position={bh_entry_position} \
                             last_opcode_position={last_opcode_position} opcode={:?} \
                             operand_reg={bh_raise_reg:?} registers_r.len={bh_regs_r_len} \
                             regs_holding_exception={:?} \
                             code[-10..]={bh_code_window:?} \
                             keeps_existing_traceback={keeps_existing_traceback} \
                             guard_exc={guard_exc:x} py_pc={:?} entry_py_pc={:?} \
                             deadframe_types={deadframe_types:?} deadframe={deadframe:x?} \
                             registers_r={bh_regs_r:x?}",
                            bh_opcode_at,
                            // Which ref registers hold a real
                            // `W_BaseException`: a hit in a slot other than
                            // `operand_reg` means the operand index is wrong,
                            // no hit anywhere means the bank never received
                            // the exception.
                            bh_regs_r
                                .iter()
                                .enumerate()
                                .filter(|(_, v)| unsafe {
                                    pyre_object::interp_exceptions::w_exception_kind_checked(
                                        **v as PyObjectRef,
                                    )
                                    .is_some()
                                })
                                .map(|(i, _)| i)
                                .collect::<Vec<_>>(),
                            jitcode_index.and_then(|index| {
                                pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                                    index,
                                    last_opcode_position as i32,
                                )
                            }),
                            jitcode_index.and_then(|index| {
                                pyre_jit_trace::py_coord::containing_py_pc_for_jitcode_pc_public(
                                    index,
                                    bh_entry_position as i32,
                                )
                            }),
                        )
                    });
                // `attach_tb` deliberately stays true on the way out.  It
                // suppresses the traceback record of the re-raising frame
                // only, and this frame's decision was already applied above by
                // skipping its own record; `handle_exception` clears the flag
                // for that same reason once a frame is done with it.  Clearing
                // it here hands the suppression one frame further instead: the
                // compiled frame never runs `handle_exception` itself, so the
                // interpreter caller is the first frame to read the flag and
                // would drop its own node.
                //
                // A residual helper can publish a raise to both exception
                // channels.  Blackhole propagation has consumed its own
                // channel into `ExitFrameWithExceptionRef`; keep the backend
                // cell in sync before the outer interpreter receives it.
                drain_backend_jit_exc();
                return BlackholeResult::ExitFrameWithExceptionRef(err);
            };
            caller_bh.last_opcode_position = caller_bh.position;
            if caller_bh.handle_exception_in_frame(exc_value) {
                bh = caller_bh;
                continue;
            }
            // `handle_exception_in_frame` records a traceback node on the way
            // through, so take the pinned value again for the store.
            caller_bh.exception_last_value =
                pyre_object::gc_roots::shadow_stack_get(exc_slot) as i64;
            caller_bh.got_exception = true;
            bh = caller_bh;
            continue;
        }

        // blackhole.py:1632-1644: pass return value to caller by _return_type.
        use majit_metainterp::blackhole::BhReturnType;
        let rt = bh.return_type;
        let next = bh.nextblackholeinterp.take();
        let caller = next.map(|b| *b);
        if caller.is_none() {
            // blackhole.py:1664-1677 _done_with_this_frame
            let result = match rt {
                BhReturnType::Void => BlackholeResult::DoneWithThisFrameVoid,
                BhReturnType::Int => BlackholeResult::DoneWithThisFrameInt(bh.get_tmpreg_i()),
                BhReturnType::Ref => {
                    BlackholeResult::DoneWithThisFrameRef(bh.get_tmpreg_r() as PyObjectRef)
                }
                BhReturnType::Float => BlackholeResult::DoneWithThisFrameFloat(f64::from_bits(
                    bh.get_tmpreg_f() as u64,
                )),
            };
            return result;
        }
        let mut caller_bh = caller.unwrap();
        // blackhole.py:1637-1644: dispatch by _return_type
        match rt {
            BhReturnType::Int => caller_bh.setup_return_value_i(bh.get_tmpreg_i()),
            BhReturnType::Ref => caller_bh.setup_return_value_r(bh.get_tmpreg_r()),
            BhReturnType::Float => caller_bh.setup_return_value_f(bh.get_tmpreg_f()),
            BhReturnType::Void => {}
        }
        leave_resumed_blackhole_frame(&bh, false);
        release_bh_rd(bh);
        bh = caller_bh;
    }
}

/// warmspot.py:961-1007 handle_jitexception parity.
///
/// RPython captures result_kind in closure (warmspot.py:913). For pyre,
/// portal result_type == REF (warmspot.py:449), so ALL CALL_ASSEMBLER
/// ops use _R. The result is always a Ref (PyObjectRef).
fn handle_blackhole_result(bh_result: BlackholeResult, _green_key: u64) -> Option<i64> {
    match bh_result {
        // warmspot.py:985-987: DoneWithThisFrameVoid → return None
        BlackholeResult::DoneWithThisFrameVoid => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameVoid");
            }
            Some(0)
        }
        // warmspot.py:988-990: DoneWithThisFrameInt → box to Ref.
        // Portal result_type == REF, so blackhole should normally raise
        // DoneWithThisFrameRef. This path handles edge cases.
        BlackholeResult::DoneWithThisFrameInt(v) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] DoneWithThisFrameInt({}) → box to Ref",
                    v
                );
            }
            Some(w_int_new(v) as i64)
        }
        // warmspot.py:991-993: DoneWithThisFrameRef → return e.result
        BlackholeResult::DoneWithThisFrameRef(r) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameRef({:?})", r);
            }
            Some(r as i64)
        }
        // warmspot.py:994-996: DoneWithThisFrameFloat → return e.result
        BlackholeResult::DoneWithThisFrameFloat(f) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] DoneWithThisFrameFloat({})", f);
            }
            Some(f.to_bits() as i64)
        }
        // warmspot.py:998-1005: ExitFrameWithExceptionRef → raise value.
        BlackholeResult::ExitFrameWithExceptionRef(mut err) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] ExitFrameWithExceptionRef → raise");
            }
            let exc_obj = err.to_exc_object();
            if exc_obj != pyre_object::PY_NULL {
                // Symmetric with the `ContinueRunningNormally` arm's
                // `portal_runner_result` error fall-through below and with
                // `lib.rs::jit_exc_raise` — every backend's blackhole resume
                // publishes the pending exception, not just cranelift.
                store_jit_exception(exc_obj as i64);
            }
            Some(0) // garbage return — GUARD_NO_EXCEPTION will fire
        }
        // warmspot.py:970-983: ContinueRunningNormally → portal_ptr(*args).
        BlackholeResult::ContinueRunningNormally {
            green_int,
            green_ref,
            green_float,
            red_int,
            red_ref,
            red_float,
        } => {
            // warmspot.py:972-975: portalfunc_ARGS extraction.
            // Build merged arg lists: all_i = gi + ri, all_r = gr + rr, all_f = gf + rf.
            // warmstate.py:41 unspecialize_value: Ref→GCREF(i64), Float→FLOATSTORAGE(i64).
            let mut all_i = green_int;
            all_i.extend(&red_int);
            let mut all_r: Vec<i64> = green_ref.iter().map(|r| *r as i64).collect();
            all_r.extend(red_ref.iter().map(|r| *r as i64));
            let mut all_f: Vec<i64> = green_float.iter().map(|f| f.to_bits() as i64).collect();
            all_f.extend(red_float.iter().map(|f| f.to_bits() as i64));
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[blackhole-resume] ContinueRunningNormally → portal_ptr(all_i={:?}, all_r=[{:#x?}])",
                    all_i, all_r,
                );
            }
            // warmspot.py:976-1005: portal_ptr(*args), and if it raises a
            // regular exception propagate it like ExitFrameWithExceptionRef
            // instead of collapsing it to a null Ref.
            let next_instr = all_i.first().copied().unwrap_or(0) as usize;
            let pycode = all_r.first().copied().unwrap_or(0) as PyObjectRef;
            let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
            let ec =
                all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
            if frame_ptr.is_null() {
                return Some(pyre_object::PY_NULL as i64);
            }
            let frame = unsafe { &mut *frame_ptr };
            // The red frame is the activation identity.  PyPy forwards the
            // CRN arguments to `portal_ptr` without mutating that frame; pyre's
            // frame-only portal entry likewise reads the pycode / execution
            // context already stored on the callee frame.  Do not overwrite
            // them from a possibly stale green snapshot.
            if majit_metainterp::majit_log_enabled()
                && ((!pycode.is_null() && frame.pycode != pycode as *const ())
                    || (!ec.is_null() && frame.execution_context != ec))
            {
                eprintln!(
                    "[blackhole-resume] CALL_ASSEMBLER CRN/frame identity mismatch: green_pycode={pycode:p} frame_pycode={:p} red_ec={ec:p} frame_ec={:p}",
                    frame.pycode, frame.execution_context,
                );
            }
            frame.set_last_instr_from_next_instr(next_instr);
            // The blackhole wrote the failing guard's recorded operand depth
            // into the frame; resuming at the merge-point `next_instr` (a
            // different pc) would carry that over-count and overflow the frame
            // at its peak stack use.  Re-derive the depth from the resume pc —
            // the CALL_ASSEMBLER-path mirror of the eval.rs CRN handoff.
            crate::eval::correct_resume_vsd(frame, next_instr);
            match crate::eval::portal_runner_result(frame) {
                Ok(result) => Some(result as i64),
                Err(mut err) => {
                    let exc_obj = err.to_exc_object();
                    if exc_obj != pyre_object::PY_NULL {
                        majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                            .with(|c| c.set(exc_obj as i64));
                        store_jit_exception(exc_obj as i64);
                    }
                    Some(0)
                }
            }
        }
        BlackholeResult::Failed => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[blackhole-resume] Failed");
            }
            None
        }
    }
}

/// Derive the (`green_key`, `trace_id`, `fail_index`) bridge-source identity
/// strictly from the failing guard's descr Arc.
///
/// `pyjitpl.py:2914 handle_guard_failure(self, resumedescr, deadframe)`
/// reads identity from `resumedescr` directly: `resumedescr.rd_loop_token
/// .loop_token_wref()` (line 2897) yields the owning `JitCellToken`,
/// `resumedescr.get_resumestorage()` (line 2893) yields the `ResumeGuardDescr`
/// carrying the per-trace `fail_index`, and `record_loop_or_bridge`
/// (`compile.py:183-185`) stamps `trace_id` onto each `ResumeDescr`.
/// Pyre's `descr_owning_jct(arc).green_key`, `arc.fail_index_per_trace()`
/// and `arc.trace_id()` are the line-by-line equivalents — together they
/// constitute the canonical bridge-source identity.
///
/// Returns `None` when the owning loop's `JitCellToken` weakref is dead
/// (the loop was evicted by the memory manager): RPython raises
/// `compile.giveup()` from `_trace_and_compile_from_bridge`
/// (`compile.py:725-729`) in the same case, falling through to
/// `resume_in_blackhole`.  Pyre's caller signals the same intent by
/// returning `false` to drop into blackhole resume.
// dont_look_inside: bridge-compile identity machinery the tracer must not enter.
#[majit_macros::dont_look_inside]
pub(crate) fn bridge_source_identity_from_descr(
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
) -> Option<(u64, u64, u32)> {
    let descr_fd = descr_arc.as_fail_descr()?;
    let green_key = majit_backend::descr_owning_jct(descr_fd)?.green_key();
    let trace_id = descr_fd.trace_id();
    let fail_index = descr_fd.fail_index_per_trace();
    Some((green_key, trace_id, fail_index))
}

/// Outcome of `trace_and_compile_from_bridge`.
///
/// `pyjitpl.py:2914 handle_guard_failure` never returns — it raises
/// `ContinueRunningNormally` (bridge attached, resume in compiled code),
/// switches to the blackhole, or raises `DoneWithThisFrame` when
/// `interpret()` runs the resumed frames forward to a `Finish`.  Pyre
/// reifies those three exits so the caller can act on each.
pub enum BridgeResolution {
    /// Bridge attached — caller re-enters compiled code
    /// (`ContinueRunningNormally`).  The legacy `true` return.
    CompiledContinue,
    /// Resume the region in the blackhole interpreter
    /// (`SwitchToBlackhole`).  The legacy `false` return.
    ResumeBlackhole,
    /// The single-frame bridge walk ran the resumed frame forward to a
    /// `Finish` (`Terminate`) and captured its concrete return value —
    /// `pyjitpl.py` `interpret()` raising `DoneWithThisFrame` from the
    /// post-walk state.  The caller returns this value directly instead of
    /// rewinding the live frame and re-running the region (#177).
    Finished(pyre_jit_trace::state::ConcreteValue),
    /// The single-frame bridge walk ended in an uncaught raise; hand its
    /// concrete exception to the guard's portal as
    /// `ExitFrameWithExceptionRef` (jitexc.py:44).
    FinishedException(pyre_jit_trace::state::ConcreteValue),
}

/// Map the legacy bool bridge outcome (`true` = continue in compiled code,
/// `false` = resume in the blackhole) onto [`BridgeResolution`].
fn bridge_resolution_from_bool(compiled_continue: bool) -> BridgeResolution {
    if compiled_continue {
        BridgeResolution::CompiledContinue
    } else {
        BridgeResolution::ResumeBlackhole
    }
}

/// compile.py:714 (_trace_and_compile_from_bridge):
/// Called when a guard failure reaches the trace_eagerness threshold.
/// Traces the alternative path from the guard failure point and compiles
/// a bridge.
///
/// pyjitpl.py:2914 handle_guard_failure:
///   initialize_state_from_guard_failure(resumedescr, deadframe)
///   prepare_resume_from_failure(deadframe, inputargs, resumedescr, excdata)
///   self.interpret()
///
/// The tracing loop mirrors pyjitpl.py interpret(): execute bytecodes
/// from the guard failure PC until a Finish (return) or CloseLoop
/// (back-edge to loop header) is reached.
/// compile.py:714 _trace_and_compile_from_bridge parity.
///
/// Returns [`BridgeResolution`]: `CompiledContinue` when the bridge was
/// compiled and attached, `ResumeBlackhole` on a trace abort / start
/// failure so the caller falls through to resume_in_blackhole (RPython
/// pyjitpl.py:2906-2907 SwitchToBlackhole → run_blackhole_interp_to_cancel_tracing),
/// or `Finished(cv)` when a single-frame walk ran forward to a `Finish`
/// (`interpret()` raising `DoneWithThisFrame`).
///
/// `allow_finish_direct_return` gates the `Finished` shortcut: only the
/// general guard path (`eval::handle_fail`) can consume a concrete result,
/// so the CALL_ASSEMBLER callback passes `false` and always takes the
/// legacy rewind/blackhole path.
// dont_look_inside: bridge-compile machinery the tracer must not enter.
#[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
#[majit_macros::dont_look_inside]
pub fn trace_and_compile_from_bridge(
    // pyjitpl.py:2914 `handle_guard_failure(self, resumedescr, deadframe)`
    // threads `resumedescr` (the descr) as the canonical identity source
    // through the entire bridge tracer.  Pyre's backend FailDescr Arc
    // plays the same role: `descr_owning_jct(arc).green_key` (mirroring
    // `pyjitpl.py:2897 resumedescr.rd_loop_token.loop_token_wref()`),
    // `arc.fail_index_per_trace()` (mirroring `compile.py:854
    // ResumeGuardDescr._attrs_`), and `arc.trace_id()` (mirroring the
    // `compile.py:183-185 record_loop_or_bridge` stamp) are the line-by-
    // line readers.  Both production callers — the general guard path
    // (eval.rs `handle_fail`) and the CALL_ASSEMBLER bridge entry
    // (call_jit.rs `jit_ca_handle_guard_failure`) — have an Arc available
    // before reaching this function (both backends recover it from
    // `FailDescrCell` thin pointers via
    // `Backend::fail_descr_arc_from_addr`), so this function takes only
    // the descr Arc and derives `(green_key, trace_id, fail_index)`
    // itself.
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    frame: &mut PyFrame,
    raw_values: &[i64],
    exit_layout: &majit_metainterp::CompiledExitLayout,
    // `cpu.grab_exc_value(deadframe)` (llmodel.py:240): the pending
    // exception this guard failure carries, or 0. Threaded so the bridge
    // tracer can decline a pending-exception resume at a non-exception
    // guard (the `pending_exc` decline below).
    guard_exc: i64,
    // Whether the caller can consume a `Finished(cv)` direct return (the
    // general guard path can; the CALL_ASSEMBLER callback, which returns a
    // bare bool to native code, cannot).  Gates the bridge `Terminate`
    // no-replay shortcut so a committed store journal never strands into a
    // blackhole re-run on a path that would ignore the concrete result.
    allow_finish_direct_return: bool,
) -> BridgeResolution {
    /// Publish the live callee frame's address to the handshake cells the CA
    /// slow path's back-to-back blackhole hook reads
    /// (`jit_blackhole_resume_from_guard`). Always re-read through
    /// `FrameRoot::frame`, i.e. off the shadow stack, and always at a return
    /// site: the walk that decides these flags runs before the bridge is
    /// optimised and compiled, and that work allocates, so an address copied
    /// while the walk was still running can name a frame the collector has
    /// since moved. The blackhole compares the adopted cell against the
    /// deadframe's slot 0, which IS forwarded across the same region, so a
    /// stale copy here misses the handshake and lets the guard-state resume
    /// re-run a region the walk already executed.
    fn stamp_ca_walk_frames(
        bridge_frame_root: &mut FrameRoot,
        stamp_resume_frame: bool,
        stamp_adopted_frame: bool,
    ) {
        if !stamp_resume_frame && !stamp_adopted_frame {
            return;
        }
        let frame = bridge_frame_root.frame() as *mut PyFrame as usize;
        if stamp_resume_frame {
            CA_WALK_RESUME_FRAME.with(|c| c.set(frame));
        }
        if stamp_adopted_frame {
            CA_WALK_ADOPTED_FRAME.with(|c| c.set(frame));
        }
    }

    {
        let (driver, _) = crate::eval::driver_pair();
        driver.clear_last_compiled_artifact_invalidation_flag();
    }

    use crate::eval::build_jit_state;
    use crate::jit::state::PyreEnv;

    // compile.py:950-953 `ResumeGuardForcedDescr.handle_fail`: "Failures of
    // a GUARD_NOT_FORCED are never compiled, but always just blackholed."
    // A bridge walked from one force flavor (pure force, live call result)
    // would also be entered for forced-and-raised failures whose call result
    // is NULL — its result-class guard then dereferences NULL.
    if descr_arc.is_guard_forced() {
        return BridgeResolution::ResumeBlackhole;
    }
    // compile.py:702-703 `must_compile() and not stack_almost_full()`: this is
    // pyre's own guard-failure entry, so the diagnostic that suppresses bridge
    // recording has to be read here as well as in the jitdriver's paths.
    if majit_metainterp::no_bridge_enabled() {
        return BridgeResolution::ResumeBlackhole;
    }
    let Some((green_key, trace_id, fail_index)) = bridge_source_identity_from_descr(descr_arc)
    else {
        // compile.py:725-729 `_trace_and_compile_from_bridge` raises
        // `compile.giveup()` when `loop_token` is None (memmgr-evicted).
        // Pyre signals the same outcome by returning `false`, dropping
        // the caller into `resume_in_blackhole` (pyjitpl.py:711).
        return BridgeResolution::ResumeBlackhole;
    };

    let info = {
        let (_, info) = crate::eval::driver_pair();
        info
    };

    // pyjitpl.py:2914-2935 handle_guard_failure parity:
    // RPython creates a fresh MetaInterp and calls
    // initialize_state_from_guard_failure(resumedescr, deadframe)
    // which internally calls rebuild_from_resumedata (resume.py:1042).
    // This restores the complete frame stack INSIDE the bridge function.
    let meta = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().get_compiled_meta(green_key).cloned()
    };
    let mut jit_state_local = build_jit_state(frame, info);
    // `num_resume_frames > 1` marks a multi-frame (inlined-callee) guard:
    // the guard fired inside a callee inlined into the trace, so the resume
    // pc is the INNERMOST frame's bytecode pc, which does not address the
    // live (outer) frame `eval_loop_jit` runs. Such a resume cannot be
    // completed by interpreting the live frame forward — see the
    // blackhole routing at the handoff below.
    let decoded_resume = meta.as_ref().and_then(|meta| {
        crate::eval::decode_and_restore_guard_failure(
            &mut jit_state_local,
            meta,
            raw_values,
            exit_layout,
        )
    });
    let Some((_, resume_pc, num_resume_frames, resume_coords)) = decoded_resume else {
        return BridgeResolution::ResumeBlackhole;
    };
    let is_multiframe_resume = num_resume_frames > 1;
    // `resume_pc` addresses the INNERMOST resumed section, so for a
    // multi-frame guard it is a pc in an inlined callee's own code object.
    // It only addresses THIS frame while that section belongs to this
    // frame's code — the same condition the paired valuestackdepth
    // correction already applies before handing the coordinate back.
    // The routing below is meant to keep such a resume off the live frame,
    // but every arm after this point can still return `ResumeBlackhole`, and
    // the blackhole itself refuses a jitcode carrying an abort opcode or a pc
    // whose liveness it cannot decode. On that path the interpreter resumes
    // this frame, so a foreign pc left in `last_instr` would step a code
    // object the frame does not run. Keep the vable-derived position instead.
    let frame_code = jit_state_local.pycode_as_usize();
    let foreign_innermost =
        is_multiframe_resume && resume_coords.last().map(|&(code, _)| code) != Some(frame_code);
    if foreign_innermost {
        pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
            frame_code,
            "Resume::ForeignInnermostLastInstr",
        );
    } else {
        frame.set_last_instr_from_next_instr(resume_pc);
    }
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame, info);

    // NOTE: guard resume_pc pointing to LOAD_CONST + RETURN_VALUE does NOT
    // mean the guard is a loop-exit guard. It means the blackhole resume
    // path leads to function return. RPython handles this correctly via
    // blackhole resume → interpreter runs remaining code → natural return.
    // Direct FINISH bridges are WRONG here — they skip the remaining loop
    // body that the blackhole should execute.
    // RPython rebuild_from_resumedata (pyjitpl.py:2901,3400)
    // restores the complete frame stack before bridge tracing.
    // Bridge tracing sees the full frame layout — no truncation.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][bridge-trace] start key={} trace={} fail={} resume_pc={}",
            green_key, trace_id, fail_index, resume_pc
        );
    }

    // compile.py:714: start_retrace_from_guard + set bridge_info.
    let started = {
        let (driver, _) = crate::eval::driver_pair();
        driver.start_bridge_tracing(descr_arc, &mut jit_state, &env, raw_values, resume_pc)
    };
    if !started {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] start_bridge_tracing failed key={} trace={} fail={}",
                green_key, trace_id, fail_index
            );
        }
        return BridgeResolution::ResumeBlackhole;
    }
    // `_prepare_exception_resumption` (pyjitpl.py:3101) +
    // `prepare_resume_from_failure` (pyjitpl.py:3132) parity: for exception
    // guard bridges (GUARD_EXCEPTION / GUARD_NO_EXCEPTION), SAVE_EXC_CLASS +
    // SAVE_EXCEPTION at trace start, then RESTORE_EXCEPTION before the guard.
    // The walker owns that whole sequence — it emits it at the bridge-entry
    // frame state, which is where the guard can capture resume data — so this
    // site only consumes the flag.
    let last_bridge_is_exception_guard = {
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard
    };
    if last_bridge_is_exception_guard {
        let (driver, _) = crate::eval::driver_pair();
        driver.last_bridge_is_exception_guard = false;
    }

    // A pending exception at a NON-exception guard means a may-force
    // residual call both raised and forced the frame, so GUARD_NOT_FORCED
    // failed before GUARD_NO_EXCEPTION ran. That guard's resume_pc is the
    // no-exception semantic fallthrough (the next opcode, e.g.
    // RETURN_VALUE), so a normal-path bridge walk would trace the return
    // of the NULL call result — emitting `Finish(NONE)` with no
    // return-value box — and run the post-call residual ops concretely on
    // a NULL operand. Decline the bridge and resume in the blackhole,
    // which propagates the pending exception to its `catch_exception`
    // handler exactly once. Skipping the walk also prevents the walk's
    // concrete side effects from double-applying the handler's mutations
    // against the post-blackhole replay.
    //
    // GUARD_NOT_FORCED is not an exception guard, so it does not stash the
    // raised value into `jf_guard_exc` (`guard_exc` is 0 here); the live
    // signal is the backend `pos_exception` cell the compiled
    // GUARD_NO_EXCEPTION reads. A stale cell only over-declines (the
    // blackhole resume with `guard_exc == 0` runs the no-exception
    // continuation, identical to the prior panic-then-fallback path), so
    // this never changes a result — only whether a bridge is attached.
    let pending_exc = guard_exc != 0 || {
        #[cfg(feature = "cranelift")]
        {
            majit_backend_cranelift::jit_exc_class_raw() != 0
        }
        #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
        {
            majit_backend_dynasm::jit_exc_class_raw() != 0
        }
        #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
        {
            false
        }
    };
    // A pending exception at an EXCEPTION guard (GUARD_NO_EXCEPTION /
    // GUARD_EXCEPTION) whose raising op sits inside an in-frame try is the
    // bridge-side analogue of the same fallthrough-not-catch resume gap.
    // pyre encodes the guard's resume_pc as the no-exception semantic
    // fallthrough (the next opcode after the call), NOT the `except`
    // handler.  The blackhole compensates at runtime — `resume_in_blackhole`
    // hands the pending exception to `handle_exception_in_frame`, which
    // `find_catch_before_resume_live`-routes to the handler and runs it
    // (e.g. `return -1`).  The bridge tracer has no such routing: it walks
    // from the fallthrough resume_pc and records the RETURN of the (NULL)
    // raised-call result — `Finish(<unbound box>)` — so the compiled bridge
    // hands a NULL up to the caller ("call failed" / a corrupted kept value
    // / a fault).  Detect the caught-in-frame case with the exact mechanism
    // the interpreter's `handle_exception` uses — an exception-table lookup
    // at the raising op (`last_instr` == resume_pc-1) — and decline so the
    // always-correct blackhole resume handles it.  (Routing the bridge walk
    // to the handler is the orthodox follow-up, gated on the in-try
    // residual-call resume-PC epic; declining is correctness-first.)
    //
    // The escaping case — an exception guard whose raising op is NOT caught in
    // this frame, so the exception unwinds OUT to the caller — has the same
    // fallthrough-not-catch resume gap: the guard's resume_pc is the
    // no-exception fallthrough, so the bridge walk records the RETURN of the
    // NULL raised-call result and the compiled bridge hands a NULL up to the
    // caller (the "call failed" crash).  The
    // `emit_exception_bridge_prologue` GUARD_EXCEPTION path that would trace
    // the propagate-out continuation needs resume-data replay pyre does not yet
    // have (its synthetic guard carries no rd_resume_position), so decline the
    // escaping case too and let the blackhole propagate the exception out of
    // the frame to the caller's handler.  Compiling a real raising bridge
    // (`Finish(exc, exit_frame_with_exception_descr_ref)`) is the orthodox
    // follow-up, gated on the same exception-edge bridge epic.
    /// The exception-table byte offset a `next_instr`-style resume coordinate
    /// maps to, mirroring the live-frame form `frame.last_instr * 2` (where
    /// `last_instr` is `next_instr - 1`).
    fn exc_table_offset(py_pc: usize) -> u32 {
        (py_pc.saturating_sub(1) as u32) * 2
    }

    /// Index of the resumed frame that catches, searched INNERMOST-first by
    /// asking each frame's own exception table about its own resume pc.
    /// `coords` is outermost-first, so `Some(0)` means the raise unwinds clear
    /// out of every inlined callee into the live frame; `None` means it
    /// escapes them all.
    fn resumed_catch_level(coords: &[(usize, usize)]) -> Option<usize> {
        for (level, &(w_code, py_pc)) in coords.iter().enumerate().rev() {
            if w_code == 0 {
                return None;
            }
            let raw = unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            };
            if raw.is_null() {
                return None;
            }
            let code = unsafe { &*raw };
            if pyre_interpreter::pycode::lookup_exceptiontable(
                &code.exceptiontable,
                exc_table_offset(py_pc),
            )
            .is_some()
            {
                return Some(level);
            }
        }
        None
    }

    let caught_in_frame = pending_exc && last_bridge_is_exception_guard && {
        if is_multiframe_resume {
            // `resume_pc` (and thus `frame.last_instr`, set above) addresses the
            // INNERMOST inlined frame while `code` is the live OUTER frame, so
            // the single-frame lookup below would consult the wrong code object.
            // Ask each resumed frame about its OWN pc instead. Only a raise that
            // unwinds clear out of every inlined callee into the live frame's
            // own handler is routable: that unwind discards the callee frames
            // outright, so there is no inlined framestack left to rebuild. A
            // catch inside a callee still needs the carrier subwalk and keeps
            // declining below, as does a raise that escapes the whole resume.
            resumed_catch_level(&resume_coords) == Some(0)
        } else {
            let off = if frame.last_instr < 0 {
                0u32
            } else {
                (frame.last_instr as u32) * 2
            };
            pyre_interpreter::pycode::lookup_exceptiontable(&code.exceptiontable, off).is_some()
        }
    };
    // Exception-edge bridge: route a resume whose handler lives in the LIVE
    // frame to that `except` handler (walker `find_catch_before_resume_live`)
    // instead of declining.  `caught_in_frame` already restricts the
    // multi-frame case to the unwind-to-live-frame shape; the escaping case
    // (uncaught) and a catch inside an inlined callee still decline below —
    // those are separate slices (raising-bridge Finish(exc) / carrier subwalk).
    // Only when the outermost resume section really IS the live frame, which is
    // what lets the unwind land in a frame that already exists. Re-pointing the
    // live frame to coordinates belonging to some other code object would
    // resume the walk against the wrong bytecode.
    let unwind_to_live_frame = is_multiframe_resume
        && caught_in_frame
        && resume_coords
            .first()
            .is_some_and(|&(outer_w_code, _)| outer_w_code == frame.pycode as usize);
    let route_exc_edge = caught_in_frame && (!is_multiframe_resume || unwind_to_live_frame);
    // The levels this route is about to throw away: `resume_coords` minus the
    // live frame, which keeps its own recorder at the handler entry.  Published
    // unconditionally on the routing path — an empty slice for the single-frame
    // shape — so a previous bridge's levels cannot survive into this walk.
    pyre_jit_trace::jitcode_dispatch::set_exc_edge_discarded_levels(if route_exc_edge {
        resume_coords.get(1..).unwrap_or(&[])
    } else {
        &[]
    });
    if route_exc_edge && guard_exc != 0 {
        // Publish the grabbed exception (`cpu.grab_exc_value` result) so the
        // walker's `seed_standing_exception_for_walk` threads it into
        // `sym.last_exc_box`, which the handler's `last_exc_value/>r` reads.
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(guard_exc));
    } else if !pending_exc {
        // No standing exception at this bridge's source guard (e.g. a
        // GUARD_NOT_FORCED that fails every iteration because the may-force
        // residual forced the frame).  A prior residual-call raise may have
        // left a stale non-null value in `BH_LAST_EXC_VALUE`; the walker's
        // exc-edge routing seeds `sym.last_exc_value` from it and would route
        // this exception-free bridge into the `except` handler (recording the
        // `v = -1` handler body as the bridge body → the handler path runs
        // every iteration).  Clear it so the walk resumes on the real
        // no-exception continuation.  The decline path below (`pending_exc &&
        // !route_exc_edge`) keeps the value for the blackhole.
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    }
    if pending_exc && !route_exc_edge {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] decline (pending exc, caught_in_frame={caught_in_frame}) key={} trace={} fail={} resume_pc={}",
                green_key, trace_id, fail_index, resume_pc
            );
        }
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        // This pre-walk decline returns before the `trace_bytecode` walk, so the
        // walk-time `TraceAction::Abort` recorder below never runs for it. Record
        // the guard here so `must_compile_with_values` does not re-enter this
        // structurally-undecidable bridge every ~`trace_eagerness` failures. Only
        // an exception guard is recorded: its `pending_exc` is guard-invariant, so
        // the decline is deterministic while the exc-edge bridge is off. A
        // `GUARD_NOT_FORCED` (even one carrying pending fields) has a
        // runtime-conditional `pending_exc` and can still bridge on a later
        // non-raising failure via the pending-field prologue, so blacklisting it
        // after one raising failure would defeat that; leave it unrecorded.
        if last_bridge_is_exception_guard {
            driver
                .meta_interp_mut()
                .record_declined_bridge_guard(descr_arc);
        }
        return BridgeResolution::ResumeBlackhole;
    }

    // The unwind discards every inlined callee, so the state the walk must
    // start from is the LIVE frame at its own call site — not the innermost
    // callee coordinate `decode_and_restore_guard_failure` installed for the
    // blackhole (its `resume_pc` override plus the matching innermost
    // `depth_based_vsd_for_wcode` correction).  Re-point the frame to the
    // outermost section before anything below reads it: `PyreJitState` holds
    // only the frame pointer, so the `jit_state` built above picks this up,
    // and `snapshot_for_tracing` then captures the coordinate the post-bridge
    // interpreter should resume at.  Only reachable where this function used
    // to decline outright, so no existing resume changes shape.
    let resume_pc = match resume_coords.first() {
        Some(&(outer_w_code, outer_py_pc)) if unwind_to_live_frame => {
            frame.set_last_instr_from_next_instr(outer_py_pc);
            if let Some(vsd) =
                pyre_jit_trace::state::depth_based_vsd_for_wcode(outer_w_code, outer_py_pc)
            {
                jit_state.set_valuestackdepth(vsd);
                jit_state.clear_stack_above(vsd);
            }
            outer_py_pc
        }
        _ => resume_pc,
    };

    // The live frame is a virtualizable GC object held by raw bridge-trace
    // locals while retracing can collect. RPython keeps the virtualizable
    // object GC-visible during retracing
    // (`rpython/jit/metainterp/pyjitpl.py:2839-2841`); pyre roots the frame
    // word so PyFrame's custom trace can forward `locals_cells_stack_w`.
    let mut bridge_frame_root = FrameRoot::new(frame);

    // pyjitpl.py:2841 interpret(): after start_retrace_from_guard, RPython
    // runs a single interpret() over the resumed frame state until the
    // bridge closes or aborts. `trace_bytecode` is the pyre equivalent of
    // that whole-loop interpreter; calling it once preserves the concrete PC
    // updates across branches/back-edges. Re-invoking it in a synthetic
    // `pc + 1` loop diverges from RPython and corrupts nested-loop bridges.
    // The bridge tracer must leave the real frame exactly at the guard
    // resume state so that a `BridgeCompiled` outcome re-enters
    // `eval_loop_jit` there — the `ContinueRunningNormally` arm does NOT
    // reconstruct the frame, it runs the interpreter forward from the live
    // frame's `last_instr` / value stack as-is. The full-body walker runs from
    // a private `snapshot_for_tracing` copy, but may-force residual calls use
    // the shared execution context during the walk, which advances the live frame's
    // `last_instr` AND its locals (e.g. a loop counter) to the walked
    // opcode's concrete state.  Snapshot the resume state here and restore it
    // after the walk so the post-bridge interpreter resumes at the guard
    // point rather than mid-body or past a dropped loop iteration (a
    // value-stack underflow / off-by-one-iteration result otherwise).
    let resume_state = frame.snapshot_for_tracing();
    let live_frame_addr = frame as *const PyFrame as usize;
    let mut adopted_walk_end_state = false;
    // Arm the bridge `Terminate` no-replay shortcut for this walk.  The walk
    // epilogue (`run_perfn_walk` in trace.rs) reads this flag: only when armed
    // does a bridge `Terminate` walk keep its finish-concrete stash + commit
    // the store journal, so the three decisions (epilogue predicate, journal
    // commit, the `!allow_finish_direct_return` consume-vs-rewind branch
    // below) stay in agreement and a committed journal never strands into a
    // blackhole re-run.
    //
    // `pyjitpl.py:2947` `interpret()` "should always raise": a resumed trace
    // that runs its frames forward to a return raises `DoneWithThisFrame`
    // from the POST-walk state, whatever the resume's frame count — there is
    // no rewind-to-the-deadframe re-run upstream.  A multi-frame resume walks
    // the inlined callees to `SubReturn` and only the LIVE frame's return
    // raises `Terminate`, so its finish-concrete is that frame's result, the
    // same value the blackhole rebuild would produce.  Replaying instead
    // re-executes every residual the walk already ran concretely and
    // double-applies its callee-internal side effects (`Counter.bump()`'s
    // `self.pos += 1` counted twice in
    // `bench/synth/nested_callee_chain_mutation_abort.py`).
    //
    // The CALL_ASSEMBLER callback (`allow_finish_direct_return == false`)
    // cannot carry the concrete itself, so it hands a kept stash to its
    // back-to-back blackhole hook through `CA_WALK_FINISHED_FRAME`.  That hook
    // does not rebuild a framestack — it returns the stashed value as the CA
    // callee's result once `ca_finished_frame == fail_values[0]` proves the
    // walk's live frame IS the frame whose guard failed — so it takes the same
    // any-frame-count arming.  Only the guard-state blackhole fallback below it
    // rebuilds frames, and a kept stash never reaches that path.
    pyre_jit_trace::jitcode_dispatch::fbw_bridge_noreplay_arm(true);
    let outcome = {
        let (driver, _) = crate::eval::driver_pair();
        driver.jit_merge_point_keyed(
            green_key,
            resume_pc,
            &mut jit_state,
            &env,
            || {},
            |meta, sym| {
                // pyjitpl.py:1571-1578: a merge point that only appends to
                // `current_merge_points` restores `self.pc` and goes on
                // interpreting, so the driver re-enters this walk instead of
                // handing the guest back.  Each entry starts from the frame
                // as the previous segment left it, so the snapshot is taken
                // here rather than once outside — `FrameBox` owns a shadow
                // stack root guard and cannot be cloned.
                let trace_frame = bridge_frame_root.frame().snapshot_for_tracing();
                let (action, executed) = trace_bytecode(
                    meta,
                    sym,
                    code,
                    resume_pc,
                    trace_frame,
                    live_frame_addr,
                    false,
                );
                // pyjitpl.py:3048-3091 raise_continue_running_normally:
                // a bridge walk that closed at a merge point and committed
                // its end-of-walk state into the trace snapshot
                // (`flush_walk_end_state_to_frame`) hands the LIVE frame
                // that end state — the walked region's residual calls
                // executed concretely, so resuming at the guard would
                // re-apply every side effect.  Uncommitted → fall through
                // to the guard-state restore below (legacy replay).
                if pyre_jit_trace::trace::take_walk_end_flush_committed() {
                    let frame = bridge_frame_root.frame();
                    frame.restore_resume_state_from(&executed);
                    adopted_walk_end_state = true;
                }
                action
            },
        )
    };
    // Disarm so the flag cannot leak into a later (non-bridge) walk on this
    // thread; the epilogue has already consumed it.
    pyre_jit_trace::jitcode_dispatch::fbw_bridge_noreplay_arm(false);
    if pyre_jit_trace::trace::take_fbw_bridge_declined() {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .record_declined_bridge_guard(descr_arc);
    }

    // #177 bridge `Terminate` no-replay: consume any finish-concrete the walk
    // kept.  A stash survives the epilogue only when `bridge_noreplay_armed`
    // held AND the walk reached `Terminate` with a materialized concrete and
    // no unjournaled effect — the epilogue then committed the store journal,
    // so the walked region's eager side effects (including callee-internal
    // `SetfieldGc`s like `self.pos += 1`) stand exactly once.  Rewinding the
    // live frame to the guard pc and re-running the region — via the
    // `ContinueRunningNormally` re-entry — would apply them a second time.
    // Hand the concrete result forward as `DoneWithThisFrame` instead,
    // mirroring the top-level portal (eval.rs `maybe_compile_and_run`) and
    // `pyjitpl.py:2841` `interpret()` raising `DoneWithThisFrame` from the
    // post-walk state.  Always take (not peek) so a kept stash cannot leak
    // into a later top-level portal `fbw_finish_concrete_take`.
    //
    // CALL_ASSEMBLER callback: this caller returns a bare bool and cannot
    // carry the concrete result itself.  Leave the stash in its GC-rooted
    // cell, record the callee frame in `CA_WALK_FINISHED_FRAME`, and let the
    // CA slow path's back-to-back blackhole hook
    // (`jit_blackhole_resume_from_guard`) take the stash and complete the
    // callee with it — the finishframe `DoneWithThisFrame` /
    // `ExitFrameWithExceptionRef` the assembler caller catches
    // (pyjitpl.py:1688-1698, jitexc.py).  `Terminate` is the LIVE frame's
    // return at any resume frame count, and the live frame here IS the CA
    // callee, so the hook's `ca_finished_frame == fail_values[0]` handshake
    // matches for a multi-frame resume exactly as for a single-frame one.
    if !allow_finish_direct_return {
        if pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_peek().is_some() {
            let finished_frame = bridge_frame_root.frame() as *mut PyFrame as usize;
            CA_WALK_FINISHED_FRAME.with(|c| c.set(finished_frame));
            // This return site predates the four below, so it needs the
            // adopted stamp too: the finished cell only answers the hook's
            // first question, and a walk that both committed its end state and
            // kept a finish-concrete stash still has to answer the second one
            // if the first handshake does not match.
            stamp_ca_walk_frames(&mut bridge_frame_root, false, adopted_walk_end_state);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] ca-finish-noreplay at resume_pc={} key={}",
                    resume_pc, green_key
                );
            }
            return BridgeResolution::ResumeBlackhole;
        }
    }
    match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
        Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
            // `Terminate` is the LIVE frame's function return, about to be
            // popped by its caller with `cv`; any inlined callee the resume
            // reconstructed already returned inside the walk.  No rewind, no
            // blackhole.
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-trace] finish-noreplay at resume_pc={} key={}",
                    resume_pc, green_key
                );
            }
            return BridgeResolution::Finished(cv);
        }
        Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
            // jitexc.py:44: hand the walk's uncaught exception to the
            // guard's portal as ExitFrameWithExceptionRef.
            return BridgeResolution::FinishedException(cv);
        }
        None => {}
    }
    if !adopted_walk_end_state {
        let frame = bridge_frame_root.frame();
        frame.restore_resume_state_from(&resume_state);
    }
    // A multi-frame (inlined-callee) guard restored on the non-flush path
    // leaves the live OUTER frame at the INNERMOST frame's resume pc with the
    // inlined call's result unmaterialized — the inlined callee frame exists
    // only in the bridge/blackhole reconstruction, never in the live frame's
    // operand stack, so the live stack carries a stale NULL where the callee
    // result belongs. `eval_loop_jit` cannot resume the live frame there
    // (it would store the NULL through the post-call ops). The freshly
    // compiled bridge stays attached for subsequent guard failures; complete
    // THIS iteration through the blackhole, which rebuilds the full inlined
    // framestack and runs it — exactly as a cold multi-frame guard does.
    // Signalled by returning `false` (→ `handle_fail` ResumeInBlackhole) even
    // though the bridge compiled. The walk took the non-flush path, so its
    // store journal was rolled back; a residual it executed concretely is not
    // in that journal, so the blackhole re-run applies such an effect twice —
    // the `Terminate` no-replay shortcut above is what keeps a returning walk
    // off this path.
    // The non-flush path applies to single-frame guards too: without the
    // committed walk-end state the live frame's last_instr / operand stack
    // were never advanced through the guard's resume region (only the
    // blackhole syncs them), so "continue running normally" resumes at the
    // enclosing loop header and silently skips the region between the guard
    // and the header — the deopt iteration's remaining inner-loop work.
    // Complete this iteration through the blackhole exactly like the
    // multi-frame case above; the compiled bridge stays attached for
    // subsequent failures.
    let resume_via_blackhole = !adopted_walk_end_state;

    // merge_point handles Finish/CloseLoop via bridge_info.
    if outcome.is_some() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={} blackhole_current={}",
                resume_pc, green_key, resume_via_blackhole
            );
        }
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && resume_via_blackhole,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(!resume_via_blackhole);
    }

    // pyjitpl.py:2982-2983 / 3095-3099 parity:
    // compile_trace() "raises in case it works". In pyre the bridge can
    // already be attached during this step even if jit_merge_point_keyed()
    // did not surface DetailedDriverRunOutcome::Jump yet. Stop tracing as
    // soon as the backend metadata shows that the bridge is attached.
    let compiled = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .bridge_was_compiled(green_key, trace_id, fail_index)
    };
    if compiled {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            driver.meta_interp_mut().abort_trace(false);
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] compiled at resume_pc={} key={} (attached) blackhole_current={}",
                resume_pc, green_key, resume_via_blackhole
            );
        }
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && resume_via_blackhole,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(!resume_via_blackhole);
    }

    // If the driver is no longer tracing, the bridge was compiled
    // (or aborted) inside merge_point. Check whether a bridge was
    // actually attached to distinguish success from abort.
    //
    // pyjitpl.py:3057 raise_continue_running_normally: a trace that
    // reached a merge point exits with ContinueRunningNormally from the
    // CURRENT state — never through the blackhole-from-guard path. A
    // committed walk already executed the region and the live frame
    // adopted its end state above, so the caller must continue running
    // normally even when the trace closed as a new loop at an inner
    // header instead of attaching a bridge to this guard; resuming in
    // the blackhole would re-apply every walked side effect from the
    // guard-time values.
    let tracing_active = {
        let (driver, _) = crate::eval::driver_pair();
        driver.is_tracing()
    };
    if !tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] trace ended at resume_pc={} key={} compiled={} adopted={}",
                resume_pc, green_key, compiled, adopted_walk_end_state
            );
        }
        let continue_compiled = compiled || adopted_walk_end_state;
        stamp_ca_walk_frames(
            &mut bridge_frame_root,
            !allow_finish_direct_return && !continue_compiled,
            !allow_finish_direct_return && adopted_walk_end_state,
        );
        return bridge_resolution_from_bool(continue_compiled);
    }

    // Trace did not converge into a bridge. Abort like RPython's
    // run_blackhole_interp_to_cancel_tracing fallback path.
    if tracing_active {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-trace] abort: no-bridge key={} trace={} fail={} resume_pc={}",
                green_key, trace_id, fail_index, resume_pc
            );
        }
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp_mut().abort_trace(false);
    }
    // A committed walk has already executed the region into the live
    // frame; the blackhole replay must not run even on this abort path.
    stamp_ca_walk_frames(
        &mut bridge_frame_root,
        !allow_finish_direct_return && !adopted_walk_end_state,
        !allow_finish_direct_return && adopted_walk_end_state,
    );
    bridge_resolution_from_bool(adopted_walk_end_state)
}

/// compile.py:701-717 handle_fail for call_assembler guard failures.
/// Checks must_compile (jitcounter.tick), and if threshold reached,
/// traces the alternate path via trace_and_compile_from_bridge.
///
/// pyjitpl.py:2914 `handle_guard_failure(self, resumedescr, deadframe)`
/// — descr identity is the only argument crossing the C-ABI boundary;
/// the receiver derives `(green_key, trace_id, fail_index)` from the
/// recovered Arc, mirroring `compile.py:706-708 _trace_and_compile_
/// from_bridge` which walks `resumedescr.rd_loop_token.loop_token_wref()`
/// for the owning JCT.
fn jit_ca_handle_guard_failure(
    raw_values_ptr: *const i64,
    num_values: usize,
    descr_addr: usize,
) -> bool {
    if raw_values_ptr.is_null() || num_values == 0 {
        return false;
    }
    // `enter_profiler_tracing` is not re-entrant (pyjitpl.py:2914 — RPython's
    // `handle_guard_failure` unwinds to the top-level `execute_token` before any
    // tracing decision, so a guard never fires while another trace is open).
    // pyre's CALL_ASSEMBLER guard callback runs synchronously from the backend
    // trampoline, so it CAN fire mid-trace: the self-recursion fold replays its
    // CALL_ASSEMBLER concretely while the outer trace is still recording, and a
    // deopt in that callee reaches here with the outer trace's profiler event
    // still open.  Starting a nested bridge trace would panic in
    // `start_retrace_from_guard`'s `enter_profiler_tracing`.  Bail to the
    // blackhole resume (return false) so the callee completes in the interpreter
    // and the outer trace keeps recording; the deferred bridge compiles later
    // when this guard fails outside a trace.
    {
        let (driver, _) = crate::eval::driver_pair();
        if driver.is_tracing() {
            return false;
        }
    }
    let raw_values_input = unsafe { std::slice::from_raw_parts(raw_values_ptr, num_values) };
    let mut raw_values_vec = raw_values_input.to_vec();

    // compile.py:706-708 _trace_and_compile_from_bridge.  Native CA code
    // crosses the backend boundary with only the raw descr pointer; recover
    // the backend FailDescr Arc before any guard-failure routing so identity
    // is read from the descr just like PyPy reads `resumedescr`.  The
    // recovery is infallible for live JIT code —
    // `Backend::fail_descr_arc_from_addr` panics if the raw value is not
    // a live `FailDescrCell` pointer, matching RPython's
    // `cpu.get_latest_descr(deadframe)` (warmspot.py:1021) which has no
    // failure mode.  `trace_and_compile_from_bridge` itself takes only
    // the descr Arc.
    let descr_arc: std::sync::Arc<dyn majit_ir::Descr> = {
        use majit_backend::Backend;
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp()
            .backend()
            .fail_descr_arc_from_addr(descr_addr)
    };
    let Some((source_green_key, source_trace_id, source_fail_index)) =
        bridge_source_identity_from_descr(&descr_arc)
    else {
        return false;
    };
    let deadframe_types = {
        let (driver, _) = crate::eval::driver_pair();
        driver.get_recovery_slot_types(source_green_key, source_trace_id, source_fail_index)
    };
    let _raw_values_roots =
        ResumeDeadframeRoots::register(&mut raw_values_vec, deadframe_types.as_deref());
    let raw_values = raw_values_vec.as_slice();

    // This callback has no channel for the exception value carried by a
    // failing CALL_ASSEMBLER exception guard.  Compiling from its post-call
    // resume state would treat the null call result as a normal operand.
    // Leave exception-guard recovery to the blackhole path, which owns the
    // callee exception and propagates it through the caller frames.
    if descr_arc.is_guard_exc() {
        return false;
    }

    // compile.py:738-784 must_compile: jitcounter.tick(guard_hash, increment)
    let (must_compile, owning_key) = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .must_compile_with_values(&descr_arc, raw_values, source_green_key)
    };
    // compile.py:702-703: must_compile() and not stack_almost_full()
    if !must_compile || majit_metainterp::MetaInterp::<()>::stack_almost_full() {
        return false;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] must_compile fired: key={} trace={} fail={}",
            source_green_key, source_trace_id, source_fail_index,
        );
    }

    // compile.py:719-726: get exit_layout from the compiled trace.
    // Use owning_key (not green_key) — after retrace the descriptor
    // may belong to a different compiled entry than green_key.
    let exit_layout = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().get_compiled_exit_layout_in_trace(
            owning_key,
            source_trace_id,
            source_fail_index,
        )
    };
    let Some(exit_layout) = exit_layout else {
        return false;
    };

    // Obtain callee frame from deadframe vable header.
    // pyre vable_boxes = [frame, ni, code, vsd, ns, locals..., stack...],
    // so raw_values[0] is the callee's PyFrame pointer.
    let frame_ptr = raw_values[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return false;
    }
    let frame = unsafe { &mut *frame_ptr };

    // compile.py:704-709 try/finally: `start_compiling()` before
    // bridge, `done_compiling()` on every unwind path.  RAII guard
    // dispatches both via `descr.as_fail_descr()` (instance-method
    // dispatch per `compile.py:786-795`); drop pairs `done_compiling`
    // with the matching `start_compiling` even on panic.
    let compiled = {
        let _guard = crate::eval::GuardCompilingScope::new(&descr_arc);
        // CALL_ASSEMBLER guard failures grab their callee exception on the
        // blackhole leg, not here; pass 0 so the non-exception-guard
        // deferral keys only off the general guard path's `guard_exc`.
        // `allow_finish_direct_return = false`: this callback returns a bare
        // bool to native code and has no channel for a concrete result; a
        // walk that terminates with a kept finish-concrete stash hands it to
        // the back-to-back blackhole hook via `CA_WALK_FINISHED_FRAME`
        // (returned as `ResumeBlackhole` here).
        match trace_and_compile_from_bridge(&descr_arc, frame, raw_values, &exit_layout, 0, false) {
            BridgeResolution::CompiledContinue => true,
            BridgeResolution::ResumeBlackhole => false,
            // Unreachable: for this caller a kept stash takes the
            // `CA_WALK_FINISHED_FRAME` handshake path, never a terminal
            // variant.
            BridgeResolution::Finished(_) | BridgeResolution::FinishedException(_) => {
                debug_assert!(
                    false,
                    "CALL_ASSEMBLER bridge returned a terminal no-replay result despite disarm"
                );
                false
            }
        }
    };
    // compile.py:204-207 record_loop_or_bridge registers every bridge's dependencies.
    crate::eval::register_quasi_immutable_deps(source_green_key);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][ca-bridge] compiled={} key={} trace={} fail={}",
            compiled, source_green_key, source_trace_id, source_fail_index,
        );
    }

    drop(_raw_values_roots);
    // compile.py:701-717 is an exclusive bridge/blackhole branch, but pyre's
    // handle_fail_resume_guard in majit-backend-dynasm always runs the
    // blackhole hook after this bridge hook. Its caller buffer is not rooted,
    // so the blackhole must read the values rooted here on both outcomes.
    CA_WALK_RESUME_DEADFRAME.with(|c| {
        *c.borrow_mut() = Some(raw_values_vec);
    });

    compiled
}

/// Feed a CALL_ASSEMBLER callee guard through the normal bridge-hotness path.
/// The caller has already recovered the live descriptor and exit values; the
/// rest is the same must-compile/bridge attachment sequence used by the native
/// CALL_ASSEMBLER guard callback above.
#[allow(dead_code)]
struct CaBridgeAttempt {
    terminal_declined: bool,
}

#[allow(dead_code)]
fn try_compile_ca_bridge(
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    raw_values: &[i64],
) -> CaBridgeAttempt {
    if raw_values.is_empty() {
        return CaBridgeAttempt {
            terminal_declined: false,
        };
    }
    let Some((source_green_key, source_trace_id, source_fail_index)) =
        bridge_source_identity_from_descr(descr_arc)
    else {
        return CaBridgeAttempt {
            terminal_declined: false,
        };
    };
    let (must_compile, owning_key) = {
        let (driver, _) = crate::eval::driver_pair();
        driver
            .meta_interp_mut()
            .must_compile_with_values(descr_arc, raw_values, source_green_key)
    };
    if !must_compile || majit_metainterp::MetaInterp::<()>::stack_almost_full() {
        let terminal_declined = {
            let (driver, _) = crate::eval::driver_pair();
            driver.meta_interp().bridge_declined_terminally(descr_arc)
        };
        return CaBridgeAttempt { terminal_declined };
    }
    let exit_layout = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().get_compiled_exit_layout_in_trace(
            owning_key,
            source_trace_id,
            source_fail_index,
        )
    };
    let Some(exit_layout) = exit_layout else {
        return CaBridgeAttempt {
            terminal_declined: false,
        };
    };
    let frame_ptr = raw_values[0] as *mut PyFrame;
    if frame_ptr.is_null() {
        return CaBridgeAttempt {
            terminal_declined: false,
        };
    }
    let frame = unsafe { &mut *frame_ptr };
    let _guard = crate::eval::GuardCompilingScope::new(descr_arc);
    let _compiled = matches!(
        trace_and_compile_from_bridge(descr_arc, frame, raw_values, &exit_layout, 0, false),
        BridgeResolution::CompiledContinue
    );
    // The wasm CALL_ASSEMBLER path likewise bypasses handle_fail's dependency drain.
    crate::eval::register_quasi_immutable_deps(owning_key);
    // `MetaInterp::compile_bridge` records a wasm `Unsupported` before the
    // walker returns.  Reuse that canonical guard identity here rather than
    // creating a separate CA-side decline table.
    let terminal_declined = {
        let (driver, _) = crate::eval::driver_pair();
        driver.meta_interp().bridge_declined_terminally(descr_arc)
    };
    CaBridgeAttempt { terminal_declined }
}

/// Host completion for an in-guest self-recursive CALL_ASSEMBLER callee that
/// deopted (`PYRE_WASM_CA`). The wasm CA arm `call_indirect`s this through the
/// shared `__indirect_function_table` (its slot published via
/// `majit_backend_wasm::set_ca_deopt_helper_slot` during `JIT_DRIVER` init)
/// when an in-guest callee frame returns a non-finish `fail_index`.
///
/// It is the wasm analog of the dynasm/cranelift CA fast path's deopt leg
/// (`call_assembler_fast_path_heap` → `jit_blackhole_resume_from_guard`): build
/// a `DeadFrame` from the callee's already-run frame, then **blackhole-resume it
/// AT the guard PC** — so the callee's pre-guard work is not re-executed (unlike
/// a force / entry re-run) — and return the call result, a boxed Ref. A callee
/// that actually finished (a base case, or a chained-bridge finish the arm's
/// static fast-path set did not recognise) short-circuits to its output slot.
///
/// `frame_ptr` is the deopted callee frame; `compiled_ptr` is the source
/// loop's `CompiledWasmLoop`, both baked into the trace by `compile_bridge`.
#[cfg(target_arch = "wasm32")]
pub extern "C" fn wasm_ca_resume_deopt(frame_ptr: i64, compiled_ptr: i64) -> i64 {
    use majit_backend::Backend;
    let frame =
        majit_backend_wasm::dead_frame_from_ran_frame(compiled_ptr as usize, frame_ptr as usize);

    // Decode the exit through the same Backend accessors the host's outermost
    // deopt handling uses (`run_compiled_detailed_with_values`); extract owned
    // values so the driver borrow is released before the blackhole re-enters it.
    enum Outcome {
        Finished(i64),
        FinishedException(i64),
        Deopt {
            descr_arc: std::sync::Arc<dyn majit_ir::Descr>,
            green_key: u64,
            exit_layout: majit_metainterp::CompiledExitLayout,
            raw_values: Vec<i64>,
            guard_exc: i64,
        },
    }
    let outcome = {
        let (driver, _) = crate::eval::driver_pair();
        let mi = driver.meta_interp();
        let backend = mi.backend();
        let descr_arc = backend.get_latest_descr_arc(&frame);
        let descr = descr_arc
            .as_fail_descr()
            .expect("CA deopt: get_latest_descr_arc returned a non-FailDescr Descr");
        if descr.is_finish() {
            let result = backend.get_ref_value(&frame, 0).as_usize() as i64;
            // compile.py:658-662 ExitFrameWithExceptionDescrRef parity: a FINISH
            // descr is either DoneWithThisFrame (return the value) or
            // ExitFrameWithException (the callee raised; slot 0 holds the
            // exception). The self-recursive callee re-raised — propagate it
            // through the exception channel like the outer Finished arm
            // (`eval.rs`'s `handle_fail`) instead of banking it as the
            // recursive-call return, which would surface as `int + <exc>`.
            if descr.is_exit_frame_with_exception() {
                Outcome::FinishedException(result)
            } else {
                Outcome::Finished(result)
            }
        } else {
            let green_key = majit_backend::descr_owning_jct(descr)
                .map(|jct| jct.green_key())
                .unwrap_or(0);
            let exit_layout = mi.build_exit_layout_for_descr(green_key, descr);
            let raw_values: Vec<i64> = descr
                .fail_arg_types()
                .iter()
                .enumerate()
                .map(|(i, &tp)| match tp {
                    majit_ir::Type::Int => backend.get_int_value(&frame, i),
                    majit_ir::Type::Ref => backend.get_ref_value(&frame, i).as_usize() as i64,
                    majit_ir::Type::Float => backend.get_float_value(&frame, i).to_bits() as i64,
                    majit_ir::Type::Void => 0,
                })
                .collect();
            let guard_exc = backend.grab_exc_value(&frame).0 as i64;
            Outcome::Deopt {
                descr_arc,
                green_key,
                exit_layout,
                raw_values,
                guard_exc,
            }
        }
    };

    match outcome {
        Outcome::Finished(r) => r,
        // warmspot.py:998-1005 ExitFrameWithExceptionRef: the callee raised.
        // Publish the exception so the caller's GUARD_NO_EXCEPTION (after the
        // CALL_ASSEMBLER) fires, and return garbage — parity with
        // `handle_blackhole_result`'s ExitFrameWithExceptionRef arm.
        Outcome::FinishedException(exc) => {
            if exc != 0 {
                store_jit_exception(exc);
            }
            0
        }
        Outcome::Deopt {
            descr_arc,
            green_key,
            exit_layout,
            raw_values,
            mut guard_exc,
        } => {
            // `grab_exc_value` cleared the only root for the pending exception
            // (dynasm `ca_helper`, llmodel.py:240); root the bare carrier while
            // the bridge-compile hook and the blackhole run — both may allocate,
            // and a moving collection would otherwise leave `guard_exc` stale.
            // Inert today (wasm host allocations never collect) but keeps the
            // carrier rooted at parity with dynasm if that invariant changes.
            let _guard_exc_root = BareRefRoot::register(&mut guard_exc);
            let attempt = try_compile_ca_bridge(&descr_arc, &raw_values);
            if attempt.terminal_declined {
                // This target cannot reach compiled steady state: each CA
                // invocation would blackhole.  Invalidate callers so the next
                // trace refuses this target and returns to the baseline path.
                majit_backend_wasm::mark_call_assembler_terminal_decline(compiled_ptr as usize);
            }
            // The walk above may have carried the callee past the guard —
            // running the resumed region to the callee's return, or committing
            // its end-of-walk state into the live frame — executing every
            // residual in it concretely.  Take the same completion the native
            // CA hook takes; the guard-state resume below would re-run that
            // region and apply its side effects a second time (#177).  This is
            // the wasm side of the `CA_WALK_*` handshake: nothing else on this
            // target reads those cells, so an unconsumed completion is silently
            // dropped into a replay.
            let ca_finished_frame = CA_WALK_FINISHED_FRAME.with(|c| c.replace(0));
            let ca_adopted_frame = CA_WALK_ADOPTED_FRAME.with(|c| c.replace(0));
            let callee_frame = raw_values.first().copied().unwrap_or(0) as usize;
            if let Some(result) =
                ca_complete_after_bridge_walk(ca_finished_frame, ca_adopted_frame, callee_frame)
            {
                return result;
            }
            // compile.py:950-958 `ResumeGuardForcedDescr.handle_fail`: only a
            // GUARD_NOT_FORCED failure fishes the cache the force saved, keyed
            // by the callee frame `raw_values[0]` names.
            let forced_cache_owner = if descr_arc.is_guard_forced() {
                callee_frame as *const pyre_interpreter::pyframe::PyFrame
            } else {
                std::ptr::null()
            };
            let bh = crate::eval::resume_in_blackhole_from_exit_layout(
                &raw_values,
                &exit_layout,
                guard_exc,
                forced_cache_owner,
                false,
            );
            handle_blackhole_result(bh, green_key).unwrap_or(0)
        }
    }
}

// ── Callee frame creation for call_assembler ─────────────────────

/// Public wrapper for trace-through inlining.
pub fn create_callee_frame_impl_pub(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    create_callee_frame_impl(caller_frame, callable, args)
}

fn fill_positional_defaults_for_jit_call<'a>(
    callable: PyObjectRef,
    w_code: *const (),
    args: &'a [PyObjectRef],
) -> Cow<'a, [PyObjectRef]> {
    let defaults = unsafe { function_get_defaults(callable) };
    if defaults.is_null() {
        return Cow::Borrowed(args);
    }

    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    let nparams = code.arg_count as usize;
    if args.len() >= nparams {
        return Cow::Borrowed(args);
    }

    let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
        unsafe { pyre_object::w_tuple_len(defaults) }
    } else {
        0
    };
    if ndefaults == 0 {
        return Cow::Borrowed(args);
    }
    let first_default = nparams.saturating_sub(ndefaults);
    if args.len() < first_default {
        // function.py:_flat_pycall_defaults is entered only after argument
        // matching proves that all required positional parameters are present.
        // Do not synthesize PY_NULL for missing required args here; callers
        // that reach this helper without enough args must keep the original
        // frame shape and let the normal call/resume path handle the error.
        return Cow::Borrowed(args);
    }

    let defaults_to_load = nparams - first_default;
    let default_start = ndefaults - defaults_to_load;
    let mut full = Vec::with_capacity(nparams);
    full.extend_from_slice(args);
    for i in args.len()..nparams {
        if i >= first_default {
            let default_idx = default_start + (i - first_default);
            let Some(default) =
                (unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) })
            else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][defaults] tuple access failed default_idx={default_idx} defaults={defaults:p}"
                    );
                }
                return Cow::Borrowed(args);
            };
            full.push(default);
        } else {
            full.push(PY_NULL);
        }
    }
    Cow::Owned(full)
}

fn create_callee_frame_impl_1_boxed(
    caller_frame: i64,
    callable: PyObjectRef,
    boxed_arg: PyObjectRef,
) -> i64 {
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let w_globals = unsafe { function_get_globals_obj(callable) };
    let one_arg = [boxed_arg];
    let args = fill_positional_defaults_for_jit_call(callable, w_code, &one_arg);
    let args = args.as_ref();

    alloc_callee_frame(w_code, args, w_globals, caller.execution_context) as i64
}

fn create_self_recursive_callee_frame_impl_1_boxed(
    caller_frame: i64,
    boxed_arg: PyObjectRef,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.pycode;
    let w_globals = caller.w_globals;
    let execution_context = caller.execution_context;

    let frame_ptr = alloc_callee_frame(func_code, &[boxed_arg], w_globals, execution_context);
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame] ptr={frame_ptr:p} locals=0x{:x} vsd={} boxed_arg=0x{:x}",
            f.locals_cells_stack_w as usize, f.valuestackdepth, boxed_arg as usize,
        );
    }
    frame_ptr as i64
}

fn create_callee_frame_impl(caller_frame: i64, callable: i64, args: &[PyObjectRef]) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    create_callee_frame_in_ctx(caller.execution_context, callable as PyObjectRef, args)
}

/// [`create_callee_frame_impl`] with the execution context passed directly.
///
/// The caller frame contributes nothing else here — `function.py funccall`
/// builds the callee frame out of the *callee's* code, globals and closure,
/// and takes the execution context from the space.  A residual helper that
/// already holds `space.getexecutioncontext()` therefore has no reason to
/// resolve a caller frame for this.
fn create_callee_frame_in_ctx(
    execution_context: *const pyre_interpreter::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> i64 {
    let w_code = unsafe { pyre_interpreter::getcode(callable) };
    let w_globals = unsafe { function_get_globals_obj(callable) };
    let args = fill_positional_defaults_for_jit_call(callable, w_code, args);
    let args = args.as_ref();

    alloc_callee_frame(w_code, args, w_globals, execution_context) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_0(caller_frame: i64, callable: i64) -> i64 {
    create_callee_frame_impl(caller_frame, callable, &[])
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1(caller_frame: i64, callable: i64, arg0: i64) -> i64 {
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, arg0 as PyObjectRef)
}

/// Self-recursive single-arg variant.
///
/// This skips rediscovering code/globals from a function object and reuses the
/// caller frame's code/namespace/execution_context directly, which matches the
/// existing self-recursive raw helper path more closely.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1(caller_frame: i64, arg0: i64) -> i64 {
    debug_assert!(
        caller_frame != 0,
        "jit_create_self_recursive_callee_frame_1: caller_frame is null"
    );
    if caller_frame == 0 {
        // Invariant violation: fall back to heap allocation with a
        // minimal frame. RPython never aborts the process for JIT
        // invariant failures — it falls back to tracing abort or
        // blackhole resume.
        return 0;
    }
    create_self_recursive_callee_frame_impl_1_boxed(caller_frame, arg0 as PyObjectRef)
}

/// Self-recursive raw-int variant: creates the frame WITHOUT boxing
/// the argument. The raw int is passed directly to compiled code via
/// CallAssemblerI inputargs. Boxing only happens on guard failure
/// (in force_fn / jit_force_self_recursive_call_raw_1).
///
/// RPython parity: compiled code uses jitframe slots, not PyFrame
/// locals. Frame locals are only needed for interpreter fallback.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_self_recursive_callee_frame_1_raw_int(
    caller_frame: i64,
    raw_int_arg: i64,
) -> i64 {
    let caller = unsafe { &*(caller_frame as *const PyFrame) };
    let func_code = caller.pycode;
    let w_globals = caller.w_globals;
    let execution_context = caller.execution_context;

    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);

    let frame_ptr = alloc_callee_frame(func_code, &[boxed], w_globals, execution_context);
    if majit_metainterp::majit_log_enabled() {
        let f = unsafe { &*frame_ptr };
        eprintln!(
            "[jit][ca-frame-raw] ptr={frame_ptr:p} locals=0x{:x} local0=0x{:x} vsd={} raw_arg={}",
            f.locals_cells_stack_w as usize,
            locals_w!(f)[0] as usize,
            f.valuestackdepth,
            raw_int_arg,
        );
    }
    frame_ptr as i64
}

/// Raw-int variant: accepts a raw int and boxes it internally.
/// Eliminates trace_box_int CallI from the trace (boxing folded into frame creation).
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_1_raw_int(
    caller_frame: i64,
    callable: i64,
    raw_int_arg: i64,
) -> i64 {
    let boxed = pyre_object::intobject::w_int_new(raw_int_arg);
    create_callee_frame_impl_1_boxed(caller_frame, callable as PyObjectRef, boxed)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_2(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_3(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
        ],
    )
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_create_callee_frame_4(
    caller_frame: i64,
    callable: i64,
    arg0: i64,
    arg1: i64,
    arg2: i64,
    arg3: i64,
) -> i64 {
    create_callee_frame_impl(
        caller_frame,
        callable,
        &[
            arg0 as PyObjectRef,
            arg1 as PyObjectRef,
            arg2 as PyObjectRef,
            arg3 as PyObjectRef,
        ],
    )
}

pub fn callee_frame_helper(nargs: usize) -> Option<*const ()> {
    match nargs {
        0 => Some(jit_create_callee_frame_0 as *const ()),
        1 => Some(jit_create_callee_frame_1 as *const ()),
        2 => Some(jit_create_callee_frame_2 as *const ()),
        3 => Some(jit_create_callee_frame_3 as *const ()),
        4 => Some(jit_create_callee_frame_4 as *const ()),
        _ => None,
    }
}

/// Force callee and return BOXED result for the retired inline-call path.
/// warmspot.py:449 result_type=REF: jit_force_callee_frame already
/// returns boxed Ref, so this is just a pass-through.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_force_callee_frame_boxed(frame_ptr: i64) -> i64 {
    jit_force_callee_frame(frame_ptr)
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_drop_callee_frame(frame_ptr: i64) {
    if frame_ptr & 1 != 0 {
        return;
    }
    let ptr = frame_ptr as *mut PyFrame;
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[jit][ca-drop] ptr={ptr:p}");
    }
    unroot_callee_frame(ptr);
}

/// Store a W_Root into a callee frame's `locals_cells_stack_w[idx]`.
///
/// Residual helper for the inline back-edge CALL_ASSEMBLER writeback
/// (do_recursive_call, pyjitpl.py:1579-1602): the callee's compiled
/// loop reads its locals from the frame object at entry, so the
/// inlined prefix's register values are stored back through these
/// helpers before the call. Plain store, same as `PyFrame::push`.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_frame_set_slot_ref(frame_ptr: i64, idx: i64, value: i64) {
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    locals_w_mut!(frame)[idx as usize] = value as PyObjectRef;
}

/// `jit_frame_set_slot_ref` for a raw int value — boxes via `w_int_new`.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_frame_set_slot_int(frame_ptr: i64, idx: i64, raw: i64) {
    let boxed = pyre_object::intobject::w_int_new(raw);
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    locals_w_mut!(frame)[idx as usize] = boxed;
}

/// `jit_frame_set_slot_ref` for a raw float value — boxes via `w_float_new`.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_frame_set_slot_float(frame_ptr: i64, idx: i64, raw: f64) {
    let boxed = pyre_object::floatobject::w_float_new(raw);
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    locals_w_mut!(frame)[idx as usize] = boxed;
}

// ===========================================================================
// Blackhole helper functions
//
// RPython blackhole.py: bhimpl_recursive_call_i, bhimpl_residual_call_*
//
// These are called by the BlackholeInterpreter through JitCode.fn_ptrs.
// Residual calls execute without accidental JIT re-entry; recursive portal
// calls are routed explicitly through the jitdriver's portal runner.
// ===========================================================================

fn bh_call_self_recursive_portal(
    ec: *const pyre_interpreter::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Option<i64> {
    // "Is the caller running this same code object?" is an identity test on
    // the chain slot, which is what `vref_referent` exists for: it returns
    // what `topframeref` NAMES, without forcing.  `gettopframe_raw` forces,
    // and a force while the trace records clears the
    // `TOKEN_TRACING_RESCALL` marker (`virtualref.py:161-167`) that
    // `tracing_after_residual_call` reads back as "the callee escaped this
    // vref" — the probe would report its own read as an escape, and the
    // recorded `VIRTUAL_REF_FINISH` then materializes the caller frame on
    // every execution of the compiled trace.
    //
    // A null referent names no reachable frame (a caller that is still
    // virtual), so there is nothing to compare: decline, and the generic
    // residual below runs the call.
    let parent_frame_ptr =
        unsafe { pyre_interpreter::executioncontext::vref_referent((*ec).topframeref) };
    if parent_frame_ptr.is_null() {
        return None;
    }
    let parent_frame = unsafe { &*parent_frame_ptr };
    let callable_code = unsafe { pyre_interpreter::getcode(callable) };
    if parent_frame.pycode != callable_code {
        return None;
    }
    if !recursive_force_cache_safe(callable) {
        return None;
    }

    // blackhole.py:1095-1116 bhimpl_recursive_call_* reaches the
    // jitdriver's portal runner.  This branch narrows pyre's generic
    // Python CALL helper back to that shape for self-recursive portal
    // calls; non-recursive residual calls below remain opaque plain calls.
    let frame_ptr = create_callee_frame_in_ctx(ec, callable, args);
    let result = {
        let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
        crate::eval::portal_runner_result(frame)
    };
    jit_drop_callee_frame(frame_ptr);
    Some(match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    })
}

/// RPython: bhimpl_recursive_call_i — call a Python function in blackhole mode.
///
/// The blackhole pops callable and args into registers before calling this.
/// blackhole.py bhimpl_residual_call parity: variable-arity call helper.
///
/// Convention: residual_call_r_r dispatches with
/// args=[callable, null_or_self, arg0, ..., argN].  RPython
/// `bhimpl_residual_call_r_r` (blackhole.py:1227) carries no frame —
/// `cpu.bh_call_r(func, None, args_r, ...)` — and `bh_call_fn_impl` carries
/// none either: it takes `space.getexecutioncontext()` and threads that
/// through the dispatch, so no branch resolves a caller frame out of
/// `topframeref`.  `null_or_self` is the CALL opcode's self slot
/// (`eval.rs`'s `PyFrame::call`): non-null means a method receiver to prepend as
/// arg0, NULL means a plain call.
///
/// For nargs=0: fn(callable, null_or_self) → 2 args
/// For nargs=1: fn(callable, null_or_self, arg0) → 3 args
/// For nargs=2: fn(callable, null_or_self, arg0, arg1) → 4 args
/// etc.
pub extern "C" fn bh_call_fn(callable: i64, null_or_self: i64, arg0: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[arg0 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_0(callable: i64, null_or_self: i64) -> i64 {
    bh_call_fn_impl(callable as PyObjectRef, null_or_self as PyObjectRef, &[])
}

pub extern "C" fn bh_call_fn_2(callable: i64, null_or_self: i64, arg0: i64, arg1: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[arg0 as PyObjectRef, arg1 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_3(callable: i64, null_or_self: i64, a0: i64, a1: i64, a2: i64) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[a0 as PyObjectRef, a1 as PyObjectRef, a2 as PyObjectRef],
    )
}

pub extern "C" fn bh_call_fn_4(
    callable: i64,
    null_or_self: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_5(
    callable: i64,
    null_or_self: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_6(
    callable: i64,
    null_or_self: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_7(
    callable: i64,
    null_or_self: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
        ],
    )
}

pub extern "C" fn bh_call_fn_8(
    callable: i64,
    null_or_self: i64,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
) -> i64 {
    bh_call_fn_impl(
        callable as PyObjectRef,
        null_or_self as PyObjectRef,
        &[
            a0 as PyObjectRef,
            a1 as PyObjectRef,
            a2 as PyObjectRef,
            a3 as PyObjectRef,
            a4 as PyObjectRef,
            a5 as PyObjectRef,
            a6 as PyObjectRef,
            a7 as PyObjectRef,
        ],
    )
}

/// Per-arity `bh_call_fn_<n>` thunks for nargs 9..=14, sharing the
/// `(callable, null_or_self, arg0..arg{n-1})` ABI of the explicit
/// `bh_call_fn_0..8` above.  nargs=14 (16 i64 params) is the ceiling
/// the backend dispatch table (`call_stub.rs::dispatch_arity_body!`,
/// `MAX_HOST_CALL_ARITY` = 16) supports; CALL with nargs > 14 falls
/// through to `emit_abort_permanent!`.
macro_rules! bh_call_fn_arity {
    ($name:ident; $($arg:ident),+ $(,)?) => {
        pub extern "C" fn $name(callable: i64, null_or_self: i64, $($arg: i64),+) -> i64 {
            bh_call_fn_impl(
                callable as PyObjectRef,
                null_or_self as PyObjectRef,
                &[$($arg as PyObjectRef),+],
            )
        }
    };
}
bh_call_fn_arity!(bh_call_fn_9; a0, a1, a2, a3, a4, a5, a6, a7, a8);
bh_call_fn_arity!(bh_call_fn_10; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
bh_call_fn_arity!(bh_call_fn_11; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
bh_call_fn_arity!(bh_call_fn_12; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
bh_call_fn_arity!(bh_call_fn_13; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
bh_call_fn_arity!(bh_call_fn_14; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);

/// CALL_KW residual shared body (`call_kw(callable, self_or_null,
/// positional, kwnames)` HLOp → per-arity `residual_call_r_r`).  Carries no
/// frame, like [`bh_call_fn_impl`]: `pyopcode.py:1402 CALL_FUNCTION_KW`
/// settles the C-profile question against its own frame and then calls the
/// frameless `space.call_args(w_function, args)`, which is what a residual
/// reaches.  Keyword resolution + the dispatched call run under
/// `force_plain_eval` (blackhole.py:1225 `bhimpl_residual_call_*` is an
/// opaque CPU call — no JIT re-entry; `call_kw`'s
/// `call_user_function_resolved` fast path routes through the JIT-aware
/// `get_eval_fn()`, so the guard is what keeps it on `eval_frame_plain`).
/// `positional` is `arg0..argN-1` in positional order (keyword tail
/// included); `kwnames` is the constant kwnames tuple.  MayForce: keyword
/// binding and the dispatched call may run Python.
fn bh_call_kw_impl(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    kwnames: PyObjectRef,
    positional: &[PyObjectRef],
) -> i64 {
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_kw_impl requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_kw_in_ctx(ec, callable, null_or_self, positional, kwnames)
    };
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// Per-arity `bh_call_kw_<n>` thunks for the CALL_KW residual, ABI
/// `(callable, null_or_self, kwnames, arg0..arg{n-1})` = 3 + n i64 params.
/// The backend dispatch tops out at `MAX_HOST_CALL_ARITY` = 16 i64 args, so
/// the kwnames slot leaves room for nargs 0..=13; CALL_KW with nargs > 13
/// falls through to `emit_abort_permanent!`.
macro_rules! bh_call_kw_arity {
    ($name:ident; $($arg:ident),* $(,)?) => {
        pub extern "C" fn $name(
            callable: i64,
            null_or_self: i64,
            kwnames: i64,
            $($arg: i64),*
        ) -> i64 {
            bh_call_kw_impl(
                callable as PyObjectRef,
                null_or_self as PyObjectRef,
                kwnames as PyObjectRef,
                &[$($arg as PyObjectRef),*],
            )
        }
    };
}
bh_call_kw_arity!(bh_call_kw_0;);
bh_call_kw_arity!(bh_call_kw_1; a0);
bh_call_kw_arity!(bh_call_kw_2; a0, a1);
bh_call_kw_arity!(bh_call_kw_3; a0, a1, a2);
bh_call_kw_arity!(bh_call_kw_4; a0, a1, a2, a3);
bh_call_kw_arity!(bh_call_kw_5; a0, a1, a2, a3, a4);
bh_call_kw_arity!(bh_call_kw_6; a0, a1, a2, a3, a4, a5);
bh_call_kw_arity!(bh_call_kw_7; a0, a1, a2, a3, a4, a5, a6);
bh_call_kw_arity!(bh_call_kw_8; a0, a1, a2, a3, a4, a5, a6, a7);
bh_call_kw_arity!(bh_call_kw_9; a0, a1, a2, a3, a4, a5, a6, a7, a8);
bh_call_kw_arity!(bh_call_kw_10; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
bh_call_kw_arity!(bh_call_kw_11; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
bh_call_kw_arity!(bh_call_kw_12; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
bh_call_kw_arity!(bh_call_kw_13; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);

/// blackhole.py:1224 bhimpl_residual_call: cpu.bh_call_r.
/// RPython: cpu.bh_call_r (llmodel.py:816) invokes calldescr.call_stub_r
/// directly — a plain function-pointer call, no portal_runner indirection.
/// Only bhimpl_recursive_call_* (blackhole.py:1095) uses the portal
/// runner to re-enter JIT.
///
/// RPython `bhimpl_residual_call_r_r` (blackhole.py:1227) carries no
/// frame: `cpu.bh_call_r(func, None, args_r, ...)`.  The user-function
/// dispatch below reads `execution_context` and the recursive-portal probe
/// walks the caller chain, so the parent frame is resolved here from the
/// execution context's top frame (`space.getexecutioncontext()
/// .gettopframe()`), matching the upstream frame-less ABI.
/// Whether the `PYRE_BH_NULL_ARG` residual-argument diagnostic is armed.
fn bh_null_arg_diag() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| std::env::var_os("PYRE_BH_NULL_ARG").is_some())
}

fn bh_call_fn_impl(callable: PyObjectRef, null_or_self: PyObjectRef, args: &[PyObjectRef]) -> i64 {
    // RPython's blackhole residual-call thunk is itself translated, so its
    // incoming GCREF arguments are shadowstack roots for the whole helper
    // call.  This Rust ABI boundary is outside that transform: the backend
    // gcmap keeps the caller objects alive, but a moving collection cannot
    // rewrite these copied native parameters.  Root them immediately and
    // dispatch only values reloaded from the forwarded slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(callable);
    pyre_object::gc_roots::pin_root(null_or_self);
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    // `eval.rs`'s `PyFrame::call` — a non-null null_or_self is the method receiver
    // (load_method_fast_path pushes `[w_descr, w_obj]`); the call proceeds
    // as `callable(null_or_self, *args)`.
    let reload_args = || {
        let null_or_self = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let mut values = Vec::with_capacity(args.len() + usize::from(!null_or_self.is_null()));
        if !null_or_self.is_null() {
            values.push(null_or_self);
        }
        values.extend(
            (0..args.len()).map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 2 + i)),
        );
        values
    };
    // `space.getexecutioncontext()` (`call.rs`'s `getexecutioncontext` →
    // TLS-pinned EC the eval loop stamps on entry).  This is the whole of the
    // caller-side state the residual needs: `bhimpl_residual_call_r_r` takes no
    // frame, and every
    // frame the dispatch below builds takes its execution context from the
    // space rather than from a caller frame.  A null here means the EC was
    // never pinned before a residual call, which is a wiring bug, so
    // fail-fast rather than building callee frames against a null context.
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_fn_impl requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    if pyre_object::gc_roots::shadow_stack_get(root_base).is_null() {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "call on null callable".to_string(),
        );
        publish_residual_call_exception(err.to_exc_object() as i64);
        return 0;
    }
    // Diagnostic (`PYRE_BH_NULL_ARG`): only `null_or_self` carries a checked
    // `PY_NULL` sentinel here (`walker_abort_if_mayforce_null_ref_arg` exempts
    // exactly that index).  A NULL in a real argument slot means the operand
    // stack the blackhole read was never published, so the callee dereferences
    // it — report the Python coordinate before that happens instead of leaving
    // a SIGSEGV inside the callee as the only evidence.
    if bh_null_arg_diag() {
        // The Python coordinate comes off the caller frame, which the dispatch
        // itself never resolves — force it here, inside the diagnostic, so an
        // unarmed run leaves the frame virtual.
        let parent_frame_ptr = unsafe { (*ec).gettopframe_raw() as *const PyFrame };
        for i in 0..args.len() {
            if !parent_frame_ptr.is_null()
                && pyre_object::gc_roots::shadow_stack_get(root_base + 2 + i).is_null()
            {
                let frame = unsafe { &*parent_frame_ptr };
                // A NULL here with a live fastlocal is an unbound blackhole
                // register (the resume never seeded the slot); a NULL fastlocal
                // means the frame itself lost the binding.
                let locals_w = locals_w!(frame);
                let shown = locals_w.len().min(8);
                let locals: Vec<Option<usize>> = (0..shown)
                    .map(|i| {
                        let v = locals_w[i];
                        (!v.is_null()).then_some(v as usize)
                    })
                    .collect();
                eprintln!(
                    "[bh-null-arg] arg[{i}]/{} NULL vsd={} base={} locals={locals:?} at {}",
                    args.len(),
                    frame.valuestackdepth,
                    frame.stack_base(),
                    frame.descr_repr(),
                );
            }
        }
    }
    // A materialized bound Method normally reaches the generic callable
    // dispatcher, which unwraps it, allocates a receiver-prefixed argument
    // vector, and dispatches the underlying Function a second time.  The
    // method fields are immutable, so an exact fixed-arity BuiltinCode can
    // take the same gateway call directly.  Keep every signature-dependent
    // shape on the generic path: only a natural arity in the fixed range with
    // an exact argument count is equivalent to the positional fast path.
    let rooted_callable = pyre_object::gc_roots::shadow_stack_get(root_base);
    let rooted_null_or_self = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let bound_builtin = unsafe {
        if !rooted_null_or_self.is_null() && pyre_interpreter::is_function(rooted_callable) {
            Some((rooted_callable, rooted_null_or_self))
        } else if rooted_null_or_self.is_null() && pyre_object::is_method(rooted_callable) {
            let function = pyre_object::w_method_get_func(rooted_callable);
            let receiver = pyre_object::w_method_get_self(rooted_callable);
            (!function.is_null() && !receiver.is_null()).then_some((function, receiver))
        } else {
            None
        }
    };
    if let Some((function, receiver)) = bound_builtin {
        let code = unsafe { pyre_interpreter::getcode(function) } as PyObjectRef;
        let positional_count = args.len() + 1;
        let exact_fixed_arity = !code.is_null()
            && unsafe { pyre_interpreter::is_builtin_code(code) }
            && positional_count <= 4
            && unsafe { pyre_interpreter::builtin_code_get_fast_natural_arity(code) as usize }
                == positional_count;
        if exact_fixed_arity {
            pyre_object::gc_roots::pin_root(code);
            pyre_object::gc_roots::pin_root(receiver);
            let code_slot = root_base + 2 + args.len();
            let receiver_slot = code_slot + 1;
            let mut call_args = [pyre_object::PY_NULL; 4];
            call_args[0] = pyre_object::gc_roots::shadow_stack_get(receiver_slot);
            for (index, slot) in call_args[1..positional_count].iter_mut().enumerate() {
                *slot = pyre_object::gc_roots::shadow_stack_get(root_base + 2 + index);
            }
            let code = pyre_object::gc_roots::shadow_stack_get(code_slot);
            return match unsafe {
                pyre_interpreter::builtin_code_call(code, &call_args[..positional_count])
            } {
                Ok(result) if !result.is_null() => result as i64,
                Ok(_) => 0,
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object() as i64);
                    0
                }
            };
        }
    }
    // llmodel.py:822 bh_call_r — calldescr.call_stub_r is callable-type-agnostic.
    // Hot path: Function callables dispatched directly here (builtin or user
    // code), matching call_user_function_plain so eval_frame_plain is used
    // and JIT is not re-entered from the blackhole.
    // Cold path: type/method/staticmethod/classmethod/callable-instance are
    // delegated to call_function_impl_result under ForcePlainEvalGuard, which
    // mirrors baseobjspace.py:1155 dispatch without re-entering the JIT.
    let callable = pyre_object::gc_roots::shadow_stack_get(root_base);
    if unsafe { is_function(callable) } {
        let code = unsafe { pyre_interpreter::getcode(callable) };
        if unsafe { pyre_interpreter::is_builtin_code(code as pyre_object::PyObjectRef) } {
            let call_args = reload_args();
            // `call_args` are raw positionals; a HOPELESS-arity Signature
            // (`*args`, optional positional) needs `_match_signature` binding
            // before the body reads its slots.  `builtin_code_call` never binds,
            // so route through the positional entry that does.
            return match pyre_interpreter::call::builtin_code_call_positional(
                code as pyre_object::PyObjectRef,
                &call_args,
            ) {
                Ok(result) if !result.is_null() => result as i64,
                Ok(_) => 0,
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object() as i64);
                    0
                }
            };
        }
        let call_args = reload_args();
        if let Some(result) = bh_call_self_recursive_portal(ec, callable, &call_args) {
            return result;
        }
        let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
        // `set_last_exec_ctx` mirrors what the portal runner does on frame
        // re-entry so user functions invoked from the residual path observe
        // the caller's execution context.  That is `ec` itself: a frame's
        // `execution_context` is the context it was built under, and the top
        // frame of this EC was built under this EC.
        pyre_interpreter::call::set_last_exec_ctx(ec);
        // `blackhole.py:1225 bhimpl_residual_call_*` is opaque to the TRACE,
        // not to the JIT: `cpu.bh_call_*(func, ...)` runs the translated
        // callee, and a callee whose graph reaches a `jit_merge_point`
        // (`execute_frame` does) enters the JIT normally.  Pinning the callee
        // and its whole subtree to `eval_frame_plain` here left a hot inner
        // loop interpreted for as long as its caller's loop was compiled.
        // `bhimpl_recursive_call_*` (`blackhole.py:1095-1132`) is not the
        // only way back to the portal — it is the form the codewriter emits
        // when the callee is *statically* the portal graph.
        //
        // Re-entrant TRACING stays blocked where upstream blocks it, on the
        // green key (`warmstate.py:473-477` JC_TRACING, mirrored by
        // `maybe_compile_and_run`'s `driver.is_tracing()`).
        //
        // Letting the callee re-enter the JIT also lets a nested compiled /
        // blackhole execution run inside this helper, and that nesting writes
        // the very cells this helper publishes to: `BH_LAST_EXC_VALUE` and the
        // backend `_store_exception` pair. Upstream cannot alias them — the
        // raise lives in `metainterp.last_exc_value`, a field of the
        // MetaInterp instance that owns the call, and `llmodel.py:194
        // _store_exception` is read back by the same `bh_call_*` that armed
        // it. Park the caller's pair across the nested run so this helper's
        // own outcome is the only thing it publishes; a nested raise the
        // callee handled internally would otherwise be read by the caller's
        // `GUARD_NO_EXCEPTION` as a spurious pending exception (the failure
        // `drain_backend_jit_exc` names for the walker's snapshot side).
        let parked = park_residual_call_exception();
        let result = pyre_interpreter::call::call_user_function_with_ctx(ec, callable, &call_args);
        unpark_residual_call_exception(parked);
        pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
        return match result {
            Ok(result) => result as i64,
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object() as i64);
                0
            }
        };
    }
    // Cold path: type/method/staticmethod/classmethod/callable-instance.
    // Ensure LAST_EXEC_CTX reflects the calling context before delegating to
    // `call_function_impl_result`. `type_descr_call_impl` →
    // `call_user_function_with_args` reads LAST_EXEC_CTX as the fallback
    // execution context for `__new__`/`__init__` (call.rs);
    // without this pin it would use whatever frame last entered
    // `eval_frame_*`, which is not guaranteed to be the blackhole caller.
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let _plain_guard = pyre_interpreter::call::force_plain_eval();
    let callable = pyre_object::gc_roots::shadow_stack_get(root_base);
    let call_args = reload_args();
    let result = pyre_interpreter::call::call_function_impl_result(callable, &call_args);
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// CALL_FUNCTION_EX residual (`call_function_ex(callable, self_or_null,
/// starargs, kwargs_or_null)` HLOp → `residual_call_r_r`).  Unpacks the
/// `*` iterable and merges the `**` mapping through the shared
/// `call::call_function_ex`, then dispatches.  Carries no frame, like
/// [`bh_call_fn_impl`]: `pyopcode.py:1429 CALL_FUNCTION_EX` settles the
/// C-profile question against its own frame and then calls the frameless
/// `space.call_args(w_function, args)`, which is what a residual reaches.
/// The nested Python call runs under `force_plain_eval` (blackhole.py:1225
/// `bhimpl_residual_call_*` is an opaque CPU call — no JIT re-entry).
/// MayForce: unpacking an arbitrary iterable / mapping and the dispatched
/// call may run Python.
pub extern "C" fn bh_call_function_ex_fn(
    callable: i64,
    self_or_null: i64,
    starargs: i64,
    kwargs_or_null: i64,
) -> i64 {
    let ec = pyre_interpreter::call::getexecutioncontext();
    assert!(
        !ec.is_null(),
        "bh_call_function_ex_fn requires a pinned execution context from \
         getexecutioncontext(); the eval loop must pin the execution context \
         before any residual call"
    );
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    pyre_interpreter::call::set_last_exec_ctx(ec);
    let result = {
        let _plain_guard = pyre_interpreter::call::force_plain_eval();
        pyre_interpreter::call::call_function_ex_in_ctx(
            ec,
            callable as PyObjectRef,
            self_or_null as PyObjectRef,
            starargs as PyObjectRef,
            kwargs_or_null as PyObjectRef,
        )
    };
    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// `_load_global` residual (pyopcode.py:958-969).  Resolves the namespace via
/// the executing frame's `get_w_globals_storage()` when the live frame OWNS this
/// `w_code` (`frame.pycode == w_code`) — honoring an `exec(code, ns)`
/// frame-specific namespace — and falls back to the callee's own promoted
/// `w_code` globals otherwise (an inlined / chained callee's frame register
/// aliases an outer frame, so it is not this code's frame).  The
/// `namespace_ptr` operand is ignored for the same aliasing reason; it
/// survives only as the cell-fold recogniser's hint.  The live frame is also
/// passed so `self.get_builtin()` works in compiled residual-call paths as
/// well as blackhole.  namei is the raw oparg from LOAD_GLOBAL:
/// name_idx = namei >> 1.
pub extern "C" fn bh_load_global_fn(
    namespace_ptr: i64,
    w_code_ptr: i64,
    frame_ptr: i64,
    namei: i64,
) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    let raw = namei as usize;
    let idx = raw >> 1;

    if idx >= code.names.len() {
        return 0;
    }

    let varname = code.names[idx].as_ref();
    let _ = namespace_ptr;
    let parent_frame_ptr = frame_ptr as *const PyFrame;
    // pypy/interpreter/pyopcode.py:958-969 `_load_global`:
    //   w_value = self.space.finditem_str(self.get_w_globals_storage(), varname)
    //   if w_value is None:
    //       w_value = self.get_builtin().getdictvalue(self.space, varname)
    //       if w_value is None:
    //           self._load_global_failed(w_varname)
    //
    // `self.get_w_globals()` (pyframe.py:49) is the executing frame's own
    // globals object: the per-frame globals an `exec(code, ns)` installs,
    // falling back to the code's bound globals when there is no override.
    // Use it whenever the live frame OWNS this `w_code`
    // (`frame.pycode == w_code`), so a compiled `LOAD_GLOBAL` resolves
    // against the executing frame's globals — not the code's original module
    // dict — exactly as the interpreter does.
    //
    // A frame that does NOT own this `w_code` is an aliased OUTER frame on a
    // chained blackhole / inlined-callee resume (the same aliasing makes the
    // `namespace_ptr` operand unusable, hence ignored above).  Resolve from
    // the callee's own promoted `w_code` constant: reading `w_globals`
    // from it yields the current module dict object the const-folding
    // `frontend_global_flow_value` resolved statically, but live, so a
    // relocated dict (a growing `memo`) is followed instead of dangling.
    let w_globals = if !parent_frame_ptr.is_null()
        && unsafe { (*parent_frame_ptr).pycode } as usize == w_code_ptr as usize
    {
        unsafe { (*parent_frame_ptr).get_w_globals() }
    } else {
        unsafe { pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef) }
    };
    if !w_globals.is_null() {
        match pyre_interpreter::baseobjspace::finditem_str(w_globals, varname) {
            Ok(Some(w_value)) => return w_value as i64,
            Ok(None) => {}
            Err(mut err) => {
                let exc_obj = err.to_exc_object();
                publish_residual_call_exception(exc_obj as i64);
                return 0;
            }
        }
    }

    // Residual helper adaptation: `self` is the live portal frame passed as
    // an explicit Ref argument, so `self.get_builtin()` maps to
    // PyFrame::get_builtin() without relying on blackhole-only TLS.
    if !parent_frame_ptr.is_null() {
        let w_builtin = unsafe { (*parent_frame_ptr).get_builtin() };
        if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if !w_dict.is_null() {
                match pyre_interpreter::baseobjspace::finditem_str(w_dict, varname) {
                    Ok(Some(w_value)) => return w_value as i64,
                    Ok(None) => {}
                    Err(mut err) => {
                        let exc_obj = err.to_exc_object();
                        publish_residual_call_exception(exc_obj as i64);
                        return 0;
                    }
                }
            }
        }
    }

    // pyopcode.py:970 `_load_global_failed`: raise NameError.
    let mut err = pyre_interpreter::PyError::new(
        pyre_interpreter::PyErrorKind::NameError,
        format!("name '{}' is not defined", varname),
    );
    let exc_obj = err.to_exc_object();
    publish_residual_call_exception(exc_obj as i64);
    0
}

/// LOAD_FROM_DICT_OR_GLOBALS residual (`load_from_dict_or_globals` HLOp →
/// `residual_call_ir_r`).  Mirrors `eval.rs::load_from_dict_or_globals`:
/// try `getitem(dict, name)` on the popped mapping first, then fall
/// back to the live frame's globals, else NameError.  `namei` is a direct
/// `code.names` index (no LOAD_GLOBAL push-null low-bit shift).
///
/// GC-safety: the globals are read from the LIVE frame when it owns this
/// `w_code` (`frame.pycode == w_code`), matching `bh_load_global_fn`, so a
/// relocated module dict is followed rather than a const-folded dangling
/// pointer.  A user `__getattr__`/`__getitem__` on the mapping may run
/// Python (`MayForce`).
pub extern "C" fn bh_load_from_dict_or_globals_fn(
    dict_ptr: i64,
    w_code_ptr: i64,
    frame_ptr: i64,
    namei: i64,
) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    let idx = namei as usize;
    if idx >= code.names.len() {
        return 0;
    }
    let varname = code.names[idx].as_ref();
    let dict = dict_ptr as pyre_object::PyObjectRef;

    // CPython/PyPy LOAD_FROM_DICT_OR_GLOBALS uses mapping subscription, not
    // attribute lookup.  Preserve __getitem__/__missing__ and propagate every
    // exception except the KeyError that selects the globals fallback.
    let key = pyre_object::w_str_new(varname);
    match pyre_interpreter::baseobjspace::getitem(dict, key) {
        Ok(val) => return val as i64,
        Err(err) if matches!(err.kind, pyre_interpreter::PyErrorKind::KeyError) => {}
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            return 0;
        }
    }

    // Fall back to the live frame's globals (GC-safe when the frame owns
    // this w_code; else the promoted w_code's own globals).
    let parent_frame_ptr = frame_ptr as *const PyFrame;
    let w_globals = if !parent_frame_ptr.is_null()
        && unsafe { (*parent_frame_ptr).pycode } as usize == w_code_ptr as usize
    {
        unsafe { (*parent_frame_ptr).get_w_globals() }
    } else {
        unsafe { pyre_interpreter::w_code_get_w_globals(w_code_ptr as pyre_object::PyObjectRef) }
    };
    if !w_globals.is_null() {
        match pyre_interpreter::baseobjspace::finditem_str(w_globals, varname) {
            Ok(Some(val)) => return val as i64,
            Ok(None) => {}
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object() as i64);
                return 0;
            }
        }
    }

    // `load_from_dict_or_globals` (eval.rs) finishes through
    // `load_global_value`, whose second leg is
    // `self.get_builtin().getdictvalue(space, varname)` (`pyopcode.py:979`).
    // This residual stopped after the globals leg, so a builtin name reached
    // here as a bogus `NameError`.
    //
    // A null frame is an anticipated input — the globals leg above already
    // treats it as one — so the builtin namespace is resolved from the globals
    // through `pick_builtin_obj_checked`, the same resolver whose result
    // `frame.get_builtin()` caches. Reading `globals["__builtins__"]` raw is
    // not equivalent: `exec`/`eval` plant the builtins *dict* rather than the
    // module (3.14 `PyEval_GetBuiltins`), and the `is_module` test below would
    // reject that spelling and report a bogus `NameError` for every builtin
    // name seen by a function created under an exec'd namespace.
    // `try_walker_load_global_cell_fold` declines the fold in that case and
    // leaves the name to this residual, so the handling has to live here.
    let w_builtin = if !parent_frame_ptr.is_null() {
        unsafe { (*parent_frame_ptr).get_builtin() }
    } else {
        match pyre_interpreter::baseobjspace::pick_builtin_obj_checked(
            w_globals,
            pyre_interpreter::call::take_last_exec_ctx(),
        ) {
            Ok(w_builtin) => w_builtin,
            Err(mut err) => {
                publish_residual_call_exception(err.to_exc_object() as i64);
                return 0;
            }
        }
    };
    if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if !w_dict.is_null() {
            match pyre_interpreter::baseobjspace::finditem_str(w_dict, varname) {
                Ok(Some(val)) => return val as i64,
                Ok(None) => {}
                Err(mut err) => {
                    publish_residual_call_exception(err.to_exc_object() as i64);
                    return 0;
                }
            }
        }
    }

    let mut err = pyre_interpreter::PyError::name_error_with_name(
        format!("name '{varname}' is not defined"),
        varname,
    );
    publish_residual_call_exception(err.to_exc_object() as i64);
    0
}

/// `LOAD_ATTR` / method-form `LOAD_ATTR` residual for the standalone
/// (blackhole / deopt) per-CodeObject jitcode.  pyre's codewriter cannot
/// rtype `getattr` into `getfield_gc` (`rclass.py:838 rtype_getattr`
/// rewrites `getattr` → `getfield_gc` after rtyping; pyre has no rtyper),
/// so the per-CodeObject jitcode lowers `LOAD_ATTR` to this residual call
/// rather than `abort_permanent`. The full-body walker traces the emitted
/// residual call.
///
/// Mirrors the interpreter `eval.rs` `load_attr` / `load_method` getattr
/// fallback (`baseobjspace::getattr_str`): returns the (possibly bound)
/// attribute.  For the method form the codewriter pushes the result plus
/// a `PY_NULL` self-slot; the bound method already carries `self`, so the
/// following `CALL` (`eval.rs`'s `PyFrame::call`) invokes it with
/// `null_or_self == NULL`, producing the same call as the unbound
/// `[w_descr, self]` fast path.  On `AttributeError` it sets
/// `BH_LAST_EXC_VALUE` and returns 0,
/// matching `bh_load_global_fn`'s NameError path.  `w_name` is the
/// interned immortal str constant the flatten driver lowers the getattr
/// HLOp's name operand to (`flatten_constant_operand` → `box_str_constant`),
/// the same interned-name ABI as `_pure_lookup_where_with_method_cache`.
pub extern "C" fn bh_getattr_fn(obj: i64, w_name: i64) -> i64 {
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    let res = pyre_interpreter::baseobjspace::getattr_str(obj as pyre_object::PyObjectRef, name);
    match res {
        Ok(w_value) => w_value as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// Resolve `getattr(obj, code.names[name_idx])` for blackhole LOAD_ATTR
/// resume — the attribute / unbound-method half of LOOKUP_METHOD.  Mirrors
/// `PyFrame::load_method`'s `getattr` call so the blackhole reproduces the
/// interpreter's attribute lookup exactly (descriptor `__get__`, super
/// proxies, weakref forcing).  On error stashes the exception in
/// `BH_LAST_EXC_VALUE` and returns 0 (the trailing `-live-` lets the
/// blackhole route it through the except handler), matching
/// [`bh_load_global_fn`].
pub extern "C" fn bh_load_attr_fn(obj: i64, w_code_ptr: i64, name_idx: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    // `name_idx` is a `co_names` index baked into the residual call by the
    // codewriter from the originating LOAD_ATTR oparg, so it is in range for
    // the code object loaded out of the resume frame.  The bound is a codegen
    // invariant rather than a runtime-reachable error (a negative index would
    // wrap to a huge `usize` and trip the same check), so assert it in debug
    // and degrade to a null result in release.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_load_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    match pyre_interpreter::baseobjspace::getattr_str(obj as pyre_object::PyObjectRef, name) {
        Ok(attr) => attr as i64,
        Err(mut err) => {
            // Publish the raise into BOTH the blackhole `BH_LAST_EXC_VALUE` and
            // the backend `_store_exception` cells (`publish_residual_call_exception`).
            // The LOAD_ATTR residual runs under the blackhole interpreter AND
            // inside a compiled trace; writing only `BH_LAST_EXC_VALUE` leaves a
            // compiled trace's `GUARD_NO_EXCEPTION` reading a stale 0, so the
            // guard wrongly passes and the NULL result flows to the consumer (a
            // raising property getter in a compiled loop is silently swallowed).
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// STORE_ATTR residual (`store_attr` HLOp → `residual_call_ir_v`).  The
/// symmetric counterpart of [`bh_load_attr_fn`]: resolves the attribute
/// name from the resume frame's code object via `name_idx` and runs the
/// generic `setattr_str` (may invoke user `__setattr__` → `MayForce`).
/// Void result, so always returns 0; an exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`.
pub extern "C" fn bh_store_attr_fn(obj: i64, value: i64, w_code_ptr: i64, name_idx: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    // Same `co_names`-index codegen invariant as `bh_load_attr_fn`.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_store_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    if let Err(mut err) = pyre_interpreter::baseobjspace::setattr_str(
        obj as pyre_object::PyObjectRef,
        name,
        value as pyre_object::PyObjectRef,
    ) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
    }
    0
}

/// DELETE_ATTR residual (`delete_attr` HLOp → `residual_call_ir_v`).
/// Resolves the `co_names` name through the jitcode's own code object
/// (same invariant as `bh_store_attr_fn`) and runs `del obj.name` through
/// `baseobjspace::delattr_str`.  A user `__delattr__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_delete_attr_fn(obj: i64, w_code_ptr: i64, name_idx: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_delete_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    if let Err(mut err) =
        pyre_interpreter::baseobjspace::delattr_str(obj as pyre_object::PyObjectRef, name)
    {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
    }
    0
}

/// IMPORT_NAME residual (`import_name` HLOp → `residual_call_ir_r`).
/// Resolves the module name from the jitcode's own code object via
/// `name_idx` (same `co_names` invariant as `bh_load_attr_fn`), fetches
/// `__import__` from the threaded frame's builtins, and calls it with the
/// frame's globals and locals. Importing a module may run its
/// top-level Python (`MayForce`); on error the exception is published
/// through `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the
/// call returns 0.  `fromlist` and `level` are the two popped operands
/// (`eval.rs import_name`: `fromlist = pop()`, `level = pop()`).
pub extern "C" fn bh_import_name_fn(
    fromlist: i64,
    level: i64,
    w_code_ptr: i64,
    frame_ptr: i64,
    name_idx: i64,
) -> i64 {
    let w_code = w_code_ptr as pyre_object::PyObjectRef;
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_import_name_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    let frame = frame_ptr as *mut PyFrame;
    debug_assert!(!frame.is_null(), "IMPORT_NAME requires a live frame");
    if frame.is_null() {
        // IMPORT_NAME produces a module or raises; it never yields a null
        // result.  A null frame cannot honour that, so fail closed by
        // publishing an exception for the trailing `GuardNoException`
        // instead of returning a bare 0 the guard would accept.
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::SystemError,
            "IMPORT_NAME residual received a null frame",
        );
        publish_residual_call_exception(err.to_exc_object() as i64);
        return 0;
    }
    match pyre_interpreter::importing::import_name(
        unsafe { &mut *frame },
        name,
        fromlist as pyre_object::PyObjectRef,
        level as pyre_object::PyObjectRef,
    ) {
        Ok(module) => module as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// IMPORT_FROM residual (`import_from` HLOp → `residual_call_ir_r`).
/// Resolves the attribute name from the jitcode's own code object via
/// `name_idx` (same `co_names` invariant as `bh_load_attr_fn`) and runs
/// `importing::import_from(module, name, ec)` — first the module's
/// namespace dict, then a submodule-import fallback (which may run a
/// module's top-level Python → `MayForce`) — with the TLS-pinned
/// execution context.  `module` is the peeked TOS (IMPORT_FROM does not
/// pop it).  On error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_import_from_fn(module: i64, w_code_ptr: i64, name_idx: i64) -> i64 {
    let w_code = w_code_ptr as pyre_object::PyObjectRef;
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_import_from_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    let ec = pyre_interpreter::call::getexecutioncontext();
    match pyre_interpreter::importing::import_from(module as pyre_object::PyObjectRef, name, ec) {
        Ok(attr) => attr as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// LOAD_SUPER_ATTR residual (`load_super_attr` HLOp → `residual_call_ir_r`).
/// Resolves the attribute name from the jitcode's code object via `name_idx`
/// (same `co_names` invariant as `bh_load_attr_fn`), calls the actual global
/// `super` value with zero or two arguments, and runs `getattr` (both calls may
/// run Python → `MayForce`).  The zero-argument builtin path reads the
/// explicitly threaded red `frame`, matching PyPy's traced
/// `ec.gettopframe()` dependency. Returns the raw resolved attribute; the
/// `is_method` form
/// post-processes it through [`bh_super_attr_unwrap_fn`].  On error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_load_super_attr_fn(
    global_super: i64,
    self_obj: i64,
    cls: i64,
    frame: i64,
    w_code_ptr: i64,
    name_idx: i64,
    is_two_arg: i64,
) -> i64 {
    let w_code = w_code_ptr as pyre_object::PyObjectRef;
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject)
    };
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_load_super_attr_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return 0;
    }
    let name = code.names[idx].as_ref();
    let global_super = global_super as pyre_object::PyObjectRef;
    let self_obj = self_obj as pyre_object::PyObjectRef;
    let cls = cls as pyre_object::PyObjectRef;
    let proxy = if is_two_arg != 0 {
        pyre_interpreter::call::call_function_impl_result(global_super, &[cls, self_obj])
    } else if pyre_interpreter::builtins::is_builtin_super_type(global_super) {
        // `descriptor.py W_Super.descr_init` traces
        // `space.getexecutioncontext().gettopframe()` into the active MIFrame.
        // The JIT threads that frame as a red operand: an inlined callee must
        // read its own `self` / `__class__` cells, never the outer portal's.
        pyre_interpreter::builtins::builtin_super_from_frame(
            frame as usize as *mut pyre_interpreter::PyFrame,
        )
    } else {
        // CPython 3.14 makes the global `super` binding authoritative.  A
        // shadowing callable still receives zero arguments exactly as the
        // interpreter path does.
        pyre_interpreter::call::call_function_impl_result(global_super, &[])
    };
    let result = proxy.and_then(|proxy| pyre_interpreter::baseobjspace::getattr_str(proxy, name));
    match result {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// LOAD_SUPER_ATTR method-form unwrap (`super_attr_unwrap` HLOp →
/// `residual_call_ir_r`).  Pure function of the raw attribute resolved by
/// [`bh_load_super_attr_fn`] — `which == 0` yields the func slot, `which ==
/// 1` the self slot.  When the attribute is a bound method, unwraps it to
/// `(func, receiver)`; otherwise (staticmethod / classmethod) yields
/// `(result, NULL)`.  Infallible and idempotent, so safe under the walk /
/// replay double-execution seam.
pub extern "C" fn bh_super_attr_unwrap_fn(raw: i64, which: i64) -> i64 {
    let result = raw as pyre_object::PyObjectRef;
    if unsafe { pyre_object::is_method(result) } {
        if which == 0 {
            unsafe { pyre_object::w_method_get_func(result) as i64 }
        } else {
            unsafe { pyre_object::w_method_get_self(result) as i64 }
        }
    } else if which == 0 {
        raw
    } else {
        pyre_object::PY_NULL as i64
    }
}

/// BINARY_SLICE residual (`binary_slice` HLOp → `residual_call_r_r`).
/// Computes `obj[start:stop]` through the shared
/// `runtime_ops::binary_slice_values` (the same code the interpreter's
/// `binary_slice` runs).  A `__getitem__` on a user object may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0, matching [`bh_load_attr_fn`].
pub extern "C" fn bh_binary_slice_fn(obj: i64, start: i64, stop: i64) -> i64 {
    match pyre_interpreter::runtime_ops::binary_slice_values(
        obj as pyre_object::PyObjectRef,
        start as pyre_object::PyObjectRef,
        stop as pyre_object::PyObjectRef,
    ) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// STORE_SLICE residual (`store_slice` HLOp → `residual_call_r_v`).
/// Runs `obj[start:stop] = value` through the shared
/// `runtime_ops::store_slice_values` (the same code the interpreter's
/// `store_slice` runs — builds a `slice(start, stop, None)` and dispatches
/// `setitem`).  A user `__setitem__` or slice-bound `__index__` may run
/// Python and force virtualizables (`MayForce`).  Void result, so always
/// returns 0; on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`, matching
/// `bh_delete_subscr_fn`.
pub extern "C" fn bh_store_slice_fn(obj: i64, start: i64, stop: i64, value: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::runtime_ops::store_slice_values(
        obj as pyre_object::PyObjectRef,
        start as pyre_object::PyObjectRef,
        stop as pyre_object::PyObjectRef,
        value as pyre_object::PyObjectRef,
    ) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
    }
    0
}

/// DELETE_SUBSCR residual (`delete_subscr` HLOp → `residual_call_r_v`).
/// Runs `del obj[index]` through the shared `baseobjspace::delitem` (the
/// same code the interpreter's `delete_subscript` runs).  A `__delitem__`
/// on a user object may run Python (`MayForce`).  Void result, so always
/// returns 0; on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException`, matching
/// `bh_store_subscr_fn`.
pub extern "C" fn bh_delete_subscr_fn(obj: i64, index: i64) -> i64 {
    // Diagnostic (#24 wasm-Linux miscompile): the delete target and index Refs
    // must be non-null at the residual boundary. A null `items` on the wasm
    // resume path would silently mis-route — `is_list(null)` is false without a
    // deref, so the slice branch is skipped and the operand reads guest offset 0.
    // Trap loudly to pin a null `items`/`index` at the residual entry.
    assert!(
        obj != 0 && index != 0,
        "bh_delete_subscr_fn: null residual operand (#24 wasm): obj={obj:#x} index={index:#x}"
    );
    if let Err(mut err) = pyre_interpreter::baseobjspace::delitem(
        obj as pyre_object::PyObjectRef,
        index as pyre_object::PyObjectRef,
    ) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
    }
    0
}

/// LIST_EXTEND residual (`list_extend` HLOp → `residual_call_r_v`).
/// Runs `list.extend(iterable)` through the shared
/// `opcode_ops::list_extend_value` (the same code the interpreter's
/// `list_extend` runs); `list` is peeked and mutated in place.  A
/// non-iterable operand or a user iterator running Python can raise
/// (`MayForce`).  Void result, so always returns 0; on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException`, matching `bh_delete_subscr_fn`.
pub extern "C" fn bh_list_extend_fn(list: i64, iterable: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::list_extend_value(
        list as pyre_object::PyObjectRef,
        iterable as pyre_object::PyObjectRef,
    ) {
        let exc_obj = err.to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
    }
    0
}

/// SET_ADD residual (`set_add` HLOp → `residual_call_r_v`).  Runs
/// `set.add(value)` (or `list.append`) through the shared
/// `opcode_ops::set_add_value`; `set` is peeked and mutated in place.
/// A user `__hash__`/`__eq__` can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_set_add_fn(set: i64, value: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::set_add_value(
        set as pyre_object::PyObjectRef,
        value as pyre_object::PyObjectRef,
    ) {
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    0
}

/// SET_UPDATE residual (`set_update` HLOp → `residual_call_r_v`).  Runs
/// `set.update(iterable)` (or `list.extend`) through the shared
/// `opcode_ops::set_update_value`; `set` is peeked and mutated in place.
/// A user iterator / `__hash__` can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_set_update_fn(set: i64, iterable: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::set_update_value(
        set as pyre_object::PyObjectRef,
        iterable as pyre_object::PyObjectRef,
    ) {
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    0
}

/// DICT_UPDATE residual (`dict_update` HLOp → `residual_call_r_v`).  Runs
/// `dict.update(source)` with the ismapping gate through the shared
/// `opcode_ops::dict_update_value`; `dict` is peeked and mutated in
/// place.  A `keys()`/`__getitem__`/`__hash__` can run Python
/// (`MayForce`).  Void result.
pub extern "C" fn bh_dict_update_fn(dict: i64, source: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::dict_update_value(
        dict as pyre_object::PyObjectRef,
        source as pyre_object::PyObjectRef,
    ) {
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    0
}

/// MAP_ADD residual (`map_add` HLOp → `residual_call_r_v`).  Runs
/// `dict[key] = value` through the shared `opcode_ops::map_add_value`;
/// `dict` is peeked and mutated in place.  A user key `__hash__`/`__eq__`
/// can run Python (`MayForce`).  Void result.
pub extern "C" fn bh_map_add_fn(dict: i64, key: i64, value: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::map_add_value(
        dict as pyre_object::PyObjectRef,
        key as pyre_object::PyObjectRef,
        value as pyre_object::PyObjectRef,
    ) {
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    0
}

/// DICT_MERGE residual (`dict_merge` HLOp → `residual_call_r_v`).  Runs
/// `dict.update(source)` with duplicate-key checks through the shared
/// `opcode_ops::dict_merge_value`; `dict` is peeked and mutated in place.
/// `w_callable` is the peeked callable used only for error-message
/// prefixes.  A `keys()`/`__getitem__`/`__hash__` can run Python
/// (`MayForce`).  Void result.
pub extern "C" fn bh_dict_merge_fn(dict: i64, source: i64, w_callable: i64) -> i64 {
    if let Err(mut err) = pyre_interpreter::opcode_ops::dict_merge_value(
        dict as pyre_object::PyObjectRef,
        source as pyre_object::PyObjectRef,
        w_callable as pyre_object::PyObjectRef,
    ) {
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    0
}

/// Compute the LOOKUP_METHOD `null_or_self` for blackhole LOAD_ATTR resume,
/// given the already-resolved `attr` from [`bh_load_attr_fn`].  Delegates to
/// the shared `compute_load_method_bound`, a pure MRO inspection that never
/// re-invokes a descriptor, so it cannot raise (returns `PY_NULL` when no
/// receiver should be prepended).
pub extern "C" fn bh_load_method_self_fn(
    obj: i64,
    attr: i64,
    w_code_ptr: i64,
    name_idx: i64,
) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    // Same `co_names`-index codegen invariant as `bh_load_attr_fn`; the
    // release-path `PY_NULL` is also the legitimate "no self prepended"
    // result, so the debug assert is what catches an invalid index.
    let idx = name_idx as usize;
    debug_assert!(
        idx < code.names.len(),
        "bh_load_method_self_fn name_idx {idx} out of range ({} names) — codegen invariant",
        code.names.len()
    );
    if idx >= code.names.len() {
        return pyre_object::PY_NULL as i64;
    }
    let name = code.names[idx].as_ref();
    pyre_interpreter::eval::compute_load_method_bound(
        obj as pyre_object::PyObjectRef,
        attr as pyre_object::PyObjectRef,
        name,
    ) as i64
}

#[inline]
fn load_special_method_name(method_kind: i64) -> &'static str {
    match method_kind {
        0 => "__enter__",
        1 => "__exit__",
        2 => "__aenter__",
        _ => "__aexit__",
    }
}

/// Resolve the LOAD_SPECIAL method named by `method_kind`.  Mirrors the
/// interpreter's `load_special` through the shared `load_special_resolve`:
/// `_PyObject_LookupSpecial` looks the method up on the type only, bypassing
/// the instance dict and any `__getattribute__` override, so a traced
/// `with`/`async with` calls the same method the interpreter would.  Returns
/// the bound method; [`bh_load_special_self_fn`] therefore reports a NULL self.
pub extern "C" fn bh_load_special_fn(obj: i64, method_kind: i64) -> i64 {
    let name = load_special_method_name(method_kind);
    match pyre_interpreter::baseobjspace::load_special_resolve(
        obj as pyre_object::PyObjectRef,
        name,
    ) {
        Ok(bound) => bound as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// The LOAD_SPECIAL `null_or_self` value.  [`bh_load_special_fn`] returns the
/// method already bound through the descriptor's `__get__`, so — as in the
/// interpreter's `load_special` — the pushed pair carries a NULL self.
pub extern "C" fn bh_load_special_self_fn(_obj: i64, _attr: i64, _method_kind: i64) -> i64 {
    pyre_object::PY_NULL as i64
}

/// WITH_EXCEPT_START residual.  The generated JitCode carries the five
/// handler-stack values, and this helper performs the same `__exit__` call as
/// `PyFrame::with_except_start` from the three semantic operands.
pub extern "C" fn bh_with_except_start_fn(exit_func: i64, exit_self: i64, val: i64) -> i64 {
    let result = pyre_interpreter::eval::with_except_start_values(
        exit_func as pyre_object::PyObjectRef,
        exit_self as pyre_object::PyObjectRef,
        val as pyre_object::PyObjectRef,
    );
    if result.is_null() {
        let mut err = pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| pyre_interpreter::PyError::type_error("__exit__ failed"));
        publish_residual_call_exception(err.to_exc_object() as i64);
    }
    result as i64
}

/// `LOAD_NAME` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py:945-955 LOAD_NAME` — when
/// `getorcreatedebug().w_locals is not get_w_globals_storage()` the lookup
/// tries `finditem_str(w_locals, varname)` first, then falls through
/// to the `LOAD_GLOBAL` globals → builtins chain.  Delegates to the
/// interpreter trait impl (`eval.rs load_name_checked_value`) so the
/// blackhole re-execution and the interpreter share one lookup order.
/// The full-body walker traces the emitted residual call. `w_name` is the
/// interned immortal str constant the flatten driver lowers the
/// `load_name` HLOp's name operand to (`box_str_constant`); `namei`
/// feeds the `pycode._globals_caches[nameindex]` global cache
/// (`celldict.py:292`).  On error it sets `BH_LAST_EXC_VALUE` and
/// returns 0, matching `bh_load_global_fn`'s NameError path.
pub extern "C" fn bh_load_name_fn(frame_ptr: i64, w_name: i64, namei: i64) -> i64 {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        frame_ptr != 0,
        "bh_load_name_fn requires a non-null PyFrame; every LOAD_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    match frame.load_name_checked_value(name, namei as usize) {
        Ok(w_value) => w_value as i64,
        Err(mut err) => {
            // Publish into BOTH the blackhole `BH_LAST_EXC_VALUE` and the
            // backend `_store_exception` cells: LOAD_NAME lowers into the
            // full-body-walk compiled trace (a handler-bearing body skips the
            // cell-fold), where the following `GUARD_NO_EXCEPTION` reads the
            // backend cells — writing only `BH_LAST_EXC_VALUE` lets the guard
            // pass on a stale 0 and a raising LOAD_NAME is silently swallowed.
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// `STORE_NAME` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py:855-859 STORE_NAME` —
/// `setitem_str(getorcreatedebug().w_locals, varname, w_newvalue)`.
/// Delegates to the interpreter trait impl (`eval.rs
/// store_name_value`), which routes non-dict mapping locals through
/// `space.setitem` and module/class namespaces through the dict
/// strategy.  Same blackhole-only execution contract and `w_name` ABI
/// as `bh_load_name_fn`.  `STORE_NAME` carries no nameindex-keyed
/// cache upstream, so the trait's `nameindex` argument is passed as 0.
/// Returns 1 on success; on error it sets `BH_LAST_EXC_VALUE` and
/// returns 0, matching `bh_store_subscr_fn`.
pub extern "C" fn bh_store_name_fn(frame_ptr: i64, w_name: i64, value: i64) -> i64 {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        frame_ptr != 0,
        "bh_store_name_fn requires a non-null PyFrame; every STORE_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    match frame.store_name_value(name, 0, value as pyre_object::PyObjectRef) {
        Ok(()) => 1,
        Err(mut err) => {
            // Publish into both exception cells: STORE_NAME lowers into the
            // full-body-walk compiled trace, whose `GUARD_NO_EXCEPTION` reads
            // the backend `_store_exception` cells; writing only
            // `BH_LAST_EXC_VALUE` would let a raising `__setitem__` on a
            // mapping-locals namespace be silently swallowed.
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// `STORE_GLOBAL` residual for the standalone (blackhole / deopt)
/// per-CodeObject jitcode.  `pyopcode.py:567 STORE_GLOBAL` writes the
/// value directly into `w_globals`, bypassing `w_locals`.  Delegates to
/// the interpreter trait impl (`eval.rs store_global_value`), which
/// routes through `w_dict_setitem_str` on the eagerly-resolved
/// `w_globals` (or the back-mirror dict storage when null).  Same
/// blackhole-only execution contract and `w_name` ABI as
/// `bh_store_name_fn`.  `STORE_GLOBAL` carries no nameindex-keyed cache,
/// so the trait's `nameindex` argument is passed as 0.  Returns 1 on
/// success; on error it sets `BH_LAST_EXC_VALUE` and returns 0.
pub extern "C" fn bh_store_global_fn(frame_ptr: i64, w_name: i64, value: i64) -> i64 {
    use pyre_interpreter::pyopcode::NamespaceOpcodeHandler;
    assert!(
        frame_ptr != 0,
        "bh_store_global_fn requires a non-null PyFrame; every STORE_GLOBAL emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    match frame.store_global_value(name, 0, value as pyre_object::PyObjectRef) {
        Ok(()) => 1,
        Err(mut err) => {
            // Publish into both exception cells, matching the STORE_NAME arm.
            // `store_global_value` is infallible today (this arm is dead), but
            // STORE_GLOBAL lowers into the compiled trace with a following
            // `GUARD_NO_EXCEPTION` that reads the backend cells, so should a
            // fallible path appear the raise must not be swallowed.
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// DELETE_NAME residual using the frame receiver and interned-name ABI.
/// pyopcode.py DELETE_NAME deletes from `w_locals` and raises `NameError`
/// when the binding is absent.
pub extern "C" fn bh_delete_name_fn(frame_ptr: i64, w_name: i64) -> i64 {
    use pyre_interpreter::pyopcode::OpcodeStepExecutor;
    assert!(
        frame_ptr != 0,
        "bh_delete_name_fn requires a non-null PyFrame; every DELETE_NAME emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    match frame.delete_name(name) {
        Ok(()) => 1,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// `LOAD_LOCALS` residual using the frame receiver.
/// `pyopcode.py:793-794` — `pushvalue(getorcreatedebug().w_locals)`, which
/// must hand back the frame's own mapping so a metaclass `__prepare__` result
/// keeps its type. Infallible, so unlike the name residuals there is no
/// exception-publishing arm.
pub extern "C" fn bh_load_locals_fn(frame_ptr: i64) -> i64 {
    assert!(
        frame_ptr != 0,
        "bh_load_locals_fn requires a non-null PyFrame; every LOAD_LOCALS emit \
         site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    frame.get_or_create_w_locals() as i64
}

/// `LOAD_BUILD_CLASS` residual using the frame receiver.
/// `pyopcode.py:866-870` — `get_builtin().getdictvalue('__build_class__')`,
/// a NameError when the selected builtin mapping has no entry. Delegates to
/// `eval.rs load_build_class_value` so the interpreter and the residual share
/// one lookup, the same contract `bh_load_name_fn` documents. On error it
/// publishes through both exception cells and returns 0.
pub extern "C" fn bh_load_build_class_fn(frame_ptr: i64) -> i64 {
    use pyre_interpreter::pyopcode::OpcodeStepExecutor;
    assert!(
        frame_ptr != 0,
        "bh_load_build_class_fn requires a non-null PyFrame; every \
         LOAD_BUILD_CLASS emit site must thread portal_frame_reg as its ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    match frame.load_build_class_value() {
        Ok(w_value) => w_value as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// DELETE_GLOBAL residual using the frame receiver and interned-name ABI.
/// pyopcode.py DELETE_GLOBAL deletes directly from `w_globals`.
pub extern "C" fn bh_delete_global_fn(frame_ptr: i64, w_name: i64) -> i64 {
    use pyre_interpreter::pyopcode::OpcodeStepExecutor;
    assert!(
        frame_ptr != 0,
        "bh_delete_global_fn requires a non-null PyFrame; every DELETE_GLOBAL emit \
         site must thread portal_frame_reg as the leading ref operand"
    );
    let frame = unsafe { &mut *(frame_ptr as *mut PyFrame) };
    let name =
        unsafe { pyre_object::unicodeobject::w_str_get_value(w_name as pyre_object::PyObjectRef) };
    match frame.delete_global(name) {
        Ok(()) => 1,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// Load a constant from the code object.
/// jtransform.py parity: code comes from getfield_vable_r(frame, pycode).
pub extern "C" fn bh_load_const_fn(w_code_ptr: i64, consti: i64) -> i64 {
    // `getconstant_w(index) -> co_consts_w[index]`: read the one shared object
    // off the virtualizable `pycode`, exactly as the interpreter does.
    let w_const = unsafe {
        pyre_interpreter::pycode::w_code_const(
            w_code_ptr as pyre_object::PyObjectRef,
            consti as usize,
        )
    };
    w_const as i64
}

/// Box a raw integer into a PyObject (w_int_new wrapper).
pub extern "C" fn bh_box_int_fn(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// `eval.rs`'s `raise_varargs` (RAISE_VARARGS) normalization for
/// blackhole/JitCode.
///
/// JitCode's `raise/r` bytecode carries only the final exception object, so
/// callers normalize `raise Type` and `raise X from Y` through this helper
/// before emitting `raise/r`.
///
/// `(frame: Ref, exc: Ref, cause: Ref) → Ref`.  The frame pointer is
/// emitted explicitly by `Instruction::RaiseVarargs` (codewriter.rs)
/// via `portal_frame_reg`, mirroring `bh_load_global_fn`'s frame-as-arg
/// ABI.  `pyopcode.py:704-722 RAISE_VARARGS` runs inside an opcode
/// dispatch where `frame` and `frame.execution_context` are always
/// valid, so `frame_ptr == 0` here signals a wiring bug — fail fast
/// at entry rather than degrade silently to a `RuntimeError`.
pub extern "C" fn bh_normalize_raise_varargs_with_frame(
    frame_ptr: i64,
    exc: i64,
    cause: i64,
) -> i64 {
    let parent_frame_ptr = frame_ptr as *const PyFrame;
    assert!(
        !parent_frame_ptr.is_null(),
        "bh_normalize_raise_varargs_with_frame requires a non-null parent \
         PyFrame; every RAISE_VARARGS emit site must thread portal_frame_reg \
         as the leading ref operand"
    );
    let exc = exc as PyObjectRef;
    let raw_cause = cause as PyObjectRef;

    // pyopcode.py:704-722 — cause and exc normalization share
    // `self.space` / `frame.execution_context`. Pin the caller frame's
    // execution_context for the whole body so the cause-class-call and
    // exc-class-call observe the same namespace.
    let frame_ctx = unsafe { (*parent_frame_ptr).execution_context };
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on blackhole re-entry.
        // Force both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            pyre_interpreter::eval::normalize_raise_cause(raw_cause)
        };
        match result {
            Ok(cause) => Some(cause),
            Err(mut err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                return err.to_exc_object() as i64;
            }
        }
    };

    let mut final_exc: PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            // pyopcode.py:711-713 — `space.call_function(w_type)` does
            // not depend on `frame.execution_context`; if the field is
            // null on a valid frame the class-call still proceeds.
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(obj) => pyre_interpreter::error::exception_from_call_type_error(exc, obj)
                    .to_exc_object(),
                Err(mut err) => err.to_exc_object(),
            }
        } else {
            pyre_interpreter::PyError::type_error("exceptions must derive from BaseException")
                .to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(mut err) = pyre_interpreter::eval::attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    final_exc as i64
}

/// Truthiness check: PyObjectRef → raw 0 or 1.
pub extern "C" fn bh_truth_fn(value: i64) -> i64 {
    let obj = value as PyObjectRef;
    if obj.is_null() {
        return 0;
    }
    match pyre_interpreter::opcode_ops::truth_value(obj) {
        Ok(truth) => truth as i64,
        Err(mut err) => {
            // A raising `__bool__` / `__len__` publishes for the trailing
            // GuardNoException, then returns 0.
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// RPython: bhimpl_int_lt, bhimpl_int_eq, etc. — comparison helper.
///
/// Performs a Python-level comparison and returns a boolean PyObject.
/// op_code encodes the CompareOp tag from CPython 3.13 COMPARE_OP.
pub extern "C" fn bh_compare_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;
    if lhs.is_null() || rhs.is_null() {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            "comparison on null operand".to_string(),
        );
        publish_residual_call_exception(err.to_exc_object() as i64);
        return 0;
    }

    // op_code 10 = CHECK_EXC_MATCH isinstance check (from codewriter CheckExcMatch).
    // lhs = exception value, rhs = exception type (or tuple of types) to match.
    // Mirror the interpreter's `check_exc_match_against` =
    // `exception_match(type(exc), match_class)` (eval.rs) so the match
    // walks the exception class MRO and accepts a tuple of classes.  The
    // earlier bespoke `ExcKind`-vs-type-name model only handled str / builtin
    // function match specs and fell through to an unconditional `true` for a
    // proper exception type object, which made every `except SomeError:`
    // appear to match — wrong for any clause beyond the first.
    if op_code == pyre_interpreter::runtime_ops::ISINSTANCE_OP_TAG {
        // Validate the match target is an exception class / tuple of exception
        // classes first (`cmp_exc_match`, pyopcode.py:1034-1039), raising
        // TypeError otherwise.  The BC handler runs
        // `validate_check_exc_match_class` before the bool-returning
        // `check_exc_match_against`, so the residual path must too — `except 5:`
        // (or a tuple with a non-exception member) raises instead of silently
        // producing a bool.
        if let Err(mut err) = pyre_interpreter::eval::validate_check_exc_match_class(rhs) {
            publish_residual_call_exception(err.to_exc_object() as i64);
            return 0;
        }
        let matched = pyre_interpreter::eval::check_exc_match_against(lhs, rhs);
        return pyre_object::w_bool_from(matched) as i64;
    }

    // op_code 6 = CONTAINS_OP `in`, 7 = `not in` (from compare_op_tag).
    // lhs = needle/item, rhs = container/haystack (flatten lowers the args
    // as `[item, container]`).
    if op_code == 6 || op_code == 7 {
        match pyre_interpreter::baseobjspace::contains(rhs, lhs) {
            Ok(found) => {
                let result = if op_code == 7 { !found } else { found };
                return pyre_object::w_bool_from(result) as i64;
            }
            Err(mut err) => {
                let exc_obj = err.to_exc_object();
                publish_residual_call_exception(exc_obj as i64);
                return 0;
            }
        }
    }

    // op_code 8 = IS_OP `is`, 9 = `is not` (from compare_op_tag).
    // `space.is_w`, not raw pointer identity: `W_AbstractIntObject.is_w`
    // (`intobject.py:44`) and `W_FloatObject.is_w` (`floatobject.py:196`)
    // compare two plain `int`s / `float`s by value, so a freshly boxed equal
    // value is identical.  Infallible — never publishes BH_LAST_EXC_VALUE.
    if op_code == 8 || op_code == 9 {
        let same = pyre_interpreter::baseobjspace::is_w(lhs, rhs);
        let result = if op_code == 9 { !same } else { same };
        return pyre_object::w_bool_from(result) as i64;
    }

    // op_code is the compact tag from compare_op_tag (0-5), NOT the raw
    // ComparisonOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_code) else {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown compare op tag {op_code}"),
        );
        publish_residual_call_exception(err.to_exc_object() as i64);
        return 0;
    };
    match pyre_interpreter::opcode_ops::compare_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// RPython: bhimpl_int_add, bhimpl_int_sub, etc. — binary op helper.
///
/// Performs a Python-level binary operation.
/// op_code is the BinaryOperator tag from CPython 3.13 BINARY_OP.
pub extern "C" fn bh_binary_op_fn(lhs: i64, rhs: i64, op_code: i64) -> i64 {
    let lhs = lhs as PyObjectRef;
    let rhs = rhs as PyObjectRef;

    // op_code is the compact tag from binary_op_tag (0-12), NOT the raw
    // BinaryOperator discriminant. Reverse the mapping to get the enum.
    let Some(op) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_code) else {
        let mut err = pyre_interpreter::PyError::new(
            pyre_interpreter::PyErrorKind::TypeError,
            format!("unknown binary op tag {op_code}"),
        );
        publish_residual_call_exception(err.to_exc_object() as i64);
        return 0;
    };
    match pyre_interpreter::opcode_ops::binary_value(lhs, rhs, op) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// BUILD_TUPLE: `space.newtuple(list_w)` (`objspace.py:332` →
/// `tupleobject.py:477` wraptuple) consuming a length-prefixed
/// `GcTypedArray` of refs — the forced `popvalues` list
/// (`pyframe.py:408-419`).  Length travels inside the array (offset-0
/// prefix), so there is no arity cap.  Allocation-only; the items are
/// pre-existing heap refs, no user code runs.
pub extern "C" fn bh_newtuple_from_array(array: i64) -> i64 {
    let arr = array as *const pyre_object::object_array::GcTypedArray;
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    pyre_interpreter::runtime_ops::build_tuple_from_refs(&items) as i64
}

/// BUILD_LIST arbitrary-arity residual: `space.newlist(items_w)`.  The
/// arity-`> 3` form of BUILD_LIST that the fixed `build_list_fn` (three
/// item slots) cannot cover; the items travel inside a length-prefixed
/// forced array exactly like [`bh_newtuple_from_array`].  Allocation-only
/// (`build_list_from_refs` = `w_list_new`), so `CallFlavor::Plain` with
/// no trailing `GuardNoException`.
pub extern "C" fn bh_newlist_from_array(array: i64) -> i64 {
    let arr = array as *const pyre_object::object_array::GcTypedArray;
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    pyre_interpreter::runtime_ops::build_list_from_refs(&items) as i64
}

/// BUILD_MAP residual — the dict counterpart of [`bh_newtuple_from_array`].
/// The length-prefixed array holds the interleaved `[k0, v0, k1, v1, ...]`
/// pairs the codewriter unrolled via `setarrayitem_gc_r`;
/// `build_map_from_refs` consumes them in `chunks_exact(2)`.  Keys are
/// hashed (may run user `__hash__` / `__eq__`) and an unhashable key raises
/// (`MayForce`, fallible); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_build_map_from_array(array: i64) -> i64 {
    let arr = array as *const pyre_object::object_array::GcTypedArray;
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    match pyre_interpreter::runtime_ops::build_map_from_refs(&items) {
        Ok(dict) => dict as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// BUILD_SET residual — the set counterpart of [`bh_build_map_from_array`].
/// The length-prefixed array holds the `count` set elements the codewriter
/// unrolled via `setarrayitem_gc_r`.  Element hashing may run user `__hash__`
/// and a non-hashable element raises (`MayForce`, fallible); on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_build_set_from_array(array: i64) -> i64 {
    let arr = array as *const pyre_object::object_array::GcTypedArray;
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    match pyre_interpreter::runtime_ops::build_set_from_refs(&items) {
        Ok(set) => set as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// BUILD_STRING residual (`build_string_from_array` HLOp →
/// `residual_call_r_r`).  Concatenates the forced fragment array into a
/// single `str` through the shared `runtime_ops::build_string_from_refs`.
/// Fragments are already strings (FORMAT_SIMPLE / CONVERT_VALUE ran first),
/// so this never runs user code → `Plain` (infallible, no exception
/// publish), mirroring `bh_newtuple_from_array`.
pub extern "C" fn bh_build_string_from_array(array: i64) -> i64 {
    let arr = array as *const pyre_object::object_array::GcTypedArray;
    let len = pyre_object::object_array::gcarray_len(arr);
    let mut items: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::object_array::getarrayitem_ref(arr, i));
    }
    pyre_interpreter::runtime_ops::build_string_from_refs(&items) as i64
}

/// FORMAT_SIMPLE residual (`format_simple` HLOp → `residual_call_r_r`).
/// Formats `value` with the empty spec (`f"{x}"` → `str(value)`) through the
/// shared `runtime_ops::format_value`.  A user `__format__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_format_simple_fn(value: i64) -> i64 {
    match pyre_interpreter::runtime_ops::format_value(
        value as pyre_object::PyObjectRef,
        pyre_object::PY_NULL,
    ) {
        Ok(s) => s as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// CONVERT_VALUE residual (`convert_value` HLOp → `residual_call_ir_r`).
/// Converts `value` per `conv` (`0=Str/3=None → str`, `1=Repr → repr`,
/// `2=Ascii → ascii`) through the shared `runtime_ops::convert_value`.  A
/// user `__str__` / `__repr__` may run Python (`MayForce`); on error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_convert_value_fn(value: i64, conv: i64) -> i64 {
    match pyre_interpreter::runtime_ops::convert_value(value as pyre_object::PyObjectRef, conv) {
        Ok(s) => s as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// FORMAT_WITH_SPEC residual (`format_with_spec` HLOp → `residual_call_r_r`).
/// Formats `value` with `spec` (`f"{x:.2f}"`) through the shared
/// `runtime_ops::format_value`.  A user `__format__` may run Python
/// (`MayForce`); on error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_format_with_spec_fn(value: i64, spec: i64) -> i64 {
    match pyre_interpreter::runtime_ops::format_value(
        value as pyre_object::PyObjectRef,
        spec as pyre_object::PyObjectRef,
    ) {
        Ok(s) => s as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// LOAD_DEREF residual (`load_deref_value` HLOp → `residual_call_ir_r`).
/// `cell` is the slot read from `locals_cells_stack_w`: a cell object whose
/// contents are the free/cell variable's value, or the raw value for a slot
/// that is not a cell.  Mirrors `load_deref` — dereference the cell and
/// raise if the result is empty.  `code` + `deref_idx` resolve the variable
/// name for the unbound-variable `NameError` (`deref_unbound_error`).  Runs no
/// user code but reads mutable heap (`CallFlavor::Plain`); on the
/// unbound-variable error the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call
/// returns 0.
pub extern "C" fn bh_load_deref_value_fn(cell: i64, w_code_ptr: i64, deref_idx: i64) -> i64 {
    let slot = cell as pyre_object::PyObjectRef;
    let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
        unsafe { pyre_object::w_cell_get(slot) }
    } else {
        slot
    };
    if value == pyre_object::PY_NULL {
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let exc_obj = pyre_interpreter::pyframe::deref_unbound_error(code, deref_idx as usize)
            .to_exc_object();
        publish_residual_call_exception(exc_obj as i64);
        return 0;
    }
    value as i64
}

/// STORE_DEREF residual (`store_deref_value` HLOp → `residual_call_r_r`).
/// `cell` is the slot read from `locals_cells_stack_w`; `value` is the
/// popped stack operand.  Mirrors `store_deref`: when the slot holds a
/// cell, mutate the cell's contents in place (`w_cell_set`, incminimark
/// write barrier inside) and return the unchanged cell so the caller
/// re-stores the same pointer into the slot; otherwise return the raw
/// `value` so the caller writes it into the slot directly.  Runs no user
/// code and never raises (`CallFlavor::PlainCannotRaise`).
pub extern "C" fn bh_store_deref_value_fn(cell: i64, value: i64) -> i64 {
    let slot = cell as pyre_object::PyObjectRef;
    if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
        unsafe { pyre_object::w_cell_set(slot, value as pyre_object::PyObjectRef) };
        cell
    } else {
        value
    }
}

/// MAKE_CELL residual (`make_cell_value` HLOp → `residual_call_r_r`).
/// `current` is the slot read from `locals_cells_stack_w`.  Mirrors
/// `make_cell`: wrap the value in a fresh cell when the slot does not
/// already hold one (`initialize_frame_scopes` installs cells for pure
/// cellvars, so only an argument slot promoted to a cellvar still holds a
/// raw value here), and return the cell the caller stores back into the
/// slot.  A slot already holding a cell is returned unchanged so a
/// never-reassigned cellvar does not become a cell wrapping a cell.
/// Allocates (may trigger a minor GC) but runs no user code and never
/// raises (`CallFlavor::Plain`).
pub extern "C" fn bh_make_cell_fn(current: i64) -> i64 {
    let cur = current as pyre_object::PyObjectRef;
    if cur.is_null() || !unsafe { pyre_object::is_cell(cur) } {
        pyre_object::w_cell_new(cur) as i64
    } else {
        current
    }
}

/// UNARY_NEGATIVE residual (`unary_negative` HLOp → `residual_call_r_r`).
/// Computes `-value` through `opcode_ops::unary_negative_value` (`neg`); a
/// user `__neg__` may run Python (`MayForce`).  On error the exception
/// is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_negative_fn(value: i64) -> i64 {
    match pyre_interpreter::opcode_ops::unary_negative_value(value as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// GET_ITER residual (`residual_call_r_r`).  Computes `iter(obj)` through
/// `baseobjspace::iter`; a user `__iter__` may run Python (`MayForce`).  On
/// error the exception is published through `BH_LAST_EXC_VALUE` for the
/// trailing `GuardNoException` and the call returns 0.
pub extern "C" fn bh_get_iter_fn(obj: i64) -> i64 {
    match pyre_interpreter::baseobjspace::iter(obj as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// UNARY_INVERT residual (`unary_invert` HLOp → `residual_call_r_r`).
/// Computes `~value` through `opcode_ops::unary_invert_value` (`invert`); a
/// user `__invert__` may run Python (`MayForce`).  On error the exception
/// is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_invert_fn(value: i64) -> i64 {
    match pyre_interpreter::opcode_ops::unary_invert_value(value as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// UNARY_POSITIVE residual (`pos` HLOp → `residual_call_r_r`).  Computes
/// `+value` through `opcode_ops::unary_positive_value` (`pos`); a user
/// `__pos__` may run Python (`MayForce`).  On error the exception is
/// published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_unary_positive_fn(value: i64) -> i64 {
    match pyre_interpreter::opcode_ops::unary_positive_value(value as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            publish_residual_call_exception(exc_obj as i64);
            0
        }
    }
}

/// CALL_INTRINSIC_1 ListToTuple residual (`list_to_tuple` HLOp →
/// `residual_call_r_r`).  Converts a list to a tuple through the shared
/// `opcode_ops::list_to_tuple_value`; allocates a fresh tuple, and a
/// non-list operand raises TypeError (`MayForce`).  On error the
/// exception is published through `BH_LAST_EXC_VALUE` for the trailing
/// `GuardNoException` and the call returns 0.
pub extern "C" fn bh_list_to_tuple_fn(value: i64) -> i64 {
    match pyre_interpreter::opcode_ops::list_to_tuple_value(value as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// GET_LEN residual (`get_len` HLOp → `residual_call_r_r`).  Pushes
/// `len(subject)` without consuming the subject.  Runs the type's
/// `__len__` (`MayForce`).
pub extern "C" fn bh_get_len_fn(subject: i64) -> i64 {
    match pyre_interpreter::baseobjspace::len(subject as pyre_object::PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// MATCH_SEQUENCE residual (`match_sequence` HLOp → `residual_call_r_r`).
/// Reads the subject type's PATMA marker; runs no user code and never
/// raises (`Plain`).
pub extern "C" fn bh_match_sequence_fn(subject: i64) -> i64 {
    pyre_interpreter::opcode_ops::match_sequence_value(subject as pyre_object::PyObjectRef) as i64
}

/// MATCH_MAPPING residual (`match_mapping` HLOp → `residual_call_r_r`).
/// Mirrors [`bh_match_sequence_fn`].
pub extern "C" fn bh_match_mapping_fn(subject: i64) -> i64 {
    pyre_interpreter::opcode_ops::match_mapping_value(subject as pyre_object::PyObjectRef) as i64
}

/// MATCH_KEYS residual (`match_keys` HLOp → `residual_call_r_r`).  Looks
/// each pattern key up in the subject via `get`, so it runs user code
/// (`MayForce`) and can raise on a duplicate key.
pub extern "C" fn bh_match_keys_fn(subject: i64, keys: i64) -> i64 {
    match pyre_interpreter::opcode_ops::match_keys_value(
        subject as pyre_object::PyObjectRef,
        keys as pyre_object::PyObjectRef,
    ) {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// MATCH_CLASS residual (`match_class` HLOp → `residual_call_ir_r`).
/// `count` is the number of positional sub-patterns.  Runs `isinstance`
/// and attribute lookups, so user code (`MayForce`).
pub extern "C" fn bh_match_class_fn(subject: i64, cls: i64, kwd_attrs: i64, count: i64) -> i64 {
    match pyre_interpreter::opcode_ops::match_class_value(
        subject as pyre_object::PyObjectRef,
        cls as pyre_object::PyObjectRef,
        kwd_attrs as pyre_object::PyObjectRef,
        count as usize,
    ) {
        Ok(result) => result as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// LOAD_COMMON_CONSTANT residual (`load_common_constant` HLOp →
/// `residual_call_ir_r`).  `disc` is the `CommonConstant` discriminant
/// (0-6).  Resolves the pushed object through the shared
/// `opcode_ops::load_common_constant_value`, matching the interpreter:
/// immortal type/exception classes for the class variants, a freshly
/// built builtin function for `all`/`any` (hence `MayForce` — it
/// allocates).  Runs no user code and never raises; an out-of-range
/// discriminant (corrupt bytecode) returns PY_NULL.
pub extern "C" fn bh_load_common_constant_fn(disc: i64) -> i64 {
    match pyre_interpreter::bytecode::CommonConstant::try_from(disc as u32) {
        Ok(cc) => pyre_interpreter::opcode_ops::load_common_constant_value(cc) as i64,
        Err(_) => pyre_object::PY_NULL as i64,
    }
}
/// UNARY_NOT residual (`unary_not` HLOp → `residual_call_r_r`).  Returns
/// `not value` as a bool object via `opcode_ops::truth_value`.  A user
/// `__bool__` / `__len__` may run Python (`MayForce`), matching the
/// interpreter's UNARY_NOT truth path; a raising `__bool__` publishes
/// through `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the
/// call returns 0.
pub extern "C" fn bh_unary_not_fn(value: i64) -> i64 {
    match pyre_interpreter::opcode_ops::truth_value(value as pyre_object::PyObjectRef) {
        Ok(truth) => pyre_object::w_bool_from(!truth) as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// LOAD_FAST_CHECK residual (`load_fast_check` HLOp → `residual_call_ir_r`).
/// The local slot is read from the vable exactly like LOAD_FAST and handed in
/// as `value` (possibly `PY_NULL` for an unbound local).  Returns `value`
/// unchanged when bound; on an unbound local raises `UnboundLocalError`,
/// resolving the variable name from the resume frame's code object via the
/// `co_varnames` index baked in by the codewriter.  Reads no heap and runs no
/// user code (`CallFlavor::Plain`); the exception is published through
/// `BH_LAST_EXC_VALUE` for the trailing `GuardNoException` and the call returns
/// 0.
pub extern "C" fn bh_load_fast_check_fn(value: i64, w_code_ptr: i64, name_idx: i64) -> i64 {
    if value as pyre_object::PyObjectRef != pyre_object::PY_NULL {
        return value;
    }
    let exc_obj = bh_unbound_local_error_fn(w_code_ptr, name_idx);
    publish_residual_call_exception(exc_obj);
    0
}

/// Construct the value raised by DELETE_FAST when its local is unbound.
/// Unlike `bh_load_fast_check_fn`, this returns the exception object without
/// publishing it through the residual-exception channel. It allocates but
/// runs no user code and never raises (`CallFlavor::PlainCannotRaise`).
pub extern "C" fn bh_unbound_local_error_fn(w_code_ptr: i64, name_idx: i64) -> i64 {
    let code = unsafe {
        &*(pyre_interpreter::w_code_get_ptr(w_code_ptr as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject)
    };
    // `name_idx` is a `co_varnames` index baked into the residual call by the
    // codewriter from the originating LOAD_FAST_CHECK oparg.  An out-of-range
    // index is a codegen invariant rather than a runtime-reachable error;
    // mirror `execute_load_fast_check`'s `idx < code_varnames_len(code)` guard,
    // falling back to the "<cell>" label rather than panicking in release.
    let idx = name_idx as usize;
    let name = if idx < code.varnames.len() {
        code.varnames[idx].as_ref()
    } else {
        "<cell>"
    };
    pyre_interpreter::PyError::unbound_local_error(format!(
        "cannot access local variable '{name}' where it is not associated with a value"
    ))
    .to_exc_object() as i64
}

#[cfg(test)]
mod tests_bh_newtuple_from_array {
    use super::bh_newtuple_from_array;
    use pyre_object::object_array::{ArrayKind, allocate_array, setarrayitem_ref};

    #[test]
    fn builds_tuple_of_any_arity_from_ref_array() {
        // The length travels inside the length-prefixed array, so any
        // arity fits without fixed argument slots.
        let arr = allocate_array(5, ArrayKind::Ref, true);
        let items: Vec<pyre_object::PyObjectRef> = (0..5)
            .map(|i| pyre_object::w_int_new(i as i64 * 7))
            .collect();
        for (i, &w) in items.iter().enumerate() {
            setarrayitem_ref(arr, i, w);
        }
        let tup = bh_newtuple_from_array(arr as i64) as pyre_object::PyObjectRef;
        unsafe {
            assert_eq!(pyre_object::w_tuple_len(tup), 5);
            for (i, &w) in items.iter().enumerate() {
                let got =
                    pyre_object::w_tuple_getitem(tup, i as i64).expect("tuple item within range");
                assert_eq!(got, w);
            }
        }
    }
}

/// BUILD_SLICE: `space.newslice(w_start, w_end, w_step)`.
/// `argc` is 2 or 3; for argc=2 the CPython/PyPy opcode semantics use None
/// for `w_step` (`pypy/interpreter/pyopcode.py:1463-1472`).
pub extern "C" fn bh_build_slice_fn(argc: i64, start: i64, stop: i64, step: i64) -> i64 {
    let step = if argc == 2 {
        pyre_object::w_none()
    } else {
        step as pyre_object::PyObjectRef
    };
    pyre_object::w_slice_new(
        start as pyre_object::PyObjectRef,
        stop as pyre_object::PyObjectRef,
        step,
    ) as i64
}

/// UNPACK_SEQUENCE: validate that `seq` has exactly `count` elements and
/// return a tuple of those items, raising ValueError/TypeError on a length
/// mismatch or non-sequence the same way the interpreter does. The portal
/// reads the items back out with `bh_unpack_item_fn`; producing the
/// validated tuple once keeps the iteration-protocol fallback single-pass.
pub extern "C" fn bh_unpack_sequence_fn(count: i64, seq: i64) -> i64 {
    let seq = seq as pyre_object::PyObjectRef;
    match pyre_interpreter::runtime_ops::unpack_sequence_exact(seq, count as usize) {
        Ok(items) => pyre_interpreter::runtime_ops::build_tuple_from_refs(&items) as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// Read item `index` out of the validated tuple produced by
/// `bh_unpack_sequence_fn`.
pub extern "C" fn bh_unpack_item_fn(index: i64, seq: i64) -> i64 {
    let seq = seq as pyre_object::PyObjectRef;
    match pyre_interpreter::runtime_ops::sequence_getitem(seq, index as usize) {
        Ok(item) => item as i64,
        Err(mut err) => {
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE
                .with(|c| c.set(err.to_exc_object() as i64));
            0
        }
    }
}

/// UNPACK_EX: split `seq` for `a, *b, c = seq` into `before` head items, a
/// starred middle list, and `after` tail items, returning the
/// `before + 1 + after` slots in TOS order as a tuple (raising ValueError on
/// too few values, or any iteration error from a non-sequence source). The
/// portal reads each slot back out with `bh_unpack_item_fn`, mirroring
/// `bh_unpack_sequence_fn`.
pub extern "C" fn bh_unpack_ex_fn(before: i64, after: i64, seq: i64) -> i64 {
    let seq = seq as pyre_object::PyObjectRef;
    match pyre_interpreter::runtime_ops::unpack_ex_slots(before as usize, after as usize, seq) {
        Ok(slots) => pyre_interpreter::runtime_ops::build_tuple_from_refs(&slots) as i64,
        Err(mut err) => {
            publish_residual_call_exception(err.to_exc_object() as i64);
            0
        }
    }
}

/// Read the current (per-thread) exception saved in
/// `pyre_interpreter::eval::CURRENT_EXCEPTION`. Matches the read at
/// `pyopcode.py:786 PUSH_EXC_INFO` (implicit via `executioncontext.sys_exc_info`).
pub extern "C" fn bh_get_current_exception() -> i64 {
    pyre_interpreter::eval::get_current_exception() as i64
}

/// `eval.rs`'s `raise_varargs(0)` — the value a bare `raise` re-raises.
///
/// Returns the active exception when `sys_exc_value` holds a live
/// `BaseException`; otherwise (null / `None` / non-exception) returns a fresh
/// `RuntimeError("No active exception to reraise")` instance.  The codewriter
/// emits this for a bare `RAISE_VARARGS(0)` whose FrameState carries no
/// `last_exception` pair — where the runtime current-exception may be absent —
/// so the following `raise/r` always receives a non-null value
/// (`blackhole.py:1002` asserts non-null).  Unlike raw `get_current_exception`,
/// this can allocate (the `RuntimeError`), so it is registered `Plain`, not
/// `PlainCannotRaiseNoHeap`.
pub extern "C" fn bh_reraise_varargs_zero() -> i64 {
    let exc = pyre_interpreter::eval::get_current_exception();
    unsafe {
        if !exc.is_null() && pyre_object::is_exception(exc) {
            exc as i64
        } else {
            pyre_interpreter::PyError::runtime_error("No active exception to reraise")
                .to_exc_object() as i64
        }
    }
}

/// Store `exc` into the per-thread `CURRENT_EXCEPTION` slot. Matches
/// the write at `pyopcode.py:778 POP_EXCEPT` (restore of saved
/// sys_exc_info) and at `pyopcode.py:786 PUSH_EXC_INFO` (new raised
/// exception becomes current).
pub extern "C" fn bh_set_current_exception(exc: i64) {
    pyre_interpreter::eval::set_current_exception(exc as pyre_object::PyObjectRef);
}

/// Complete the `PUSH_EXC_INFO` ownership transfer after the caught exception
/// has become the execution context's current exception.  The interpreter's
/// `push_exc_info` clears this propagation carrier at the same point; leaving
/// it populated roots the handled exception (and its traceback/frame) after
/// the handler returns.
pub extern "C" fn bh_clear_in_flight_exception() {
    pyre_interpreter::eval::set_in_flight_exception(pyre_object::PY_NULL);
}

/// On-demand resume callback (pyre-jit side).  Registered into cranelift
/// via `register_resumedata_deopt` (eval.rs:init_callbacks) and called
/// from the `recovery_layout_ref()` consumer sites that have migrated off
/// the pre-baked `ExitRecoveryLayout` cache.
///
/// PyPy parity target: `pyjitpl.py:3424
/// MetaInterp.rebuild_state_after_failure(resumedescr, deadframe)`
/// drives `resume.rebuild_from_resumedata` to materialise virtuals +
/// replay pending fields from `rd_numb` / `rd_consts` / `rd_virtuals` /
/// `rd_pendingfields` carried on the `ResumeGuardDescr`.
///
/// `outputs` enters carrying the failing-guard fail_args (the JITed
/// exit code stored them).  We treat that as the `deadframe` input to
/// the decoder and replace `*outputs` with the per-section
/// concatenation that the recovery_layout walker produced before
/// (innermost-first per `compiler.rs`'s `recovery.frames` reverse walk).
///
/// Implementation lives in `call_jit.rs` rather than `eval.rs` because
/// `pyre-jit-trace`'s build.rs reads `eval.rs` through the JIT
/// translator's RPython subset, which rejects the trait-object
/// downcast pattern used here.
#[cfg(feature = "cranelift")]
pub fn cranelift_resumedata_deopt(
    descr_addr: usize,
    outputs: &mut Vec<i64>,
    types: &[majit_ir::Type],
    _bridge_num_inputs: usize,
) -> bool {
    use majit_backend::Backend;
    use majit_ir::FailDescr;
    use majit_metainterp::resume;

    if std::env::var_os("PYRE_DEOPT_PROBE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ENTRIES: AtomicU64 = AtomicU64::new(0);
        let n = ENTRIES.fetch_add(1, Ordering::Relaxed);
        eprintln!("[deopt-probe] entry #{n} descr_addr={descr_addr:#x}");
    }

    // 1. Recover descr Arc.
    let (driver, driver_vinfo) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let descr = backend.fail_descr_arc_from_addr(descr_addr);

    // 2. Downcast to ResumeGuardDescr.  Synthetic FINISH /
    //    ExitFrameWithException / external-JUMP descrs have no `rd_*`
    //    payload upstream (compile.py:624-662) — they short-circuit
    //    here.  Callers fall back to the recovery_layout walker for
    //    these until the synthetic construction path is restructured.
    let Some(any) = descr.as_any() else {
        return false;
    };
    let Some(rgd) = any.downcast_ref::<majit_backend::ResumeGuardDescr>() else {
        return false;
    };

    // A novable driver's resume data has no virtualizable section. Derive the
    // owning driver from the guard descr, matching the blackhole path's
    // caller-supplied distinction, rather than guessing from this callback's
    // backend registration.
    let novable = match rgd
        .rd_loop_token_clt()
        .and_then(|clt_any| {
            clt_any.downcast_ref::<std::sync::Arc<majit_backend::CompiledLoopToken>>()
        })
        .and_then(|clt| clt.upgrade_loop_token())
        .and_then(|token| token.outermost_jitdriver_index)
        .and_then(|idx| driver.meta_interp().jitdriver_has_vinfo(idx))
    {
        Some(has_vinfo) => !has_vinfo,
        // Synthetic descr, evicted token, or stale driver slot: decline to the
        // existing recovery-layout fallback instead of choosing the wrong
        // resume-data shape for one of the two driver kinds.
        None => return false,
    };

    // 3. Extract resume payload.  Empty rd_numb → nothing to decode.
    let Some(rd_numb) = rgd.payload.rd_numb() else {
        return false;
    };
    if rd_numb.is_empty() {
        return false;
    }
    let rd_consts = rgd.payload.rd_consts().unwrap_or(&[]);
    let rd_virtuals_rcs = rgd.payload.rd_virtuals();
    let rd_pendingfields = rgd.payload.rd_pendingfields();

    // 4. resume.py:983-991 _prepare_virtuals — convert RdVirtualInfo →
    //    VirtualInfo so the decoder can materialise lazily.
    let count = outputs.len() as i32;
    let rd_virtuals_converted: Option<Vec<resume::VirtualInfo>> = rd_virtuals_rcs.map(|rcs| {
        let num_virtuals = rcs.len();
        rcs.iter()
            .map(|rd| resume::rd_virtual_to_virtual_info(rd, rd_consts, count, num_virtuals))
            .collect()
    });
    let rd_virtuals_slice = rd_virtuals_converted.as_deref();

    // 5. Construct ResumeDataDirectReader.  `outputs` enters as the
    //    deadframe (the JITed exit code stored fail_args here).
    //    Snapshot all_liveness once so the slice outlives the reader.
    let all_liveness = pyre_jit_trace::state::liveness_info_snapshot();
    let deadframe: Vec<i64> = outputs.clone();
    let allocator = crate::eval::PyreBlackholeAllocator;
    let mut reader = resume::ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        &all_liveness,
        &deadframe,
        Some(types),
        None,
        &allocator,
    );

    // 6. resume.py:1324-1325 — prepare virtuals/pendingfields, then
    //    consume the vref + vable sections that precede the per-frame
    //    sections.
    reader.prepare(rd_virtuals_slice, rd_pendingfields);
    let vinfo_dyn: &dyn resume::VirtualizableInfo = driver_vinfo.as_ref();
    let vrefinfo_dyn: &dyn resume::VRefInfo = driver.meta_interp().virtualref_info();
    if std::env::var_os("PYRE_DEOPT_PROBE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let n = HITS.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[deopt-probe] cranelift vinfo hand-off #{n} descr_addr={descr_addr:#x} novable={novable}"
        );
    }
    let vinfo_arg: Option<&dyn resume::VirtualizableInfo> =
        if novable { None } else { Some(vinfo_dyn) };
    reader.consume_vref_and_vable(Some(vrefinfo_dyn), vinfo_arg, None, None);

    // 7. resume.py:1339 jitcodes[jitcode_pos] lookup — same shape as
    //    blackhole_resume_via_rd_numb's resolve_jitcode,
    //    but returns the (jitcode, pc, op_live) triple
    //    consume_all_sections_into_vec needs to compute the per-section
    //    liveness offset.
    let (op_live_i32, _op_catch_exception, _op_rvmprof_code) =
        pyre_jit_trace::state::blackhole_control_opcodes();
    // op_live is the `-live-` opcode byte that JitCode::get_live_vars_info
    // (translate/codewriter/jitcode.rs) uses to skip past the
    // op header.  state.rs returns it as i32 for the
    // `setup_cached_control_opcodes` API; we narrow here.  A negative
    // or out-of-range value means the control opcodes were not set up,
    // which would break the per-section walk — short-circuit to the
    // recovery_layout fallback instead.
    if !(0..=255).contains(&op_live_i32) {
        return false;
    }
    let op_live = op_live_i32 as u8;
    let resolve_jitcode = |jitcode_index: i32,
                           pc: i32|
     -> Option<(
        std::sync::Arc<majit_metainterp::jitcode::JitCode>,
        usize,
        u8,
    )> {
        if pc < 0 {
            return None;
        }
        let pyjitcode = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)?;
        if pyjitcode.has_abort_opcode() {
            return None;
        }
        let resolved_pc = if pyjitcode.jitcode.can_decode_live_vars(pc as usize, op_live) {
            pc as usize
        } else {
            return None;
        };
        Some((pyjitcode.jitcode.clone(), resolved_pc, op_live))
    };

    // 8. Drive the per-section consume loop, appending decoded values
    //    into a fresh rebuilt vec.  Mirrors the
    //    `rebuild_state_after_failure(outputs, types, recovery)` walker:
    //    innermost-first concatenation of (i, r, f) sections.
    let mut rebuilt: Vec<i64> = Vec::with_capacity(outputs.len());
    if !reader.consume_all_sections_into_vec(&resolve_jitcode, &mut rebuilt) {
        // resolve_jitcode failure — leave outputs as-is so the
        // recovery_layout fallback path can take over.
        return false;
    }

    // 9. Replace outputs with rebuilt.
    *outputs = rebuilt;
    true
}

/// Pyre-jit side of the on-demand
/// `ExitRecoveryLayout` reconstruction callback registered into
/// cranelift via `register_recovery_layout` (eval.rs:init_callbacks).
/// Used by `CraneliftFailDescr::recovery_layout_ref` to derive the
/// layout from the metainterp-side `StoredExitLayout.resume_layout`
/// summary instead of reading the
/// `ResumeGuardDescr.recovery_layout` cache.
///
/// Returns `None` for synthetic descrs (FINISH / external-JUMP /
/// overlay) without a `ResumeGuardDescr` meta_descr or for descrs
/// whose `compiled_loops` entry has been evicted; callers fall back
/// to the meta-side slot read in that case.
#[cfg(feature = "cranelift")]
pub fn cranelift_recovery_layout_for_descr(
    descr_addr: usize,
    caller_prefix: Option<&majit_backend::ExitRecoveryLayout>,
) -> Option<majit_backend::ExitRecoveryLayout> {
    use majit_backend::Backend;

    let (driver, _) = crate::eval::driver_pair();
    let backend = driver.meta_interp().backend();
    let descr = backend.fail_descr_arc_from_addr(descr_addr);
    let fd = descr.as_fail_descr()?;
    driver
        .meta_interp()
        .compute_recovery_layout_for_descr(fd, caller_prefix)
}

#[cfg(test)]
mod tests_bh_normalize_raise {
    use super::*;
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JF_FRAME_OFS};
    use pyre_interpreter::{PyErrorKind, compile_exec};

    #[test]
    fn jitframe_layout_descrs_uses_frame_relative_offsets() {
        let descrs = jitframe_layout_descrs();
        assert_eq!(descrs.jf_frame_baseitemofs, FIRST_ITEM_OFFSET);
        assert_eq!(descrs.jf_frame_lengthofs, JF_FRAME_OFS);
    }

    #[test]
    fn bh_normalize_raise_varargs_rejects_builtin_callables_that_are_not_exception_classes() {
        let callable = pyre_interpreter::make_builtin_function(
            "len",
            pyre_interpreter::builtins::__pyre_wrap_builtin_len,
        );
        let code = compile_exec("").expect("empty module should compile");
        let frame = pyre_interpreter::PyFrame::new(code);

        let frame_ptr = (&*frame as *const pyre_interpreter::PyFrame) as i64;
        let result = bh_normalize_raise_varargs_with_frame(
            frame_ptr,
            callable as i64,
            pyre_object::PY_NULL as i64,
        );
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result as PyObjectRef) };
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message_text(),
            "exceptions must derive from BaseException"
        );
    }
}
