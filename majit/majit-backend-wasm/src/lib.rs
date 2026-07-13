/// WebAssembly backend for majit.
///
/// Generates wasm bytecodes via wasm-encoder. On wasm32 targets,
/// instantiates the emitted trace modules through a host binding (see
/// `glue`): the `web` feature uses the browser `WebAssembly` API via
/// wasm-bindgen, the `host-import` feature uses plain wasm imports that a
/// native embedder (wasmi / wasmtime) supplies. On native targets,
/// compile_loop succeeds but execute_token requires a wasm host
/// (unreachable natively).
pub mod codegen;
pub mod failguard;

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
mod glue;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Diagnostic-only `compile_bridge` outcome tallies, read out via the
/// `pyre_jit_bridge_diag` guest export (the runner prints them at
/// `PYRE_WASM_JIT_STATS` time). A static counter — NOT a host import, which
/// would shift the wasm function-index space and break the JIT's baked
/// `fn as usize` table indices. Index legend: 0 = compile_bridge entered,
/// 1 = declined CALL_ASSEMBLER, 2 = declined multi-label peeled,
/// 3 = declined not-a-direct-loop-guard, 4 = declined ref-home overflow,
/// 5 = bridge compiled (chained in-module), 6 = loop-closing shape seen,
/// 7 = source loop has a preamble. Sub-breakdown of the index-2 multi-label
/// decline (TEMP, for the resume-at-last-label measurement): 8 = JUMP descr
/// did not resolve (target_ord None), 9 = target_ord Some but != last label,
/// 10 = arity mismatch, 11 = loop-closing bridge advances no loop-carried value
/// (guard side-trace that would livelock the chained loop), 15 = declined CA
/// because a trace would use the host call trampoline on a movable CA frame.
pub static BRIDGE_DIAG: [AtomicU64; 16] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z]
};

/// Read a `BRIDGE_DIAG` tally (saturating index). Surfaced to the host through
/// the `pyre_jit_bridge_diag` export in the `pyre-wasm` crate.
pub fn bridge_diag(i: usize) -> u64 {
    BRIDGE_DIAG
        .get(i)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0)
}

#[inline]
fn diag_bump(i: usize) {
    BRIDGE_DIAG[i].fetch_add(1, Ordering::Relaxed);
}

// A source token is compiled before a later guard may become a CA bridge.
// Freeze modest room for that bridge at first compilation; a later trace that
// exceeds either bound is declined rather than changing the live frame's
// offsets. The fib CA path measured raw maxima of 9 positional slots and 15
// Ref homes, but that is not a bridge fail-arg census for the full suite; keep
// both existing floors until such a census establishes smaller safe bounds.
const FROZEN_CHAIN_VALUE_SLOTS: usize = 32;
const FROZEN_CHAIN_REF_HOMES: usize = 16;

/// An arithmetic op whose result advances a loop-carried numeric value (the
/// `IntAdd`/`IntSub`/… and float-arithmetic block plus the overflow-checked
/// variants and the unary `IntNeg`/`IntInvert`). Excludes copies (`SameAs*`),
/// casts, comparisons, and allocations: those feed a JUMP arg without making
/// the loop's induction walk toward its exit condition. Used to tell a
/// state-advancing loop-closing bridge from a guard side-trace that re-presents
/// the same loop state every pass.
fn is_inductive_arith(opcode: majit_ir::OpCode) -> bool {
    use majit_ir::OpCode::*;
    matches!(
        opcode,
        IntAdd
            | IntSub
            | IntMul
            | UintMulHigh
            | IntFloorDiv
            | IntMod
            | IntAnd
            | IntOr
            | IntXor
            | IntRshift
            | IntLshift
            | UintRshift
            | IntSignext
            | FloatAdd
            | FloatSub
            | FloatMul
            | FloatTrueDiv
            | FloatFloorDiv
            | FloatMod
            | FloatNeg
            | FloatAbs
            | IntNeg
            | IntInvert
            | IntAddOvf
            | IntSubOvf
            | IntMulOvf
    )
}

/// Per-guard (per-trace order), per-fail-arg: whether the value was produced
/// by induction-advancing arithmetic in the part of the trace that re-runs on
/// every pass — the ops after the loop-header (last) LABEL, or the WHOLE trace
/// when it has no LABEL (a bridge, or a Label-less recursion loop, whose body
/// runs in full each pass). Such a fail arg is fresh in the failing iteration,
/// so a loop-closing bridge that JUMPs it verbatim still advances the chained
/// loop⇄bridge cycle (`compile_bridge`'s livelock check).
fn guard_fail_args_advanced(
    ops: &[majit_ir::Op],
    guard_exits: &[codegen::GuardExit],
) -> Vec<Vec<bool>> {
    let start = ops
        .iter()
        .rposition(|op| op.opcode == majit_ir::OpCode::Label)
        .map_or(0, |p| p + 1);
    let advanced_ids: std::collections::HashSet<u32> = ops[start..]
        .iter()
        .filter(|op| is_inductive_arith(op.opcode))
        .map(|op| op.pos.get())
        .filter(|r| *r != majit_ir::OpRef::NONE && !r.is_constant())
        .map(|r| r.raw())
        .collect();
    guard_exits
        .iter()
        .map(|g| {
            g.fail_arg_refs
                .iter()
                .map(|r| !r.is_constant() && advanced_ids.contains(&r.raw()))
                .collect()
        })
        .collect()
}

use failguard::{
    ChainedTraceMeta, CompiledWasmLoop, LabelTarget, WasmFailDescr, WasmFrameData, fail_descr_base,
    global_fail_descr, label_target, publish_label_target, register_fail_descrs,
};
use majit_backend::{AsmInfo, BackendError, DeadFrame, JitCellToken};
use majit_gc::GcAllocator;
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpRc, Value};

/// JIT exception state, mirroring the native backends' `JIT_EXC_VALUE` /
/// `JIT_EXC_TYPE` globals. A can-raise helper publishes the pending exception
/// here via `jit_exc_raise`; the compiled trace's `GuardNoException` /
/// `GuardException` read these slots by absolute address through the shared
/// linear memory (host and trace import the same `env.memory`) and fail the
/// guard accordingly. Single-slot per process, matching the single-threaded
/// dynasm/cranelift backends.
static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);

/// llmodel.py:194-199 _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0); the exception class is read from
/// `value.typeptr` (offset 0).
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        // `typeptr` is a machine pointer (32-bit on wasm32); read it at
        // pointer width and zero-extend, so the high bits stay clear and
        // `GuardException`'s type comparison matches the baked class pointer.
        unsafe { *(value as *const usize) as i64 }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// grab_exc_value parity: read the pending exception value and clear both
/// slots. Called host-side after a trace returns through a guard exit.
pub fn jit_exc_take() -> i64 {
    let value = JIT_EXC_VALUE.swap(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
    value
}

/// Clear both exception slots without reading the value.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

/// Address of `JIT_EXC_VALUE`, embedded as an immediate in JIT-emitted wasm
/// so the trace can load/store it over the shared linear memory
/// (`_store_and_reset_exception` parity).
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

/// Address of `JIT_EXC_TYPE`, embedded as an immediate in JIT-emitted wasm.
pub fn jit_exc_type_addr() -> usize {
    &JIT_EXC_TYPE as *const _ as usize
}

thread_local! {
    /// llmodel.py self.gc_ll_descr — owned by the active wasm
    /// backend on this thread. Stored as a thread-local so the
    /// backend-agnostic `majit_gc::ActiveGcGuardHooks` shims can
    /// reach the live allocator without taking a wasm dependency.
    /// Mirrors `cranelift::compiler::CRANELIFT_ACTIVE_GC` and
    /// `dynasm::runner::DYNASM_ACTIVE_GC` — RPython's
    /// `cpu.gc_ll_descr` parity, single-slot per thread.
    static WASM_ACTIVE_GC: RefCell<Option<Box<dyn GcAllocator>>> = const { RefCell::new(None) };
    /// Raw mirror of the boxed allocator, read by `wasm_gc_owns_object`'s
    /// reentrant fallback: the interpreter-safepoint major holds the
    /// `WASM_ACTIVE_GC` mutable borrow while extra-root walkers ask whether a
    /// slot is GC-managed, so that query routes through the raw pointer instead
    /// of a second borrow. Mirrors `dynasm::runner::DYNASM_ACTIVE_GC_RAW`.
    static WASM_ACTIVE_GC_RAW: std::cell::Cell<Option<*mut dyn GcAllocator>> =
        const { std::cell::Cell::new(None) };
}

/// Read-only GC query for the guard hooks and codegen helpers.
///
/// - **Test box present** (`WASM_ACTIVE_GC` is `Some`): a test owns the
///   GC directly — apply `f` to the boxed allocator.
/// - **No box, `gc_sync` initialized** (production): route to the
///   process-global singleton via `gc_query_reentrant`. Some of these
///   callers can fire during a collection's guard evaluation / extra-root
///   walk, so the reentrant read-only path (no second `gc_mutex`, no
///   second `&mut`) is required.
/// - **No box, `gc_sync` uninitialized** (no GC at all — unit tests):
///   returns `None` so callers keep their existing `.unwrap_or(default)`
///   / `.flatten()` behaviour.
fn with_wasm_active_gc<R>(f: impl Fn(&dyn GcAllocator) -> R) -> Option<R> {
    match WASM_ACTIVE_GC.with(|cell| cell.try_borrow().map(|g| g.as_deref().map(|gc| f(gc)))) {
        Ok(Some(r)) => return Some(r),
        Ok(None) => {}
        Err(_) => {
            if let Some(ptr) = WASM_ACTIVE_GC_RAW.with(|raw| raw.get()) {
                // SAFETY: the raw mirror points at the same live test box whose
                // mutable borrow is held by the in-progress mutation. This is a
                // read-only reentrant query, so it does not create a second &mut.
                return Some(f(unsafe { &*ptr }));
            }
        }
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_query_reentrant(f));
    }
    None
}

/// `&mut` counterpart of `with_wasm_active_gc` for GC mutations
/// (allocation, write barriers, collection). Test box → box; production
/// (no box, `gc_sync` initialized) → `gc_sync::gc_op`; no GC at all →
/// `None` so callers keep their non-GC fallback. Top-level mutator/
/// blackhole trampolines, never inside a collection, so `gc_op` is correct.
fn with_wasm_active_gc_mut<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
    if WASM_ACTIVE_GC.with(|cell| cell.borrow().is_some()) {
        return WASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let raw: *mut dyn GcAllocator = guard.as_deref_mut()?;
            // SAFETY: `guard` holds the borrow for the whole `f` call and
            // these are non-reentrant top-level trampolines, so the
            // reborrow is exclusive and outlives `f`.
            Some(f(unsafe { &mut *raw }))
        });
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_op(f));
    }
    None
}

/// Register all backend-agnostic `majit_gc::set_active_*` hooks to the
/// wasm trampolines. Shared by `install_gc_box` (test path: also stores a
/// box in TLS) and `install_gc_standalone` (production: hooks only, no box
/// — the trampolines then route to the `gc_sync` singleton).
fn register_active_hooks(supports_guard_gc_type: bool) {
    majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
        check_is_object: Some(wasm_check_is_object),
        is_tagged_immediate: Some(wasm_is_tagged_immediate),
        get_actual_typeid: Some(wasm_get_actual_typeid),
        subclass_range: Some(wasm_subclass_range),
        typeid_subclass_range: Some(wasm_typeid_subclass_range),
        typeid_is_object: Some(wasm_typeid_is_object),
        can_move: None,
        supports_guard_gc_type,
    });
    majit_gc::set_active_alloc_nursery_typed(Some(wasm_alloc_nursery_typed));
    majit_gc::set_active_alloc_oldgen_typed(Some(wasm_alloc_oldgen_typed));
    majit_gc::set_active_root_hooks(Some(wasm_gc_add_root), Some(wasm_gc_remove_root));
    majit_gc::set_active_gc_owns_object(Some(wasm_gc_owns_object));
    majit_gc::set_active_write_barrier(Some(wasm_active_gc_write_barrier));
    majit_gc::set_active_collect_oldgen(Some(wasm_collect_oldgen_nonmoving));
    majit_gc::set_active_heap_stats(Some(active_gc_heap_stats));
    majit_gc::set_active_finalizer_hooks(
        Some(wasm_register_finalizer),
        Some(wasm_finalizer_next_dead),
    );
}

/// Store a GC allocator in the wasm backend thread-local and register
/// the `majit_gc::set_active_*` function-pointer hooks, without
/// requiring a `WasmBackend` instance.
/// Install a GC box into TLS and register all `set_active_*` hooks. Test
/// path only — `set_gc_allocator` hands ownership of a real allocator to
/// the backend thread. Production uses [`install_gc_standalone`], which
/// registers the same hooks WITHOUT a box so the trampolines fall through
/// to `gc_sync`.
fn install_gc_box(gc: Box<dyn majit_gc::GcAllocator>) {
    let supports_guard_gc_type = gc.supports_guard_gc_type();
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        *guard = Some(gc);
        let raw = guard.as_deref_mut().map(|gc| gc as *mut dyn GcAllocator);
        WASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
    });
    register_active_hooks(supports_guard_gc_type);
}

/// Production path: register all `set_active_*` hooks WITHOUT storing a
/// box. `WASM_ACTIVE_GC` stays `None`, so every trampoline routes to the
/// process-global `gc_sync` singleton (the per-thread GC box is the
/// free-threading gap R4 removes).
pub fn install_gc_standalone() {
    majit_gc::gc_sync::gc_op(|gc| gc.freeze_types());
    let supports_guard_gc_type = majit_gc::gc_sync::gc_query(|gc| gc.supports_guard_gc_type());
    register_active_hooks(supports_guard_gc_type);
}

/// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)` of the GC owned
/// by this thread's wasm backend, or `(0, 0)` if none is installed. Lets a host
/// runner split GC-retained memory from host-heap growth.
pub fn active_gc_heap_stats() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.heap_byte_stats()).unwrap_or((0, 0))
}

/// Diagnostic: `(minor_collections, major_collections)` of the active GC, or
/// `(0, 0)` when none is installed. Companion to [`active_gc_heap_stats`].
pub fn active_gc_collection_counts() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.collection_counts()).unwrap_or((0, 0))
}

/// Assemble the inline nursery-bump parameters for this trace's `New` /
/// `NewWithVtable` ops (rewrite.py malloc-fast-path eligibility over the
/// gc.py:525-531 nursery address surface), or `None` when no GC is active,
/// the `gc_stress` feature is compiled in (the fast path would bypass its
/// per-allocation stress collections), or no allocation op qualifies.
fn nursery_alloc_params(ops: &[Op]) -> Option<codegen::NurseryAllocParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let tids: std::collections::HashSet<u32> = ops
        .iter()
        .filter_map(|op| match op.opcode {
            majit_ir::OpCode::New | majit_ir::OpCode::NewWithVtable => {
                Some(op.getdescr()?.as_size_descr()?.type_id())
            }
            majit_ir::OpCode::NewArray | majit_ir::OpCode::NewArrayClear => {
                Some(op.getdescr()?.as_array_descr()?.type_id())
            }
            _ => None,
        })
        .collect();
    if tids.is_empty() {
        return None;
    }
    with_wasm_active_gc(|gc| {
        let free_addr = gc.nursery_free_addr();
        let top_addr = gc.nursery_top_addr();
        if free_addr == 0 || top_addr == 0 {
            return None;
        }
        let plain_tids: std::collections::HashSet<u32> = tids
            .iter()
            .copied()
            .filter(|&t| gc.type_alloc_is_plain(t))
            .collect();
        if plain_tids.is_empty() {
            return None;
        }
        Some(codegen::NurseryAllocParams {
            free_addr: free_addr as u32,
            top_addr: top_addr as u32,
            large_threshold: gc.max_nursery_object_size(),
            plain_tids,
        })
    })?
}

/// Assemble the direct CA arm's fixed-size nursery/frame parameters. This is
/// deliberately separate from ordinary `New*` eligibility: a CA frame needs
/// both the nursery words and the JitFrame shadow-stack top/limit cells.
/// Missing active GC (or gc_stress) leaves the pre-existing helper path intact.
fn ca_inline_params(frame_bytes: u32) -> Option<codegen::CaInlineParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let jitframe_tid = wasm_jitframe_tid();
    let depth = frame_bytes as usize / std::mem::size_of::<isize>();
    let total = ((majit_gc::header::GcHeader::SIZE
        + majit_backend::jitframe::JitFrame::alloc_size(depth))
    .max(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE)
        + 7)
        & !7;
    with_wasm_active_gc(|gc| {
        assert_ne!(
            jitframe_tid, 0,
            "wasm CA inline frame path requires the registered JitFrame type id"
        );
        if total > gc.max_nursery_object_size() || !gc.type_alloc_is_plain(jitframe_tid) {
            return None;
        }
        let nursery_free_addr = gc.nursery_free_addr();
        let nursery_top_addr = gc.nursery_top_addr();
        let jf_top_addr = majit_gc::shadow_stack::get_root_stack_top_addr();
        let jf_limit_addr = majit_gc::shadow_stack::get_root_stack_limit_addr();
        (nursery_free_addr != 0 && nursery_top_addr != 0 && jf_top_addr != 0 && jf_limit_addr != 0)
            .then_some(codegen::CaInlineParams {
                nursery_free_addr: nursery_free_addr as u32,
                nursery_top_addr: nursery_top_addr as u32,
                jf_top_addr: jf_top_addr as u32,
                jf_limit_addr: jf_limit_addr as u32,
                jitframe_tid,
            })
    })?
}

/// Address of the active jitframe shadow-stack top cell for ordinary trace
/// body reloads. This does not depend on nursery fast-path eligibility: the
/// reload is valid whenever a GC is active at compilation time.
fn jf_top_addr() -> Option<u32> {
    with_wasm_active_gc(|_| majit_gc::shadow_stack::get_root_stack_top_addr())
        .and_then(|addr| u32::try_from(addr).ok())
        .filter(|&addr| addr != 0)
}

/// `majit_gc::CollectOldgenFn` installed by `set_gc_allocator`. Drives the
/// interpreter-safepoint non-moving old-gen major (`gc_interp::safepoint`,
/// default-on on wasm) through the active GC. Needs mutable access, so it
/// routes via `with_wasm_active_gc_mut` (test box → box; production → the
/// `gc_sync` singleton). Mirrors dynasm's `dynasm_collect_oldgen_nonmoving`
/// and cranelift's `collect_oldgen_nonmoving_via_active_runtime`.
fn wasm_collect_oldgen_nonmoving() {
    with_wasm_active_gc_mut(|gc| gc.collect_oldgen_nonmoving());
}

fn wasm_register_finalizer(fq_index: usize, obj: GcRef, trigger: majit_gc::FinalizerTriggerFn) {
    with_wasm_active_gc_mut(|gc| gc.register_finalizer(fq_index, obj, trigger));
}

fn wasm_finalizer_next_dead(fq_index: usize) -> Option<GcRef> {
    with_wasm_active_gc_mut(|gc| gc.finalizer_next_dead(fq_index)).flatten()
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`.
/// Mirrors cranelift's `check_is_object_via_active_runtime`: dispatches
/// through the wasm-thread-local GC allocator.
fn wasm_check_is_object(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn wasm_is_tagged_immediate(addr: usize) -> bool {
    with_wasm_active_gc(|gc| gc.is_tagged_immediate(addr)).unwrap_or(false)
}

fn wasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_wasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn wasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn wasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

fn wasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_wasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` so backend-agnostic
/// callers (pyre-object `w_int_new`, …) can route through the
/// wasm-owned GC.
fn wasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // See cranelift/dynasm counterparts: host-side allocation must not
    // trigger collection because the caller holds a raw pointer that
    // is not a registered GC root.
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_no_collect_typed(type_id, size))
        .unwrap_or(GcRef(0))
}

/// Host-side old-gen allocation trampoline. Stable
/// across minor/major collections — see dynasm counterpart.
fn wasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_oldgen_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// JIT-trace allocation trampoline target for `New` / `NewWithVtable`.
///
/// A compiled trace cannot allocate directly (the GC lives behind the
/// `WASM_ACTIVE_GC` thread-local), so the `New` codegen routes through the
/// host `jit_call` trampoline, which resolves this function via the module's
/// `__indirect_function_table` (its address is taken in `compile_loop`, so it
/// lands in the table) and invokes it with `(type_id, size)`. Returns the new
/// object pointer, or 0 when no GC is installed. The `ob_type` field for
/// `NewWithVtable` is written inline by codegen at `vtable_offset`.
///
/// Unlike the general [`wasm_alloc_nursery_typed`] host hook (which must not
/// collect — its callers hold unrooted raw pointers), this JIT-trace path is
/// safe to collect: the trace registers every live Ref's frame home slot as a
/// GC root and reloads its locals from the (forwarded) homes after each
/// allocation. So it uses the *collecting* `alloc_nursery_typed`, which
/// triggers a minor collection on nursery-full instead of leaking to old-gen.
pub extern "C" fn wasm_jit_alloc(type_id: i64, size: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_typed(type_id as u32, size as usize).0 as i64)
        .unwrap_or(0)
}

/// JIT-trace variable-size allocation trampoline target for `NewArray` /
/// `NewArrayClear`. Allocates `length` items and writes the length field at
/// `len_offset`, mirroring [`WasmBackend::bh_new_array`].
pub extern "C" fn wasm_jit_alloc_array(
    type_id: i64,
    base_size: i64,
    item_size: i64,
    length: i64,
    len_offset: i64,
) -> i64 {
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    with_wasm_active_gc_mut(|gc| {
        let obj = gc.alloc_varsize_typed(
            type_id as u32,
            base_size as usize,
            item_size as usize,
            length,
        );
        if obj.is_null() {
            0
        } else {
            unsafe {
                *((obj.0 as *mut u8).add(len_offset as usize) as *mut usize) = length;
            }
            obj.0 as i64
        }
    })
    .unwrap_or(0)
}

/// JIT-trace write-barrier trampoline target for ref-storing `SetfieldGc` /
/// `SetarrayitemGc` / `SetinteriorfieldGc`. Routes through the host `jit_call`
/// trampoline; invokes the active GC's `write_barrier`, which adds an old
/// object that may now hold a young reference to the remembered set (and clears
/// TRACK_YOUNG_PTRS). A young base (no flag) or a null base is a no-op. wasm
/// skips the native GC rewrite pass, so the trace emits this barrier directly
/// instead of `COND_CALL_GC_WB`. Returns 0 — the store codegen ignores it.
pub extern "C" fn wasm_jit_write_barrier(obj: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(GcRef(obj as usize)));
    0
}

/// Self-recursive CALL_ASSEMBLER (`PYRE_WASM_CA`) callee-frame allocation
/// helper. Allocates the callee's execution frame as a young nursery
/// GC-managed `JitFrame`, mirroring rewrite.py's nursery frame allocation:
/// steady recursive frames die young, while only frames alive across a
/// collection are promoted. The frame is traced through the jitframe type id's
/// custom trace using its per-frame `jf_gcmap`, rooted by pushing it on the
/// jitframe shadow stack, and reloaded after the recursive call because a
/// nursery frame may move. Returns the frame base (codegen adds
/// `FIRST_ITEM_OFFSET` for the bespoke-layout frame pointer), or 0 on
/// allocation failure.
///
/// Each callee frame self-describes through its own per-frame gcmap, so
/// mixed-geometry frames from distinct CA bridges are each forwarded by their
/// own geometry — no shared coarse single-stride scan that mis-reads a larger
/// frame's interior as a smaller frame's slots.
pub extern "C" fn wasm_jit_ca_alloc_frame(frame_bytes: i64, gcmap_ptr: i64) -> i64 {
    use majit_backend::jitframe::JitFrame;
    assert!(frame_bytes >= 0);
    assert_eq!(frame_bytes as usize % std::mem::size_of::<isize>(), 0);
    let depth = frame_bytes as usize / std::mem::size_of::<isize>();
    // Slice A1: collecting nursery allocation, matching rewrite.py's
    // `gen_malloc_nursery_varsize_frame`. The caller frame remains rooted at
    // the shadow-stack top during a collection; wasm reloads it from there
    // after this call, then this freshly allocated callee is pushed below its
    // own execution. Steady recursive frames die young; only frames that live
    // through a collection are promoted instead of inflating the old-gen major
    // collection threshold on every call.
    let jf_ref = with_wasm_active_gc_mut(|gc| {
        gc.alloc_nursery_typed(wasm_jitframe_tid(), JitFrame::alloc_size(depth))
    })
    .unwrap_or(GcRef(0));
    if jf_ref.0 == 0 {
        return 0;
    }
    let jf = jf_ref.0 as *mut JitFrame;
    unsafe {
        JitFrame::init(jf, std::ptr::null(), depth);
        (*jf).jf_gcmap = gcmap_ptr as *const u8;
    }
    majit_gc::shadow_stack::push_jf(jf_ref);
    jf_ref.0 as i64
}

/// Companion to [`wasm_jit_ca_alloc_frame`]: pop the top jitframe shadow-stack
/// entry on CA-arm exit. The CA recursion is strict LIFO — each level pushes
/// one frame before its `call_indirect` and pops after, and a deopt resume runs
/// on the host's own shadow stack — so removing the top entry releases exactly
/// this callee's frame.
pub extern "C" fn wasm_jit_ca_pop_frame(_frame_base: i64) -> i64 {
    majit_gc::shadow_stack::pop_jf_top();
    0
}

/// Reload the current CA callee frame pointer after a recursive call. The GC
/// may have moved the callee frame during the recursive call; `jf_top_ptr()`
/// reads the forwarded base from the jitframe shadow-stack slot. At this point
/// this recursion level's frame is the top — deeper levels have already popped.
/// Analog of `_reload_frame_if_necessary`; returns the ITEMS base held in the
/// CA arm's `ca_cfp_local`.
pub extern "C" fn wasm_jit_ca_reload_frame() -> i64 {
    majit_gc::shadow_stack::jf_top_ptr().0 as i64
        + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// Reload the CA caller's frame pointer after the callee-frame allocation.
/// The allocation occurs before the callee is pushed, so while the callee is
/// live the caller remains one entry below the shadow-stack top. Returns that
/// caller's ITEMS base for local 0.
pub extern "C" fn wasm_jit_ca_reload_caller_frame() -> i64 {
    majit_gc::shadow_stack::jf_under_top_ptr().0 as i64
        + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// Build the per-frame `jf_gcmap` for a CA callee frame: mark the input slots
/// (at `FRAME_SLOT_BASE`) and the home slots (at `HOME_SLOT_BASE`), in the
/// `JitFrame`'s Signed-granular item indexing (see [`build_home_gcmap`] for the
/// wasm32 layout). The collector's `is_nursery_object_start` gate skips any
/// marked slot that does not hold a live nursery object base, so a slot holding
/// a scalar or an already-promoted Ref is traced harmlessly.
///
/// Returned buffer is leaked by the caller (one per bridge) and lives for the
/// program's life.
fn build_callee_gcmap(
    input_types: &[majit_ir::Type],
    frame: codegen::FrameGeometry,
) -> Box<[usize]> {
    let sign = std::mem::size_of::<isize>();
    let bits_per_word = std::mem::size_of::<usize>() * 8;
    let input_count = input_types.len();
    let mut indices: Vec<usize> = Vec::with_capacity(input_count + frame.home_slots);
    for i in 0..input_count {
        indices.push((codegen::FRAME_SLOT_BASE as usize + i * 8) / sign);
    }
    for h in 0..frame.home_slots {
        indices.push((frame.home_slot_base as usize + h * 8) / sign);
    }
    let max_index = indices.iter().copied().max().unwrap_or(0);
    // `wasm_jit_ca_alloc_frame` sets `jf_frame` from `ca_frame_bytes`, not the
    // full geometry. Inputs and homes must therefore fit that actual item
    // allocation; fail/deopt outputs live in the low value slots and are
    // covered by the same bound.
    debug_assert!(
        max_index < frame.ca_frame_bytes as usize / sign,
        "CA gcmap exceeds the allocated JitFrame item area"
    );
    let num_words = max_index / bits_per_word + 1;
    let mut buf = vec![0usize; 1 + num_words];
    buf[0] = num_words;
    for index in indices {
        buf[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    buf.into_boxed_slice()
}

/// Host-side root-register trampoline.
///
/// # Safety
/// Caller must keep `slot` valid until [`wasm_gc_remove_root`] is
/// called with the same pointer.
unsafe fn wasm_gc_add_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| unsafe { gc.add_root(slot) });
}

/// Companion to [`wasm_gc_add_root`].
fn wasm_gc_remove_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| gc.remove_root(slot));
}

/// Host-side write-barrier trampoline for the interpreter (mapdict / list /
/// set / dict stores route through `majit_gc::gc_write_barrier`). Mirrors
/// `dynasm_gc_write_barrier`; without it every interpreter ref-store is a
/// silent no-op, so a collecting nursery loses old→young pointers.
fn wasm_active_gc_write_barrier(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(obj));
}

/// Host-side `is_managed_heap_object` trampoline.
///
/// This read-only ownership query can fire reentrantly from an extra-root
/// walker mid-collection (the interpreter-safepoint major holds the
/// `WASM_ACTIVE_GC` mutable borrow while asking whether a slot is
/// GC-managed), so:
///   - `Ok(Some)`: a test box is present and free-borrowable → use it.
///   - `Ok(None)`: no box (production) → route to `gc_sync` reentrantly.
///   - `Err`: the test box's mutable borrow is held by an in-progress
///     mutation → reach it through the raw mirror (read-only), or fall
///     back to `gc_sync` if there is no box at all.
fn wasm_gc_owns_object(addr: usize) -> bool {
    match WASM_ACTIVE_GC.with(|cell| {
        cell.try_borrow()
            .map(|g| g.as_deref().map(|gc| gc.is_managed_heap_object(addr)))
    }) {
        Ok(Some(r)) => r,
        Ok(None) => {
            if majit_gc::gc_sync::is_initialized() {
                majit_gc::gc_sync::gc_query_reentrant(|g| g.is_managed_heap_object(addr))
            } else {
                false
            }
        }
        Err(_) => WASM_ACTIVE_GC_RAW.with(|raw| match raw.get() {
            Some(ptr) => unsafe { (*ptr).is_managed_heap_object(addr) },
            None => {
                if majit_gc::gc_sync::is_initialized() {
                    majit_gc::gc_sync::gc_query_reentrant(|g| g.is_managed_heap_object(addr))
                } else {
                    false
                }
            }
        }),
    }
}

pub struct WasmBackend {
    /// `rpython/jit/backend/model.py:28-29 self.tracker =
    /// CPUTotalTracker()` parity — per-instance `cpu.tracker`
    /// exposed via [`majit_backend::Backend::cpu_tracker`].
    cpu_tracker: std::sync::Arc<majit_backend::CpuTotalTracker>,
    trace_counter: u64,
    /// Optimizer constant pool (constant-namespace OpRef → i64 value).
    constants: indexmap::IndexMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset.
    vtable_offset: Option<usize>,
}

/// GC type id of the `JitFrame`. The single registration authority is `eval.rs`
/// (the type is registered there alongside the rest of the heap types, before
/// `freeze_types`); it pushes the id here through `set_wasm_jitframe_tid`,
/// mirroring how it feeds `majit_backend_{cranelift,dynasm}::set_jitframe_gc_type_id`.
/// The orthodox (`PYRE_WASM_CA`) frame path allocates the host-entry frame as a
/// real GC-managed `JitFrame` of this type so the collector forwards its Ref item
/// slots through the `jf_gcmap` custom trace. 0 = not yet pushed (the orthodox
/// path stays disabled until then).
static WASM_JITFRAME_TID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Host entry point used by `eval.rs` to publish the registered `JitFrame` type
/// id (counterpart to `set_jitframe_gc_type_id` on the native backends).
pub fn set_wasm_jitframe_tid(id: u32) {
    WASM_JITFRAME_TID.store(id, std::sync::atomic::Ordering::Relaxed);
}

// Only read on the wasm32 execute_token path and by CA callee-frame allocation.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn wasm_jitframe_tid() -> u32 {
    WASM_JITFRAME_TID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Build a `jf_gcmap` bitmap marking the surviving Ref-home region as the
/// frame's traced GC roots, in the `JitFrame`'s Signed-granular item indexing.
///
/// On wasm32 `isize` is 4 bytes, so `jf_frame` items are 4-byte Signed slots and
/// each 8-byte data slot spans two items — the orthodox PyPy 32-bit layout where
/// a one-word value (a `GcRef`) occupies a single item and a two-word value
/// (i64) occupies a pair. A Ref home written as an i64 keeps the guest pointer in
/// its LOW word (little-endian), at Signed item index `(HOME_SLOT_BASE + h *
/// 8) / sign`. `jitframe_trace` strides items by `sign` and forwards one word per
/// marked bit, so marking those indices exposes each home's `GcRef` (the high
/// word stays unmarked). Returns `[data_word_count, word0, ...]` in `usize`
/// words (GCMAP array layout: `gcmap[0]` = number of data words).
#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        dead_code,
        reason = "native test builds compile the wasm backend without running wasm frame entry"
    )
)]
fn build_home_gcmap(frame: codegen::FrameGeometry) -> Box<[usize]> {
    let sign = std::mem::size_of::<isize>();
    let bits_per_word = std::mem::size_of::<usize>() * 8;
    if frame.home_slots == 0 {
        // One empty data word: a non-null jf_gcmap that traces nothing.
        return vec![1usize, 0usize].into_boxed_slice();
    }
    let last_index = (frame.home_slot_base as usize + (frame.home_slots - 1) * 8) / sign;
    let num_words = last_index / bits_per_word + 1;
    let mut buf = vec![0usize; 1 + num_words];
    buf[0] = num_words;
    for h in 0..frame.home_slots {
        let index = (frame.home_slot_base as usize + h * 8) / sign;
        buf[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    buf.into_boxed_slice()
}

/// `__indirect_function_table` slot of `call_jit::wasm_ca_resume_deopt`,
/// published by pyre-jit at boot (`init_jit_hooks`). When an in-guest
/// self-recursive CALL_ASSEMBLER callee leaves its trace through a guard with no
/// bridge — a deopt the in-guest fast path cannot finish — the CA arm
/// `call_indirect`s this slot to blackhole-resume that callee on the host (no
/// re-execution of its pre-guard work) and read back its result. `0` (unset)
/// makes `compile_bridge` decline the CA lift, since the arm would have no way
/// to complete a deopt. Stored as `u64` to reuse the imported atomics.
static CA_DEOPT_HELPER_SLOT: AtomicU64 = AtomicU64::new(0);

/// Host entry point publishing [`CA_DEOPT_HELPER_SLOT`] (called from pyre-jit's
/// `init_jit_hooks` with `wasm_ca_resume_deopt as *const () as usize`, which on
/// wasm32 is the function's table index).
pub fn set_ca_deopt_helper_slot(slot: u32) {
    CA_DEOPT_HELPER_SLOT.store(slot as u64, Ordering::Relaxed);
}

/// Current CA deopt-helper table slot (0 = unset).
pub fn ca_deopt_helper_slot() -> u32 {
    CA_DEOPT_HELPER_SLOT.load(Ordering::Relaxed) as u32
}

/// A legacy pool-indexed const (`ConstInt(u32)` etc.) reached the wasm backend
/// without a value in the constants pool. `set_constants_pool` runs before
/// `assemble`, so every legitimate legacy const is already present; an arg
/// landing here means the optimizer producer failed to seed it. RPython
/// `ConstInt.value` (history.py:227) is always present, so never register a
/// placeholder `0` — that would emit the constant as zero. Panic at the parity
/// hole, matching the dynasm/cranelift backends.
fn missing_legacy_const(arg: majit_ir::OpRef) -> ! {
    panic!(
        "wasm collect_constants_from_ops: legacy pool-indexed const OpRef \
         (raw={}) is absent from the constants pool — the optimizer producer \
         must seed it (or mint an inline Const) instead of registering 0.",
        arg.raw()
    );
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            cpu_tracker: std::sync::Arc::new(majit_backend::CpuTotalTracker::default()),
            trace_counter: 0,
            constants: indexmap::IndexMap::new(),
            vtable_offset: None,
        }
    }

    /// Active vtable_offset for wasm codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    // `set_constants_pool`, `set_next_trace_id`, and `set_next_header_pc`
    // are provided via the `Backend` trait impl below.

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Mirrors `CraneliftBackend::set_gc_allocator`: stores the box in
    /// the wasm thread-local seam and publishes the same five
    /// `ActiveGcGuardHooks` so the backend-agnostic optimizer /
    /// blackhole executor reach the live allocator without taking a
    /// wasm dependency.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        install_gc_box(gc);
    }

    /// No-op: present for API parity with the dynasm backend so
    /// backend-agnostic consumers can call it uniformly. The wasm `New`
    /// allocation path is out of scope for the GC-routed-New opt-in.
    pub fn set_new_via_gc(&mut self, _enabled: bool) {}

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr.
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_wasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }

    /// Resolve the vtable integer carried by GuardClass /
    /// GuardNonnullClass / GuardSubclass `arg(1)`.
    ///
    /// RPython represents these class operands as `ConstInt` vtable
    /// addresses: `model.py:199-201 cls_of_box()` returns
    /// `ConstInt(ptr2int(obj.typeptr))`, `virtualstate.py:748` builds
    /// `ConstInt(descr.get_vtable())`, and backends read
    /// `op.getarg(1).getint()` (aarch64/regalloc.py:829). Inline ConstInt
    /// carries the value directly (history.py:227 `ConstInt.value`).
    fn const_class_vtable(&self, arg: majit_ir::OpRef) -> Option<i64> {
        arg.const_int_value()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. wasm codegen runs without a
    /// borrow of `self`, so we materialize the resolver as a HashMap.
    /// Only GuardClass / GuardNonnullClass need this table — GuardGcType
    /// already carries an immediate typeid (assembler.py:1919-1922) and
    /// GUARD_IS_OBJECT / GUARD_SUBCLASS use a different lookup path.
    fn collect_classptr_typeid_table(&self, ops: &[Op]) -> HashMap<i64, u32> {
        let mut table = HashMap::new();
        if self.vtable_offset.is_some() {
            return table;
        }
        if with_wasm_active_gc(|_| ()).is_none() {
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.num_args() >= 2
            {
                if let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref()) {
                    if let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize) {
                        table.insert(classptr, tid);
                    }
                }
            }
        }
        table
    }

    /// Pre-fetch `GuardGcTypeInfo` from the installed `gc_ll_descr`.
    ///
    /// Mirrors the `self.cpu.gc_ll_descr.get_translated_info_*` /
    /// `cpu.subclassrange_min_offset` lookups that RPython's
    /// `genop_guard_guard_is_object` (x86/assembler.py:1924-1943) and
    /// `genop_guard_guard_subclass` (x86/assembler.py:1945-1980) do at
    /// codegen time. The returned struct is handed to
    /// `codegen::build_wasm_module`; the codegen arms assert
    /// `supports_guard_gc_type` before reading any other field.
    ///
    /// Also pre-computes `(subclassrange_min, subclassrange_max)` for
    /// every constant classptr argument of a `GuardSubclass` op
    /// (assembler.py:1971-1974 reads these bounds at codegen time).
    fn collect_guard_gc_type_info(&self, ops: &[Op]) -> codegen::GuardGcTypeInfo {
        with_wasm_active_gc(|gc| {
            let mut info = codegen::GuardGcTypeInfo::default();
            info.supports_guard_gc_type = gc.supports_guard_gc_type();
            if !info.supports_guard_gc_type {
                return info;
            }
            // assembler.py:1934-1937: gc_ll_descr lookups.
            let (base, shift, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            info.base_type_info = base;
            info.shift_by = shift;
            info.sizeof_ti = sizeof_ti;
            let (infobits_off, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            info.infobits_offset = infobits_off;
            info.is_object_flag = is_object_flag;
            // assembler.py:1951: cpu.subclassrange_min_offset.
            info.subclassrange_min_offset = gc.subclassrange_min_offset();
            // assembler.py:1971-1974: (subclassrange_min, subclassrange_max)
            // for every constant GuardSubclass arg1.
            for op in ops {
                if op.opcode == majit_ir::OpCode::GuardSubclass && op.num_args() >= 2 {
                    if let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref()) {
                        if let Some(range) = gc.subclass_range(classptr as usize) {
                            info.subclass_ranges.insert(classptr, range);
                        }
                    }
                }
            }
            info
        })
        .unwrap_or_default()
    }

    /// Validate that every constant OpRef appearing as an arg is resolvable.
    ///
    /// Inline-Const variants (`ConstInt`/`ConstFloat`/
    /// `ConstPtr`) carry `.value` on the OpRef itself (history.py:
    /// 227/268/314), so they need no `self.constants` side-table entry and
    /// are skipped. A legacy idx-keyed `ConstInt(u32)` / `ConstFloat(u32)` /
    /// `ConstPtr(u32)` must have been seeded by `set_constants_pool`; one that
    /// is missing is a producer gap and panics rather than defaulting to 0.
    fn collect_constants_from_ops(&mut self, ops: &[Op]) {
        for op in ops {
            for arg in op.getarglist().iter() {
                let arg = arg.to_opref();
                if arg.is_constant()
                    && arg.inline_const_bits().is_none()
                    && !self.constants.contains_key(&arg.raw())
                {
                    missing_legacy_const(arg);
                }
            }
            if let Some(fail_args) = op.getfailargs() {
                for arg in fail_args.iter() {
                    let arg = arg.to_opref();
                    if arg.is_constant()
                        && arg.inline_const_bits().is_none()
                        && !self.constants.contains_key(&arg.raw())
                    {
                        missing_legacy_const(arg);
                    }
                }
            }
        }
    }
}

unsafe impl Send for WasmBackend {}

/// Report why a trace cannot be compiled by the wasm backend, or `None` if it
/// can. Declined traces fall back to the interpreter (correct, unaccelerated)
/// instead of producing an invalid trace module. `is_loop` is true for
/// `compile_loop`, false for `compile_bridge`. `allow_ca` (set by
/// `compile_bridge` only when `PYRE_WASM_CA` is on and the bridge is a
/// self-recursive single-int `CallAssemblerR` shape) lifts the CALL_ASSEMBLER
/// decline so the CA arm (guest→guest `call_indirect`) lowers it instead.
fn wasm_unsupported_trace_reason(ops: &[Op], is_loop: bool, allow_ca: bool) -> Option<String> {
    for op in ops {
        if op.opcode.is_call_assembler() && !allow_ca {
            // CALL_ASSEMBLER inlines a loop-bearing callee by jumping into another
            // trace's compiled token. The wasm backend has no inter-module trace
            // chaining (each trace is its own module), so it cannot execute the
            // target — decline (#62 loop-callee gap).
            return Some(format!(
                "wasm backend: {:?} (loop-callee inline)",
                op.opcode
            ));
        }
    }
    if is_loop {
        // A JUMP with no local LABEL is lowered by codegen (`Jump if !has_loop`)
        // to `return_call_indirect(external_jump_slot)`. Only `compile_bridge`
        // knows the re-entry target (the source loop's table slot) and plumbs it
        // through `external_jump_slot`; `compile_loop` passes 0, so such a trace
        // here is a jump-to-existing-trace (terminal JUMP into a *different*
        // loop) that would tail-call table slot 0 — the wrong function. Decline
        // it; the interpreter performs the cross-loop jump correctly.
        let has_label = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Label);
        let has_jump = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Jump);
        if has_jump && !has_label {
            return Some(
                "wasm backend: loop trace with cross-loop terminal JUMP (no local LABEL)".into(),
            );
        }
    }
    // A JUMP with no local LABEL inside a bridge (a loop-closing bridge) is
    // lowered to a `return_call_indirect` into the source loop's table slot — a
    // wasm tail call — so it is accepted.
    None
}

/// True when `ops` is a bridge whose every `CallAssemblerR` op is a
/// self-recursive single-int call into the source loop (`source_loop_number`):
/// `[Ref, Ref] -> Ref` (the inline-built callee PyFrame + the EC), targeting the
/// SAME token the bridge attaches to. This is the fib recursion shape — the only
/// CA shape the wasm CA arm (codegen) lowers. `.all()` (not `.any()`) keeps a
/// mixed self+foreign or wrong-arity bridge declined.
fn bridge_is_self_recursive_int_ca(ops: &[Op], source_loop_number: u64) -> bool {
    let cas: Vec<&Op> = ops
        .iter()
        .filter(|o| o.opcode.is_call_assembler())
        .collect();
    if cas.is_empty() {
        return false;
    }
    cas.iter().all(|op| {
        op.opcode == majit_ir::OpCode::CallAssemblerR
            && op
                .getdescr()
                .and_then(|d| {
                    d.as_call_descr().map(|cd| {
                        let at = cd.arg_types();
                        cd.call_target_token() == Some(source_loop_number)
                            && at.len() == 2
                            && at[0] == majit_ir::Type::Ref
                            && at[1] == majit_ir::Type::Ref
                            && cd.result_type() == majit_ir::Type::Ref
                    })
                })
                .unwrap_or(false)
    })
}

/// Reconstruct a [`DeadFrame`] from a callee frame an in-guest `call_indirect`
/// already ran to a guard/finish exit (the self-recursive CALL_ASSEMBLER fast
/// path, `PYRE_WASM_CA`). This is the post-`glue::execute` tail of
/// [`WasmBackend::execute_token`] factored for a frame the host did not itself
/// enter: `frame[0]` holds the exit `fail_index`, `frame[1..]` the exit slots,
/// and the pending-exception cell is captured with `jit_exc_take` exactly as
/// `execute_token` does after a GuardNoException / GuardException exit.
/// `pyre-jit`'s `call_jit::wasm_ca_resume_deopt` calls this, then drives the
/// resulting `DeadFrame` through the same `get_latest_descr_arc` /
/// `get_*_value` / `grab_exc_value` Backend path the host's outermost deopt
/// handling uses, so the in-guest deopt completes identically.
///
/// `frame[0]` resolves through the GLOBAL fail-index space
/// (`failguard::global_fail_descr`) — the exit may belong to a bridge chained
/// past the source loop. `_compiled_ptr` (the source loop's metadata address,
/// baked into the CA arm) is kept in the trace ABI but no longer consulted.
pub fn dead_frame_from_ran_frame(_compiled_ptr: usize, frame_ptr: usize) -> DeadFrame {
    let frame = frame_ptr as *const i64;
    let exc_value = jit_exc_take();
    let fail_index = unsafe { *frame } as u32;
    let fail_descr =
        global_fail_descr(fail_index).expect("invalid fail_index from in-guest CA callee frame");
    let num_outputs = fail_descr.fail_arg_types.len();
    let raw_values: Vec<i64> = (0..num_outputs)
        .map(|i| unsafe { *frame.add(1 + i) })
        .collect();
    DeadFrame {
        data: Box::new(WasmFrameData {
            raw_values,
            fail_descr,
            exc_value,
        }),
    }
}

impl majit_backend::Backend for WasmBackend {
    fn cpu_tracker(&self) -> &std::sync::Arc<majit_backend::CpuTotalTracker> {
        &self.cpu_tracker
    }

    fn backend_name(&self) -> &'static str {
        "wasm"
    }

    fn bridge_decline_is_terminal(&self) -> bool {
        // Every `compile_bridge` `Unsupported` return is a deterministic
        // structural decline — a function of the (ops, source-loop) shape that
        // re-tracing the same guard reproduces identically: CALL_ASSEMBLER /
        // cross-loop JUMP shape, missing source loop, loop-closing bridge into a
        // peeled preamble, non-direct loop guard, ref-home overflow, or the
        // codegen frame-slot / unhandled-opcode declines. So re-firing the guard
        // only rebuilds the same unsupported bridge; record it terminal.
        true
    }

    // ── Blackhole allocation (llmodel.py:775-790) ──
    //
    // The blackhole interpreter materializes virtuals (e.g. a virtualized
    // `W_IntObject` loop variable forced at loop exit) through these. Without
    // a real implementation `bhimpl_new*` returns 0 and the resumed frame
    // carries null operands. Mirrors `CraneliftBackend`'s overrides but routes
    // through the wasm thread-local GC; allocation inputs carry no unrooted GC
    // refs, so no collection-suppression beyond the no-collect fixed-size path
    // is required.

    /// llmodel.py:775 bh_new(sizedescr).
    fn bh_new(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        // TODO: get_type_id() returns the u64 path_hash cache key; the GC tid
        // is its low 32 bits until gc_cache routing resolves the real tid.
        let type_id = sizedescr.get_type_id() as u32;
        with_wasm_active_gc_mut(|gc| gc.alloc_nursery_no_collect_typed(type_id, size).0 as i64)
            .unwrap_or(0)
    }

    /// llmodel.py:778-782 bh_new_with_vtable(sizedescr): allocate, then write
    /// the type pointer at `vtable_offset`.
    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let vtable = sizedescr.get_vtable();
        let type_id = sizedescr.get_type_id() as u32;
        let ptr =
            with_wasm_active_gc_mut(|gc| gc.alloc_nursery_no_collect_typed(type_id, size).0 as i64)
                .unwrap_or(0);
        if ptr != 0 && vtable != 0 {
            if let Some(vt_off) = self.vtable_offset {
                unsafe {
                    *((ptr as *mut u8).add(vt_off) as *mut usize) = vtable;
                }
            }
        }
        ptr
    }

    /// llmodel.py:788 bh_new_array(length, arraydescr).
    fn bh_new_array(&self, length: i64, arraydescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let length = usize::try_from(length).expect("bh_new_array length must be non-negative");
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let len_offset = arraydescr
            .array_len_offset()
            .expect("bh_new_array requires ArrayDescr.lendescr");
        let type_id = arraydescr.get_type_id() as u32;
        with_wasm_active_gc_mut(|gc| {
            let obj = gc.alloc_varsize_typed(type_id, base_size, itemsize, length);
            if obj.is_null() {
                0
            } else {
                unsafe {
                    *((obj.0 as *mut u8).add(len_offset) as *mut usize) = length;
                }
                obj.0 as i64
            }
        })
        .unwrap_or(0)
    }

    /// llmodel.py:790 bh_new_array_clear = bh_new_array (allocator zeroes).
    fn bh_new_array_clear(
        &self,
        length: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        self.bh_new_array(length, arraydescr)
    }

    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[OpRc],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        // `x86/assembler.py:514` parity — bump
        // `cpu.tracker.total_compiled_loops` at the same point PyPy
        // creates the `CompiledLoopToken`.
        if let Some(clt) = token.compiled_loop_token.as_ref() {
            majit_backend::record_compiled_loop_token(&self.cpu_tracker, clt);
        }
        let ops_owned: Vec<Op> = ops.iter().map(|rc| (**rc).clone()).collect();
        let ops: &[Op] = &ops_owned;
        // This must use the same direct-vs-trampoline predicates as codegen:
        // a CA callee runs this source-loop body on a movable nursery frame.
        let has_trampoline_calls = codegen::has_trampoline_calls(inputargs, ops, false);

        // Decline traces the wasm backend cannot compile correctly, so the
        // metainterp falls back to the interpreter (correct, if unaccelerated)
        // rather than installing a structurally-invalid trace module:
        //   * CALL_ASSEMBLER inlines a loop-bearing callee by jumping into
        //     another trace's compiled token. The wasm backend has no
        //     inter-module trace chaining (each trace is its own module), so it
        //     cannot execute the target — declining is the #62 loop-callee gap.
        //   * A JUMP with no LABEL targets a *different* existing loop
        //     (jump-to-existing-trace); compile_loop cannot supply the target
        //     table slot, so codegen would tail-call slot 0 — the wrong
        //     function. Declined here (is_loop=true).
        if let Some(reason) = wasm_unsupported_trace_reason(ops, true, false) {
            return Err(BackendError::Unsupported(reason));
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        // Allocation helpers reached from a compiled trace through the host
        // `jit_call` trampoline. `fn as usize` is the `__indirect_function_table`
        // index on wasm32; taking it here keeps the function in the table.
        let alloc_fn_ptr = wasm_jit_alloc as *const () as usize as i64;
        let alloc_array_fn_ptr = wasm_jit_alloc_array as *const () as usize as i64;
        let wb_fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;
        // Freeze this token's generated frame layout at first compilation.
        // Every geometry retains its tail call area for later bridges. CA
        // callee frames use only the homes prefix (`ca_frame_bytes`) below.
        let raw_frame_value_slots = codegen::frame_value_slots(inputargs, ops);
        let raw_num_ref_homes = codegen::count_ref_homes(inputargs, ops);
        let frame = codegen::FrameGeometry::compact(
            raw_frame_value_slots.max(FROZEN_CHAIN_VALUE_SLOTS),
            raw_num_ref_homes.max(FROZEN_CHAIN_REF_HOMES),
        );
        // Exit indices come from the global fail-index space so a cross-trace
        // chain's `frame[0]` resolves regardless of which module wrote it
        // (`failguard::FAIL_DESCR_REGISTRY`).
        let fail_index_base = fail_descr_base();
        let (wasm_bytes, guard_exits, num_ref_homes, bridge_cells_base, bridge_cells_owner) =
            codegen::build_wasm_module(
                inputargs,
                ops,
                &self.constants,
                self.vtable_offset,
                &typeid_table,
                &guard_gc_type_info,
                alloc_fn_ptr,
                alloc_array_fn_ptr,
                wb_fn_ptr,
                nursery_alloc_params(ops).as_ref(),
                fail_index_base,
                0, // external_jump_slot: a loop's JUMP is a local back-edge `br`
                0, // external_jump_key: unused without an external JUMP
                frame,
                codegen::CaParams {
                    // Loops do not emit the CA arm, but they can run on a
                    // nursery CA frame and must reload local 0 after a collect.
                    ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                    jf_top_addr: jf_top_addr(),
                    ..codegen::CaParams::default()
                },
            )?;

        // Build fail descriptors
        let fail_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        register_fail_descrs(&fail_descrs);

        let max_output_slots = guard_exits
            .iter()
            .map(|g| g.fail_arg_refs.len())
            .max()
            .unwrap_or(0)
            .max(inputargs.len());

        // Instantiate via the host binding on wasm32, or store bytes for
        // testing on native (no wasm host available).
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let func_handle = glue::compile_module(&wasm_bytes);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let func_handle = 0u32; // Placeholder — no wasm host available

        // `jit_compile_wasm` returns 0 when the host runtime rejects the emitted
        // module (e.g. a function body exceeding the parser's size limit — a
        // trace within the metainterp `trace_limit` can still overflow it once
        // the optimizer peels/unrolls the loop). Storing a token whose handle is
        // this dead sentinel would let `execute_token` dispatch table slot 0 (not
        // a trace), leaving `frame[0]` unwritten and resolving a wrong exit descr.
        // Decline the compile so the metainterp keeps the interpreter fallback —
        // a backend capability limit, reported like any other unsupported shape.
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        if func_handle == 0 {
            return Err(BackendError::Unsupported(
                "wasm host rejected the compiled trace module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }

        // A peeled loop carries real work before its (last) LABEL — the
        // unrolled first iteration. codegen emits the `loop` at that LABEL, so
        // the preamble runs once on entry and is NOT part of the iterating body.
        // A loop-closing bridge that re-enters through `func_handle` would
        // re-run this preamble; record the shape so `compile_bridge` can decline
        // such a bridge (see `has_preamble` doc on the struct).
        // A peeled loop (the resume-at-LABEL wrapper's shape) — real work before
        // the last LABEL. Computed through the same predicate codegen's wrapper
        // gates on, so the recorded field and the emitted wrapper cannot drift.
        let has_preamble = codegen::is_resumable_peeled(ops);
        // Stamp each LABEL's loop-target descr with its ordinal (0, 1, 2, …) so a
        // loop-closing bridge can recover which label its terminal JUMP targets:
        // the JUMP and the LABEL share the descr by Arc identity, so the ordinal
        // written here is readable from the bridge's JUMP in `compile_bridge`.
        // Pure metadata — emits no wasm bytes, so the module shape is unchanged.
        // Skip a LABEL whose descr is not loop-target-backed (`set_label_block_id`
        // would panic on a non-`AtomicU32` slot).
        let mut label_block_id: u32 = 0;
        let mut label_descrs: Vec<usize> = Vec::new();
        for op in ops.iter() {
            if op.opcode != majit_ir::OpCode::Label {
                continue;
            }
            // Descr identity of each label, in ordinal order, so
            // `compile_bridge` can resolve which of THIS loop's labels a
            // closing JUMP targets by Arc identity (the JUMP and the LABEL
            // share the descr). The stamped `label_block_id` alone cannot: a
            // loop retraced into several specializations re-stamps a shared
            // descr, and every specialization's start label carries ordinal
            // 0 — a bridge targeting ANOTHER specialization's label would
            // otherwise be mis-chained into this one.
            label_descrs.push(
                op.getdescr()
                    .map(|d| std::sync::Arc::as_ptr(&d) as *const () as usize)
                    .unwrap_or(0),
            );
            if let Some(descr) = op.getdescr() {
                if let Some(target) = descr.as_loop_target_descr() {
                    target.set_label_block_id(label_block_id);
                }
            }
            label_block_id += 1;
        }
        // Per-label resume metadata (ordinal order) for `compile_bridge`'s
        // accept condition: a loop-closing bridge may resume at ANY label via
        // the entry `br_table`, provided its JUMP arity matches that label's
        // arg count and the label's args are the complete live set of the
        // trace remainder.
        let label_num_args = codegen::label_arg_counts(ops);
        let label_resume_safe = codegen::label_resume_safety(ops);
        // Per-guard, per-fail-arg induction-advance flags for
        // `compile_bridge`'s livelock check (see `guard_fail_args_advanced`).
        let guard_fail_arg_advanced = guard_fail_args_advanced(ops, &guard_exits);

        // Publish this loop's enterable labels so a loop-closing bridge from
        // ANY loop can chain into them in-module (jump-to-existing-trace). A
        // peeled loop's labels are each enterable through the entry br_table
        // (key = ordinal + 1). A non-peeled loop has no dispatch: only its
        // FIRST label is enterable — through the plain entry (key 0), whose
        // input loader reads `num_inputs` positional slots — and only when
        // the label's arity equals that (the standard loop shape, whose
        // first label's args ARE the inputargs).
        if has_preamble {
            let last = label_descrs.len().saturating_sub(1);
            for (j, &id) in label_descrs.iter().enumerate() {
                if id == 0 {
                    continue;
                }
                publish_label_target(
                    id,
                    LabelTarget {
                        func_handle,
                        key: j as u32 + 1,
                        num_args: label_num_args[j],
                        resume_safe: label_resume_safe[j],
                        is_last_label: j == last,
                        frame,
                    },
                );
            }
        } else if let Some(&id) = label_descrs.first() {
            if id != 0 && label_num_args.first() == Some(&inputargs.len()) {
                publish_label_target(
                    id,
                    LabelTarget {
                        func_handle,
                        key: 0,
                        num_args: inputargs.len(),
                        resume_safe: true,
                        // No real ops precede a non-peeled loop's header, so
                        // an entry re-run lands at the header without any
                        // advancing segment — the livelock check applies.
                        is_last_label: true,
                        frame,
                    },
                );
            }
        }

        let compiled = CompiledWasmLoop {
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp).collect(),
            func_handle,
            fail_descrs: std::cell::RefCell::new(fail_descrs),
            num_inputs: inputargs.len(),
            max_output_slots,
            num_ref_homes,
            frame,
            has_trampoline_calls: std::cell::Cell::new(has_trampoline_calls),
            bridge_cells_base,
            num_guard_cells: guard_exits.len(),
            has_preamble,
            label_descrs,
            guard_fail_arg_advanced,
            bridge_descr_ranges: std::cell::RefCell::new(Vec::new()),
            chained_trace_meta: std::cell::RefCell::new(std::collections::HashMap::new()),
            _bridge_cells_owner: bridge_cells_owner,
            _bridge_owned_cells: std::cell::RefCell::new(Vec::new()),
            ca_active: std::cell::Cell::new(false),
        };

        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: 0,
            code_size: wasm_bytes.len(),
        })
    }

    fn set_constants_pool(&mut self, constants: majit_ir::ConstMap<majit_ir::Const>) {
        self.constants.clear();
        for (&k, c) in constants.iter() {
            self.constants.insert(k, c.as_raw_i64());
        }
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.trace_counter = trace_id;
    }

    // `set_next_header_pc` uses the trait default (no-op) — wasm does
    // not currently honour it.

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[OpRc],
        original_token: &JitCellToken,
        _previous_tokens: &[std::sync::Arc<JitCellToken>],
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        // A bridge is a fresh trace that continues from a source loop's guard
        // exit. Instead of returning that guard's index to the host and
        // round-tripping through the interpreter, the source loop's epilogue
        // `call_indirect`s the bridge in-module (see `codegen` epilogue). The
        // bridge runs in the SOURCE loop's reused frame: the guard spilled its
        // fail args positionally into `frame[1..]`, exactly where the bridge's
        // `build_function` reads its inputs (`inputargs[k].index == k`), so no
        // argument-recovery layout is needed — hence `caller_recovery_layout`
        // and `previous_tokens` are unused.
        let ops_owned: Vec<Op> = ops.iter().map(|rc| (**rc).clone()).collect();
        let ops: &[Op] = &ops_owned;
        diag_bump(0); // compile_bridge entered

        // is_loop=false: a bridge's terminal JUMP with no LABEL is a loop-closing
        // bridge whose re-entry target is plumbed via `external_jump_slot`.
        // Lift the CALL_ASSEMBLER decline for a self-recursive single-int bridge
        // (the fib shape) so the CA arm lowers it to an in-module `call_indirect`
        // into the source loop instead.
        // The CA arm must be able to complete a callee deopt; without the
        // registered `wasm_ca_resume_deopt` slot it could not, so decline the
        // lift (the host round-trip path still handles the CALL_ASSEMBLER).
        let ca_candidate = ca_deopt_helper_slot() != 0
            && bridge_is_self_recursive_int_ca(ops, original_token.number);
        // The CA candidate's `CallAssemblerR` is a dedicated direct arm; all
        // other ops are scanned against their normal emission paths.
        let bridge_has_trampoline_calls =
            codegen::has_trampoline_calls(inputargs, ops, ca_candidate);

        // Decline exception-resume bridges (`GuardException`): the guarded call
        // raised, so the bridge resumes into the exception handler by re-entering
        // the interpreter at the raising bytecode. That re-entry reads the
        // pre-call operand stack, whose constant entries (e.g. a float dividend)
        // live only in the guard's `rd_consts` resume data, not in a spilled
        // frame slot. The compiled bridge reconstructs its state from inputargs
        // (spilled slots) alone and cannot see `rd_consts`, so such a constant
        // materialises as NULL and the re-entered interpreter runs e.g.
        // `truediv(NULL, int)`, raising a spurious `unsupported operand type(s)
        // for /`. Declining routes the deopt through the blackhole interpreter,
        // which rematerialises constants from `rd_consts` (the native path).
        // Non-raising (`GuardNoException`) loop-closing bridges keep their
        // compiled fast path — the CallMayForce bridges wasm relies on for speed
        // are unaffected, so this does not regress them.
        if ops
            .iter()
            .any(|op| op.opcode == majit_ir::OpCode::GuardException)
        {
            return Err(BackendError::Unsupported(
                "wasm backend: exception-resume bridge (GuardException) re-enters \
                 the interpreter with unreconstructable rd_consts stack entries"
                    .into(),
            ));
        }

        // The source guard this bridge attaches to. `fail_index` is its index in
        // the source loop's `fail_descrs` / cell array; `trace_id` identifies the
        // owning trace.
        let source_trace_id = fail_descr.trace_id();
        let source_fail_index = fail_descr.fail_index();

        // Scalars read from the source loop up front, so the immutable borrow of
        // `original_token` is released before the `&mut self` codegen calls.
        let (
            source_guard,
            source_is_direct,
            source_func_handle,
            source_has_preamble,
            source_frame,
            source_input_types,
            source_loop_finish_fi,
            source_compiled_ptr,
            source_ca_active,
            source_has_trampoline_calls,
        ) = {
            let source_loop = original_token
                .compiled
                .as_ref()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm backend: bridge source token has no compiled loop".into(),
                    )
                })?;
            // `fail_index` the source loop's DoneWithThisFrame Finish writes to
            // frame[0] on the base-case return (the CA arm accepts it as a clean
            // callee finish alongside this bridge's own finish).
            let loop_finish_fi = source_loop
                .fail_descrs
                .borrow()
                .iter()
                .find(|d| d.is_finish)
                .map(|d| d.fail_index)
                .unwrap_or(0);
            // Resolve the failing guard's owning trace by the descr's
            // `trace_id`: the source loop itself, or one of the bridges
            // already chained onto it (`chained_trace_meta`) — a NESTED
            // sub-bridge source. Either way the resolution yields the owning
            // trace's guard-cell array, cell count, and the guard's
            // per-fail-arg advance flags. `None` = foreign trace (declined
            // below, diag 3).
            let is_direct = source_trace_id == source_loop.trace_id;
            let guard = if is_direct {
                Some((
                    source_loop.bridge_cells_base,
                    source_loop.num_guard_cells,
                    source_loop
                        .guard_fail_arg_advanced
                        .get(source_fail_index as usize)
                        .cloned()
                        .unwrap_or_default(),
                ))
            } else {
                source_loop
                    .chained_trace_meta
                    .borrow()
                    .get(&source_trace_id)
                    .map(|m| {
                        (
                            m.cells_base,
                            m.num_cells,
                            m.guard_fail_arg_advanced
                                .get(source_fail_index as usize)
                                .cloned()
                                .unwrap_or_default(),
                        )
                    })
            };
            (
                guard,
                is_direct,
                source_loop.func_handle,
                source_loop.has_preamble,
                source_loop.frame,
                source_loop.input_types.clone(),
                loop_finish_fi,
                // Address of the source loop's metadata, baked into the CA arm
                // (opaque cookie in the deopt-helper ABI; `frame[0]` resolution
                // itself goes through the global fail-index space). Same
                // lifetime assumption as `source_func_handle` below: a
                // recompile invalidates this bridge before the loop's
                // `CompiledWasmLoop` is replaced, so the arm is unreachable
                // with a stale pointer.
                source_loop as *const CompiledWasmLoop as usize as u64,
                source_loop.ca_active.get(),
                source_loop.has_trampoline_calls.get(),
            )
        };

        // The failing guard must belong to the source loop or to a bridge
        // already chained onto it, and its per-trace index must have a cell in
        // that trace's array. A foreign descr has no cell to flip; decline so
        // the metainterp keeps the correct interpreter fallback rather than
        // installing an unreachable bridge module.
        let Some((source_cells_base, source_num_cells, source_fail_arg_advanced)) = source_guard
        else {
            diag_bump(3); // declined: source guard's trace is not chained here
            return Err(BackendError::Unsupported(
                "wasm backend: bridge source guard is not a direct loop guard".into(),
            ));
        };
        if source_fail_index as usize >= source_num_cells {
            diag_bump(3);
            return Err(BackendError::Unsupported(
                "wasm backend: bridge source guard index has no dispatch cell".into(),
            ));
        }
        // The CA arm bakes source-LOOP metadata (finish index, compiled ptr);
        // restrict it to direct loop guards. A CA-shaped bridge on a nested
        // guard then fails codegen's CALL_ASSEMBLER handling — a deterministic
        // decline.
        let mut allow_ca = ca_candidate && source_is_direct;
        let ca_trampoline_decline = if allow_ca && source_has_trampoline_calls {
            Some(
                "wasm backend: self-recursive CA source token or chained bridge \
                 uses the host call trampoline",
            )
        } else if allow_ca && bridge_has_trampoline_calls {
            Some("wasm backend: self-recursive CA bridge uses the host call trampoline")
        } else {
            None
        };
        if ca_trampoline_decline.is_some() {
            // Let the ordinary non-CA CALL_ASSEMBLER decline path retain the
            // interpreter fallback, but make this soundness floor observable.
            diag_bump(15);
            allow_ca = false;
        }
        if let Some(reason) = wasm_unsupported_trace_reason(ops, false, allow_ca) {
            diag_bump(1); // declined: CALL_ASSEMBLER
            return Err(BackendError::Unsupported(
                ca_trampoline_decline.unwrap_or(reason.as_str()).to_string(),
            ));
        }

        // A chained bridge executes in the source token's *same* frame. Its
        // offsets are frozen when that token is compiled, so accept it only if
        // its positional spill region and Ref-home region fit exactly within
        // that layout. Declining here preserves the normal blackhole fallback;
        // it is never safe to grow an already-allocated CA frame underneath a
        // later bridge.
        let bridge_value_slots = codegen::frame_value_slots(inputargs, ops);
        let bridge_ref_homes = codegen::count_ref_homes(inputargs, ops);
        if bridge_value_slots > source_frame.value_slots
            || bridge_ref_homes > source_frame.home_slots
            || (source_ca_active && bridge_has_trampoline_calls)
        {
            if source_ca_active && bridge_has_trampoline_calls {
                // Guard exits in a CA-active token execute on movable callee
                // frames. Do not chain a later bridge whose own body would
                // re-enter the stale-pointer host trampoline.
                diag_bump(15);
                return Err(BackendError::Unsupported(
                    "wasm backend: CA-active source cannot chain a bridge that \
                     uses the host call trampoline"
                        .into(),
                ));
            }
            diag_bump(4);
            return Err(BackendError::Unsupported(format!(
                "wasm backend: bridge frame needs values={bridge_value_slots}, homes={bridge_ref_homes}; \
                 source frozen layout has values={}, homes={}",
                source_frame.value_slots, source_frame.home_slots,
            )));
        }

        // A loop-closing bridge (terminal JUMP, no local LABEL) re-enters the
        // source loop through `source_func_handle` — the function entry. For a
        // peeled source loop, entering at the function entry re-runs the preamble
        // (the unrolled first iteration) against the bridge's mid-loop state
        // instead of resuming at the LABEL, so the induction variable never
        // advances: an infinite loop (the wasm chaining hang on nbody / fannkuch).
        //
        // A peeled loop carries the resume-at-LABEL dispatch: the loop-closing
        // JUMP arm sets the frame dispatch key to `target label ordinal + 1`,
        // so re-entering through `source_func_handle` `br_table`s to that
        // label's resume loader — chaining stays in-module. The bridge is
        // accepted when its JUMP's target label is recoverable from the descr,
        // the arities match, and the label's args are the complete live set of
        // the trace remainder (`label_resume_safe`); otherwise decline — the
        // guard then falls back to blackhole resume and
        // `declined_bridge_guards` stops the metainterp re-tracing it.
        // Non-peeled loops (entry == LABEL) re-enter correctly and keep
        // chaining.
        let bridge_is_loop_closing = {
            let has_label = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Label);
            let has_jump = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Jump);
            has_jump && !has_label
        };
        if bridge_is_loop_closing {
            diag_bump(6); // loop-closing shape
        }
        if source_has_preamble {
            diag_bump(7); // source loop has preamble
        }
        // Resolve the terminal JUMP's target label BY DESCR IDENTITY through
        // the `LABEL_TARGETS` registry — the JUMP and its target LABEL share
        // the loop-target descr Arc, and every compiled loop published its
        // enterable labels there. The stamped `label_block_id` ordinal is NOT
        // identity: a retraced loop has several sibling specializations whose
        // start labels all carry ordinal 0, and a bridge legitimately closes
        // into a SIBLING (jump-to-existing-trace) — the registry resolves the
        // owning module's table slot and resume key, so the tail call chains
        // into the RIGHT loop, own or sibling. Decline when the target is
        // unpublished (descr stripped, or its loop declined/was dropped), the
        // JUMP arity differs from the label's arg count (the resume loader
        // reads exactly that many positional frame slots), the label's args
        // are not the complete live set of the target trace's remainder
        // (`resume_safe` — resuming there would read a null local), or the
        // target loop's frozen frame geometry differs from the source frame.
        // A declined guard falls back to blackhole resume and
        // `declined_bridge_guards` stops the metainterp re-tracing it.
        let mut external_jump_key: u32 = 0;
        let mut external_jump_slot: u32 = source_func_handle;
        let mut resumes_at_loop_header = false;
        if bridge_is_loop_closing {
            let closing_jump = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump);
            let target = closing_jump
                .and_then(|j| j.getdescr())
                .map(|d| std::sync::Arc::as_ptr(&d) as *const () as usize)
                .filter(|id| *id != 0)
                .and_then(label_target);
            let arity = closing_jump.map_or(0, |j| j.getarglist().len());
            let accepted_target = match target {
                // Descr stripped, or the target label was never published.
                None => {
                    diag_bump(8);
                    false
                }
                Some(t) if arity != t.num_args => {
                    diag_bump(10); // arity mismatch
                    false
                }
                Some(t) if !t.resume_safe => {
                    diag_bump(9); // label args not the full live set
                    false
                }
                Some(t) if t.frame != source_frame => {
                    diag_bump(4); // target uses different frozen frame offsets
                    false
                }
                Some(t) => {
                    external_jump_key = t.key;
                    external_jump_slot = t.func_handle;
                    resumes_at_loop_header = t.is_last_label;
                    true
                }
            };
            if !accepted_target {
                diag_bump(2); // declined: JUMP target not chainable
                return Err(BackendError::Unsupported(
                    "wasm backend: loop-closing bridge JUMP target is not a \
                     chainable published label"
                        .into(),
                ));
            }
        }

        // A loop-closing bridge carries the source loop's loop-carried state in
        // its terminal JUMP args and tail-calls the loop to iterate again. If no
        // JUMP arg is the result of an induction-advancing arithmetic op — i.e.
        // every loop-carried value is a verbatim input reload, a fresh
        // allocation, or a baked constant — the bridge re-presents byte-identical
        // induction/guard state on every pass, so the loop's exit guard never
        // flips and the loop⇄bridge cycle spins forever (a control-flow
        // livelock at constant stack depth and heap state). Such a bridge is a
        // guard side-trace that omits the loop body's advancing arithmetic; it
        // has no correct in-module resume, so decline it — the guard falls back
        // to blackhole resume and `declined_bridge_guards` stops the metainterp
        // re-tracing it. A genuinely advancing loop-closing bridge (an `i += 1`
        // counter feeding a JUMP arg) passes and keeps chaining.
        //
        // The check only concerns a bridge that lands directly AT the loop
        // header (the target's last label, or the entry of a non-peeled
        // loop): only then can the guard re-fail on byte-identical state. A
        // resume at an EARLIER label executes the segment between that label
        // and the header — the peeled iteration — which advances the state
        // before the loop re-runs, so no advance is required of the bridge
        // itself.
        if bridge_is_loop_closing && resumes_at_loop_header {
            // Bridge input position `k` reads frame slot `k`, where the source
            // guard spilled its k-th fail arg — so an `InputArg` JUMP arg is a
            // verbatim reload of source fail arg `k`. The advance for such an
            // arg may have happened in the SOURCE loop's body before the guard
            // (an `i += 1` preceding the failing branch): the source recorded
            // per-fail-arg whether the value was produced by induction-
            // advancing arithmetic within the failing iteration
            // (`guard_fail_arg_advanced`), so consult that alongside the
            // in-bridge producers.
            let input_pos: std::collections::HashMap<u32, usize> = inputargs
                .iter()
                .enumerate()
                .map(|(k, ia)| (ia.index, k))
                .collect();
            let advances = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump)
                .is_some_and(|jump| {
                    jump.getarglist().iter().any(|arg| match arg {
                        majit_ir::operand::Operand::Op(producer) => {
                            is_inductive_arith(producer.opcode)
                        }
                        majit_ir::operand::Operand::InputArg(ia) => {
                            input_pos.get(&ia.index).is_some_and(|&k| {
                                source_fail_arg_advanced.get(k).copied().unwrap_or(false)
                            })
                        }
                        _ => false,
                    })
                });
            // Loop state carried on the HEAP (a permutation array flipped via
            // setarrayitem, an object field bumped via setfield, a residual
            // call's arbitrary effects) advances the cycle without any JUMP
            // arg showing inductive arithmetic. The shield only exists to
            // refuse PROVABLY static bridges, so any state-mutating op counts
            // as an advance.
            let mutates_heap = ops.iter().any(|op| {
                use majit_ir::OpCode::*;
                op.opcode.is_call()
                    || matches!(
                        op.opcode,
                        SetfieldGc
                            | SetfieldRaw
                            | SetarrayitemGc
                            | SetarrayitemRaw
                            | GcStore
                            | GcStoreIndexed
                            | RawStore
                            | Strsetitem
                            | Unicodesetitem
                    )
            });
            if !advances && !mutates_heap {
                diag_bump(11); // declined: loop-closing bridge advances no loop-carried value
                return Err(BackendError::Unsupported(
                    "wasm backend: loop-closing bridge advances no loop-carried value \
                     (guard side-trace would livelock the chained loop)"
                        .into(),
                ));
            }
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        let alloc_fn_ptr = wasm_jit_alloc as *const () as usize as i64;
        let alloc_array_fn_ptr = wasm_jit_alloc_array as *const () as usize as i64;
        let wb_fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;

        // Self-recursive CALL_ASSEMBLER (PYRE_WASM_CA): the CA arm allocates a
        // fresh callee using the source token's frozen geometry. The earlier
        // frame-fit decline guarantees this bridge uses those same offsets.
        let ca_params = if allow_ca {
            // Per-bridge callee-frame gcmap (real Ref inputs + home Ref slots),
            // leaked to live for the program — each callee frame's `jf_gcmap`
            // points at it.
            let callee_gcmap_ptr =
                Box::leak(build_callee_gcmap(&source_input_types, source_frame)).as_ptr() as i64;
            codegen::CaParams {
                emit_ca: true,
                // `compile_bridge`'s trampoline-decline floor above guarantees
                // no trampoline-lowered op executes on this movable CA callee
                // frame, so its tail call area is never touched.
                callee_frame_bytes: source_frame.ca_frame_bytes,
                loop_finish_fi: source_loop_finish_fi,
                deopt_helper_slot: ca_deopt_helper_slot(),
                source_compiled_ptr,
                ca_alloc_fn_ptr: wasm_jit_ca_alloc_frame as *const () as usize as i64,
                ca_pop_fn_ptr: wasm_jit_ca_pop_frame as *const () as usize as i64,
                ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                ca_reload_caller_fn_ptr: wasm_jit_ca_reload_caller_frame as *const () as usize
                    as i64,
                callee_gcmap_ptr,
                inline: ca_inline_params(source_frame.ca_frame_bytes),
                jf_top_addr: jf_top_addr(),
            }
        } else {
            codegen::CaParams {
                ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                jf_top_addr: jf_top_addr(),
                ..codegen::CaParams::default()
            }
        };

        // This bridge's exit indices come from the global fail-index space,
        // like every trace's (`failguard::FAIL_DESCR_REGISTRY`).
        let base = fail_descr_base();
        let (wasm_bytes, guard_exits, _num_ref_homes, bridge_cells_base, bridge_cells_owner) =
            codegen::build_wasm_module(
                inputargs,
                ops,
                &self.constants,
                self.vtable_offset,
                &typeid_table,
                &guard_gc_type_info,
                alloc_fn_ptr,
                alloc_array_fn_ptr,
                wb_fn_ptr,
                nursery_alloc_params(ops).as_ref(),
                base,
                // A loop-closing bridge's terminal JUMP re-enters the target
                // loop (own or sibling, resolved via `LABEL_TARGETS`) through
                // its table slot via a tail call, resuming at the label
                // `external_jump_key` selects.
                external_jump_slot,
                external_jump_key,
                source_frame,
                ca_params,
            )?;

        // Bridge exit descrs (fail_index already base-offset by build_wasm_module).
        let bridge_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        register_fail_descrs(&bridge_descrs);

        // Register the bridge module into the shared table, then publish its
        // descrs and flip the source guard's cell. Order matters: the descrs
        // must be resolvable (appended) before the cell makes the guard dispatch
        // into the bridge.
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let bridge_slot = glue::compile_module(&wasm_bytes);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let bridge_slot = 0u32;
        // A 0 handle means the host rejected the bridge module (see the
        // `compile_loop` decline). Flipping the source guard's cell to dispatch
        // into slot 0 would tail-call a non-trace; decline instead so the guard
        // keeps its host round-trip (correct, unaccelerated).
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        if bridge_slot == 0 {
            return Err(BackendError::Unsupported(
                "wasm host rejected the compiled bridge module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }
        diag_bump(5); // bridge compiled — chained in-module

        {
            let source_loop = original_token
                .compiled
                .as_ref()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .expect("source loop disappeared between borrows");
            // Append the bridge's exit descrs to the source loop's flat
            // `fail_descrs` and record the slice they occupy, keyed by the
            // source guard's `fail_index`. `compiled_bridge_fail_descr_layouts`
            // / `store_bridge_guard_hashes` use that range to stamp jitcounter
            // hashes onto these bridge-internal guards (compile.py:826-830
            // store_hash). `start` is captured inside the same `borrow_mut`
            // critical section as the `extend`, so the range stays in lockstep
            // with the vec.
            let count = bridge_descrs.len();
            {
                let mut descrs = source_loop.fail_descrs.borrow_mut();
                let start = descrs.len();
                descrs.extend(bridge_descrs);
                source_loop.bridge_descr_ranges.borrow_mut().push((
                    source_trace_id,
                    source_fail_index,
                    start,
                    count,
                ));
            }
            source_loop.record_chained_bridge_trampoline_calls(bridge_has_trampoline_calls);
            // Publish this bridge's own guard-dispatch metadata so a hot guard
            // INSIDE it can chain a nested sub-bridge (same resolution the
            // loop's own guards get, keyed by this bridge's trace_id).
            source_loop.chained_trace_meta.borrow_mut().insert(
                trace_id,
                ChainedTraceMeta {
                    cells_base: bridge_cells_base,
                    num_cells: guard_exits.len(),
                    guard_fail_arg_advanced: guard_fail_args_advanced(ops, &guard_exits),
                },
            );
            // The bridge module lives as long as this source loop, so hand its
            // own cell array (if any) to the loop, freed when the loop drops.
            if let Some(owner) = bridge_cells_owner {
                source_loop._bridge_owned_cells.borrow_mut().push(owner);
            }
            if allow_ca {
                // Freeze this recursion to the CA mechanism: no further bridge
                // chains here (see the decline above the codegen call).
                source_loop.ca_active.set(true);
            }
        }

        // CA dispatch diagnostics (guest `eprintln` is a no-op on wasm32, so
        // route through the BRIDGE_DIAG tallies the host surfaces): 12 = CA bridge
        // cell actually written (loop epilogue will tail into it); 13 = CA bridge
        // but the source loop reserved no bridge cells (cells_base 0) so the guard
        // never dispatches in-module — the recursion stays a host round-trip.
        if allow_ca {
            if source_cells_base != 0 && bridge_slot != 0 {
                diag_bump(12);
            } else {
                diag_bump(13);
            }
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        if source_cells_base != 0 && bridge_slot != 0 {
            // cells[source_fail_index] = bridge_slot — the loop epilogue now
            // tails into this bridge instead of returning to the host.
            let cell = (source_cells_base as usize + source_fail_index as usize * 4) as *mut u32;
            unsafe {
                core::ptr::write(cell, bridge_slot);
            }
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let _ = (source_cells_base, bridge_slot);

        Ok(AsmInfo {
            code_addr: 0,
            code_size: wasm_bytes.len(),
        })
    }

    /// `compile.py:826-830` store_hash relies on a per-guard fail-descr layout
    /// to know which exits are real guards (vs FINISH) and to count them.
    /// `assign_guard_hashes` fetches one jitcounter hash per non-finish guard
    /// from this list, so without it no guard ever gets a hash, `must_compile`
    /// never fires, and a hot guard exit round-trips to the host forever instead
    /// of triggering a bridge. Build one layout per exit from the metainterp
    /// `ResumeGuardDescr` the optimizer stamped on the guard (`meta_descr`); the
    /// wasm backend keeps no machine-code recovery metadata (resume runs through
    /// the frontend `WasmFrameData` path), so the recovery / rd_* / gc-slot
    /// fields stay empty — `merge_backend_exit_layouts` keeps the frontend's own
    /// entry (`or_insert_with`) and only consumes `is_finish` + `source_op_index`.
    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        let trace_id = compiled.trace_id;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    gc_ref_slots: Vec::new(),
                    force_token_slots: Vec::new(),
                    recovery_layout: None,
                    frame_stack: None,
                    rd_numb: meta.and_then(|fd| fd.rd_numb().map(|s| s.to_vec())),
                    rd_consts: meta.and_then(|fd| fd.rd_consts().map(|s| s.to_vec())),
                    rd_virtuals: meta.and_then(|fd| fd.rd_virtuals().map(|s| s.to_vec())),
                    rd_pendingfields: meta.and_then(|fd| fd.rd_pendingfields().map(|s| s.to_vec())),
                }
            })
            .collect();
        Some(layouts)
    }

    /// `compile.py:826-830` store_hash: stamp the jitcounter hashes assigned by
    /// `assign_guard_hashes` onto each guard's metainterp `ResumeGuardDescr`
    /// (`meta_descr`) — the descr `must_compile_with_values` reads the status
    /// from. Same `ResumeDescr`-family + status-0 gate as the native backends.
    fn store_guard_hashes(&self, token: &JitCellToken, hashes: &[u64]) {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let descrs = compiled.fail_descrs.borrow();
        for (i, &hash) in hashes.iter().enumerate() {
            let Some(wfd) = descrs.get(i) else { break };
            let Some(meta) = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr()) else {
                continue;
            };
            if (meta.is_resume_guard() || meta.is_resume_guard_copied()) && meta.get_status() == 0 {
                meta.store_hash(hash);
            }
        }
    }

    /// `compile.py:826-830` store_hash for the guards INSIDE a compiled bridge.
    /// `compile_bridge` appends a bridge's exit descrs to the source loop's flat
    /// `fail_descrs` and records their `(source_fail_index, start, count)` slice
    /// in `bridge_descr_ranges`. Return one layout per descr in that slice so
    /// `assign_bridge_guard_hashes` stamps a jitcounter hash on each non-finish
    /// bridge guard — without it they stay status 0 and collide in jitcounter
    /// bucket 0. `fail_index` is the 0-based position within the bridge's own
    /// exit list (matching the bridge's frontend `exit_layouts` keying and the
    /// native backends' `compiled_bridge_fail_descr_layouts`); `trace_id` is the
    /// bridge's own id, stamped on each appended `WasmFailDescr`.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = original_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        // The most recently chained bridge at this source guard (last range).
        let (start, count) = compiled
            .bridge_descr_ranges
            .borrow()
            .iter()
            .rev()
            .find(|r| r.0 == source_trace_id && r.1 == source_fail_index)
            .map(|&(_, _, start, count)| (start, count))?;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .get(start..start + count)?
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id: wfd.trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    gc_ref_slots: Vec::new(),
                    force_token_slots: Vec::new(),
                    recovery_layout: None,
                    frame_stack: None,
                    rd_numb: meta.and_then(|fd| fd.rd_numb().map(|s| s.to_vec())),
                    rd_consts: meta.and_then(|fd| fd.rd_consts().map(|s| s.to_vec())),
                    rd_virtuals: meta.and_then(|fd| fd.rd_virtuals().map(|s| s.to_vec())),
                    rd_pendingfields: meta.and_then(|fd| fd.rd_pendingfields().map(|s| s.to_vec())),
                }
            })
            .collect();
        Some(layouts)
    }

    /// `compile.py:826-830` store_hash: stamp the hashes `assign_bridge_guard_hashes`
    /// assigned onto the metainterp `ResumeGuardDescr` of each guard inside the
    /// bridge attached at `source_fail_index`. Same `ResumeDescr`-family +
    /// status-0 gate as `store_guard_hashes`; iterates the same slice in the
    /// same order as `compiled_bridge_fail_descr_layouts` so the hash vector
    /// lines up positionally.
    fn store_bridge_guard_hashes(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
        hashes: &[u64],
    ) {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let Some((start, _count)) = compiled
            .bridge_descr_ranges
            .borrow()
            .iter()
            .rev()
            .find(|r| r.0 == source_trace_id && r.1 == source_fail_index)
            .map(|&(_, _, start, count)| (start, count))
        else {
            return;
        };
        let descrs = compiled.fail_descrs.borrow();
        for (i, &hash) in hashes.iter().enumerate() {
            let Some(wfd) = descrs.get(start + i) else {
                break;
            };
            let Some(meta) = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr()) else {
                continue;
            };
            if (meta.is_resume_guard() || meta.is_resume_guard_copied()) && meta.get_status() == 0 {
                meta.store_hash(hash);
            }
        }
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");

        // Host entry allocates the complete frozen geometry, including the tail
        // call area. Chained bridges share these exact offsets; only CA callee
        // frames use the smaller homes prefix (`ca_frame_bytes`).
        let frame_size = (compiled.frame.frame_bytes as usize).div_ceil(8);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            let _ = (frame_size, args);
            panic!("wasm backend execute_token requires a wasm host");
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            // The pending-exception cell is global, unlike the native
            // per-jitframe `jf_guard_exc`. A residual raise on a blackhole
            // resume path (publish_residual_call_exception) writes it outside
            // any trace and nothing clears it, so clear it before running this
            // trace; otherwise jit_exc_take below would surface a stale
            // exception from a previous frame's resume as this trace's.
            jit_exc_clear();

            // Orthodox frame path (PYRE_WASM_CA): run the trace on a real
            // GC-managed `JitFrame` so a collecting allocation forwards the live
            // Ref-home slots through the `jf_gcmap` custom trace, discovered via
            // the jitframe shadow stack — replacing the bespoke add_root-over-
            // homes scheme. The frame is old-gen (non-moving), so the frame
            // pointer held across `glue::execute` never dangles without a reload
            // protocol. The data region (fail_index at 0, inputs/outputs at
            // FRAME_SLOT_BASE, call area, dispatch key, Ref homes) lives in the
            // `jf_frame` items area; passing `jf + FIRST_ITEM_OFFSET` as the wasm
            // frame pointer keeps every local-0-relative codegen access
            // unchanged. (See `build_home_gcmap` for the wasm32 Signed-item
            // layout.)
            if wasm_jitframe_tid() != 0 {
                use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame};
                let sign = std::mem::size_of::<isize>();
                // Data region (frame_size i64 slots) expressed in Signed items.
                let depth = frame_size * 8 / sign;
                let jf_ref =
                    wasm_alloc_oldgen_typed(wasm_jitframe_tid(), JitFrame::alloc_size(depth));
                assert!(jf_ref.0 != 0, "wasm JitFrame allocation failed");
                let jf = jf_ref.0 as *mut JitFrame;
                unsafe { JitFrame::init(jf, std::ptr::null(), depth) };

                // Per-loop gcmap over the surviving Ref-home region. Held in this
                // stack frame (jf_gcmap points at it) until the outputs are read
                // after the trace returns.
                let gcmap = build_home_gcmap(compiled.frame);
                unsafe { (*jf).jf_gcmap = gcmap.as_ptr() as *const u8 };

                let items_base = jf as usize + FIRST_ITEM_OFFSET;
                let fsb = codegen::FRAME_SLOT_BASE as usize;
                for (i, arg) in args.iter().enumerate() {
                    let v = match arg {
                        Value::Int(v) => *v,
                        Value::Float(v) => v.to_bits() as i64,
                        Value::Ref(r) => r.0 as i64,
                        Value::Void => 0,
                    };
                    unsafe { *((items_base + fsb + i * 8) as *mut i64) = v };
                }

                let saved = majit_gc::shadow_stack::push_jf(jf_ref);
                glue::execute(compiled.func_handle, items_base as u32);

                let exc_value = jit_exc_take();
                let fail_index = unsafe { *(items_base as *const i64) } as u32;
                // Global fail-index space: a cross-trace chain may exit through
                // a sibling loop's guard, so `frame[0]` never resolves against
                // this loop's own `fail_descrs`.
                let fail_descr =
                    global_fail_descr(fail_index).expect("invalid fail_index from compiled wasm");
                let num_outputs = fail_descr.fail_arg_types.len();
                let raw_values: Vec<i64> = (0..num_outputs)
                    .map(|i| unsafe { *((items_base + fsb + i * 8) as *const i64) })
                    .collect();

                // Done reading the frame; release it from the jf shadow stack
                // (it becomes collectible) and free the gcmap. Null the frame's
                // jf_gcmap before dropping the gcmap Box: the old-gen frame can
                // outlive this call still marked VISITED (grayed while it was a
                // root at a major-cycle start), reachable through the major
                // gray stack or the remembered set. A later collection would
                // then custom-trace it and read the freed gcmap. A null gcmap
                // makes jitframe_trace forward nothing (jitframe.py:115-116),
                // which is correct here: the outputs were already read out.
                unsafe { (*jf).jf_gcmap = std::ptr::null() };
                majit_gc::shadow_stack::pop_jf_to(saved);
                drop(gcmap);

                return DeadFrame {
                    data: Box::new(WasmFrameData {
                        raw_values,
                        fail_descr,
                        exc_value,
                    }),
                };
            }

            // Legacy host-Vec frame path (default, PYRE_WASM_CA off): fail_index
            // at frame[0], inputs/outputs at frame[1 + i], surviving Ref homes
            // manually rooted across the trace. A home slot only ever holds null
            // (entry init) or a valid GcRef (store-on-def), so forwarding is
            // safe. The path to
            // `wasm_gc_remove_root` is straight-line and the wasm32 build is
            // `panic=abort`, so `glue::execute` cannot unwind and leak roots.
            let mut frame = vec![0i64; frame_size];
            for (i, arg) in args.iter().enumerate() {
                frame[1 + i] = match arg {
                    Value::Int(v) => *v,
                    Value::Float(v) => v.to_bits() as i64,
                    Value::Ref(r) => r.0 as i64,
                    Value::Void => 0,
                };
            }
            let frame_ptr = frame.as_mut_ptr() as usize as u32;
            let home_base = compiled.frame.home_slot_base as usize / 8;
            for h in 0..compiled.frame.home_slots {
                let slot = unsafe { frame.as_mut_ptr().add(home_base + h) } as *mut GcRef;
                unsafe { wasm_gc_add_root(slot) };
            }
            glue::execute(compiled.func_handle, frame_ptr);
            for h in 0..compiled.frame.home_slots {
                let slot = unsafe { frame.as_mut_ptr().add(home_base + h) } as *mut GcRef;
                wasm_gc_remove_root(slot);
            }
            let exc_value = jit_exc_take();
            let fail_index = frame[0] as u32;
            // Global fail-index space (see the CA-path resolution above).
            let fail_descr =
                global_fail_descr(fail_index).expect("invalid fail_index from compiled wasm");
            let num_outputs = fail_descr.fail_arg_types.len();
            let raw_values: Vec<i64> = (0..num_outputs).map(|i| frame[1 + i]).collect();
            DeadFrame {
                data: Box::new(WasmFrameData {
                    raw_values,
                    fail_descr,
                    exc_value,
                }),
            }
        }
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.fail_descr.as_ref()
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn majit_ir::Descr> {
        // `history.py:125` parity — when the optimizer stamped a
        // metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
        // `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
        // `op.descr`, the wasm backend snapshotted it into
        // `WasmFailDescr.meta_descr`.  Forward through that Arc so
        // identity (`Arc::ptr_eq`) matches dynasm/cranelift; otherwise
        // fall back to the backend Arc upcast (synthetic backend-only
        // descrs).
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        if let Some(meta) = data.fail_descr.meta_descr.as_ref() {
            return Arc::clone(meta);
        }
        Arc::clone(&data.fail_descr) as Arc<dyn majit_ir::Descr>
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        data.raw_values[index]
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        f64::from_bits(data.raw_values[index] as u64)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        GcRef(data.raw_values[index] as usize)
    }

    /// llmodel.py:240 grab_exc_value parity: the exception captured when the
    /// trace exited through a GuardNoException / GuardException.
    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        let data = frame
            .data
            .downcast_ref::<WasmFrameData>()
            .expect("not WasmFrameData");
        GcRef(data.exc_value as usize)
    }

    fn clear_stored_exception(&self) {
        crate::jit_exc_clear();
    }

    fn invalidate_loop(&self, _token: &JitCellToken) {
        // No native code to invalidate — wasm modules are immutable.
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::Backend;
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the wasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping.
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x3333_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    /// S0 spike for the Option A wasm-JITFRAME refactor: prove the shared
    /// `MiniMarkGC` forwards a JitFrame's interior Ref item through the
    /// `jf_gcmap` custom-trace when the frame is discovered via the jitframe
    /// shadow stack. This is the exact GC path the orthodox wasm loop would
    /// depend on — a non-moving old-gen JitFrame whose live Ref item slots are
    /// traced by `jf_gcmap` during a minor collection (`do_collect_nursery`
    /// Phase 1c → `trace_and_update_object` → `jitframe_custom_trace`). The
    /// wasm backend has none of the feeders yet; this confirms the collector
    /// side works so the feeders can be built (S1–S3).
    #[test]
    fn jitframe_oldgen_gcmap_minor_forwards_ref_item() {
        use majit_backend::jitframe::{
            FIRST_ITEM_OFFSET, JF_FRAME_OFS, JF_GCMAP_OFS, JitFrame, jitframe_type_info,
        };
        use majit_gc::GcAllocator;

        let mut gc = MiniMarkGC::new();
        let jf_tid = gc.register_type(jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        let depth = 2usize;
        // Non-moving old-gen JitFrame (jitframe_prefer_oldgen()).
        let frame = gc.alloc_oldgen_typed(jf_tid, JitFrame::alloc_size(depth));
        assert_ne!(frame.0, 0, "old-gen JitFrame alloc failed");
        let frame_ptr = frame.0 as *mut JitFrame;
        unsafe { JitFrame::init(frame_ptr, std::ptr::null(), depth) };

        // A fresh nursery object reachable ONLY through the frame's item slot 0.
        let young = gc.alloc_nursery_typed(payload_tid, 16);
        assert_ne!(young.0, 0, "nursery alloc failed");
        let young_before = young.0;
        unsafe {
            let item0 = (frame_ptr as *mut u8).add(FIRST_ITEM_OFFSET) as *mut usize;
            *item0 = young_before;
        }

        // Per-loop gcmap marking item slot 0 as a Ref: [data_word_count, bits].
        // jitframe_trace reads gcmap_lgt at +0, a data word at +GCMAPBASEOFS(8),
        // and maps bit i (of word 0) to jf_frame item i.
        let gcmap: [usize; 2] = [1, 0b1];
        unsafe {
            let gcmap_field = (frame_ptr as *mut u8).add(JF_GCMAP_OFS as usize) as *mut *const u8;
            *gcmap_field = gcmap.as_ptr() as *const u8;
        }

        // Discover the frame the orthodox way: push it on the jitframe shadow
        // stack so Phase 1c traces its interior via the gcmap.
        let saved = majit_gc::shadow_stack::push_jf(frame);
        gc.do_collect_nursery();
        majit_gc::shadow_stack::pop_jf_to(saved);

        // The young object must have been forwarded out of the nursery and the
        // item slot rewritten to its new address — proving the gcmap bit was
        // honored. An untraced slot would still hold young_before (now dangling).
        let item0_after =
            unsafe { *((frame_ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize) };
        assert_ne!(item0_after, 0, "item0 cleared: frame interior not traced");
        assert_ne!(
            item0_after, young_before,
            "item0 not forwarded: gcmap bit was not honored by the collector"
        );
        assert!(
            gc.is_managed_heap_object(item0_after),
            "forwarded item0 is not a live managed object"
        );

        // The old-gen frame must NOT have moved: its length header stays intact
        // in place, so a wasm local holding frame_ptr would remain valid.
        let len_after = unsafe { *((frame_ptr as *const u8).add(JF_FRAME_OFS) as *const isize) };
        assert_eq!(len_after, depth as isize, "old-gen frame moved/corrupted");
    }
}
