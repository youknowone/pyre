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
use std::sync::atomic::{AtomicI64, Ordering};

use failguard::{CompiledWasmLoop, WasmFailDescr, WasmFrameData};
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

fn with_wasm_active_gc<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> Option<R> {
    WASM_ACTIVE_GC.with(|cell| {
        let guard = cell.borrow();
        guard.as_deref().map(f)
    })
}

/// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)` of the GC owned
/// by this thread's wasm backend, or `(0, 0)` if none is installed. Lets a host
/// runner split GC-retained memory from host-heap growth.
pub fn active_gc_heap_stats() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.heap_byte_stats()).unwrap_or((0, 0))
}

/// `majit_gc::CollectOldgenFn` installed by `set_gc_allocator`. Drives the
/// interpreter-safepoint non-moving old-gen major (`gc_interp::safepoint`,
/// default-on on wasm) through the wasm-thread-local GC. Needs mutable access,
/// so it borrows `WASM_ACTIVE_GC` directly rather than via `with_wasm_active_gc`.
/// Mirrors dynasm's `dynasm_collect_oldgen_nonmoving` and cranelift's
/// `collect_oldgen_nonmoving_via_active_runtime`.
fn wasm_collect_oldgen_nonmoving() {
    WASM_ACTIVE_GC.with(|cell| {
        if let Some(gc) = cell.borrow_mut().as_deref_mut() {
            gc.collect_oldgen_nonmoving();
        }
    });
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`.
/// Mirrors cranelift's `check_is_object_via_active_runtime`: dispatches
/// through the wasm-thread-local GC allocator.
fn wasm_check_is_object(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
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
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_nursery_no_collect_typed(type_id, size),
            None => GcRef(0),
        }
    })
}

/// Host-side old-gen allocation trampoline. Stable
/// across minor/major collections — see dynasm counterpart.
fn wasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_oldgen_typed(type_id, size),
            None => GcRef(0),
        }
    })
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
    WASM_ACTIVE_GC.with(|cell| match cell.borrow_mut().as_deref_mut() {
        Some(gc) => gc.alloc_nursery_typed(type_id as u32, size as usize).0 as i64,
        None => 0,
    })
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
    WASM_ACTIVE_GC.with(|cell| match cell.borrow_mut().as_deref_mut() {
        Some(gc) => {
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
        }
        None => 0,
    })
}

/// JIT-trace write-barrier trampoline target for ref-storing `SetfieldGc` /
/// `SetarrayitemGc` / `SetinteriorfieldGc`. Routes through the host `jit_call`
/// trampoline; invokes the active GC's `write_barrier`, which adds an old
/// object that may now hold a young reference to the remembered set (and clears
/// TRACK_YOUNG_PTRS). A young base (no flag) or a null base is a no-op. wasm
/// skips the native GC rewrite pass, so the trace emits this barrier directly
/// instead of `COND_CALL_GC_WB`. Returns 0 — the store codegen ignores it.
pub extern "C" fn wasm_jit_write_barrier(obj: i64) -> i64 {
    WASM_ACTIVE_GC.with(|cell| {
        if let Some(gc) = cell.borrow_mut().as_deref_mut() {
            gc.write_barrier(GcRef(obj as usize));
        }
    });
    0
}

/// Host-side root-register trampoline.
///
/// # Safety
/// Caller must keep `slot` valid until [`wasm_gc_remove_root`] is
/// called with the same pointer.
unsafe fn wasm_gc_add_root(slot: *mut GcRef) {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            unsafe { gc.add_root(slot) };
        }
    });
}

/// Companion to [`wasm_gc_add_root`].
fn wasm_gc_remove_root(slot: *mut GcRef) {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            gc.remove_root(slot);
        }
    });
}

/// Host-side write-barrier trampoline for the interpreter (mapdict / list /
/// set / dict stores route through `majit_gc::gc_write_barrier`). Mirrors
/// `dynasm_gc_write_barrier`; without it every interpreter ref-store is a
/// silent no-op, so a collecting nursery loses old→young pointers.
fn wasm_active_gc_write_barrier(obj: GcRef) {
    WASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            gc.write_barrier(obj);
        }
    });
}

/// Host-side `is_managed_heap_object` trampoline.
fn wasm_gc_owns_object(addr: usize) -> bool {
    WASM_ACTIVE_GC.with(|cell| {
        let guard = match cell.try_borrow() {
            Ok(guard) => guard,
            Err(_) => {
                // The interpreter-safepoint major holds the mutable borrow
                // while its extra-root walker asks whether a slot is
                // GC-managed. Answer the read-only ownership query through the
                // raw mirror rather than panicking on the second borrow.
                return WASM_ACTIVE_GC_RAW.with(|raw| match raw.get() {
                    Some(ptr) => unsafe { (*ptr).is_managed_heap_object(addr) },
                    None => false,
                });
            }
        };
        match guard.as_deref() {
            Some(gc) => gc.is_managed_heap_object(addr),
            None => false,
        }
    })
}

pub struct WasmBackend {
    /// `rpython/jit/backend/model.py:28-29 self.tracker =
    /// CPUTotalTracker()` parity — per-instance `cpu.tracker`
    /// exposed via [`majit_backend::Backend::cpu_tracker`].
    cpu_tracker: std::sync::Arc<majit_backend::CpuTotalTracker>,
    trace_counter: u64,
    /// Optimizer constant pool (constant-namespace OpRef → i64 value).
    constants: majit_ir::VecMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset.
    vtable_offset: Option<usize>,
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
            constants: majit_ir::VecMap::new(),
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
        // gctypelayout.encode_type_shapes_now parity: close the
        // type-registration phase before any compile embeds the
        // type_info_group base address. Mirrors
        // `CraneliftBackend::set_gc_allocator`.
        gc.freeze_types();
        let supports_guard_gc_type = gc.supports_guard_gc_type();
        WASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            *guard = Some(gc);
            let raw = guard.as_deref_mut().map(|gc| gc as *mut dyn GcAllocator);
            WASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
        });
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            check_is_object: Some(wasm_check_is_object),
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
    }

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
        if WASM_ACTIVE_GC.with(|cell| cell.borrow().is_none()) {
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
/// `compile_loop`, false for `compile_bridge`.
fn wasm_unsupported_trace_reason(ops: &[Op], is_loop: bool) -> Option<String> {
    for op in ops {
        if op.opcode.is_call_assembler() {
            // CALL_ASSEMBLER enters another trace's compiled token; the wasm
            // backend has no inter-module trace chaining (#62).
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

impl majit_backend::Backend for WasmBackend {
    fn cpu_tracker(&self) -> &std::sync::Arc<majit_backend::CpuTotalTracker> {
        &self.cpu_tracker
    }

    fn backend_name(&self) -> &'static str {
        "wasm"
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
        WASM_ACTIVE_GC.with(|cell| match cell.borrow_mut().as_deref_mut() {
            Some(gc) => gc.alloc_nursery_no_collect_typed(type_id, size).0 as i64,
            None => 0,
        })
    }

    /// llmodel.py:778-782 bh_new_with_vtable(sizedescr): allocate, then write
    /// the type pointer at `vtable_offset`.
    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let vtable = sizedescr.get_vtable();
        let type_id = sizedescr.get_type_id() as u32;
        let ptr = WASM_ACTIVE_GC.with(|cell| match cell.borrow_mut().as_deref_mut() {
            Some(gc) => gc.alloc_nursery_no_collect_typed(type_id, size).0 as i64,
            None => 0,
        });
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
        WASM_ACTIVE_GC.with(|cell| match cell.borrow_mut().as_deref_mut() {
            Some(gc) => {
                let obj = gc.alloc_varsize_typed(type_id, base_size, itemsize, length);
                if obj.is_null() {
                    0
                } else {
                    unsafe {
                        *((obj.0 as *mut u8).add(len_offset) as *mut usize) = length;
                    }
                    obj.0 as i64
                }
            }
            None => 0,
        })
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
        if let Some(reason) = wasm_unsupported_trace_reason(ops, true) {
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
        let (wasm_bytes, guard_exits, num_ref_homes, bridge_cells_base) =
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
                0, // fail_index_base: a loop owns fail indices [0, num_guards)
                0, // external_jump_slot: a loop's JUMP is a local back-edge `br`
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

        // A peeled loop carries real work before its (last) LABEL — the
        // unrolled first iteration. codegen emits the `loop` at that LABEL, so
        // the preamble runs once on entry and is NOT part of the iterating body.
        // A loop-closing bridge that re-enters through `func_handle` would
        // re-run this preamble; record the shape so `compile_bridge` can decline
        // such a bridge (see `has_preamble` doc on the struct).
        let last_label = ops
            .iter()
            .rposition(|op| op.opcode == majit_ir::OpCode::Label);
        let has_preamble = last_label.is_some_and(|idx| {
            ops[..idx]
                .iter()
                .any(|op| op.opcode != majit_ir::OpCode::Label)
        });

        let compiled = CompiledWasmLoop {
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp).collect(),
            func_handle,
            fail_descrs: std::cell::RefCell::new(fail_descrs),
            num_inputs: inputargs.len(),
            max_output_slots,
            num_ref_homes,
            bridge_cells_base,
            num_guard_cells: guard_exits.len(),
            has_preamble,
        };

        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: 0,
            code_size: wasm_bytes.len(),
        })
    }

    fn set_constants_pool(&mut self, constants: majit_ir::VecMap<u32, majit_ir::Const>) {
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

        // is_loop=false: a bridge's terminal JUMP with no LABEL is a loop-closing
        // bridge whose re-entry target is plumbed via `external_jump_slot`.
        if let Some(reason) = wasm_unsupported_trace_reason(ops, false) {
            return Err(BackendError::Unsupported(reason));
        }

        // The source guard this bridge attaches to. `fail_index` is its index in
        // the source loop's `fail_descrs` / cell array; `trace_id` identifies the
        // owning trace.
        let source_trace_id = fail_descr.trace_id();
        let source_fail_index = fail_descr.fail_index();

        // Scalars read from the source loop up front, so the immutable borrow of
        // `original_token` is released before the `&mut self` codegen calls.
        let (
            source_loop_trace_id,
            source_cells_base,
            source_num_cells,
            source_num_ref_homes,
            source_func_handle,
            source_has_preamble,
            base,
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
            (
                source_loop.trace_id,
                source_loop.bridge_cells_base,
                source_loop.num_guard_cells,
                source_loop.num_ref_homes,
                source_loop.func_handle,
                source_loop.has_preamble,
                source_loop.fail_descrs.borrow().len() as u32,
            )
        };

        // A loop-closing bridge (terminal JUMP, no local LABEL) re-enters the
        // source loop through `source_func_handle` — the function entry. For a
        // peeled source loop that re-runs the preamble (the unrolled first
        // iteration) against the bridge's mid-loop state instead of resuming at
        // the LABEL, so the induction variable never advances: an infinite loop
        // (observed as the wasm chaining hang on nbody / fannkuch). Decline so
        // the guard falls back to blackhole resume; `declined_bridge_guards`
        // then stops the metainterp re-tracing it. Non-peeled loops (entry ==
        // LABEL) re-enter correctly and keep chaining.
        let bridge_is_loop_closing = {
            let has_label = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Label);
            let has_jump = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Jump);
            has_jump && !has_label
        };
        if bridge_is_loop_closing && source_has_preamble {
            return Err(BackendError::Unsupported(
                "wasm backend: loop-closing bridge re-enters a peeled loop (preamble re-run)"
                    .into(),
            ));
        }

        // This simple chaining handles a bridge attached directly to one of the
        // source loop's own guards (the common loop-exit continuation). A nested
        // bridge (source guard living in another bridge) or a foreign descr has
        // no cell in this loop's array; decline so the metainterp keeps the
        // correct interpreter fallback rather than installing an unreachable
        // bridge module.
        if source_trace_id != source_loop_trace_id || source_fail_index as usize >= source_num_cells
        {
            return Err(BackendError::Unsupported(
                "wasm backend: bridge source guard is not a direct loop guard".into(),
            ));
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        let alloc_fn_ptr = wasm_jit_alloc as *const () as usize as i64;
        let alloc_array_fn_ptr = wasm_jit_alloc_array as *const () as usize as i64;
        let wb_fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;

        let (wasm_bytes, guard_exits, num_ref_homes, _bridge_cells_base) =
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
                base,
                // A loop-closing bridge's terminal JUMP re-enters the source loop
                // through its table slot via a tail call.
                source_func_handle,
            )?;

        // The bridge runs in the source loop's fixed-size frame, so it must not
        // address more Ref-home slots than the loop reserved (value/output slots
        // sit below the constant call area and always fit). If it would, decline:
        // the host round-trip path allocates a frame sized for the bridge.
        if num_ref_homes > source_num_ref_homes {
            return Err(BackendError::Unsupported(format!(
                "wasm backend: bridge needs {num_ref_homes} ref homes, source loop has \
                 {source_num_ref_homes}"
            )));
        }

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

        // Register the bridge module into the shared table, then publish its
        // descrs and flip the source guard's cell. Order matters: the descrs
        // must be resolvable (appended) before the cell makes the guard dispatch
        // into the bridge.
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        let bridge_slot = glue::compile_module(&wasm_bytes);
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        let bridge_slot = 0u32;

        {
            let source_loop = original_token
                .compiled
                .as_ref()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .expect("source loop disappeared between borrows");
            source_loop.fail_descrs.borrow_mut().extend(bridge_descrs);
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
                    rd_numb: None,
                    rd_consts: None,
                    rd_virtuals: None,
                    rd_pendingfields: None,
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

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");

        // Allocate frame area large enough for slots + call trampoline area +
        // the Ref-home region. MIN_FRAME_BYTES accommodates the call area at
        // offset 2000+; the Ref-home region (`codegen::HOME_SLOT_BASE`) follows
        // it, one slot per Ref-typed value (`num_ref_homes`).
        let min_slots = codegen::MIN_FRAME_BYTES / 8;
        let base_slots = min_slots.max(1 + compiled.max_output_slots.max(compiled.num_inputs));
        let frame_size = base_slots + compiled.num_ref_homes;
        let mut frame = vec![0i64; frame_size];

        // Write inputs to frame[1..]
        for (i, arg) in args.iter().enumerate() {
            frame[1 + i] = match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            };
        }

        // frame_ptr = byte offset of frame[0] in wasm linear memory
        let _frame_ptr = frame.as_mut_ptr() as usize as u32;

        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            panic!("wasm backend execute_token requires a wasm host");
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            // Register each Ref-home slot (codegen::HOME_SLOT_BASE region) as a
            // GC root so a collecting allocation inside the trace (epic B)
            // forwards the live refs. A home slot only ever holds null (its
            // entry init) or a valid GcRef (store-on-def), so forwarding is
            // always safe without precise liveness.
            //
            // No RAII guard is needed for the removal below: the path from here
            // to `wasm_gc_remove_root` is straight-line (no `?`/early return),
            // and the wasm32 build is `panic=abort`, so `glue::execute` cannot
            // unwind past this frame — a trap aborts the process rather than
            // leaking the roots.
            let home_base = codegen::HOME_SLOT_BASE as usize / 8;
            for h in 0..compiled.num_ref_homes {
                let slot = unsafe { frame.as_mut_ptr().add(home_base + h) } as *mut GcRef;
                unsafe { wasm_gc_add_root(slot) };
            }

            // The pending-exception cell is global, unlike the native
            // per-jitframe `jf_guard_exc`. A residual raise on a blackhole
            // resume path (publish_residual_call_exception) writes it outside
            // any trace and nothing clears it, so clear it before running this
            // trace; otherwise jit_exc_take below would surface a stale
            // exception from a previous frame's resume as this trace's.
            jit_exc_clear();
            glue::execute(compiled.func_handle, _frame_ptr);

            // Companion to the add_root loop above: drop the home-slot roots
            // now the trace has returned (the host frame is freed on return).
            for h in 0..compiled.num_ref_homes {
                let slot = unsafe { frame.as_mut_ptr().add(home_base + h) } as *mut GcRef;
                wasm_gc_remove_root(slot);
            }

            // A GuardNoException / GuardException exit leaves the pending
            // exception in the global slot; capture and clear it here so
            // grab_exc_value surfaces it to the meta-interpreter. Mirrors
            // cranelift `emit_guard_exit`'s `must_save_exception` move into
            // `jf_guard_exc`, done host-side after the trace returns.
            let exc_value = jit_exc_take();

            // Read fail_index from frame[0]. A chained bridge exit writes its
            // own (base-offset) index here, resolved through the same array
            // because `compile_bridge` appended the bridge's descrs to it.
            let fail_index = frame[0] as u32;
            let fail_descr = compiled
                .fail_descrs
                .borrow()
                .get(fail_index as usize)
                .expect("invalid fail_index from compiled wasm")
                .clone();

            // Read output values
            let num_outputs = fail_descr.fail_arg_types.len();
            let raw_values: Vec<i64> = (0..num_outputs).map(|i| frame[1 + i]).collect();

            DeadFrame {
                data: Box::new(WasmFrameData {
                    raw_values,
                    fail_descr: fail_descr.clone(),
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
}
