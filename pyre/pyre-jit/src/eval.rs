//! JIT-enabled evaluation — the sole entry point for JIT execution.
#![allow(non_camel_case_types, non_upper_case_globals)]
//!
//! This module owns the JitDriver, tracing hooks, and compiled-code
//! execution. pyre-interpreter provides the pure interpreter (eval_frame_plain)
//! and the opcode trait implementations on PyFrame.
//!
//! Equivalent to PyPy's `pypyjit/interp_jit.py` — the JIT is injected
//! from outside the interpreter.

use crate::jit::state::{PyreEnv, PyreJitState};
use crate::jit::trace::trace_bytecode;
use pyre_interpreter::PyExecutionContext;
use pyre_interpreter::executioncontext::ActionFlagOps;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{
    PyError, PyResult, StepResult, decode_instruction_for_dispatch, execute_opcode_step,
};
use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;

use majit_backend::Backend;
use majit_gc::GcAllocator;
use majit_gc::trace::TypeInfo;
use majit_ir::{Type, Value};
use majit_metainterp::blackhole::ExceptionState;
use majit_metainterp::{CompiledExitLayout, DetailedDriverRunOutcome, JitState};

/// Host tracer registered with majit-gc so `walk_jf_roots` can reach
/// the interior Ref slots of our libc-allocated jitframes. The
/// collector expects a callback that, given a jitframe payload
/// address, reports each Ref slot pointer via `update`.
///
/// `jitframe_trace` reads `jf_gcmap` to know which of the trailing
/// `jf_frame` slots hold Refs and calls back for each bit.
unsafe fn pyre_libc_jitframe_tracer(obj_addr: usize, update: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe {
        majit_backend::jitframe::jitframe_trace(
            obj_addr as *mut majit_backend::jitframe::JitFrame,
            |slot_ptr| {
                update(slot_ptr as *mut majit_ir::GcRef);
            },
        );
    }
}

/// Bridge pyre-object's `GcAllocHookFn` to `majit_gc::alloc_nursery_typed`.
/// pyre-object deliberately carries no majit-gc dep, so pyre-jit owns
/// the `GcRef` → `*mut u8` conversion.
fn pyre_object_gc_alloc_trampoline(type_id: u32, size: usize) -> *mut u8 {
    majit_gc::alloc_nursery_typed(type_id, size).0 as *mut u8
}

/// Trampoline for stable-address host-side allocations.
/// Routes pyre-object's stable-allocation hook to the backend's
/// `alloc_oldgen_typed`. MiniMark's old-gen is mark-sweep
/// (non-moving), so the returned pointer is safe to hold on the Rust
/// stack across subsequent allocations.
fn pyre_object_gc_alloc_stable_trampoline(type_id: u32, size: usize) -> *mut u8 {
    majit_gc::alloc_oldgen_typed(type_id, size).0 as *mut u8
}

/// Trampoline for *collecting* nursery host-side allocations — routes
/// pyre-object's collecting-allocation hook to the backend's collecting nursery
/// allocator (minor-on-full). Only the elidable bigint payload helpers use it,
/// from a gcmap-carrying residual call holding no unrooted pointer across the
/// allocation, so the embedded minor cycle is safe.
fn pyre_object_gc_alloc_collecting_trampoline(type_id: u32, size: usize) -> *mut u8 {
    majit_gc::alloc_nursery_collecting_typed(type_id, size).0 as *mut u8
}

/// Trampoline for off-heap memory-pressure charges — routes pyre-object's
/// memory-pressure hook to the backend's GC. The bignum collecting-alloc site
/// charges its limb-`Vec` bytes here so minor cadence reflects true footprint;
/// the charge may force a minor, safe because the caller is the same gcmap-rooted
/// residual call as [`pyre_object_gc_alloc_collecting_trampoline`].
fn pyre_object_gc_charge_memory_pressure_trampoline(bytes: usize) {
    majit_gc::charge_memory_pressure(bytes);
}

/// Old-gen external-byte charge trampoline. Bridges a host stable bignum alloc
/// (`alloc_bigint_stable`) to the active backend's major threshold without
/// forcing a minor.
fn pyre_object_gc_charge_oldgen_external_trampoline(obj_addr: usize, bytes: usize) {
    majit_gc::charge_oldgen_external(obj_addr, bytes);
}

/// `gc.collect()` (interp_gc.py:7-26) trampoline. Bridges
/// pyre-object's `try_gc_collect` to `majit_gc::collect_full`, which
/// fans out to the active backend's `dynasm_collect_full` /
/// `collect_full_via_active_runtime`. pyre-object intentionally has
/// no majit-gc dep, hence the indirection lives here.
///
/// # Safety hazard (documented gap)
///
/// `do_collect_full` always runs a minor cycle first; the nursery is
/// moving. Any live PyObjectRef held on the Rust stack of the
/// bytecode interpreter that is NOT registered as a GC root (via
/// `pyframe_root_walker` / shadow stack / `try_gc_add_root`) will
/// dangle after collection. pyre's interpreter has no shadowstack
/// pass and does not register every per-handler temporary, so a
/// user-triggered `gc.collect()` from a JIT-initialised context can
/// segfault on the next memory access. The trampoline is wired up, but
/// safe enablement is not yet implemented: it requires a shadowstack
/// pass that registers every live PyObjectRef as a GC root.
fn pyre_object_gc_collect_trampoline() {
    majit_gc::collect_full();
}

/// Non-moving old-gen-only major trampoline for the interpreter GC safepoint.
/// Bridges pyre-object's `try_gc_collect_oldgen` to
/// `majit_gc::collect_oldgen_nonmoving`. Unlike the full-collect trampoline it
/// runs no minor, so it reclaims stable-allocated interp int/float without
/// moving the nursery — safe to fire under an active JIT.
fn pyre_object_gc_collect_oldgen_trampoline() {
    majit_gc::collect_oldgen_nonmoving();
}

fn pyre_object_gc_set_enabled_trampoline(enabled: bool) {
    majit_gc::gc_set_enabled(enabled);
}

fn pyre_object_gc_register_finalizer_trampoline(
    fq_index: usize,
    obj: pyre_object::PyObjectRef,
    trigger: pyre_object::gc_hook::GcFinalizerTriggerFn,
) {
    majit_gc::gc_register_finalizer(fq_index, majit_ir::GcRef(obj as usize), trigger);
}

fn pyre_object_gc_finalizer_next_dead_trampoline(fq_index: usize) -> pyre_object::PyObjectRef {
    majit_gc::gc_fq_next_dead(fq_index)
        .map(|obj| obj.0 as pyre_object::PyObjectRef)
        .unwrap_or(pyre_object::PY_NULL)
}

/// Jitframe-empty trampoline for the interpreter GC safepoint. Bridges
/// pyre-object's hook to `majit_gc::jitframe_shadow_stack_empty`, so the
/// safepoint can skip collecting while a compiled trace is suspended.
fn pyre_object_gc_jitframe_empty_trampoline() -> bool {
    majit_gc::jitframe_shadow_stack_empty()
}

/// Trampoline: register a caller-owned slot as
/// a GC root with the active backend. Bridges `*mut *mut u8` (the
/// pyre-object-facing shape that does not depend on majit-gc) to
/// `*mut GcRef` expected by `majit_gc::gc_add_root`. `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the pointer-pointer and
/// `*mut GcRef` share representation.
///
/// # Safety
/// Caller must keep `slot` valid until
/// [`pyre_object_gc_remove_root_trampoline`] is called with the same
/// pointer.
unsafe fn pyre_object_gc_add_root_trampoline(slot: *mut *mut u8) {
    unsafe { majit_gc::gc_add_root(slot as *mut majit_ir::GcRef) };
}

/// Companion to [`pyre_object_gc_add_root_trampoline`].
fn pyre_object_gc_remove_root_trampoline(slot: *mut *mut u8) {
    majit_gc::gc_remove_root(slot as *mut majit_ir::GcRef);
}

struct FrameLocalsRoot {
    slot: *mut *mut u8,
    registered: bool,
}

/// A forwarding root for the red `frame` argument while native JIT glue is
/// active.  `PyFrame.dispatch` keeps this identity as a GC reference; Rust
/// callback parameters are raw addresses and must be reloaded after a moving
/// collection.
struct FrameRoot {
    depth: usize,
}

impl FrameRoot {
    #[majit_macros::dont_look_inside]
    fn new(frame: &mut PyFrame) -> Self {
        let depth = majit_gc::shadow_stack::push(majit_ir::GcRef(frame as *mut PyFrame as usize));
        Self { depth }
    }

    #[majit_macros::dont_look_inside]
    fn frame(&mut self) -> &mut PyFrame {
        let frame = majit_gc::shadow_stack::get(self.depth).0 as *mut PyFrame;
        unsafe { &mut *frame }
    }

    #[majit_macros::dont_look_inside]
    fn release(&mut self) {
        majit_gc::shadow_stack::try_pop_to(self.depth);
    }
}

impl Drop for FrameRoot {
    fn drop(&mut self) {
        self.release();
    }
}

impl FrameLocalsRoot {
    fn new(frame: &mut PyFrame) -> Self {
        let slot = &mut frame.locals_cells_stack_w as *mut _ as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

/// Bridge pyre-object's `is_managed_heap_object` query to
/// `majit_gc::gc_owns_object`. Used by host-side allocators
/// (`pyre_object::dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks.
fn pyre_object_gc_owns_object_trampoline(addr: usize) -> bool {
    majit_gc::gc_owns_object(addr)
}

fn pyre_object_gc_current_object_address_trampoline(addr: usize) -> usize {
    majit_gc::gc_current_object_address(addr)
}

fn pyre_object_gc_identity_hash_trampoline(addr: usize) -> usize {
    majit_gc::gc_id_or_identityhash(addr)
}

fn pyre_object_gc_write_barrier_trampoline(obj: *mut u8) {
    majit_gc::gc_write_barrier(majit_ir::GcRef(obj as usize));
}

/// `pypy/objspace/std/dictmultiobject.py:1209 ObjectDictStrategy` key
/// equality bridge: ObjectDictStrategy stores its dstorage as
/// `r_dict(space.eq_w, space.hash_w)` so user `__eq__` is honoured on
/// lookup.  pyre-object cannot depend on pyre-interpreter for the
/// dispatch, so this trampoline routes through
/// `pyre_interpreter::baseobjspace::eq_w` (line-by-line port of
/// `baseobjspace.py:823-825 W_ObjectSpace.eq_w`).  Registered at
/// JIT init so all subsequent `dict_keys_equal` calls reach the full
/// comparison protocol.  A raising `__eq__` (or `__bool__` of its
/// result) cannot return a `Result` across the bucket probe, so the
/// `PyError` is stashed on the shared pending slot and flagged via
/// `dict_eq_hook::signal_eq_error`; the checked dict op converts the
/// flag to a `DictKeyError` after the probe.
unsafe fn pyre_object_eq_w_trampoline(
    a: pyre_object::PyObjectRef,
    b: pyre_object::PyObjectRef,
) -> bool {
    match pyre_interpreter::baseobjspace::eq_w(a, b) {
        Ok(v) => v,
        Err(e) => {
            pyre_interpreter::baseobjspace::set_pending_hash_error(e);
            pyre_object::dict_eq_hook::signal_eq_error(a);
            false
        }
    }
}

/// `pypy/objspace/std/dictmultiobject.py:1210 r_dict(space.eq_w,
/// space.hash_w)` hash bridge: ObjectDictStrategy uses both eq_w and
/// hash_w; pyre's `dict_keys_equal` enforces the bucket invariant
/// (same eq_w + same hash_w → same key, different hash_w → distinct).
/// Routes through `try_hash_value` (the strict Result-bearing hash)
/// so unhashable types, user `__hash__ = None`, and user `__hash__`
/// exceptions are all caught.  On error, signals via
/// `dict_eq_hook::signal_hash_error` and stores the `PyError` in
/// `PENDING_HASH_ERROR` for the caller to retrieve.
unsafe fn pyre_object_hash_w_trampoline(obj: pyre_object::PyObjectRef) -> i64 {
    match pyre_interpreter::builtins::try_hash_value(obj) {
        Ok(h) => h,
        Err(e) => {
            pyre_interpreter::baseobjspace::set_pending_hash_error(e);
            pyre_object::dict_eq_hook::signal_hash_error(obj);
            0
        }
    }
}

/// `space.hash_w` for a `str` straight from its WTF-8 bytes — the str-keyed
/// `getitem_str` companion to [`pyre_object_hash_w_trampoline`], so a str-key
/// dict probe lands in the same bucket without building a `W_UnicodeObject`.
/// `ptr`/`len` describe a valid WTF-8 range for the duration of the call.
unsafe fn pyre_object_hash_str_trampoline(ptr: *const u8, len: usize) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    pyre_interpreter::builtins::hash_str_bytes(bytes)
}

/// `pypy/objspace/std/typeobject.py:353-371
/// W_TypeObject.compares_by_identity` trampoline.  Routes through
/// `pyre_interpreter::baseobjspace::compares_by_identity` which
/// walks the MRO and caches the result on
/// `W_TypeObject.compares_by_identity_status`.  Registered at
/// JIT init so `EmptyDictStrategy::switch_to_correct_strategy`
/// (`dictmultiobject.py:702-705`) reaches the full
/// `__eq__`/`__hash__` resolution.
unsafe fn pyre_object_compares_by_identity_trampoline(w_type: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_interpreter::baseobjspace::compares_by_identity(w_type) }
}

/// Custom trace for `W_TypeObject`.
///
/// Forwards every GC-reachable edge a heap type owns so that, once heap
/// types are GC-managed, a type kept live by reachability keeps its own
/// children live (`typeobject.py:176-180` `_immutable_fields_` lists
/// `'mro_w?[*]'`, `'bases_w?[*]'`, the namespace `dict_w`, `terminator`):
///
///   * `ob_header.w_class` — the metaclass, the type's own class edge
///     (the inline header word, same as `object_object_custom_trace`).
///   * `bases` — the movable bases tuple.
///   * `mro_w` — the out-of-line MRO type list.
///   * `weak_subclasses` — the out-of-line list populated by
///     `w_type_ready` / `add_subclass` (`typeobject.py:373-377`,
///     `:640-662`).  Each slot is a strong root to the WEAKREF GcStruct
///     itself — its `weakptr` payload is invalidated separately by the
///     collector's `invalidate_young_weakrefs` / `invalidate_old_weakrefs`
///     (incminimark.py:3058-3126), so passing the slot to `f` keeps the
///     WEAKREF alive without forcing the target alive.
///   * the managed namespace object for heap types, or the off-GC
///     `DictStorage` values for static builtin types.
///
/// Heap types are stable old-gen GC objects, so this trace keeps their owned
/// GC edges live and forwards their slots.  The separate builtin-type walk
/// covers Box-immortal builtin types, whose custom trace never fires.
unsafe fn type_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let t = unsafe { &mut *(obj_addr as *mut pyre_object::typeobject::W_TypeObject) };
    f(&mut t.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut t.bases as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !t.mro_w.is_null() {
        if pyre_object::gc_hook::try_gc_owns_object(t.mro_w as *mut u8) {
            // GC-owned type-9 block: forward the `mro_w` field slot; the
            // varsize walker forwards items[0..len]. Forwarding each element
            // instead would mark the elements but leave the block itself
            // unmarked, so a major collection would sweep it (UAF). Mirrors
            // `list_object_custom_trace`'s GC-owned branch.
            let mro_slot = std::ptr::addr_of_mut!(t.mro_w);
            f(mro_slot as *mut majit_ir::GcRef);
        } else {
            // std::alloc fallback block (no GC hook): forward each element in
            // place — the block is stationary and the collector does not own it.
            let mro = unsafe { &mut *t.mro_w };
            for slot in mro.as_mut_slice().iter_mut() {
                f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            }
        }
    }
    if !t.weak_subclasses.is_null() {
        let subs = unsafe { &mut *t.weak_subclasses };
        for slot in subs.iter_mut() {
            f(slot as *mut *mut pyre_object::weakref::Weakref as *mut majit_ir::GcRef);
        }
    }
    f(&mut t.dict as *mut *mut u8 as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
}

/// Reclaim the two Rust-owned, out-of-line containers of a swept heap type.
/// `mro_w` is a GC-managed `FixedObjectArray` reclaimed by the collector and
/// must not be freed here. The managed namespace object is also reclaimed by
/// the collector, while the shared/uncertain `terminator` ownership remains
/// deferred by #528.
unsafe fn type_object_destructor(obj_addr: usize) {
    let t = obj_addr as *const pyre_object::typeobject::W_TypeObject;
    let name = unsafe { (*t).name };
    if !name.is_null() {
        drop(unsafe { Box::from_raw(name) });
    }
    let weak_subclasses = unsafe { (*t).weak_subclasses };
    if !weak_subclasses.is_null() {
        drop(unsafe { Box::from_raw(weak_subclasses) });
    }
}

/// Custom trace for `GeneratorIterator` (generator.py GeneratorIterator).
///
/// The suspended frame is held behind an opaque `frame_ptr`
/// (`Box<PyFrame>`, off the active `CURRENT_FRAME` chain), so none of its
/// slots are reachable from `walk_pyframe_roots`.  Forward the suspended
/// frame's own GC slots — pycode, the locals/cells/valuestack array and
/// its elements, the generator/yield-from slots, the globals/builtin
/// object pointers, and the debug-data locals — through
/// `walk_suspended_generator_frame` so a value live only through a
/// suspended generator (e.g. a local held across a `yield` while
/// `gc.collect()` runs) is not reclaimed.
unsafe fn generator_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let gen_obj = unsafe { &mut *(obj_addr as *mut pyre_object::generator::GeneratorIterator) };
    f(&mut gen_obj.name as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut gen_obj.qualname as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !gen_obj.frame_ptr.is_null() {
        let frame = gen_obj.frame_ptr as *mut PyFrame;
        if pyre_object::gc_hook::try_gc_owns_object(gen_obj.frame_ptr) {
            // GC-managed suspended frame: forward the `frame_ptr` slot as a
            // managed edge so mark greys the frame block, and its own
            // `pyframe_object_custom_trace` recursively forwards the
            // locals/cells/valuestack — keeping both the block and its
            // contents live without sweeping the frame the generator holds.
            f(&mut gen_obj.frame_ptr as *mut *mut u8 as *mut majit_ir::GcRef);
        } else {
            // `std::alloc` fallback frame (no GC hook at generator birth):
            // the block is never swept, so only its contents need
            // forwarding, in place.
            let mut adapter = |slot: &mut majit_ir::GcRef| f(slot as *mut majit_ir::GcRef);
            pyre_interpreter::eval::walk_suspended_generator_frame(frame, &mut adapter);
        }
    }
}

/// Custom trace for `PyTraceback` (`pytraceback.py:17 PyTraceback`).
///
/// PyPy's `PyTraceback.frame` is a normal `PyFrame` W_Root, so its
/// tracer reaches the frame (and thence its locals / `f_backref`
/// chain) through the ordinary reference.  Pyre stores `frame` as a
/// raw `*mut PyFrame` and the two `PyObjectRef` slots (`w_next`,
/// `w_code`) inline; none are reachable through `gc_ptr_offsets`
/// (the type was registered with empty offsets), so forward all three
/// here:
///
///   * `w_next` — the chained caller-side traceback link.
///   * `w_code` — the raising frame's PyCode snapshot (kept alive so
///     source-path / function-name metadata survives).
///   * `frame` — forwarded as a managed edge **only** when
///     `try_gc_owns_object(frame)` holds (an executing/oldgen frame,
///     `PYFRAME_GC_TYPE_ID`).  Mark greys the frame block and its own
///     `pyframe_object_custom_trace` recurses into locals/cells/
///     valuestack and the `f_backref` chain, so a frame reachable only
///     through a live traceback (the whole point of `tb_frame`) is not
///     reclaimed.  A non-Gc frame (Box tracer snapshot / arena callee,
///     already freed by the time the traceback escapes) is left
///     dangling exactly as before — never dereferenced.
unsafe fn pytraceback_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    let tb = unsafe { &mut *(obj_addr as *mut pyre_interpreter::pytraceback::PyTraceback) };
    f(&mut tb.w_next as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut tb.w_code as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    if !tb.frame.is_null() && pyre_object::gc_hook::try_gc_owns_object(tb.frame as *mut u8) {
        f(&mut tb.frame as *mut *mut pyre_interpreter::pyframe::PyFrame as *mut majit_ir::GcRef);
    }
}

unsafe fn dict_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    // Strategy-side dispatch — `W_DictObject.dstorage: *mut u8` erases
    // the storage layout, so each strategy walks its own native shape
    // through `DictStrategy::walk_gc_refs` (`dictmultiobject.rs`).  PyPy's
    // counterpart is the per-`rerased`-pair GC trace fn generated from
    // `new_erasing_pair("name")` at translation time
    // (`rpython/rlib/rerased.py:24-72`); the trait method is pyre's
    // runtime dispatch equivalent.
    let w_dict = obj_addr as pyre_object::PyObjectRef;
    let strategy = unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(w_dict) };
    let mut adapter = |slot: *mut pyre_object::PyObjectRef| {
        f(slot as *mut majit_ir::GcRef);
    };
    unsafe { strategy.walk_gc_refs(w_dict, &mut adapter) };
}

/// Reclaim the Rust-owned erased storage container of a swept regular dict.
/// The strategy reconstructs the exact Box type; contained PyObjectRefs remain
/// collector-owned.  MapDictStrategy's dstorage is a borrowed GC edge and its
/// strategy deallocator is deliberately a no-op.
unsafe fn dict_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    let dict = unsafe { &*(obj as *const pyre_object::dictmultiobject::W_DictObject) };
    if !dict.dstorage.is_null() {
        unsafe { dict.dstrategy.dealloc_storage(obj) };
    }
}

/// Sweep-time destructor for `W_ModuleDictObject`: reclaim the three
/// off-GC storage Boxes (`dstorage`/`mstrategy`/`object_storage`) the GC
/// does not own.  Mirrors `dict_object_destructor`.
unsafe fn module_dict_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    unsafe { pyre_object::dictmultiobject::w_module_dict_dealloc_storage(obj) };
}

/// Reclaim the off-GC byte buffer of a swept bytes object.
unsafe fn bytes_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    unsafe { pyre_object::bytesobject::w_bytes_dealloc(obj) };
}

/// Reclaim the off-GC byte buffer of a swept bytearray object.
unsafe fn bytearray_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    unsafe { pyre_object::bytearrayobject::w_bytearray_dealloc(obj) };
}

/// Reclaim the off-GC item container of a swept set object.
unsafe fn set_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    unsafe { pyre_object::setobject::w_set_dealloc_items(obj) };
}

/// Reclaim the off-GC name string of a swept function object.
unsafe fn function_object_destructor(obj_addr: usize) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    unsafe { pyre_interpreter::function::function_dealloc_name(obj) };
}

/// Custom trace for `W_ObjectObject` (instance `map`+`storage`,
/// `mapdict.py:907-910`).  The `storage` list is an off-GC
/// `Box<Vec<PyObjectRef>>`, so — exactly as `dict_object_custom_trace`
/// reaches the off-GC dict entries — this forwards each boxed
/// attribute-value slot in place via `instance_walk_boxed_storage`,
/// which consults the map to skip erased unboxed (`Vec<i64>`) slots
/// (`mapdict.py:438/447` boxed `erase_item` vs `:601/612`
/// `erase_unboxed`).  The off-GC `Vec` stays put; only its
/// `PyObjectRef` contents are relocated.
///
/// `ob_header.w_class` is the instance's class reachability edge — the
/// equivalent of PyPy reaching the class through the traced
/// `terminator.w_cls` (`mapdict.py:751-752`, a strong `_immutable_field_`).
/// Pyre stores the class in the inline header word
/// (`objectobject.rs:24`, `typeptr` in `rclass.py`), so it must be
/// forwarded here or an instance whose class is reachable only through
/// it would have that class reclaimed once heap types become
/// GC-managed.  Inert while heap types remain `malloc_typed`
/// Box-immortal — the visitor's `is_in_nursery` / `is_managed_heap_object`
/// guard skips the non-managed type pointer — exactly as
/// `generator_object_custom_trace` forwards `pycode` ahead of the
/// code-object migration.
unsafe fn object_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let obj = obj_addr as pyre_object::PyObjectRef;
    let inst = unsafe { &mut *(obj_addr as *mut pyre_object::objectobject::W_ObjectObject) };
    f(&mut inst.ob_header.w_class as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    // Mark the `storage` block (`W_MAPDICT_STORAGE_GC_TYPE_ID`, a stable leaf
    // GcArray) live: forward the block-pointer field slot itself so a major GC
    // greys the block (its interior is a GC leaf, so the collector never walks
    // it — this instance's walk below is the only thing that forwards the boxed
    // element slots). Non-moving, so the minor-GC forward is a no-op; the value
    // is keeping the block off the sweep list. Mirrors `list_object_custom_trace`
    // forwarding `int_items.block` / `float_items.block`. Guard on GC ownership:
    // a `std::alloc` fallback block (no GC hook) is not GC-managed.
    if !inst.storage.is_null() && pyre_object::gc_hook::try_gc_owns_object(inst.storage as *mut u8)
    {
        let storage_slot = std::ptr::addr_of_mut!(inst.storage);
        f(storage_slot as *mut majit_ir::GcRef);
    }
    pyre_interpreter::objspace::std::mapdict::instance_walk_boxed_storage(
        obj,
        &mut |slot: *mut pyre_object::PyObjectRef| {
            f(slot as *mut majit_ir::GcRef);
        },
    );
}

/// Custom trace for `W_ModuleDictObject`
/// (`dictmultiobject.py:328 W_ModuleDictObject`).
///
/// PyPy's tracer follows `W_DictMultiObject.dstorage` (a real
/// RPython `{str: cell_or_value}` dict) plus
/// `ModuleDictStrategy.caches` (the per-name `GlobalCache` registry
/// whose `cell` fields hold live values).  Pyre's W_ModuleDictObject
/// carries four indirect storages behind raw pointers — none of them
/// reachable through inline `gc_ptr_offsets`:
///
///   * `dstorage` → `ModuleDictStorage.entries` (Vec<(String,
///     PyObjectRef)>) — every entry's value
///   * `mstrategy` → `ModuleDictStrategy.caches` (Option<HashMap<...,
///     Rc<RefCell<GlobalCache>>>>) — every live cache's `cell`
///   * `object_storage` → post-`switch_to_object_strategy`
///     Vec<(PyObjectRef, PyObjectRef)> — both halves of every entry
unsafe fn module_dict_object_custom_trace(
    obj_addr: usize,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    // Delegate to the shared module-dict walk so this (GC-managed dict)
    // path and `walk_pyframe_roots`' Box-immortal path forward exactly
    // the same movable slots — including unwrapping the Box-immortal
    // MutableCells to reach the inner `w_value`, which a bare cell-pointer
    // visit (the slot itself never moves) would miss.
    let mut forward = |slot: &mut pyre_object::PyObjectRef| {
        f(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    };
    unsafe {
        pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(
            obj_addr as pyre_object::PyObjectRef,
            &mut forward,
        );
    }
}

unsafe fn set_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let set = unsafe { &mut *(obj_addr as *mut pyre_object::setobject::W_SetObject) };
    let entries = unsafe { &mut *set.items };
    for (key, _) in entries.iter_mut() {
        // ObjectKey.hash is identity-stable across GC moves, so writing the
        // relocated pointer through the key's `obj` slot keeps the bucket
        // index valid — mirrors `dict_object_custom_trace`.
        let key_ptr = key as *const pyre_object::dictmultiobject::ObjectKey
            as *mut pyre_object::dictmultiobject::ObjectKey;
        f(
            std::ptr::addr_of_mut!((*key_ptr).obj) as *mut pyre_object::PyObjectRef
                as *mut majit_ir::GcRef,
        );
    }
}

/// Custom trace for `W_TupleObject`. `wrappeditems` points at an off-GC
/// `std::alloc`'d `ItemsBlock` (`tupleobject.rs:56`), so the element
/// slots are unreachable through inline `gc_ptr_offsets` — the collector
/// would see `wrappeditems` as a single non-managed pointer and stop.
/// Forward each element slot in place, exactly as `set_object_custom_trace`
/// walks the off-GC `Vec`, so a moving collector relocates young tuple
/// elements and rewrites the block. The block is exact-size for tuples
/// (`capacity == len`, every slot written by `alloc_tuple_items_block`).
unsafe fn tuple_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let tuple_ptr = obj_addr as *mut pyre_object::tupleobject::W_TupleObject;
    let tuple = unsafe { &*tuple_ptr };
    let block = tuple.wrappeditems;
    if block.is_null() {
        return;
    }
    if pyre_object::gc_hook::try_gc_owns_object(block as *mut u8) {
        // Phase L2: forward the `wrappeditems` field slot; the type-9 varsize
        // walker forwards items[0..capacity] (tuples are exact-size).
        let items_slot = unsafe { std::ptr::addr_of_mut!((*tuple_ptr).wrappeditems) };
        f(items_slot as *mut majit_ir::GcRef);
    } else {
        // std::alloc stationary block: forward each element in place.
        let cap = unsafe { pyre_object::object_array::items_block_capacity(block) };
        let base = unsafe { pyre_object::object_array::items_block_items_base(block) };
        for i in 0..cap {
            f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
        }
    }
}

/// Custom trace for `W_ListObject` under the Object strategy. `items`
/// points at an off-GC `std::alloc`'d `ItemsBlock`
/// (`object_array::alloc_items_block`), so the element slots are
/// unreachable through inline `gc_ptr_offsets` — the collector would see
/// `items` as a single non-managed pointer and stop, leaving list elements
/// untraced (a major collection then sweeps an element reachable only via
/// the list).  Forward each live element slot in place, exactly as
/// `tuple_object_custom_trace`, so a moving collector relocates young
/// elements and a major collection marks them.  Only the Object strategy
/// stores `PyObjectRef`s; Integer/Float keep unboxed arrays (`items` null)
/// and Empty has no block.  Trace `length` live slots, not capacity — the
/// spare tail past the live length may hold stale pointers a shrink left
/// behind.
unsafe fn list_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let list_ptr = obj_addr as *mut pyre_object::listobject::W_ListObject;
    let list = unsafe { &*list_ptr };
    if list.strategy == pyre_object::listobject::ListStrategy::Object && !list.items.is_null() {
        if pyre_object::gc_hook::try_gc_owns_object(list.items as *mut u8) {
            // Phase L2: a GC-managed (moving) block is forwarded by handing the
            // collector the `items` field slot itself; the type-9 varsize walker
            // then forwards items[0..capacity] (spare slots are NULL). This is
            // the `gc_ptr_offsets = [offset_of!(items)]` edge that collector.rs:377
            // declines while the block stays std::alloc.
            let items_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).items) };
            f(items_slot as *mut majit_ir::GcRef);
        } else {
            // std::alloc stationary block: forward each live element in place.
            let base = unsafe { pyre_object::object_array::items_block_items_base(list.items) };
            for i in 0..list.length {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }
    // Integer/Float backing blocks (`int_items.block` / `float_items.block`) are
    // `GcArray(Signed)` / `GcArray(Float)` leaf arrays — no inner refs — so the
    // collector relocates one by forwarding the owner slot itself. Forwarded for
    // every strategy so a collection keeps the slots valid even when the strategy
    // does not read them (`Drop` deallocs through them); a std::alloc block (gate
    // off) is not GC-owned and stays in place.
    let int_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).int_items.block) };
    if pyre_object::gc_hook::try_gc_owns_object(unsafe { *int_block_slot } as *mut u8) {
        f(int_block_slot as *mut majit_ir::GcRef);
    }
    let float_block_slot = unsafe { std::ptr::addr_of_mut!((*list_ptr).float_items.block) };
    if pyre_object::gc_hook::try_gc_owns_object(unsafe { *float_block_slot } as *mut u8) {
        f(float_block_slot as *mut majit_ir::GcRef);
    }
}

/// Custom trace for `W_MemoryView`.  Its geometry and backing live in an
/// off-heap `*const BufferView` (`memoryview.rs`), so the macro's empty
/// `gc_ptr_offsets` reach none of the refs the collector must keep alive.
/// Forward the root exporter `PyObjectRef` of a `Buffer`, descending through
/// any `Sub` window to the leaf whose exporter actually owns the storage.
fn trace_buffer_exporter(
    buf: &mut pyre_object::buffer::Buffer,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    match buf {
        pyre_object::buffer::Buffer::String { w_obj }
        | pyre_object::buffer::Buffer::Byte { w_obj }
        | pyre_object::buffer::Buffer::Array { w_obj } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
        }
        pyre_object::buffer::Buffer::Sub { parent, .. } => {
            trace_buffer_exporter(parent, f);
        }
    }
}

/// Forward, in place, every `PyObjectRef` a view tree owns — the `.obj`
/// exporter, any stored format / shape / strides objects, the backing
/// exporter inside its `Buffer`, and (for a `Slice` wrapper) everything the
/// boxed parent view owns — exactly as `W_ListObject` walks its off-block
/// elements.  The boxes are `std::alloc` stationary, so nothing here moves.
fn trace_bufferview(
    view: &mut pyre_object::bufferview::BufferView,
    f: &mut dyn FnMut(*mut majit_ir::GcRef),
) {
    match view {
        // Simple / Raw derive their shape / strides (and Simple its format),
        // so only the `.obj` exporter, the backing, and — for Raw — the
        // explicit format object are ref slots to forward.
        pyre_object::bufferview::BufferView::Simple { backing, w_obj, .. } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_buffer_exporter(backing, f);
        }
        pyre_object::bufferview::BufferView::Raw {
            backing,
            w_obj,
            w_fmt,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_fmt as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_buffer_exporter(backing, f);
        }
        pyre_object::bufferview::BufferView::Slice { parent, w_obj, .. } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        pyre_object::bufferview::BufferView::View1D {
            parent,
            w_obj,
            w_fmt,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_fmt as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        pyre_object::bufferview::BufferView::ViewND {
            parent,
            w_obj,
            w_shape,
            w_strides,
            ..
        } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_shape as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(w_strides as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(parent, f);
        }
        pyre_object::bufferview::BufferView::Readonly { view, w_obj } => {
            f(w_obj as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            trace_bufferview(view, f);
        }
    }
}

unsafe fn memoryview_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let mv = obj_addr as *const pyre_object::memoryview::W_MemoryView;
    let view_ptr = unsafe { (*mv).view } as *mut pyre_object::bufferview::BufferView;
    if view_ptr.is_null() {
        return;
    }
    trace_bufferview(unsafe { &mut *view_ptr }, f);
}

/// Reclaim the off-heap `BufferView` box (and any nested `Buffer::Sub`
/// boxes it owns, via `Box`'s recursive drop glue) when a dead
/// `W_MemoryView` header is swept.  The custom trace only keeps the box's
/// `PyObjectRef`s alive while the header lives; without this the
/// `std::alloc` box would leak on every memoryview / slice / cast that
/// dies.  `release()` drops the box and nulls `view` eagerly, so the null
/// guard covers both a released view and the brief
/// header-allocated-before-`set_view` window.
unsafe fn memoryview_object_destructor(obj_addr: usize) {
    let mv = obj_addr as *const pyre_object::memoryview::W_MemoryView;
    let view_ptr = unsafe { (*mv).view } as *mut pyre_object::bufferview::BufferView;
    if !view_ptr.is_null() {
        drop(unsafe { Box::from_raw(view_ptr) });
    }
}

/// Custom trace for `PyFrame` (type id [`PYFRAME_GC_TYPE_ID`]).
///
/// Forwards exactly the frame-owned GC slots that the interpreter root
/// walker `pyre-interpreter::eval::walk_pyframe_roots` visits per frame,
/// so a type-directed trace of a header-tagged `PyFrame` sees the same
/// root set the ad-hoc walker does.  A flat `gc_ptr_offsets` list cannot
/// express two of those slots: the `locals_cells_stack_w` items when the
/// array is a stationary `std::alloc` block (regime-a — the collector
/// never enters `trace_and_update_object` on a non-nursery array so its
/// varsize walker never runs), and the in-place scan of old-gen / Box
/// `FrameDebugData` fields. Both require a custom trace.
///
/// Forwarded (mirrors `walk_pyframe_roots` eval.rs:496-556):
///   - `f_backref` — the parent frame pointer.
///   - `pycode` — visited to match the walker; inert while code objects
///     are Box-immortal (`is_nursery_object_start` short-circuits).
///   - `locals_cells_stack_w` — the array pointer.  A GC-managed nursery
///     block forwards through its field slot and its type-9 walker owns the
///     items. An old-gen GC block also visits the field slot and walks its
///     items in place: barrier-less interpreter stores require that at minors,
///     and it is harmless duplicate marking at majors. A stationary
///     `std::alloc` block always forwards its items in place.
///   - `f_generator_nowref`, `w_yielding_from`, `w_builtin`, `w_globals`
///     — the ref-bearing statics.
///   - `debugdata` / `lastblock` — managed field slots are forwarded.
///   - `debugdata->{w_locals, w_f_trace, hidden_operationerr}` — null-guarded.
///
/// Excluded (matches the walker): `execution_context` (persistent, not
/// GC), the module-dict / method-cache / prebuilt-family global walks (those are not frame-owned; the
/// root walker performs them once per collection, not per frame).
unsafe fn pyframe_object_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    let frame = unsafe { &mut *(obj_addr as *mut PyFrame) };

    f(&mut frame.f_backref as *mut *mut PyFrame as *mut majit_ir::GcRef);
    f(&mut frame.pycode as *mut *const () as *mut majit_ir::GcRef);

    // locals_cells_stack_w: visit the field slot for every GC array so major
    // marking reaches it. A nursery array is subsequently scanned by its own
    // type-9 walker; an old-gen array also needs this in-place scan because
    // interpreter stores do not write-barrier its items. At a major, its own
    // walker reaches them too, so this is harmless duplicate marking.
    // RPython's phase-agnostic precedent is jitframe.py:104 `jitframe_trace`.
    let array = frame.locals_cells_stack_w;
    if !array.is_null() {
        let managed = pyre_object::gc_hook::try_gc_owns_object(array as *mut u8);
        if managed {
            f(
                &mut frame.locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray
                    as *mut majit_ir::GcRef,
            );
        }
        // The visitor forwards a promoted nursery array in place, so
        // `frame.locals_cells_stack_w` now holds the old-gen copy. Re-read it:
        // the local `array` read above still points at the pre-copy location,
        // whose header is a forwarding marker (arraylen there is the
        // forwarding address). `walk_items` and the item walk must use the
        // live destination.
        let array = frame.locals_cells_stack_w;
        let walk_items = if managed {
            !majit_gc::gc_is_nursery_object(array as usize)
        } else {
            true
        };
        // Stationary `std::alloc` blocks (never entered by
        // `trace_and_update_object`) and old-gen GC blocks forward the FULL
        // fixed-length array, not just the live prefix. The old-gen major
        // walk is idempotent duplicate marking; the minor walk covers
        // barrier-less interpreter stores. This matches RPython's
        // phase-agnostic jitframe.py:104 `jitframe_trace`.
        // This matches `walk_pyframe_roots` (eval.rs:626), which forwards
        // popped-in-transit argument slots past `valuestackdepth`.
        if walk_items {
            let arr = unsafe { &mut *array };
            let base = arr.items_mut_ptr();
            for i in 0..arr.len() {
                f(unsafe { base.add(i) } as *mut majit_ir::GcRef);
            }
        }
    }

    f(&mut frame.f_generator_nowref as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.w_yielding_from as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.w_builtin as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
    f(&mut frame.w_globals as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);

    if !frame.debugdata.is_null() {
        let debugdata = frame.debugdata;
        let walk_fields = if pyre_object::gc_hook::try_gc_owns_object(debugdata as *mut u8) {
            f(
                &mut frame.debugdata as *mut *mut pyre_interpreter::pyframe::FrameDebugData
                    as *mut majit_ir::GcRef,
            );
            !majit_gc::gc_is_nursery_object(debugdata as usize)
        } else {
            true
        };
        // A nursery payload is scanned by FRAME_DEBUG_DATA_GC_TYPE_ID's
        // ordinary offset walker. Box payloads and old-gen payloads need this
        // in-place walk because interpreter stores do not individually
        // write-barrier w_f_trace and its sibling fields. The old-gen major
        // walk is harmless duplicate marking, matching RPython's
        // phase-agnostic jitframe.py:104 `jitframe_trace` contract.
        if walk_fields {
            let d = unsafe { &mut *frame.debugdata };
            f(&mut d.w_locals as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.w_f_trace as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
            f(&mut d.hidden_operationerr as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef);
        }
    }

    if !frame.lastblock.is_null()
        && pyre_object::gc_hook::try_gc_owns_object(frame.lastblock as *mut u8)
    {
        // FRAME_BLOCK_GC_TYPE_ID's `previous` walker forwards the rest of
        // the chain. Blocks themselves contain no PyObjectRefs.
        f(
            &mut frame.lastblock as *mut *mut pyre_interpreter::pyframe::FrameBlock
                as *mut majit_ir::GcRef,
        );
    }
}

/// RPython jitexc.py:53 ContinueRunningNormally parity.
pub(crate) enum LoopResult {
    Done(PyResult),
    ContinueRunningNormally,
}

/// Action from handle_jit_outcome for eval_loop_jit dispatch.
enum JitAction {
    Return(PyResult),
    Continue,
    /// RPython jitexc.py:53: guard-restored → restart portal.
    ContinueRunningNormally,
}

use crate::jit::descr::{
    BUILTIN_CODE_GC_TYPE_ID, FRAME_BLOCK_GC_TYPE_ID, FRAME_DEBUG_DATA_GC_TYPE_ID,
    FUNCTION_GC_TYPE_ID, GC_FLOAT_ARRAY_GC_TYPE_ID, GC_INT_ARRAY_GC_TYPE_ID, JITFRAME_GC_TYPE_ID,
    OBJECT_GC_TYPE_ID, PY_OBJECT_ARRAY_GC_TYPE_ID, PYFRAME_GC_TYPE_ID, RANGE_ITER_GC_TYPE_ID,
    SPECIALISED_TUPLE_FF_GC_TYPE_ID, SPECIALISED_TUPLE_II_GC_TYPE_ID,
    SPECIALISED_TUPLE_OO_GC_TYPE_ID, VREF_GC_TYPE_ID, W_BASE_EXCEPTION_GC_TYPE_ID,
    W_BOOL_GC_TYPE_ID, W_BYTEARRAY_GC_TYPE_ID, W_BYTES_GC_TYPE_ID, W_CELL_GC_TYPE_ID,
    W_CLASSMETHOD_GC_TYPE_ID, W_COUNT_GC_TYPE_ID, W_DICT_GC_TYPE_ID, W_DICT_PROXY_GC_TYPE_ID,
    W_FLOAT_GC_TYPE_ID, W_GENERATOR_GC_TYPE_ID, W_INT_GC_TYPE_ID, W_LIST_GC_TYPE_ID,
    W_LONG_GC_TYPE_ID, W_MEMBER_GC_TYPE_ID, W_METHOD_GC_TYPE_ID, W_MODULE_DICT_GC_TYPE_ID,
    W_MODULE_GC_TYPE_ID, W_PROPERTY_GC_TYPE_ID, W_REPEAT_GC_TYPE_ID, W_SEQ_ITER_GC_TYPE_ID,
    W_SET_GC_TYPE_ID, W_SLICE_GC_TYPE_ID, W_STATICMETHOD_GC_TYPE_ID, W_SUPER_GC_TYPE_ID,
    W_TUPLE_GC_TYPE_ID, W_TYPE_GC_TYPE_ID, W_UNICODE_GC_TYPE_ID, W_UNION_GC_TYPE_ID,
};
use majit_gc::collector::MiniMarkGC;
use majit_metainterp::JitDriver;
use pyre_jit_trace::frame_layout::build_pyframe_virtualizable_info;
use pyre_object::floatobject::{FLOAT_FLOATVAL_OFFSET, W_FloatObject};
use pyre_object::intobject::{INT_INTVAL_OFFSET, W_IntObject};
use pyre_object::{w_bool_from, w_int_new, w_none, w_str_new, w_tuple_new};

const JIT_THRESHOLD: u32 = 200;
type JitDriverPair = (
    JitDriver<PyreJitState>,
    std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>,
);

thread_local! {
    /// Per-thread flag: this thread has registered with the gc_sync
    /// mutator registry and installed the backend GC handle into the
    /// backend's per-thread TLS box. The majit_gc set_active_* fn-ptrs and
    /// pyre_object gc_hook cells are now process-global (#396), so they are
    /// installed once (not gated by this flag); only the per-thread backend
    /// box and mutator registration remain per-thread here.
    static GC_TLS_INSTALLED: Cell<bool> = const { Cell::new(false) };

    /// Initialized after shadow_stack::register_mutator has captured all four
    /// root TLS slots. Its destructor therefore removes the registry entry
    /// before those slots are destroyed, then removes the thread from RUNNING.
    static GC_MUTATOR_REGISTRATION: GcMutatorRegistration = const { GcMutatorRegistration };
}

struct GcMutatorRegistration;

impl Drop for GcMutatorRegistration {
    fn drop(&mut self) {
        majit_gc::shadow_stack::unregister_mutator();
        majit_gc::gc_sync::unregister_thread();
    }
}

/// Build and configure the MiniMarkGC with all type registrations,
/// vtable mappings, and subclass ranges.
fn build_gc() -> Box<dyn majit_gc::GcAllocator> {
    // translationoption.py:185 `taggedpointers` — kept in lockstep with the
    // pyre-object representation switch so the collector-core immediate
    // guards (`is_tagged_immediate`) go live exactly when small ints start
    // arriving as `(v<<1)|1` immediates. Both default false; the flip lands
    // in the enablement slice. majit-gc cannot read `pyre_object`
    // (dependency points the other way), so the constructor mirrors it here.
    let mut gc = MiniMarkGC::with_config(majit_gc::collector::GcConfig {
        taggedpointers: pyre_object::tagged_int::CAN_BE_TAGGED,
        ..majit_gc::collector::GcConfig::default()
    });
    // rclass.OBJECT root (rclass.py:160-166). pyre's static
    // `INSTANCE_TYPE` is the `name = "object"` PyType — every
    // other `PyObject`-layout class chains its `parent` field to
    // this id so `assign_inheritance_ids` (normalizecalls.py:373-389)
    // produces a `subclassrange_{min,max}` covering every
    // descendant. The size is `sizeof(PyObject)` because instances
    // tagged with `&INSTANCE_TYPE` (i.e. user `object()` calls)
    // carry only the `ob_type` header.
    let object_tid = gc.register_type(TypeInfo::object(
        std::mem::size_of::<pyre_object::PyObject>(),
    ));
    debug_assert_eq!(object_tid, OBJECT_GC_TYPE_ID);
    // W_IntObject / W_FloatObject carry `PyObject.ob_type` at offset 0,
    // matching RPython `rclass.OBJECT` layout (T_IS_RPYTHON_INSTANCE,
    // gc.py:642). They are NewWithVtable allocation targets so the
    // payload size must be the actual struct size, and they sit one
    // level below the OBJECT root (`int.__bases__ == (object,)`,
    // `float.__bases__ == (object,)`).
    let w_int_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<W_IntObject>(),
        object_tid,
    ));
    debug_assert_eq!(w_int_tid, W_INT_GC_TYPE_ID);
    let w_float_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<W_FloatObject>(),
        object_tid,
    ));
    debug_assert_eq!(w_float_tid, W_FLOAT_GC_TYPE_ID);
    // jitframe.py:49 — rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    debug_assert_eq!(jitframe_tid, JITFRAME_GC_TYPE_ID);
    // pyre allocates jitframes via `libc::calloc` (not nursery/oldgen),
    // so the collector's standard `walk_jf_roots` visitor can't
    // route them through `trace_and_update_object`. Register a
    // host-side tracer that invokes `jitframe_trace` directly so
    // Refs pinned to frame slots are visible to GC across minor
    // collections triggered by CallMallocNursery slow paths.
    majit_gc::shadow_stack::register_libc_jitframe_tracer(pyre_libc_jitframe_tracer);
    // virtualref.py — JIT_VIRTUAL_REF as a proper GC type.
    // Layout: super_.typeptr(u64, offset 0) | virtual_token(*mut u8, offset 8) | forced(*mut u8, offset 16)
    //
    // Note (GC trace divergence).  Upstream
    // `virtualref.py:17-20` declares both `virtual_token` and
    // `forced` as GC slots (`llmemory.GCREF` / `OBJECTPTR`); pyre
    // registers only `forced` (offset 16) in `gc_ptr_offsets`.
    // The `virtual_token` slot is intentionally outside the GC's
    // view because every runtime value it can hold lives outside
    // any GC heap: TOKEN_NONE (null), `token_tracing_rescall()`
    // (program-lifetime leaked `Box<ObjectHeader>` dummy lazily
    // allocated by `allocate_tracing_rescall_dummy` and cached in
    // `TRACING_RESCALL_DUMMY_PTR`, see `majit-metainterp/src/
    // virtualref.rs:140-180`), and active JITFRAME addresses
    // (libc::calloc'd, see `register_libc_jitframe_tracer` above).
    // The optimizer-side descriptor at
    // `majit-metainterp/src/optimizeopt/virtualize.rs:make_vref_field_descr`
    // still uses `Type::Ref` so `setfield_gc_r` / `getfield_gc_r`
    // emit correctly; only the collector's view of the slot
    // diverges.  Convergence requires both `_dummy` and JITFRAME
    // allocation to move under the GC.
    let vref_tid = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
        std::mem::size_of::<majit_metainterp::virtualref::JitVirtualRef>(),
        vec![std::mem::offset_of!(
            majit_metainterp::virtualref::JitVirtualRef,
            forced
        )],
    ));
    debug_assert_eq!(vref_tid, VREF_GC_TYPE_ID);
    // Tell the virtualref optimizer about the registered type id.
    majit_metainterp::virtualref::set_vref_gc_type_id(vref_tid);
    // Dedicated typeids for the JIT-NEW'd / JIT-guard'd PyObject
    // subclasses whose payload is NOT `sizeof(PyObject)`. RPython
    // registers one typeid per distinct STRUCT through
    // `heaptracker.setup_cache_gcstruct2vtable` (heaptracker.py:23-30)
    // and `add_vtable_after_typeinfo` (gctypelayout.py:359-374). pyre's
    // earlier one-typeid-per-root-layout approximation under-walked
    // lists/tuples/range-iters as soon as their descr groups carried
    // `type_id = 0`. `gc_ptr_offsets` stays empty for all four — these
    // registrations are pure bookkeeping; their pointer fields are
    // not modeled here.
    let w_bool_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::boolobject::W_BoolObject>(),
        w_int_tid,
    ));
    debug_assert_eq!(w_bool_tid, W_BOOL_GC_TYPE_ID);
    let range_iter_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::functional::W_IntRangeIterator>(),
        object_tid,
    ));
    debug_assert_eq!(range_iter_tid, RANGE_ITER_GC_TYPE_ID);
    // rlist.py:116 parity: W_ListObject has a single GC pointer
    // field — `items: Ptr(GcArray(OBJECTPTR))` — directly at
    // `offset_of!(items)`. The GC offset points straight at `items`
    // with no intermediate block-start field.
    //
    // `items` points at an off-GC `std::alloc`'d `ItemsBlock`
    // (`alloc_items_block` in `pyre_object::object_array`), so inline
    // `gc_ptr_offsets` tracing stops at the non-managed block pointer
    // (`is_managed_heap_object` rejects it) and never reaches the
    // elements — a major collection then sweeps a list element
    // reachable only through the list.  Trace through the block with a
    // custom hook instead (mirrors `W_TupleObject` / `W_SetObject`).
    let w_list_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::listobject::W_ListObject>(),
        object_tid,
        list_object_custom_trace,
    ));
    debug_assert_eq!(w_list_tid, W_LIST_GC_TYPE_ID);
    // Full tuple convergence additionally requires specialised arity-2
    // variants (per `pypy/objspace/std/specialisedtupleobject.py`),
    // which are not yet modeled here.
    // `wrappeditems` points at an off-GC `std::alloc`'d ItemsBlock, so
    // inline `gc_ptr_offsets` tracing stops at the non-managed block
    // pointer and never reaches the elements. Trace through the block
    // with a custom hook instead (mirrors `W_SetObject`); the tuple's
    // explicit write barrier at creation (`tupleobject.rs`) keeps the
    // old-gen tuple in the remembered set so this runs on minor GC.
    let w_tuple_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::tupleobject::W_TupleObject>(),
        object_tid,
        tuple_object_custom_trace,
    ));
    debug_assert_eq!(w_tuple_tid, W_TUPLE_GC_TYPE_ID);
    // `rlist.py Ptr(GcArray(OBJECTPTR))` — the variable-length
    // backing block behind `PyObjectArray`. `base=8` single-slot
    // header (`capacity`), `item_size=8` Ref, `length_offset=0`
    // so `gctypelayout.py:266-291` reads `capacity` as the
    // GcArray length (rlist.py:251 `len(l.items)` = allocated
    // slot count — upstream's GcArray header IS the capacity,
    // not live length).  `items_have_gc_ptrs=true` activates
    // `T_IS_GCARRAY_OF_GCPTR` so the nursery walker traces every
    // item slot as a Ref; NULL-initialized spare slots past the
    // live length are benign.
    //
    // This typeid governs blocks allocated *through the GC*, which is
    // the default path (`object_array::alloc_*_block_gc` →
    // `try_gc_alloc`); the nursery walker traces each item slot of such
    // a block, and the list/tuple custom traces forward the block
    // pointer. Under the `PYRE_GC_ITEMSBLOCK=0` fallback the blocks come
    // from `std::alloc` instead and no allocation carries this typeid.
    // See comments on `pyre_jit_trace::descr::PY_OBJECT_ARRAY_GC_TYPE_ID`
    // and `pyre_object::object_array::ItemsBlock` for the companion
    // notices.
    let py_object_array_tid = gc.register_type(TypeInfo::varsize(
        pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET,
        std::mem::size_of::<pyre_object::pyobject::PyObjectRef>(),
        0,
        true,
        Vec::new(),
    ));
    debug_assert_eq!(py_object_array_tid, PY_OBJECT_ARRAY_GC_TYPE_ID);
    // `pypy/objspace/std/specialisedtupleobject.py` `Cls_ii / Cls_ff
    // / Cls_oo` — three subclasses of `W_AbstractTupleObject` with
    // inline `value0` / `value1` fields. Each gets a distinct
    // `ob_type` so the JIT's `GUARD_CLASS` reaches the inline-field
    // shape directly. `Cls_oo` carries two GC-pointer slots; the
    // other two are GC-leaf for the payload (header still has w_class).
    let mut spec_tuple_ii_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ii>(),
        object_tid,
    );
    spec_tuple_ii_ti.has_gc_ptrs = false;
    let spec_tuple_ii_tid = gc.register_type(spec_tuple_ii_ti);
    debug_assert_eq!(spec_tuple_ii_tid, SPECIALISED_TUPLE_II_GC_TYPE_ID);
    let mut spec_tuple_ff_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_ff>(),
        object_tid,
    );
    spec_tuple_ff_ti.has_gc_ptrs = false;
    let spec_tuple_ff_tid = gc.register_type(spec_tuple_ff_ti);
    debug_assert_eq!(spec_tuple_ff_tid, SPECIALISED_TUPLE_FF_GC_TYPE_ID);
    let mut spec_tuple_oo_ti = TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::specialisedtupleobject::W_SpecialisedTupleObject_oo>(),
        object_tid,
    );
    spec_tuple_oo_ti.gc_ptr_offsets = vec![
        pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE0_OFFSET,
        pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_VALUE1_OFFSET,
    ];
    spec_tuple_oo_ti.has_gc_ptrs = true;
    let spec_tuple_oo_tid = gc.register_type(spec_tuple_oo_ti);
    debug_assert_eq!(spec_tuple_oo_tid, SPECIALISED_TUPLE_OO_GC_TYPE_ID);
    // Tell the cranelift backend which type id to use for the
    // nursery allocations that it issues for jitframes. Without
    // this, the backend's default u32::MAX sentinel would trip the
    // allocation assert in run_compiled_code_inner, or — worse,
    // before this fix — the backend's stale hard-coded `2` would
    // collide with W_FLOAT_GC_TYPE_ID and GC would copy jitframes
    // with the wrong TypeInfo (24-byte float payload instead of
    // the real 64 + 8*depth layout), silently truncating every
    // ref root slot past the first three bytes.
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::set_jitframe_gc_type_id(jitframe_tid);
    #[cfg(feature = "dynasm")]
    majit_backend_dynasm::set_jitframe_gc_type_id(jitframe_tid);
    // The orthodox (PYRE_WASM_CA) frame path allocates host-entry frames as
    // GC-managed JitFrames of this type so the collector forwards their Ref
    // item slots via the jf_gcmap custom trace.
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_wasm_jitframe_tid(jitframe_tid);
    // llsupport/gc.py:563 vtable→typeid mapping. RPython derives the
    // typeid arithmetically from gc_get_type_info_group; pyre keeps an
    // explicit table because every PyType is a static global
    // unrelated to the GC's internal layout. The OBJECT root and
    // INT/FLOAT are wired up first so subsequent foreign-pytype
    // entries can resolve their parents through the same map.
    let mut pytype_to_tid: HashMap<usize, u32> = HashMap::new();
    // Helper for `#[pyre_class]`-emitted types: register the GC
    // payload + vtable + `pytype_to_tid` entry in one call.  Asserts
    // that the descriptor's `gc_type_id` matches the id `gc.register_type`
    // returns — drift indicates the manual constant in the
    // `#[pyre_class(... type_id = N)]` attribute is out of step
    // with the registration order here.
    let register_pyre_class = |gc: &mut MiniMarkGC,
                               pytype_to_tid: &mut HashMap<usize, u32>,
                               descr: &'static pyre_object::lltype::PyreClassDescriptor|
     -> u32 {
        let tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            descr.object_size,
            object_tid,
            descr.ptr_offsets.to_vec(),
        ));
        // Auto-id mode (cell == UNASSIGNED): stamp the cell with
        // the freshly-assigned tid so runtime readers see it.
        // Explicit-id mode (cell pre-initialized): drift-check that
        // the declared id matches registration order.
        if descr.gc_type_id.is_unassigned() {
            descr.gc_type_id.set(tid);
        } else {
            debug_assert_eq!(
                tid,
                descr.gc_type_id.get(),
                "PyreClassDescriptor::gc_type_id mismatch — adjust `#[pyre_class(type_id = N)]` or drop the explicit id",
            );
        }
        let pytype_ptr = descr.pytype_ptr as usize;
        majit_gc::GcAllocator::register_vtable_for_type(gc, pytype_ptr, tid);
        pytype_to_tid.insert(pytype_ptr, tid);
        tid
    };
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        object_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as usize,
        object_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::INT_TYPE as *const _ as usize,
        w_int_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::INT_TYPE as *const _ as usize,
        w_int_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
        w_float_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::FLOAT_TYPE as *const _ as usize,
        w_float_tid,
    );
    // Bind the four dedicated typeids registered above to their
    // static PyType pointers. The foreign-pytype loop below skips
    // any PyType already present in `pytype_to_tid`, so these four
    // pre-bindings override the loop's would-be
    // `object_subclass(sizeof(PyObject))` registration with the
    // correct per-struct size.
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        w_bool_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::BOOL_TYPE as *const _ as usize,
        w_bool_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        range_iter_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::functional::RANGE_ITER_TYPE as *const _ as usize,
        range_iter_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        w_list_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::LIST_TYPE as *const _ as usize,
        w_list_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        w_tuple_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::TUPLE_TYPE as *const _ as usize,
        w_tuple_tid,
    );
    // BuiltinCode is pre-registered (rather than picked up by the
    // foreign-pytype loop below) because the loop hard-codes
    // `size_of::<PyObject>()` as the payload size, while the
    // GC needs `size_of::<BuiltinCode>()` to walk live instances
    // correctly. Mirror W_INT/W_FLOAT pattern so future GC
    // integration finds an already-registered tid + size pair.
    let builtin_code_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_interpreter::gateway::BuiltinCode>(),
        object_tid,
    ));
    debug_assert_eq!(builtin_code_tid, BUILTIN_CODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
        builtin_code_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::gateway::BUILTIN_CODE_TYPE as *const _ as usize,
        builtin_code_tid,
    );
    // Function carries inline `PyObjectRef` fields (code / closure /
    // defs_w / w_kw_defs / w_module / cached metadata) that the
    // collector must walk — `object_subclass_with_gc_ptrs` records
    // the offsets so mark traversal reaches them. `BUILTIN_FUNCTION_TYPE`
    // is a separate static `PyType` for module-level builtins
    // (`pypy/interpreter/function.py:706 BuiltinFunction`) but its
    // instances are the same Rust struct, so the vtable map sends
    // both PyTypes to `function_tid`.
    let function_tid = gc.register_type(
        TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_interpreter::function::Function>(),
            object_tid,
            pyre_interpreter::function::FUNCTION_GC_PTR_OFFSETS.to_vec(),
        )
        .with_destructor_fn(function_object_destructor),
    );
    debug_assert_eq!(function_tid, FUNCTION_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE as *const _ as usize,
        function_tid,
    );
    // Cell / Method / W_SliceObject — typed payload
    // via `#[pyre_class]`.  Pre-registered ahead of the foreign-
    // pytype loop because that loop's `size_of::<PyObject>()`
    // approximation drops the GC ptr offsets, leaving cells / bound
    // methods / slices unscanned across a minor collection.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::nestedscope::Cell as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::Method as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::sliceobject::W_SliceObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Super (super proxy) — typed payload via `#[pyre_class]`;
    // GC descriptor carries the 2 inline `PyObjectRef` fields
    // (super_type / obj).  Pre-registered ahead of the foreign-pytype
    // loop for the same reason as W_Cell/W_Method/W_Slice.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::descriptor::W_Super as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Property (3 PyObjectRef fields: fget/fset/fdel),
    // StaticMethod and ClassMethod (1 PyObjectRef
    // field each: w_function) — typed payload via `#[pyre_class]`.
    // Pre-registered ahead of the foreign-pytype loop so the GC
    // walker reaches the inline descriptor refs.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::descriptor::W_Property as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::StaticMethod as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::function::ClassMethod as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // UnionType (PEP 604 `X | Y`) — typed payload via `#[pyre_class]`.
    // Pre-registered ahead of the foreign-pytype loop because that
    // loop's `size_of::<PyObject>()` approximation drops gc_ptr_offsets,
    // leaving live unions unscanned across a minor collection.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::_pypy_generic_alias::UnionType
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_SeqIterObject (list/tuple iterator) — typed payload via
    // `#[pyre_class]`.  Pre-registered ahead of the foreign-pytype
    // loop so the GC walker reaches the inline `seq` field.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_SeqIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Count / W_Repeat (`itertools.count` / `itertools.repeat`) —
    // typed payload via `#[pyre_class]`.  Neither PyType is in
    // `all_foreign_pytypes()`, so pre-registration here is the only
    // path through which their instances become GC-managed.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Count
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Repeat
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_MemberDescr (`__slots__` member descriptor) carries one
    // inline `PyObjectRef` field (`w_cls`) plus a `*const String`
    // (`name`) and a `u32` index. The `#[pyre_class]` macro's
    // auto-detection skips both non-PyObjectRef fields, so the
    // descriptor's ptr_offsets only includes `w_cls`.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::typedef::W_MemberDescr as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_BytesObject (immutable byte sequence) carries a raw
    // `*const Vec<u8>` (`data`) and a `usize` length, neither a
    // `PyObjectRef`. Pre-registered with `object_subclass(size, ...)`
    // so the foreign-pytype loop's `sizeof(PyObject)` approximation
    // does not under-count the payload.
    let w_bytes_tid = gc.register_type(
        TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::bytesobject::W_BytesObject>(),
            object_tid,
        )
        .with_destructor_fn(bytes_object_destructor),
    );
    debug_assert_eq!(w_bytes_tid, W_BYTES_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
        w_bytes_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytesobject::BYTES_TYPE as *const _ as usize,
        w_bytes_tid,
    );
    // W_BytearrayObject (mutable byte sequence) carries a raw
    // `*mut Vec<u8>` (`data`). Same registration shape as
    // W_BytesObject.
    let w_bytearray_tid = gc.register_type(
        TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::bytearrayobject::W_BytearrayObject>(),
            object_tid,
        )
        .with_destructor_fn(bytearray_object_destructor),
    );
    debug_assert_eq!(w_bytearray_tid, W_BYTEARRAY_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
        w_bytearray_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const _ as usize,
        w_bytearray_tid,
    );
    // W_DictObject carries `entries: *mut Vec<(PyObjectRef,
    // PyObjectRef)>` behind a raw pointer. Register a custom trace
    // hook so the GC updates those indirect key/value slots just as it
    // updates inline object fields.
    let w_dict_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_object::dictmultiobject::W_DictObject>(),
            object_tid,
            dict_object_custom_trace,
        )
        .with_destructor_fn(dict_object_destructor),
    );
    debug_assert_eq!(w_dict_tid, W_DICT_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::DICT_TYPE as *const _ as usize,
        w_dict_tid,
    );
    pytype_to_tid.insert(&pyre_object::DICT_TYPE as *const _ as usize, w_dict_tid);
    // W_SetObject carries `items: *mut IndexMap<ObjectKey, ()>`. Register a
    // custom trace hook so GC forwarding updates indirect key object slots.
    // Both `set` and `frozenset` PyTypes share this Rust struct/tid.
    let w_set_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_object::setobject::W_SetObject>(),
            object_tid,
            set_object_custom_trace,
        )
        .with_destructor_fn(set_object_destructor),
    );
    debug_assert_eq!(w_set_tid, W_SET_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::SET_TYPE as *const _ as usize,
        w_set_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::SET_TYPE as *const _ as usize,
        w_set_tid,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
        w_set_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::setobject::FROZENSET_TYPE as *const _ as usize,
        w_set_tid,
    );
    // W_BaseException carries an `ExcKind` tag, a `*mut String`
    // pointer (raw heap, not a `PyObjectRef`), and a `args_w`
    // tuple `PyObjectRef` (`interp_exceptions.py:123-124
    // W_BaseException.descr_init` parity — the constructor stores
    // the args tuple inline on the instance).  Register the
    // `args_w` offset so the GC traces it across minor
    // collections.
    let w_exception_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::interp_exceptions::W_BaseException>(),
        object_tid,
        pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_exception_tid, W_BASE_EXCEPTION_GC_TYPE_ID);
    // Pre-register every per-ExcKind PyType to the same
    // `W_BaseException` GC tid — they share one storage layout
    // (the per-kind discriminator lives in `ob_type`, payload is
    // identical) so the GC must size them identically.  The
    // `all_foreign_pytypes` loop below skips entries already in
    // `pytype_to_tid`, so this pre-registration wins over its
    // generic `object_subclass(sizeof(PyObject), parent_tid)`
    // default which would underallocate `W_BaseException`.
    for kind_idx in 0u8..=(pyre_object::interp_exceptions::ExcKind::UnboundLocalError as u8) {
        // Round-trip the byte through the enum so we don't depend
        // on unsafe transmute; every value in [0, UnboundLocalError]
        // is a valid `ExcKind` variant by construction.
        let kind = match kind_idx {
            0 => pyre_object::interp_exceptions::ExcKind::BaseException,
            1 => pyre_object::interp_exceptions::ExcKind::Exception,
            2 => pyre_object::interp_exceptions::ExcKind::TypeError,
            3 => pyre_object::interp_exceptions::ExcKind::ValueError,
            4 => pyre_object::interp_exceptions::ExcKind::ZeroDivisionError,
            5 => pyre_object::interp_exceptions::ExcKind::NameError,
            6 => pyre_object::interp_exceptions::ExcKind::IndexError,
            7 => pyre_object::interp_exceptions::ExcKind::KeyError,
            8 => pyre_object::interp_exceptions::ExcKind::AttributeError,
            9 => pyre_object::interp_exceptions::ExcKind::RuntimeError,
            10 => pyre_object::interp_exceptions::ExcKind::StopIteration,
            11 => pyre_object::interp_exceptions::ExcKind::OverflowError,
            12 => pyre_object::interp_exceptions::ExcKind::ArithmeticError,
            13 => pyre_object::interp_exceptions::ExcKind::ImportError,
            14 => pyre_object::interp_exceptions::ExcKind::NotImplementedError,
            15 => pyre_object::interp_exceptions::ExcKind::AssertionError,
            16 => pyre_object::interp_exceptions::ExcKind::ReferenceError,
            17 => pyre_object::interp_exceptions::ExcKind::GeneratorExit,
            18 => pyre_object::interp_exceptions::ExcKind::RecursionError,
            19 => pyre_object::interp_exceptions::ExcKind::OSError,
            20 => pyre_object::interp_exceptions::ExcKind::FileNotFoundError,
            21 => pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError,
            22 => pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError,
            23 => pyre_object::interp_exceptions::ExcKind::SystemExit,
            24 => pyre_object::interp_exceptions::ExcKind::MemoryError,
            25 => pyre_object::interp_exceptions::ExcKind::SystemError,
            26 => pyre_object::interp_exceptions::ExcKind::LookupError,
            27 => pyre_object::interp_exceptions::ExcKind::UnicodeError,
            28 => pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError,
            29 => pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError,
            30 => pyre_object::interp_exceptions::ExcKind::SyntaxError,
            31 => pyre_object::interp_exceptions::ExcKind::BufferError,
            32 => pyre_object::interp_exceptions::ExcKind::UnboundLocalError,
            _ => unreachable!(),
        };
        let pytype_ptr =
            pyre_object::interp_exceptions::exc_kind_to_pytype(kind) as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, w_exception_tid);
        pytype_to_tid.insert(pytype_ptr, w_exception_tid);
    }
    // GeneratorIterator carries `frame_ptr: *mut u8` (opaque
    // PyFrame pointer, owned by the generator) plus three bools.
    // The suspended frame is held behind an opaque `frame_ptr`; a
    // custom trace visits the frame's `pycode` so a code object
    // reachable only via a suspended generator stays a GC root once
    // code objects are GC-managed.  The frame's other PyObjectRefs
    // remain reachable only through the PyFrame indirection
    // (pre-existing limitation).
    let w_generator_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_object::generator::GeneratorIterator>(),
        object_tid,
        generator_object_custom_trace,
    ));
    debug_assert_eq!(w_generator_tid, W_GENERATOR_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::generator::GENERATOR_TYPE as *const _ as usize,
        w_generator_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::generator::GENERATOR_TYPE as *const _ as usize,
        w_generator_tid,
    );
    // W_TypeObject carries one inline `PyObjectRef` (`bases`)
    // plus several non-PyObject raw pointers (`name`, `dict`,
    // `mro_w`, `layout`) and a `weak_subclasses: *mut
    // Vec<PyObjectRef>` that must be walked manually
    // (`typeobject.py:640-689` add/get/remove_subclass).
    // Pre-registered ahead of the foreign-pytype loop because
    // `TYPE_TYPE` is in `all_foreign_pytypes()` and the
    // loop's `sizeof(PyObject)` approximation drastically
    // under-counts the W_TypeObject payload.
    let w_type_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_object::typeobject::W_TypeObject>(),
            object_tid,
            type_object_custom_trace,
        )
        .with_destructor_fn(type_object_destructor),
    );
    debug_assert_eq!(w_type_tid, W_TYPE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::TYPE_TYPE as *const _ as usize,
        w_type_tid,
    );
    pytype_to_tid.insert(&pyre_object::TYPE_TYPE as *const _ as usize, w_type_tid);
    // W_UnicodeObject carries an off-heap WTF-8 buffer. Managed subclass
    // instances additionally need the header's `w_class` traced, matching
    // W_ObjectObject's instance-class edge; exact strings remain immortal.
    let w_str_tid = gc.register_type(
        TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::unicodeobject::W_UnicodeObject>(),
            object_tid,
            vec![pyre_object::pyobject::W_CLASS_OFFSET],
        )
        .with_destructor_fn(pyre_object::unicodeobject::unicode_object_destructor),
    );
    debug_assert_eq!(w_str_tid, W_UNICODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::STR_TYPE as *const _ as usize,
        w_str_tid,
    );
    pytype_to_tid.insert(&pyre_object::STR_TYPE as *const _ as usize, w_str_tid);
    // W_LongObject carries a `value: *mut BigInt` that now points at a
    // GC-managed bigint payload (BIGINT_GC_TYPE_ID, registered below), so
    // the collector must trace/forward it — register the `value` offset as
    // a gc-pointer rather than the old size-only shape.
    let w_long_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::longobject::W_LongObject>(),
        object_tid,
        vec![pyre_object::longobject::LONG_VALUE_OFFSET],
    ));
    debug_assert_eq!(w_long_tid, W_LONG_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::LONG_TYPE as *const _ as usize,
        w_long_tid,
    );
    pytype_to_tid.insert(&pyre_object::LONG_TYPE as *const _ as usize, w_long_tid);
    // Module carries `name: *mut String` (raw heap),
    // `dict: *mut u8` (DictStorage*, non-PyObject), and
    // `w_dict: PyObjectRef` (aliased `W_DictObject`,
    // `pypy/interpreter/module.py:22 self.w_dict = w_dict`).  Only
    // the last is GC-traceable.
    let w_module_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::module::Module>(),
        object_tid,
        pyre_object::module::W_MODULE_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_module_tid, W_MODULE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::MODULE_TYPE as *const _ as usize,
        w_module_tid,
    );
    pytype_to_tid.insert(&pyre_object::MODULE_TYPE as *const _ as usize, w_module_tid);
    // `pyre-interpreter::pyframe::PyFrame` — execution frame for a
    // Python code block. NOT an `rclass.OBJECT`-shaped instance
    // (no `ob_type` header — virtualizable struct laid out for the
    // JIT virtualize pass), so register with a bare size + trace hook
    // rather than `object_subclass`.
    //
    // The trace hook `pyframe_object_custom_trace` forwards exactly
    // the frame-owned GC slots `walk_pyframe_roots` visits per frame.
    // A flat `gc_ptr_offsets` list cannot express two of them — the
    // `locals_cells_stack_w` items when the array is a stationary
    // `std::alloc` block, and the `debugdata->{w_locals, w_f_trace}`
    // refs one indirection away — so a custom trace is required.
    // `custom_trace` fully replaces offset tracing on both the minor
    // (collector.rs:1471) and major-mark (collector.rs:1746) paths.
    //
    // Frames stamped with this type id: JIT-built inline frames
    // (`emit_new_pyframe_inline_self_recursive`, whose locals array is a
    // GC-managed `PY_OBJECT_ARRAY_GC_TYPE_ID` block) AND executing /
    // generator `FrameBox` frames (`FrameBox::new` via
    // `try_gc_alloc_stable`, whose locals array is a stationary
    // `std::alloc` block).  The custom trace's regime split
    // (`try_gc_owns_object`) handles both.  Callee-arena JIT frames
    // remain `type_id = 0` off-GC blocks reached only as roots via
    // `walk_jit_callee_frame_roots` (S2c).
    //
    // Frame-owned locals arrays, debug data, and block-stack nodes are all
    // GC-managed.  The collector reclaims them with the frame once it is
    // unreachable; `FrameBox::drop` only frees the `std::alloc` snapshot /
    // bootstrap fallback regime.  With no destructor or weakref flag,
    // `type_alloc_is_plain` admits PYFRAME's normal allocation fast paths.
    let pyframe_tid = gc.register_type(majit_gc::trace::TypeInfo::with_custom_trace(
        std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>(),
        pyframe_object_custom_trace,
    ));
    debug_assert_eq!(pyframe_tid, PYFRAME_GC_TYPE_ID);
    // `W_DictProxyObject` carries a single GC-traceable
    // `w_mapping: PyObjectRef` slot (the wrapped W_DictObject —
    // `pypy/objspace/std/dictproxyobject.py:17 self.w_mapping =
    // w_mapping`).  Pre-register it here so that
    // `MAPPING_PROXY_TYPE` resolves to a TypeInfo with the
    // correct payload size + gc_ptr offsets (the foreign-pytype
    // loop below would otherwise approximate it as
    // `sizeof(PyObject)` and miss the `w_mapping` trace slot,
    // dropping the wrapped dict on minor collection).
    let w_dict_proxy_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::dictproxyobject::W_DictProxyObject>(),
        object_tid,
        pyre_object::dictproxyobject::W_DICT_PROXY_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(w_dict_proxy_tid, W_DICT_PROXY_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::MAPPING_PROXY_TYPE as *const _ as usize,
        w_dict_proxy_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::MAPPING_PROXY_TYPE as *const _ as usize,
        w_dict_proxy_tid,
    );
    // `pypy/objspace/std/dictmultiobject.py` — three sibling
    // W_DictView*Object classes (Keys / Values / Items) each
    // carry a `w_dict` PyObjectRef back to the source.  Pyre
    // folds the three into one `W_DictViewObject` struct + tag; all
    // three Python-visible PyTypes (`DICT_KEYS_TYPE` /
    // `DICT_VALUES_TYPE` / `DICT_ITEMS_TYPE`) share the same tid
    // / vtable / size / offsets so the view's `w_dict` slot is
    // traced regardless of which kind it represents.
    let w_dict_view_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::dictmultiobject::W_DictViewObject>(),
        object_tid,
        pyre_object::dictmultiobject::W_DICT_VIEW_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(
        w_dict_view_tid,
        pyre_object::dictmultiobject::W_DICT_VIEW_GC_TYPE_ID
    );
    for tp in [
        &pyre_object::dictmultiobject::DICT_KEYS_TYPE,
        &pyre_object::dictmultiobject::DICT_VALUES_TYPE,
        &pyre_object::dictmultiobject::DICT_ITEMS_TYPE,
    ] {
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            tp as *const _ as usize,
            w_dict_view_tid,
        );
        pytype_to_tid.insert(tp as *const _ as usize, w_dict_view_tid);
    }
    // `pypy/interpreter/typedef.py:312-326 class GetSetProperty`
    // — fget/fset/fdel/doc/reqcls/name are W_Root references.
    // Pyre's `GetSetProperty` ports them as inline fields; the
    // GC must trace each so descriptors built before
    // `init_typeobjects` (e.g. function.__doc__ / __annotations__)
    // survive minor collection.  Registered after the dict-view
    // tid so `W_GETSET_PROPERTY_GC_TYPE_ID = 40` lines up with
    // the post-`W_DICT_VIEW_GC_TYPE_ID = 39` slot.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::typedef::GetSetProperty
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // resume.py:1444-1447 allocate_array(length, arraydescr, clear)
    // delegates to cpu.bh_new_array(), which in turn requires the
    // live ArrayDescr to carry the GC type id set by
    // GcLLDescr_framework.init_array_descr (gc.py:544-549).  These
    // two primitive GcArray lltypes have the same trace shape but are
    // distinct ARRAY identities, so register separate tids.
    let gc_int_array_tid = gc.register_type(TypeInfo::varsize(
        pyre_object::GC_TYPED_ARRAY_ITEMS_OFFSET,
        std::mem::size_of::<i64>(),
        pyre_object::GC_TYPED_ARRAY_LEN_OFFSET,
        false,
        Vec::new(),
    ));
    debug_assert_eq!(gc_int_array_tid, GC_INT_ARRAY_GC_TYPE_ID);
    let gc_float_array_tid = gc.register_type(TypeInfo::varsize(
        pyre_object::GC_TYPED_ARRAY_ITEMS_OFFSET,
        std::mem::size_of::<f64>(),
        pyre_object::GC_TYPED_ARRAY_LEN_OFFSET,
        false,
        Vec::new(),
    ));
    debug_assert_eq!(gc_float_array_tid, GC_FLOAT_ARRAY_GC_TYPE_ID);
    // `pypy/interpreter/pycode.py:52 class PyCode(W_Root)` — code
    // objects are normal GC heap objects in PyPy.  Pre-register
    // `PyCode` here, immediately after the GcArray tids and
    // before the foreign-pytype loop, so it takes tid 43 and the
    // loop skips `CODE_TYPE` via the `pytype_to_tid.contains_key`
    // guard below.  This keeps the net register-call count up to
    // `W_MODULE_DICT_GC_TYPE_ID = 48` unchanged (one explicit
    // registration here, one fewer from the loop), so no downstream
    // hardcoded tid shifts.  Allocation routes through `Box::into_raw`
    // (`w_code_new`), so this TypeInfo trace never fires and it registers
    // with empty gc_ptr offsets.  Its one movable GCREF slot, `w_globals`
    // (the cached globals dict object — movable for `exec`/custom-globals
    // dicts), is instead forwarded as a root by
    // `pyre_interpreter::eval::walk_raw_code_roots`, reached through
    // `walk_raw_function_roots` (`func.code`) and the frame root walk
    // (`frame.pycode`); a Box-immortal code object is never reachable by
    // tracing into it.  This registration stays inert until `w_code_new`
    // switches to `try_gc_alloc_stable`.
    let w_code_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_interpreter::pycode::PyCode>(),
        object_tid,
    ));
    debug_assert_eq!(w_code_tid, pyre_interpreter::pycode::W_CODE_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize,
        w_code_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::pycode::CODE_TYPE as *const _ as usize,
        w_code_tid,
    );
    // `pytraceback.py:17 PyTraceback` — pre-registered here, right
    // after PyCode, with a custom trace that forwards `w_next` /
    // `w_code` and (when GC-owned) the raw `frame` edge so a frame
    // reachable only through `tb.tb_frame` survives.  Like PyCode it
    // stays in `all_foreign_pytypes()` and is skipped by the loop's
    // `contains_key` guard, so the net register-call count through
    // `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged (one explicit
    // registration here, one fewer from the loop) — no downstream
    // hardcoded tid shifts.  Allocation routes through
    // `try_gc_alloc_stable` (`w_pytraceback_new`), so the trace fires
    // for real oldgen tracebacks.
    let w_pytraceback_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        std::mem::size_of::<pyre_interpreter::pytraceback::PyTraceback>(),
        object_tid,
        pytraceback_object_custom_trace,
    ));
    debug_assert_eq!(
        w_pytraceback_tid,
        pyre_interpreter::pytraceback::PYTRACEBACK_GC_TYPE_ID
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE as *const _ as usize,
        w_pytraceback_tid,
    );
    pytype_to_tid.insert(
        &pyre_interpreter::pytraceback::PYTRACEBACK_TYPE as *const _ as usize,
        w_pytraceback_tid,
    );
    // W_ObjectObject's PyType (`INSTANCE_TYPE`) stays bound to
    // `object_tid` (`OBJECT_GC_TYPE_ID = 0`) in `pytype_to_tid`:
    // it is the `object` root, and giving the *vtable* a separate
    // preorder id would corrupt the `subclass_range` hierarchy
    // (disjoint sub-ranges for one root, breaking `object ⊇ int` —
    // see eval::tests::test_subclass_range_preorder_bounds). The
    // dedicated `W_OBJECT_OBJECT_GC_TYPE_ID` registered above is a GC
    // *header* id (size + custom trace), an independent axis that
    // the collector reads off the header `w_instance_new` stamps;
    // it is deliberately absent from `pytype_to_tid`.
    // Walk every remaining built-in PyType and register one
    // `TypeInfo::object_subclass` per class, mirroring how
    // `assign_inheritance_ids` (normalizecalls.py:373-389) walks
    // `bk.bookkeeper.classdefs`. Each entry resolves its parent
    // through `pytype_to_tid`, so the resulting hierarchy obeys
    // `int_between(cls.min, subcls.min, cls.max)` (rclass.py:1133).
    // `pyre_object::pyobject::all_foreign_pytypes()` covers object
    // module PyTypes; `pyre_interpreter::all_foreign_pytypes()`
    // covers interpreter-level PyTypes (FUNCTION_TYPE /
    // BUILTIN_CODE_TYPE) that flow through tracing as constant
    // callable/code pointers.  `CODE_TYPE` and `PYTRACEBACK_TYPE` are
    // both pre-registered above and so skipped here by the
    // `contains_key` guard.
    for (pytype, parent) in pyre_object::pyobject::all_foreign_pytypes()
        .iter()
        .chain(pyre_interpreter::all_foreign_pytypes().iter())
    {
        let pytype_ptr = *pytype as *const _ as usize;
        // BOOL_TYPE / LIST_TYPE / TUPLE_TYPE / RANGE_ITER_TYPE are
        // pre-registered above with their real struct sizes. Leave
        // those bindings intact instead of overwriting them with a
        // `sizeof(PyObject)` approximation.
        if pytype_to_tid.contains_key(&pytype_ptr) {
            continue;
        }
        let parent_tid = *pytype_to_tid
            .get(&(*parent as *const _ as usize))
            .expect("foreign pytype parent must be registered before its subclass");
        let tid = gc.register_type(TypeInfo::object_subclass(
            std::mem::size_of::<pyre_object::PyObject>(),
            parent_tid,
        ));
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, tid);
        pytype_to_tid.insert(pytype_ptr, tid);
    }
    // `pypy/objspace/std/dictmultiobject.py:328 W_ModuleDictObject`
    // — module / globals dict carrying its own storage + strategy
    // pair (the celldict.py:ModuleDictStrategy port).  Separate GC
    // tid (`W_MODULE_DICT_GC_TYPE_ID=48`) so the allocator can tell
    // module dicts apart from regular dicts even though both
    // surface as Python's `dict` via the `MODULE_DICT_TYPE` static.
    // Registered after the foreign_pytypes loop so it occupies the
    // tail slot 48, one past the five tids the loop assigns to
    // NONE_TYPE (43), NOTIMPLEMENTED_TYPE (44), ELLIPSIS_TYPE (45),
    // CODE_TYPE (46) and PYTRACEBACK_TYPE (47); placing it between
    // W_DICT and W_SET would shift every subsequent tid by one and
    // break descr ↔ GC tid correspondence.
    // W_ModuleDictObject carries `dstorage: *mut ModuleDictStorage`
    // (`Vec<(String, PyObjectRef)>` of cells / raw values),
    // `mstrategy: *mut ModuleDictStrategy` (whose `caches`
    // GlobalCache.cell fields hold live cells), and
    // `object_storage: *mut Vec<(PyObjectRef, PyObjectRef)>` (active
    // after `switch_to_object_strategy`).  Register a custom trace
    // hook so the GC walks all three indirect storages — matching
    // the W_DictObject pattern at line 851.
    let w_module_dict_tid = gc.register_type(
        TypeInfo::object_subclass_with_custom_trace(
            std::mem::size_of::<pyre_object::dictmultiobject::W_ModuleDictObject>(),
            object_tid,
            module_dict_object_custom_trace,
        )
        .with_destructor_fn(module_dict_object_destructor),
    );
    debug_assert_eq!(w_module_dict_tid, W_MODULE_DICT_GC_TYPE_ID);
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::dictmultiobject::MODULE_DICT_TYPE as *const _ as usize,
        w_module_dict_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::dictmultiobject::MODULE_DICT_TYPE as *const _ as usize,
        w_module_dict_tid,
    );
    // `pypy/objspace/std/typeobject.py:22-71` cell layer:
    // `MutableCell` subclasses (`ObjectMutableCell`,
    // `IntMutableCell`) live inside `ModuleDictStorage` entries
    // and are unwrapped on the way out of the strategy.  They
    // never surface to user code so the static `PyType`s are
    // internal-only; allocate distinct GC tids so the bump
    // allocator can size them independently.
    //
    // `ObjectMutableCell.w_value` is a live `PyObjectRef` field
    // that must be traced during minor collection — otherwise the
    // wrapped value could be reclaimed while a still-installed
    // cell holds the pointer.  Mirrors `Cell`'s
    // `contents` registration (`nestedscope.rs:42`).
    let w_object_mutable_cell_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::celldict::ObjectMutableCell>(),
        object_tid,
        pyre_object::celldict::W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(
        w_object_mutable_cell_tid,
        pyre_object::celldict::W_OBJECT_MUTABLE_CELL_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::celldict::OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_object_mutable_cell_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::celldict::OBJECT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_object_mutable_cell_tid,
    );
    let w_int_mutable_cell_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::celldict::IntMutableCell>(),
        object_tid,
    ));
    debug_assert_eq!(
        w_int_mutable_cell_tid,
        pyre_object::celldict::W_INT_MUTABLE_CELL_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::celldict::INT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_int_mutable_cell_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::celldict::INT_MUTABLE_CELL_TYPE as *const _ as usize,
        w_int_mutable_cell_tid,
    );
    // WEAKREF GcStruct (gctypelayout.py:587). TypeInfo::weakref()
    // sets T_IS_WEAKREF so minor / major collections invalidate
    // the single weakptr slot when its target dies
    // (incminimark.py:3058-3126). pyre-object's
    // `pyre_object::weakref::Weakref` mirrors the layout; the
    // assert below pins the runtime tid to the constant it
    // hardcodes.
    let weakref_tid = gc.register_type(majit_gc::trace::TypeInfo::weakref());
    debug_assert_eq!(weakref_tid, pyre_object::weakref::WEAKREF_GC_TYPE_ID);
    debug_assert_eq!(
        std::mem::size_of::<pyre_object::weakref::Weakref>(),
        majit_gc::weakref::SIZEOF_WEAKREF,
        "pyre_object::weakref::Weakref layout must match majit_gc::weakref::Weakref",
    );
    debug_assert_eq!(
        std::mem::offset_of!(pyre_object::weakref::Weakref, weakptr),
        majit_gc::weakref::WEAKPTR_OFFSET,
        "weakptr field must sit at the offset majit_gc expects",
    );
    // GcWeakrefBox — instance-dict-slot wrapper around `*mut Weakref`.
    // Carries a single inline GcRef-shaped field (`inner`) so the
    // Weakref struct itself survives across collections; the
    // weakptr inside the Weakref is invalidated separately by the
    // collector's invalidate_*_weakrefs hooks.
    let gc_weakref_box_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
        std::mem::size_of::<pyre_object::weakref::GcWeakrefBox>(),
        object_tid,
        pyre_object::weakref::GC_WEAKREF_BOX_GC_PTR_OFFSETS.to_vec(),
    ));
    debug_assert_eq!(
        gc_weakref_box_tid,
        pyre_object::weakref::GC_WEAKREF_BOX_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::weakref::GC_WEAKREF_BOX_TYPE as *const _ as usize,
        gc_weakref_box_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::weakref::GC_WEAKREF_BOX_TYPE as *const _ as usize,
        gc_weakref_box_tid,
    );
    // `W_ObjectObject` keeps its attributes in an off-GC
    // `Box<Vec<PyObjectRef>>` `storage` list reachable only via a
    // custom trace (instance map+storage, `mapdict.py:907-910`).
    // Register a dedicated GC type id — stamped into the GC header
    // by `w_instance_new` — so a collection traces those value
    // slots (and reclaims dead instances; the storage `Vec` itself
    // forwards in place). `INSTANCE_TYPE` stays bound to `object_tid`
    // (above) for isinstance / `subclass_range`: the GC header id
    // (read by the collector for size + custom trace) and the
    // vtable preorder id are independent axes, so this id is NOT
    // inserted into `pytype_to_tid` and gets no `register_vtable`.
    let w_object_object_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
        pyre_object::objectobject::W_OBJECT_OBJECT_SIZE,
        object_tid,
        object_object_custom_trace,
    ));
    debug_assert_eq!(
        w_object_object_tid,
        pyre_object::objectobject::W_OBJECT_OBJECT_GC_TYPE_ID,
    );
    // W_ComplexObject carries two f64s after the `PyObject` header and
    // no managed pointers — a GC leaf like W_FloatObject.  Registered
    // immediately after the last hardcoded-constant tid (W_ObjectObject = 53)
    // so its fixed id 54 precedes the auto-numbered `#[pyre_class]` /
    // per-ExcKind tids registered below.  Bound to `COMPLEX_TYPE` so the
    // collector reads the correct size + leaf trace when a managed
    // container holds a complex.
    let w_complex_tid = gc.register_type(TypeInfo::object_subclass(
        std::mem::size_of::<pyre_object::complexobject::W_ComplexObject>(),
        object_tid,
    ));
    debug_assert_eq!(
        w_complex_tid,
        pyre_object::complexobject::W_COMPLEX_GC_TYPE_ID,
    );
    majit_gc::GcAllocator::register_vtable_for_type(
        &mut gc,
        &pyre_object::pyobject::COMPLEX_TYPE as *const _ as usize,
        w_complex_tid,
    );
    pytype_to_tid.insert(
        &pyre_object::pyobject::COMPLEX_TYPE as *const _ as usize,
        w_complex_tid,
    );
    // `W_ObjectObject.storage` block — the mapdict instance attribute-value
    // array (`mapdict.py:910`, `Ptr(GcArray(OBJECTPTR))`).  Registered as a
    // varsize leaf: the custom trace on the W_ObjectObject instance walks
    // each boxed storage slot by consulting the map to skip unboxed slots
    // (`instance_walk_boxed_storage`), and forwards this block pointer to
    // keep the (non-moving, stable-allocated) block marked live.  Length at
    // offset 0 (the `ItemsBlock.capacity` header), 8-byte ref items.
    // Registered here, immediately after `W_COMPLEX_GC_TYPE_ID = 54`, so it
    // takes tid 55 before the runtime-numbered `#[pyre_class]` / per-ExcKind
    // registrations below.
    let w_mapdict_storage_tid = gc.register_type(TypeInfo::varsize(
        pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET,
        std::mem::size_of::<pyre_object::pyobject::PyObjectRef>(),
        0,
        false,
        Vec::new(),
    ));
    debug_assert_eq!(
        w_mapdict_storage_tid,
        pyre_object::object_array::W_MAPDICT_STORAGE_GC_TYPE_ID,
    );
    // `#[pyre_class]`-emitted typed-payload registrations.  Each
    // entry is one line consuming the macro-generated
    // `PyreClassDescriptor` static; `register_pyre_class` asserts
    // the descriptor's `gc_type_id` matches the order here so the
    // hardcoded `type_id` constants on the `#[pyre_class]`
    // attribute cannot silently drift.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_random::W_Random
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // Per-`ExcKind` GC type ids.  The pre-registration loop at the
    // top of this function mapped every exception PyType to a
    // single `W_BASE_EXCEPTION_GC_TYPE_ID` so `new_with_vtable` knows
    // the `W_BaseException` payload size for allocation; the
    // shared tid also meant `gc.subclass_range(any_exception_
    // pytype)` returned the same range for every subclass, which
    // collapses RPython's per-class `subclassrange_{min,max}`
    // discrimination (rclass.py:167-174 `OBJECT.typeptr = specific
    // class` + rclass.py:1133-1137 `ll_issubclass`).
    //
    // To restore per-class ranges without renumbering the post-31
    // hardcoded tid constants (W_GENERATOR_GC_TYPE_ID = 32, …,
    // PYTRACEBACK_GC_TYPE_ID = 43) or the W_MODULE_DICT /
    // W_*MUTABLE_CELL tids registered above, register a fresh tid
    // per `ExcKind` (except BaseException, which keeps
    // `W_BASE_EXCEPTION_GC_TYPE_ID`) AFTER all hardcoded registrations.
    // Each new TypeInfo carries the W_BaseException layout
    // (size + GC ptr offsets) so allocation still works, and the
    // correct `parent_typeid` so `freeze_types` builds the
    // preorder subclass tree.  Then `register_vtable_for_type`
    // overrides the earlier pytype → 31 mapping so
    // `subclass_range(pytype)` resolves to the per-class range.
    //
    // Order is topological: each entry's `parent_kind` is already
    // registered by the time the entry is reached.  `None` parent
    // means "direct child of BaseException" — the parent_tid is
    // `W_BASE_EXCEPTION_GC_TYPE_ID`.
    use pyre_object::interp_exceptions::{
        EXC_KIND_COUNT, ExcKind, W_BASE_EXCEPTION_GC_PTR_OFFSETS, exc_kind_to_pytype,
    };
    let exc_hierarchy: &[(ExcKind, Option<ExcKind>)] = &[
        (ExcKind::Exception, None),
        (ExcKind::SystemExit, None),
        (ExcKind::GeneratorExit, None),
        (ExcKind::ArithmeticError, Some(ExcKind::Exception)),
        (ExcKind::OverflowError, Some(ExcKind::ArithmeticError)),
        (ExcKind::ZeroDivisionError, Some(ExcKind::ArithmeticError)),
        (ExcKind::TypeError, Some(ExcKind::Exception)),
        (ExcKind::ValueError, Some(ExcKind::Exception)),
        // `pypy/module/exceptions/interp_exceptions.py:418
        // W_UnicodeError = _new_exception('UnicodeError',
        // W_ValueError, ...)` — intermediate parent for the two
        // Unicode error variants; must register before children
        // because `parent_kind` is resolved by `per_exc_tid`
        // lookup in this same loop.
        (ExcKind::UnicodeError, Some(ExcKind::ValueError)),
        (ExcKind::UnicodeDecodeError, Some(ExcKind::UnicodeError)),
        (ExcKind::UnicodeEncodeError, Some(ExcKind::UnicodeError)),
        // `pypy/module/exceptions/interp_exceptions.py:426
        // W_UnicodeTranslateError = _new_exception(...,
        // W_UnicodeError, ...)`.
        (ExcKind::UnicodeTranslateError, Some(ExcKind::UnicodeError)),
        (ExcKind::NameError, Some(ExcKind::Exception)),
        // `pypy/module/exceptions/interp_exceptions.py:474
        // W_LookupError = _new_exception('LookupError',
        // W_Exception, ...)` — intermediate parent for IndexError
        // and KeyError.
        (ExcKind::LookupError, Some(ExcKind::Exception)),
        (ExcKind::IndexError, Some(ExcKind::LookupError)),
        (ExcKind::KeyError, Some(ExcKind::LookupError)),
        (ExcKind::AttributeError, Some(ExcKind::Exception)),
        (ExcKind::RuntimeError, Some(ExcKind::Exception)),
        (ExcKind::NotImplementedError, Some(ExcKind::RuntimeError)),
        (ExcKind::RecursionError, Some(ExcKind::RuntimeError)),
        (ExcKind::StopIteration, Some(ExcKind::Exception)),
        (ExcKind::ImportError, Some(ExcKind::Exception)),
        (ExcKind::AssertionError, Some(ExcKind::Exception)),
        (ExcKind::ReferenceError, Some(ExcKind::Exception)),
        (ExcKind::OSError, Some(ExcKind::Exception)),
        (ExcKind::FileNotFoundError, Some(ExcKind::OSError)),
        (ExcKind::MemoryError, Some(ExcKind::Exception)),
        (ExcKind::SystemError, Some(ExcKind::Exception)),
    ];
    // Per-kind tid lookup, seeded so BaseException resolves to
    // `W_BASE_EXCEPTION_GC_TYPE_ID`; unmapped slots also fall back to
    // it which is harmless because every reachable kind is
    // assigned its own tid by the loop below.
    let mut per_exc_tid: [u32; EXC_KIND_COUNT] = [W_BASE_EXCEPTION_GC_TYPE_ID; EXC_KIND_COUNT];
    per_exc_tid[ExcKind::BaseException as u8 as usize] = w_exception_tid;
    for (kind, parent_kind) in exc_hierarchy {
        let parent_tid = parent_kind
            .map(|p| per_exc_tid[p as u8 as usize])
            .unwrap_or(W_BASE_EXCEPTION_GC_TYPE_ID);
        let new_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            std::mem::size_of::<pyre_object::interp_exceptions::W_BaseException>(),
            parent_tid,
            W_BASE_EXCEPTION_GC_PTR_OFFSETS.to_vec(),
        ));
        per_exc_tid[*kind as u8 as usize] = new_tid;
        let pytype_ptr = exc_kind_to_pytype(*kind) as *const _ as usize;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, pytype_ptr, new_tid);
        pytype_to_tid.insert(pytype_ptr, new_tid);
    }
    // W_SRE_Pattern / W_SRE_Match / W_SRE_Scanner (`_sre` compiled
    // pattern, match result, and finditer scanner) — typed payloads
    // via `#[pyre_class]` in AUTO-ID mode.  The leaked engine buffers
    // (`code`, `spans`) are non-GC raw pointers the macro's
    // auto-detection skips; scanner's pattern/string refs must be
    // traced like PyPy's W_SRE_Scanner fields.  Registered at the
    // tail of the tid chain: every earlier slot is pinned by an
    // explicit `type_id = N` constant or a hardcoded comment-counted
    // position, so an insertion anywhere above would shift them all.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Pattern
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Match
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_sre::W_SRE_Scanner
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // GenericAlias (`types.GenericAlias`, PEP 585) — typed payload
    // via `#[pyre_class]` in AUTO-ID mode.  Its three `PyObjectRef`
    // fields (origin/args/parameters) are traced edges; registered at
    // the tail of the tid chain alongside the `_sre` types so no
    // explicit-id slot above shifts.  Absent from `all_foreign_pytypes`,
    // so this is the only path that GC-manages it.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::_pypy_generic_alias::GenericAlias
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Pickler / W_Unpickler (`_pickle` accelerator) — typed payloads
    // via `#[pyre_class]` in AUTO-ID mode.  Both carry inline
    // `PyObjectRef` fields (the pickler's output file; the unpickler's
    // read/readline callables, result stack, and active frame) that the
    // collector must walk.  Registered at the tail of the tid chain so
    // no earlier explicit-id slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_pickle::W_Pickler
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_pickle::W_Unpickler
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_PickleBuffer (`__pypy__.PickleBuffer`) — typed payload via
    // `#[pyre_class]` in AUTO-ID mode; its `w_obj` field is a traced
    // edge the collector must walk.  Tail of the tid chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::__pypy__::W_PickleBuffer
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // PicklerMemoProxy / UnpicklerMemoProxy — typed payloads via
    // `#[pyre_class]` in AUTO-ID mode; each holds one traced `PyObjectRef`
    // back-reference to its owning pickler/unpickler. Tail of the tid chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_pickle::PicklerMemoProxy
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_interpreter::module::_pickle::UnpicklerMemoProxy
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_ReversedIterator (`reversed`) — typed payload via `#[pyre_class]`
    // in AUTO-ID mode; its `w_sequence` field is a traced edge the
    // collector must walk. Tail of the tid chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_ReversedIterator
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Filter (`filter`) — AUTO-ID; its `w_predicate` / `w_iterable`
    // fields are traced edges the collector must walk.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Filter as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Map (`map`) — AUTO-ID; `w_fun` / `w_iterators` are traced edges.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Map as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Zip (`zip`) — AUTO-ID; `w_iterators` is a traced edge.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::functional::W_Zip as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Cycle (`itertools.cycle`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Unlike the other itertools iterators, its `saved`
    // list is owned solely by the W_Cycle (no external root), so the
    // collector must trace both the `w_iterable` source and the `saved`
    // replay buffer.  Tail of the tid chain so no earlier slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Cycle
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Array (`array.array`) — typed payload via `#[pyre_class]`
    // in AUTO-ID mode; its elements are unboxed scalars in an off-GC
    // `*mut Vec<u8>` buffer (the bytearray storage model), so the
    // descriptor reports zero traced pointer fields.  Tail of the tid
    // chain.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_array::W_Array as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_Chain (`itertools.chain`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Both the `w_iterables` source iterator and the current
    // sub-iterator `w_it` are owned solely by the W_Chain (no external
    // root), so the collector must trace both edges.  Tail of the
    // register_pyre_class chain so no earlier slot shifts.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::interp_itertools::W_Chain
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // W_MemoryView (`memoryview`) — typed payload via `#[pyre_class]` in
    // AUTO-ID mode.  Its geometry and backing live in an off-heap
    // `*const BufferView`, so the macro's empty `gc_ptr_offsets` reach
    // none of the view's refs; register a custom trace
    // (`memoryview_object_custom_trace`) that walks the box, mirroring
    // `W_ListObject` / `W_TupleObject`, plus a lightweight destructor
    // (`memoryview_object_destructor`) that frees the `std::alloc` box
    // itself when a dead header is swept so repeated view/slice/cast
    // churn does not leak.  Absent from `all_foreign_pytypes`, so this is
    // the only path that GC-manages it.  Registered at the tail of the
    // tid chain so no earlier explicit-id / hardcoded-constant slot
    // shifts; this replicates `register_pyre_class`'s tid stamp /
    // vtable / `pytype_to_tid` wiring with the custom-trace `TypeInfo`.
    {
        let mv_descr = <pyre_object::memoryview::W_MemoryView
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR;
        let mv_tid = gc.register_type(
            TypeInfo::object_subclass_with_custom_trace(
                mv_descr.object_size,
                object_tid,
                memoryview_object_custom_trace,
            )
            .with_destructor_fn(memoryview_object_destructor),
        );
        mv_descr.gc_type_id.set(mv_tid);
        majit_gc::GcAllocator::register_vtable_for_type(
            &mut gc,
            mv_descr.pytype_ptr as usize,
            mv_tid,
        );
        pytype_to_tid.insert(mv_descr.pytype_ptr as usize, mv_tid);
    }
    // Raw `BigInt` payload backing every `W_LongObject.value` (and the JIT
    // `jit_w_long_*_raw` results). Not an `rclass.OBJECT` instance — a bare
    // payload with no gc-pointer fields (malachite's limb `Vec` is off-GC),
    // carrying a lightweight destructor that runs `BigInt`'s drop glue so
    // the limbs are freed instead of leaked when the collector reclaims a
    // dead bigint. Registered at runtime id (no fixed const) and published
    // to pyre-object via `set_bigint_gc_type_id`; the id is never embedded
    // in a JIT descr (bigints are host-allocated, never `NewWithVtable`'d).
    let bigint_tid = gc.register_type(
        TypeInfo::with_destructor(
            pyre_object::longobject::BIGINT_PAYLOAD_SIZE,
            pyre_object::longobject::bigint_destructor,
        )
        .with_external_size(pyre_object::longobject::bigint_external_size),
    );
    pyre_object::longobject::set_bigint_gc_type_id(bigint_tid);
    // PyPy's FrameDebugData is a plain GC object. It owns three PyObjectRef
    // fields; once the frame custom trace greys the payload, the ordinary
    // offset walker finds all of them during a major mark.
    let frame_debug_data_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        std::mem::size_of::<pyre_interpreter::pyframe::FrameDebugData>(),
        vec![
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_locals),
            std::mem::offset_of!(pyre_interpreter::pyframe::FrameDebugData, w_f_trace),
            std::mem::offset_of!(
                pyre_interpreter::pyframe::FrameDebugData,
                hidden_operationerr
            ),
        ],
    ));
    debug_assert_eq!(frame_debug_data_tid, FRAME_DEBUG_DATA_GC_TYPE_ID);
    // Block-stack nodes are young GC objects. `previous` is their only GC
    // edge, so the normal walker forwards and major-marks an entire chain.
    let frame_block_tid = gc.register_type(TypeInfo::with_gc_ptrs(
        std::mem::size_of::<pyre_interpreter::pyframe::FrameBlock>(),
        vec![std::mem::offset_of!(
            pyre_interpreter::pyframe::FrameBlock,
            previous
        )],
    ));
    debug_assert_eq!(frame_block_tid, FRAME_BLOCK_GC_TYPE_ID);
    // setobject.py W_SetIterObject — AUTO-ID typed payload. Its live `w_set`
    // edge is traced, preserving a source owned solely by an iterator. Keep
    // this at the absolute tail of the type-id chain: inserting it before the
    // fixed FrameDebugData/FrameBlock slots changes their generated ids and
    // corrupts JIT frame payloads during guard/resume.
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::setobject::W_SetIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_ListIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_ListReverseIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    register_pyre_class(
        &mut gc,
        &mut pytype_to_tid,
        <pyre_object::iterobject::W_TupleIterObject
            as pyre_object::lltype::PyreClassPyTypeOf>::DESCRIPTOR,
    );
    // rclass.py:340-346 — assign subclassrange_{min,max} to each
    // vtable entry. freeze_types() runs assign_inheritance_ids
    // (normalizecalls.py:373-389), then we write the computed ranges
    // back into the static PyType structs so that ll_issubclass
    // (rclass.py:1133-1137) can read them directly from the typeptr.
    gc.freeze_types();
    // This writeback replaces every static `subclassrange_{min,max}`
    // with the GC-tid numbering, which differs from the preorder
    // numbering the interpreter seeds via `compute_subclass_ranges_from`.
    // Publish the whole batch inside one seqlock write section so a
    // concurrent interpreter `ll_issubclass` observes either the
    // all-preorder or the all-GC set, never a half-swapped mix — a mixed
    // read makes `ll_issubclass(TypeError, BaseException)` spuriously
    // false.
    {
        let _range_guard = pyre_object::pyobject::subclass_range_write_guard();
        for (&classptr, &_tid) in &pytype_to_tid {
            if let Some((min, max)) = gc.subclass_range(classptr) {
                let tp = unsafe { &*(classptr as *const pyre_object::pyobject::PyType) };
                pyre_object::pyobject::assign_subclass_range(tp, min, max);
            }
        }
    }
    Box::new(gc)
}

/// Store a zero-size `GcHandle` (routes through `gc_sync`) into the
/// active backend's thread-local.
#[cfg(target_arch = "wasm32")]
fn install_gc_into_backend() {
    majit_backend_wasm::install_gc_standalone();
}
#[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
fn install_gc_into_backend() {
    majit_backend_cranelift::install_gc_standalone();
}
#[cfg(all(
    feature = "dynasm",
    not(feature = "cranelift"),
    not(target_arch = "wasm32")
))]
fn install_gc_into_backend() {
    majit_backend_dynasm::runner::install_gc_standalone();
}

/// Phase B: root walkers that reference interpreter state (immortal dicts,
/// mapdict side table, etc.).  Called on first eval entry, after the
/// interpreter is initialized.
fn install_gc_root_walkers() {
    pyre_interpreter::eval::register_pyframe_root_walker();
}

fn register_thread_root_areas() {
    let register = majit_gc::shadow_stack::register_mutator_extra_area;
    let jit_driver = JIT_DRIVER.with(|cell| cell as *const _ as *const ());
    // SAFETY: every `data` pointer is this thread's own TLS / `'static` root
    // area, valid for the thread's registered lifetime, and every walk fn
    // dereferences addresses derived solely from its supplied `data` pointer.
    unsafe {
        register(
            pyframe_root_walker_area,
            pyre_interpreter::eval::capture_pyframe_root_area(),
        );
        register(
            pyre_object_root_walker_area,
            pyre_object::gc_roots::capture_shadow_stack_area(),
        );
        register(
            jitcode_constants_root_walker_area,
            pyre_jit_trace::state::capture_jitcode_constants_root_area(),
        );
        register(
            fbw_store_journal_root_walker_area,
            pyre_jit_trace::jitcode_dispatch::capture_fbw_store_journal_root_area(),
        );
        register(
            fbw_finish_concrete_root_walker_area,
            pyre_jit_trace::jitcode_dispatch::capture_fbw_finish_concrete_root_area(),
        );
        register(
            mapdict_root_walker_area,
            pyre_interpreter::objspace::std::mapdict::capture_mapdict_root_area(),
        );
        #[cfg(not(target_arch = "wasm32"))]
        register(
            signal_handler_root_walker_area,
            pyre_interpreter::module::signal::interp_signal::capture_signal_handler_root_area(),
        );
        register(
            weakref_box_inner_root_walker_area,
            pyre_object::weakref::capture_gc_weakref_box_root_area(),
        );
        register(
            sre_pattern_root_walker_area,
            pyre_object::interp_sre::capture_sre_pattern_root_area(),
        );
        register(
            jit_callee_frame_root_walker_area,
            crate::call_jit::capture_jit_callee_frame_root_area(),
        );
        register(rd_consts_root_walker_area, jit_driver);
        register(partial_trace_root_walker_area, jit_driver);
        register(active_trace_root_walker_area, jit_driver);
        register(compile_snapshot_root_walker_area, jit_driver);
    }
}

/// pyre-object GC hook trampolines — safe to install at boot because
/// they only store function pointers in pyre-object's thread-local
/// `Cell` slots and do not touch interpreter state.
fn install_pyre_object_hooks() {
    pyre_object::register_gc_alloc_hook(pyre_object_gc_alloc_trampoline);
    pyre_object::register_gc_alloc_stable_hook(pyre_object_gc_alloc_stable_trampoline);
    pyre_object::gc_hook::register_gc_alloc_collecting_hook(
        pyre_object_gc_alloc_collecting_trampoline,
    );
    pyre_object::gc_hook::register_gc_charge_memory_pressure_hook(
        pyre_object_gc_charge_memory_pressure_trampoline,
    );
    pyre_object::gc_hook::register_gc_charge_oldgen_external_hook(
        pyre_object_gc_charge_oldgen_external_trampoline,
    );
    pyre_object::register_gc_collect_hook(pyre_object_gc_collect_trampoline);
    pyre_object::gc_hook::register_gc_collect_oldgen_hook(pyre_object_gc_collect_oldgen_trampoline);
    pyre_object::gc_hook::register_gc_set_enabled_hook(pyre_object_gc_set_enabled_trampoline);
    pyre_object::gc_hook::register_gc_finalizer_hooks(
        pyre_object_gc_register_finalizer_trampoline,
        pyre_object_gc_finalizer_next_dead_trampoline,
    );
    pyre_object::gc_hook::register_gc_jitframe_empty_hook(pyre_object_gc_jitframe_empty_trampoline);
    pyre_object::register_gc_root_hooks(
        pyre_object_gc_add_root_trampoline,
        pyre_object_gc_remove_root_trampoline,
    );
    pyre_object::register_gc_owns_object_hook(pyre_object_gc_owns_object_trampoline);
    pyre_object::register_gc_current_object_address_hook(
        pyre_object_gc_current_object_address_trampoline,
    );
    pyre_object::register_gc_write_barrier_hook(pyre_object_gc_write_barrier_trampoline);
    pyre_object::gc_hook::register_gc_identity_hash_hook(pyre_object_gc_identity_hash_trampoline);
}

/// Build the GC once and store it in `gc_sync::GC_STORE`.
/// `gc_sync::is_initialized()` + `gc_sync::store_singleton()` ensures
/// exactly one GC is created even under cargo test's parallel threads.
fn build_gc_global() {
    if majit_gc::gc_sync::is_initialized() {
        return;
    }
    let gc = build_gc();
    // Publish the eval-breaker word address before store_singleton flips the
    // GC-initialized flag (Release). A concurrent initializer that observes the
    // flag set (Acquire) early-returns above; ordering the publish first makes
    // that observer also sees a non-zero address when recording the
    // back-edge eval-breaker poll.
    majit_ir::eval_breaker_word::publish_addr();
    majit_gc::gc_sync::store_singleton(gc);
}

/// Test-support: give the calling `gc_stress` worker a pristine GC heap by
/// installing a fresh GC and leaking the shared singleton. The per-test worker
/// threads share the process-global GC singleton; without this, a
/// class-defining test leaves oldgen residue or stale roots that a later
/// test's collection, run on a different worker thread with a different
/// thread-local root set, can mishandle and corrupt immortal state.
pub fn reset_gc_fresh_for_test() {
    let gc = build_gc();
    majit_gc::gc_sync::replace_singleton_leaking_old(gc);
}

/// Initialize the GC subsystem independently of the JIT driver.
///
/// Phase 1 (process-global, once): build MiniMarkGC, type registry,
/// subclass ranges, store in gc_sync singleton.
/// Phase 2a (per-thread): register this thread with the gc_sync mutator
/// registry and install the backend GC handle. `install_gc_into_backend`
/// also registers the `majit_gc::set_active_*` fn-pointer cells (add_root,
/// write_barrier, guard hooks, …). The backend still keeps a per-thread TLS
/// box (removed by #396 R4), so this part stays per-thread.
/// Phase 2b (process-global, once): install the pyre-object hook
/// trampolines. These are process-global fn-pointer cells (#396), so a
/// single install is visible to every thread. They route pyre-object
/// through the `set_active_*` cells from phase 2a, so phase 2a runs first —
/// a thread never publishes the pyre-object hooks before its own backend
/// `set_active_*` install.
pub fn init_gc_subsystem() {
    build_gc_global();
    if !GC_TLS_INSTALLED.with(|c| c.get()) {
        majit_gc::gc_sync::register_thread();
        majit_gc::shadow_stack::register_mutator();
        register_thread_root_areas();
        GC_MUTATOR_REGISTRATION.with(|_| {});
        install_gc_into_backend();
        GC_TLS_INSTALLED.with(|c| c.set(true));
    }
    PYRE_OBJECT_HOOKS_INSTALLED.call_once(install_pyre_object_hooks);
}

/// Guards the one-time install of the process-global pyre-object GC hooks.
static PYRE_OBJECT_HOOKS_INSTALLED: std::sync::Once = std::sync::Once::new();

thread_local! {
    static GC_ROOT_WALKERS_INSTALLED: Cell<bool> = const { Cell::new(false) };
}

/// Phase B of GC init: register root walkers that touch interpreter
/// state (immortal dicts, mapdict side table, etc.).  Must run after
/// the interpreter is initialized — called on first eval entry.
/// Idempotent.
pub fn init_gc_root_walkers() {
    if GC_ROOT_WALKERS_INSTALLED.with(|c| c.get()) {
        return;
    }
    install_gc_root_walkers();
    GC_ROOT_WALKERS_INSTALLED.with(|c| c.set(true));
}

thread_local! {
    static JIT_DRIVER: UnsafeCell<Option<JitDriverPair>> = const { UnsafeCell::new(None) };
}

fn build_jit_driver_pair() -> JitDriverPair {
    let info = build_pyframe_virtualizable_info();
    let mut d = JitDriver::new(JIT_THRESHOLD);
    d.set_virtualizable_info(info.clone());
    d.meta_interp_mut().num_scalar_inputargs =
        pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — install pyre's
    // `W_UnicodeObject` length reader so constant STRLEN / UNICODELEN ops
    // fold to `IntBound::from_constant(len)` during intbounds
    // postprocessing.
    //
    // PyPy returns the exact length for both modes:
    //
    //     def getstrlen1(self, mode):
    //         from rpython.jit.metainterp.optimizeopt import vstring
    //         if mode is vstring.mode_string:
    //             s = self._unpack_str(vstring.mode_string)
    //             ...
    //             return len(s)
    //         elif mode is vstring.mode_unicode:
    //             s = self._unpack_str(vstring.mode_unicode)
    //             ...
    //             return len(s)
    //
    // Pyre's `W_UnicodeObject.value` is a Rust `String` whose
    // `len()` returns the UTF-8 BYTE length and whose
    // `chars().count()` returns the codepoint count, so the resolver
    // needs different reads per mode:
    //
    //   * mode == 0 (`vstring.mode_string`, byte string) — return the
    //     UTF-8 byte length, which is what PyPy's `str.len()` would
    //     produce for an RPython byte string.
    //   * mode == 1 (`vstring.mode_unicode`, unicode string) — return
    //     the codepoint count, which is what Python 3's
    //     `len(str_object)` produces.
    d.meta_interp_mut()
        .set_string_length_resolver(std::sync::Arc::new(
            |gcref: majit_ir::GcRef, mode: u8| -> Option<i64> {
                if gcref.is_null() {
                    return None;
                }
                let obj = gcref.0 as pyre_object::pyobject::PyObjectRef;
                if !unsafe { pyre_object::unicodeobject::is_str(obj) } {
                    return None;
                }
                match mode {
                    // vstring.mode_string — UTF-8 byte length per
                    // `rstr.py:1226 Array(Char)` / `llmodel.py:667 bh_strlen`.
                    0 => {
                        let s = unsafe { pyre_object::unicodeobject::w_str_get_value(obj) };
                        Some(s.len() as i64)
                    }
                    // vstring.mode_unicode — codepoint count.
                    1 => {
                        let s = unsafe { pyre_object::unicodeobject::w_str_get_value(obj) };
                        Some(s.chars().count() as i64)
                    }
                    _ => None,
                }
            },
        ));
    d.set_vtable_offset(Some(pyre_object::pyobject::OB_TYPE_OFFSET));
    // resume.py:1367 — BlackholeAllocator for virtual materialization.
    d.register_blackhole_allocator(PyreBlackholeAllocator);
    // warmspot.py:1039 handle_jitexception_from_blackhole parity:
    // portal_runner is called when ContinueRunningNormally is raised
    // at a recursive portal level during blackhole execution.
    d.register_portal_runner(pyre_portal_runner);
    // pypy/module/pypyjit/interp_jit.py:72-78 PyPyJitDriver(..., is_recursive=True).
    // Drives MetaInterp.is_main_jitcode() / is_portal_jitcode dispatch
    // — without this flag the recursive-portal bookkeeping stays
    // disabled while is_main_jitcode() callers still assume it was
    // set, leaving the metadata internally inconsistent.
    d.set_is_recursive(true);
    // warmspot.py:449 — jd.result_type = getkind(portal.getreturnvar().concretetype)[0]
    // PyPy dispatch() returns W_Root → Ref.
    d.set_result_type(majit_ir::Type::Ref);
    // rlib/jit.py:842 set_user_param — the translation-time `--jit STR`
    // option's analog. `PYRE_JIT="vec_all=1"` opts vectorization in the
    // PyPy way (parameter; the defaults stay off). `PYRE_JIT=0` keeps its
    // existing disable meaning (handled on the can_enter_jit gate), so it
    // is skipped here.
    if let Ok(text) = std::env::var("PYRE_JIT") {
        let text = text.trim();
        if !text.is_empty() && text != "0" {
            let ws = d.meta_interp_mut().warm_state_mut();
            let _ = apply_jit_param_string(ws, text);
        }
    }
    // Publish the wasm CA deopt-helper's `__indirect_function_table` slot so
    // `compile_bridge` can lift a self-recursive CALL_ASSEMBLER bridge: the
    // CA arm `call_indirect`s it to blackhole-resume a callee that left its
    // trace through a guard. Taking the function's address keeps it in the
    // table; on wasm32 the address IS the table index. Done here (not only in
    // `init_jit_hooks`) because the wasm entry path reaches `driver_pair`
    // without `init_jit_hooks`.
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_ca_deopt_helper_slot(
        crate::call_jit::wasm_ca_resume_deopt as *const () as usize as u32,
    );
    #[cfg(target_arch = "wasm32")]
    majit_backend_wasm::set_ca_baseline_helper_slot(
        crate::call_jit::wasm_ca_baseline_call as *const () as usize as u32,
    );
    (d, info)
}

// dont_look_inside: JIT-driver global accessor (JIT_DRIVER TLS).
#[majit_macros::dont_look_inside]
pub fn driver_pair() -> &'static mut JitDriverPair {
    init_gc_subsystem();
    JIT_DRIVER.with(|cell| unsafe {
        let slot = &mut *cell.get();
        if slot.is_none() {
            *slot = Some(build_jit_driver_pair());
        }
        slot.as_mut().unwrap()
    })
}

/// framework.py `root_walker.walk_roots` hook for
/// `storage.rd_consts` (resume.py:451) across every live compiled
/// trace.
///
/// Registered once during `JIT_DRIVER` init (see
/// `register_thread_root_areas`). Routes into the thread-local
/// `JitDriver`'s `walk_rd_consts_refs`, which in turn iterates
/// `MetaInterp::compiled_loops` and visits the Ref-typed entries in
/// every `StoredExitLayout::rd_consts`.
unsafe fn rd_consts_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_rd_consts_refs(visitor);
    }
}

/// framework.py `root_walker.walk_roots` hook for the inline-Const
/// `ConstPtr` slots inside `MetaInterp.partial_trace.ops` —
/// history.py:314 `ConstPtr.value` lives on the OpRef itself, so the
/// walker iterates `partial.ops` and visits each `OpRef::ConstPtr`
/// arg / fail-arg directly. Routes into
/// `JitDriver::walk_partial_trace_refs`, which forwards to
/// `MetaInterp::walk_partial_trace_refs`.
unsafe fn partial_trace_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_partial_trace_refs(visitor);
    }
}

/// framework.py `root_walker.walk_roots` hook for the active recorder's
/// op-graph. Visits every inline `OpRef::ConstPtr(GcRef)` slot in
/// `op.args` / `op.fail_args` (history.py:314 `ConstPtr.value`).
/// No-op when no trace is in progress. Routes into
/// `JitDriver::walk_active_trace_refs`, which forwards to
/// `MetaInterp::walk_active_trace_refs`.
unsafe fn active_trace_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_active_trace_refs(visitor);
    }
}

/// GC walker for ConstPtr GcRefs extracted from snapshot maps
/// during compilation. history.py:314 ConstPtr.value is traced through
/// the Python object graph; pyre's SnapshotBox.opref slots in Rust Vecs
/// need explicit walking. See `MetaInterp::walk_compile_snapshot_refs`.
unsafe fn compile_snapshot_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if let Some(pair) = unsafe { jit_driver_pair_from_root_area(data) } {
        pair.0.walk_compile_snapshot_refs(visitor);
    }
}

/// Re-derives the thread-local `JitDriverPair` for a GC root walk from the
/// registered `JIT_DRIVER` cell pointer.
///
/// # Safety / aliasing
///
/// This mints a second `&mut JitDriverPair` from the `JIT_DRIVER` cell. A
/// collection triggered by an allocation under a live `driver_pair()` `&mut`
/// arrives here on the *same* thread — the allocating thread becomes the
/// collector and walks its own registered areas — so the outer borrow is a
/// suspended frame on this stack, not another thread's. Two live `&mut` to one
/// cell is undefined under stacked/tree borrows.
///
/// It has not been observed to miscompile: the walker only rewrites `GcRef`
/// slots in place (moving-GC forwarding) and the outer borrow re-reads them as
/// forwarded pointers when it resumes. That explains why it survives; it does
/// not license the aliasing.
///
/// The minimum sound configuration is a load-bearing triple — dropping any one
/// leg leaves the aliasing intact:
///   1. this walker stops forming a `&mut` (or `&`) to the pair;
///   2. `driver_pair()` stops returning `&'static mut` — the exclusivity is
///      minted there, and that is what invalidates every other access;
///   3. the GC-visible root fields (`compiled_loops` / `partial_trace` /
///      `exported_state` / `framestack` / `tracing`) move behind interior
///      mutability.
///
/// Wrapping the fields alone does NOT help: `UnsafeCell` only opts out of the
/// immutability of `&T`, while a `&mut T` retag is `Unique` over the whole
/// pointee, UnsafeCell bytes included. Nor does switching this walker to raw
/// pointers: its `data` is captured at registration, so its tag sits below the
/// outer `&mut`'s on the borrow stack and even a read through it pops that tag.
unsafe fn jit_driver_pair_from_root_area(data: *const ()) -> Option<&'static mut JitDriverPair> {
    let cell = unsafe { &*(data as *const UnsafeCell<Option<JitDriverPair>>) };
    unsafe { (&mut *cell.get()).as_mut() }
}

/// `framework.shadowstack walk_stack_root` adapter — walk every
/// pinned slot in `pyre_object::gc_roots`'s thread-local shadow
/// stack and forward each `&mut PyObjectRef` slot to the GC's
/// `&mut GcRef` visitor. Both types are pointer-sized:
/// `PyObjectRef = *mut PyObject` and `GcRef` is
/// `#[repr(transparent)]` over `usize`, so the cast is layout-safe.
fn pyre_object_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::gc_roots::walk_shadow_stack(|slot: &mut pyre_object::PyObjectRef| {
        // SAFETY: `PyObjectRef` and `GcRef` are both pointer-sized
        // and trivially transmutable (one is `*mut PyObject`, the
        // other is `#[repr(transparent)] struct GcRef(pub usize)`).
        // Reinterpreting the slot in place lets a moving collector
        // rewrite the address through the visitor — the rewrite is
        // observed by subsequent `pin_root` / `shadow_stack_get`
        // callers.
        let gcref: &mut majit_ir::GcRef =
            unsafe { &mut *(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) };
        visitor(gcref);
    });
}

unsafe fn pyre_object_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_object::gc_roots::walk_shadow_stack_area(data, |slot| {
            visit_pyobject_root(slot, visitor);
        });
    }
}

unsafe fn pyframe_root_walker_area(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    unsafe { pyre_interpreter::eval::walk_pyframe_roots_area(data, visitor) };
}

unsafe fn jitcode_constants_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { pyre_jit_trace::state::walk_jitcode_constants_refs_area(data, visitor) };
}

unsafe fn fbw_store_journal_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { pyre_jit_trace::jitcode_dispatch::fbw_store_journal_root_walker_area(data, visitor) };
}

unsafe fn fbw_finish_concrete_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_root_walker_area(data, visitor)
    };
}

unsafe fn mapdict_root_walker_area(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    unsafe {
        pyre_interpreter::objspace::std::mapdict::walk_mapdict_roots_area(data, |slot| {
            visit_pyobject_root(slot, visitor);
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn signal_handler_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_interpreter::module::signal::interp_signal::walk_signal_handler_roots_area(
            data,
            |slot| visit_pyobject_root(slot, visitor),
        );
    }
}

unsafe fn weakref_box_inner_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_object::weakref::walk_gc_weakref_box_inner_roots_area(data, |slot| {
            visit_pyobject_root(slot, visitor);
        });
    }
}

unsafe fn sre_pattern_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        pyre_object::interp_sre::walk_sre_pattern_roots_area(data, |slot| {
            visit_pyobject_root(slot, visitor);
        });
    }
}

unsafe fn jit_callee_frame_root_walker_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { crate::call_jit::walk_jit_callee_frame_roots_area(data, visitor) };
}

fn visit_pyobject_root(
    slot: &mut pyre_object::PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let gcref: &mut majit_ir::GcRef =
        unsafe { &mut *(slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) };
    visitor(gcref);
}

fn pyre_interpreter_side_table_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_interpreter::objspace::std::mapdict::walk_mapdict_roots(|slot| {
        visit_pyobject_root(slot, visitor);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn signal_handler_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_interpreter::module::signal::interp_signal::walk_signal_handler_roots(|slot| {
        visit_pyobject_root(slot, visitor);
    });
}

fn weakref_box_inner_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::weakref::walk_gc_weakref_box_inner_roots(|slot| {
        visit_pyobject_root(slot, visitor);
    });
}

fn sre_pattern_root_walker(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    pyre_object::interp_sre::walk_sre_pattern_roots(|slot| {
        visit_pyobject_root(slot, visitor);
    });
}

// GREEN_KEY_ALIASES removed: compile.py:269 parity — cross-loop cut
// traces are now stored directly under the inner loop's green_key
// (cut_inner_green_key) in compile_loop, matching RPython's
// jitcell_token = cross_loop.jitcell_token. No alias dispatch needed.

/// Return a raw pointer to the thread-local VirtualizableInfo.
/// Used by the blackhole to implement BC_GETFIELD_VABLE_* bytecodes.
pub(crate) fn get_virtualizable_info() -> *const majit_metainterp::virtualizable::VirtualizableInfo
{
    let pair = driver_pair();
    std::sync::Arc::as_ptr(&pair.1)
}

/// pypy/module/pypyjit/interp_jit.py → PyPyJitDriver(JitDriver).
///
/// Mirrors RPython JitDriver (`rpython/rlib/jit.py:610-693`) field set:
/// class-level attrs (`virtualizables`, `greens`, `reds`) from
/// interp_jit.py:67-71 and constructor kwargs from interp_jit.py:72-78
/// frozen onto a single static instance, matching the upstream
/// `pypyjitdriver = PyPyJitDriver(...)` module-scope binding.
#[derive(Clone, Copy)]
pub struct PyPyJitDriver {
    /// rlib/jit.py:617 `active = True` — class attr controlling whether
    /// the marker fires.
    pub active: bool,
    /// rlib/jit.py:618 / interp_jit.py:70 `virtualizables = ['frame']`.
    pub virtualizables: &'static [&'static str],
    /// rlib/jit.py:619 / interp_jit.py:77 `name = 'pypyjit'`.
    pub name: &'static str,
    /// rlib/jit.py:620 `inline_jit_merge_point = False`.
    pub inline_jit_merge_point: bool,
    /// rlib/jit.py:649-650 / interp_jit.py:69
    /// `greens = ['next_instr', 'is_being_profiled', 'pycode']`.
    pub greens: &'static [&'static str],
    /// rlib/jit.py:652-662 / interp_jit.py:68 `reds = ['frame', 'ec']`.
    pub reds: &'static [&'static str],
    /// rlib/jit.py:653/661 — True iff `reds='auto'`.
    pub autoreds: bool,
    /// rlib/jit.py:655/662 — `len(reds)`; `None` when `autoreds`.
    pub numreds: Option<usize>,
    /// rlib/jit.py:684 — `has_unique_id = (get_unique_id is not None)`.
    /// Stays in sync with `get_unique_id` below.
    pub has_unique_id: bool,
    /// rlib/jit.py:691 `check_untranslated=True` default.
    pub check_untranslated: bool,
    /// rlib/jit.py:692 / interp_jit.py:78 `is_recursive=True`.
    pub is_recursive: bool,
    /// rlib/jit.py:693 `vec = vectorize` default False.
    pub vec: bool,

    /// rlib/jit.py:682 — `get_printable_location` hook callable.
    pub get_printable_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> String>,
    /// rlib/jit.py:683 — `get_location` hook callable.
    pub get_location: Option<fn(usize, bool, pyre_object::PyObjectRef) -> pyre_object::PyObjectRef>,
    /// rlib/jit.py:685-687 — `get_unique_id` hook callable.
    pub get_unique_id: Option<fn(usize, bool, pyre_object::PyObjectRef) -> usize>,
    /// rlib/jit.py:690 — `should_unroll_one_iteration` hook callable.
    pub should_unroll_one_iteration: Option<fn(usize, bool, pyre_object::PyObjectRef) -> bool>,
    /// rlib/jit.py:688 — `confirm_enter_jit` hook (concrete pyre signature
    /// is wired alongside S1.3 specialize_call; until then, `None`).
    pub confirm_enter_jit: Option<fn() -> bool>,
    /// rlib/jit.py:689 — `can_never_inline` hook (signature ported with S1.3).
    pub can_never_inline: Option<fn() -> bool>,
}

impl PyPyJitDriver {
    /// interp_jit.py:85-87 — jit_merge_point inside dispatch loop.
    /// API-parity stub: the merge point is handled inside
    /// `eval_loop_jit`'s `jit_merge_point_hook` until the S3 cutover
    /// replaces this with the upstream marker call.
    pub fn jit_merge_point(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (frame, ec, next_instr, pycode, is_being_profiled);
    }

    /// interp_jit.py:114-117 — can_enter_jit at back-edge.
    /// API-parity stub: handled by `eval_loop_jit`'s
    /// `maybe_compile_and_run` on `StepResult::CloseLoop`.
    pub fn can_enter_jit(
        &self,
        frame: &mut PyFrame,
        ec: *const PyExecutionContext,
        next_instr: usize,
        pycode: pyre_object::PyObjectRef,
        is_being_profiled: bool,
    ) {
        let _ = (frame, ec, next_instr, pycode, is_being_profiled);
    }
}

/// pypy/module/pypyjit/interp_jit.py:72-78 —
/// `pypyjitdriver = PyPyJitDriver(...)`.
///
/// All four upstream hook kwargs that interp_jit.py:72-76 passes are
/// wired to the per-hook pyre implementations defined later in this
/// file (`get_printable_location`, `get_location`, `get_unique_id`,
/// `should_unroll_one_iteration`). `has_unique_id` mirrors
/// `get_unique_id` per rlib/jit.py:684 so the two cannot drift.
///
/// Field defaults that match `JitDriver.__init__` (rlib/jit.py:610-693)
/// when the corresponding kwarg is not passed:
///
///   - `active = true`               ← rlib/jit.py:617 class attr.
///   - `inline_jit_merge_point = false` ← rlib/jit.py:670.
///   - `autoreds = false`            ← interp_jit.py passes a list, not 'auto'.
///   - `check_untranslated = true`   ← rlib/jit.py:674.
///   - `vec = false`                 ← rlib/jit.py:693.
///   - `confirm_enter_jit = None`    ← interp_jit.py omits the kwarg, so
///                                     `JitDriver.__init__` (rlib/jit.py:680)
///                                     leaves the slot as the class-level
///                                     `confirm_enter_jit = None` default.
///   - `can_never_inline = None`     ← same path: rlib/jit.py:681 default
///                                     because interp_jit.py omits it.
pub const pypyjitdriver: PyPyJitDriver = PyPyJitDriver {
    active: true,
    virtualizables: &["frame"],
    name: "pypyjit",
    inline_jit_merge_point: false,
    greens: &["next_instr", "is_being_profiled", "pycode"],
    reds: &["frame", "ec"],
    autoreds: false,
    numreds: Some(2),
    has_unique_id: true,
    check_untranslated: true,
    is_recursive: true,
    vec: false,
    get_printable_location: Some(get_printable_location),
    get_location: Some(get_location),
    get_unique_id: Some(get_unique_id),
    should_unroll_one_iteration: Some(should_unroll_one_iteration),
    // interp_jit.py:72-78 omits these kwargs — keep at upstream
    // default `None` (see field-default block above).
    confirm_enter_jit: None,
    can_never_inline: None,
};

/// interp_jit.py:77 — class __extend__(PyFrame)
///
/// In RPython, __extend__ adds methods to PyFrame. In Rust, PyFrame methods
/// are defined directly; this struct provides the interp_jit.py API surface.
pub struct __extend__;

impl __extend__ {
    /// interp_jit.py:79-96 — dispatch(self, pycode, next_instr, ec).
    ///
    /// RPython:
    ///   while True:
    ///       pypyjitdriver.jit_merge_point(ec=ec, frame=self, ...)
    ///       next_instr = self.handle_bytecode(co_code, next_instr, ec)
    ///   except Yield: ...
    ///   except ExitFrame: ...
    ///
    /// In pyre, the JIT-instrumented dispatch loop is eval_loop_jit().
    /// pycode and ec are stored on the frame; eval_loop_jit reads them
    /// from frame.pycode and frame.execution_context respectively.
    pub fn dispatch(
        frame: &mut PyFrame,
        _pycode: pyre_object::PyObjectRef,
        next_instr: usize,
        _ec: *const PyExecutionContext,
    ) -> PyResult {
        majit_gc::gc_sync::safepoint_poll();
        frame.set_last_instr_from_next_instr(next_instr);
        // interp_jit.py:79-96 dispatch: the while-True loop runs until
        // Yield or ExitFrame. ContinueRunningNormally means portal
        // re-entry (warmspot.py:976), not a silent return.
        handle_jitexception(frame)
    }

    /// interp_jit.py:102-121 — jump_absolute(self, jumpto, next_instr, ec).
    ///
    /// RPython:
    ///   def jump_absolute(self, jumpto, next_instr, ec):
    ///       jumpto *= 2
    ///       if jumpto >= next_instr:    # no backward jump
    ///           return jumpto
    ///       if we_are_jitted():
    ///           decr_by = 0
    ///           if self.space.actionflag.has_bytecode_counter:
    ///               if self.space.threadlocals.gil_ready:
    ///                   decr_by = _get_adapted_tick_counter()
    ///           self.last_instr = intmask(jumpto)
    ///           ec.bytecode_trace(self, decr_by)
    ///           jumpto = r_uint(self.last_instr)
    ///       pypyjitdriver.can_enter_jit(frame=self, ec=ec,
    ///           next_instr=jumpto, pycode=self.getcode(),
    ///           is_being_profiled=self.get_is_being_profiled())
    ///       return jumpto
    pub fn jump_absolute(
        frame: &mut PyFrame,
        mut jumpto: usize,
        next_instr: usize,
        ec: *mut PyExecutionContext,
    ) -> Result<usize, pyre_interpreter::PyError> {
        // interp_jit.py:103 — `jumpto *= 2`. RPython encodes PCs in
        // 16-bit code-words; pyre's `JumpBackward` opcode arg is
        // already the absolute byte offset, so the `*= 2` scaling
        // does not apply.  Kept as a comment marker so the line-by-
        // line correspondence stays explicit.
        // interp_jit.py:104-105 — `if jumpto >= next_instr: return jumpto`.
        if jumpto >= next_instr {
            return Ok(jumpto);
        }
        if majit_metainterp::we_are_jitted() {
            // interp_jit.py:108-112 — has_bytecode_counter +
            // gil_ready quasi-immutable gate.  Pyre's actionflag
            // does not carry a constant-folded `has_bytecode_counter`
            // flag yet, so use the adapted tick directly.  When the
            // actionflag port lands the gate flips back on.
            let decr_by = _get_adapted_tick_counter();
            // interp_jit.py:114 — `self.last_instr = intmask(jumpto)`.
            frame.set_last_instr_from_next_instr(jumpto);
            if !ec.is_null() {
                // interp_jit.py:115 — `ec.bytecode_trace(self, decr_by)`.
                // executioncontext.py:392-395 re-raises callback
                // exceptions; propagate via `?`.
                unsafe { (*ec).bytecode_trace(frame as *mut PyFrame, decr_by) }?;
            }
            // interp_jit.py:116 — `jumpto = r_uint(self.last_instr)`.
            jumpto = frame.next_instr();
        }
        // interp_jit.py:118-120 — `pypyjitdriver.can_enter_jit(...)`.
        // Not invoked here: this function is a documentation-only
        // line-by-line port of PyPy `interp_jit.py:102-121` kept for
        // parity audit (no Rust caller exists yet).  Pyre's live
        // can_enter_jit dispatch happens out-of-band at
        // `eval_loop_jit`'s `StepResult::CloseLoop` →
        // `maybe_compile_and_run`, which fires for every backward
        // jump independently of this shim.
        Ok(jumpto)
    }
}

/// interp_jit.py:119-131 — _get_adapted_tick_counter().
///
/// Normally the tick counter is decremented by 100 for every Python opcode.
/// Here, to better support JIT compilation of small loops, we decrement it
/// by a possibly smaller constant.  We get the maximum 100 when the
/// (unoptimized) trace length is at least 3200 (a bit randomly).
#[inline]
fn _get_adapted_tick_counter() -> usize {
    let (driver, _) = driver_pair();
    let trace_length = driver.current_trace_length();
    // current_trace_length() returns -1 when not tracing
    let decr_by = if trace_length < 0 {
        100 // also if current_trace_length() returned -1
    } else {
        (trace_length as usize) / 32
    };
    decr_by.clamp(1, 100)
}

#[derive(Clone, Copy)]
pub struct W_NotFromAssembler {
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
}

impl W_NotFromAssembler {
    pub fn __init__(
        &mut self,
        space: pyre_object::PyObjectRef,
        w_callable: pyre_object::PyObjectRef,
    ) {
        self.space = space;
        self.w_callable = w_callable;
    }

    pub fn descr_call(&self, __args__: &[pyre_object::PyObjectRef]) -> Self {
        _call_not_in_trace(self.space, self.w_callable, __args__);
        *self
    }
}

pub fn not_from_assembler_new(
    space: pyre_object::PyObjectRef,
    _w_subtype: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
) -> W_NotFromAssembler {
    let _ = _w_subtype;
    W_NotFromAssembler { space, w_callable }
}

#[allow(unused_variables)]
pub fn _call_not_in_trace(
    space: pyre_object::PyObjectRef,
    w_callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) {
    let _ = space;
    let _ = pyre_interpreter::baseobjspace::call_function(w_callable, args);
}

#[inline]
fn green_key_from_pycode(next_instr: usize, w_pycode: pyre_object::PyObjectRef) -> Option<u64> {
    // Safety: this follows existing wrappers that treat `PyCode`
    // as an owned pointer to a `CodeObject`.
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if code_ptr.is_null() {
        return None;
    }
    Some(make_green_key(code_ptr, next_instr))
}

/// RPython interp_jit.py helper: get_printable_location.
pub fn get_printable_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> String {
    let mut opcode = "<eof>".to_string();
    let mut code_name = "<unknown>".to_string();
    let code_ptr = unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) };
    if !code_ptr.is_null() {
        let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
        code_name = code.obj_name.to_string();
        if let Some((instr, _)) = pyre_interpreter::decode_instruction_at(code, next_instr) {
            opcode = format!("{:?}", instr);
        }
    }
    format!("{code_name} #{next_instr} {opcode}")
}

/// RPython interp_jit.py helper: get_unique_id.
pub fn get_unique_id(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> usize {
    // A stable process-local unique-id equivalent using the code pointer.
    unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) as usize }
}

/// RPython interp_jit.py helper: get_location.
pub fn get_location(
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let (filename, line, name, opcode) =
        match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
            x if x.is_null() => (
                "<unknown>".to_string(),
                0,
                "<unknown>".to_string(),
                "<eof>".to_string(),
            ),
            code_ptr => {
                let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
                let (_opcode, opname) =
                    match pyre_interpreter::decode_instruction_at(code, next_instr) {
                        Some((instruction, _)) => {
                            (format!("{instruction:?}"), format!("{:?}", instruction))
                        }
                        None => ("<eof>".to_string(), "<eof>".to_string()),
                    };
                let line = code
                    .locations
                    .get(next_instr)
                    .and_then(|(start, _)| Some(start.line.get() as usize))
                    .unwrap_or_else(|| {
                        code.first_line_number
                            .map(|line| line.get())
                            .unwrap_or(0)
                            .saturating_add(next_instr)
                    });
                (
                    code.source_path.to_string(),
                    line,
                    code.obj_name.to_string(),
                    opname,
                )
            }
        };
    let _ = opcode;
    w_tuple_new(vec![
        w_str_new(&filename),
        w_int_new(line as i64),
        w_str_new(&name),
        w_int_new(next_instr as i64),
        w_str_new(&opcode),
    ])
}

/// RPython interp_jit.py helper: should_unroll_one_iteration.
pub fn should_unroll_one_iteration(
    _next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> bool {
    match unsafe { pyre_interpreter::pycode::w_code_get_ptr(w_pycode) } {
        ptr if ptr.is_null() => false,
        code_ptr => {
            let code = unsafe { &*code_ptr.cast::<pyre_interpreter::CodeObject>() };
            code.flags.contains(pyre_interpreter::CodeFlags::GENERATOR)
        }
    }
}

/// interp_jit.py:216 — get_jitcell_at_key.
///
/// Returns True if a jitcell exists for this green key, regardless of
/// whether machine code has been compiled. A cell is created when the
/// counter first ticks, so this returns True even before compilation.
/// interp_jit.py:215 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn get_jitcell_at_key(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    let key = green_key_from_pycode(next_instr, w_pycode);
    let (driver, _) = driver_pair();
    w_bool_from(key.is_some_and(|green_key| {
        driver
            .meta_interp_mut()
            .warm_state_mut()
            .get_cell(green_key)
            .is_some()
    }))
}

/// interp_jit.py:222 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn dont_trace_here(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .disable_noninlinable_function(green_key);
}

/// interp_jit.py:233 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn mark_as_being_traced(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .mark_as_being_traced(green_key);
}

/// interp_jit.py:245 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration(
    _space: pyre_object::PyObjectRef,
    next_instr: usize,
    _is_being_profiled: bool,
    w_pycode: pyre_object::PyObjectRef,
) {
    let Some(green_key) = green_key_from_pycode(next_instr, w_pycode) else {
        return;
    };
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key);
}

/// interp_jit.py:253 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn trace_next_iteration_hash(_space: pyre_object::PyObjectRef, green_key_hash: usize) {
    let _ = _space;
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .trace_next_iteration(green_key_hash as u64);
}

/// interp_jit.py:169 — `@dont_look_inside`
#[majit_macros::dont_look_inside]
pub fn residual_call(
    _space: pyre_object::PyObjectRef,
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    let _ = _space;
    pyre_interpreter::baseobjspace::call_function(callable, args)
}

/// rlib/jit.py:842-862 `set_user_param` — apply a JIT-parameter string
/// (`"name=value,…"`, `"off"`, or `"default"`) to the warmstate. Shared by
/// the Python-level `set_param` positional-string branch and the `PYRE_JIT`
/// env lever (the translation-time `--jit STR` option's analog) so both
/// parse identically. `Err(())` signals a malformed string (rlib/jit.py:853
/// ValueError).
fn apply_jit_param_string(
    ws: &mut majit_metainterp::warmstate::WarmEnterState,
    text: &str,
) -> Result<(), ()> {
    // rlib/jit.py:842-845
    if text == "off" {
        ws.set_param("threshold", -1);
        ws.set_param("function_threshold", -1);
    } else if text == "default" {
        ws.set_default_params();
    } else {
        // rlib/jit.py:850-862 — "name=value,name=value"
        for s in text.split(',') {
            let s = s.trim();
            if s.is_empty() {
                continue;
            }
            // rlib/jit.py:853 — len(parts) != 2 → raise ValueError
            let Some((name, value)) = s.split_once('=') else {
                return Err(());
            };
            let value = value.trim();
            if name == "enable_opts" {
                ws.set_param_enable_opts(value);
            } else if let Ok(parsed) = value.parse::<i64>() {
                ws.set_param(name, parsed);
            } else {
                return Err(());
            }
        }
    }
    Ok(())
}

/// interp_jit.py:138-167 — set_param(space, __args__).
///
/// Configure the tunable JIT parameters.
///   * set_param(name=value, ...)            # as keyword arguments
///   * set_param("name=value,name=value")    # as a user-supplied string
///   * set_param("off")                      # disable the jit
///   * set_param("default")                  # restore all defaults
pub fn set_param(
    _space: pyre_object::PyObjectRef,
    __args__: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let _ = _space;
    let (driver, _) = driver_pair();

    // Separate positional args from kwargs dict (last arg with __pyre_kw__ marker).
    let (pos_args, kwds) = split_kwargs(__args__);

    // interp_jit.py:147-148
    if pos_args.len() > 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos_args.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → jit.set_user_param(None, text)
    if pos_args.len() == 1 {
        // `space.text_w` rejects a non-str positional with TypeError.
        let text = pyre_interpreter::baseobjspace::text_w(pos_args[0])?;
        // rlib/jit.py:842-862 set_user_param.
        let ws = driver.meta_interp_mut().warm_state_mut();
        if apply_jit_param_string(ws, &text).is_err() {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    // interp_jit.py:157-167 — keyword arguments.  Routed through
    // strategy-dispatched `w_dict_items` (dictmultiobject.py:308 items)
    // rather than reaching past the strategy slot into `dstorage` —
    // the raw cast would tear once a non-Object strategy backs `kwds`.
    if let Some(kw_dict) = kwds {
        let ws = driver.meta_interp_mut().warm_state_mut();
        let items = unsafe { pyre_object::dictmultiobject::w_dict_items(kw_dict) };
        for (k, v) in items {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_value(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            // interp_jit.py:158-159 — `space.text_w` rejects a non-str value.
            if key == "enable_opts" {
                ws.set_param_enable_opts(pyre_interpreter::baseobjspace::text_w(v)?);
                continue;
            }
            // interp_jit.py:160-167 — `intval = space.int_w(w_value)` is computed
            // (rejecting a non-int value with TypeError) before the parameter
            // name is validated.
            let intval = pyre_interpreter::baseobjspace::int_w(v)?;
            if !is_known_jit_param(key) {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "no JIT parameter '{key}'"
                )));
            }
            ws.set_param(key, intval);
        }
    }

    Ok(w_none())
}

/// rlib/jit.py:588-605 PARAMETERS — valid parameter names.
fn is_known_jit_param(name: &str) -> bool {
    matches!(
        name,
        "threshold"
            | "function_threshold"
            | "trace_eagerness"
            | "decay"
            | "trace_limit"
            | "inlining"
            | "loop_longevity"
            | "retrace_limit"
            | "pureop_historylength"
            | "max_retrace_guards"
            | "max_unroll_loops"
            | "disable_unrolling"
            | "enable_opts"
            | "max_unroll_recursion"
            | "vec"
            | "vec_all"
            | "vec_cost"
    )
}

/// Split args into (positional, optional kwargs dict).
fn split_kwargs(
    args: &[pyre_object::PyObjectRef],
) -> (
    &[pyre_object::PyObjectRef],
    Option<pyre_object::PyObjectRef>,
) {
    if let Some(&last) = args.last() {
        if !last.is_null()
            && unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__"))
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
            }
        {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// interp_jit.py:258 — `@dont_look_inside`
///
/// Mark all current machine code objects as ready to release.
/// They will be released at the next GC (unless in use on a thread stack).
#[majit_macros::dont_look_inside]
pub fn releaseall(_space: pyre_object::PyObjectRef) {
    let _ = _space;
    let (driver, _) = driver_pair();
    // memmgr.py:85 release_all_loops parity.
    driver.mark_all_loops_for_release();
}

fn init_callbacks() {
    use pyre_jit_trace::callbacks::{self, CallJitCallbacks};
    thread_local! {
        static INIT: Cell<bool> = const { Cell::new(false) };
    }
    INIT.with(|c| {
        if !c.get() {
            c.set(true);
            let cb = Box::leak(Box::new(CallJitCallbacks {
                callee_frame_helper: crate::call_jit::callee_frame_helper,
                recursive_force_cache_safe: crate::call_jit::recursive_force_cache_safe,
                jit_drop_callee_frame: crate::call_jit::jit_drop_callee_frame as *const (),
                jit_frame_set_slot_ref: crate::call_jit::jit_frame_set_slot_ref as *const (),
                jit_frame_set_slot_int: crate::call_jit::jit_frame_set_slot_int as *const (),
                jit_frame_set_slot_float: crate::call_jit::jit_frame_set_slot_float as *const (),
                jit_force_callee_frame: crate::call_jit::jit_force_callee_frame as *const (),
                jit_force_recursive_call_1: crate::call_jit::jit_force_recursive_call_1
                    as *const (),
                jit_force_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_recursive_call_argraw_boxed_1 as *const (),
                jit_force_self_recursive_call_argraw_boxed_1:
                    crate::call_jit::jit_force_self_recursive_call_argraw_boxed_1 as *const (),
                jit_create_callee_frame_1: crate::call_jit::jit_create_callee_frame_1 as *const (),
                jit_create_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_callee_frame_1_raw_int as *const (),
                jit_create_self_recursive_callee_frame_1:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1 as *const (),
                jit_create_self_recursive_callee_frame_1_raw_int:
                    crate::call_jit::jit_create_self_recursive_callee_frame_1_raw_int as *const (),
                driver_pair: || driver_pair() as *mut JitDriverPair as *mut u8,
                ensure_majit_jitcode: |code, w_code| {
                    if !code.is_null() {
                        let _ =
                            crate::jit::codewriter::ensure_trace_jitcode_for_w_code(code, w_code);
                    }
                },
                drain_backend_jit_exc: crate::call_jit::drain_backend_jit_exc,
            }));
            callbacks::init(cb);
        }
    });
}

// JIT_TRACING_DEPTH removed — now MetaInterp.tracing_call_depth field.
// RPython portal_call_depth parity: state colocated with tracing context.

/// Read the call depth from pyre-interpreter's CALL_DEPTH TLS.
/// Replaces the separate JIT_CALL_DEPTH — single source of truth.
// dont_look_inside: reads CALL_DEPTH TLS; no registry-resolvable accessor.
#[majit_macros::dont_look_inside]
pub(crate) fn call_depth() -> u32 {
    pyre_interpreter::call::call_depth()
}

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
// dont_look_inside: green-key construction (pypyjit_greenkey_uhash); JIT-driver machinery.
#[majit_macros::dont_look_inside]
pub fn make_green_key(code_ptr: *const (), pc: usize) -> u64 {
    // Full `JitCell.get_uhash` over the pypyjit green tuple
    // `[next_instr, is_being_profiled, pycode]` (warmstate.py:584-593),
    // computed allocation-free. `is_being_profiled` folds to 0 (the JIT
    // path is never profiled), so this matches the typed marker-path key
    // and both lookups resolve to the same cell.
    majit_ir::pypyjit_greenkey_uhash(pc, false, code_ptr as u64)
}

// JIT_CALL_DEPTH removed — pyre-interpreter::call::CALL_DEPTH is the single
// source of truth. call_depth() reads it. No more Box<dyn Any> allocation.

/// RPython compile.py:204-207 (record_loop_or_bridge) parity:
/// Register the compiled loop's invalidation flag with all quasi-immutable
/// dependencies collected during optimization. The optimizer records
/// namespace pointers in quasi_immutable_deps when processing
/// QUASIIMMUT_FIELD ops. After compilation, this function reads them
/// from MetaInterp and registers watchers so GUARD_NOT_INVALIDATED
/// fails when the namespace mutates.
fn register_quasi_immutable_deps(green_key: u64) {
    let (driver, _) = driver_pair();
    let deps: Vec<(u64, u32)> =
        std::mem::take(&mut driver.meta_interp_mut().last_quasi_immutable_deps);
    if deps.is_empty() {
        return;
    }
    let Some(token) = driver.get_loop_token(green_key) else {
        return;
    };
    let flag = token.invalidation_flag();
    // `celldict.py:34 _immutable_fields_ = ["version?"]`: the global cell
    // fast path's `QUASIIMMUT_FIELD(ns, slot)` is keyed on the module
    // dict's `ModuleDictStrategy.version`, not a per-slot index, so every
    // recorded dep registers the loop flag against that single version
    // watcher.  `mutated()` (new key, `del`, `switch_to_object_strategy`)
    // then flips the flag; a same-key value reassign mutates the cell in
    // place without bumping the version and is observed by the live
    // `cell.w_value` read instead.  `ns_ptr` is the `const_ref`-folded
    // `w_globals` object pointer; `slot` is unused for version keying.
    for (ns_ptr, _slot) in deps {
        let obj = ns_ptr as pyre_object::PyObjectRef;
        unsafe {
            pyre_object::dictmultiobject::module_dict_register_version_watcher(obj, &flag);
        }
    }
}

/// rpython/rlib/rstack.py:75-90 `stack_almost_full` parity — delegates
/// to [`pyre_interpreter::stack_check::stack_almost_full`], which reads
/// the shared [`PYRE_STACKTOOBIG`](pyre_interpreter::stack_check::
/// PYRE_STACKTOOBIG) budget maintained by `sys.setrecursionlimit`. Kept
/// as a thin wrapper so existing call sites in this module stay short.
#[inline]
fn stack_almost_full() -> bool {
    pyre_interpreter::stack_check::stack_almost_full()
}

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-jit.
pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    eval_with_jit_inner(frame)
}

/// Hook target for `pyre_interpreter::call::set_jit_param`. Routes
/// `executioncontext.py:296-298 jit.set_param(None, name, value)` calls
/// from `ExecutionContext::settrace` into the live `WarmState`.
fn set_jit_param_via_warmstate(name: &str, value: i64) {
    let (driver, _) = driver_pair();
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .set_param(name, value);
}

/// Eagerly register pyre-jit's hooks into pyre-interpreter so callers
/// like `sys.settrace` see the JIT side from the very first user call,
/// not only after the first JIT-eligible eval.  Idempotent (the
/// `OnceLock::set` semantics inside the registrars discard repeats).
///
/// `register_eval_override` and `register_set_jit_param_hook` are also
/// invoked from `eval_with_jit_inner` as a lazy safety net — pyrex
/// calls this once at boot so user code that touches `sys.settrace`
/// before its first JIT-traced bytecode still routes through to the
/// real `WarmState::set_param("trace_limit", 10000)`.
pub fn init_jit_hooks() {
    // Phase A: build the GC and install it into the backend + pyre-object
    // hooks.  Safe at boot — no interpreter state referenced.  This makes
    // frames GC-owned even under PYRE_JIT=0 (#383).
    init_gc_subsystem();
    pyre_interpreter::call::register_eval_override(eval_with_jit);
    pyre_interpreter::call::register_set_jit_param_hook(set_jit_param_via_warmstate);
    // Install the dict key `eq_w` / `hash_w` / `compares_by_identity`
    // trampolines here, at boot, before any user statement runs. They are
    // also registered inside the `JIT_DRIVER` initializer for the
    // standalone/test path that touches `driver_pair()` without going
    // through `main_entry`; doing it here too makes them live before the
    // first `{}` literal is filled. Otherwise a str-keyed dict built at
    // module level hashes its keys through `object_key_for`'s structural
    // fallback (dictmultiobject.py:95-101), and once the real hook installs
    // on the first JIT entry every later lookup recomputes the siphash and
    // misses its bucket. The trampolines only call interpreter-side
    // `eq_w`/`try_hash_value`/`compares_by_identity`, so they need neither
    // the GC allocator nor the JIT driver — safe to install this early.
    pyre_object::dict_eq_hook::register_eq_w_hook(pyre_object_eq_w_trampoline);
    pyre_object::dict_eq_hook::register_hash_w_hook(pyre_object_hash_w_trampoline);
    pyre_object::dict_eq_hook::register_hash_str_hook(pyre_object_hash_str_trampoline);
    pyre_object::dict_eq_hook::register_compares_by_identity_hook(
        pyre_object_compares_by_identity_trampoline,
    );
}

thread_local! {
    static JIT_SUPPRESSED_BY_UNSUPPORTED_FRAME: Cell<usize> = const { Cell::new(0) };
}

struct JitSuppressionGuard;

impl JitSuppressionGuard {
    fn new() -> Self {
        JIT_SUPPRESSED_BY_UNSUPPORTED_FRAME.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for JitSuppressionGuard {
    fn drop(&mut self) {
        JIT_SUPPRESSED_BY_UNSUPPORTED_FRAME.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

// dont_look_inside: reads JIT_SUPPRESSED_BY_UNSUPPORTED_FRAME TLS.
#[majit_macros::dont_look_inside]
fn jit_suppressed_by_unsupported_frame() -> bool {
    JIT_SUPPRESSED_BY_UNSUPPORTED_FRAME.with(|depth| depth.get() != 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnsupportedJitShape {
    None,
    CurrentFrameOnly,
    StructuralRegion,
    /// The frame's constant pool plus register file would exceed the
    /// single-byte (`< 256`) register-or-constant index encoding
    /// (`assembler.py:72 chr()`, `assembler.py:132-133`
    /// `count_regs + len(constants) < 256`). RPython builds jitcodes at
    /// translation time from the hand-written interpreter, where this
    /// ceiling is a bounded programming invariant; pyre builds a jitcode
    /// per *user* Python frame at runtime, so a data-heavy frame (e.g. a
    /// module body with thousands of literal constants) genuinely exceeds
    /// it. Such a frame cannot be encoded and must run in the interpreter.
    ConstEncodingOverflow,
}

/// True for opcodes that may appear in a `FOR_ITER` loop body without ever
/// reaching the orthodox-sub-walk `list.append`/`STORE_SUBSCR` path whose
/// walk-abort silently drops an iteration (#57). This is an ALLOW-LIST:
/// arithmetic/comparison (implicit dunder dispatch resumes past the call on
/// abort — verified), local/const reads and frame-slot writes, stack
/// manipulation, read-only attribute loads, and intra-body control flow. Every
/// other opcode — heap-mutating stores outside the direct-store widening and
/// unsupported mutators — is treated as unsafe so the frame keeps running in
/// the interpreter. Unknown/future opcodes default to unsafe.
fn for_iter_body_op_is_jit_safe(instr: pyre_interpreter::Instruction) -> bool {
    use pyre_interpreter::Instruction as I;
    matches!(
        instr,
        // local / const: frame slots and constants, no heap mutation
        I::LoadFast { .. }
            | I::LoadFastBorrow { .. }
            | I::LoadFastLoadFast { .. }
            | I::LoadFastBorrowLoadFastBorrow { .. }
            | I::LoadFastCheck { .. }
            | I::LoadFastAndClear { .. }
            | I::StoreFast { .. }
            | I::StoreFastLoadFast { .. }
            | I::StoreFastStoreFast { .. }
            | I::LoadConst { .. }
            | I::LoadSmallInt { .. }
            | I::LoadCommonConstant { .. }
            // arithmetic / comparison: implicit dunder dispatch recovers on abort
            | I::BinaryOp { .. }
            | I::CompareOp { .. }
            | I::IsOp { .. }
            | I::UnaryNegative
            | I::UnaryNot
            | I::UnaryInvert
            | I::ToBool
            // stack manipulation
            | I::Copy { .. }
            | I::Swap { .. }
            | I::PopTop
            | I::PushNull
            | I::Nop
            | I::NotTaken
            // intra-body control flow
            | I::PopJumpIfFalse { .. }
            | I::PopJumpIfTrue { .. }
            | I::PopJumpIfNone { .. }
            | I::PopJumpIfNotNone { .. }
            | I::JumpForward { .. }
            | I::JumpBackward { .. }
            | I::JumpBackwardNoInterrupt { .. }
            // nested FOR_ITER: the inner loop's iterator setup and iteration
            | I::GetIter
            | I::ForIter { .. }
            | I::EndFor
            // sequence unpacking / tuple/slice build: stack-only operations
            // that produce immutable objects, no heap mutation
            | I::UnpackSequence { .. }
            | I::UnpackEx { .. }
            | I::BuildTuple { .. }
            | I::BuildSlice { .. }
            // read-only subscript/membership: lowered to residual calls,
            // no heap mutation
            | I::BinarySlice
            | I::ContainsOp { .. }
            // string formatting: produces immutable strings
            | I::FormatSimple
            | I::ConvertValue { .. }
            | I::BuildString { .. }
            // misc read-only: len(), iterator cleanup, local delete,
            // closure variable read
            | I::GetLen
            | I::PopIter
            | I::DeleteFast { .. }
            | I::LoadDeref { .. }
            // An aborting method call resumes exactly via forward-exc-delivery / CALL-forward.
            | I::LoadAttr { .. }
            // function calls and global reads: the Layer 2 dynamic defense
            // (body_effect_candidate + fbw_foriter_inflight_take) handles
            // walk-abort safety, and inline sub-walks are declined when a
            // FOR_ITER item is in-flight (try_walker_inline_user_call).
            | I::Call { .. }
            | I::CallKw { .. }
            | I::LoadGlobal { .. }
            | I::Resume { .. }
            // container builders: produce new heap objects but do not mutate
            // existing ones; walk-abort just drops the incomplete object
            | I::BuildList { .. }
            | I::BuildSet { .. }
            | I::BuildMap { .. }
            // oparg prefix + inline-cache padding (no-ops in the body scan)
            | I::ExtendedArg
            | I::Cache
    )
}

/// True iff every `FOR_ITER` loop body in `code` is admissible. The BASE rule is
/// `for_iter_body_op_is_jit_safe`. A nested `FOR_ITER` appears as a body
/// instruction of its enclosing loop, and its own body is scanned when the
/// outer instruction walk reaches it. The iterable setup (`range(n)`,
/// `GET_ITER`) precedes the `FOR_ITER` and is therefore not part of any body
/// range.
///
/// This whole gate is a conservative adaptation, not an upstream mechanism.
/// Mid-body walk aborts now resume exactly through `try_commit_midbody_abort`:
/// forward-exception-delivery handles propagated exceptions and CALL-forward
/// re-runs the outer call, instead of using the FBW refuse-drop path. This
/// matches the forward-only resume of `blackhole_from_resumedata`
/// (resume.py:1312), which never drops or doubles an iteration.
///
/// The direct heap-mutation opcodes `STORE_SUBSCR`, `STORE_ATTR`, `STORE_NAME`,
/// `STORE_GLOBAL`, `STORE_DEREF`, `DELETE_SUBSCR`, `DELETE_ATTR`, and the
/// `LOAD_NAME` that reads module globals are admitted in any body, including one
/// with a call, branch or nested loop. A mid-body abort after a committed
/// un-journaled store, cell write, or delete now resumes exactly through
/// `try_commit_midbody_abort`:
/// forward-exception-delivery or CALL-forward replaces the FBW refuse-drop that
/// skipped the iteration tail. The mutation therefore commits exactly once and
/// the tail is never dropped, matching the forward-only resume of
/// `blackhole_from_resumedata`. STORE_DEREF's cell slot and every subsequent
/// operand-stack slot are reconstructed independently, including both method
/// slots consumed by a tail CALL. A mutation that raises, or a later exception
/// after the cell write, propagates through forward-exception-delivery; Fix A's
/// exit-frame traceback recording preserves its traceback exactly. `LOAD_ATTR`
/// is admitted because a mid-body abort from its method call follows the same
/// exact-resume path rather than dropping the remainder of the iteration.
fn for_iter_bodies_all_jit_safe(code: &pyre_interpreter::CodeObject) -> bool {
    use pyre_interpreter::Instruction as I;
    let instructions = &code.instructions;
    let mut arg_state = pyre_interpreter::OpArgState::default();
    for (pc, unit) in instructions.iter().copied().enumerate() {
        let (instr, op_arg) = arg_state.get(unit);
        if let pyre_interpreter::Instruction::ForIter { delta } = instr {
            let exit = pyre_interpreter::jump_target_forward(
                instructions,
                pc + 1,
                delta.get(op_arg).as_usize(),
            );
            let mut body_state = pyre_interpreter::OpArgState::default();
            let mut body_pc = pc + 1;
            while body_pc < exit && body_pc < instructions.len() {
                let (body_instr, _) = body_state.get(instructions[body_pc]);
                let permitted = for_iter_body_op_is_jit_safe(body_instr)
                    || matches!(
                        body_instr,
                        I::StoreSubscr
                            | I::StoreAttr { .. }
                            | I::StoreName { .. }
                            | I::StoreGlobal { .. }
                            | I::StoreDeref { .. }
                            | I::DeleteSubscr
                            | I::DeleteAttr { .. }
                            | I::LoadName { .. }
                    );
                if !permitted {
                    return false;
                }
                body_pc += 1;
            }
        }
    }
    true
}

/// True when `code` holds more than one `FOR_ITER` and at least one of them
/// sits inside an exception-table-covered range — the signature of a loop
/// DUPLICATED into a `finally` block's normal and exceptional copies (3.14
/// emits the `finally` body twice). The JIT compiles only the normal copy; on
/// loop exhaustion the side-exit resumes the live frame through the dense
/// carry-forward `pc_map`, which collapses the un-traced exhaustion-exit region
/// and the un-traced exceptional copy into a single marker. The resume then
/// lands at the exceptional-copy `FOR_ITER` with an empty value stack ("stack
/// underflow during interpreter peek"). The correct resume coordinate was never
/// emitted into the trace, so no inverse-map rule can recover it; the frame must
/// run in the interpreter (#57).
///
/// A single `FOR_ITER` inside a `try`/`except` (no duplication) keeps JITting
/// because the count stays at one, and genuinely nested `FOR_ITER` frames are
/// already declined by [`for_iter_bodies_all_jit_safe`].
fn for_iter_frame_is_finally_duplicated(code: &pyre_interpreter::CodeObject) -> bool {
    let mut arg_state = pyre_interpreter::OpArgState::default();
    let mut for_iter_count = 0usize;
    let mut any_in_handler = false;
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        if let pyre_interpreter::Instruction::ForIter { .. } = arg_state.get(unit).0 {
            for_iter_count += 1;
            // The exception table is keyed by byte offset; pyre's `pc` is the
            // instruction-unit index (two bytes per unit).
            if pyre_interpreter::pycode::lookup_exceptiontable(
                &code.exceptiontable,
                (pc * 2) as u32,
            )
            .is_some()
            {
                any_in_handler = true;
            }
        }
    }
    for_iter_count > 1 && any_in_handler
}

/// Upper bound on the constant-pool slots one code-object constant can
/// contribute to the assembled jitcode. A `Tuple`/`Frozenset`/`Slice`
/// constant can be unpacked into its elements during graph construction —
/// e.g. a `BUILD_CONST_KEY_MAP` keys tuple lowered to one `ConstRef` per key,
/// which is how a module body like `html.entities` turns a handful of literal
/// `code.constants` entries into thousands of pool slots — so every nested
/// leaf is counted. A `Code` constant belongs to a *separate* callee frame
/// with its own pool, so it counts as a single ref here (its inner constants
/// never enter this frame's pool).
fn const_pool_slot_upper_bound(c: &pyre_interpreter::ConstantData) -> usize {
    use pyre_interpreter::ConstantData;
    match c {
        ConstantData::Tuple { elements } | ConstantData::Frozenset { elements } => {
            1 + elements
                .iter()
                .map(const_pool_slot_upper_bound)
                .sum::<usize>()
        }
        ConstantData::Slice { elements } => {
            1 + elements
                .iter()
                .map(const_pool_slot_upper_bound)
                .sum::<usize>()
        }
        _ => 1,
    }
}

fn unsupported_jit_shape(code: &pyre_interpreter::CodeObject) -> UnsupportedJitShape {
    // Structural adaptation: RPython/PyPy traces these bytecodes with
    // fully translated support. Pyre's codewriter still lowers
    // `WITH_EXCEPT_START` through a pyre-local `abort_permanent`
    // path. A frame containing this unsupported shape must run in the
    // interpreter. While that frame is active, nested helper calls are
    // also kept out of the JIT by `JitSuppressionGuard`; this mirrors
    // the structural unsupported region instead of keying on a
    // benchmark filename.
    //
    // The gate cannot be dropped yet: removing it lets a `with` frame trace,
    // which miscompiles (a raw SIGSEGV in the exception-link path plus
    // guard-failure storms — test_strftime/test_shlex/test_textwrap, #389).
    // The eventual fix is a real `WITH_EXCEPT_START` lowering; until then the
    // decline is recorded in the census (see the `StructuralRegion` arm) so it
    // is not a silent no-token gap.
    //
    // `FOR_ITER` is narrower: a FOR_ITER frame may enter the JIT only when
    // every loop body contains exclusively allow-listed opcodes
    // (`for_iter_bodies_all_jit_safe`). The exclusion exists because a FBW walk
    // that aborts mid-loop while an inlined sub-walk has performed a direct
    // `list.append`/`STORE_SUBSCR` cannot deliver or rewind that effect, so the
    // iteration is silently dropped (#57). Bodies with no explicit
    // mutation/call and no nested `FOR_ITER` cannot reach that path — verified
    // against the battery and adversarial mutation probes.

    // Single-byte register-or-constant index ceiling (`assembler.py:72`
    // `chr(reg)`; `assembler.py:132-133` / `check_result` assert
    // `count_regs[kind] + len(constants) <= 256`). RPython assembles jitcodes
    // at translation time from the hand-written interpreter, where the ceiling
    // is a bounded invariant; pyre assembles a jitcode per *user* Python frame
    // at runtime, so a data-heavy frame (a module body with thousands of
    // literal constants — e.g. `html.entities`, whose `<module>` frame carries
    // ~4000 string constants) genuinely overruns it and the assembler asserts
    // mid-emission. Decline such a frame to the interpreter before tracing.
    //
    // The predicate over-approximates the assembled counts, so it never admits
    // an unencodable frame: the per-kind constant pool is bounded above by the
    // whole constant table (flattened through nestable constants, see
    // `const_pool_slot_upper_bound`) plus the name table, and each kind's
    // register file is bounded above by the frame's live-slot capacity — the
    // operand stack plus every local/cell/free slot, padded for the always-live
    // vable red args and the transient temporaries one opcode lowering keeps
    // live at once. When even this loose sum cannot fit in a byte, no assembler
    // kind bank can encode the frame.
    const SYNTHESIZED_REG_HEADROOM: usize = 16;
    let register_slots_upper_bound = SYNTHESIZED_REG_HEADROOM
        + code.max_stackdepth as usize
        + code.varnames.len()
        + code.cellvars.len()
        + code.freevars.len();
    let constant_slots_upper_bound = code
        .constants
        .iter()
        .map(const_pool_slot_upper_bound)
        .sum::<usize>()
        + code.names.len();
    if register_slots_upper_bound + constant_slots_upper_bound > 256 {
        return UnsupportedJitShape::ConstEncodingOverflow;
    }

    let mut arg_state = pyre_interpreter::OpArgState::default();
    let mut has_for_iter = false;
    for unit in code.instructions.iter().copied() {
        if let pyre_interpreter::Instruction::ForIter { .. } = arg_state.get(unit).0 {
            has_for_iter = true;
        }
    }
    if has_for_iter {
        // A `finally`-duplicated loop stays interpreted: its exhaustion
        // side-exit resumes through the lossy carry-forward `pc_map` and lands
        // at the exceptional copy with an empty stack (see
        // `for_iter_frame_is_finally_duplicated`).
        if !for_iter_bodies_all_jit_safe(code) || for_iter_frame_is_finally_duplicated(code) {
            return UnsupportedJitShape::CurrentFrameOnly;
        }
    }
    UnsupportedJitShape::None
}

fn eval_with_jit_inner(frame: &mut PyFrame) -> PyResult {
    // Phase B of GC init: register root walkers that reference
    // interpreter state.  Safe here — the interpreter is initialized.
    // Phase A (GC build + backend install) ran at boot in init_jit_hooks.
    init_gc_root_walkers();
    // PYRE_JIT=0 disables JIT entirely, falling back to plain interpreter.
    static PYRE_JIT_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *PYRE_JIT_DISABLED.get_or_init(|| std::env::var("PYRE_JIT").as_deref() == Ok("0")) {
        return frame.execute_frame(None, None);
    }
    if jit_suppressed_by_unsupported_frame() {
        return frame.execute_frame(None, None);
    }
    let mut frame_root = FrameRoot::new(frame);
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    pyre_interpreter::call::register_eval_override(eval_with_jit);
    pyre_interpreter::call::register_set_jit_param_hook(set_jit_param_via_warmstate);
    // The backend-agnostic registrations here — notably the JIT exception
    // raiser (`register_jit_exc_raiser`) that `jit_publish_exception` routes
    // residual-call raises through — are required on every backend; the
    // cranelift/dynasm-specific blocks inside are already `cfg`-gated, so this
    // is safe on wasm32 (where it is the only thing that installs the raiser).
    crate::call_jit::install_jit_call_bridge();
    init_callbacks();
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::register_resumedata_deopt(crate::call_jit::cranelift_resumedata_deopt);
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::register_recovery_layout(
        crate::call_jit::cranelift_recovery_layout_for_descr,
    );
    match unsupported_jit_shape(code) {
        UnsupportedJitShape::None => {}
        UnsupportedJitShape::CurrentFrameOnly => {
            // A FOR_ITER frame the #57 gate cannot trace (a non-journalable
            // mutator body, or a `finally`-duplicated loop): run it in the plain
            // interpreter.  The tracer never sees it, so record the frame-shape
            // decline in the census (deduped per code object) — otherwise a hot
            // loop declined here reads as a silent no-token gap, indistinguishable
            // from a loop the JIT never noticed.
            pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
                code as *const _ as usize,
                "FrameShape::CurrentFrameOnly",
            );
            return frame_root.frame().execute_frame(None, None);
        }
        UnsupportedJitShape::StructuralRegion => {
            // A `with` frame whose `WITH_EXCEPT_START` exception-link lowering
            // the codewriter still residualizes: tracing it (or its callees)
            // miscompiles — a raw SIGSEGV in the exception path and guard-failure
            // storms (#389 gate-regression evidence).  Keep the frame AND its
            // nested helper frames interpreted via `JitSuppressionGuard`, and
            // record the census entry so the decline is visible, not a silent
            // no-token gap.
            pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
                code as *const _ as usize,
                "FrameShape::StructuralRegion",
            );
            let _guard = JitSuppressionGuard::new();
            return frame_root.frame().execute_frame(None, None);
        }
        UnsupportedJitShape::ConstEncodingOverflow => {
            // The frame's constant pool plus register file overruns the
            // single-byte register-or-constant index encoding, so no jitcode
            // can be assembled for it (`unsupported_jit_shape`).  Run this frame
            // in the plain interpreter; nested callees stay JIT-eligible (the
            // overflow is per-frame), so no `JitSuppressionGuard`.  Record the
            // decline in the census so the interpreted frame is visible, not a
            // silent no-token gap.
            pyre_jit_trace::jitcode_dispatch::census_record_frame_shape_decline(
                code as *const _ as usize,
                "FrameShape::ConstEncodingOverflow",
            );
            return frame_root.frame().execute_frame(None, None);
        }
    }
    frame_root.frame().fix_array_ptrs();
    // Set CURRENT_FRAME so zero-arg super() can find __class__ in the caller.
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame_root.frame());

    // RPython blackhole.py parity: during bridge tracing, concrete
    // (force helper) calls must use the plain interpreter to avoid
    // corrupting the bridge trace's symbolic state via eval_loop_jit's
    // jit_merge_point_hook. RPython's blackhole interpreter has no
    // JIT hooks; pyre's equivalent is eval_frame_plain.
    {
        let (drv, _) = driver_pair();
        if drv.is_bridge_tracing() {
            return frame_root.frame().execute_frame(None, None);
        }
    }

    // RPython warmspot.py ll_portal_runner:
    //   maybe_compile_and_run(increment_threshold, *args)
    //   return portal_ptr(*args)
    //
    // maybe_compile_and_run = try_function_entry_jit: checks for compiled
    // code (dispatch) or threshold (start tracing). Internally guards on
    // JC_TRACING (driver.is_tracing()) to avoid re-entry during tracing.
    //
    // portal_ptr = eval_loop_jit at depth 0 (has jit_merge_point +
    // can_enter_jit back-edge), plain interpreter at depth > 0.
    if let Some(result) = try_function_entry_jit(frame_root.frame()) {
        if majit_metainterp::majit_log_enabled() {
            log_named_global_result(
                frame_root.frame(),
                "eval_with_jit_inner.try_function_entry_jit",
            );
        }
        return result;
    }
    let result = handle_jitexception(frame_root.frame());
    if majit_metainterp::majit_log_enabled() {
        log_named_global_result(
            frame_root.frame(),
            "eval_with_jit_inner.handle_jitexception",
        );
    }
    result
}

fn log_named_global_result(frame: &PyFrame, label: &str) {
    unsafe {
        let w_globals = frame.get_w_globals();
        if w_globals.is_null() {
            return;
        }
        let Some(value) = pyre_object::w_dict_getitem_str(w_globals, "result") else {
            return;
        };
        if value.is_null() {
            eprintln!("[jit][{label}] result=NULL");
            return;
        }
        // pyobject.rs:308 `is_int` returns true for both INT_TYPE and
        // BOOL_TYPE (bool is a W_IntObject subclass sharing `intval`). Match
        // INT_TYPE strictly here so the log labels a bool result distinctly
        // in the branch below.
        if pyre_object::pyobject::py_type_check(value, &pyre_object::pyobject::INT_TYPE) {
            eprintln!(
                "[jit][{label}] result_ptr=0x{:x} kind=int intval={}",
                value as usize,
                pyre_object::intobject::w_int_get_value(value),
            );
        } else if pyre_object::pyobject::is_bool(value) {
            eprintln!("[jit][{label}] result_ptr=0x{:x} kind=bool", value as usize,);
        } else {
            eprintln!(
                "[jit][{label}] result_ptr=0x{:x} kind=other",
                value as usize,
            );
        }
    }
}

/// warmspot.py:970-983 ContinueRunningNormally → portal_ptr(*args) parity.
///
/// warmspot.py:961-983 handle_jitexception: ContinueRunningNormally path.
///
/// Called from handle_jitexception_in_portal (via portal_runner callback)
/// when ContinueRunningNormally is raised at a recursive portal level.
/// Extracts the red_ref values (frame locals as PyObjectRef pointers)
/// and calls the portal function (eval_with_jit) with those values.
///
/// Returns Ok((return_type, value)) or Err(JitException) if the portal
/// itself raises a JitException (warmspot.py:979-980 loop back).
pub(crate) fn pyre_portal_runner(
    exc: &majit_metainterp::jitexc::JitException,
) -> Result<(majit_metainterp::blackhole::BhReturnType, i64), majit_metainterp::jitexc::JitException>
{
    use majit_metainterp::blackhole::BhReturnType;
    use majit_metainterp::jitexc::JitException;

    let JitException::ContinueRunningNormally {
        green_int,
        green_ref,
        green_float,
        red_int,
        red_ref,
        red_float,
    } = exc
    else {
        return Ok((BhReturnType::Void, 0));
    };

    // warmspot.py:972-975: portalfunc_ARGS extraction.
    // Build merged arg lists like blackhole.py:1113-1116.
    let mut all_i = green_int.clone();
    all_i.extend(red_int);
    let mut all_r = green_ref.clone();
    all_r.extend(red_ref);
    let _all_f = (green_float, red_float);

    // warmspot.py:976-978: result = portal_ptr(*args)
    let next_instr = all_i.first().copied().unwrap_or(0) as usize;
    let pycode = all_r.first().copied().unwrap_or(0) as pyre_object::PyObjectRef;
    let frame_ptr = all_r.get(1).copied().unwrap_or(0) as *mut PyFrame;
    let ec = all_r.get(2).copied().unwrap_or(0) as *const pyre_interpreter::PyExecutionContext;
    if frame_ptr.is_null() {
        return Err(JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(0)));
    }
    let frame = unsafe { &mut *frame_ptr };
    if !pycode.is_null() {
        frame.pycode = pycode as *const ();
    }
    if !ec.is_null() {
        frame.execution_context = ec;
    }
    frame.set_last_instr_from_next_instr(next_instr);
    match portal_runner_result(frame) {
        Ok(result) => Ok((BhReturnType::Ref, result as i64)),
        Err(err) => Err(JitException::ExitFrameWithExceptionRef(majit_ir::GcRef(
            err.exc_object as usize,
        ))),
    }
}

/// warmspot.py:961-1007 handle_jitexception.
///
/// RPython: CRN → portal_ptr(*args) re-invokes the interpreter.
/// pyre: CRN → re-loop eval_loop_jit(frame). This does NOT call
/// maybe_compile_and_run (warmspot.py:948); portal_ptr is a plain
/// interpreter dispatch, and pyre's eval_loop_jit is the equivalent.
/// TODO: exact portal_ptr(*args) parity (currently `continue`
/// re-enters without re-extracting CRN args from the exception).
#[inline(always)]
fn handle_jitexception(frame: &mut PyFrame) -> PyResult {
    let mut frame_root = FrameRoot::new(frame);
    loop {
        let loop_outcome = eval_loop_jit(frame_root.frame());
        // Drain pyre's call-error stash (see `pyre_interpreter::call::set_call_error`).
        // Several PY_NULL-returning helpers (e.g. `call_args_and_c_profile`,
        // `c_call_trace` / `c_return_trace` / `c_exception_trace` callbacks)
        // park their `PyError` here when their signature cannot carry one.
        if let Some(err) = pyre_interpreter::call::take_call_error() {
            return Err(err);
        }
        match loop_outcome {
            LoopResult::Done(result) => return result,
            LoopResult::ContinueRunningNormally => {
                // RPython warmspot.py:976-978: result = portal_ptr(*args).
                // The blackhole has already written back the merge point
                // state to the frame (call_jit.rs:999-1013). Re-enter
                // eval_loop_jit with that state — do NOT reset to entry.
                frame_root.frame().fix_array_ptrs();
                continue;
            }
        }
    }
}

// dont_look_inside: JIT debug/driver helper.
#[majit_macros::dont_look_inside]
fn debug_first_arg_int(frame: &PyFrame) -> Option<i64> {
    if frame.locals_w().len() == 0 {
        return None;
    }
    let value = frame.locals_w()[0];
    if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
        return None;
    }
    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
}

/// warmspot.py:941 ll_portal_runner parity: execute a frame through the
/// JIT-enabled portal runner. Used by bhimpl_recursive_call
/// (blackhole.py:1101-1116) for recursive portal depth.
///
/// warmspot.py:941-959:
///   maybe_compile_and_run(state.increment_function_threshold, *args)
///   return portal_ptr(*args)
///
/// warmspot.py:997-1005: ExitFrameWithExceptionRef → re-raise.
pub(crate) fn portal_runner_result(frame: &mut PyFrame) -> PyResult {
    // warmspot.py:941-955 ll_portal_runner:
    //   maybe_compile_and_run(state.increment_function_threshold, *args)
    //   return portal_ptr(*args)
    //
    // portal_ptr is the JIT-aware interpreter (jit_merge_point +
    // can_enter_jit). pyre's equivalent is handle_jitexception →
    // eval_loop_jit, NOT eval_frame_plain. Routing through
    // eval_frame_plain here would skip maybe_enter_jit at every
    // opcode of the recursive portal frame, which breaks parity for
    // bhimpl_recursive_call_* paths.
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().fix_array_ptrs();
    let _frame_guard = pyre_interpreter::eval::install_current_frame(frame_root.frame());
    // Mirror `eval_with_jit_inner`'s structural-region suppression so a
    // recursive portal entry whose code contains `WITH_EXCEPT_START`
    // keeps nested helper Python frames out of the JIT too. The current
    // frame is already kept out of trace by `try_function_entry_jit` and
    // `jit_merge_point_hook`'s `unsupported_jit_shape` check; the guard
    // extends that to callees.
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    let _suppression = match unsupported_jit_shape(code) {
        UnsupportedJitShape::StructuralRegion => Some(JitSuppressionGuard::new()),
        UnsupportedJitShape::None
        | UnsupportedJitShape::CurrentFrameOnly
        | UnsupportedJitShape::ConstEncodingOverflow => None,
    };
    portal_runner_dispatch(frame_root.frame())
}

fn portal_runner_dispatch(frame: &mut PyFrame) -> PyResult {
    if let Some(result) = try_function_entry_jit(frame) {
        result
    } else {
        handle_jitexception(frame)
    }
}

pub fn portal_runner(frame: &mut PyFrame) -> pyre_object::PyObjectRef {
    match portal_runner_result(frame) {
        Ok(r) => r,
        Err(err) => {
            crate::call_jit::store_jit_exception(err.exc_object as i64);
            pyre_object::PY_NULL
        }
    }
}

/// pyre-local debug instrumentation (no PyPy counterpart).
/// `@not_in_trace` so that compiled code does not include this call.
#[majit_macros::not_in_trace]
fn trace_jit_bytecode(_pc: usize, _instruction_name: &str) {
    // Debug logging disabled — per-bytecode eprintln causes O(n) slowdown.
}

/// warmspot.py portal_runner parity: execute a frame through the JIT-enabled
/// interpreter. Used by bhimpl_recursive_call (blackhole.py:1074-1093) for
/// recursive portal depth. Returns PyObjectRef (NULL on void/exception).
/// JIT hooks are thin inline checks; all heavy logic is in #[cold] helpers.
fn eval_loop_jit(frame: &mut PyFrame) -> LoopResult {
    let mut frame_root = FrameRoot::new(frame);
    // Bump the monotonic frame eval-loop entry odometer (mirrors the plain
    // `eval_loop` entry): a user Python frame is about to run bytecode.  The
    // FBW FOR_ITER Option-C guard snapshots this around a residual call to
    // detect a body effect that ran through user code.
    pyre_interpreter::call::bump_frame_entry_count();
    // Count this eval-loop activation for the GC safepoint's
    // at_outermost_activation gate (gh#393). The gate allows collection
    // at depth ≤ 2 (module + one called function) where the CALL opcode
    // has completed and no Rust-stack ref is live, but blocks at depth
    // ≥ 3 (a callback re-entry from FOR_ITER→__getitem__, exception
    // handler, etc., where the outer opcode handler holds a PyObjectRef
    // on the Rust stack that walk_pyframe_roots cannot reach).
    let _eval_activation = pyre_object::gc_interp::EvalActivationGuard::enter();
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    let env = PyreEnv;
    let (driver, info) = driver_pair();
    // The codewriter-side portal check
    // (`CallControl::jitdriver_sd_from_portal_graph`, codewriter.py:37)
    // is the canonical "is this code a portal" answer once
    // `setup_jitdriver` has registered it.
    //
    // Note: pyre routes every
    // CodeObject through `jit_merge_point_hook` and `can_enter_jit`
    // so that recursive calls into a previously-traced function reach
    // `maybe_compile_and_run` even before the function's own loop
    // runs. RPython does not need this because portals are an
    // explicit registry (`jitdrivers_sd`), not an inferred property,
    // and recursion goes through the portal_runner. Two narrowing
    // alternatives both regress benchmarks:
    //   - "is registered portal" alone (post-`setup_jitdriver`):
    //     non-loop function frames never trigger registration, so
    //     recursive entry never reaches `maybe_compile_and_run` —
    //     surfaces as a TLS-drop panic in
    //     `test_inline_residual_user_call_with_many_args_stays_correct`.
    //   - "has back-edge AND name != <module>": same problem —
    //     non-loop function frames are skipped.
    //
    // interp_jit.py:81-99 `PyFrame.dispatch` applies `pypyjitdriver`
    // (`jit_merge_point` :87, `can_enter_jit` :117) to EVERY frame
    // uniformly — there is no `co_name == "<module>"` gate and no env
    // switch, so `<module>` frames trace exactly like function frames.
    // The parity-correct value is unconditional `true`.
    //
    // This was briefly gated (a `<module>` exclusion, then a
    // PYRE_MODULE_LOOP_TRACE env switch) while module-loop tracing was a
    // deopt-storm regression: a dynamic driver-loop call to a loop-bearing
    // callee is not inlinable by the full-body walker yet (#62).  That is
    // resolved — the walk now declines such a key to the trait leg
    // (`DispatchError::LoopBearingCalleeInlineUnsupported` ->
    // `FBW_DECLINED_KEYS`), which inlines the callee via
    // `recursive-call-assembler` — so module-loop tracing is a win
    // (nbody_50k 0.22s interpreter -> 0.09s traced) and the gate is gone.
    let is_portal: bool = true;
    // interp_jit.py:66 — next_instr, pycode are greens (managed by jit_merge_point).
    // No explicit promote needed; the JitDriver green-key mechanism handles this.

    loop {
        // Interpreter-path GC safepoint (PYRE_GC_INTERP). Between opcodes the
        // only live refs are in the frame, reachable through the registered
        // pyframe root walker; no bytecode handler holds a Rust-stack temporary
        // here. A no-op unless the flag is on and enough interpreter objects
        // have accumulated to warrant a collection.
        pyre_object::gc_interp::safepoint();

        // Stop-the-world safepoint: a compiled loop's back-edge poll deopts
        // here when a collector has requested STW; park until it completes.
        // Between opcodes no bytecode handler holds a Rust-stack ref (see the
        // note above), so this is a walkable safepoint.
        majit_gc::gc_sync::safepoint_poll();

        if frame_root.frame().next_instr() >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame_root.frame().next_instr();
        let (opcode_pc, instruction, op_arg) = match decode_instruction_for_dispatch(code, pc) {
            Ok(decoded) => decoded,
            Err(err) => return LoopResult::Done(Err(err.into())),
        };

        // ── jit_merge_point (RPython interp_jit.py:85-87) ──
        // Runtime no-op. Only handles trace feed when tracing is active.
        if is_portal {
            let tracing_depth: Option<u32> = driver.meta_interp().tracing_call_depth;
            let mut merge_point_active = if let Some(depth) = tracing_depth {
                call_depth() == depth
            } else {
                driver.is_tracing()
            };
            // A frame running under trace-continuation suspend is a
            // residual-executed callee — the walk reached a self-recursive
            // call and ran it concretely through `execute_residual_call`
            // (jitdriver.rs `TraceContinuationSuspendGuard`).  It re-enters
            // eval_loop_jit at the same call depth as the trace, so the depth
            // check above mis-identifies it as a merge point.  It is opaque to
            // the active trace's merge points (`do_residual_call` never
            // re-enters the portal), so skip the merge-point feed and run it as
            // plain interpretation.
            if merge_point_active && majit_metainterp::trace_continuation_suspended() {
                merge_point_active = false;
            }
            if merge_point_active {
                if let Some(loop_result) =
                    jit_merge_point_hook(frame_root.frame(), code, pc, driver, info, &env)
                {
                    return loop_result;
                }
            }
        }

        // ── handle_bytecode (RPython interp_jit.py:90) ──
        trace_jit_bytecode(pc, "");
        frame_root.frame().last_instr = pc as isize;
        frame_root
            .frame()
            .set_last_instr_from_next_instr(opcode_pc + 1);
        // pyopcode.py:170-176 dispatch_bytecode parity: fire
        // `ec.bytecode_trace(self)` each opcode while warming up,
        // with the default `TICK_COUNTER_STEP` decrement.  This is
        // NOT the same call site as interp_jit.py:115
        // `jump_absolute`'s `ec.bytecode_trace(self, decr_by)` —
        // jump_absolute fires on backward jumps only, with an
        // adapted tick (`_get_adapted_tick_counter()`); that path
        // is in `__extend__::jump_absolute` above and its
        // `pypyjitdriver.can_enter_jit` half is dispatched by
        // `StepResult::CloseLoop` → `maybe_compile_and_run` below.
        // The naive call (`(*ec).bytecode_trace(...)`) regresses
        // hot benchmarks 28-29% because the function-call boundary
        // hides the no-tracer fast path from the optimizer. Inline
        // the gate here — read `ec.w_tracefunc` directly and skip
        // the trace-only slow path when null. The ticker decrement
        // (executioncontext.py:163-165) runs unconditionally so
        // signal handlers / async actions fire periodically (matches
        // PyPy's `actionflag.decrement_ticker(decr_by)` invariant);
        // the `action_dispatcher` slow path itself is still a stub
        // pending the actionflag port.
        let ec_ptr = frame_root.frame().execution_context as *mut PyExecutionContext;
        if !ec_ptr.is_null() {
            let needs_trace = unsafe { !(*ec_ptr).w_tracefunc.is_null() };
            if needs_trace {
                if let Err(err) = unsafe {
                    (*ec_ptr).bytecode_trace(
                        frame_root.frame() as *mut PyFrame,
                        pyre_interpreter::executioncontext::TICK_COUNTER_STEP,
                    )
                } {
                    return LoopResult::Done(Err(err));
                }
                // A trace callback may perform a debugger line-jump by
                // setting `frame.f_lineno` (`fset_f_lineno` → `last_instr
                // = best_addr`).  The opcode for this iteration was
                // decoded from the pre-jump `pc`; honour the jump by
                // restarting the loop so the target is re-decoded,
                // mirroring the interpreter loop's post-trace redirect.
                // The baseline set before the trace is `last_instr =
                // opcode_pc` (line above); a moved `last_instr` is the
                // jump target.  This loop reads `pc = frame.next_instr()`
                // (= `last_instr + 1`) at the top, so rebase the target
                // through `set_last_instr_from_next_instr` for the next
                // iteration to land on it rather than one past it.
                if frame_root.frame().last_instr as usize != opcode_pc {
                    let jump_target = frame_root.frame().last_instr as usize;
                    frame_root
                        .frame()
                        .set_last_instr_from_next_instr(jump_target);
                    continue;
                }
            } else {
                // executioncontext.py:163-165 — `actionflag.
                // decrement_ticker(decr_by)` runs every bytecode, and
                // `action_dispatcher` runs once it goes negative.
                // bytecode_trace bundles both when a tracer is set; the
                // no-tracer fast path inlines them.  The OS signal
                // handler forces the ticker to -1 (signalstate::
                // signal_pushback), so this is where Ctrl-C is delivered
                // during JIT warm-up.  The negative branch is rarely
                // taken — the fast path stays a load + not-taken compare.
                let ticker = unsafe {
                    (*ec_ptr).actionflag.decrement_ticker(
                        pyre_interpreter::executioncontext::TICK_COUNTER_STEP as isize,
                    )
                };
                if ticker < 0 {
                    if let Err(mut err) =
                        unsafe { (*ec_ptr).perform_actions(frame_root.frame() as *mut PyFrame) }
                    {
                        // Deliver the action's exception (e.g. a signal
                        // handler's KeyboardInterrupt) as if raised at the
                        // current opcode so the frame's try/except can
                        // catch it — CPython runs the eval-breaker
                        // exception through the same `goto error` path.
                        // `frame.last_instr` was set to `pc` above, so
                        // `handle_exception` finds the covering handler.
                        let mut next_instr = frame_root.frame().next_instr();
                        if pyre_interpreter::eval::handle_exception(
                            frame_root.frame(),
                            &mut err,
                            &mut next_instr,
                        ) {
                            frame_root
                                .frame()
                                .set_last_instr_from_next_instr(next_instr);
                            continue;
                        }
                        return LoopResult::Done(Err(err));
                    }
                }
            }
        }
        let mut next_instr = frame_root.frame().next_instr();
        let raw_arg: u32 = op_arg.into();
        let delta = instruction.stack_effect(raw_arg);
        if delta > 0 {
            let frame = frame_root.frame();
            let pushed_top = frame.valuestackdepth.saturating_add(delta as usize);
            let next_pc = opcode_pc + 1;
            // A JIT handoff can arrive with the stack depth for the point just
            // after a super-instruction while `last_instr` still names the
            // super-instruction itself. If metadata proves the current depth
            // belongs to the next opcode, advance the pc instead of re-running
            // pushes that are already reflected in the frame stack.
            if pushed_top > frame.locals_w().len()
                && pyre_jit_trace::state::depth_based_vsd_for_wcode(frame.pycode as usize, next_pc)
                    == Some(frame.valuestackdepth)
            {
                frame_root.frame().set_last_instr_from_next_instr(next_pc);
                continue;
            }
        }
        let step_result =
            execute_opcode_step(frame_root.frame(), code, instruction, op_arg, next_instr);
        match step_result {
            Ok(StepResult::Continue) => {
                // pyjitpl.py:2843 blackhole_if_trace_too_long — check after
                // every traced step to prevent infinite trace recording.
                driver.blackhole_if_trace_too_long();
            }
            Ok(StepResult::CloseLoop { loop_header_pc, .. }) if is_portal => {
                // ── can_enter_jit (RPython interp_jit.py:114) ──
                // RPython interp_jit.py:114 → warmstate.py:446
                let green_key = make_green_key(frame_root.frame().pycode, loop_header_pc);
                if let Some(loop_result) = maybe_compile_and_run(
                    frame_root.frame(),
                    green_key,
                    loop_header_pc,
                    driver,
                    info,
                    &env,
                ) {
                    return loop_result;
                }
            }
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(mut err) => {
                if pyre_interpreter::eval::handle_exception(
                    frame_root.frame(),
                    &mut err,
                    &mut next_instr,
                ) {
                    frame_root
                        .frame()
                        .set_last_instr_from_next_instr(next_instr);
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// pyjitpl.py:2837-2845 _interpret() parity for bridge tracing.
///
/// RPython's bridge tracing uses the same MetaInterp._interpret() loop
/// as normal tracing. This function provides the same eval loop as
/// eval_loop_jit, but always calls jit_merge_point_hook since tracing
/// is already active from start_bridge_tracing.
pub(crate) fn eval_loop_jit_bridge(frame: &mut PyFrame) -> LoopResult {
    let mut frame_root = FrameRoot::new(frame);
    // Same as eval_loop_jit: count the activation for the safepoint's
    // depth gate (gh#393). See the comment in eval_loop_jit.
    let _eval_activation = pyre_object::gc_interp::EvalActivationGuard::enter();
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    let env = PyreEnv;
    let (driver, info) = driver_pair();

    loop {
        if frame_root.frame().next_instr() >= code.instructions.len() {
            return LoopResult::Done(Ok(w_none()));
        }

        let pc = frame_root.frame().next_instr();
        let (opcode_pc, instruction, op_arg) = match decode_instruction_for_dispatch(code, pc) {
            Ok(decoded) => decoded,
            Err(err) => return LoopResult::Done(Err(err.into())),
        };

        // pyjitpl.py:1892-1914 run_one_step: trace + execute.
        if driver.is_tracing() {
            if let Some(loop_result) =
                jit_merge_point_hook(frame_root.frame(), code, pc, driver, info, &env)
            {
                return loop_result;
            }
        } else {
            // Tracing ended (bridge compiled or aborted).
            return LoopResult::Done(Ok(w_none()));
        }

        // handle_bytecode: execute the bytecode on the concrete frame.
        let next_instr = opcode_pc + 1;
        frame_root
            .frame()
            .set_last_instr_from_next_instr(next_instr);
        let step_result =
            execute_opcode_step(frame_root.frame(), code, instruction, op_arg, next_instr);
        match step_result {
            Ok(StepResult::Continue) => {}
            Ok(StepResult::CloseLoop { .. }) => {}
            Ok(StepResult::Return(result)) => return LoopResult::Done(Ok(result)),
            Ok(StepResult::Yield(result)) => return LoopResult::Done(Ok(result)),
            Err(mut err) => {
                let mut next_instr = frame_root.frame().next_instr();
                if pyre_interpreter::eval::handle_exception(
                    frame_root.frame(),
                    &mut err,
                    &mut next_instr,
                ) {
                    frame_root
                        .frame()
                        .set_last_instr_from_next_instr(next_instr);
                    continue;
                }
                return LoopResult::Done(Err(err));
            }
        }
    }
}

/// #57 Option C (deliver): on a FOR_ITER trace abort, deliver the in-flight
/// iteration to the live frame instead of dropping it.
///
/// The aborted walk advanced the real shared heap iterator once (an
/// irreversible side effect with no journal undo) and the recording was
/// discarded, leaving the live frame parked at the FOR_ITER loop header with
/// the iterator on TOS but the consumed item neither pushed nor its body run
/// — the `ContinueRunningNormally` re-entry would then re-run FOR_ITER on the
/// already-advanced iterator, consume the next item, and drop the in-flight
/// one.  Instead reconstruct the interpreter resume
/// state at the point AFTER the consume: push the already-consumed item onto
/// the live value stack (above the kept iterator, the FOR_ITER continue-arm
/// shape) and reposition the frame at the loop BODY (`body_pc`, the FOR_ITER
/// fallthrough).  The `ContinueRunningNormally` re-entry then runs the body
/// exactly once for that item and continues the loop from the already-
/// advanced iterator — the `_copy_data_from_miframe` continue-forward analog
/// (blackhole.py:1711), no drop and no double.
///
/// The repositioning is the load-bearing effect, encoded in the frame itself
/// (its value stack and pc), not in the return value: on delivery the frame is
/// moved to `body_pc` with the item pushed, so the caller's
/// `ContinueRunningNormally` re-entry runs the body once; on refusal or no
/// in-flight item the frame is left untouched, so the SAME
/// `ContinueRunningNormally` re-entry takes the legacy drop-on-abort (the
/// conservative never-double fallback).  Both call sites therefore continue
/// identically and need not branch on the result — `true` (delivered /
/// repositioned-to-body) vs `false` (refused or empty → frame unchanged) is
/// informational (the debug log distinguishes the two `false` cases).  The R1
/// double-apply guard lives in `fbw_foriter_inflight_take`.
fn deliver_inflight_foriter_item(frame: &mut PyFrame) -> bool {
    let Some((item, body_pc)) = pyre_jit_trace::jitcode_dispatch::fbw_foriter_inflight_take()
    else {
        return false;
    };
    // #57 Option C (Finding #3, loud-failure assert): the R1 guard in
    // `fbw_foriter_inflight_take` returns `Some` (delivers) ONLY when no body
    // effect committed for the in-flight iteration, so re-running the body
    // cannot double.  With Finding #1's inverted predicate this is unreachable;
    // the assert turns any future regression (a missed mutator that lets a
    // delivery slip past a standing body-effect signal) into a loud debug
    // abort instead of a silent double-apply.  `take` leaves the signals
    // intact, so `fbw_foriter_any_body_effect_signal()` reads the same state
    // the guard just checked.
    debug_assert!(
        !pyre_jit_trace::jitcode_dispatch::fbw_foriter_any_body_effect_signal(),
        "Option C delivered an in-flight FOR_ITER item while a body-effect \
         signal stands (body_pc={body_pc}) — re-running the body would double \
         a committed effect (R1 guard regression)"
    );
    // #57 Option C (header-state guard): the push+reposition below assumes the
    // live frame is parked at the loop-header FOR_ITER state for `body_pc` —
    // the iterator on TOS, the body's STORE_FAST expecting `item` one slot
    // above.  `body_pc` is nested-aware (derived from the consumed FOR_ITER
    // op's own pc), so it can name an INNER FOR_ITER reached deeper in a
    // traced body.  For such an inner consume the live frame is parked at the
    // OUTER loop header (the walk-entry / jit_merge_point pc), NOT at the
    // inner header — its value stack carries the outer body state and the
    // outer iterator, not the inner iterator on TOS.  Pushing there and
    // jumping to the inner `body_pc` corrupts the operand stack (a later
    // FOR_ITER/GET_ITER then reads a wrong slot as an iterator).  Deliver only
    // when the frame is PROVABLY at the header for `body_pc`: it is parked at
    // the FOR_ITER opcode whose fallthrough is `body_pc`
    // (`next_instr() == body_pc - 1`) and that opcode really is a `FOR_ITER`.
    // The walk parks the live frame at the loop header it entered, so a
    // header-entry consume satisfies this and still DELIVERS; a non-header
    // inner consume fails it and is REFUSED — the stash is dropped (already
    // taken above) and the legacy bypass keeps the conservative drop-on-abort,
    // never a stack-corrupting push.  This is the `fbw_foriter_inflight_take`
    // refuse-when-not-provably-safe model applied to the stack-state axis.
    // `body_pc` is the FOR_ITER `orgpc + 1`, so it is always >= 1; the header
    // pc is one before it.  A `body_pc == 0` (impossible) wraps to `usize::MAX`
    // and fails the `next_instr()` match, so the guard stays safe without a
    // separate zero check.
    let header_pc = body_pc.wrapping_sub(1);
    let at_loop_header = frame.next_instr() == header_pc && {
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
        matches!(
            pyre_interpreter::decode_instruction_at(code, header_pc),
            Some((pyre_interpreter::Instruction::ForIter { .. }, _))
        )
    };
    if !at_loop_header {
        if pyre_jit_trace::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!(
                "[fbw-foriter] deliver REFUSED (live frame not at the loop header for \
                 body_pc={body_pc}) frame.next_instr()={} — keeping legacy drop-on-abort \
                 to avoid a non-header stack-corrupting push",
                frame.next_instr()
            );
        }
        return false;
    }
    // The continue arm keeps the iterator on the stack and pushes `next`
    // above it (codewriter.rs FOR_ITER continue arm; opcode_for_iter never
    // pops the iterator).  The live frame is still at the loop-header state
    // with the iterator on TOS, so a single push lands `item` exactly where
    // the body's STORE_FAST expects TOS.
    frame.push(item);
    // Resume at the FOR_ITER fallthrough body opcode.  `next_instr` /
    // `last_instr` are Python bytecode coordinates, matching `body_pc`
    // (the FOR_ITER `orgpc + 1`).
    frame.set_last_instr_from_next_instr(body_pc);
    true
}

/// RPython jit_merge_point slow path — only called when tracing is active.
// dont_look_inside: jit_merge_point slow path; the tracer must not enter the driver.
#[cold]
#[majit_macros::dont_look_inside]
fn jit_merge_point_hook(
    frame: &mut PyFrame,
    code: &pyre_interpreter::CodeObject,
    pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    if jit_suppressed_by_unsupported_frame()
        || unsupported_jit_shape(code) != UnsupportedJitShape::None
    {
        return None;
    }
    let concrete_frame = frame as *mut PyFrame as usize;
    let green_key = make_green_key(frame.pycode, pc);

    // The trace-START decision (counter / threshold / start-tracing) lives
    // in the warmstate marker path — `maybe_compile_with_key` (back-edge)
    // and `force_start_tracing_for_key` (function-entry/recursion) walk the
    // cell chain by `comparekey_matches` and own the decision. This hook is
    // only the trace FEED: it runs once tracing is already active and hands
    // each merge-point opcode to `jit_merge_point_keyed`. `make_green_key`
    // and the warmstate cell key are the same allocation-free
    // `pypyjit_greenkey_uhash`, so the feed key and the decision key agree.

    let mut jit_state = build_jit_state(frame, info);
    let current_depth = call_depth();
    let was_tracing = driver.is_tracing();
    // warmstate.py:437-444: capture the starting cell's key before
    // entering the trace body so we can unconditionally clear its
    // TRACING flag in the post-trace finally block. May differ from
    // `green_key` when we are mid-trace and the current merge point's
    // key is not the tracing origin.
    let starting_tracing_key = driver.starting_green_key();
    let mut propagated_exception = None;
    let driver_outcome = driver.jit_merge_point_keyed(
        green_key,
        pc,
        &mut jit_state,
        env,
        || {},
        |meta, sym| {
            meta.tracing_call_depth = Some(current_depth);
            // RPython parity: codewriter.make_jitcodes() runs before tracing
            // starts, populating all_liveness. In pyre, JitCode compilation is
            // lazy — ensure the code's JitCode (with liveness) exists before
            // tracing so get_list_of_active_boxes can use it.
            crate::jit::codewriter::register_portal_jitdriver(code);
            let snapshot = frame.snapshot_for_tracing();
            let _ = concrete_frame;
            let live_frame_addr = &*frame as *const PyFrame as usize;
            let (action, executed_frame) =
                trace_bytecode(meta, sym, code, pc, snapshot, live_frame_addr, true);
            // pyjitpl.py:3048-3091 raise_continue_running_normally: tracing
            // IS execution — a walk that committed its end-of-walk state
            // into the snapshot (CloseLoop / CompileTracePending flush)
            // hands that state to the LIVE frame, so the
            // ContinueRunningNormally re-entry continues from the walked
            // iteration's end instead of replaying it (re-applying every
            // concretely executed side effect).  An uncommitted flush
            // leaves the snapshot at entry state — adopting it is a no-op.
            let walk_end_flushed = pyre_jit_trace::trace::take_walk_end_flush_committed();
            let walk_end_restart_pc = pyre_jit_trace::trace::take_walk_end_restart_pc();
            if walk_end_flushed {
                frame.restore_resume_state_from(&executed_frame);
            } else if let Some(restart_pc) = walk_end_restart_pc {
                // When `fbw_has_unjournaled_effect` / `PYRE_FBW_END_FLUSH=0`
                // leaves the end flush uncommitted, the live frame stays at trace entry.
                // At a super-instruction loop close it only corrects `last_instr` to the
                // marker-consistent restart pc; walked locals/stack stay out for consistent replay.
                frame.set_last_instr_from_next_instr(restart_pc);
            }
            propagated_exception = pyre_jit_trace::trace::take_walk_end_propagated_exception();
            action
        },
    );
    if let Some(err) = propagated_exception {
        return Some(LoopResult::Done(Err(err)));
    }
    if let Some(outcome) = driver_outcome {
        match handle_jit_outcome(outcome, &jit_state, frame, info, green_key) {
            JitAction::Return(result) => return Some(LoopResult::Done(result)),
            JitAction::ContinueRunningNormally => return Some(LoopResult::ContinueRunningNormally),
            JitAction::Continue => {}
        }
    }
    // Trace completed or aborted — clear tracing depth.
    if !driver.is_tracing() {
        driver.meta_interp_mut().tracing_call_depth = None;
        // compile.py:269: cross-loop cut stores under inner key.
        // Use the actual compiled key for post-compilation steps.
        let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
        // warmstate.py:444 `finally: cell.flags &= ~JC_TRACING` parity.
        // `starting_tracing_key` was captured before jit_merge_point_keyed;
        // its TRACING must be cleared unconditionally — even if cross-loop
        // cut compiled under a different key, or if the trace aborted.
        if let Some(k) = starting_tracing_key {
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .clear_tracing_flag(k);
        }
        register_quasi_immutable_deps(compiled_key);
        // RPython pyjitpl.py:3048-3061 raise_continue_running_normally:
        // after trace compilation, restart so maybe_compile_and_run
        // (try_function_entry_jit) dispatches to compiled code.
        if was_tracing {
            // #57 Option C (deliver): a FOR_ITER trace that aborted advanced
            // the real iterator once but discarded its recording.  Deliver
            // the in-flight item to the live frame (push + reposition at the
            // body) so the ContinueRunningNormally re-entry runs the body
            // once for it, instead of bypassing past the now-orphaned
            // FOR_ITER and dropping the iteration.
            deliver_inflight_foriter_item(frame);
            // No-replay portal exit for a loop-free function trace: when the
            // walk captured its concrete return (the `run_perfn_walk`
            // epilogue kept the stash only when the walk's eager side
            // effects stand and no symbolic-only effect needs the replay),
            // hand that result back directly.  Re-running the freshly
            // compiled trace for THIS invocation would re-read the heap the
            // walk already consumed (a side-effecting residual ran once) and
            // deopt; the compiled trace serves only subsequent invocations.
            // No capture → the legacy ContinueRunningNormally replay.
            match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
                Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
                    let result = match cv {
                        // A void return stashes `Null`, i.e. Python `None`
                        // (`ConcreteValue::to_pyobj` would map it to PY_NULL).
                        pyre_jit_trace::state::ConcreteValue::Null => w_none(),
                        other => other.to_pyobj(),
                    };
                    return Some(LoopResult::Done(Ok(result)));
                }
                Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
                    return Some(LoopResult::Done(Err(finish_concrete_raise_error(cv))));
                }
                None => {}
            }
            return Some(LoopResult::ContinueRunningNormally);
        }
    }
    None
}

/// RPython warmstate.py:446-511 maybe_compile_and_run.
///
/// Entry point to the JIT. Called at can_enter_jit (back-edge).
///
/// RPython order: cell lookup (JC_TRACING → skip, JC_COMPILED → enter)
/// BEFORE counter.tick(). This prevents compiled loops from occupying
/// counter hash-table slots and evicting non-compiled loops (the 5-way
/// associative cache has only 5 slots per bucket).
// dont_look_inside: warmstate maybe_compile_and_run; the tracer must not enter the driver.
#[cold]
#[majit_macros::dont_look_inside]
fn maybe_compile_and_run(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    // pyre-local extension: PYRE_NO_JIT disables JIT entirely.
    // No RPython counterpart — kept for development debugging only.
    // TODO: remove when JIT is stable enough to not need a kill switch.
    static NO_JIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT.get_or_init(|| std::env::var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame) };
    if jit_suppressed_by_unsupported_frame()
        || unsupported_jit_shape(code) != UnsupportedJitShape::None
    {
        return None;
    }
    if let Some(expected_vsd) =
        pyre_jit_trace::state::depth_based_vsd_for_wcode(frame.pycode as usize, loop_header_pc)
    {
        if frame.valuestackdepth != expected_vsd {
            // A pyre super-instruction compiled-loop exit can rewind next_instr to the header
            // while retaining an over-counted vable stack depth. Valid Python loops have one depth per bytecode offset,
            // so this never permanently interprets a traceable loop; PyPy's MIFrame.pc JitCode-offset
            // plus ResumeDataDirectReader exact-resume invariant structurally preclude it.
            return None;
        }
    }
    // warmstate.py:473-477: JC_TRACING → skip entirely (no counter tick)
    if driver.is_tracing() {
        return None;
    }
    // warmstate.py:503-511: procedure_token exists → EnterJitAssembler.
    // RPython enters assembler unconditionally when a compiled loop is
    // available for this green_key.
    if driver.has_compiled_loop(green_key) {
        return execute_assembler(frame, green_key, loop_header_pc, driver, info, env);
    }
    // warmstate.py:484: DONT_TRACE_HERE → skip counter tick entirely
    if driver
        .meta_interp()
        .warm_state_ref()
        .is_dont_trace_here(green_key)
    {
        return None;
    }
    // warmstate.py:496-511: counter.tick → threshold reached → bound_reached
    // TODO(parity): warmstate.py:473-496 funnels every back-edge through
    // `maybe_compile_and_run`, which checks JC_TRACING, compiled-loop
    // presence, DONT_TRACE_HERE, has_seen_a_procedure_token, and
    // counter.tick in one linear sequence.  Pyre splits the checks
    // across this function and `counter_tick_checked` (warmstate.rs:559).
    // The flag-based DONT_TRACE_HERE path above duplicates part of the
    // warmstate logic; verify that `counter_tick_checked` still covers
    // the `has_seen_a_procedure_token` guard and the full `bound_reached`
    // flow identically to warmstate.py:496-511.
    if driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter_tick_checked(green_key)
    {
        if driver
            .meta_interp()
            .is_tracing_key((frame.pycode as usize, loop_header_pc))
        {
            return None;
        }
        return bound_reached(frame, green_key, loop_header_pc, driver, info, env);
    }
    None
}

/// Panic-safe RAII pairing for `FailDescr::start_compiling` /
/// `done_compiling`.  `compile.py:704-709`:
///
/// ```python
/// self.start_compiling()
/// try:
///     self._trace_and_compile_from_bridge(...)
/// finally:
///     self.done_compiling()
/// ```
///
/// brackets bridge compilation with `start_compiling()` (set
/// `ST_BUSY_FLAG`) and `done_compiling()` (clear it) in a try/finally —
/// even when bridge compilation raises, `done_compiling` runs on the
/// unwind path.  Pyre would otherwise leave the busy flag latched if
/// the inner `trace_and_compile_from_bridge` panics, blocking every
/// subsequent guard-fail retry on the same descriptor.  Holding an
/// `Arc<dyn Descr>` clone keeps the descr alive across the scope and
/// lets the drop call `as_fail_descr().done_compiling()` directly,
/// matching `compile.py:786-795` instance-method dispatch.
#[must_use = "drop the guard to clear ST_BUSY_FLAG"]
pub(crate) struct GuardCompilingScope {
    descr: std::sync::Arc<dyn majit_ir::Descr>,
}

impl GuardCompilingScope {
    pub(crate) fn new(descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Self {
        // `compile.py:786-795 ResumeGuardDescr.start_compiling` is an
        // instance method on `FailDescr` upstream — PyPy structurally
        // cannot reach this code path with a non-fail descriptor (the
        // `handle_fail` caller is itself a method on `FailDescr`).
        // Pyre takes a `&Arc<dyn Descr>` to avoid an upfront downcast
        // at the call site; PyPy raises `AttributeError` on a non-fail
        // descr at the very first `start_compiling` lookup, so we
        // panic in both debug and release builds via `expect` to match
        // that fail-fast contract instead of silently skipping the
        // start/done pair.
        let fd = descr
            .as_fail_descr()
            .expect("GuardCompilingScope built on a non-fail descr; PyPy can only reach handle_fail through a FailDescr instance");
        fd.start_compiling();
        Self {
            descr: std::sync::Arc::clone(descr),
        }
    }
}

impl Drop for GuardCompilingScope {
    fn drop(&mut self) {
        // The constructor's `expect` guarantees the underlying
        // concrete type behind `dyn Descr` is a `FailDescr`; that
        // type does not change for the lifetime of the Arc, so the
        // downcast on the unwind path must also succeed.
        let fd = self
            .descr
            .as_fail_descr()
            .expect("GuardCompilingScope dropped with a descr that lost its FailDescr identity");
        fd.done_compiling();
    }
}

/// Build the exception result the same way as blackhole.py:1679-1682
/// `_exit_frame_with_exception`.
fn finish_concrete_raise_error(value: pyre_jit_trace::state::ConcreteValue) -> PyError {
    let pyre_jit_trace::state::ConcreteValue::Ref(exc_ref) = value else {
        unreachable!("FinishConcrete::Raise must hold a concrete Ref")
    };
    debug_assert!(!exc_ref.is_null());
    // blackhole.py:1679-1682 constructs ExitFrameWithExceptionRef directly
    // from the uncaught exception object and leaves the exception latches as-is.
    unsafe { PyError::from_exc_object(exc_ref) }
}

/// compile.py:701-717: handle_fail NEVER returns in RPython — it raises
/// ContinueRunningNormally or DoneWithThisFrame. In pyre, we return the
/// equivalent BlackholeResult.
enum HandleFailOutcome {
    /// Bridge compiled successfully — continue in compiled code.
    BridgeCompiled,
    /// Resume in blackhole interpreter.
    ResumeInBlackhole,
    /// The single-frame bridge walk ran the resumed frame forward to a
    /// `Finish` and captured its concrete return value (`interpret()`
    /// raising `DoneWithThisFrame`).  Return it directly instead of
    /// rewinding + re-running the region (#177).
    BridgeFinished(pyre_object::PyObjectRef),
    /// The bridge walk ended in an uncaught raise; propagate the same
    /// `PyError` as `ExitFrameWithExceptionRef` (jitexc.py:44).
    BridgeRaised(PyError),
}

/// Re-derive `frame.valuestackdepth` from the resume pc `resume_pc` and null
/// the operand slots above it.  A blackhole guard-failure resume writes the
/// failing guard's own recorded operand depth into the frame; when the handoff
/// resumes at a different pc (a merge point) that depth over-counts, so the loop
/// header's pushes overflow the frame at its peak stack use.  Re-derive the
/// operand depth from the actual resume pc, mirroring the bridge path's
/// `depth_based_vsd_for_wcode` correction.  Shared by every blackhole resume leg
/// (the eval.rs CRN arms via [`apply_blackhole_crn_handoff`] and the
/// CALL_ASSEMBLER arm in `handle_blackhole_result`) so the resume coordinate and
/// its operand depth stay consistent.  A `None` depth (missing liveness) leaves
/// the frame untouched, matching the bridge path's skip-on-None.
#[majit_macros::dont_look_inside]
pub(crate) fn correct_resume_vsd(frame: &mut PyFrame, resume_pc: usize) {
    if let Some(corrected) =
        pyre_jit_trace::state::depth_based_vsd_for_wcode(frame.pycode as usize, resume_pc)
    {
        frame.valuestackdepth = corrected;
        frame.clear_stack_above(corrected);
    }
}

/// Blackhole `ContinueRunningNormally` handoff: resume `frame` at the
/// merge-point next_instr carried in `green_int[0]` and re-derive its
/// `valuestackdepth` from that resume pc via [`correct_resume_vsd`].
///
/// warmspot.py:961 `handle_jitexception` parity — the CRN carries the
/// merge-point args, so the frame restarts at the merge point, not the
/// guard-failure pc.
#[majit_macros::dont_look_inside]
fn apply_blackhole_crn_handoff(frame: &mut PyFrame, green_int: &[i64]) {
    let Some(&ni) = green_int.first() else {
        return;
    };
    frame.set_last_instr_from_next_instr(ni as usize);
    correct_resume_vsd(frame, ni as usize);
}

/// compile.py:701-717 handle_fail.
///
/// Single function containing the complete guard failure handling:
/// compile.py:701-717 handle_fail.
///
/// RPython: handle_fail NEVER returns — both paths raise
/// ContinueRunningNormally or DoneWithThisFrame.
/// pyre: returns BlackholeResult (equivalent to RPython's exceptions).
// dont_look_inside: compile.py handle_fail; post-trace outcome dispatch.
#[majit_macros::dont_look_inside]
fn handle_fail(
    frame: &mut PyFrame,
    green_key: u64,
    _trace_id: u64,
    fail_index: u32,
    descr_arc: &std::sync::Arc<dyn majit_ir::Descr>,
    should_bridge: bool,
    _owning_key: u64,
    exit_layout: &CompiledExitLayout,
    raw_values: &[i64],
    guard_exc: i64,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> HandleFailOutcome {
    // The range FOR_ITER `GuardClass(RANGE_ITER)` proves its own site
    // polymorphic on the first failure.  Demote before `should_bridge` can
    // spend another retrace-limit cycle trying to close a bridge at the same
    // failing loop header; blackhole resumes this invocation without the
    // invalidated compiled loop.
    if pyre_jit_trace::trace::range_foriter_guard_failed(green_key, fail_index) {
        let (driver, _) = driver_pair();
        driver.invalidate_loop(green_key);
        // This is an intentional replacement, unlike ordinary
        // GUARD_NOT_INVALIDATED handling: discard the range trace's target
        // tokens so the next walk compiles the generic FOR_ITER residual.
        driver.remove_compiled_loop(green_key);
        return HandleFailOutcome::ResumeInBlackhole;
    }

    // compile.py:702-703: must_compile() AND not stack_almost_full()
    if should_bridge && !stack_almost_full() {
        let is_tracing = {
            let (driver, _) = driver_pair();
            driver.is_tracing()
        };
        if !is_tracing {
            // compile.py:704-709 try/finally: start_compiling() before
            // bridge, done_compiling() on every unwind path.  The RAII
            // guard packages both halves: ctor fires `start_compiling`
            // via `descr.as_fail_descr()` (direct instance-method
            // dispatch matching `compile.py:786-795`); drop fires
            // `done_compiling` so a panic inside
            // `trace_and_compile_from_bridge` cannot latch
            // `ST_BUSY_FLAG`.
            let resolution = {
                let _guard = GuardCompilingScope::new(descr_arc);
                // force_plain_eval prevents concrete calls during bridge
                // tracing from re-entering compiled code.
                let _plain = pyre_interpreter::call::force_plain_eval();
                // `allow_finish_direct_return = true`: the general guard path
                // can hand a concrete `Finish` result back to its portal.
                crate::call_jit::trace_and_compile_from_bridge(
                    descr_arc,
                    frame,
                    raw_values,
                    exit_layout,
                    guard_exc,
                    true,
                )
            };
            match resolution {
                crate::call_jit::BridgeResolution::CompiledContinue => {
                    // compile.py:708: bridge compiled → ContinueRunningNormally.
                    // RPython: the bridge is attached to the guard descr;
                    // re-entering compiled code will follow the bridge.
                    return HandleFailOutcome::BridgeCompiled;
                }
                crate::call_jit::BridgeResolution::Finished(cv) => {
                    // #177: the walk ran the resumed frame forward to its
                    // return and captured the concrete result; hand it back
                    // as `DoneWithThisFrame` (`interpret()` raising it from
                    // the post-walk state) rather than rewinding + re-running.
                    // The bridge stays attached for subsequent guard failures.
                    let v = match cv {
                        // A void return stashes `Null`, i.e. Python `None`.
                        pyre_jit_trace::state::ConcreteValue::Null => w_none(),
                        other => other.to_pyobj(),
                    };
                    return HandleFailOutcome::BridgeFinished(v);
                }
                crate::call_jit::BridgeResolution::FinishedException(cv) => {
                    return HandleFailOutcome::BridgeRaised(finish_concrete_raise_error(cv));
                }
                crate::call_jit::BridgeResolution::ResumeBlackhole => {}
            }
        }
    }
    // compile.py:710-716 / pyjitpl.py:2906 (SwitchToBlackhole):
    // resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
    HandleFailOutcome::ResumeInBlackhole
}

/// Short tag for a `BlackholeResult` variant, for the `[bh-rd-numb]`
/// blackhole-resume log line.
fn blackhole_result_tag(r: &crate::call_jit::BlackholeResult) -> &'static str {
    use crate::call_jit::BlackholeResult as R;
    match r {
        R::ContinueRunningNormally { .. } => "ContinueRunningNormally",
        R::DoneWithThisFrameVoid => "DoneWithThisFrameVoid",
        R::DoneWithThisFrameInt(_) => "DoneWithThisFrameInt",
        R::DoneWithThisFrameRef(_) => "DoneWithThisFrameRef",
        R::DoneWithThisFrameFloat(_) => "DoneWithThisFrameFloat",
        R::ExitFrameWithExceptionRef(_) => "ExitFrameWithExceptionRef",
        R::Failed => "Failed",
    }
}

/// compile.py:710-716 resume_in_blackhole parity.
///
/// RPython: resume_in_blackhole → blackhole_from_resumedata →
/// consume_one_section → _run_forever → raises.
///
// dont_look_inside: post-trace blackhole resume machinery.
#[majit_macros::dont_look_inside]
pub(crate) fn resume_in_blackhole_from_exit_layout(
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
    guard_exc: i64,
) -> crate::call_jit::BlackholeResult {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] resume_in_blackhole: raw_values.len={} exit_types.len={} rd_numb={:?}",
            raw_values.len(),
            exit_layout.exit_types.len(),
            exit_layout.storage.as_deref().map(|s| s.rd_numb.len())
        );
    }

    // resume.py:1312 blackhole_from_resumedata is the single blackhole
    // resume mechanism: every exit_layout that carries resume storage
    // decodes through the orthodox rd_numb reader
    // `blackhole_resume_via_rd_numb`. It walks jitcode liveness once per
    // resume frame, so it reconstructs the full inline framestack.
    // exit_layout already carries (rd_loop_token, trace_id, fail_index,
    // storage), mirroring the CALL_ASSEMBLER caller
    // `jit_blackhole_resume_from_guard` (call_jit.rs:1855-1881) without the
    // green_key recovery that path needs.
    if let Some(storage) = exit_layout.storage.as_deref() {
        let deadframe_types = {
            let (driver, _) = driver_pair();
            driver.get_recovery_slot_types(
                exit_layout.rd_loop_token,
                exit_layout.trace_id,
                exit_layout.fail_index,
            )
        };
        let result = crate::call_jit::blackhole_resume_via_rd_numb(
            &storage.rd_numb,
            storage.rd_consts(),
            raw_values,
            Some(&storage.rd_pendingfields),
            Some(&storage.rd_virtuals),
            deadframe_types.as_deref(),
            guard_exc,
        );
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[bh-rd-numb] trace={} fail_idx={} result={}",
                exit_layout.trace_id,
                exit_layout.fail_index,
                blackhole_result_tag(&result),
            );
        }
        return result;
    }
    // resume.py:1369-1372 `ResumeDataDirectReader._prepare` dereferences
    // `storage.rd_numb_list` with no fallback: a guard reaching the
    // blackhole resume MUST carry rd_numb. `storage` is None only for
    // terminal FINISH/JUMP exit layouts (compile.rs
    // `infer_terminal_exit_layout`) and synthesized fallback layouts,
    // none of which reach the guard blackhole path. If one ever did,
    // there would be no resume data to decode — fail loudly rather than
    // silently mis-resume.
    panic!(
        "resume_in_blackhole_from_exit_layout: exit_layout.storage missing \
         (trace={} fail_idx={})",
        exit_layout.trace_id, exit_layout.fail_index,
    );
}

/// Replace every tagged-immediate int in a frame's local slots with a real
/// heap `W_IntObject`, in place, before a compiled loop reads them.
///
/// `w_int_new` tags small ints as `(value << 1) | 1` during interpretation, so
/// a hot-loop-carried local can enter the JIT as a tagged immediate. The trace
/// body would then record the tag-arithmetic unbox (`CastPtrToInt`+`IntRshift`)
/// against the entry InputArg; the unroll replicates that into the steady loop,
/// where the loop-carried value is a heap box (low bit 0) and the
/// `GuardTrue(lowbit)` fails on every back-edge → deopt storm / hang. Converting
/// the slot to a heap box here — at the concrete boundary, recording no IR —
/// makes the compiled loop read heap `W_IntObject`, so the trace is the
/// flag-false `GuardClass`+`GetfieldGcPure` shape that virtualizes to a raw
/// carry with zero in-loop tag ops.
///
/// Only the locals region (`0..nlocals`) is scanned; cell/free vars and stack
/// temps are left untouched. GC-safe: `w_int_new_unique` returns a managed heap
/// box and the store lands in the frame's `locals_cells_stack_w` array, which
/// is a GC root for the duration of the compiled run (`FrameLocalsRoot`) and is
/// forwarded via the current-frame chain during any collection the allocation
/// itself triggers.
#[inline]
fn untag_tagged_frame_locals(frame: &mut PyFrame) {
    if !pyre_object::tagged_int::CAN_BE_TAGGED {
        return;
    }
    let nlocals = frame.nlocals();
    let locals = frame.locals_w_mut();
    let n = nlocals.min(locals.len());
    for i in 0..n {
        let slot = locals[i];
        if !slot.is_null() && pyre_object::tagged_int::is_tagged_int(slot) {
            let value = pyre_object::tagged_int::untag_int(slot);
            let boxed = pyre_object::intobject::w_int_new_unique(value);
            locals[i] = boxed;
        }
    }
}

/// RPython warmstate.py:387-423 execute_assembler.
///
/// Run compiled machine code for a given green_key. Handles the
/// fail_descr outcomes: DoneWithThisFrame, GuardFailure, etc.
// dont_look_inside: runs compiled machine code; the tracer must not enter it.
#[cold]
#[majit_macros::dont_look_inside]
fn execute_assembler(
    frame: &mut PyFrame,
    green_key: u64,
    entry_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let mut frame_root = FrameRoot::new(frame);
    frame_root.frame().set_last_instr_from_next_instr(entry_pc);

    // Convert tagged-immediate frame locals to heap `W_IntObject` before the
    // compiled loop reads them, so the trace body operates on the flag-false
    // heap-int representation (no in-loop tag test). See `untag_tagged_frame_locals`.
    untag_tagged_frame_locals(frame);

    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..frame_root.frame().locals_w().len().min(5))
            .map(|i| {
                let value = frame_root.frame().locals_w()[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!("[jit][execute-assembler][locals] {:?}", locals);
    }

    let mut jit_state = build_jit_state(frame_root.frame(), info);

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][execute-assembler] key={} pc={} arg0={:?}",
            green_key,
            entry_pc,
            debug_first_arg_int(frame_root.frame()),
        );
    }

    // warmstate.py:395 func_execute_token(loop_token, *args) → deadframe
    let outcome = {
        let _frame_locals_root = FrameLocalsRoot::new(frame_root.frame());
        driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            entry_pc,
            &mut jit_state,
            env,
            || {},
        )
    };

    // rstack.stack_check_slowpath → _StackOverflow parity: drain the
    // JIT-overflow flag the backend probe records when it trips. The
    // backend detects the overflow inside compiled code and exits via
    // the dedicated stack-overflow block; we surface the user-visible
    // RecursionError here on the way back to the interpreter loop.
    if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
        return Some(LoopResult::Done(Err(exc)));
    }

    // warmspot.py:998 ExitFrameWithExceptionRef: check for exceptions
    // stashed by blackhole/force callbacks across FFI boundaries.
    if let Some(exc) = crate::call_jit::take_ca_exception() {
        return Some(LoopResult::Done(Err(exc)));
    }

    if majit_metainterp::majit_log_enabled() {
        let kind = match &outcome {
            DetailedDriverRunOutcome::Finished { .. } => "finished",
            DetailedDriverRunOutcome::Jump { .. } => "jump",
            DetailedDriverRunOutcome::Abort { .. } => "abort",
            DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
        };
        eprintln!(
            "[jit][execute-assembler] outcome key={} pc={} kind={}",
            green_key, entry_pc, kind
        );
    }

    // warmstate.py:402-422 handle fail_descr outcome
    match outcome {
        // warmstate.py:402-415 fast path: DoneWithThisFrame
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
            // warmspot.py:998 handle_jitexception:
            //   ExitFrameWithExceptionRef.handle_fail raises the stored Ref
            //   into the outer interpreter's exception machinery.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return Some(LoopResult::Done(Err(
                            pyre_interpreter::PyError::type_error(
                                "compiled exit_frame_with_exception did not produce a single Ref value",
                            ),
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                return Some(LoopResult::Done(Err(err)));
            }
            let [value] = typed_values.as_slice() else {
                return Some(LoopResult::Done(Err(
                    pyre_interpreter::PyError::type_error(
                        "compiled finish did not produce a single object return value",
                    ),
                )));
            };
            let result = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py:631 DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py:640 DoneWithThisFrameDescrRef parity:
                    // return get_result() as-is. jitframe GC trace hook
                    // (jitframe.rs:293) keeps interior refs alive.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return Some(LoopResult::Done(Err(
                        pyre_interpreter::PyError::type_error(
                            "compiled finish produced a void return value",
                        ),
                    )));
                }
            };
            Some(LoopResult::Done(Ok(result)))
        }
        // warmstate.py:416-422 general: handle_fail
        // compile.py:701-717 → bridge or blackhole
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref raw_values,
            ref exit_layout,
            guard_exc,
        } => {
            match handle_fail(
                frame_root.frame(),
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => Some(LoopResult::ContinueRunningNormally),
                // #177: single-frame bridge walk returned a concrete Finish.
                HandleFailOutcome::BridgeFinished(v) => Some(LoopResult::Done(Ok(v))),
                HandleFailOutcome::BridgeRaised(err) => Some(LoopResult::Done(Err(err))),
                HandleFailOutcome::ResumeInBlackhole => {
                    // compile.py:710-716 / pyjitpl.py:2906 SwitchToBlackhole
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(raw_values, exit_layout, guard_exc);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally {
                            green_int,
                            ..
                        } => {
                            apply_blackhole_crn_handoff(frame_root.frame(), green_int);
                            Some(LoopResult::ContinueRunningNormally)
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrameRef(v) => {
                            Some(LoopResult::Done(Ok(*v)))
                        }
                        crate::call_jit::BlackholeResult::DoneWithThisFrameInt(v) => {
                            // warmspot.py:988-990: box Int to Ref for portal result_type=Ref
                            Some(LoopResult::Done(Ok(
                                pyre_object::intobject::w_int_new(*v) as pyre_object::PyObjectRef
                            )))
                        }
                        crate::call_jit::BlackholeResult::ExitFrameWithExceptionRef(exc) => {
                            // warmspot.py:998-1005 ExitFrameWithExceptionRef:
                            // propagate the Python exception, don't swallow it.
                            Some(LoopResult::Done(Err(exc.clone())))
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            // RPython: blackhole resume never fails — rd_numb
                            // is always complete (`blackhole.py:1679` raises
                            // `ExitFrameWithExceptionRef` for uncaught
                            // exceptions, never returns a failure code).
                            // Pyre's `BlackholeResult::Failed` is a layered
                            // adaptation; SSA-authoritative live_r encoder /
                            // decoder work should eliminate the remaining
                            // triggers. Until then
                            // the bare `invalidate_loop` keeps the cell
                            // retraceable; the failure surfaces in
                            // check.py rather than being masked.
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} trace={} guard={} — invalidating",
                                    green_key, trace_id, fail_index,
                                );
                            }
                            driver.invalidate_loop(green_key);
                            None
                        }
                        _ => bh_result.to_pyresult().map(LoopResult::Done),
                    }
                }
            }
        }
        DetailedDriverRunOutcome::Jump { .. } | DetailedDriverRunOutcome::Abort { .. } => None,
    }
}

/// RPython warmstate.py:425-444 bound_reached.
///
/// Called when counter threshold fires and no compiled code exists.
/// Starts tracing via back_edge_or_run_compiled_keyed.
// dont_look_inside: JIT-driver counter/back-edge slow path the tracer must not enter.
#[cold]
#[majit_macros::dont_look_inside]
fn bound_reached(
    frame: &mut PyFrame,
    green_key: u64,
    loop_header_pc: usize,
    driver: &mut JitDriver<PyreJitState>,
    info: &majit_metainterp::virtualizable::VirtualizableInfo,
    env: &PyreEnv,
) -> Option<LoopResult> {
    let mut frame_root = FrameRoot::new(frame);
    if majit_metainterp::majit_log_enabled() {
        let locals: Vec<(usize, Option<i64>)> = (0..frame_root.frame().locals_w().len().min(5))
            .map(|i| {
                let value = frame_root.frame().locals_w()[i];
                let decoded = if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) }
                {
                    None
                } else {
                    Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                };
                (value as usize, decoded)
            })
            .collect();
        eprintln!(
            "[jit][bound-reached] key={} pc={} arg0={:?} locals={:?}",
            green_key,
            loop_header_pc,
            debug_first_arg_int(frame_root.frame()),
            locals,
        );
    }
    // warmstate.py:429: jitcounter.decay_all_counters()
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .decay_all_counters();
    // warmstate.py:430
    if stack_almost_full() {
        return None;
    }
    // warmstate.py:437-444: MetaInterp.compile_and_run_once
    frame_root
        .frame()
        .set_last_instr_from_next_instr(loop_header_pc);
    let mut jit_state = build_jit_state(frame_root.frame(), info);
    // warmstate.py:473-477 JC_TRACING
    if driver
        .meta_interp()
        .is_tracing_key((frame_root.frame().pycode as usize, loop_header_pc))
    {
        return None;
    }
    // warmstate.py:503-511: procedure_token → EnterJitAssembler.
    let outcome = if driver.has_compiled_loop(green_key) {
        let _frame_locals_root = FrameLocalsRoot::new(frame_root.frame());
        Some(driver.run_compiled_detailed_with_bridge_keyed(
            green_key,
            loop_header_pc,
            &mut jit_state,
            env,
            || {},
        ))
    } else if !driver.is_tracing() {
        // warmstate.py:425-444 bound_reached: enter tracing if the cell's
        // counter / flags allow.  Pyre's `driver.bound_reached` does NOT
        // compile synchronously — it returns `BackEdgeAction::StartedTracing`
        // and the actual trace is driven by `jit_merge_point_keyed` below
        // when `is_tracing()` becomes true after this call.
        //
        // PyPy parity: `maybe_compile_and_run` (warmstate.py:482-511)
        // identifies "the compile we just made" through `cell.procedure_token`
        // (per-greenkey cell), NOT by reading any global last-compiled
        // value.  Pyre's equivalent is `has_compiled_loop(green_key)` —
        // never `last_compiled_key()`, which is a single global slot that
        // accumulates across iterations and cannot tell "stale prior
        // compile" from "fresh same-key compile this round".  If a
        // cross-loop cut compiles an INNER key, attachment goes to the
        // INNER cell; the next iteration's `has_compiled_loop` query at
        // the inner entry point dispatches to it (warmstate.py:482-483).
        let had_compiled = driver.has_compiled_loop(green_key);
        driver.bound_reached(green_key, loop_header_pc, &mut jit_state, env);
        if driver.is_tracing() {
            // RPython pyjitpl.py:2876-2888 _compile_and_run_once:
            // interpret() traces the entire loop synchronously.
            // Set tracing_call_depth so inner function calls (which
            // run their own eval_loop_jit) don't trigger jit_merge_point_hook.
            driver.meta_interp_mut().tracing_call_depth = Some(call_depth());
            let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
            let mut propagated_exception = None;
            let outcome = driver.jit_merge_point_keyed(
                green_key,
                loop_header_pc,
                &mut jit_state,
                env,
                || {},
                |meta, sym| {
                    use pyre_jit_trace::trace::trace_bytecode;
                    crate::jit::codewriter::register_portal_jitdriver(code);
                    let concrete_frame = frame_root.frame().snapshot_for_tracing();
                    let live_frame_addr = frame_root.frame() as *const PyFrame as usize;
                    let (action, executed_frame) = trace_bytecode(
                        meta,
                        sym,
                        code,
                        loop_header_pc,
                        concrete_frame,
                        live_frame_addr,
                        true,
                    );
                    // raise_continue_running_normally seam — see the
                    // jit_merge_point_hook tracing site for the contract.
                    let walk_end_flushed = pyre_jit_trace::trace::take_walk_end_flush_committed();
                    let walk_end_restart_pc = pyre_jit_trace::trace::take_walk_end_restart_pc();
                    if walk_end_flushed {
                        frame_root
                            .frame()
                            .restore_resume_state_from(&executed_frame);
                    } else if let Some(restart_pc) = walk_end_restart_pc {
                        frame_root
                            .frame()
                            .set_last_instr_from_next_instr(restart_pc);
                    }
                    propagated_exception =
                        pyre_jit_trace::trace::take_walk_end_propagated_exception();
                    action
                },
            );
            driver.meta_interp_mut().tracing_call_depth = None;
            if let Some(err) = propagated_exception {
                return Some(LoopResult::Done(Err(err)));
            }
            let compiled_key = driver.last_compiled_key().unwrap_or(green_key);
            if !had_compiled && driver.has_compiled_loop(compiled_key) {
                register_quasi_immutable_deps(compiled_key);
            }
            // pyjitpl.py:3048-3061 raise_continue_running_normally:
            // after compilation, restart so execute_assembler runs.
            if !driver.is_tracing() {
                // warmstate.py:444 `finally: cell.flags &= ~JC_TRACING`
                // — green_key is the starting cell. Cross-loop cut
                // (compile.py:269) installs the token on an inner cell,
                // so attach_procedure_to_interp does not clear TRACING
                // on green_key. Restore the clear here. The full
                // gate `!had_compiled && has_compiled_loop(compiled_key)
                // && compiled_key != green_key` narrows to "this round
                // cross-loop-compiled under a different inner key";
                // without it stale `last_compiled_key` values from
                // prior iterations trigger spurious clears that can
                // destabilize active traces (cranelift fannkuch regresses
                // without this gate).
                if !had_compiled
                    && driver.has_compiled_loop(compiled_key)
                    && compiled_key != green_key
                {
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .clear_tracing_flag(green_key);
                }
                // No-replay portal exit for a walk that started at this
                // loop header but fell through to `done_with_this_frame`
                // (the back-edge counter tripped on the loop's terminal
                // iteration, so the loop test exited immediately and the
                // walk traced the post-loop tail to the frame return).
                // The walk executed the tail's residual calls concretely
                // and captured the concrete return value; re-running the
                // freshly compiled trace for THIS invocation
                // (ContinueRunningNormally re-enters the live frame still
                // parked at the loop header) would re-apply those already
                // executed side effects.  Hand the captured result back
                // directly, mirroring the `jit_merge_point_hook` tracing
                // site (which carries the same no-replay logic for the
                // merge-point-driven trace path).
                // #57 Option C (deliver): a FOR_ITER trace that aborted on
                // the back-edge `can_enter_jit` path advanced the real
                // iterator once but discarded its recording.  Deliver the
                // in-flight item to the live frame so the
                // ContinueRunningNormally re-entry runs the body once for it
                // (the same continuation as the `jit_merge_point_hook`
                // tracing site).
                deliver_inflight_foriter_item(frame_root.frame());
                match pyre_jit_trace::jitcode_dispatch::fbw_finish_concrete_take() {
                    Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Return(cv)) => {
                        let result = match cv {
                            // A void return stashes `Null`, i.e. Python `None`.
                            pyre_jit_trace::state::ConcreteValue::Null => w_none(),
                            other => other.to_pyobj(),
                        };
                        return Some(LoopResult::Done(Ok(result)));
                    }
                    Some(pyre_jit_trace::jitcode_dispatch::FinishConcrete::Raise(cv)) => {
                        return Some(LoopResult::Done(Err(finish_concrete_raise_error(cv))));
                    }
                    None => {}
                }
                return Some(LoopResult::ContinueRunningNormally);
            }
            outcome
        } else {
            None
        }
    } else {
        None
    };
    if let Some(outcome) = outcome {
        // rstack.stack_check_slowpath → _StackOverflow parity: drain
        // the JIT-overflow flag the backend probe records when it
        // trips. The backend's prologue exits via the dedicated
        // stack-overflow block; we surface RecursionError here on the
        // way back to the interpreter loop.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(LoopResult::Done(Err(exc)));
        }
        // compile.py:701-717 handle_fail: bridge/blackhole decision.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref raw_values,
            ref exit_layout,
            guard_exc,
        } = outcome
        {
            match handle_fail(
                frame_root.frame(),
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    return Some(LoopResult::ContinueRunningNormally);
                }
                // #177: single-frame bridge walk returned a concrete Finish.
                HandleFailOutcome::BridgeFinished(v) => {
                    return Some(LoopResult::Done(Ok(v)));
                }
                HandleFailOutcome::BridgeRaised(err) => {
                    return Some(LoopResult::Done(Err(err)));
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(raw_values, exit_layout, guard_exc);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally {
                            green_int,
                            ..
                        } => {
                            apply_blackhole_crn_handoff(frame_root.frame(), green_int);
                            return Some(LoopResult::ContinueRunningNormally);
                        }
                        crate::call_jit::BlackholeResult::Failed => {}
                        _ => {
                            if let Some(r) = bh_result.to_pyresult() {
                                return Some(LoopResult::Done(r));
                            }
                        }
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame_root.frame(), info, green_key) {
                JitAction::Return(result) => return Some(LoopResult::Done(result)),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }
    }
    driver.meta_interp_mut().tracing_call_depth = None;
    None
}

/// RPython warmstate.py maybe_compile_and_run parity.
///
/// Called at every portal entry (function call). Must be fast for the
/// common case (no compiled code, not tracing, threshold not reached).
pub fn try_function_entry_jit(frame: &mut PyFrame) -> Option<PyResult> {
    let mut frame_root = FrameRoot::new(frame);
    // warmstate.py parity: PYRE_NO_JIT disables ALL JIT paths.
    static NO_JIT_FN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *NO_JIT_FN.get_or_init(|| std::env::var_os("PYRE_NO_JIT").is_some()) {
        return None;
    }
    let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(frame_root.frame()) };
    if jit_suppressed_by_unsupported_frame()
        || unsupported_jit_shape(code) != UnsupportedJitShape::None
    {
        return None;
    }
    if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
        if code.obj_name.as_str() == "fannkuch" && frame_root.frame().next_instr() == 0 {
            use std::sync::OnceLock;
            static DUMPED: OnceLock<()> = OnceLock::new();
            if DUMPED.get().is_none() {
                let _ = DUMPED.set(());
                let mut state = pyre_interpreter::OpArgState::default();
                eprintln!("-- fannkuch bytecode dump --");
                for (pc, unit) in code.instructions.iter().copied().enumerate() {
                    let (instr, oparg) = state.get(unit);
                    eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
                }
                for pc in [
                    72usize, 99, 129, 131, 141, 155, 168, 179, 234, 245, 447, 449,
                ] {
                    eprintln!(
                        "decode[{pc}] = {:?}",
                        pyre_interpreter::decode_instruction_at(code, pc)
                    );
                }
            }
        }
    }
    let green_key = make_green_key(frame_root.frame().pycode, frame_root.frame().next_instr());
    let (driver, info) = driver_pair();

    // RPython warmstate.py maybe_compile_and_run fast path:
    // if no compiled loop and not tracing, just tick the counter.
    if !driver.has_compiled_loop(green_key) && !driver.is_tracing() {
        let should_trace = driver
            .meta_interp_mut()
            .warm_state_mut()
            .should_trace_function_entry(green_key);
        if !should_trace {
            return None;
        }
    }

    // RPython warmstate.py:473-477: per-cell JC_TRACING.
    if driver.meta_interp().is_tracing_key((
        frame_root.frame().pycode as usize,
        frame_root.frame().next_instr(),
    )) {
        return None;
    }
    if driver.has_compiled_loop(green_key) {
        // Same gate as maybe_compile_and_run: only enter compiled code
        // when a compiled loop exists for this green_key.
        // warmstate.py:503-511: procedure_token → enter unconditionally.
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][func-entry] run compiled frame=0x{:x} locals=0x{:x} key={} arg0={:?} depth={} raw_finish_known={}",
                frame_root.frame() as *mut PyFrame as usize,
                frame_root.frame().locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame_root.frame()),
                call_depth(),
                driver.has_raw_int_finish()
            );
        }
        let env = PyreEnv;
        // Same concrete boundary as `execute_assembler`: a function-entry
        // compiled trace reads its arg locals as heap `W_IntObject`
        // (`GuardClass`+`GetfieldGcPure`, entry-local tag test omitted), but
        // this path never runs `execute_assembler`, so convert here too. A
        // recursive callee (`fib(n-1)`) arrives with a tagged-immediate arg
        // local; without this the compiled `GuardClass(n)` derefs the tag.
        untag_tagged_frame_locals(frame_root.frame());
        let mut jit_state = build_jit_state(frame_root.frame(), info);
        let outcome = {
            let _frame_locals_root = FrameLocalsRoot::new(frame_root.frame());
            driver.run_compiled_detailed_with_bridge_keyed(
                green_key,
                frame_root.frame().next_instr(),
                &mut jit_state,
                &env,
                || {},
            )
        };
        // rstack.stack_check_slowpath → _StackOverflow parity: drain
        // the JIT-overflow flag the backend probe records when it
        // trips during compiled execution at function entry.
        if let Err(exc) = pyre_interpreter::stack_check::drain_jit_pending_exception() {
            return Some(Err(exc));
        }
        // warmspot.py:998 ExitFrameWithExceptionRef: check for exceptions
        // stashed by blackhole/force callbacks across FFI boundaries.
        if let Some(exc) = crate::call_jit::take_ca_exception() {
            return Some(Err(exc));
        }
        if majit_metainterp::majit_log_enabled() {
            let kind = match &outcome {
                DetailedDriverRunOutcome::Finished { .. } => "finished",
                DetailedDriverRunOutcome::Jump { .. } => "jump",
                DetailedDriverRunOutcome::Abort { .. } => "abort",
                DetailedDriverRunOutcome::GuardFailure { .. } => "guard-failure",
            };
            eprintln!(
                "[jit][func-entry] compiled outcome frame=0x{:x} locals=0x{:x} key={} arg0={:?} kind={}",
                frame_root.frame() as *mut PyFrame as usize,
                frame_root.frame().locals_cells_stack_w as usize,
                green_key,
                debug_first_arg_int(frame_root.frame()),
                kind
            );
        }

        // compile.py:701-717 handle_fail parity.
        if let DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            ref descr_arc,
            should_bridge,
            owning_key,
            ref raw_values,
            ref exit_layout,
            guard_exc,
        } = outcome
        {
            match handle_fail(
                frame_root.frame(),
                green_key,
                trace_id,
                fail_index,
                descr_arc,
                should_bridge,
                owning_key,
                exit_layout,
                raw_values,
                guard_exc,
                info,
            ) {
                HandleFailOutcome::BridgeCompiled => {
                    // Bridge compiled → ContinueRunningNormally → re-enter
                    // compiled code which will follow the new bridge.
                    // Fall through to eval_loop_jit below.
                }
                // #177: single-frame bridge walk returned a concrete Finish.
                // This site returns `Option<PyResult>` (not `LoopResult`).
                HandleFailOutcome::BridgeFinished(v) => {
                    return Some(Ok(v));
                }
                HandleFailOutcome::BridgeRaised(err) => {
                    return Some(Err(err));
                }
                HandleFailOutcome::ResumeInBlackhole => {
                    let bh_result =
                        resume_in_blackhole_from_exit_layout(raw_values, exit_layout, guard_exc);
                    match &bh_result {
                        crate::call_jit::BlackholeResult::ContinueRunningNormally {
                            green_int,
                            ..
                        } => {
                            apply_blackhole_crn_handoff(frame_root.frame(), green_int);
                            // Fall through to eval_loop_jit
                        }
                        crate::call_jit::BlackholeResult::Failed => {
                            // RPython blackhole resume cannot fail
                            // (`blackhole.py:1679` raises
                            // `ExitFrameWithExceptionRef` instead).  The
                            // `BlackholeResult::Failed` variant is a pyre
                            // layering; reading/writing registers_r at
                            // post-regalloc color instead of semantic slot
                            // index would eliminate the triggers.
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][BUG] blackhole failed key={} — invalidating",
                                    green_key,
                                );
                            }
                            let (driver, _) = driver_pair();
                            driver.invalidate_loop(green_key);
                        }
                        _ => {
                            if let Some(r) = bh_result.to_pyresult() {
                                if majit_metainterp::majit_log_enabled() {
                                    let returned_intval = match &r {
                                        Ok(obj)
                                            if !obj.is_null()
                                                && unsafe {
                                                    pyre_object::pyobject::is_int(*obj)
                                                } =>
                                        {
                                            Some(unsafe {
                                                pyre_object::intobject::w_int_get_value(*obj)
                                            })
                                        }
                                        _ => None,
                                    };
                                    eprintln!(
                                        "[jit][handle-outcome] bh-return arg0={:?} intval={:?}",
                                        debug_first_arg_int(frame_root.frame()),
                                        returned_intval,
                                    );
                                }
                                return Some(r);
                            }
                        }
                    }
                }
            }
        } else {
            match handle_jit_outcome(outcome, &jit_state, frame_root.frame(), info, green_key) {
                JitAction::Return(result) => return Some(result),
                JitAction::ContinueRunningNormally | JitAction::Continue => {}
            }
        }

        // After compiled code guard-restored fallback, re-establish the
        // frame's array pointer.
        frame_root.frame().fix_array_ptrs();
        return None;
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] probe key={} arg0={:?} tracing={}",
            green_key,
            debug_first_arg_int(frame_root.frame()),
            driver.is_tracing(),
        );
    }

    if driver.is_tracing() {
        return None;
    }

    // warmstate.py:467 jitcounter.tick(hash, increment_threshold). The
    // fast path above already fired the counter for this entry, so go
    // straight to bound_reached without re-ticking.
    if majit_metainterp::majit_log_enabled() {
        let function_threshold = driver.meta_interp().warm_state_ref().function_threshold();
        eprintln!(
            "[jit][func-entry] fired key={} arg0={:?} threshold={}",
            green_key,
            debug_first_arg_int(frame),
            function_threshold,
        );
    }
    // warmstate.py:425-444 bound_reached parity:
    //   if not confirm_enter_jit(*args): return
    //   jitcounter.decay_all_counters()
    //   if rstack.stack_almost_full(): return
    //   metainterp.compile_and_run_once(jitdriver_sd, *args)
    driver
        .meta_interp_mut()
        .warm_state_mut()
        .counter
        .decay_all_counters();
    if stack_almost_full() {
        return None;
    }
    let env = PyreEnv;
    let mut jit_state = build_jit_state(frame_root.frame(), info);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][func-entry] start tracing key={} arg0={:?}",
            green_key,
            debug_first_arg_int(frame_root.frame()),
        );
    }
    {
        let _frame_locals_root = FrameLocalsRoot::new(frame_root.frame());
        driver.force_start_tracing(
            green_key,
            frame_root.frame().next_instr(),
            &mut jit_state,
            &env,
        );
    }
    None
}

// dont_look_inside: JIT-driver outcome dispatch the tracer must not enter.
#[majit_macros::dont_look_inside]
fn handle_jit_outcome(
    outcome: DetailedDriverRunOutcome,
    _jit_state: &PyreJitState,
    frame: &mut PyFrame,
    _info: &majit_metainterp::virtualizable::VirtualizableInfo,
    green_key: u64,
) -> JitAction {
    match outcome {
        DetailedDriverRunOutcome::Finished {
            typed_values,
            raw_int_result,
            is_exit_frame_with_exception,
            ..
        } => {
            let (driver, _) = driver_pair();
            let raw_int_result = raw_int_result || driver.has_raw_int_finish();
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][handle-outcome] finished key={} raw_flag={} exc_exit={} typed_values={:?}",
                    green_key, raw_int_result, is_exit_frame_with_exception, typed_values
                );
            }
            // compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
            if is_exit_frame_with_exception {
                let exc_ref = match typed_values.as_slice() {
                    [majit_ir::Value::Ref(r)] => r.as_usize() as pyre_object::PyObjectRef,
                    _ => {
                        return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                            "compiled exit_frame_with_exception did not produce a single Ref value",
                        )));
                    }
                };
                let err = unsafe { pyre_interpreter::PyError::from_exc_object(exc_ref) };
                return JitAction::Return(Err(err));
            }
            let [value] = typed_values.as_slice() else {
                return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                    "compiled finish did not produce a single object return value",
                )));
            };
            let value = match value {
                majit_ir::Value::Int(raw) => {
                    // compile.py:631 DoneWithThisFrameDescrInt parity —
                    // unused in pyre (result_type=Ref), but handle
                    // gracefully just in case.
                    pyre_object::intobject::w_int_new(*raw)
                }
                majit_ir::Value::Ref(value) => {
                    // compile.py:640 DoneWithThisFrameDescrRef parity.
                    value.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Float(f) => pyre_object::floatobject::w_float_new(*f),
                majit_ir::Value::Void => {
                    return JitAction::Return(Err(pyre_interpreter::PyError::type_error(
                        "compiled finish produced a void return value",
                    )));
                }
            };
            if majit_metainterp::majit_log_enabled() {
                let returned_intval =
                    if !value.is_null() && unsafe { pyre_object::pyobject::is_int(value) } {
                        Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
                    } else {
                        None
                    };
                eprintln!(
                    "[jit][handle-outcome] return arg0={:?} intval={:?} ref=0x{:x}",
                    debug_first_arg_int(frame),
                    returned_intval,
                    value as usize
                );
            }
            JitAction::Return(Ok(value))
        }
        DetailedDriverRunOutcome::Jump {
            continue_running_normally_values,
            continue_running_normally_pc,
            ..
        } => {
            if let Some(values) = continue_running_normally_values {
                // pyjitpl.py:3072-3085 raise_continue_running_normally:
                // commit the back-edge live boxes, then restart at the loop
                // header so the next portal check can enter the compiled loop.
                let restart_pc = continue_running_normally_pc.unwrap_or_else(|| frame.next_instr());
                let env = PyreEnv;
                let mut restart_state = build_jit_state(frame, _info);
                let meta = restart_state.build_meta(restart_pc, &env);
                restart_state.restore_values(&meta, &values);
                frame.set_last_instr_from_next_instr(restart_pc);
                frame.fix_array_ptrs();
            }
            JitAction::Continue
        }
        DetailedDriverRunOutcome::GuardFailure { .. } => {
            // Guard failure handled by handle_fail() before reaching here.
            // If we reach handle_jit_outcome with a GuardFailure, state was
            // already restored — proceed to blackhole resume.
            JitAction::ContinueRunningNormally
        }
        DetailedDriverRunOutcome::Abort { .. } => JitAction::Continue,
    }
}

/// resume.py:1441-1442 allocate_struct(typedescr) → cpu.bh_new(typedescr).
fn allocate_struct(typedescr: &dyn majit_ir::SizeDescr) -> usize {
    let size = typedescr.size();
    let descr = majit_translate::jitcode::BhDescr::Size {
        size,
        // `descr.py:108-118` cache identity — `SizeDescr.cache_key()`
        // returns the `LLType::Struct(path_hash)` slot stamped at
        // `get_size_descr` cache-miss-mint.
        type_id: typedescr.cache_key(),
        vtable: 0,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(typedescr),
        is_gc_managed: typedescr.is_gc_managed(),
    };
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new(&descr) as usize
}

fn bh_array_descr_from_descr(arraydescr: &majit_ir::DescrRef) -> majit_translate::jitcode::BhDescr {
    let ad = arraydescr
        .as_array_descr()
        .expect("resume array path requires an ArrayDescr");
    majit_translate::jitcode::BhDescr::from_array_descr(ad)
}

fn bh_new_array_from_descr(length: usize, arraydescr: &majit_ir::DescrRef, clear: bool) -> i64 {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    let backend = driver.meta_interp().backend();
    if clear {
        backend.bh_new_array_clear(length as i64, &bh_descr)
    } else {
        backend.bh_new_array(length as i64, &bh_descr)
    }
}

fn bh_setarrayitem_int_from_descr(
    array: i64,
    index: usize,
    value: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver
        .meta_interp()
        .backend()
        .bh_setarrayitem_gc_i(array, index as i64, value, &bh_descr);
}

fn bh_setarrayitem_ref_from_descr(
    array: i64,
    index: usize,
    value: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_setarrayitem_gc_r(
        array,
        index as i64,
        majit_ir::GcRef(value as usize),
        &bh_descr,
    );
}

fn bh_setarrayitem_float_from_descr(
    array: i64,
    index: usize,
    value_bits: i64,
    arraydescr: &majit_ir::DescrRef,
) {
    let bh_descr = bh_array_descr_from_descr(arraydescr);
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_setarrayitem_gc_f(
        array,
        index as i64,
        f64::from_bits(value_bits as u64),
        &bh_descr,
    );
}

/// resume.py:1437-1439 allocate_with_vtable(descr) → exec_new_with_vtable(cpu, descr).
/// llmodel.py:778-782: bh_new_with_vtable uses sizedescr.get_vtable().
fn allocate_with_vtable(descr: &dyn majit_ir::SizeDescr) -> usize {
    let size = descr.size();
    let vtable = descr.vtable();
    let bh_descr = majit_translate::jitcode::BhDescr::Size {
        size,
        // `descr.py:108-118` cache identity via `SizeDescr.cache_key()`.
        type_id: descr.cache_key(),
        vtable: vtable as u64,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(descr),
        is_gc_managed: descr.is_gc_managed(),
    };
    let (driver, _) = driver_pair();
    driver.meta_interp().backend().bh_new_with_vtable(&bh_descr) as usize
}

/// resume.py:945-956 getvirtual_ptr parity.
///
/// Lazily materializes a virtual from rd_virtuals[vidx].
/// Pattern: check cache → allocate_with_vtable/allocate_struct → cache → setfields.
/// RPython caches the REAL object pointer before filling fields, enabling
/// recursive/shared virtual resolution without NULL placeholders.
fn materialize_virtual_from_rd(
    vidx: usize,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    // resume.py:951: v = self.virtuals_cache.get_ptr(index)
    if let Some(cached) = virtuals_cache.get(&vidx) {
        return cached.clone();
    }
    // resume.py:953: assert self.rd_virtuals is not None
    let virtuals = rd_virtuals.expect("resume.py:953 getvirtual_ptr: rd_virtuals is not None");
    // resume.py:954: v = self.rd_virtuals[index].allocate(self, index) — direct
    // index; a corrupt resume stream (out-of-range vidx) raises IndexError here
    // rather than being swallowed as a NULL ref.
    let entry = &virtuals[vidx];
    // resume.py:1552-1588 decode_* parity.
    fn decode_tagged_fieldnum(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Option<Value> {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return None;
        }
        let (val, tagbits) = majit_metainterp::resume::untag(tagged);
        Some(match tagbits {
            majit_ir::resumedata::TAGBOX => {
                // resume.py:1562-1564: if num < 0: num += self.count; then
                // cpu.get_*_value(self.deadframe, num) — direct deadframe
                // access, so a corrupt stream (out-of-range num) raises here
                // rather than being papered over with a 0.
                let idx = if val < 0 {
                    (val + num_failargs) as usize
                } else {
                    val as usize
                };
                dead_frame[idx].clone()
            }
            majit_ir::resumedata::TAGINT => Value::Int(val as i64),
            majit_ir::resumedata::TAGCONST => {
                // resume.py:1568-1570 decode_ref: if tagged_eq(tagged, NULLREF):
                //   return ConstPtr.value
                if tagged == majit_ir::resumedata::NULLREF {
                    return Some(Value::Ref(majit_ir::GcRef::NULL));
                }
                // resume.py:1554/1571/1582: self.consts[num - TAG_CONST_OFFSET]
                // — direct index; a corrupt stream raises IndexError here.
                let ci = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
                rd_consts[ci].to_value()
            }
            majit_ir::resumedata::TAGVIRTUAL => {
                // resume.py:278-284 nested virtuals are numbered negatively;
                // getvirtual_ptr resolves them via Python negative list
                // indexing into rd_virtuals (resume.py:951-954).
                let vidx = if val < 0 {
                    (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
                } else {
                    val as usize
                };
                return Some(materialize_virtual_from_rd(
                    vidx,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                ));
            }
            // untag masks to 2 bits (TAGMASK), so tagbits is exhaustively one
            // of TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above. resume.py's decode_*
            // encode this with `assert tag == TAGBOX` in the final else.
            _ => unreachable!("untag yields a 2-bit tag; all four are handled"),
        })
    }
    /// resume.py:1549 decode_int(fieldnum)
    /// Returns the raw i64 value for integer-typed fields.
    fn decode_tagged_fieldnum_int(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> i64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Int(n)) => n,
            Some(Value::Float(f)) => f.to_bits() as i64,
            Some(Value::Ref(gc)) => gc.0 as i64,
            _ => 0,
        }
    }

    /// resume.py:1546 decode_float(fieldnum)
    /// Returns the raw f64 value for float-typed fields.
    fn decode_tagged_fieldnum_float(
        tagged: i16,
        dead_frame: &[Value],
        num_failargs: i32,
        rd_consts: &[majit_ir::Const],
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> f64 {
        match decode_tagged_fieldnum(
            tagged,
            dead_frame,
            num_failargs,
            rd_consts,
            rd_virtuals,
            virtuals_cache,
        ) {
            Some(Value::Float(f)) => f,
            Some(Value::Int(n)) => f64::from_bits(n as u64),
            _ => 0.0,
        }
    }

    fn box_opt_value(v: &Option<Value>) -> pyre_object::PyObjectRef {
        match v {
            Some(Value::Ref(gc)) => gc.0 as pyre_object::PyObjectRef,
            Some(Value::Int(n)) => pyre_object::intobject::w_int_new(*n),
            Some(Value::Float(f)) => pyre_object::floatobject::w_float_new(*f),
            _ => std::ptr::null_mut(),
        }
    }
    // resume.py:643-760: dispatch by virtual kind.
    match entry.as_ref() {
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py:650-670: allocate_array(len, arraydescr, clear)
            let arraydescr = arraydescr
                .as_ref()
                .expect("VArrayInfo.allocate requires self.arraydescr");
            let ad = arraydescr
                .as_array_descr()
                .expect("VArrayInfo.arraydescr must be an ArrayDescr");
            let array = bh_new_array_from_descr(fieldnums.len(), arraydescr, clear);
            // resume.py:654: cache BEFORE filling — recursive/shared virtuals
            // may reference this vidx during element decoding.
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:656-670: element kind dispatch + UNINITIALIZED skip.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue; // resume.py:659: skip UNINITIALIZED
                }
                // resume.py:656-670: dispatch by arraydescr kind and pass
                // the same arraydescr through to setarrayitem_*.
                if ad.is_array_of_pointers() {
                    let value = match decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    ) {
                        Some(Value::Ref(gc)) => gc.0 as i64,
                        Some(other) => box_opt_value(&Some(other)) as i64,
                        None => 0,
                    };
                    bh_setarrayitem_ref_from_descr(array, i, value, arraydescr);
                } else if ad.is_array_of_floats() {
                    let value = decode_tagged_fieldnum_float(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    )
                    .to_bits() as i64;
                    bh_setarrayitem_float_from_descr(array, i, value, arraydescr);
                } else {
                    let value = decode_tagged_fieldnum_int(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    bh_setarrayitem_int_from_descr(array, i, value, arraydescr);
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            size,
            fielddescrs,
            item_size,
            fieldnums,
            ..
        } => {
            // resume.py:748-760: VArrayStructInfo.allocate
            let num_fields = fielddescrs.len();
            // resume.py:749: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
            // item_size from arraydescr (RPython: self.arraydescr)
            let is = arraydescr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(*item_size);
            let array = pyre_object::allocate_array_struct(*size, is);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            let result = Value::Ref(majit_ir::GcRef(array as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    // resume.py:755: num = self.fieldnums[p] — direct index; a
                    // short stream raises IndexError here (encoder bug), a
                    // longer one leaves its tail unread once p exhausts
                    // size * len(fielddescrs).
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let v = decode_tagged_fieldnum(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    if let Some(val) = v {
                        // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        let raw = match val {
                            Value::Int(i) => i,
                            Value::Float(f) => f.to_bits() as i64,
                            Value::Ref(r) => r.0 as i64,
                            Value::Void => 0,
                        };
                        let (fo, fs, ft) = extract_interior_field_info(&fielddescrs[j]);
                        pyre_object::setinteriorfield(array, i, fo, fs, is, ft, raw);
                    }
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:701-703: buffer = decoder.allocate_raw_buffer(func, size)
            let (driver, _) = driver_pair();
            // resume.py:1453-1455 allocate_raw_buffer:
            //   cic = self.callinfocollection
            //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            // The calldescr comes from the shared callinfocollection, not a
            // freshly minted MOST_GENERAL descr.  func is NOT read from the
            // callinfo (resume.py:1453 discards it as `_`; several malloc
            // variants share the oopspec) — it stays the VRawBufferInfo.func.
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VRawBufferInfo recovery (resume.py:1453)",
                )
                .clone();
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::effectinfo::OopSpecIndex::RawMallocVarsizeChar);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_RAW_MALLOC_VARSIZE_CHAR");
            let cd = calldescr
                .as_call_descr()
                .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr must downcast to CallDescr");
            let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1456: self.cpu.bh_call_i(func, [size], None, None, calldescr)
            let buffer = driver.meta_interp().backend().bh_call_i(
                *func,
                Some(&[*size as i64]),
                None,
                None,
                &bh_calldescr,
            );
            // resume.py:704: cache BEFORE filling fields.
            let result = Value::Int(buffer);
            virtuals_cache.insert(vidx, result.clone());
            let backend = driver.meta_interp().backend();
            // resume.py:705-708: for i in range(len(self.offsets)):
            //     offset = self.offsets[i]; descr = self.descrs[i]
            //     decoder.setrawbuffer_item(buffer, self.fieldnums[i], offset, descr)
            // Drive by len(self.offsets) (not fieldnums): indexing
            // descrs[i]/fieldnums[i] makes a short list an out-of-bounds error
            // (IndexError parity), a longer one leaves its tail unread.
            for i in 0..offsets.len() {
                let fnum = fieldnums[i];
                let di = &descrs[i];
                let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(di);
                // resume.py:1544: assert not descr.is_array_of_pointers()
                assert!(
                    !bh_descr.is_array_of_pointers(),
                    "raw buffer entry must not be pointer type"
                );
                let offset = offsets[i] as i64;
                // resume.py:1545-1550: descr drives decode AND store
                if di.item_type == 2 {
                    // resume.py:1546: newvalue = self.decode_float(fieldnum)
                    let fval = decode_tagged_fieldnum_float(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1547: self.cpu.bh_raw_store_f(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_f(buffer, offset, fval, &bh_descr);
                } else {
                    // resume.py:1549: newvalue = self.decode_int(fieldnum)
                    let ival = decode_tagged_fieldnum_int(
                        fnum,
                        dead_frame,
                        num_failargs,
                        rd_consts,
                        rd_virtuals,
                        virtuals_cache,
                    );
                    // resume.py:1550: self.cpu.bh_raw_store_i(buffer, offset, newvalue, descr)
                    backend.bh_raw_store_i(buffer, offset, ival, &bh_descr);
                }
            }
            return result;
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1 — a slice carries
            // exactly its base buffer; any other count is an encoder bug.
            assert!(
                fieldnums.len() == 1,
                "resume.py:724 VRawSliceInfo.allocate_int: len(self.fieldnums) == 1"
            );
            // resume.py:725: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            // resume.py:726: buffer = decoder.int_add_const(base_buffer, self.offset)
            let result = Value::Int(base + *offset as i64);
            // resume.py:727: decoder.virtuals_cache.set_int(index, buffer)
            virtuals_cache.insert(vidx, result.clone());
            return result;
        }
        majit_ir::RdVirtualInfo::Empty => {
            panic!("[jit] materialize_virtual: rd_virtuals[{vidx}] is Empty");
        }
        // resume.py:763-775 VStrPlainInfo.allocate /
        // resume.py:817-829 VUniPlainInfo.allocate —
        //     string = decoder.allocate_string(length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i, fieldnum in enumerate(self.fieldnums):
        //         if not tagged_eq(fieldnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, fieldnum)
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len() as i64;
            let (driver, _) = driver_pair();
            let backend = driver.meta_interp().backend();
            // resume.py:1449 allocate_string / resume.py:1482 allocate_unicode.
            let string = if is_unicode {
                backend.bh_newunicode(length)
            } else {
                backend.bh_newstr(length)
            };
            // resume.py:766/820 virtuals_cache.set_ptr BEFORE filling.
            let result = Value::Ref(majit_ir::GcRef(string as usize));
            virtuals_cache.insert(vidx, result.clone());
            // resume.py:771-774/824-827 per-char string_setitem loop.
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let char_val = decode_tagged_fieldnum_int(
                    fnum,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                if is_unicode {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_unicodesetitem(string, i as i64, char_val);
                } else {
                    driver
                        .meta_interp()
                        .backend()
                        .bh_strsetitem(string, i as i64, char_val);
                }
            }
            return result;
        }
        // resume.py:781-793 VStrConcatInfo.allocate /
        // resume.py:836-848 VUniConcatInfo.allocate —
        //     left  = decoder.decode_ref(self.fieldnums[0])
        //     right = decoder.decode_ref(self.fieldnums[1])
        //     string = decoder.concat_strings(left, right)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let left_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let right_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Concat recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_STR_CONCAT / OS_UNI_CONCAT");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Concat calldescr must downcast to CallDescr");
            let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1462-1470 concat_strings / resume.py:1489-1497
            // concat_unicodes — cpu.bh_call_r(func, [left, right], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                func as i64,
                None,
                Some(&[left_val, right_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        // resume.py:799-813 VStrSliceInfo.allocate /
        // resume.py:854-868 VUniSliceInfo.allocate —
        //     largerstr = decoder.decode_ref(self.fieldnums[0])
        //     start     = decoder.decode_int(self.fieldnums[1])
        //     length    = decoder.decode_int(self.fieldnums[2])
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let str_val = decode_tagged_fieldnum_int(
                fieldnums[0],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let start_val = decode_tagged_fieldnum_int(
                fieldnums[1],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            let length_val = decode_tagged_fieldnum_int(
                fieldnums[2],
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            );
            // resume.py:1474 / 1501 — slice_string(str, start, start + length)
            // passes the stop index, not the length.
            let stop_val = start_val + length_val;
            let (driver, _) = driver_pair();
            let cic = driver
                .meta_interp()
                .callinfocollection()
                .expect(
                    "materialize_virtual_from_rd: MetaInterp.callinfocollection \
                     required for VStr/VUni Slice recovery (resume.py:1143)",
                )
                .clone();
            let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
            let calldescr =
                calldescr.expect("callinfo_for_oopspec missing OS_STR_SLICE / OS_UNI_SLICE");
            let cd = calldescr
                .as_call_descr()
                .expect("VStr/VUni Slice calldescr must downcast to CallDescr");
            let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
            // resume.py:1472-1480 slice_string / resume.py:1499-1507
            // slice_unicode — cpu.bh_call_r(func, [str, start, stop], descr).
            let backend = driver.meta_interp().backend();
            let result = backend.bh_call_r(
                func as i64,
                Some(&[start_val, stop_val]),
                Some(&[str_val]),
                None,
                &bh_calldescr,
            );
            let value = Value::Ref(majit_ir::GcRef(result.0));
            virtuals_cache.insert(vidx, value.clone());
            return value;
        }
        _ => {} // Instance/Struct: fall through
    }
    // Instance/Struct: extract fields for ob_type-based materialization.
    // resume.py:593 fielddescrs + fieldnums
    enum VirtualKind<'a> {
        /// resume.py:612 VirtualInfo — allocate_with_vtable(descr=self.descr).
        Instance {
            descr: &'a Option<majit_ir::DescrRef>,
            known_class: Option<i64>,
        },
        /// resume.py:628 VStructInfo — allocate_struct(self.typedescr).
        Struct {
            typedescr: &'a Option<majit_ir::DescrRef>,
        },
    }
    let (kind, fielddescrs, fieldnums, descr_size) = match entry.as_ref() {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Instance {
                descr,
                known_class: *known_class,
            },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            descr_size,
            ..
        } => (
            VirtualKind::Struct { typedescr },
            fielddescrs.as_slice(),
            fieldnums.as_slice(),
            *descr_size,
        ),
        _ => unreachable!(),
    };

    // resume.py:617-621 VirtualInfo.allocate / resume.py:634-637 VStructInfo.allocate
    //   Phase 1: allocate (allocate_with_vtable or allocate_struct)
    //   Phase 2: virtuals_cache.set_ptr(index, struct)  ← BEFORE setfields
    //   Phase 3: self.setfields(decoder, struct)         ← fields filled AFTER

    // Phase 1: allocate.
    // A virtual is materialized empty (`intval: 0`) and its field is written by
    // Phase 3 `setfields`; a tagged immediate carries its value in the pointer
    // and cannot be mutated field-by-field, so this reconstruction path stays
    // boxed regardless of `CAN_BE_TAGGED`. Fresh int *values* are made via
    // `w_int_new` (which takes the tag path); this is not that.
    let obj_ptr: usize = match kind {
        // resume.py:617-621: VirtualInfo.allocate(descr) → allocate_with_vtable.
        VirtualKind::Instance { descr, known_class } => {
            let ob_type = known_class.unwrap_or(0);
            let int_type_addr = &pyre_object::INT_TYPE as *const _ as i64;
            let float_type_addr = &pyre_object::FLOAT_TYPE as *const _ as i64;
            if ob_type == int_type_addr {
                let tp = unsafe { &*(ob_type as *const pyre_object::pyobject::PyType) };
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: tp,
                        w_class: pyre_object::pyobject::get_instantiate(tp),
                    },
                    intval: 0,
                });
                let raw = Box::into_raw(obj) as usize;
                raw
            } else if ob_type == float_type_addr {
                let tp = unsafe { &*(ob_type as *const pyre_object::pyobject::PyType) };
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: tp,
                        w_class: pyre_object::pyobject::get_instantiate(tp),
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as usize
            } else if ob_type != 0 {
                // resume.py:619: allocate_with_vtable(descr=self.descr).
                if let Some(d) = descr {
                    allocate_with_vtable(
                        d.as_size_descr()
                            .expect("VirtualInfo descr must be SizeDescr"),
                    )
                } else {
                    // Fallback: no live descr (decoded from EncodedResumeData).
                    debug_assert!(descr_size > 0, "VirtualInfo must have descr_size");
                    let size = if descr_size > 0 { descr_size } else { 16 };
                    let fallback =
                        majit_ir::make_size_descr_with_vtable(0, size, 0, ob_type as usize);
                    allocate_with_vtable(fallback.as_size_descr().unwrap())
                }
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] materialize_virtual: vidx={vidx} Instance with no known_class",
                    );
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
        // resume.py:635: VStructInfo.allocate → allocate_struct(self.typedescr)
        VirtualKind::Struct { typedescr, .. } => {
            if let Some(td) = typedescr {
                let sd = td
                    .as_size_descr()
                    .expect("VStruct typedescr must be SizeDescr");
                allocate_struct(sd)
            } else if descr_size > 0 {
                let fallback = majit_ir::make_size_descr_full(0, descr_size, 0);
                let sd = fallback.as_size_descr().unwrap();
                allocate_struct(sd)
            } else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit] materialize_virtual: vidx={vidx} Struct with no typedescr",);
                }
                return Value::Ref(majit_ir::GcRef::NULL);
            }
        }
    };

    // Phase 2: cache REAL object pointer BEFORE setting fields.
    // resume.py:620: decoder.virtuals_cache.set_ptr(index, struct)
    let obj_ref = Value::Ref(majit_ir::GcRef(obj_ptr));
    virtuals_cache.insert(vidx, obj_ref.clone());

    // Phase 3: setfields — decode each field and write to object.
    // resume.py:596-603: for each fielddescr, decoder.setfield(struct, num, descr)
    let is_instance = matches!(kind, VirtualKind::Instance { .. });
    match kind {
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::INT_TYPE as *const _ as i64) =>
        {
            // W_IntObject fast path: find intval field.
            // fielddescrs may include ob_type (offset 0) first.
            let intval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == INT_INTVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(intval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let intval = match val {
                    Value::Int(n) => n,
                    Value::Ref(gc) if !gc.is_null() => unsafe {
                        pyre_object::intobject::w_int_get_value(gc.0 as pyre_object::PyObjectRef)
                    },
                    _ => 0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::intobject::W_IntObject)).intval = intval;
                }
            }
        }
        VirtualKind::Instance { known_class, .. }
            if known_class == Some(&pyre_object::FLOAT_TYPE as *const _ as i64) =>
        {
            // W_FloatObject fast path: find floatval field.
            let floatval_idx = fielddescrs
                .iter()
                .position(|fd| fd.offset == FLOAT_FLOATVAL_OFFSET)
                .unwrap_or(0);
            if let Some(&tagged) = fieldnums.get(floatval_idx) {
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let floatval = match val {
                    Value::Float(f) => f,
                    Value::Int(bits) => f64::from_bits(bits as u64),
                    _ => 0.0,
                };
                unsafe {
                    (*(obj_ptr as *mut pyre_object::floatobject::W_FloatObject)).floatval =
                        floatval;
                }
            }
        }
        _ => {
            // resume.py:598-602 AbstractVirtualStructInfo.setfields:
            // for each fielddescr, decoder.setfield(struct, num, descr)
            for (i, &tagged) in fieldnums.iter().enumerate() {
                if tagged == majit_ir::resumedata::NULLREF
                    || tagged == majit_ir::resumedata::UNINITIALIZED_TAG
                {
                    continue;
                }
                let val = decode_tagged_value(
                    tagged,
                    dead_frame,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                );
                let raw = match val {
                    Value::Int(n) => n,
                    Value::Float(f) => f.to_bits() as i64,
                    Value::Ref(gc) => gc.0 as i64,
                    _ => 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    debug_assert!(false, "fielddescrs missing for field {}", i);
                    continue;
                };
                // Skip vtable slot (offset 0) for Instance — already set by allocate_with_vtable.
                if descr.offset == 0 && is_instance {
                    continue;
                }
                unsafe {
                    let addr = (obj_ptr as *mut u8).add(descr.offset);
                    match descr.field_type {
                        majit_ir::Type::Ref => {
                            let p = match val {
                                Value::Ref(gc) => gc.0 as i64,
                                Value::Int(n) => n,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut i64, p);
                        }
                        majit_ir::Type::Float => {
                            let bits = match val {
                                Value::Float(f) => f.to_bits(),
                                Value::Int(n) => n as u64,
                                _ => 0,
                            };
                            std::ptr::write(addr as *mut u64, bits);
                        }
                        _ => match descr.field_size {
                            1 => std::ptr::write(addr, raw as u8),
                            2 => std::ptr::write(addr as *mut u16, raw as u16),
                            4 => std::ptr::write(addr as *mut u32, raw as u32),
                            _ => std::ptr::write(addr as *mut i64, raw),
                        },
                    }
                }
            }
        }
    }
    obj_ref
}

/// resume.py:1552-1588 ResumeDataDirectReader decode_int/decode_ref parity.
///
/// Decode a tagged value from rd_numb into a concrete Value.
/// Handles TAGBOX (deadframe), TAGINT (inline), TAGCONST (constant pool),
/// and TAGVIRTUAL (lazy materialization via materialize_virtual_from_rd).
fn decode_tagged_value(
    tagged: i16,
    dead_frame: &[Value],
    num_failargs: i32,
    rd_consts: &[majit_ir::Const],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    virtuals_cache: &mut HashMap<usize, Value>,
) -> Value {
    let (val, tagbits) = majit_metainterp::resume::untag(tagged);
    match tagbits {
        majit_metainterp::resume::TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            dead_frame.get(idx).cloned().unwrap_or(Value::Int(0))
        }
        majit_metainterp::resume::TAGINT => Value::Int(val as i64),
        majit_metainterp::resume::TAGCONST => rd_consts
            .get((val - majit_metainterp::resume::TAG_CONST_OFFSET) as usize)
            .copied()
            .unwrap_or(majit_ir::Const::Int(0))
            .to_value(),
        majit_metainterp::resume::TAGVIRTUAL => {
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num).
            // resume.py:278-284 nested virtuals are numbered negatively;
            // resolve via negative indexing into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            materialize_virtual_from_rd(
                vidx,
                dead_frame,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            )
        }
        _ => Value::Int(0),
    }
}

fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
    layout
        .exit_types
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            let raw = raw_values.get(index).copied().unwrap_or(0);
            match tp {
                majit_ir::Type::Int => Value::Int(raw),
                majit_ir::Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                majit_ir::Type::Float => Value::Float(f64::from_bits(raw as u64)),
                majit_ir::Type::Void => Value::Void,
            }
        })
        .collect()
}

/// Phase A: decode rd_numb + materialize virtuals + restore frame state.
/// RPython: this corresponds to rebuild_from_resumedata (resume.py:1042)
/// which decodes the deadframe into typed values and writes them to the
/// virtualizable/MIFrames. Returns typed values for Phase B and resume PC.
// dont_look_inside: post-trace deopt resume machinery (rebuild_from_resumedata).
#[majit_macros::dont_look_inside]
pub(crate) fn decode_and_restore_guard_failure(
    jit_state: &mut PyreJitState,
    meta: &crate::jit::state::PyreMeta,
    raw_values: &[i64],
    exit_layout: &CompiledExitLayout,
) -> Option<(Vec<Value>, usize, usize)> {
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] exit-layout trace_id={} fail_idx={} source_op={:?} rd_numb={} recovery={} resume_layout={}",
            exit_layout.trace_id,
            exit_layout.fail_index,
            exit_layout.source_op_index,
            exit_layout
                .storage
                .as_deref()
                .map(|s| s.rd_numb.len())
                .unwrap_or(0),
            exit_layout.recovery_layout.is_some(),
            exit_layout.resume_layout.is_some(),
        );
    }
    if majit_metainterp::majit_log_enabled() {
        let nraw = raw_values.len();
        let slots: Vec<String> = (0..nraw)
            .map(|i| format!("{:#x}", raw_values[i] as usize))
            .collect();
        eprintln!(
            "[jit] guard-fail: fail_idx={} types={:?} raw_len={} raw=[{}]",
            exit_layout.fail_index,
            exit_layout.exit_types,
            nraw,
            slots.join(", ")
        );
    }
    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    // resume.py:1042 rebuild_from_resumedata: decode rd_numb into typed values.
    // compile.py:853 `ResumeGuardDescr` storage — borrow rd_numb / rd_consts
    // from the guard-owned shared Arc instead of a per-guard Vec copy.
    let (typed, mut pending_virtuals_cache) = {
        let storage = exit_layout.storage.as_deref();
        let rd_numb = storage.map(|s| s.rd_numb.as_slice()).unwrap_or(&[]);
        let empty_consts: Vec<majit_ir::Const> = Vec::new();
        let rd_consts: &[majit_ir::Const] = storage.map(|s| s.rd_consts()).unwrap_or(&empty_consts);
        if rd_numb.is_empty() {
            (dead_frame_typed.clone(), HashMap::new())
        } else {
            let (t, rd_numb_pc, virtuals_cache) =
                rebuild_typed_from_rd_numb(raw_values, rd_numb, rd_consts, exit_layout);
            // blackhole.py:337 parity: setposition(jitcode, pc) before
            // consume_one_section. rd_numb_pc = orgpc used by
            // get_list_of_active_boxes during encoding.
            jit_state.resume_pc = rd_numb_pc;
            (t, virtuals_cache)
        }
    };
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] rebuilt typed prefix: {:?}",
            typed.iter().take(6).collect::<Vec<_>>()
        );
    }
    // resume.py:924-926 + 993 parity: `_prepare_next_section` already
    // materializes rd_virtuals lazily via `materialize_virtual_from_rd`.
    // Replay pending fields against the original exit slots plus that
    // shared virtual cache; do not run the legacy pyre-only
    // `recovery_layout` materialization pass here.
    replay_pending_fields(&dead_frame_typed, exit_layout, &mut pending_virtuals_cache);

    // resume.py:1042 rebuild_from_resumedata + pyjitpl.py:3400-3430
    // rebuild_state_after_failure parity: decode rd_numb to reconstruct
    // per-frame values AND write the captured virtualizable_boxes back
    // onto the physical frame via synchronize_virtualizable/write_boxes.
    // pyjitpl.py:3419-3430 — `if vinfo is not None: ... self.synchronize_virtualizable()` —
    // fires on bridge tracing entry so the tracer's subsequent
    // vable_getarrayitem_ref reads see the resume-data values, not the
    // pre-guard heap. pyre mirrors this by selecting the guard-failure
    // vable-sync mode inside `build_resumed_frames`.
    //
    // RPython parity: every guard reaching this path MUST carry rd_numb.
    // `store_final_boxes_in_guard` (optimizeopt/mod.rs:2936) populates
    // it for tracer-origin guards; backend-origin layouts propagate it
    // via `FailDescrLayout.rd_numb`. An empty
    // `rd_numb` here indicates an unported guard-emission site — hard
    // assert so the gap surfaces rather than silently degrade via a
    // pyre-only single-frame synthesis.
    let resumed_frames = {
        // compile.py:853 `ResumeGuardDescr` storage — borrow rd_numb /
        // rd_consts from the guard-owned shared Arc instead of a
        // per-guard Vec copy.
        let storage = exit_layout
            .storage
            .as_deref()
            .expect("rebuild_guard_fail_state: exit_layout.storage missing");
        assert!(
            !storage.rd_numb.is_empty(),
            "rebuild_guard_fail_state: storage.rd_numb is empty (fail_index={})",
            exit_layout.fail_index
        );
        // GuardFailureSync mode writes the captured vable boxes back onto
        // the physical frame (see comment above). The decoded frame chain
        // is also consumed below to recover the innermost frame's section
        // pc (its resume opcode), which the full-body walk does not track
        // in the vable `last_instr` field.
        build_resumed_frames(
            raw_values,
            storage.rd_numb.as_slice(),
            storage.rd_consts(),
            exit_layout,
            ResumeVableMode::GuardFailureSync,
        )
    };

    // virtualizable.py:126: write fields from resumedata to frame.
    let restored = jit_state.restore_guard_failure_values(meta, &typed, &ExceptionState::default());
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail restored: ni={} vsd={}",
            jit_state.next_instr(),
            jit_state.valuestackdepth(),
        );
    }

    if restored {
        // `next_instr()` is derived from the vable `last_instr` field.  The
        // full-body walk sets the concrete frame's `last_instr` once at the
        // loop header and does not advance it per opcode, so for a mid-body
        // guard that field — and hence `next_instr()` — carries the loop
        // header pc instead of the guard's resume opcode.  The per-frame
        // section pc (`ResumedFrame.py_pc`, the same coordinate
        // `resume_in_blackhole` resumes at) is the correct resume point.
        // Prefer it when the two disagree; for the trait tracer they always
        // match (the frame's `last_instr` tracks the Python pc), so this is
        // a no-op there. With flipped pc words, a single-frame resume uses
        // the restored vable position and a multi-frame resume uses the
        // innermost decoded section position.
        let ni = jit_state.next_instr();
        let innermost = resumed_frames.last();
        let resume_pc = if resumed_frames.len() == 1 {
            ni
        } else {
            innermost.map(|f| f.py_pc).unwrap_or(ni)
        };
        // When the resume pc is overridden to the innermost section's
        // `py_pc` (a multi-frame inlined-callee guard), the positional
        // `write_from_resume_data_partial` has left the physical frame's
        // `valuestackdepth` at the CHAIN frame's depth (the outer
        // section's).  Correct it to the innermost section's depth so the
        // interpreter does not resume at the inner pc carrying the outer
        // depth — an over-count that materializes a stray operand slot and
        // shifts every subsequent push by one (`PyFrame::push` overflow at
        // the function's peak stack use).  `last_instr` is already handled
        // via `resume_pc`; only the vsd lags.  Clear the slots above the
        // corrected depth so a GC scan before the first re-executed push
        // does not see a stale operand pointer.
        //
        // The correction must also run when a deeper inlined-callee frame is
        // present (`resumed_frames.len() > 1`) even if the innermost
        // section's `py_pc` numerically coincides with `ni`: the positional
        // vsd left by `write_from_resume_data_partial` is still the CHAIN
        // (outer) frame's depth, and the matching pc value does not make it
        // correct.  Single-frame guards keep the prior `resume_pc != ni`
        // behavior.
        if resume_pc != ni || resumed_frames.len() > 1 {
            if let Some(code) = innermost.map(|f| f.code as usize) {
                if let Some(corrected_vsd) =
                    pyre_jit_trace::state::depth_based_vsd_for_wcode(code, resume_pc)
                {
                    jit_state.set_valuestackdepth(corrected_vsd);
                    jit_state.clear_stack_above(corrected_vsd);
                }
            }
        }
        Some((typed, resume_pc, resumed_frames.len()))
    } else {
        None
    }
}

/// Decode rd_numb to produce typed values via
/// `majit_ir::resumedata::rebuild_from_numbering`. Each slot is TAGBOX
/// (deadframe), TAGCONST (constant), TAGINT (small int), or TAGVIRTUAL
/// (virtual to materialize). Consumes only the outermost frame's values,
/// but splits frames by per-jitcode liveness so the box-section boundary is
/// correct for multi-frame (inlined-callee) guards.
///
/// Returns `(typed_values, rd_numb_frame_pc)`. The frame PC from rd_numb
/// is the liveness PC used by get_list_of_active_boxes during encoding.
/// The recovery side MUST use this same PC for expand — NOT next_instr
/// (which may differ by 1+ due to cache slots).
fn rebuild_typed_from_rd_numb(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
) -> (Vec<Value>, Option<usize>, HashMap<usize, Value>) {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py:1049-1055 parity: bound each frame's box section by jitcode
    // liveness (the same per-(jitcode,pc) count the encoder used). Without it,
    // the single-frame fallback makes `frames[0]` swallow every remaining
    // item — including subsequent inline frames' headers — which is benign
    // only as long as the over-read lands on valid tagged values. This
    // function consumes only `frames.first()`, but it must still consume the
    // header word stream symmetrically so that boundary is correct for
    // multi-frame (inlined-callee) guards.
    let cb = pyre_jit_trace::state::frame_value_count_at;
    let num_virtuals = exit_layout
        .storage
        .as_deref()
        .map_or(0, |s| s.rd_virtuals.len());
    let (_num_failargs, vable_values, _vref_values, frames) = rebuild_from_numbering(
        rd_numb,
        rd_consts,
        &exit_layout.exit_types,
        Some(&cb),
        num_virtuals,
    );

    // resume.py:1045 consume_vref_and_vable_boxes parity.
    // vable_array format: [frame_ptr, ni, code, vsd, ns, locals..., stack...]
    // (opencoder.py:722 moves virtualizable_ptr to front).
    if majit_metainterp::majit_log_enabled() && !vable_values.is_empty() {
        eprintln!(
            "[jit] guard-fail: vable_values={} items: {:?}",
            vable_values.len(),
            vable_values.iter().take(6).collect::<Vec<_>>()
        );
    }

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1083 + pyjitpl.py:3400-3428 parity:
    // Decode vable_values into typed prefix [frame_ptr, ni, code, vsd, ns, locals..., stack...].
    // In RPython, virtualizable_boxes are restored first, then synchronize_virtualizable
    // writes them back to the actual frame object.
    fn decode_rv(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py:1042-1057 rebuild_from_resumedata parity:
    // RPython produces TWO streams:
    //   1. virtualizable_boxes (consume_vref_and_vable → synchronize_virtualizable)
    //   2. frame registers (consume_boxes per frame)
    // pyjitpl.py:3419-3430: virtualizable_boxes restored, then
    // synchronize_virtualizable writes them back to the heap.
    // Frame registers fill frame.registers_i/r/f independently.

    // `vable_values` is heap-layout (opencoder.py:718 `_list_of_boxes_virtualizable`):
    //   [frame_ptr, vable_static_fields..., array_items...]
    // `_list_of_boxes_virtualizable` excludes any reds that are not virtualizable
    // static fields (ec is a per-thread global), so the encoded prefix has
    // `1 + NUM_VABLE_SCALARS` entries — never `NUM_SCALAR_INPUTARGS`, which counts
    // `NUM_EXTRA_REDS` ec slot(s) on the trace inputarg side.
    //
    // `restore_guard_failure_values` and downstream consumers index this header
    // with `SYM_*_IDX`, which include the `NUM_EXTRA_REDS` shift. Inject placeholder
    // ec slot(s) between the frame and the static fields here so the trace-layout
    // indices align. The ec value itself is never written back (ec is reloaded from
    // `get_execution_context()` on resume), so a `Value::Void` placeholder is safe.
    let num_scalars = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    let num_extra_reds = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let heap_scalar_count = 1 + pyre_jit_trace::virtualizable_gen::NUM_VABLE_SCALARS;
    let header: Vec<Value> = if vable_values.len() >= heap_scalar_count {
        let mut h = Vec::with_capacity(num_scalars);
        h.push(decode_rv(
            &vable_values[0],
            &dead_frame_typed,
            exit_layout,
            &mut virtuals_cache,
        ));
        for _ in 0..num_extra_reds {
            h.push(Value::Void);
        }
        for i in 1..heap_scalar_count {
            h.push(decode_rv(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            ));
        }
        h
    } else {
        Vec::new()
    };

    // resume.py:1049-1056: rebuild_from_resumedata iterates all frames
    // via newframe()+consume_boxes(). For guard-failure restore into the
    // outer pyre interpreter state (restore_guard_failure_values), only
    // the JIT-entry frame's values are needed; the decoded inner frames
    // are unused here (build_resumed_frames runs only for its vable-sync
    // side effect on the guard-failure path).
    // After `opencoder.py:217` `framestack.reverse()` parity (encoder at
    // `trace_opcode.rs::build_framestack_snapshot`) `frames[0]` is the
    // outermost (caller / JIT-driver) frame, so `frames.first()` is the
    // restoration target for both single- and multi-frame guards.
    let mut typed = header;
    if let Some(outermost) = frames.first() {
        _prepare_next_section(
            outermost,
            &dead_frame_typed,
            exit_layout,
            &mut typed,
            &mut virtuals_cache,
        );
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] guard-fail: rd_numb decoded {} slots from {} frame(s)",
            typed.len(),
            frames.len(),
        );
    }

    // The outer frame's decoded Python position is retained for resume-state
    // hygiene. The live resume selection uses the rebuilt frame chain.
    // pc=-1 = no-snapshot sentinel; screen it out (as build_resumed_frames does)
    // so the negative word never reaches the `as usize` cast.
    let rd_numb_pc = frames
        .first()
        .filter(|f| f.pc >= 0)
        .map(|f| pyre_jit_trace::state::backxlat_py_pc(f.jitcode_index, f.pc) as usize);
    (typed, rd_numb_pc, virtuals_cache)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResumeVableMode {
    GuardFailureSync,
}

fn value_to_static_vable_bits(value: &Value, expected_type: Type, field_index: usize) -> i64 {
    match (expected_type, value) {
        (Type::Int, Value::Int(v)) => *v,
        (Type::Float, Value::Float(v)) => v.to_bits() as i64,
        (Type::Ref, Value::Ref(r)) => r.as_usize() as i64,
        (ty, other) => {
            panic!("virtualizable static field {field_index} expected {ty:?}, got {other:?}")
        }
    }
}

fn value_to_vable_array_item_bits(
    value: &Value,
    expected_type: Type,
    array_index: usize,
    item_index: usize,
) -> i64 {
    match expected_type {
        Type::Ref => match value {
            Value::Ref(r) => r.as_usize() as i64,
            Value::Int(i) => pyre_object::intobject::w_int_new(*i) as i64,
            Value::Float(f) => pyre_object::floatobject::w_float_new(*f) as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Ref, got {other:?}"
            ),
        },
        Type::Int => match value {
            Value::Int(v) => *v,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Int, got {other:?}"
            ),
        },
        Type::Float => match value {
            Value::Float(v) => v.to_bits() as i64,
            other => panic!(
                "virtualizable array item [{array_index}][{item_index}] expected Float, got {other:?}"
            ),
        },
        ty => {
            panic!("virtualizable array item [{array_index}][{item_index}] unsupported type {ty:?}")
        }
    }
}

fn value_to_vable_identity_bits(value: &Value) -> i64 {
    match value {
        Value::Ref(r) => r.as_usize() as i64,
        other => panic!("virtualizable identity expected Ref, got {other:?}"),
    }
}

fn sync_virtualizable_after_guard_failure(
    resolved_vable: &[Value],
    frame_u8: *mut u8,
    vinfo: &majit_metainterp::virtualizable::VirtualizableInfo,
) {
    unsafe {
        // pyjitpl.py:3427-3429: reset token before synchronize_virtualizable().
        vinfo.reset_vable_token(frame_u8);
    }
    let expected_total_without_identity = vinfo.num_static_extra_boxes
        + (0..vinfo.array_fields.len())
            .map(|array_index| unsafe {
                vinfo.get_array_length(frame_u8.cast_const(), array_index)
            })
            .sum::<usize>();
    assert_eq!(
        resolved_vable.len(),
        expected_total_without_identity + 1,
        "rebuild_guard_fail_state: virtualizable box count mismatch (expected {}, got {})",
        expected_total_without_identity + 1,
        resolved_vable.len(),
    );

    let mut boxes: Vec<i64> = Vec::with_capacity(expected_total_without_identity + 1);
    let mut cursor = 1;
    for (field_index, field) in vinfo.static_fields.iter().enumerate() {
        boxes.push(value_to_static_vable_bits(
            &resolved_vable[cursor],
            field.field_type,
            field_index,
        ));
        cursor += 1;
    }
    for (array_index, array_field) in vinfo.array_fields.iter().enumerate() {
        let array_len = unsafe { vinfo.get_array_length(frame_u8.cast_const(), array_index) };
        for item_index in 0..array_len {
            boxes.push(value_to_vable_array_item_bits(
                &resolved_vable[cursor],
                array_field.item_type,
                array_index,
                item_index,
            ));
            cursor += 1;
        }
    }
    debug_assert_eq!(cursor, resolved_vable.len());
    boxes.push(value_to_vable_identity_bits(&resolved_vable[0]));

    unsafe {
        vinfo.write_boxes_to_heap(frame_u8, &boxes);
    }
}

/// Decode rd_numb into per-frame ResumedFrame chain via
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// Single-frame only (RPython's blackhole_from_resumedata uses
/// per-jitcode liveness for multi-frame decode).
fn build_resumed_frames(
    raw_values: &[i64],
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    exit_layout: &CompiledExitLayout,
    vable_mode: ResumeVableMode,
) -> Vec<crate::call_jit::ResumedFrame> {
    use majit_ir::resumedata::rebuild_from_numbering;

    // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
    // RPython uses jitcode liveness (jitcode.position_info) to know how many
    // boxes each frame contributes. There is no out-of-band frame size — the
    // decoder reads jitcode liveness at the frame's resume pc.
    let cb = pyre_jit_trace::state::frame_value_count_at;
    let num_virtuals = exit_layout
        .storage
        .as_deref()
        .map_or(0, |s| s.rd_virtuals.len());
    let (_num_failargs, vable_values, _vref_values, frames) = rebuild_from_numbering(
        rd_numb,
        rd_consts,
        &exit_layout.exit_types,
        Some(&cb),
        num_virtuals,
    );

    let dead_frame_typed = decode_exit_layout_values(raw_values, exit_layout);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume] exit_types={:?} dead_frame={:?} vable={} frames={}",
            exit_layout.exit_types,
            dead_frame_typed,
            vable_values.len(),
            frames.len()
        );
    }
    let mut virtuals_cache: HashMap<usize, Value> = HashMap::new();

    // resume.py:1045 consume_vref_and_vable parity:
    // Reconstruct header [frame_ptr, ni, code, vsd, ns] from vable_values.
    fn resolve_rebuilt_value(
        rv: &majit_ir::resumedata::RebuiltValue,
        dead_frame_typed: &[Value],
        exit_layout: &CompiledExitLayout,
        virtuals_cache: &mut HashMap<usize, Value>,
    ) -> Value {
        use majit_ir::resumedata::RebuiltValue;
        match rv {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            RebuiltValue::Virtual(vidx) => {
                let storage = exit_layout.storage.as_deref();
                let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
                let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
                materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    exit_layout.exit_types.len() as i32,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )
            }
            _ => Value::Int(0),
        }
    }
    // resume.py:1045 consume_vref_and_vable: vable header is extracted
    // AFTER _prepare_next_section materializes virtuals. The post-section
    // block below is the authoritative extraction. vable_values is always
    // non-empty for guards with complete resume data (resume.py:397 asserts
    // resume_position >= 0). The no-snapshot fallback in store_final_boxes_in_guard
    // now encodes fail_args[0..3] as vable_array to maintain this invariant.

    let mut all_values: Vec<Vec<Value>> = Vec::with_capacity(frames.len());
    for (fidx, frame) in frames.iter().enumerate() {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] _prepare_next_section frame={}/{} pc={} values_len={}",
                fidx,
                frames.len(),
                frame.pc,
                frame.values.len()
            );
        }
        let mut values = Vec::new();
        _prepare_next_section(
            frame,
            &dead_frame_typed,
            exit_layout,
            &mut values,
            &mut virtuals_cache,
        );
        all_values.push(values);
    }
    // RPython parity: _prepare_next_section + materialize_virtual_from_rd
    // is the authoritative path for virtual materialization.
    // Pending-field replay must consume the same deadframe slots and shared
    // virtual cache; the legacy pyre-only recovery_layout materializer has
    // been removed.
    // resume.py:993 _prepare_pendingfields: apply ONCE for the whole reader.
    // No header — values = slot registers only.
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[dynasm-debug] before replay_pending_fields, frames={}",
            all_values.len()
        );
    }
    replay_pending_fields(&dead_frame_typed, exit_layout, &mut virtuals_cache);
    if majit_metainterp::majit_log_enabled() {
        eprintln!("[dynasm-debug] after replay_pending_fields");
    }

    // opencoder.py:722 _list_of_boxes_virtualizable: snapshot reorders
    // virtualizable_ptr from end to front. `vable_values` is heap-layout
    // (no ec): [frame, vable_static_fields..., array_items...]. SYM_*_IDX
    // include the `NUM_EXTRA_REDS` shift for trace inputarg layout, so
    // subtract `NUM_EXTRA_REDS` to land on heap-layout positions.
    let extra = pyre_jit_trace::virtualizable_gen::NUM_EXTRA_REDS;
    let ni_idx = pyre_jit_trace::virtualizable_gen::SYM_LAST_INSTR_IDX as usize - extra;
    let code_idx = pyre_jit_trace::virtualizable_gen::SYM_PYCODE_IDX as usize - extra;
    let vsd_idx = pyre_jit_trace::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize - extra;
    let ns_idx = pyre_jit_trace::virtualizable_gen::SYM_W_GLOBALS_IDX as usize - extra;

    // Resolve ALL vable fields from resume data.
    // vable_values = [frame_ptr(0), last_instr(1), pycode(2),
    //                  valuestackdepth(3), debugdata(4),
    //                  lastblock(5), w_globals(6), array...]
    // RPython reader.load_next_value_of_type reads ALL values sequentially.
    let resolved_vable: Vec<Value> = (0..vable_values.len())
        .map(|i| {
            resolve_rebuilt_value(
                &vable_values[i],
                &dead_frame_typed,
                exit_layout,
                &mut virtuals_cache,
            )
        })
        .collect();
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][resume][vable-values] rebuilt={:?} resolved={:?}",
            vable_values, resolved_vable
        );
    }

    let vable_frame_ptr = resolved_vable
        .first()
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *mut pyre_interpreter::pyframe::PyFrame,
            Value::Int(v) => *v as *mut pyre_interpreter::pyframe::PyFrame,
            _ => std::ptr::null_mut(),
        })
        .unwrap_or(std::ptr::null_mut());
    let vable_ni = resolved_vable
        .get(ni_idx)
        .map(|v| match v {
            Value::Int(v) => (*v + 1) as usize,
            _ => 0,
        })
        .unwrap_or(0);

    let vable_vsd = resolved_vable
        .get(vsd_idx)
        .map(|v| match v {
            Value::Int(v) => *v as usize,
            _ => 0,
        })
        .unwrap_or(0);

    // virtualizable.py:86-99 read_boxes: ALL static fields in declared order.
    let vable_pycode: *const () = resolved_vable
        .get(code_idx)
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *const (),
            Value::Int(v) => *v as *const (),
            _ => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null());

    let vable_ns: *const () = resolved_vable
        .get(ns_idx)
        .map(|v| match v {
            Value::Ref(r) => r.as_usize() as *const (),
            Value::Int(v) => *v as *const (),
            _ => std::ptr::null(),
        })
        .unwrap_or(std::ptr::null());

    // pyjitpl.py:3419-3430 synchronize_virtualizable on guard-failure
    // bridge entry: stores `self.virtualizable_boxes`, resets the token,
    // then calls `self.synchronize_virtualizable()` which ends at
    // virtualizable.py:101-113 `write_boxes`. `ResumeVableMode::GuardFailureSync`
    // models that path: it writes the captured vable boxes back onto the
    // physical frame so the tracer's subsequent vable reads see the
    // resume-data values, not the pre-guard heap. (The blackhole resume
    // path performs its own consume_vable_info write inside
    // `blackhole_resume_via_rd_numb` (resume.py:1399-1408).)
    if !vable_frame_ptr.is_null() {
        let frame_u8 = vable_frame_ptr as *mut u8;
        // resume.py:1312-1314 blackhole_from_resumedata parity:
        //     vinfo = self.jitdriver_sd.virtualizable_info
        // Use the JIT driver's cached `Arc<VirtualizableInfo>` set once by
        // `set_virtualizable_info` at JIT_DRIVER init rather than rebuilding
        // a fresh instance, so the guard-failure recovery path shares a
        // single vinfo identity with the tracing / blackhole consumers.
        let vinfo = crate::eval::driver_pair().1.clone();
        match vable_mode {
            ResumeVableMode::GuardFailureSync => {
                sync_virtualizable_after_guard_failure(&resolved_vable, frame_u8, &vinfo);
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][resume][vable-sync] mode={vable_mode:?} frame {:?}",
                vable_frame_ptr
            );
            if !vable_frame_ptr.is_null() {
                let f = unsafe { &*vable_frame_ptr };
                eprintln!(
                    "[jit][resume][vable-sync] frame after write: ni={} vsd={} code={:?} ns={:?} debugdata={:?} lastblock={:?} vable_token={} array_len={}",
                    f.next_instr(),
                    f.valuestackdepth,
                    f.pycode,
                    f.w_globals,
                    f.debugdata,
                    f.lastblock,
                    f.vable_token,
                    f.locals_w().len(),
                );
            }
        }
    }

    let mut result = Vec::with_capacity(frames.len());
    for (idx, (frame, values)) in frames.iter().zip(all_values.into_iter()).enumerate() {
        // pc=0 is valid (function start). pc=-1 = no-snapshot sentinel.
        let decoded_py_pc = (frame.pc >= 0)
            .then(|| pyre_jit_trace::state::backxlat_py_pc(frame.jitcode_index, frame.pc) as usize);
        let py_pc = decoded_py_pc.unwrap_or(vable_ni);
        // resume.py:1339 jitcodes[jitcode_pos]:
        // Outermost frame: code from vable resume data.
        // Inner frames: code from jitcode_index registry (inlined calls).
        // After `opencoder.py:217` `framestack.reverse()` parity (encoder at
        // `trace_opcode.rs::build_framestack_snapshot`), `frames[0]` is the
        // outermost (caller / JIT-driver) frame and the last entry is the
        // innermost (deepest callee).
        let is_outermost = idx == 0;
        let w_code = if is_outermost {
            // virtualizable.py:86-99: code from resume data, not heap.
            if !vable_pycode.is_null() {
                vable_pycode
            } else if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).pycode }
            } else {
                std::ptr::null()
            }
        } else {
            pyre_jit_trace::state::code_for_jitcode_index(frame.jitcode_index)
                .unwrap_or(std::ptr::null())
        };
        let raw_code = if !w_code.is_null() {
            unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            }
        } else {
            std::ptr::null()
        };
        // resume.py:1095/1067 parity: virtualizable_ptr is the JIT driver's
        // PyFrame, shared across the entire blackhole chain. RPython's
        // newframe(jitcode) creates a fresh MIFrame for each inner section
        // (inlined call) but the virtualizable on the JIT driver is the same.
        // pyre's BlackholeInterpreter holds its own register state per
        // section, so inner frames don't need a PyFrame — they only need
        // virtualizable_ptr to write back to the outermost frame at the
        // merge point. Use vable_frame_ptr for ALL sections.
        let frame_ptr = vable_frame_ptr;
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[dynasm-debug] frame_ptr={:?} py_pc={} idx={}",
                frame_ptr, py_pc, idx
            );
        }
        // Per-frame VSD: outermost uses vable_vsd, inner frames derive
        // from their code's nlocals + snapshot stack depth.
        let vsd = if is_outermost {
            // resume.py:1399 parity: outermost frame's vsd comes directly
            // from the virtualizable. RPython does not sentinel-check 0.
            vable_vsd
        } else if !raw_code.is_null() {
            let nlocals = unsafe { &*raw_code }.varnames.len();
            nlocals + values.len().saturating_sub(nlocals)
        } else {
            values.len()
        };
        // virtualizable.py:86-99: namespace from resume data.
        let namespace = if is_outermost {
            if !vable_ns.is_null() {
                vable_ns
            } else if !vable_frame_ptr.is_null() {
                unsafe { (*vable_frame_ptr).w_globals as *const () }
            } else {
                std::ptr::null()
            }
        } else {
            // Inner frames share the chain virtualizable's namespace.
            vable_ns
        };
        result.push(crate::call_jit::ResumedFrame {
            code: w_code,
            py_pc,
            rd_numb_pc: decoded_py_pc,
            frame_ptr,
            vsd,
            namespace,
            values,
        });
    }

    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit] build_resumed_frames: {} frame(s) from rd_numb",
            result.len()
        );
    }

    result
}

/// resume.py:1017-1026 _prepare_next_section: decode one frame's slots
/// from rd_numb tagged values into typed Value vector.
fn _prepare_next_section(
    frame: &majit_ir::resumedata::RebuiltFrame,
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    typed: &mut Vec<Value>,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    use majit_ir::resumedata::RebuiltValue;
    let storage = exit_layout.storage.as_deref();
    let rd_consts = storage.map(|s| s.rd_consts()).unwrap_or(&[]);
    let rd_virtuals = storage.map(|s| s.rd_virtuals.as_slice());
    let num_failargs = exit_layout.exit_types.len() as i32;
    for val in &frame.values {
        typed.push(match val {
            RebuiltValue::Box(idx, _tp) => {
                dead_frame_typed.get(*idx).cloned().unwrap_or(Value::Int(0))
            }
            // history.py:220-360 Const → Value: direct variant projection.
            RebuiltValue::Const(c) => c.to_value(),
            // resume.py:1572: decode_ref(TAGVIRTUAL) → getvirtual_ptr(num)
            RebuiltValue::Virtual(vidx) => materialize_virtual_from_rd(
                *vidx,
                dead_frame_typed,
                num_failargs,
                rd_consts,
                rd_virtuals,
                virtuals_cache,
            ),
            // resume.py:131 UNINITIALIZED parity: dead/uninitialized slots
            // stay at default. In pyre, PY_NULL via Value::Void.
            RebuiltValue::Unassigned => Value::Void,
        });
    }
}

// `cranelift_resumedata_deopt` lives in `call_jit.rs` so it stays
// outside `pyre-jit-trace`'s build-script translator file set
// (build.rs:66 reads pyre-jit/src/eval.rs verbatim; `eval.rs` must
// remain expressible in the translator's RPython subset, which the
// downcast-driven on-demand decode implementation is not).

/// virtual's slot to NONE and appends field values (ob_type, intval).
/// On guard failure, we detect contiguous null Ref slots at the end
/// of the locals/stack region and pair them with trailing Int fields.
///
/// resume.py:993-1007 _prepare_pendingfields: replay deferred field writes.
///
/// After virtual materialization, pending SETFIELD_GC/SETARRAYITEM_GC
/// ops stored in rd_pendingfields are replayed on the materialized objects.
/// This ensures lazy field writes that were deferred during optimization
/// take effect when the guard fires.
fn replay_pending_fields(
    dead_frame_typed: &[Value],
    exit_layout: &CompiledExitLayout,
    virtuals_cache: &mut HashMap<usize, Value>,
) {
    let Some(ref recovery) = exit_layout.recovery_layout else {
        return;
    };
    if recovery.pending_field_layouts.is_empty() {
        return;
    }

    let empty_consts: Vec<majit_ir::Const> = Vec::new();
    let rd_consts: &[majit_ir::Const] = exit_layout
        .storage
        .as_deref()
        .map(|s| s.rd_consts())
        .unwrap_or(&empty_consts);
    let rd_virtuals = exit_layout
        .storage
        .as_deref()
        .map(|s| s.rd_virtuals.as_slice());
    let num_failargs = exit_layout.exit_types.len() as i32;
    let value_to_raw_bits = |value: Value| match value {
        Value::Int(i) => i,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.0 as i64,
        Value::Void => 0,
    };
    let mut resolve_value = |src: &majit_backend::ExitValueSourceLayout| -> Option<i64> {
        match src {
            majit_backend::ExitValueSourceLayout::ExitValue(idx) => {
                dead_frame_typed.get(*idx).cloned().map(value_to_raw_bits)
            }
            majit_backend::ExitValueSourceLayout::Constant(c, _) => Some(*c),
            majit_backend::ExitValueSourceLayout::Virtual(vidx) => {
                Some(value_to_raw_bits(materialize_virtual_from_rd(
                    *vidx,
                    dead_frame_typed,
                    num_failargs,
                    rd_consts,
                    rd_virtuals,
                    virtuals_cache,
                )))
            }
            majit_backend::ExitValueSourceLayout::Uninitialized
            | majit_backend::ExitValueSourceLayout::Unavailable => None,
        }
    };

    for pf in &recovery.pending_field_layouts {
        let Some(target_ptr) = resolve_value(&pf.target) else {
            continue;
        };
        let Some(value_raw) = resolve_value(&pf.value) else {
            continue;
        };
        if target_ptr == 0 {
            continue; // null target — skip
        }
        // resume.py:1000 PENDINGFIELDSTRUCT.lldescr is always present in
        // RPython — captured directly off the Setfield_gc / Setarrayitem_gc op
        // that produced the pending field (heap.py force_lazy_sets_for_guard).
        let descr = pf
            .descr
            .as_ref()
            .expect("resume.py:1000 PENDINGFIELDSTRUCT.lldescr must be set");
        // resume.py:1003-1007 _prepare_pendingfields:
        //   if itemindex < 0: setfield(struct, fieldnum, descr)
        //   else:             setarrayitem(struct, itemindex, fieldnum, descr)
        //
        // resume.py:1509-1518 setfield: descr.is_pointer_field()
        //   → bh_setfield_gc_r; is_float_field() → bh_setfield_gc_f;
        //   else → bh_setfield_gc_i.
        // resume.py:1531-1541 setarrayitem_{int,ref,float}: dispatched by
        //   resume.py:1009-1014 setarrayitem via arraydescr.is_array_of_pointers
        //   / is_array_of_floats.
        let (addr, value_type, value_size) = if pf.is_array_item {
            let ad = descr
                .as_array_descr()
                .expect("setarrayitem pending field must carry an ArrayDescr");
            let item_index = pf.item_index.unwrap_or(0);
            let addr = target_ptr as usize + ad.base_size() + item_index * ad.item_size();
            (addr, ad.item_type(), ad.item_size())
        } else {
            let fd = descr
                .as_field_descr()
                .expect("setfield pending field must carry a FieldDescr");
            let addr = target_ptr as usize + fd.offset();
            (addr, fd.field_type(), fd.field_size())
        };
        unsafe {
            match value_type {
                majit_ir::Type::Ref => {
                    // bh_setfield_gc_r / bh_setarrayitem_gc_r: store pointer.
                    // Emit the write barrier on the target object so a young
                    // ref stored into an existing old object is tracked by
                    // the next minor collection (`rd_pendingfields` can
                    // target pre-existing deadframe objects).
                    majit_gc::gc_write_barrier(majit_ir::GcRef(target_ptr as usize));
                    std::ptr::write(addr as *mut usize, value_raw as usize);
                }
                majit_ir::Type::Float => {
                    // bh_setfield_gc_f / bh_setarrayitem_gc_f: store f64.
                    std::ptr::write(addr as *mut u64, value_raw as u64);
                }
                majit_ir::Type::Int | majit_ir::Type::Void => {
                    // bh_setfield_gc_i / bh_setarrayitem_gc_i: size-aware int.
                    match value_size {
                        8 => std::ptr::write(addr as *mut i64, value_raw),
                        4 => std::ptr::write(addr as *mut i32, value_raw as i32),
                        2 => std::ptr::write(addr as *mut i16, value_raw as i16),
                        1 => std::ptr::write(addr as *mut u8, value_raw as u8),
                        _ => std::ptr::write(addr as *mut i64, value_raw),
                    }
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit] replay_pending_field: type={:?} size={} target={:#x} value={:#x}",
                value_type, value_size, target_ptr as usize, value_raw as usize
            );
        }
    }
}

// dont_look_inside: JIT-state construction machinery the tracer must not enter.
#[majit_macros::dont_look_inside]
pub(crate) fn build_jit_state(
    frame: &PyFrame,
    virtualizable_info: &majit_metainterp::virtualizable::VirtualizableInfo,
) -> PyreJitState {
    let mut jit_state = PyreJitState {
        frame: frame as *const PyFrame as usize,
        resume_pc: None,
    };
    assert!(
        jit_state.sync_from_virtualizable(virtualizable_info),
        "build_jit_state: frame must be a valid PyFrame with readable fields"
    );
    jit_state
}

/// Extract (field_offset, field_size, field_type_code) from a live InteriorFieldDescr.
/// field_type_code: 0=ref, 1=int, 2=float — matches pyre_object::setinteriorfield API.
fn extract_interior_field_info(descr: &majit_ir::DescrRef) -> (usize, usize, u8) {
    if let Some(ifd) = descr.as_interior_field_descr() {
        let fld = ifd.field_descr();
        let ft = if fld.is_pointer_field() {
            0u8
        } else if fld.is_float_field() {
            2u8
        } else {
            1u8
        };
        (fld.offset(), fld.field_size(), ft)
    } else {
        (0, 8, 1)
    }
}

/// resume.py:1437-1541 — BlackholeAllocator for pyre's object model.
///
/// Used by ResumeDataDirectReader during guard failure blackhole resume
/// to allocate virtual objects and replay pending field writes.
/// RPython delegates to self.cpu (metainterp_sd.cpu) for allocation.
pub(crate) struct PyreBlackholeAllocator;

/// `resume.py:1509-1518 setfield(struct, fieldnum, descr)` byte-write
/// helper for integer and float fields. Ref fields use a pointer-width
/// store in `bh_setfield_gc_r`, matching `llmodel.py:723`.
fn bh_setfield_gc_byte_write(struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
    let field_offset = descr_info.offset;
    if struct_ptr == 0 {
        return;
    }
    unsafe {
        let ptr = (struct_ptr as *mut u8).add(field_offset);
        match descr_info.field_size {
            8 => (ptr as *mut i64).write(value),
            4 => (ptr as *mut i32).write(value as i32),
            2 => (ptr as *mut i16).write(value as i16),
            1 => ptr.write(value as u8),
            _ => (ptr as *mut i64).write(value),
        }
    }
}

const LOWLEVEL_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
const LOWLEVEL_STRING_CHARS_OFFSET: usize = 2 * std::mem::size_of::<usize>();
const LOWLEVEL_STR_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET + 1;
const LOWLEVEL_UNICODE_BASE_SIZE: usize = LOWLEVEL_STRING_CHARS_OFFSET;

fn bh_alloc_lowlevel_string(length: usize, base_size: usize, item_size: usize) -> i64 {
    let Some(items_size) = length.checked_mul(item_size) else {
        return 0;
    };
    let Some(total_size) = base_size.checked_add(items_size) else {
        return 0;
    };
    let layout = std::alloc::Layout::from_size_align(total_size, std::mem::align_of::<usize>())
        .expect("low-level string layout");
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        (ptr.add(LOWLEVEL_STRING_LEN_OFFSET) as *mut usize).write(length);
    }
    ptr as i64
}

fn bh_lowlevel_string_len(string: i64) -> usize {
    if string == 0 {
        return 0;
    }
    unsafe { *((string as *const u8).add(LOWLEVEL_STRING_LEN_OFFSET) as *const usize) }
}

fn bh_lowlevel_chars_offset(item_size: usize) -> usize {
    if item_size == 1 {
        LOWLEVEL_STR_BASE_SIZE - 1
    } else {
        LOWLEVEL_UNICODE_BASE_SIZE
    }
}

fn bh_read_lowlevel_string(string: i64, item_size: usize) -> Vec<i64> {
    let len = bh_lowlevel_string_len(string);
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    let mut chars = Vec::with_capacity(len);
    for index in 0..len {
        let addr = unsafe { (string as *const u8).add(chars_offset + index * item_size) };
        let value = unsafe {
            match item_size {
                1 => *addr as i64,
                4 => *(addr as *const u32) as i64,
                _ => *(addr as *const i64),
            }
        };
        chars.push(value);
    }
    chars
}

fn bh_write_lowlevel_char(string: i64, index: usize, char: i64, item_size: usize) {
    if string == 0 {
        return;
    }
    let chars_offset = bh_lowlevel_chars_offset(item_size);
    unsafe {
        let addr = (string as *mut u8).add(chars_offset + index * item_size);
        match item_size {
            1 => addr.write(char as u8),
            4 => (addr as *mut u32).write(char as u32),
            _ => (addr as *mut i64).write(char),
        }
    }
}

fn bh_concat_lowlevel_strings(left: i64, right: i64, item_size: usize) -> i64 {
    let mut chars = bh_read_lowlevel_string(left, item_size);
    chars.extend(bh_read_lowlevel_string(right, item_size));
    let (base_size, item_size) = if item_size == 1 {
        (LOWLEVEL_STR_BASE_SIZE, 1)
    } else {
        (LOWLEVEL_UNICODE_BASE_SIZE, 4)
    };
    let result = bh_alloc_lowlevel_string(chars.len(), base_size, item_size);
    for (index, char) in chars.into_iter().enumerate() {
        bh_write_lowlevel_char(result, index, char, item_size);
    }
    result
}

fn bh_slice_lowlevel_string(string: i64, start: i64, stop: i64, item_size: usize) -> i64 {
    let chars = bh_read_lowlevel_string(string, item_size);
    let len = chars.len();
    let start = start.clamp(0, len as i64) as usize;
    let stop = stop.clamp(start as i64, len as i64) as usize;
    let slice = &chars[start..stop];
    let (base_size, item_size) = if item_size == 1 {
        (LOWLEVEL_STR_BASE_SIZE, 1)
    } else {
        (LOWLEVEL_UNICODE_BASE_SIZE, 4)
    };
    let result = bh_alloc_lowlevel_string(slice.len(), base_size, item_size);
    for (index, char) in slice.iter().copied().enumerate() {
        bh_write_lowlevel_char(result, index, char, item_size);
    }
    result
}

impl majit_metainterp::resume::BlackholeAllocator for PyreBlackholeAllocator {
    fn bh_new_array_clear(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1446 cpu.bh_new_array_clear(length, arraydescr)
        bh_new_array_from_descr(length, arraydescr, /* clear */ true)
    }

    fn bh_new_array(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1447 cpu.bh_new_array(length, arraydescr)
        bh_new_array_from_descr(length, arraydescr, /* clear */ false)
    }

    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        // resume.py:1442 cpu.bh_new(typedescr)
        // llmodel.py:775-776 bh_new(sizedescr): plain malloc, no vtable.
        let sd = typedescr
            .as_size_descr()
            .expect("allocate_struct: not a SizeDescr");
        let bh_descr = majit_translate::jitcode::BhDescr::Size {
            size: sd.size(),
            // `descr.py:108-118` cache identity via `SizeDescr.cache_key()`.
            type_id: sd.cache_key(),
            vtable: 0,
            owner: String::new(),
            all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(sd),
            is_gc_managed: sd.is_gc_managed(),
        };
        let (driver, _) = driver_pair();
        driver.meta_interp().backend().bh_new(&bh_descr)
    }

    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        // resume.py:1437-1439 allocate_with_vtable →
        //   exec_new_with_vtable(self.cpu, descr)
        // llmodel.py:778-782 bh_new_with_vtable: allocate AND set vtable.
        use pyre_jit_trace::descr::{W_FLOAT_GC_TYPE_ID, W_INT_GC_TYPE_ID};
        let sd = descr
            .as_size_descr()
            .expect("allocate_with_vtable: not a SizeDescr");
        let descr_index = sd.type_id();
        let descr_size = sd.size();
        match descr_index {
            W_INT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::intobject::W_IntObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::INT_TYPE as *const _,
                        w_class: pyre_object::pyobject::get_instantiate(
                            &pyre_object::pyobject::INT_TYPE,
                        ),
                    },
                    intval: 0,
                });
                let raw = Box::into_raw(obj) as i64;
                raw
            }
            W_FLOAT_GC_TYPE_ID => {
                let obj = Box::new(pyre_object::floatobject::W_FloatObject {
                    ob_header: pyre_object::pyobject::PyObject {
                        ob_type: &pyre_object::pyobject::FLOAT_TYPE as *const _,
                        w_class: pyre_object::pyobject::get_instantiate(
                            &pyre_object::pyobject::FLOAT_TYPE,
                        ),
                    },
                    floatval: 0.0,
                });
                Box::into_raw(obj) as i64
            }
            _ => {
                let bh_descr = majit_translate::jitcode::BhDescr::Size {
                    size: descr_size,
                    // Note: u32 gc tid widened to u64 cache key slot.
                    type_id: descr_index as u64,
                    vtable: vtable as u64,
                    owner: String::new(),
                    all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(sd),
                    is_gc_managed: sd.is_gc_managed(),
                };
                let (driver, _) = driver_pair();
                driver.meta_interp().backend().bh_new_with_vtable(&bh_descr)
            }
        }
    }

    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        bh_setfield_gc_byte_write(struct_ptr, value, descr_info);
    }

    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        unsafe {
            ((struct_ptr as *mut u8).add(descr_info.offset) as *mut usize).write(value as usize);
        }
    }

    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        bh_setfield_gc_byte_write(struct_ptr, value, descr_info);
    }

    fn bh_setarrayitem_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_int_from_descr(array, index, value, descr);
    }

    fn bh_setarrayitem_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_ref_from_descr(array, index, value, descr);
    }

    fn bh_setarrayitem_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        bh_setarrayitem_float_from_descr(array, index, value, descr);
    }

    // resume.py:1520-1529: setinteriorfield dispatch by descr
    // llmodel.py:648-665: bh_setinteriorfield_gc_{i,r,f}
    fn bh_setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        if array != 0 {
            let (fo, fs, ft) = extract_interior_field_info(descr);
            let is = descr
                .as_interior_field_descr()
                .map(|ifd| ifd.array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(fo + fs);
            pyre_object::setinteriorfield(array as *mut _, index, fo, fs, is, ft, value);
        }
    }

    fn bh_setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        self.bh_setinteriorfield_gc_i(array, index, value, descr);
    }

    fn bh_setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        self.bh_setinteriorfield_gc_i(array, index, value, descr);
    }

    fn bh_newstr(&self, length: usize) -> i64 {
        bh_alloc_lowlevel_string(length, LOWLEVEL_STR_BASE_SIZE, 1)
    }

    fn bh_strsetitem(&self, string: i64, index: usize, char: i64) {
        bh_write_lowlevel_char(string, index, char, 1);
    }

    fn os_str_concat(&self, str1: i64, str2: i64) -> i64 {
        bh_concat_lowlevel_strings(str1, str2, 1)
    }

    fn os_str_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        bh_slice_lowlevel_string(str, start, stop, 1)
    }

    fn bh_newunicode(&self, length: usize) -> i64 {
        bh_alloc_lowlevel_string(length, LOWLEVEL_UNICODE_BASE_SIZE, 4)
    }

    fn bh_unicodesetitem(&self, string: i64, index: usize, char: i64) {
        bh_write_lowlevel_char(string, index, char, 4);
    }

    fn os_uni_concat(&self, str1: i64, str2: i64) -> i64 {
        bh_concat_lowlevel_strings(str1, str2, 4)
    }

    fn os_uni_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        bh_slice_lowlevel_string(str, start, stop, 4)
    }

    /// resume.py:1452-1456 allocate_raw_buffer(func, size)
    /// Concrete reader: cpu.bh_call_i(func, [size], None, None, calldescr)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let (driver, _) = driver_pair();
        // resume.py:1453-1455: calldescr, _ = cic.callinfo_for_oopspec(
        //   OS_RAW_MALLOC_VARSIZE_CHAR). The calldescr comes from the shared
        // callinfocollection, not a freshly minted MOST_GENERAL descr; func is
        // the caller's argument (resume.py discards the callinfo's func as `_`).
        let cic = driver
            .meta_interp()
            .callinfocollection()
            .expect(
                "allocate_raw_buffer: MetaInterp.callinfocollection required \
                 (resume.py:1453)",
            )
            .clone();
        let (calldescr, _) =
            cic.callinfo_for_oopspec(majit_ir::effectinfo::OopSpecIndex::RawMallocVarsizeChar);
        let calldescr = calldescr.expect("callinfo_for_oopspec missing OS_RAW_MALLOC_VARSIZE_CHAR");
        let cd = calldescr
            .as_call_descr()
            .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr must downcast to CallDescr");
        let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
        driver.meta_interp().backend().bh_call_i(
            func,
            Some(&[size as i64]),
            None,
            None,
            &bh_calldescr,
        )
    }

    /// resume.py:1547 cpu.bh_raw_store_f(buffer, offset, value, descr).
    fn bh_raw_store_f(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(descr);
        let (driver, _) = driver_pair();
        let backend = driver.meta_interp().backend();
        backend.bh_raw_store_f(buffer, offset, f64::from_bits(value as u64), &bh_descr);
    }

    /// resume.py:1550 cpu.bh_raw_store_i(buffer, offset, value, descr).
    fn bh_raw_store_i(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(descr);
        let (driver, _) = driver_pair();
        let backend = driver.meta_interp().backend();
        backend.bh_raw_store_i(buffer, offset, value, &bh_descr);
    }

    fn box_int(&self, value: i64) -> i64 {
        pyre_object::intobject::w_int_new(value) as i64
    }

    fn box_float(&self, bits: i64) -> i64 {
        pyre_object::floatobject::w_float_new(f64::from_bits(bits as u64)) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a global by name from the frame's canonical `w_globals` object.
    fn frame_global(frame: &PyFrame, name: &str) -> pyre_object::PyObjectRef {
        unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), name) }
            .unwrap_or_else(|| panic!("namespace should contain {name}"))
    }

    struct TestJitParamsGuard;

    impl TestJitParamsGuard {
        fn low_threshold() -> Self {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
            driver.set_param("threshold", 2);
            driver.set_param("function_threshold", 2);
            Self
        }
    }

    impl Drop for TestJitParamsGuard {
        fn drop(&mut self) {
            let (driver, _) = driver_pair();
            driver
                .meta_interp_mut()
                .warm_state_mut()
                .set_default_params();
            driver.set_param("threshold", JIT_THRESHOLD as i64);
        }
    }

    fn function_code_from_module(
        module: &pyre_interpreter::CodeObject,
        name: &str,
    ) -> pyre_interpreter::CodeObject {
        use pyre_interpreter::ConstantData;

        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == name => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("test source should contain function code {name}"))
    }

    #[test]
    fn for_iter_flat_arithmetic_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def f(n):\n    s = 0\n    for i in range(n):\n        s = (s + i * i + 3) % 1000000007\n    return s\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "f");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_single_level_binaryop_mutation_body_is_jit_safe() {
        // single-level `s += t` (in-place list extend via BINARY_OP) recovers on
        // abort (verified by /tmp/inplace_probe.py) -> body is all allow-listed.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def h(src, t):\n    s = []\n    for x in src:\n        s += t\n    return len(s)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "h");
        assert!(for_iter_bodies_all_jit_safe(&code));
    }

    #[test]
    fn for_iter_nested_append_body_is_jit_safe() {
        // Both loop bodies are scanned; the inner LOAD_ATTR+CALL is admitted now
        // that a mid-body method-call abort resumes exactly.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def g(n, acc):\n    for a in range(n):\n        for b in range(a):\n            acc.append(a)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "g");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_single_level_explicit_append_body_is_jit_safe() {
        // LOAD_ATTR followed by CALL is admitted now that a mid-body method-call
        // abort resumes exactly instead of dropping the remainder of the iteration.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def k(src, acc):\n    for x in src:\n        acc.append(x)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "k");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_subscr_load_body_is_jit_safe() {
        // A subscript LOAD (`tbl[i]`) lowers to `BinaryOp(Subscr)`, a dispatching
        // op: `__getitem__` runs in a separate user frame and recovers on a walk
        // abort like any other `BinaryOp`, so the body is admitted. Only the
        // mutating write `STORE_SUBSCR` is admitted by the direct-store widening.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def s(src, tbl):\n    acc = 0\n    for i in src:\n        acc = acc + tbl[i]\n    return acc\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "s");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_straight_line_store_subscr_body_is_jit_safe() {
        // The direct STORE_SUBSCR is admitted; a later mid-body abort resumes
        // exactly, so the store commits once and the iteration tail is preserved.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, buf):\n    for i in src:\n        buf[i] = i\n    return len(buf)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_subscr_with_branch_body_is_jit_safe() {
        // Exact resume makes STORE_SUBSCR safe even in a body with a branch.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, buf):\n    for i in src:\n        if i & 1:\n            buf[i] = i\n    return len(buf)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_subscr_with_call_body_is_jit_safe() {
        // Exact resume preserves the committed store and the tail when the call
        // aborts the walk in a body containing both STORE_SUBSCR and append.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, d, fn, acc):\n    total = 0\n    for i in src:\n        d[i % 8] = i\n        total += fn(i)\n        acc.append(i)\n    return total\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_store_deref_with_branch_and_call_body_is_jit_safe() {
        use pyre_interpreter::{Instruction, compile_exec};
        let module = compile_exec(
            "def outer(src, fn, acc):\n    n = -1\n    def read():\n        return n\n    for i in src:\n        if i & 1:\n            n = i * 17 + 3\n        n += fn(i)\n        acc.append((i, n))\n    return n, read()\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "outer");
        let mut arg_state = pyre_interpreter::OpArgState::default();
        assert!(
            code.instructions
                .iter()
                .copied()
                .any(|unit| { matches!(arg_state.get(unit).0, Instruction::StoreDeref { .. }) })
        );
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_delete_subscr_with_branch_and_call_body_is_jit_safe() {
        // Exact resume preserves a committed delete and the tail when a call
        // aborts the walk in a body that also contains a branch.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, d, fn, acc):\n    for i in src:\n        if i & 1:\n            del d[i]\n        fn(i)\n        acc.append(i)\n    return len(d)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_delete_attr_with_branch_and_call_body_is_jit_safe() {
        // DELETE_ATTR uses the same exact-resume path in a body containing a
        // branch, an aborting user call, and an append tail.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(src, o, fn, acc):\n    for i in src:\n        if i & 1:\n            del o.x\n        fn(i)\n        acc.append(i)\n    return len(acc)\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_straight_line_store_attr_body_is_jit_safe() {
        // The direct STORE_ATTR is admitted through the direct-store widening.
        use pyre_interpreter::compile_exec;
        let module =
            compile_exec("def w(src, o):\n    for i in src:\n        o.x = i\n    return o.x\n")
                .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn for_iter_load_attr_method_body_is_jit_safe() {
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def w(objs):\n    s = 0\n    for o in objs:\n        s += o.bump()\n    return s\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "w");
        assert!(for_iter_bodies_all_jit_safe(&code));
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    #[test]
    fn with_block_frame_is_admitted() {
        // A `with` block compiles to `WITH_EXCEPT_START` for its exception link,
        // lowered as a residual with the exception disposition preserved across
        // the guard-failure bridge. The frame is admitted for tracing.
        // The body is kept free of `FOR_ITER` so the only classification axis
        // is the `WITH_EXCEPT_START` shape; a `for` loop whose body is not
        // allow-listed declines independently via `for_iter_bodies_all_jit_safe`.
        use pyre_interpreter::compile_exec;
        let module = compile_exec(
            "def wf(cm):\n    total = 0\n    with cm:\n        total += 1\n    return total\n",
        )
        .expect("test code should compile");
        let code = function_code_from_module(&module, "wf");
        assert_eq!(unsupported_jit_shape(&code), UnsupportedJitShape::None);
    }

    fn ensure_test_jit_callbacks() {
        super::init_callbacks();
        let _ = crate::jit::codewriter::CodeWriter::instance();
    }

    /// Drive the codewriter `register_portal_jitdriver` setup path so
    /// `pyre_jit_trace::state::ensure_jitcode_ptr` can resolve the
    /// installed entry. Mirrors RPython warmspot.py:281-282 — the
    /// trace-side staticdata is populated only by the make_jitcodes
    /// drain.
    ///
    /// The portal must be registered using the canonical `CodeObject*`
    /// that backs `w_code` (the inner pointer obtained by
    /// `w_code_get_ptr`), not an arbitrary copy of the same source —
    /// `CallControl.jitcodes` is keyed by raw pointer identity.
    fn register_test_portal(_unused: &pyre_interpreter::CodeObject, w_code: *const ()) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject
        };
        let canonical_code = unsafe { &*raw_code };
        crate::jit::codewriter::register_portal_jitdriver(canonical_code);
    }

    /// Post-regalloc Ref-bank color of semantic slot `slot` at `py_pc`,
    /// read from the per-PC `pcdep_color_slots` entries (local `i` is
    /// slot `i`, operand-stack depth `d` is slot `nlocals + d`). Colors
    /// are per-program-point (chordal coloring may coalesce
    /// disjointly-live slots and re-color across PCs), so there is no
    /// flat slot → color lookup; `None` when the slot carries no live
    /// restorable entry at that PC.
    fn pcdep_color_for_slot(jitcode_index: i32, py_pc: usize, slot: usize) -> Option<u32> {
        pyre_jit_trace::state::pcdep_color_slots_at(jitcode_index, py_pc as i32)
            .iter()
            .find(|&&(b, _, s)| b == 1 && s as usize == slot)
            .map(|&(_, c, _)| u32::from(c))
    }

    /// Find the first JitCode `-live-` offset where every requested local
    /// slot and operand-stack depth carries a color AND the compiled live set
    /// contains all of those colors. Returns that JitCode offset, its full
    /// live set, and the mapped colors (locals first, then stack depths in
    /// order).
    fn live_pc_with_slot_colors(
        jitcode_index: i32,
        code: &pyre_interpreter::CodeObject,
        local_slots: &[u32],
        stack_depths: &[usize],
    ) -> (usize, Vec<u32>, Vec<u32>) {
        let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
        let payload = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)
            .expect("compiled trace-side jitcode must be registered");
        let op_live = pyre_jit_trace::state::op_live();
        (0..payload.jitcode.body().code.len())
            .filter(|&jit_pc| payload.jitcode.can_decode_live_vars(jit_pc, op_live))
            .find_map(|jit_pc| {
                let pcdep = payload.pcdep_for_jitcode_pc(jit_pc)?;
                let colors: Vec<u32> = local_slots
                    .iter()
                    .map(|&slot| {
                        pcdep
                            .iter()
                            .find(|&&(b, _, s)| b == 1 && s as usize == slot as usize)
                            .map(|&(_, c, _)| u32::from(c))
                    })
                    .chain(stack_depths.iter().map(|&d| {
                        pcdep
                            .iter()
                            .find(|&&(b, _, s)| b == 1 && s as usize == stack_base + d)
                            .map(|&(_, c, _)| u32::from(c))
                    }))
                    .collect::<Option<Vec<u32>>>()?;
                let live = pyre_jit_trace::state::frame_liveness_reg_indices_at(
                    jitcode_index,
                    jit_pc as i32,
                );
                colors
                    .iter()
                    .all(|c| live.contains(c))
                    .then_some((jit_pc, live, colors))
            })
            .unwrap_or_else(|| {
                panic!(
                    "no JitCode -live- offset carries colors for local slots \
                     {local_slots:?} + stack depths {stack_depths:?}"
                )
            })
    }

    fn live_pc_containing_all(
        jitcode_index: i32,
        _code: &pyre_interpreter::CodeObject,
        regs: &[u32],
    ) -> (usize, Vec<u32>) {
        let payload = pyre_jit_trace::state::pyjitcode_for_jitcode_index(jitcode_index)
            .expect("compiled trace-side jitcode must be registered");
        let op_live = pyre_jit_trace::state::op_live();
        let live_by_jit_pc: Vec<(usize, Vec<u32>)> = (0..payload.jitcode.body().code.len())
            .filter(|&jit_pc| payload.jitcode.can_decode_live_vars(jit_pc, op_live))
            .map(|jit_pc| {
                let live = pyre_jit_trace::state::frame_liveness_reg_indices_at(
                    jitcode_index,
                    jit_pc as i32,
                );
                (jit_pc, live)
            })
            .collect();
        live_by_jit_pc
            .iter()
            .find_map(|(jit_pc, live)| {
                regs.iter()
                    .all(|reg| live.contains(reg))
                    .then_some((*jit_pc, live.clone()))
            })
            .unwrap_or_else(|| {
                panic!(
                    "compiled JitCode liveness should expose regs {regs:?}; got {:?}",
                    live_by_jit_pc
                )
            })
    }

    fn compiled_trace_fixture(
        source: &str,
        function_name: &str,
        live_locals: &[u32],
        live_stack_depths: &[usize],
        init: impl FnOnce(&mut PyFrame),
    ) -> (pyre_interpreter::pyframe::FrameBox, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = PyFrame::new(code.clone());
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Both the `register_color = nlocals + depth` (stack) and the
        // `color == slot` (locals) identities were removed: colors are
        // per-program-point, so resolve the requested semantic slots
        // through the per-PC `pcdep_color_slots` entries while picking
        // the resume PC. `live_locals` are semantic local SLOT indices.
        let (resume_pc, _, _) =
            live_pc_with_slot_colors(jitcode_index, &code, live_locals, live_stack_depths);
        (frame, jitcode_ptr, resume_pc)
    }

    /// Stack-depth-based variant of `compiled_trace_fixture`. Locates the
    /// first Python PC where the bytecode-level forward stack analysis
    /// reports `target_depth`, independent of which Ref-bank colors land
    /// in the encoded `-live-` set. Stable across force-add
    /// removal: the codewriter `live_r` is now SSA-driven only, so
    /// stack-slot colors no longer always appear there even when those
    /// slots are runtime-live (the consume_one_section heap-read
    /// fallback fills them at decode time). Tests that need a PC with
    /// a specific symbolic stack depth should use this helper.
    fn compiled_trace_fixture_at_depth(
        source: &str,
        function_name: &str,
        target_depth: u16,
        init: impl FnOnce(&mut PyFrame),
    ) -> (pyre_interpreter::pyframe::FrameBox, *const (), usize) {
        use pyre_interpreter::compile_exec;
        use pyre_jit_trace::state as trace_state;

        ensure_test_jit_callbacks();
        let module = compile_exec(source).expect("test code should compile");
        let code = function_code_from_module(&module, function_name);
        let mut frame = PyFrame::new(code.clone());
        init(&mut frame);
        frame.fix_array_ptrs();

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let _ = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let depth_table =
            pyre_jit_trace::liveness::liveness_for(&code as *const _).depth_at_py_pc();
        let resume_pc = depth_table
            .iter()
            .position(|&d| d == target_depth)
            .unwrap_or_else(|| {
                panic!(
                    "test source should reach stack depth {target_depth}; depth_table={depth_table:?}"
                )
            });
        (frame, jitcode_ptr, resume_pc)
    }

    fn single_local_test_state(
        ctx: &mut majit_metainterp::TraceCtx,
        frame: &PyFrame,
        frame_ptr: usize,
        jitcode_ptr: *const (),
        resume_pc: usize,
        local_type: majit_ir::Type,
        local: majit_ir::OpRef,
    ) -> pyre_jit_trace::state::TestSymState {
        use pyre_jit_trace::state as trace_state;

        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(ctx, frame_ref);
        pyre_jit_trace::state::TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 1,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![local_type],
            symbolic_stack_types: vec![],
            registers_r: vec![local],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(1),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        }
    }

    // emit_store_local_with_mirror no longer
    // emits the inline `ref_copy(reg, stored_reg)` on portal frames
    // (matches upstream `jtransform.py:1898 do_fixed_list_setitem`
    // vable branch which emits only `setarrayitem_vable_r`).  This
    // test's precondition — `frame_liveness_reg_indices_at` must
    // expose local `i`'s color at some PC — relied on the walker
    // writing local `i` into `Reg(Ref, color_i)` via that retired
    // ref_copy.  Locals now live exclusively in the vable array;
    // `restore_guard_failure_values` recovers them through the
    // virtualizable array path.  Rewriting this test against the
    // vable-array recovery shape is tracked separately.
    #[test]
    #[ignore = "walker no longer mirrors locals into Ref-bank registers on portal frames; rewrite against vable-array recovery path"]
    fn test_restore_guard_failure_uses_runtime_value_kinds_with_compiled_trace_jitcode() {
        use majit_ir::{GcRef, Type, Value};
        use majit_metainterp::JitState;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, PyreJitState, PyreMeta};
        use pyre_object::pyobject::is_int;
        use pyre_object::{w_int_get_value, w_int_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        // Resolve the per-local Ref-bank colors via the per-PC
        // `pcdep_color_slots` entries.  Hardcoding reg indices (e.g. `&3`
        // for local `i`) couples the test to the walker's pre-canonical
        // regalloc strategy; querying the per-PC map keeps the assertion
        // shape regardless of which lowering path emits the jitcode.
        // Pick the resume PC where local `i` (slot 3) is both colored and
        // in the compiled `-live-` set.
        let resume_pc = (0..code.instructions.len())
            .find(|&pc| {
                pcdep_color_for_slot(jitcode_index, pc, 3).is_some_and(|c| {
                    trace_state::frame_liveness_reg_indices_at(jitcode_index, pc as i32)
                        .contains(&c)
                })
            })
            .expect("compiled liveness should expose local i at some Python PC");
        let color_i = pcdep_color_for_slot(jitcode_index, resume_pc, 3)
            .expect("regalloc must assign a color to local `i`");
        // A local dead at the resume PC carries no per-PC entry (and thus
        // no live reg); the `u32::MAX` sentinel keeps its match arm inert.
        let color_a = pcdep_color_for_slot(jitcode_index, resume_pc, 0).unwrap_or(u32::MAX);
        let color_b = pcdep_color_for_slot(jitcode_index, resume_pc, 1).unwrap_or(u32::MAX);
        let color_c = pcdep_color_for_slot(jitcode_index, resume_pc, 2).unwrap_or(u32::MAX);
        let live_regs = trace_state::frame_liveness_reg_indices_at(jitcode_index, resume_pc as i32);
        assert!(
            live_regs.contains(&color_i),
            "selected resume pc must decode the raw-int local slot"
        );
        assert_eq!(
            trace_state::frame_value_count_at(jitcode_index, resume_pc as i32),
            live_regs.len(),
            "frame-value count must come from the same compiled jitcode liveness block"
        );

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: Some(resume_pc),
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 4,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            // Trace-entry slot types can be stale; guard failure must still
            // respect the runtime Value tags recovered from resume data.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };

        let ec_value = unsafe { (*(frame_ptr as *const PyFrame)).execution_context as usize };
        let mut values = vec![
            Value::Ref(GcRef(frame_ptr)),                // frame
            Value::Ref(GcRef(ec_value)),                 // ec extra red
            Value::Int(8),                               // last_instr
            Value::Ref(GcRef(frame.pycode as usize)),    // pycode
            Value::Int(4),                               // valuestackdepth
            Value::Ref(GcRef(0)),                        // debugdata
            Value::Ref(GcRef(0)),                        // lastblock
            Value::Ref(GcRef(frame.w_globals as usize)), // w_globals
        ];
        for reg in live_regs.iter() {
            match *reg {
                r if r == color_a => values.push(Value::Ref(GcRef(w_int_new(1) as usize))),
                r if r == color_b => values.push(Value::Ref(GcRef(w_int_new(2) as usize))),
                r if r == color_c => values.push(Value::Ref(GcRef(w_int_new(3) as usize))),
                r if r == color_i => values.push(Value::Int(7)),
                // pypy/module/pypyjit/interp_jit.py:68 reds = ['frame',
                // 'ec'] — portal red args ride the live_r mask. The two
                // trailing live regs are portal_frame_reg /
                // portal_ec_reg holding the runtime frame_ptr and ec.
                _ if Some(reg) == live_regs.iter().rev().nth(1) => {
                    values.push(Value::Ref(GcRef(frame_ptr)));
                }
                _ if Some(reg) == live_regs.iter().rev().next() => {
                    values.push(Value::Ref(GcRef(ec_value)));
                }
                other => panic!("unexpected live reg {other} at resume pc {resume_pc}"),
            }
        }

        assert!(<PyreJitState as JitState>::restore_guard_failure_values(
            &mut state,
            &meta,
            &values,
            &majit_metainterp::blackhole::ExceptionState::default(),
        ));

        assert_eq!(state.next_instr(), 9);
        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    fn test_current_fail_args_flushes_header_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::{FrameBlock, PyFrame};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        let _ = frame.getorcreatedebug(123);
        frame.append_block(FrameBlock {
            valuestackdepth: 0,
            handlerposition: 55,
            previous: std::ptr::null_mut(),
        });
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(999),
            vable_pycode: ctx.const_ref(0xdead),
            vable_valuestackdepth: ctx.const_int(111),
            vable_debugdata: ctx.const_ref(0xbeef),
            vable_lastblock: ctx.const_ref(0xcafe),
            vable_w_globals: ctx.const_ref(0xfeed),
        });
        let ec_ref = ctx.const_ref(frame.execution_context as usize as i64);
        sym.set_test_execution_context(ec_ref);
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let fail_args = state.capture_current_fail_args();

        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[1], ec_ref);
        // last_instr / valuestackdepth are guard-time-overridden by
        // flush_to_frame_for_guard (orgpc - 1, pre-opcode depth).
        // Compare via constants_get_value rather than re-minting a
        // ConstInt and asserting OpRef identity — `history.py:220`
        // ConstInt is fresh-alloc per construction; value-equality is
        // the upstream invariant (`Const.same_constant`, history.py:204).
        assert_eq!(
            ctx.constants_get_value(fail_args[2]),
            Some(majit_ir::Value::Int(resume_pc as i64 - 1)),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[4]),
            Some(majit_ir::Value::Int(4)),
        );
        // pycode / debugdata / lastblock / w_globals are JIT-scope
        // invariant under CPython 3.14 bytecode (`lastblock` is mutated
        // only by SETUP_*/POP_BLOCK paths the tracer never enters) and
        // stay bound to the trace-start inputarg OpRefs the fixture
        // seeded above.
        assert_eq!(
            ctx.constants_get_value(fail_args[3]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xdead))),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[5]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xbeef))),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[6]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xcafe))),
        );
        assert_eq!(
            ctx.constants_get_value(fail_args[7]),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(0xfeed))),
        );
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, stack_colors) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let stack0 = ctx.const_ref(0xb0);
        let stack1 = ctx.const_ref(0xb1);
        // Materialized fail args must carry no holes: fill every live Ref
        // color with a non-NONE placeholder. The semantic mirror keeps the
        // two stack temps at `nlocals + depth` (2, 3); the encoder reads the
        // live colors (placeholders are fine there — the test asserts only
        // count + no-holes + the semantic mirror, not per-color identity).
        let max_idx = live_regs.iter().copied().max().unwrap_or(0).max(3) as usize;
        let mut registers_r = vec![OpRef::NONE; max_idx + 1];
        for &c in &live_regs {
            registers_r[c as usize] = ctx.const_ref(0xfa11_0000 + i64::from(c));
        }
        registers_r[2] = stack0;
        registers_r[3] = stack1;
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r,
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let ec_ref = ctx.const_ref(frame.execution_context as usize as i64);
        sym.set_test_execution_context(ec_ref);
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, stack0), (1, stack1)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let fail_args = state.capture_current_fail_args();

        for &color in &stack_colors {
            assert!(live_regs.contains(&color));
        }
        assert_eq!(
            fail_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + live_regs.len(),
        );
        assert_eq!(fail_args[0], frame_ref);
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
        // `registers_r` remains the semantic frame mirror: stack values
        // stay at `nlocals + depth`. Guard capture materializes the
        // color-indexed bank separately from this mirror/vable state.
        for depth in 0..stack_colors.len() {
            let stack_value = [stack0, stack1][depth];
            let semantic_idx = 2 + depth;
            assert_eq!(
                state.symbolic_registers_r()[semantic_idx],
                stack_value,
                "stack depth {} must be in semantic registers_r[{}]",
                depth,
                semantic_idx,
            );
        }
    }

    #[test]
    fn test_load_local_checked_value_respects_symbolic_local_type_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{LocalOpcodeHandler, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module =
            compile_exec("def f(b):\n    return b\nf(1)\n").expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Resolve local `b`'s Ref-bank color via the per-PC
        // `pcdep_color_slots` entries.  Hardcoding reg index 0 couples the
        // test to walker's pre-canonical local-slot identity; canonical
        // `flatten_graph`'s regalloc-coalesced coloring may emit a
        // different color for the inputarg.  Mirrors the splice-gate
        // convergence pattern landed for
        // test_restore_guard_failure_uses_runtime_value_kinds_... .
        //
        // `b`'s Ref color is not in any `-live-` set under precise liveness
        // (a local restores from the virtualizable, not a register), so the
        // resume PC is picked for validity only; the load reads local slot 0
        // from the symbolic state, and `registers_r` carries `b` at its color.
        let (resume_pc, live_regs) = live_pc_containing_all(jitcode_index, &code, &[]);
        let color_b = pcdep_color_for_slot(jitcode_index, resume_pc, 0)
            .expect("regalloc must assign a color to local `b`");
        let max_color = live_regs.iter().copied().max().unwrap_or(0).max(color_b) as usize;

        let run_case = |symbolic_type: Type, name: &str, expected_guard: Option<OpCode>| {
            let mut ctx = TraceCtx::for_test_types(&[symbolic_type]);
            // resoperation.py:719/727/739 — InputArg has only Int/Float/Ref
            // variants; `input_arg_typed` panics on Type::Void.
            let local = OpRef::input_arg_typed(0, symbolic_type);
            let frame_ref = ctx.const_ref(frame_ptr as i64);
            let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
            let mut sym = PyreSym::from_test_state(TestSymState {
                frame: frame_ref,
                jitcode: jitcode_ptr,
                nlocals: 1,
                valuestackdepth: 1,
                locals_cells_stack_array_ref: locals_array,
                symbolic_local_types: vec![symbolic_type],
                symbolic_stack_types: vec![],
                registers_r: {
                    let mut r = vec![OpRef::NONE; max_color + 1];
                    r[color_b as usize] = local;
                    r
                },
                concrete_stack: vec![],
                concrete_namespace: frame.w_globals,
                vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
                vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
                vable_valuestackdepth: ctx.const_int(1),
                vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
                vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
                vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
            });
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

            let loaded =
                <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, name)
                    .expect("local should load");
            assert_eq!(loaded.opref, local);

            let recorder = ctx.into_recorder();
            match expected_guard {
                Some(opcode) => {
                    assert!(
                        recorder.ops().iter().any(|op| op.opcode == opcode),
                        "expected guard opcode {opcode:?} in {:?}",
                        recorder.ops()
                    );
                }
                None => assert_eq!(recorder.num_guards(), 0),
            }
        };

        run_case(Type::Int, "j", None);
        run_case(Type::Ref, "b", Some(OpCode::GuardNonnull));
    }

    #[test]
    fn test_guard_class_uses_guard_nonnull_class_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{INT_TYPE, w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let obj = OpRef::input_arg_ref(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        state.capture_guard_class(obj, &INT_TYPE as *const _);

        let recorder = ctx.into_recorder();
        let op = recorder.ops().last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardClass);
        assert_eq!(op.arg(0).to_opref(), obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        let module = compile_exec("def f(x):\n    i = 7\n    return x[i - 7]\nf([1])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_list_new(vec![w_int_new(11)]);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.locals_w_mut()[3] = w_int_new(5);
        frame.valuestackdepth = 4;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[0, 1]);
        let max_color = live_regs.iter().copied().max().unwrap_or(0) as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let int_obj = OpRef::input_arg_ref(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 2,
            valuestackdepth: 4,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Int],
            symbolic_stack_types: vec![Type::Ref, Type::Int],
            registers_r: vec![OpRef::NONE; max_color + 1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(0),
            vable_pycode: ctx.const_ref(0),
            vable_valuestackdepth: ctx.const_int(0),
            vable_debugdata: ctx.const_ref(0),
            vable_lastblock: ctx.const_ref(0),
            vable_w_globals: ctx.const_ref(0),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        let _ = state.capture_trace_guarded_int_payload(int_obj);

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        let recorded_ops: Vec<(OpCode, Vec<OpRef>)> = recorder
            .ops()
            .iter()
            .map(|op| {
                (
                    op.opcode,
                    op.getarglist().iter().map(|a| a.to_opref()).collect(),
                )
            })
            .collect();
        for op in recorder.ops() {
            if op.opcode == OpCode::GuardClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI
                && op
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()
                    == vec![int_obj]
            {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardClass: {:?}",
            recorded_ops
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI: {:?}",
            recorded_ops
        );
    }

    #[test]
    fn test_branch_guard_preserves_pre_pop_stack_shape_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::SnapshotTagged;
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        // Splice's precise canonical coloring coalesces the DEEPEST live
        // operand-stack Ref slot with the portal `ec` red (ec is dead in
        // a call-free body), and the encoder substitutes ec unconditionally
        // at that color — so the depth-0 slot is not separately observable.
        // Use a nested-subscript expression whose `-live-` resume marker
        // (the inner BINARY_SUBSCR) keeps four operand slots live: the
        // deepest coalesces with ec, but depths 1 and 2 above it retain
        // distinct, portal-disjoint colors that ARE observable in the
        // snapshot.
        let module = compile_exec("def f(a, b, c):\n    return [a, b, c[0]]\nf(1,2,[3])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_int_new(11);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.valuestackdepth = 7;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Operand stack depths 1 and 2 carry the two observable slots.
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[1, 2]);

        let run_case = |record_branch_guard: bool| {
            let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int, Type::Ref, Type::Ref]);
            let lower_stack = OpRef::input_arg_ref(0);
            let truth = OpRef::input_arg_int(1);
            // Pre-seed the deepest (ec-coalesced) slot and the topmost slot
            // too: every live operand-stack slot must carry a value so the
            // snapshot reads the seeded mirror rather than lazy-filling from
            // the heap. Under splice's distinct color/semantic indices a
            // lazy stack-fill writes `registers_r[color_idx]`, which can
            // alias a sibling slot's seeded semantic index and clobber it.
            let deep_slot = OpRef::input_arg_ref(2);
            let top_slot = OpRef::input_arg_ref(3);
            let frame_ref = ctx.const_ref(frame_ptr as i64);
            let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
            let mut sym = PyreSym::from_test_state(TestSymState {
                frame: frame_ref,
                jitcode: jitcode_ptr,
                nlocals: 3,
                valuestackdepth: 7,
                locals_cells_stack_array_ref: locals_array,
                symbolic_local_types: vec![Type::Ref, Type::Ref, Type::Ref],
                symbolic_stack_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
                registers_r: vec![OpRef::NONE; 8],
                concrete_stack: vec![],
                concrete_namespace: frame.w_globals,
                vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
                vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
                vable_valuestackdepth: ctx.const_int(7),
                vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
                vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
                vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
            });
            trace_state::seed_compiled_trace_jitcode_test_state(
                &mut sym,
                &mut ctx,
                jitcode_index,
                resume_pc as i32,
                &[(0, deep_slot), (1, lower_stack), (2, truth), (3, top_slot)],
            );
            let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
            // `record_branch_guard` captures at its `other_target`, which this
            // fixture passes as the same JitCode `-live-` resume coordinate.
            state.set_resume_marker_for_test(resume_pc);
            if record_branch_guard {
                state.capture_record_branch_guard(OpRef::NONE, truth, true, resume_pc);
            } else {
                state.capture_generate_guard(OpCode::GuardTrue, &[truth]);
            }

            // Production guard recording goes through
            // `record_guard_typed` + `capture_resumedata` —
            // `op.fail_args` stays None until the optimizer's
            // `store_final_boxes_in_guard` writes it back from the
            // snapshot.  Inspect the snapshot directly (the canonical
            // RPython resume oracle) instead of the raw recorder buffer.
            //
            // Snapshot layout (opencoder.py:806 / build_framestack_snapshot):
            //  - `vable_boxes` = full virtualizable image
            //    `[frame_ptr, scalar_fields..., array_items...]`
            //    (NUM_SCALAR_INPUTARGS scalars + locals/stack array slots).
            //  - `frames[0].boxes` = top frame's active boxes (one per
            //    live register at the resume PC).
            let guard = ctx
                .ops()
                .last()
                .expect("branch guard should be recorded")
                .clone();
            assert_eq!(guard.opcode, OpCode::GuardTrue);
            let snapshot_id = guard.rd_resume_position.get();
            assert!(
                snapshot_id >= 0,
                "branch guard must carry rd_resume_position pointing at its captured snapshot",
            );
            let snapshot = &ctx.snapshots()[snapshot_id as usize];
            let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            assert!(
                snapshot.vable_boxes.len() >= n,
                "vable_boxes must contain at least the scalar virtualizable header: {:?}",
                snapshot.vable_boxes
            );
            // vable_boxes[0] = frame_ptr — the encoded form of `frame_ref`,
            // which the test seeded as `ctx.const_ref(frame_ptr as i64)`.
            assert_eq!(
                snapshot.vable_boxes[0],
                SnapshotTagged::Const(frame_ptr as i64, Type::Ref)
            );
            let active_boxes = snapshot
                .frames
                .first()
                .map(|f| f.boxes.as_slice())
                .unwrap_or(&[]);
            assert_eq!(active_boxes.len(), live_regs.len());
            // Kind-segregated liveness emission was restored
            // (Int regs first, then Ref); additionally, the
            // the `register_color = nlocals + depth` identity, so the
            // active_boxes order no longer reflects Python stack
            // depth. Verify both stack OpRefs are present without
            // asserting an order that the protocol no longer
            // guarantees.
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(li, _) if *li == lower_stack
                )),
                "pre-pop snapshot must capture lower stack slot: {:?}",
                active_boxes
            );
            assert!(
                active_boxes.iter().any(|b| matches!(
                    b,
                    SnapshotTagged::Box(ti, _) if *ti == truth
                )),
                "pre-pop snapshot must capture truth slot: {:?}",
                active_boxes
            );
        };

        run_case(true);
        run_case(false);
    }

    #[test]
    fn test_branch_truth_uses_concrete_parameter_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use majit_metainterp::recorder::SnapshotTagged;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{BranchOpcodeHandler, compile_exec};
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::{w_int_new, w_list_new};

        ensure_test_jit_callbacks();
        // Splice's precise coloring coalesces the deepest live operand-stack
        // Ref slot with the portal `ec` red, so the depth-0 slot is not
        // separately observable. Use a nested-subscript expression whose
        // inner-BINARY_SUBSCR `-live-` marker keeps four operand slots live;
        // depths 1 and 2 above the coalesced bottom retain distinct,
        // portal-disjoint colors that ARE observable in the snapshot.
        let module = compile_exec("def f(a, b, c):\n    return [a, b, c[0]]\nf(1,2,[3])\n")
            .expect("test code should compile");
        let code = function_code_from_module(&module, "f");

        let mut frame = PyFrame::new(code.clone());
        frame.locals_w_mut()[0] = w_int_new(11);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.locals_w_mut()[2] = w_list_new(vec![w_int_new(21)]);
        frame.valuestackdepth = 7;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        register_test_portal(&code, frame.pycode as *const ());
        let jitcode_ptr = trace_state::ensure_jitcode_ptr(frame.pycode as *const ())
            .expect("real trace-side jitcode registration must succeed");
        let jitcode_index = trace_state::ensure_jitcode_index(frame.pycode as *const ())
            .expect("real trace-side jitcode index must exist");
        // Operand stack depths 1 and 2 carry the two observable slots.
        let (resume_pc, live_regs, _) =
            live_pc_with_slot_colors(jitcode_index, &code, &[], &[1, 2]);

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int, Type::Ref, Type::Ref]);
        let lower_stack = OpRef::input_arg_ref(0);
        let truth = OpRef::input_arg_int(1);
        // Pre-seed every live operand-stack slot so the snapshot reads the
        // seeded mirror instead of lazy-filling from the heap; under splice
        // a lazy stack-fill writes `registers_r[color_idx]`, which can alias
        // a sibling slot's seeded semantic index and clobber it.
        let deep_slot = OpRef::input_arg_ref(2);
        let top_slot = OpRef::input_arg_ref(3);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 3,
            valuestackdepth: 7,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref, Type::Ref, Type::Ref],
            symbolic_stack_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
            registers_r: vec![OpRef::NONE; 8],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(resume_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(7),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        });
        trace_state::seed_compiled_trace_jitcode_test_state(
            &mut sym,
            &mut ctx,
            jitcode_index,
            resume_pc as i32,
            &[(0, deep_slot), (1, lower_stack), (2, truth), (3, top_slot)],
        );
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);
        state.set_resume_marker_for_test(resume_pc);

        state.capture_generate_guard(OpCode::GuardTrue, &[truth]);
        assert_eq!(
            state
                .capture_concrete_branch_truth_for_value(truth, w_int_new(1))
                .unwrap(),
            true
        );
        <MIFrame as BranchOpcodeHandler>::leave_branch_truth(&mut state).unwrap();

        // Snapshot is the resume-data oracle, not
        // `op.fail_args` (None until the optimizer's
        // `store_final_boxes_in_guard` writes it back).
        let guard = ctx
            .ops()
            .last()
            .expect("guard op should be present")
            .clone();
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        let snapshot_id = guard.rd_resume_position.get();
        assert!(
            snapshot_id >= 0,
            "guard must carry rd_resume_position pointing at its captured snapshot",
        );
        let snapshot = &ctx.snapshots()[snapshot_id as usize];
        let n = pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        assert!(
            snapshot.vable_boxes.len() >= n,
            "vable_boxes must contain at least the scalar virtualizable header: {:?}",
            snapshot.vable_boxes
        );
        let active_boxes = snapshot
            .frames
            .first()
            .map(|f| f.boxes.as_slice())
            .unwrap_or(&[]);
        assert_eq!(active_boxes.len(), live_regs.len());
        // See note on the sibling test
        // `test_branch_guard_preserves_pre_pop_stack_shape_*`: kind
        // segregation + color reassignment mean
        // active_boxes order is dictated by liveness format, not by
        // stack depth.
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(li, _) if *li == lower_stack
            )),
            "mixed-bank guard must capture lower stack slot: {:?}",
            active_boxes
        );
        assert!(
            active_boxes.iter().any(|b| matches!(
                b,
                SnapshotTagged::Box(ti, _) if *ti == truth
            )),
            "mixed-bank guard must capture truth slot: {:?}",
            active_boxes
        );
    }

    #[test]
    fn test_close_loop_args_at_target_pc_preserves_virtualizable_stack_with_compiled_trace_jitcode()
    {
        use majit_ir::Type;
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{self as trace_state, MIFrame, PyreSym, TestSymState};
        use pyre_object::w_int_new;

        let _ = driver_pair();
        init_callbacks();
        // Symbolic state below has nlocals=1 + 2 stack slots, so the
        // target PC needs depth=2 (post force-add removal,
        // stack-slot colors no longer always appear in `live_r`, so a
        // depth-based locator is needed instead of `&[1, 2]` regs).
        let (mut frame, jitcode_ptr, target_pc) = compiled_trace_fixture_at_depth(
            "def f(x):\n    return (x, x)\nf(1)\n",
            "f",
            2,
            |frame| {
                frame.locals_w_mut()[0] = w_int_new(7);
            },
        );
        // `live_args_shape_at` and
        // `close_loop_args_at` both derive their JUMP-args shape from
        // `concrete_valuestackdepth()`.  The symbolic state below
        // advertises `valuestackdepth=3` (one local + two stack slots);
        // seed the concrete `PyFrame.valuestackdepth` to match so the
        // shape derivation reflects the same user-side stack the
        // symbolic mirror is testing.
        frame.valuestackdepth = 3;
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(frame_ptr as i64);
        let local0 = ctx.const_ref(w_int_new(11) as usize as i64);
        let stack0 = ctx.const_ref(w_int_new(22) as usize as i64);
        let stack1 = ctx.const_ref(w_int_new(33) as usize as i64);
        let locals_array = trace_state::frame_locals_cells_stack_array_ref(&mut ctx, frame_ref);
        let mut sym = PyreSym::from_test_state(TestSymState {
            frame: frame_ref,
            jitcode: jitcode_ptr,
            nlocals: 1,
            valuestackdepth: 3,
            locals_cells_stack_array_ref: locals_array,
            symbolic_local_types: vec![Type::Ref],
            symbolic_stack_types: vec![Type::Ref, Type::Ref],
            registers_r: vec![local0, stack0, stack1],
            concrete_stack: vec![],
            concrete_namespace: frame.w_globals,
            vable_last_instr: ctx.const_int(target_pc as i64 - 1),
            vable_pycode: ctx.const_ref(frame.pycode as usize as i64),
            vable_valuestackdepth: ctx.const_int(3),
            vable_debugdata: ctx.const_ref(frame.debugdata as usize as i64),
            vable_lastblock: ctx.const_ref(frame.lastblock as usize as i64),
            vable_w_globals: ctx.const_ref(frame.w_globals as usize as i64),
        });
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, target_pc, target_pc);

        let jump_args = state.capture_close_loop_args_at(Some(target_pc), None);

        assert_eq!(
            jump_args.len(),
            pyre_jit_trace::virtualizable_gen::NUM_SCALAR_INPUTARGS + 3,
            "JUMP carries local and stack slots from the virtualizable array"
        );
        assert_eq!(state.symbolic_valuestackdepth(), 3);
        let nlocals = state.symbolic_nlocals();
        let stack_only = state.symbolic_valuestackdepth() - nlocals;
        // The closing `GuardFutureCondition` lazy-inits every register the
        // jitcode reports live at `target_pc`. When that PC is a real
        // result-producing op (BUILD_TUPLE / BINARY_OP / BUILD_LIST), its
        // destination color sits one past the virtualizable window
        // `[nlocals..nlocals+stack_only]`, so `registers_r` may extend
        // beyond the window with a tail slot the synthetic state never
        // produced (production fills it via `materialize_fail_arg_slot`).
        // The invariant under test is that the window itself — the slots
        // the JUMP carries — is fully covered and preserved.
        assert!(
            state.symbolic_registers_r().len() >= nlocals + stack_only,
            "register file must cover the virtualizable window"
        );
        assert!(
            state.symbolic_registers_r()[nlocals..nlocals + stack_only]
                .iter()
                .all(|opref| !opref.is_none()),
            "live stack slots carried by the JUMP must be preserved"
        );
    }

    #[test]
    fn test_trace_dynamic_list_index_typed_int_skips_object_unbox_with_compiled_trace_jitcode() {
        use majit_ir::{OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};
        use pyre_object::w_int_new;

        let (frame, jitcode_ptr, resume_pc) =
            compiled_trace_fixture("def f(b):\n    return b\nf(1)\n", "f", &[], &[], |frame| {
                frame.locals_w_mut()[0] = w_int_new(2);
            });
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Int, Type::Int]);
        let key = OpRef::input_arg_int(0);
        let len = OpRef::input_arg_int(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Int,
            key,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let raw_index = state.capture_trace_dynamic_list_index(key, len, 2);
        assert_eq!(raw_index, key);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_guards(), 2);
        assert!(
            recorder
                .ops()
                .iter()
                .all(|op| op.opcode != majit_ir::OpCode::GuardClass),
            "typed-int index should not guard object class for an unbox fast path: {:?}",
            recorder.ops()
        );
        assert!(
            recorder
                .ops()
                .iter()
                .all(|op| op.opcode != majit_ir::OpCode::GetfieldGcPureI),
            "typed-int index should not read boxed int payloads: {:?}",
            recorder.ops()
        );
    }

    #[test]
    fn test_direct_len_value_returns_typed_raw_len_for_integer_list_with_compiled_trace_jitcode() {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};
        use pyre_object::{w_int_new, w_list_new};

        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_int_storage(list));
        }
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(x):\n    return len(x)\nf([1, 2, 3])\n",
            "f",
            &[],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = list;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Ref]);
        let value = OpRef::input_arg_ref(0);
        let callable = OpRef::input_arg_ref(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            value,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let len = state
            .capture_direct_len_value(callable, value, list)
            .expect("integer-list len fast path should trace");
        assert_eq!(state.capture_value_type(len), Type::Int);

        let recorder = ctx.into_recorder();
        assert_ne!(
            recorder.ops().last().map(|op| op.opcode),
            Some(OpCode::CallI)
        );
        let mut saw_len_field = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op.getdescr().map(|d| d.index())
                    == Some(pyre_jit_trace::descr::list_int_items_len_descr().index())
            {
                saw_len_field = true;
            }
        }
        assert!(saw_len_field);
        assert!(!saw_new);
    }

    #[test]
    fn test_trace_direct_float_list_getitem_uses_gc_field_loads_for_list_object_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_jit_trace::state::{MIFrame, PyreSym};

        let float_list = pyre_object::w_list_new(vec![
            pyre_object::floatobject::w_float_new(1.5),
            pyre_object::floatobject::w_float_new(2.5),
            pyre_object::floatobject::w_float_new(3.5),
        ]);
        unsafe {
            assert!(pyre_object::listobject::w_list_uses_float_storage(
                float_list
            ));
        }
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(x):\n    return x[2]\nf([1.5, 2.5, 3.5])\n",
            "f",
            &[],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = float_list;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Int]);
        let list = OpRef::input_arg_ref(0);
        let key = OpRef::input_arg_int(1);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            list,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let result = state.capture_generated_list_getitem_by_strategy(list, key, 2, 2);
        assert_eq!(state.capture_value_type(result), Type::Float);

        let recorder = ctx.into_recorder();
        let mut saw_gc_field = false;
        let mut saw_raw_field = false;
        let mut saw_gc_array = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(list) =>
                {
                    saw_gc_field = true
                }
                OpCode::GetfieldRawI
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(list) =>
                {
                    saw_raw_field = true
                }
                OpCode::GetarrayitemGcF => saw_gc_array = true,
                _ => {}
            }
        }
        assert!(saw_gc_field);
        assert!(!saw_raw_field);
        assert!(saw_gc_array);
    }

    #[test]
    fn test_iter_next_value_for_range_iterator_uses_gc_fields_and_returns_raw_int_with_compiled_trace_jitcode()
     {
        use majit_ir::{OpCode, OpRef, Type};
        use majit_metainterp::TraceCtx;
        use pyre_interpreter::IterOpcodeHandler;
        use pyre_jit_trace::state::{MIFrame, PyreSym};

        let range_iter = pyre_object::w_range_iter_new(0, 2, 1);
        let (frame, jitcode_ptr, resume_pc) = compiled_trace_fixture(
            "def f(it):\n    return it\nf(range(2))\n",
            "f",
            &[],
            &[],
            |frame| {
                frame.locals_w_mut()[0] = range_iter;
            },
        );
        let frame_ptr = (&*frame) as *const PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let iter = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::from_test_state(single_local_test_state(
            &mut ctx,
            &frame,
            frame_ptr,
            jitcode_ptr,
            resume_pc,
            Type::Ref,
            iter,
        ));
        let mut state = MIFrame::from_sym(&mut ctx, &mut sym, frame_ptr, resume_pc, resume_pc);

        let next = state
            .capture_iter_next(iter, range_iter)
            .expect("range iterator fast path should trace")
            .expect("two-element range iterator should yield a value");
        assert_eq!(state.capture_value_type(next.opref), Type::Int);
        <MIFrame as IterOpcodeHandler>::guard_optional_value(&mut state, next, true)
            .expect("for-iter next should guard the optional result");

        let recorder = ctx.into_recorder();
        let mut saw_getfield_gc = false;
        let mut saw_setfield_gc = false;
        let mut saw_setfield_raw = false;
        let mut saw_getfield_raw = false;
        let mut saw_new = false;
        let mut saw_optional_guard = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(iter) =>
                {
                    saw_getfield_gc = true
                }
                OpCode::SetfieldGc
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(iter) =>
                {
                    saw_setfield_gc = true
                }
                OpCode::SetfieldRaw
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(iter) =>
                {
                    saw_setfield_raw = true
                }
                OpCode::GetfieldRawI
                    if op.getarglist().first().map(|a| a.to_opref()) == Some(iter) =>
                {
                    saw_getfield_raw = true
                }
                OpCode::New => saw_new = true,
                OpCode::GuardNonnull | OpCode::GuardIsnull => saw_optional_guard = true,
                _ => {}
            }
        }
        assert!(saw_getfield_gc);
        assert!(saw_setfield_gc);
        assert!(!saw_setfield_raw);
        assert!(!saw_getfield_raw);
        assert!(!saw_new);
        assert!(!saw_optional_guard);
    }

    #[test]
    fn test_eval_simple_addition() {
        let source = "x = 1 + 2";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let x = frame_global(&frame, "x");
            assert_eq!(pyre_object::intobject::w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_eval_while_loop() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
i = 0
s = 0
while i < 20:
    s = s + i
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = frame_global(&frame, "s");
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 190);
        }
    }

    #[test]
    fn test_eval_with_jit_redecodes_opargs_after_extended_arg_jumps() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let mut source = String::from(
            "\
i = 0
acc = 0
if i == 1:
",
        );
        for _ in 0..80 {
            source.push_str("    acc = acc + 1000\n");
        }
        source.push_str(
            "\
while i < 6:
    acc = acc + 1
    i = i + 1
r = acc",
        );
        let code = pyre_interpreter::compile_exec(&source).expect("compile failed");
        assert!(
            code.instructions.windows(2).any(|pair| {
                matches!(
                    pair[0].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) && !matches!(
                    pair[1].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                )
            }),
            "expected an instruction with an ExtendedArg prefix"
        );
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut state = pyre_interpreter::OpArgState::default();
            for (pc, unit) in code.instructions.iter().copied().enumerate() {
                let (instr, oparg) = state.get(unit);
                eprintln!("{pc:03}: {instr:?} oparg={oparg:?}");
            }
            for (pc, pair) in code.instructions.windows(2).enumerate() {
                if matches!(
                    pair[0].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) && !matches!(
                    pair[1].op,
                    pyre_interpreter::bytecode::Instruction::ExtendedArg
                ) {
                    let target_pc = pc + 1;
                    eprintln!(
                        "decode[{target_pc}] = {:?}",
                        pyre_interpreter::decode_instruction_at(&code, target_pc)
                    );
                    break;
                }
            }
        }
        let mut frame = PyFrame::new(code);
        let result = eval_with_jit(&mut frame);
        if std::env::var_os("MAJIT_DUMP_BYTECODE").is_some() {
            let mut keys: Vec<String> =
                unsafe { pyre_object::w_dict_str_entries(frame.get_w_globals()) }
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect();
            keys.sort();
            eprintln!("module result: {:?}", result);
            eprintln!("module namespace keys: {:?}", keys);
        }
        unsafe {
            let r = frame_global(&frame, "r");
            assert_eq!(pyre_object::intobject::w_int_get_value(r), 6);
        }
    }

    /// Regression test for the recursive portal Ref ABI.
    ///
    /// RPython portal return type is always REF (warmspot.py:449).
    /// The self-recursive call uses CALL_ASSEMBLER_R, FINISH records with
    /// done_with_this_frame_descr_ref, and the caller unboxes via
    /// GuardClass + GetfieldGcPureI (pyjitpl.py:3198-3220).
    ///
    /// A previous bug used CALL_ASSEMBLER_I + FINISH(Int) + forced unbox
    /// at the blackhole boundary, causing pointer-like-integer corruption
    /// in the recursive return path.
    #[test]
    fn test_recursive_fib_returns_correct_result_through_jit() {
        let source = "\
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(12)
";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let result = frame_global(&frame, "result");
            assert_eq!(
                pyre_object::intobject::w_int_get_value(result),
                144,
                "fib(12) should be 144 — recursive portal Ref ABI regression"
            );
        }
    }

    #[test]
    #[cfg_attr(
        feature = "cranelift",
        ignore = "cranelift CALL_ASSEMBLER Rust trampoline \
                  (call_assembler_guard_failure_inner / call_assembler_shim_inner) \
                  adds a native frame per recursive compiled entry. At the low JIT \
                  threshold used here, g(9)×2 runs enough compiled invocations to \
                  overflow the 2 MiB default cargo-test thread stack. Dynasm is \
                  unaffected (jmp trampoline). See \
                  memory/fib_recursive_sigbus_2026_04_19.md."
    )]
    fn test_recursive_global_reads_do_not_reuse_force_cache_across_global_mutation() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
factor = 1
def g(n):
    if n < 2:
        return n * factor
    return g(n - 1) + g(n - 2) + factor

first = g(9)
factor = 2
second = g(9)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        // Production shape (pyrex real_main): the frame carries an
        // ExecutionContext and the TLS slot is seeded, so
        // `getexecutioncontext().gettopframe()` is live when the
        // self-recursive CALL_ASSEMBLER path concretely executes the
        // recursive `g(n - 1)` during the walk (`bh_call_fn_impl`
        // resolves the parent frame from it).  A bare `PyFrame::new`
        // frame is never entered onto the EC and trips the fail-fast
        // topframe assert — same fixture shape as
        // `test_nested_direct_helper_calls_stay_correct`.
        let execution_context = std::rc::Rc::new(pyre_interpreter::PyExecutionContext::default());
        pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
        let mut frame =
            pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context)
                .expect("frame construction failed");
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let first = frame_global(&frame, "first");
            let second = frame_global(&frame, "second");
            assert_eq!(pyre_object::intobject::w_int_get_value(first), 88);
            assert_eq!(pyre_object::intobject::w_int_get_value(second), 176);
        }
    }

    #[test]
    fn test_inline_residual_user_call_with_many_args_stays_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
def helper(a, b, c, d, e):
    return a + b + c + d + e

def outer(x):
    return helper(x, x, x, x, x)

s = 0
i = 0
while i < 40:
    s = s + outer(i)
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        // Production shape (see `test_nested_direct_helper_calls_stay_correct`):
        // the module-level loop is a portal (interp_jit.py:81-99 applies the
        // jitdriver to every frame), so the full-body walk concrete-executes
        // the `outer(i)` residual during tracing — `bh_call_fn_impl` resolves
        // the parent frame from `getexecutioncontext().gettopframe()`, which a
        // bare `PyFrame::new` frame never seeds.
        let execution_context = std::rc::Rc::new(pyre_interpreter::PyExecutionContext::default());
        pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
        let mut frame =
            pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context)
                .expect("frame construction failed");
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = frame_global(&frame, "s");
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 3_900);
        }
    }

    #[test]
    fn test_nested_direct_helper_calls_stay_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(i):
    return add(square(i), i)

s = 0
i = 0
while i < 40:
    s = add(s, compute(i))
    i = add(i, 1)";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        // Production shape (pyrex real_main): the frame carries an
        // ExecutionContext and the TLS slot is seeded, so
        // `getexecutioncontext().gettopframe()` is live during blackhole
        // resume — `bh_call_fn_impl` resolves the parent frame from it
        // when a guard deopt re-executes a `call_fn` residual.  A bare
        // `PyFrame::new` frame is never entered onto the EC and trips
        // the fail-fast topframe assert.
        let execution_context = std::rc::Rc::new(pyre_interpreter::PyExecutionContext::default());
        pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
        let mut frame =
            pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context)
                .expect("frame construction failed");
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = frame_global(&frame, "s");
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 21_320);
        }
    }

    /// rclass.py:1133-1137 `ll_issubclass(subcls, cls)` parity. After
    /// `set_gc_allocator` runs `freeze_types`, the materialized
    /// `(subclassrange_min, subclassrange_max)` for each registered
    /// PyType must satisfy `int_between(cls.min, subcls.min, cls.max)`
    /// for every (cls, subcls) pair where `subcls` Python-inherits from
    /// `cls`. This test exercises the `assign_inheritance_ids`
    /// (normalizecalls.py:373-389) preorder walk by verifying:
    ///   1. `INSTANCE_TYPE` (root `object`) range contains every other
    ///      PyType's range.
    ///   2. `INT_TYPE` range contains `BOOL_TYPE` range
    ///      (`bool.__bases__ == (int,)`).
    ///   3. Sibling classes (`INT_TYPE` vs `FLOAT_TYPE`, `STR_TYPE` vs
    ///      `LIST_TYPE`) are disjoint.
    #[test]
    fn test_subclass_range_preorder_bounds() {
        // Force JIT_DRIVER initialization so set_gc_allocator runs and
        // installs the active subclass_range hook.
        let _ = driver_pair();

        fn range(t: &pyre_object::pyobject::PyType) -> (i64, i64) {
            majit_gc::subclass_range(t as *const _ as usize)
                .expect("every built-in PyType must be registered with the GC")
        }

        // ll_issubclass(subcls, cls): a <= b < c.
        let contains = |outer: (i64, i64), inner: (i64, i64)| {
            outer.0 <= inner.0 && inner.0 < outer.1 && inner.1 <= outer.1
        };
        let disjoint = |a: (i64, i64), b: (i64, i64)| a.1 <= b.0 || b.1 <= a.0;

        let object_r = range(&pyre_object::pyobject::INSTANCE_TYPE);
        let int_r = range(&pyre_object::pyobject::INT_TYPE);
        let float_r = range(&pyre_object::pyobject::FLOAT_TYPE);
        let bool_r = range(&pyre_object::pyobject::BOOL_TYPE);
        let str_r = range(&pyre_object::pyobject::STR_TYPE);
        let list_r = range(&pyre_object::pyobject::LIST_TYPE);
        let none_r = range(&pyre_object::pyobject::NONE_TYPE);

        // (1) object encompasses every descendant.
        assert!(contains(object_r, int_r), "object ⊇ int");
        assert!(contains(object_r, float_r), "object ⊇ float");
        assert!(contains(object_r, bool_r), "object ⊇ bool");
        assert!(contains(object_r, str_r), "object ⊇ str");
        assert!(contains(object_r, list_r), "object ⊇ list");
        assert!(contains(object_r, none_r), "object ⊇ NoneType");

        // (2) int ⊇ bool (PyPy: W_BoolObject inherits from W_IntObject).
        assert!(contains(int_r, bool_r), "int ⊇ bool");

        // (3) Disjoint siblings.
        assert!(disjoint(int_r, float_r), "int ⊥ float");
        assert!(disjoint(int_r, str_r), "int ⊥ str");
        assert!(disjoint(float_r, str_r), "float ⊥ str");
        assert!(disjoint(str_r, list_r), "str ⊥ list");
        assert!(disjoint(float_r, bool_r), "float ⊥ bool");

        // (4) rclass.py:340-346 parity: subclassrange_{min,max} assigned
        // directly on the PyType (OBJECT_VTABLE) struct, not only in
        // the GC's TypeInfo table. ll_issubclass reads them from the
        // typeptr without a GC indirection.
        use pyre_object::pyobject::{BOOL_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE};
        use std::sync::atomic::Ordering;
        assert_eq!(
            INSTANCE_TYPE.subclassrange_min.load(Ordering::Relaxed),
            object_r.0
        );
        assert_eq!(
            INSTANCE_TYPE.subclassrange_max.load(Ordering::Relaxed),
            object_r.1
        );
        assert_eq!(INT_TYPE.subclassrange_min.load(Ordering::Relaxed), int_r.0);
        assert_eq!(INT_TYPE.subclassrange_max.load(Ordering::Relaxed), int_r.1);
        assert_eq!(
            BOOL_TYPE.subclassrange_min.load(Ordering::Relaxed),
            bool_r.0
        );
        assert_eq!(
            BOOL_TYPE.subclassrange_max.load(Ordering::Relaxed),
            bool_r.1
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_min.load(Ordering::Relaxed),
            float_r.0
        );
        assert_eq!(
            FLOAT_TYPE.subclassrange_max.load(Ordering::Relaxed),
            float_r.1
        );

        // (5) ll_issubclass direct PyType reads match GC callback.
        assert!(pyre_object::pyobject::ll_issubclass(&BOOL_TYPE, &INT_TYPE));
        assert!(pyre_object::pyobject::ll_issubclass(
            &INT_TYPE,
            &INSTANCE_TYPE
        ));
        assert!(!pyre_object::pyobject::ll_issubclass(
            &INT_TYPE,
            &FLOAT_TYPE
        ));
        assert!(!pyre_object::pyobject::ll_issubclass(
            &FLOAT_TYPE,
            &INT_TYPE
        ));
    }

    #[test]
    fn test_dynamic_int_list_indexing_stays_correct() {
        let _jit_params = TestJitParamsGuard::low_threshold();
        let source = "\
q = [0, 1, 2, 3, 4]
i = 0
s = 0
while i < 40:
    q0 = i % 5
    s = s + q[q0]
    q[q0] = q[q0] + 1
    i = i + 1";
        let code = pyre_interpreter::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = frame_global(&frame, "s");
            let q = frame_global(&frame, "q");
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 220);
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 0).unwrap()
                ),
                8
            );
            assert_eq!(
                pyre_object::intobject::w_int_get_value(
                    pyre_object::listobject::w_list_getitem(q, 4).unwrap()
                ),
                12
            );
        }
    }
}
