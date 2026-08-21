/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `history.py:125 id(descr)` parity — when the optimizer
    /// (`store_final_boxes_in_guard` / `make_and_attach_done_descrs`)
    /// stamps a metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
    /// `op.descr`, we keep it here so `get_latest_descr_arc` returns the
    /// canonical metainterp Arc (matching dynasm/cranelift).  `None`
    /// for synthetic backend-only descrs (`compile_bridge` placeholders,
    /// test scaffolds).
    pub meta_descr: Option<DescrRef>,
}

impl Descr for WasmFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for WasmFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame::Boxed` after `execute_token` returns.
pub struct WasmFrameData {
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
    /// Pending exception value captured by `execute_token` after the trace
    /// exited through a GuardNoException / GuardException (0 = none), surfaced
    /// via `grab_exc_value`.
    pub exc_value: i64,
    /// Slots handed to [`crate::wasm_gc_add_roots`] by [`WasmFrameData::boxed`],
    /// released again in `Drop`.
    roots: Vec<usize>,
}

impl WasmFrameData {
    /// `llmodel.py` reads `get_ref_value` straight out of the JITFRAME,
    /// which stays a GC root (its `jf_gcmap` covers the exit slots) for as long
    /// as the deadframe lives. wasm has no host-visible JITFRAME to hand back:
    /// `execute_token` copies the exit values into `raw_values` and drops the
    /// guest frame, so the copies must carry that rooting themselves. Between
    /// the copy and the last `get_ref_value`, resume/blackhole reconstruction
    /// allocates freely, and a minor collection there moves exactly the objects
    /// these slots name.
    ///
    /// Only `Type::Ref` exit slots are rooted, matching the gcmap the guest
    /// frame carried. A wasm32 `GcRef` occupies the low half of its `i64` slot,
    /// so the root address is the slot address (same aliasing the Ref home
    /// slots already rely on).
    pub fn boxed(
        raw_values: Vec<i64>,
        fail_descr: Arc<WasmFailDescr>,
        exc_value: i64,
    ) -> Box<Self> {
        let mut data = Box::new(WasmFrameData {
            raw_values,
            fail_descr,
            exc_value,
            roots: Vec::new(),
        });
        let ref_count = data
            .fail_descr
            .fail_arg_types
            .iter()
            .take(data.raw_values.len())
            .filter(|ty| **ty == Type::Ref)
            .count();
        if ref_count != 0 || data.exc_value != 0 {
            let mut roots = Vec::with_capacity(ref_count + usize::from(data.exc_value != 0));
            for i in 0..data.raw_values.len() {
                if data.fail_descr.fail_arg_types.get(i) == Some(&Type::Ref) {
                    roots.push(&mut data.raw_values[i] as *mut i64 as usize);
                }
            }
            // `grab_exc_value` hands this out as a `GcRef` too, and the resume path
            // reads it after it has already allocated. A null `GcRef` needs no root.
            if data.exc_value != 0 {
                roots.push(&mut data.exc_value as *mut i64 as usize);
            }
            unsafe { crate::wasm_gc_add_roots(&roots) };
            data.roots = roots;
        }
        data
    }
}

impl Drop for WasmFrameData {
    fn drop(&mut self) {
        if self.roots.is_empty() {
            return;
        }
        // Remove in reverse push order so RootSet::remove stays on its
        // O(1) stack-pop path.
        crate::wasm_gc_remove_roots(self.roots.drain(..).rev());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use majit_gc::GcAllocator;
    use majit_ir::GcRef;

    use super::{Type, WasmFailDescr, WasmFrameData};
    use super::{fail_descr_base, global_fail_descr, register_fail_descrs};

    struct RootCountingGc(Arc<AtomicUsize>);

    impl GcAllocator for RootCountingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_nursery_no_collect(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize(&mut self, _base_size: usize, _item_size: usize, _length: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize_no_collect(
            &mut self,
            _base_size: usize,
            _item_size: usize,
            _length: usize,
        ) -> GcRef {
            GcRef(0)
        }

        fn write_barrier(&mut self, _obj: GcRef) {}

        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}

        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }

        fn collect_nursery(&mut self) {}

        fn collect_full(&mut self) {}

        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }

        fn nursery_free_addr(&self) -> usize {
            0
        }

        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }

        fn nursery_top_addr(&self) -> usize {
            0
        }

        fn max_nursery_object_size(&self) -> usize {
            0
        }

        unsafe fn add_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn remove_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn install_root_counting_gc() -> Arc<AtomicUsize> {
        let roots = Arc::new(AtomicUsize::new(0));
        crate::install_gc_box(Box::new(RootCountingGc(Arc::clone(&roots))));
        roots
    }

    fn fail_descr(fail_arg_types: Vec<Type>) -> Arc<WasmFailDescr> {
        Arc::new(WasmFailDescr {
            fail_index: 0,
            trace_id: 0,
            fail_arg_types,
            is_finish: false,
            meta_descr: None,
        })
    }

    #[test]
    fn a_finish_singleton_resolves_to_its_reserved_exit() {
        // The emitted FINISH writes the index this returns and the emitted
        // CALL_ASSEMBLER check compares against the same constant, so a
        // singleton that failed to bind would send every clean callee finish
        // back to the host to be decoded and handed straight over.
        let descr: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        super::attach_finish_descr(super::FINISH_EXIT_INDEX_REF, Arc::clone(&descr));
        assert_eq!(
            super::attached_finish_exit_index(&Some(Arc::clone(&descr))),
            Some(super::FINISH_EXIT_INDEX_REF),
        );
        assert_eq!(
            super::done_with_this_frame_exit_index(Type::Ref),
            super::FINISH_EXIT_INDEX_REF,
        );
        // A descr this cpu was never handed has no shared identity, so it keeps
        // its own exit.
        let unattached: majit_ir::DescrRef =
            Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        assert_eq!(super::attached_finish_exit_index(&Some(unattached)), None);
    }

    #[test]
    fn the_reserved_finish_exits_precede_every_trace_base() {
        // The emitted CALL_ASSEMBLER check compares against a baked reserved
        // index, so a trace whose own exits started below the reserved block
        // would collide with it.
        let base = fail_descr_base();
        assert!(base >= super::FINISH_EXIT_INDEX_COUNT);
        for (index, types) in [
            (super::FINISH_EXIT_INDEX_VOID, &[][..]),
            (super::FINISH_EXIT_INDEX_INT, &[Type::Int][..]),
            (super::FINISH_EXIT_INDEX_REF, &[Type::Ref][..]),
            (super::FINISH_EXIT_INDEX_FLOAT, &[Type::Float][..]),
            (super::FINISH_EXIT_INDEX_EXC, &[Type::Ref][..]),
        ] {
            let descr = global_fail_descr(index).expect("reserved finish exit is unregistered");
            assert_eq!(descr.fail_index, index);
            assert!(descr.is_finish, "reserved exit {index} is not a finish");
            assert_eq!(descr.fail_arg_types, types, "reserved exit {index} layout");
        }

        let descrs: Vec<Arc<WasmFailDescr>> = (0..3)
            .map(|i| {
                Arc::new(WasmFailDescr {
                    fail_index: base + i,
                    trace_id: 0,
                    fail_arg_types: vec![Type::Ref],
                    is_finish: false,
                    meta_descr: None,
                })
            })
            .collect();
        register_fail_descrs(&descrs);
        for i in 0..3 {
            assert_eq!(
                global_fail_descr(base + i)
                    .expect("trace exit is unregistered")
                    .fail_index,
                base + i,
            );
        }
    }

    #[test]
    fn boxed_roots_ref_slots_and_nonzero_exception_until_drop() {
        let roots = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(
            vec![0x10, 42, 0, 0x20],
            fail_descr(vec![Type::Ref, Type::Int, Type::Float, Type::Ref]),
            0x30,
        );
        assert_eq!(roots.load(Ordering::SeqCst), before + 3);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn boxed_without_refs_or_exception_does_not_bracket_roots() {
        let roots = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(vec![1, 2], fail_descr(vec![Type::Int, Type::Float]), 0);
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }
}

/// A resumable `LABEL` of a compiled loop, published in `LABEL_TARGETS` so a
/// loop-closing bridge can chain into ANY compiled loop's label in-module
/// (jump-to-existing-trace), not only its own source loop's. Keyed by the
/// label's loop-target descr identity (`Arc::as_ptr`), which the JUMP shares.
#[derive(Clone, Copy, Debug)]
pub struct LabelTarget {
    /// Table slot of the owning loop's compiled function.
    pub func_handle: u32,
    /// Resume dispatch key (`label ordinal + 1`) the bridge's JUMP writes.
    pub key: u32,
    /// The label's arg count — the resume loader reads exactly this many
    /// positional frame slots, so the JUMP arity must equal it.
    pub num_args: usize,
    /// Whether every live-in can be reconstructed from LABEL args or frozen
    /// backend capture slots (`codegen::label_resume_info`).
    pub resume_safe: bool,
    /// Backend capture slots are populated by this target loop's own
    /// fall-through path. If true, a bridge may resume only its source loop,
    /// not a sibling specialization that happens to share the geometry.
    pub requires_own_frame: bool,
    /// Whether this is the owning loop's LAST label (the loop header). A
    /// bridge landing here re-runs no segment code before the `loop`, so the
    /// livelock advance-check applies; earlier labels execute the peeled
    /// segment, which advances the state by itself.
    pub is_last_label: bool,
    /// Frozen frame geometry of the target token. A tail-call can only reuse
    /// a frame when its offsets agree exactly, not merely when its allocation
    /// is large enough.
    pub frame: crate::codegen::FrameGeometry,
}

/// Frozen metadata for entering a compiled loop from a `CALL_ASSEMBLER` arm.
/// The table slot and frame layout are published only after the loop module is
/// installed, so a caller can decline before baking an unresolved target.
#[derive(Clone, Debug)]
pub struct CallAssemblerTarget {
    /// Owning `JitCellToken` number. Used only by the dormant wasm regression
    /// hook to select one target deterministically.
    pub token_number: u64,
    pub func_handle: u32,
    pub input_types: Vec<Type>,
    pub callee_frame_bytes: u32,
    pub callee_gcmap_ptr: i64,
    pub compiled_ptr: u64,
}

/// Compiled loop targets keyed by their `JitCellToken` number. Unlike label
/// targets, CALL_ASSEMBLER identifies its callee by that number directly.
pub static CALL_ASSEMBLER_TARGETS: std::sync::Mutex<
    Option<std::collections::HashMap<u64, CallAssemblerTarget>>,
> = std::sync::Mutex::new(None);

// ── CALL_ASSEMBLER dispatch table ──
//
// A trace module imports the guest's linear memory, so a boxed wasm-side
// allocation is addressable by every trace with an ordinary i32.load.  The
// box keeps that address stable while the map grows; the emitted code must
// never bake a table slot because redirects and the pending->real transition
// replace it after the caller module was compiled.
#[repr(C)]
pub struct WasmCaDispatchEntry {
    /// `__indirect_function_table` slot. Zero means pending/unavailable.
    pub func_handle: AtomicU32,
    /// `CompiledWasmLoop` address for the deopt helper, in wasm32 memory.
    pub compiled_ptr: AtomicU32,
}

pub const WASM_CA_DISPATCH_FUNC_HANDLE_OFS: u64 = 0;
pub const WASM_CA_DISPATCH_COMPILED_PTR_OFS: u64 = 4;

/// `make_and_attach_done_descrs` gives every cpu one `DoneWithThisFrame*` per
/// result kind plus one `ExitFrameWithExceptionDescrRef`, and
/// `compile_done_with_this_frame` / `compile_exit_frame_with_exception` stamp
/// that singleton on the FINISH — so every trace that finishes the same way
/// writes the same `jf_descr`, and `_call_assembler_check_descr` recognises a
/// clean callee finish by comparing against one value.
///
/// A wasm frame slot holds an index into the global exit space rather than a
/// descr pointer, so the shared identity is a reserved index. The five sit at
/// the front of [`FAIL_DESCR_REGISTRY`], claimed before any trace takes a
/// `fail_descr_base`, and carry the attached `Arc` as their `meta_descr` so
/// `get_latest_descr_arc` still answers with the metainterp's own descr.
pub const FINISH_EXIT_INDEX_VOID: u32 = 0;
pub const FINISH_EXIT_INDEX_INT: u32 = 1;
pub const FINISH_EXIT_INDEX_REF: u32 = 2;
pub const FINISH_EXIT_INDEX_FLOAT: u32 = 3;
pub const FINISH_EXIT_INDEX_EXC: u32 = 4;
const FINISH_EXIT_INDEX_COUNT: u32 = 5;

/// Reserved exit index for the `done_with_this_frame_descr_*` of `ty`.
pub fn done_with_this_frame_exit_index(ty: Type) -> u32 {
    match ty {
        Type::Void => FINISH_EXIT_INDEX_VOID,
        Type::Int => FINISH_EXIT_INDEX_INT,
        Type::Ref => FINISH_EXIT_INDEX_REF,
        Type::Float => FINISH_EXIT_INDEX_FLOAT,
    }
}

/// The four `DoneWithThisFrameDescr*` classes carry the one result of their
/// kind; `ExitFrameWithExceptionDescrRef` carries the exception Ref.
fn reserved_fail_arg_types(exit_index: u32) -> Vec<Type> {
    match exit_index {
        FINISH_EXIT_INDEX_VOID => Vec::new(),
        FINISH_EXIT_INDEX_INT => vec![Type::Int],
        FINISH_EXIT_INDEX_FLOAT => vec![Type::Float],
        _ => vec![Type::Ref],
    }
}

fn reserved_finish_descr(exit_index: u32, meta_descr: Option<DescrRef>) -> Arc<WasmFailDescr> {
    Arc::new(WasmFailDescr {
        fail_index: exit_index,
        trace_id: 0,
        fail_arg_types: reserved_fail_arg_types(exit_index),
        is_finish: true,
        meta_descr,
    })
}

/// Claim the reserved block if the registry has not been opened yet. Called
/// under the registry lock from every entry point that can grow or read it, so
/// a trace can never take a `fail_descr_base` below `FINISH_EXIT_INDEX_COUNT`.
fn reserve_finish_exit_block(vec: &mut Vec<Arc<WasmFailDescr>>) {
    if !vec.is_empty() {
        return;
    }
    for index in 0..FINISH_EXIT_INDEX_COUNT {
        vec.push(reserved_finish_descr(index, None));
    }
}

fn thin_descr_ptr(descr: &DescrRef) -> usize {
    Arc::as_ptr(descr) as *const () as usize
}

/// `make_and_attach_done_descrs` pointer identity: the reserved exit index for
/// `descr` when it is one of the five singletons this cpu was handed, else
/// `None`.
/// Mirrors `AttachedDescrPtrs::is_done_with_this_frame_descr`, which is how the
/// native backends recognise the same descrs on the FINISH fast path.
///
/// Answered from the reserved entries themselves, so an index this returns
/// always names a registry entry carrying `descr` — `get_latest_descr_arc`
/// keeps its `AbstractDescr` identity whatever order attachment and the first
/// compile happened in.
pub fn attached_finish_exit_index(descr: &Option<DescrRef>) -> Option<u32> {
    let ptr = thin_descr_ptr(descr.as_ref()?);
    let mut reg = FAIL_DESCR_REGISTRY.lock().unwrap();
    let vec = reg.get_or_insert_with(Default::default);
    reserve_finish_exit_block(vec);
    vec[..FINISH_EXIT_INDEX_COUNT as usize]
        .iter()
        .position(|reserved| {
            reserved
                .meta_descr
                .as_ref()
                .is_some_and(|attached| thin_descr_ptr(attached) == ptr)
        })
        .map(|index| index as u32)
}

/// `make_and_attach_done_descrs`' per-target attachment for one of the five.
/// Binds the singleton to its reserved exit, which is what the emitted FINISH
/// writes and the emitted CALL_ASSEMBLER check compares against. Rebinding an
/// already-claimed entry keeps the fast path available to a process that
/// compiled something before the attachment landed.
pub fn attach_finish_descr(exit_index: u32, descr: DescrRef) {
    let mut reg = FAIL_DESCR_REGISTRY.lock().unwrap();
    let vec = reg.get_or_insert_with(Default::default);
    reserve_finish_exit_block(vec);
    vec[exit_index as usize] = reserved_finish_descr(exit_index, Some(descr));
}

/// Stable, guest-memory dispatch entries, keyed by CALL_ASSEMBLER token.
/// `Box` is intentional: an emitted module bakes the entry address.
pub static WASM_CA_DISPATCH: std::sync::Mutex<
    Option<std::collections::HashMap<u64, Box<WasmCaDispatchEntry>>>,
> = std::sync::Mutex::new(None);

/// Return the stable guest-memory address for `number`, creating a pending
/// (zero-slot) entry when needed.
pub fn ca_dispatch_slot(number: u64) -> u32 {
    let mut table = WASM_CA_DISPATCH.lock().unwrap();
    let entry = table
        .get_or_insert_with(Default::default)
        .entry(number)
        .or_insert_with(|| {
            Box::new(WasmCaDispatchEntry {
                func_handle: AtomicU32::new(0),
                compiled_ptr: AtomicU32::new(0),
            })
        });
    (&**entry as *const WasmCaDispatchEntry as usize) as u32
}

pub fn ca_dispatch_exists(number: u64) -> bool {
    WASM_CA_DISPATCH
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|table| table.contains_key(&number))
}

/// Publish an installed loop after its module has acquired a shared-table
/// slot. Release stores pair with the runtime loads in emitted trace modules;
/// wasm execution cannot begin until this compile call returns.
pub fn ca_dispatch_publish(number: u64, func_handle: u32, compiled_ptr: u32) {
    let _ = ca_dispatch_slot(number);
    let table = WASM_CA_DISPATCH.lock().unwrap();
    let entry = table
        .as_ref()
        .and_then(|table| table.get(&number))
        .expect("CALL_ASSEMBLER dispatch entry disappeared while publishing");
    entry.compiled_ptr.store(compiled_ptr, Ordering::Release);
    entry.func_handle.store(func_handle, Ordering::Release);
}

/// Redirect existing callers of `old_number` to the installed target.
pub fn ca_dispatch_redirect(old_number: u64, func_handle: u32, compiled_ptr: u32) {
    ca_dispatch_publish(old_number, func_handle, compiled_ptr);
}

/// Remove every dispatch entry that still resolves to `compiled_ptr`.  This
/// also retracts redirects into a dropped replacement loop, while preserving
/// an old token whose entry has already been redirected elsewhere.
pub fn ca_dispatch_remove_compiled_ptr(compiled_ptr: u32) {
    if let Some(table) = WASM_CA_DISPATCH.lock().unwrap().as_mut() {
        table.retain(|_, entry| entry.compiled_ptr.load(Ordering::Acquire) != compiled_ptr);
    }
}

pub fn ca_dispatch_remove(number: u64) {
    if let Some(table) = WASM_CA_DISPATCH.lock().unwrap().as_mut() {
        table.remove(&number);
    }
}

pub fn call_assembler_target(number: u64) -> Option<CallAssemblerTarget> {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|targets| targets.get(&number).cloned())
}

pub fn publish_call_assembler_target(number: u64, target: CallAssemblerTarget) {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(number, target);
}

/// Register the frontend's compile_tmp_callback placeholder.  The geometry is
/// deliberately zero: Stage-1 admission continues to require a fully live
/// target, while the stable dispatch entry lets a later self-recursive compile
/// use this token without baking a transient table slot.
pub fn register_pending_call_assembler_target(number: u64, input_types: Vec<Type>) {
    ca_dispatch_remove(number);
    let _ = ca_dispatch_slot(number);
    publish_call_assembler_target(
        number,
        CallAssemblerTarget {
            token_number: number,
            func_handle: 0,
            input_types,
            callee_frame_bytes: 0,
            callee_gcmap_ptr: 0,
            compiled_ptr: 0,
        },
    );
}

/// Remove metadata and the dispatch entry for an invalidated token.
pub fn remove_call_assembler_target(number: u64) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().unwrap().as_mut() {
        targets.remove(&number);
    }
    ca_dispatch_remove(number);
}

/// Retract all metadata aliases which point at a dropped compiled loop.
pub fn remove_call_assembler_targets_for_compiled_ptr(compiled_ptr: u32) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().unwrap().as_mut() {
        targets.retain(|_, target| target.compiled_ptr as u32 != compiled_ptr);
    }
    ca_dispatch_remove_compiled_ptr(compiled_ptr);
}

/// Global `frame[0]` fail-index space.
///
/// Cross-trace chaining (`LABEL_TARGETS`) means the module that last wrote
/// `frame[0]` is not necessarily the loop `execute_token` entered: a bridge's
/// terminal JUMP may tail-call a SIBLING loop, whose guards then write THEIR
/// exit indices. Per-loop index spaces would make those writes ambiguous at
/// the host — resolving `frame[0]` against the entry loop's own `fail_descrs`
/// picks a wrong descr (wrong arg types/resume ⇒ type confusion). So every
/// compile (`compile_loop` and `compile_bridge`) allocates its exits from this
/// one global space: it passes the registry length as codegen's
/// `fail_index_base`, guards write `base + local` into `frame[0]`, and the
/// registered descrs land at exactly those registry positions — any `frame[0]`
/// then resolves here regardless of which chained module wrote it. The
/// per-guard bridge-cell epilogue keeps its local indexing by subtracting the
/// owning module's base (`codegen`'s cell lookup).
///
/// Entries are never removed: a dropped loop's modules are unreachable (its
/// label targets are retracted and its token is gone), so its entries are just
/// retained memory, bounded by the total number of compiled exits.
static FAIL_DESCR_REGISTRY: std::sync::Mutex<Option<Vec<Arc<WasmFailDescr>>>> =
    std::sync::Mutex::new(None);

/// The next free global fail index — pass as `fail_index_base` to
/// `codegen::build_wasm_module`, then register the built descrs with
/// `register_fail_descrs`. The wasm host is single-threaded, so no other
/// compile can interleave between the two calls.
pub fn fail_descr_base() -> u32 {
    let mut reg = FAIL_DESCR_REGISTRY.lock().unwrap();
    let vec = reg.get_or_insert_with(Default::default);
    reserve_finish_exit_block(vec);
    vec.len() as u32
}

/// Append a compile's exit descrs to the global space. Each descr's
/// `fail_index` (already base-offset by `build_wasm_module`) must equal the
/// registry position it lands at.
pub fn register_fail_descrs(descrs: &[Arc<WasmFailDescr>]) {
    let mut reg = FAIL_DESCR_REGISTRY.lock().unwrap();
    let vec = reg.get_or_insert_with(Default::default);
    reserve_finish_exit_block(vec);
    for d in descrs {
        debug_assert_eq!(
            d.fail_index as usize,
            vec.len(),
            "fail descr registered out of lockstep with its global fail_index"
        );
        vec.push(Arc::clone(d));
    }
}

/// Resolve a `frame[0]` value through the global fail-index space.
pub fn global_fail_descr(fail_index: u32) -> Option<Arc<WasmFailDescr>> {
    FAIL_DESCR_REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|v| v.get(fail_index as usize).cloned())
}

/// Global `label descr identity → LabelTarget` registry (see `LabelTarget`).
/// The wasm host is single-threaded; the `Mutex` is for `static` soundness
/// only. `compile_loop` inserts every resumable label of a peeled loop;
/// `CompiledWasmLoop::drop` removes its own entries (guarded by
/// `func_handle`, so a recompile that re-stamped the same descr keeps the
/// replacement's entry).
pub static LABEL_TARGETS: std::sync::Mutex<Option<std::collections::HashMap<usize, LabelTarget>>> =
    std::sync::Mutex::new(None);

/// Look up a label target by descr identity.
pub fn label_target(descr_id: usize) -> Option<LabelTarget> {
    LABEL_TARGETS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&descr_id).copied())
}

/// Publish a label target (see `LABEL_TARGETS`).
pub fn publish_label_target(descr_id: usize, target: LabelTarget) {
    LABEL_TARGETS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(descr_id, target);
}

/// Guard-dispatch metadata of a bridge chained onto a loop, kept on the
/// source loop's `CompiledWasmLoop.chained_trace_meta` keyed by the bridge's
/// backend `trace_id`. Lets `compile_bridge` chain a NESTED sub-bridge onto a
/// guard that lives inside an already-chained bridge: the failing guard's
/// meta descr carries `(trace_id, per-trace fail_index)`, and this record
/// supplies the owning bridge's cell array and livelock advance flags — the
/// same data `CompiledWasmLoop` holds for the loop's own guards.
pub struct ChainedTraceMeta {
    /// Base address of the bridge's per-guard bridge-slot cell array
    /// (`CompiledWasmLoop::bridge_cells_base` analog); `0` = no dispatch.
    pub cells_base: u32,
    /// Cell count = the bridge's own guard count.
    pub num_cells: usize,
    /// Per-guard, per-fail-arg induction-advance flags
    /// (`CompiledWasmLoop::guard_fail_arg_advanced` analog).
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// Number of values each guard transfers to a bridge.  A parameter entry
    /// is admitted only when this agrees with the bridge's input list.
    pub guard_fail_arg_counts: Vec<usize>,
    /// Whether this trace's guard epilogue has typed parameter dispatch arms.
    pub bridge_param_dispatch: bool,
}

/// Compiled wasm loop metadata, stored in `JitCellToken.compiled`.
pub struct CompiledWasmLoop {
    /// Owning `JitCellToken` number, used to retract this loop's
    /// CALL_ASSEMBLER target metadata on drop.
    pub token_number: u64,
    pub trace_id: u64,
    pub input_types: Vec<Type>,
    /// Shared-table slot of the materialized wasm function.  Straight-line
    /// function-entry traces may keep this at zero until their first actual
    /// execution: an invalidated trace that never reaches `execute_token`
    /// must not pay the host Wasmtime compilation cost.
    pub(crate) func_handle: Cell<u32>,
    /// Encoded module retained until lazy host materialization.  This is
    /// backend assembler state, not metainterpreter state: the optimized trace
    /// and all per-token descriptors have already been installed exactly as
    /// in the eager path.
    #[cfg_attr(any(not(target_arch = "wasm32"), target_os = "wasi"), allow(dead_code))]
    pub(crate) pending_wasm_bytes: RefCell<Option<Vec<u8>>>,
    /// This loop's own guard/finish exit descriptors (positions `[0,
    /// num_guard_cells)`, per-trace order), followed by the descr slices of
    /// every chained bridge `compile_bridge` appended (positional bookkeeping
    /// for `bridge_descr_ranges` — layouts and jitcounter hashes). `frame[0]`
    /// exit resolution does NOT index this vec: exit indices live in the
    /// GLOBAL fail-index space (`register_fail_descrs`), because a cross-trace
    /// chain can exit through a sibling loop's guard. `RefCell` because the
    /// append happens through the shared `&JitCellToken` the bridge attaches
    /// to; the wasm host is single-threaded so no cross-thread access occurs.
    pub fail_descrs: RefCell<Vec<Arc<WasmFailDescr>>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
    /// Number of Ref-typed values given a home slot in the frame's Ref-home
    /// region (`codegen::HOME_SLOT_BASE`). `execute_token` sizes the host
    /// frame to include this region and registers each home slot as a GC root.
    pub num_ref_homes: usize,
    /// Geometry frozen when this token was first compiled. Every bridge
    /// chained onto it is emitted against this exact layout.
    pub frame: crate::codegen::FrameGeometry,
    /// Base address (shared linear memory) of this loop's per-guard bridge-slot
    /// cell array — one i32 per `fail_index`, `0` = no bridge. The trace's
    /// epilogue reads `cells[fail_index]` and `compile_bridge` writes a bridge's
    /// table slot here. `0` when the trace has no in-module dispatch (native, or
    /// a guardless / straight-line trace).
    pub bridge_cells_base: Cell<u32>,
    /// Number of cells in the `bridge_cells_base` array = this loop's own guard
    /// count at compile time. A bridge attaches only to one of these original
    /// guards (`source_fail_index < num_guard_cells`); descrs appended past this
    /// range belong to already-chained bridges and have no cell of their own.
    pub num_guard_cells: Cell<usize>,
    /// True when this is a peeled loop (`codegen::is_resumable_peeled`) — there
    /// is real work (a preamble = the unrolled first iteration) before the last
    /// `LABEL`, single- or multi-label. Such a loop carries the resume-at-LABEL
    /// entry `br_table` (key = label ordinal + 1) so a loop-closing bridge can
    /// re-enter at any of its labels. A loop-closing bridge re-enters through
    /// the loop's table slot (the function entry); for a peeled loop,
    /// re-running the preamble against mid-loop state would never advance the
    /// induction variable — an infinite loop. `compile_bridge` therefore
    /// declines a loop-closing bridge UNLESS its JUMP's
    /// target label resolves to a published, resumable `LabelTarget`.
    pub has_preamble: bool,
    /// Descr identity (`Arc::as_ptr`) of each `LABEL`, in ordinal order; `0`
    /// for a descr-less label. `compile_bridge` resolves a closing JUMP's
    /// target label by matching its descr identity against this list — a JUMP
    /// whose descr is not here targets ANOTHER trace's label (e.g. a sibling
    /// retrace specialization, whose start label carries the same stamped
    /// ordinal) and must not be chained into this loop.
    pub label_descrs: Vec<usize>,
    /// Per-guard (indexed by this loop's own `fail_index`), per-fail-arg:
    /// whether the value was produced by induction-advancing arithmetic after
    /// the loop-header label — fresh in the failing iteration. Consulted by
    /// `compile_bridge`'s livelock check: a loop-closing bridge that JUMPs
    /// such a fail arg verbatim still advances the chained cycle.
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// Number of fail arguments for every guard/finish exit in this trace.
    pub guard_fail_arg_counts: Vec<usize>,
    /// Whether this module transfers a compiled bridge's fail arguments as
    /// wasm call parameters instead of reloading their positional frame slots.
    pub bridge_param_dispatch: bool,
    /// `(source_trace_id, source_fail_index, start, count)` ranges into
    /// `fail_descrs` for each chained bridge `compile_bridge` appended (lib.rs
    /// extend site). Lets `compiled_bridge_fail_descr_layouts` /
    /// `store_bridge_guard_hashes` map a source guard back to its bridge's
    /// appended descr slice — the wasm analog of dynasm's
    /// `lookup_bridge_addr` (runner.rs). Keyed by BOTH the source guard's
    /// owning trace and its per-trace fail index: with nested chaining, the
    /// loop's guard `k` and a chained bridge's guard `k` are distinct sources.
    /// Recorded in lockstep with the `extend`, inside the same `borrow_mut`
    /// critical section.
    pub bridge_descr_ranges: RefCell<Vec<(u64, u32, usize, usize)>>,
    /// Guard-dispatch metadata of every bridge chained onto this loop, keyed
    /// by the bridge's backend `trace_id` (see [`ChainedTraceMeta`]). Lets a
    /// guard INSIDE a chained bridge chain its own nested sub-bridge.
    pub chained_trace_meta: RefCell<std::collections::HashMap<u64, ChainedTraceMeta>>,
    /// Owns this loop's current cell array and every bridge cell array chained
    /// onto it. A re-emission retains the old array for an already-running
    /// module before switching its baked base to a new array.
    pub _bridge_owned_cells: RefCell<Vec<Box<[u32]>>>,
    /// Direct-loop guard index to bridge table slot. `patch_jump_for_descr`
    /// rewrites the guard's own jump to reach a newly attached bridge; a wasm
    /// module is immutable once compiled, so the branch instead reads a slot
    /// out of a mutable cell array, and these are the writes a re-emission has
    /// to replay into its fresh array.
    pub bridge_slots: RefCell<std::collections::HashMap<u32, u32>>,
    /// The same, for a guard that lives inside a trace chained onto this loop,
    /// keyed by `(owning trace_id, per-trace fail index)`. A standalone chained
    /// bridge keeps its cells in its own module's array, which survives; a
    /// region merged into this loop does not, because a re-emission reallocates
    /// the loop array its guards are carved out of. Replayed once the rebuilt
    /// `chained_trace_meta` names the new bases.
    pub chained_bridge_slots: RefCell<std::collections::HashMap<(u64, u32), u32>>,
    /// Post-intern module inputs retained for a loop re-emission. Entry
    /// bridges store `None` because they tail-call another loop.
    pub reemit: RefCell<Option<crate::codegen::ModuleBuildInputs>>,
    /// The environment-gated identity re-emission runs once per token.
    pub reemitted: Cell<bool>,
    /// `(descr identity, table slot)` for every label published by a bridge
    /// chained onto this loop. The bridge module lives as long as its source
    /// loop, so `Drop` retracts entries that still name that bridge's slot.
    pub bridge_owned_label_targets: RefCell<Vec<(usize, u32)>>,
    /// Set when `compile_bridge` accepts a self-recursive `CallAssemblerR`
    /// bridge (`PYRE_WASM_CA`) for this loop. While set, `compile_bridge`
    /// declines chaining any FURTHER bridge into this recursion (the guard
    /// falls back to host round-trips): a chained bridge deopting inside the
    /// CA recursion trips a resume seam that reads a clobbered class — see
    /// the decline site for the failing suite shapes.
    pub ca_active: Cell<bool>,
    /// A guard reached through this loop as a wasm CALL_ASSEMBLER callee was
    /// structurally declined by `compile_bridge`.  Admission refuses this
    /// target, because entering it from compiled wasm would only blackhole.
    pub ca_terminal_declined: Cell<bool>,
    /// Compiled callers that baked this loop as their CALL_ASSEMBLER target.
    /// A terminal callee decline invalidates them for a no-CA retrace.
    pub ca_callers: RefCell<Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>>,
}

// Compiled loop metadata is transferred through the token's `Any + Send`
// holder, but all access to its IR snapshot and cell arrays is confined to the
// single wasm execution thread. The contained `RefCell`s enforce that runtime
// ownership model; moving the holder does not permit concurrent access.
unsafe impl Send for CompiledWasmLoop {}

impl CompiledWasmLoop {
    pub fn eager_func_handle(&self) -> u32 {
        self.func_handle.get()
    }

    /// Materialize a lazily-installed root trace.  The wasm host is
    /// single-threaded, matching the RefCell/Cell ownership used throughout
    /// this structure, so one trace can only cross this gate once.
    pub fn materialize_func_handle(&self) -> Result<u32, majit_backend::BackendError> {
        let current = self.func_handle.get();
        if current != 0 {
            return Ok(current);
        }
        #[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
        {
            Ok(0)
        }
        #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
        {
            let pending = self.pending_wasm_bytes.borrow();
            let Some(bytes) = pending.as_deref() else {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm trace has neither a function handle nor pending module bytes".into(),
                ));
            };
            let handle = crate::glue::compile_module_cached(bytes);
            if handle == 0 {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm host rejected the lazily compiled trace module".into(),
                ));
            }
            self.func_handle.set(handle);
            drop(pending);
            self.pending_wasm_bytes.borrow_mut().take();
            Ok(handle)
        }
    }
}

impl Drop for CompiledWasmLoop {
    fn drop(&mut self) {
        // Remove every token alias still targeting this module, including a
        // redirect source. A source redirected to a newer module survives an
        // old-loop drop because its dispatch `compiled_ptr` no longer matches.
        remove_call_assembler_targets_for_compiled_ptr(self as *const Self as usize as u32);
        // Retract this loop's published label targets so a later bridge
        // cannot chain into a dropped loop's stale table slot. Guarded by
        // `func_handle`: a recompile that re-stamped the same descr onto its
        // replacement loop has already overwritten the entry, which must
        // survive the old loop's drop.
        let mut reg = LABEL_TARGETS.lock().unwrap();
        if let Some(map) = reg.as_mut() {
            for (id, func_handle) in self
                .label_descrs
                .iter()
                .copied()
                .map(|id| (id, self.func_handle.get()))
                .chain(self.bridge_owned_label_targets.get_mut().iter().copied())
            {
                if id != 0
                    && let Some(t) = map.get(&id)
                    && t.func_handle == func_handle
                {
                    map.remove(&id);
                    crate::BRIDGE_DIAG[22].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
}
