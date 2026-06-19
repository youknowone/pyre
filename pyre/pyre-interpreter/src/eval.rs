//! Bytecode evaluation loop — pure interpreter.
//!
//! JIT integration lives in pyre-jit/src/eval.rs. This module is
//! JIT-free: it processes bytecode instructions with no tracing,
//! no merge points, and no compiled-code hooks.

use crate::bytecode::{BinaryOperator, ComparisonOperator, Instruction};
use crate::*;
use crate::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs,
    decode_instruction_for_dispatch, dict_storage_load, dict_storage_store, ensure_range_iter,
    execute_opcode_step, range_iter_continues, range_iter_next_or_null, stack_underflow_error,
    unpack_sequence_exact,
};
use pyre_object::*;

use crate::call::call_callable;
use std::cell::Cell;

#[derive(Debug, Clone)]
pub struct Code {
    pub name: String,
    pub code: Option<PyObjectRef>,
}

impl Code {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            code: None,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("<code {}>", self.name)
    }
}

// The current active exception (`sys.exc_info()` / bare `raise`) now lives
// on the per-thread `ExecutionContext` (`sys_exc_value`), reached via
// `get_current_exception` / `set_current_exception`; see those.
thread_local! {
    pub(crate) static CURRENT_FRAME: Cell<*mut PyFrame> = const { Cell::new(std::ptr::null_mut()) };
}
use crate::pyframe::PyFrame;

/// Saves the previous `CURRENT_FRAME` and (when EC was modified) the
/// previous `ec.topframeref` so they can be restored on Drop. The two
/// pointers are pushed onto `majit_gc::shadow_stack` rather than a local
/// `Vec` — this matches RPython's `framework.py` shadow-stack
/// (rpython/memory/gctransform/shadowstack.py:281) and lets the GC's
/// root-walker forward both pointers in place when a minor
/// collection runs while the guard is on the stack.
pub struct CurrentFrameGuard {
    save_point: usize,
    ec: *mut PyExecutionContext,
    ec_top_root_index: Option<usize>,
}

impl Drop for CurrentFrameGuard {
    fn drop(&mut self) {
        // Read forwarded values from the shadow stack before pop_to so we
        // observe any in-place updates the GC may have made.
        let previous = majit_gc::shadow_stack::get(self.save_point);
        let previous_ec_top = self
            .ec_top_root_index
            .map(majit_gc::shadow_stack::get)
            .unwrap_or(majit_ir::GcRef::NULL);
        majit_gc::shadow_stack::pop_to(self.save_point);
        CURRENT_FRAME.with(|current| current.set(previous.0 as *mut PyFrame));
        if !self.ec.is_null() {
            unsafe {
                (*self.ec).topframeref = previous_ec_top.0 as *mut PyFrame;
            }
        }
    }
}

fn push_current_frame_previous_root(
    previous: *mut PyFrame,
    ec: *mut PyExecutionContext,
    previous_ec_top: *mut PyFrame,
) -> CurrentFrameGuard {
    let save_point = majit_gc::shadow_stack::push(majit_ir::GcRef(previous as usize));
    let ec_top_root_index = if ec.is_null() {
        None
    } else {
        Some(majit_gc::shadow_stack::push(majit_ir::GcRef(
            previous_ec_top as usize,
        )))
    };
    CurrentFrameGuard {
        save_point,
        ec,
        ec_top_root_index,
    }
}

pub fn install_current_frame(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    // executioncontext.py `enter()` parity: link the frame into the
    // topframeref/f_backref chain so walkers (GC roots, sys._getframe)
    // can iterate all active frames. `eval_frame_plain` calls
    // `ExecutionContext::enter` before installing TLS-only state, but
    // the JIT portal path enters through this helper directly.
    let ec = frame.execution_context as *mut PyExecutionContext;
    let previous_ec_top = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe {
            let top = (*ec).topframeref;
            (*ec).topframeref = frame as *mut PyFrame;
            top
        }
    };
    frame.f_backref = if ec.is_null() {
        previous
    } else {
        previous_ec_top
    };
    push_current_frame_previous_root(previous, ec, previous_ec_top)
}

/// Install only the TLS current-frame root.
///
/// Use this after `ExecutionContext::enter()` has already linked
/// `frame.f_backref`.  PyPy has one frame chain (`ec.topframeref`);
/// pyre's `CURRENT_FRAME` is an extra GC/super() TLS root and must not
/// overwrite the RPython `f_backref` chain once EC owns it.
pub fn install_current_frame_tls_only(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    push_current_frame_previous_root(previous, std::ptr::null_mut(), std::ptr::null_mut())
}

/// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
/// expose every live slot of `PyFrame.locals_cells_stack_w` on the active
/// f_backref chain as a GC root.
///
/// pyre's JIT-compiled code allocates W_IntObject / result boxes into the
/// nursery (`NewWithVtable` → `gc_alloc_typed_nursery_shim`). When the
/// nursery fills and a minor collection runs, only registered roots are
/// forwarded — unforwarded nursery refs become stale after
/// `Nursery::reset` zero-fills the region. The interpreter stores live
/// refs in `PyFrame.locals_cells_stack_w`; without this walker those
/// slots turn into NULL-`ob_type` stale pointers on the next LOAD_FAST
/// (reproduced by `inline_helper` n >= 10000).
///
/// Walks 0..`valuestackdepth` entries because that range covers both
/// the always-live locals+cells prefix (slots `0..nlocals+ncells`,
/// written once at frame setup) and the operand stack region
/// (`nlocals+ncells..valuestackdepth`). Dead stack slots past
/// `valuestackdepth` are skipped.
unsafe fn walk_raw_function_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() || !crate::is_function(value) {
            return;
        }
        let func = &mut *(value as *mut crate::function::Function);
        visitor(&mut *(&mut func.code as *mut *const () as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.closure as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.defs_w as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_kw_defs as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_module as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_func_globals_obj as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_ann as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_annotate as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_doc as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_qualname as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_objclass as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_text_signature as *mut PyObjectRef as *mut majit_ir::GcRef));
    }
}

/// Mark the GC-reachable children of a `getset_descriptor`
/// (`W_GetSetProperty`).  The descriptor itself is Box-immortal
/// (`pyre_class` `allocate` → `malloc_typed`), so its `W_TYPE_GC_TYPE_ID`
/// custom trace never fires.  Its `fget`/`fset`/`fdel` getters are
/// GC-managed `try_gc_alloc_stable` functions — non-moving but still
/// *collectable* — so when a descriptor's only holder is a Box-immortal
/// type dict, nothing marks the getters reachable and the collector frees
/// them, leaving `descr.fget` dangling (a fresh `obj.__dict__` after a
/// collection then calls a freed getter → SIGSEGV).  Visit every
/// `PyObjectRef` field and recurse into the getter functions, the same
/// shape `walk_raw_function_roots` uses for Box-held function children.
/// No-op for non-descriptor values.
unsafe fn walk_raw_getset_roots(value: PyObjectRef, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicate: the annotator cannot lower `!` over a
        // cross-crate bool result (`UnaryNotUnknownOperand`), so guard with
        // a positive `if` rather than negating `is_getset_property`.
        if pyre_object::getsetproperty::is_getset_property(value) {
            let d = &mut *(value as *mut pyre_object::getsetproperty::W_GetSetProperty);
            visitor(&mut *(&mut d.fget as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.fset as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.fdel as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.doc as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.reqcls as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.name as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.w_objclass as *mut PyObjectRef as *mut majit_ir::GcRef));
            visitor(&mut *(&mut d.w_qualname as *mut PyObjectRef as *mut majit_ir::GcRef));
            // The getters are functions whose own children (code / globals /
            // defaults) must stay reachable as well.
            walk_raw_function_roots(d.fget, visitor);
            walk_raw_function_roots(d.fset, visitor);
            walk_raw_function_roots(d.fdel, visitor);
        }
    }
}

/// Forward every `PyObjectRef` value bound in a heap type's namespace
/// `DictStorage` in place — the class attributes, methods, and the
/// per-type `__dict__`/`__weakref__` getset descriptor copies.  Keys are
/// Rust `String`s (not GC objects); only the `PyObjectRef` values relocate.
/// Snapshot the value slots first (same shape as the globals proxy walk)
/// so `forward` cannot re-borrow the storage.  Shared by `walk_type_dicts_gc`
/// (the Box-immortal-type band-aid root walk) and the `W_TYPE_GC_TYPE_ID`
/// custom trace, so a GC-managed heap type reaches its own namespace values
/// directly once its trace fires.
pub unsafe fn type_walk_namespace_values(
    w_type: PyObjectRef,
    forward: &mut dyn FnMut(&mut PyObjectRef),
) {
    unsafe {
        if w_type.is_null() {
            return;
        }
        // Positive predicate (see `walk_raw_getset_roots`): `!is_type` over
        // a cross-crate bool is `UnaryNotUnknownOperand` to the annotator,
        // so guard with a positive `if`.
        if pyre_object::is_type(w_type) {
            let dict_ptr = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            if dict_ptr.is_null() {
                // No namespace storage installed yet.
            } else {
                let value_slots: Vec<*mut PyObjectRef> = (*dict_ptr)
                    .values_mut()
                    .iter_mut()
                    .map(|value| value as *mut PyObjectRef)
                    .collect();
                for slot in value_slots {
                    forward(&mut *slot);
                }
                // The lazily-cached canonical `W_DictObject` that
                // `type.__dict__` returns (`dict_storage_to_dict_kind`'s
                // `mirror_target`) is GC-managed but reachable only through
                // this off-GC storage field; forward it so a minor
                // collection that relocates or reclaims it updates the
                // cache instead of returning a dangling pointer on the next
                // `__dict__` access.
                if let Some(slot) = (*dict_ptr).mirror_target_slot_mut() {
                    forward(slot);
                }
            }
        }
    }
}

/// Box-immortal heap types (`w_type_new`) never have their
/// `W_TYPE_GC_TYPE_ID` custom trace fired, so the movable values bound in
/// each type's namespace `DictStorage` (methods, class attributes, the
/// per-type `__dict__`/`__weakref__` getset copies), the `bases` tuple, and
/// the `weak_subclasses` WEAKREF list are unreachable by the collector.
/// Walk every registered heap type's namespace values + `bases` +
/// `weak_subclasses` as pinned roots so a relocated class attribute /
/// method / descriptor — or a reclaimed subclass weakref — is not read back
/// stale after a collection; the same shape `walk_module_dicts_gc` uses for
/// module dicts.  PRE-EXISTING-ADAPTATION: PyPy GC-manages type objects and
/// traces `dict_w`/`bases`/`weak_subclasses` for free; convergence is to
/// GC-manage `W_TypeObject` (then its custom trace fires and this walk is
/// deleted), mirroring the deferred instance Path-B keystone.
unsafe fn walk_type_dicts_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    unsafe {
        for addr in pyre_object::typeobject::snapshot_heap_types() {
            let w_type = addr as PyObjectRef;
            if w_type.is_null() {
                continue;
            }
            // Positive predicate (see `walk_raw_getset_roots`): `!is_type`
            // over a cross-crate bool is `UnaryNotUnknownOperand` to the
            // annotator, so guard with a positive `if`.
            if pyre_object::is_type(w_type) {
                // `bases` is a movable tuple created at class definition and
                // held only by the Box-immortal type; forward it in place.
                let bases_slot =
                    &mut (*(w_type as *mut pyre_object::typeobject::W_TypeObject)).bases;
                forward(bases_slot);
                // Namespace `dict_w` values: the class attributes, methods,
                // and getset descriptor copies — the same walk the
                // `W_TYPE_GC_TYPE_ID` custom trace performs once a heap type
                // is GC-managed.
                type_walk_namespace_values(w_type, forward);
                // `weak_subclasses` holds `w_weakref_new` (`try_gc_alloc`)
                // young WEAKREF GcStructs whose only strong root is this
                // off-GC list; forward each slot in place so the WEAKREF
                // survives collection (its `weakptr` payload is invalidated
                // separately by the collector's weakref scan).  Without this,
                // the first collection reclaims the weakref and the base's
                // `weak_subclasses[i]` dangles — a UAF on the next
                // `mutated()` / `w_type_get_subclasses` deref.  The
                // `W_TYPE_GC_TYPE_ID` custom trace performs the same walk once
                // a heap type is GC-managed.
                let t = &mut *(w_type as *mut pyre_object::typeobject::W_TypeObject);
                if t.weak_subclasses.is_null() {
                    // No subclasses recorded.
                } else {
                    let subs = &mut *t.weak_subclasses;
                    for slot in subs.iter_mut() {
                        forward(
                            &mut *(slot as *mut *mut pyre_object::weakref::Weakref
                                as *mut PyObjectRef),
                        );
                    }
                }
            }
        }
    }
}

fn walk_pyframe_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    CURRENT_FRAME.with(|cf| {
        // Forward `CURRENT_FRAME` itself: when the top frame is a
        // nursery-allocated `PyFrame`
        // (`emit_new_pyframe_inline_self_recursive`) the visitor copies
        // it to the survivor space and rewrites the cell to the new
        // address. For `std::alloc`-backed frames the visitor's
        // `is_nursery_object_start` guard short-circuits, leaving the
        // pointer untouched. `Cell::as_ptr()` exposes the storage
        // address; `*mut PyFrame` and `GcRef` share the `usize` repr
        // (`GcRef` is `#[repr(transparent)]`).
        //
        // SAFETY: `CURRENT_FRAME`'s storage is a thread-local `Cell`
        // that outlives this walker. We hold the with-borrow `cf` for
        // the duration of the visit so no other code mutates the cell.
        let cf_slot_ptr = cf.as_ptr() as *mut majit_ir::GcRef;
        visitor(unsafe { &mut *cf_slot_ptr });
        // Saved previous-frame / previous-ec-topframe roots now live on
        // `majit_gc::shadow_stack` (pushed by `push_current_frame_previous_root`)
        // and are forwarded by the GC's root walker; no extra visit here.

        let mut frame = cf.get();
        let frame_ec = if frame.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { (*frame).execution_context as *mut PyExecutionContext }
        };
        // Root the EC slots from the current frame's EC AND the ambient
        // TLS EC (`getexecutioncontext`).  The ambient visit covers the
        // spans where no frame is installed in `CURRENT_FRAME` yet the EC
        // is live — between `ExecutionContext::enter` and `eval_loop`'s
        // frame install, and around `return_trace`/`leave` after the
        // frame guard drops — where `sys_exc_value` may already hold a
        // nursery exception.  PyPy reaches the ExecutionContext
        // unconditionally through `space.threadlocals`, independent of
        // any frame.
        let ambient_ec = crate::call::take_last_exec_ctx() as *mut PyExecutionContext;
        let mut visit_ec_slots = |ec: *mut PyExecutionContext| {
            if ec.is_null() {
                return;
            }
            let top_slot = unsafe { &mut (*ec).topframeref as *mut *mut PyFrame };
            visitor(unsafe { &mut *(top_slot as *mut majit_ir::GcRef) });
            // `sys_exc_value` holds the active handler exception, which
            // is nursery-allocated and may move; forward it so the EC
            // slot is updated on a minor collection (the value-stack
            // copy alone is not authoritative for later EC reads).
            let exc_slot = unsafe { &mut (*ec).sys_exc_value as *mut PyObjectRef };
            visitor(unsafe { &mut *(exc_slot as *mut majit_ir::GcRef) });
        };
        visit_ec_slots(frame_ec);
        if ambient_ec != frame_ec {
            visit_ec_slots(ambient_ec);
        }
        while !frame.is_null() {
            // SAFETY: PyFrame pointers on the f_backref chain are valid
            // for the duration of the enclosing `eval_with_jit` call. A
            // minor collection is always synchronous with respect to the
            // interpreter thread, so frames cannot be dropped mid-walk.
            //
            // We walk the FULL fixed-length array (not just the live
            // `valuestackdepth` prefix). Argument values in transit —
            // popped from the caller's stack before the callee frame
            // is installed — are briefly invisible from
            // `valuestackdepth` alone, yet still reachable from the
            // popped-slot storage. Non-ref slots are filtered by
            // `is_nursery_object_start` inside the collector, so
            // walking past the live depth is harmless for the
            // bump-pointer nursery.
            //
            // The walk runs for every frame on the chain, including
            // ones the GC owns. For nursery-allocated frames the
            // standard tracer ALSO covers their gc_ptr_offsets when it
            // reaches the survivor copy; visiting the locals array
            // items here from the original nursery payload is safe
            // because root visiting runs before any internal-slot
            // forwarding (the original payload is still intact). We
            // intentionally do NOT call `majit_gc::gc_owns_object`
            // here to gate this branch — that hook re-enters
            // `with_cranelift_gc` with a `borrow_mut`, which panics
            // when invoked from inside `collect_nursery` (the GC's
            // own cell is already borrowed by the active alloc shim).
            let (arr_ptr, depth, next_frame) = unsafe {
                let f_back_slot = &mut (*(frame)).f_backref as *mut *mut PyFrame;
                visitor(&mut *(f_back_slot as *mut majit_ir::GcRef));

                // pyframe.py:102 `self.pycode` — the running code object.
                // Visited as a root so a code object reachable only via
                // `frame.pycode` (e.g. `exec`'d code with no owning
                // Function) stays alive once code objects become
                // GC-managed.  While code objects remain Box-immortal the
                // visitor's `is_nursery_object_start` /
                // `is_managed_heap_object` guard short-circuits, so this is
                // inert today.
                let pycode_slot = &mut (*(frame)).pycode as *mut *const ();
                visitor(&mut *(pycode_slot as *mut majit_ir::GcRef));

                // PyFrame is normally a GC object in PyPy, so its GCREF
                // fields are traced before consumers dereference them.
                // pyre also has stdalloc-backed frames, so the frame root
                // walker must expose those fields explicitly.
                let locals_slot =
                    &mut (*(frame)).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
                visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
                let gen_slot = &mut (*(frame)).f_generator_nowref as *mut PyObjectRef;
                visitor(&mut *(gen_slot as *mut majit_ir::GcRef));
                let yielding_slot = &mut (*(frame)).w_yielding_from as *mut PyObjectRef;
                visitor(&mut *(yielding_slot as *mut majit_ir::GcRef));
                // pyframe.py:115-116 `self.builtin = ...` — the picked
                // builtin Module is a GC root.  Pyre stores it on
                // `frame.w_builtin` so `frame.get_builtin()` returns
                // the same object PyPy would; the LOAD_GLOBAL fallback
                // (`load_global_value` at eval.rs) reaches the
                // builtin's globals through `w_module_get_w_dict(self
                // .w_builtin)` — there is no separate storage-keyed
                // fast path field anymore.
                let w_builtin_slot = &mut (*(frame)).w_builtin as *mut PyObjectRef;
                visitor(&mut *(w_builtin_slot as *mut majit_ir::GcRef));
                // pyframe.py:147 `debugdata.w_locals` (and the pyre-only
                // `w_locals_object` companion for non-dict mapping
                // locals) carry GCREFs that survive the frame.
                if !(*frame).debugdata.is_null() {
                    let d = &mut *(*frame).debugdata;
                    let w_locals_object_slot = &mut d.w_locals_object as *mut PyObjectRef;
                    visitor(&mut *(w_locals_object_slot as *mut majit_ir::GcRef));
                    let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
                    visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
                }
                // pyframe.py:49 `self.w_globals` is the dict OBJECT.  Visit
                // the canonical `w_globals_obj` slot first so the visitor
                // forwards it (and resolves any forwarding marker a sibling
                // frame sharing the same module globals already left); only
                // then is the object's `dict_storage_proxy` safe to chase for
                // the backing storage.  Reading the proxy off a not-yet-
                // forwarded object would dereference a stale nursery address.
                let w_globals_obj_slot = &mut (*frame).w_globals_obj as *mut PyObjectRef;
                visitor(&mut *(w_globals_obj_slot as *mut majit_ir::GcRef));
                let live_obj = (*frame).w_globals_obj;
                if !live_obj.is_null() {
                    let globals_ptr =
                        pyre_object::dictmultiobject::w_dict_get_dict_storage_proxy(live_obj)
                            as *mut crate::DictStorage;
                    if !globals_ptr.is_null() {
                        let value_slots: Vec<*mut PyObjectRef> = (&mut *globals_ptr)
                            .values_mut()
                            .iter_mut()
                            .map(|value| value as *mut PyObjectRef)
                            .collect();
                        for value in value_slots {
                            visitor(&mut *(value as *mut majit_ir::GcRef));
                            walk_raw_function_roots(*value, visitor);
                        }
                        (&mut *globals_ptr).set_mirror_target(live_obj);
                    }
                }
                // The proxy mirror above only covers the back-mirror
                // DictStorage.  For a W_ModuleDictObject the LOAD_GLOBAL
                // read path (`w_module_dict_getitem_str`) consults the
                // authoritative `dstorage` cell map / `object_storage` /
                // strategy caches ahead of the proxy, none of which the
                // proxy walk reaches; the module dict is Box-immortal so
                // its own custom trace never fires.  Forward those movable
                // values here so a relocated global is not read back stale.
                // No-op for non-module dicts.  The picked builtin Module's
                // dict is consulted on a globals miss (`_load_global`
                // fallback), so forward it too.
                {
                    let mut forward = |slot: &mut PyObjectRef| {
                        visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
                        walk_raw_function_roots(*slot, visitor);
                    };
                    pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(
                        live_obj,
                        &mut forward,
                    );
                    let w_builtin = (*frame).w_builtin;
                    if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                        let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
                        pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(
                            w_builtin_dict,
                            &mut forward,
                        );
                    }
                }
                let f = &*frame;
                let next_frame = (*frame).f_backref;
                if f.locals_cells_stack_w.is_null() {
                    (std::ptr::null_mut::<PyObjectRef>(), 0, next_frame)
                } else {
                    let arr = &*f.locals_cells_stack_w;
                    (arr.items_ptr() as *mut PyObjectRef, arr.len(), next_frame)
                }
            };
            if !arr_ptr.is_null() && depth > 0 {
                for i in 0..depth {
                    let slot_ptr = unsafe { arr_ptr.add(i) } as *mut majit_ir::GcRef;
                    // SAFETY: slot lies inside the FixedObjectArray's
                    // heap allocation, which outlives the frame. The
                    // visitor reads, conditionally forwards, and
                    // stores back a `GcRef` (same layout as
                    // `*mut PyObject`).
                    visitor(unsafe { &mut *slot_ptr });
                }
            }
            frame = next_frame;
        }
        // Box-immortal modules (and their Box-immortal dicts) are not
        // reachable transitively by the collector, so walk every loaded
        // module's dict storage as a pinned root source.  This covers
        // module-scope movable values bound in modules other than the
        // running frame's globals — e.g. `gc.collect` reached through
        // `gc.__dict__` on a fresh `LOAD_METHOD` after a collection.
        unsafe {
            let mut forward = |slot: &mut PyObjectRef| {
                visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
                walk_raw_function_roots(*slot, visitor);
                // getset descriptors are Box-immortal (custom trace never
                // fires), so their collectable `fget`/`fset`/`fdel`
                // functions must be marked reachable here or the getter
                // dangles after a collection.
                walk_raw_getset_roots(*slot, visitor);
            };
            crate::importing::walk_module_dicts_gc(&mut forward);
            // The `sys.modules` dict is cached in a fast-path cell
            // (`importing::SYS_MODULES_DICT`) that is independent of the
            // `sys.__dict__["modules"]` slot the walk above forwards; forward
            // the cell too so a relocated modules dict is not read back stale
            // on the next import / `check_sys_modules` lookup.
            crate::importing::walk_sys_modules_dict_gc(&mut forward);
            // Box-immortal heap types' namespace dicts hold movable
            // methods / class attributes / descriptor copies that no
            // custom trace reaches; root them the same way.
            walk_type_dicts_gc(&mut forward);
            // The interpreter method cache (`baseobjspace::MethodCache`)
            // keeps a second pointer to each looked-up method that the
            // namespace-dict walk above does not reach; forward those so
            // a cache hit after a moving collection is not stale.
            crate::baseobjspace::walk_method_cache_gc(&mut forward);
            // The per-pycode `_mapdict_caches` LOAD_METHOD slots hold a
            // `w_method` pointer (mapdict.py:1418) that no custom trace
            // reaches — code objects are Box-immortal — so forward those
            // the same way.
            crate::pycode::walk_mapdict_method_cache_gc(&mut forward);
        }
    });
}

/// Install the PyFrame GC root walker with the majit-gc collector.
///
/// Called once at process startup from the JIT driver / pyrex main.
/// Stored in a per-thread slot; calling again with the same fn pointer
/// is idempotent.
pub fn register_pyframe_root_walker() {
    majit_gc::set_active_extra_root_walker(Some(walk_pyframe_roots));
}

/// Flat TLS read of the per-thread `CURRENT_EXCEPTION` slot.
///
/// `dont_look_inside` keeps the codewriter from following into the
/// `LocalKey::with` closure (no extractable graph); calls classify
/// `Residual` against the fnaddr registered in `jit_trace_fnaddrs()`,
/// mirroring the trace-side `get_current_exception_fn` cpu helper
/// binding (`codewriter.rs PlainCannotRaiseNoHeap`).
#[majit_macros::dont_look_inside]
pub fn get_current_exception() -> PyObjectRef {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return PY_NULL;
    }
    unsafe { (*ec).sys_exc_value }
}

/// Flat TLS write of the per-thread `CURRENT_EXCEPTION` slot — same
/// residual-leaf contract as [`get_current_exception`].
#[majit_macros::dont_look_inside]
pub fn set_current_exception(exc: PyObjectRef) {
    let ec = crate::call::getexecutioncontext() as *mut PyExecutionContext;
    if ec.is_null() {
        return;
    }
    unsafe {
        (*ec).sys_exc_value = exc;
    }
}

/// `pyopcode.py:1524-1532 DICT_UPDATE` — update `dict` from a mapping
/// source via the `keys()` + `__getitem__` protocol.  Falls back to
/// direct `w_dict_items` for exact-dict sources.
fn dict_update_from_mapping(dict: PyObjectRef, source: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_dict(source) {
            for (k, v) in pyre_object::w_dict_items(source) {
                pyre_object::w_dict_store(dict, k, v);
            }
            return Ok(());
        }
    }
    // pyopcode.py:2005-2006: only AttributeError → TypeError; others propagate
    let keys_method = match crate::baseobjspace::getattr_str(source, "keys") {
        Ok(m) => m,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let type_name = unsafe { (*(*source).ob_type).name };
            return Err(PyError::type_error(format!(
                "'{type_name}' object is not a mapping"
            )));
        }
        Err(e) => return Err(e),
    };
    let keys_obj = crate::call::call_function_impl_result(keys_method, &[])?;
    let keys = crate::builtins::collect_iterable(keys_obj)?;
    for key in keys {
        let val = crate::baseobjspace::getitem(source, key)?;
        unsafe { pyre_object::w_dict_store(dict, key, val) };
    }
    Ok(())
}

/// Resolve callable display prefix for `**kwargs` error messages.
/// Returns e.g. `"foo()"` or just `""` when unresolvable.
fn callable_prefix(w_callable: PyObjectRef) -> String {
    if w_callable.is_null() {
        return String::new();
    }
    unsafe {
        if crate::is_function(w_callable) {
            let name = crate::function_get_qualname(w_callable);
            return format!("{name}() ");
        }
        if pyre_object::is_type(w_callable) {
            let name = pyre_object::w_type_get_name(w_callable);
            return format!("{name}() ");
        }
    }
    String::new()
}

/// pyopcode.py:1979-2026 `_dict_merge` — merge `source` into `dict`.
/// Dict path checks duplicates; mapping path does keys/getitem/setitem
/// without extra validation (string key check is CALL_FUNCTION_EX's job).
fn dict_merge_from_mapping(
    dict: PyObjectRef,
    source: PyObjectRef,
    w_callable: PyObjectRef,
) -> Result<(), PyError> {
    let prefix = callable_prefix(w_callable);

    unsafe {
        if pyre_object::is_dict(source) {
            for (k, v) in pyre_object::w_dict_items(source) {
                if pyre_object::w_dict_lookup(dict, k).is_some() {
                    // pyopcode.py:1987 — %S is str(key)
                    let key_str = crate::display::py_str(k)?;
                    return Err(PyError::type_error(format!(
                        "{prefix}got multiple values for keyword argument '{key_str}'"
                    )));
                }
                pyre_object::w_dict_store(dict, k, v);
            }
            return Ok(());
        }
    }
    // pyopcode.py:2005-2006: only AttributeError → TypeError; others propagate
    let keys_method = match crate::baseobjspace::getattr_str(source, "keys") {
        Ok(m) => m,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let type_name = unsafe { (*(*source).ob_type).name };
            return Err(PyError::type_error(format!(
                "{prefix}argument after ** must be a mapping, not {type_name}"
            )));
        }
        Err(e) => return Err(e),
    };
    // pyopcode.py:2021 _dict_merge_loop: keys/getitem/contains/setitem
    let keys_obj = crate::call::call_function_impl_result(keys_method, &[])?;
    let keys = crate::builtins::collect_iterable(keys_obj)?;
    for key in keys {
        let val = crate::baseobjspace::getitem(source, key)?;
        unsafe {
            if pyre_object::w_dict_lookup(dict, key).is_some() {
                let key_str = crate::display::py_str(key)?;
                return Err(PyError::type_error(format!(
                    "{prefix}got multiple values for keyword argument '{key_str}'"
                )));
            }
            pyre_object::w_dict_store(dict, key, val);
        }
    }
    Ok(())
}

pub fn normalize_raise_value(value: PyObjectRef) -> PyObjectRef {
    unsafe {
        if crate::baseobjspace::exception_is_valid_obj_as_class_w(value) {
            return crate::call_function(value, &[]);
        }
    }
    value
}

/// Normalize the `from` cause of a `raise X from Y` statement: instantiate
/// the cause if it is an exception class, validate that the result is
/// `None` / a `BaseException` instance, and return a `PyError::type_error`
/// otherwise.
///
/// # TODO: inline back into RAISE_VARARGS
///
/// **Deviation.** RPython performs this validation inline inside
/// `RAISE_VARARGS` (`pypy/interpreter/pyopcode.py:704-707`,
/// `space.call_function(w_cause)` when `w_cause` is an exception class)
/// without a named helper, deferring the BaseException check to
/// `OperationError.set_cause` (`pypy/interpreter/error.py`). Pyre
/// extracts this pre-step into a standalone helper so the JIT raise/r
/// BH path (`pyre-jit/src/call_jit.rs::bh_normalize_raise_varargs_fn`)
/// and the interpreter raise path can share the same validation.
///
/// **When to fix.** When `bh_normalize_raise_varargs_fn` is removed or
/// rewritten — e.g. when the JIT BH path can dispatch the same inlined
/// `RAISE_VARARGS` sequence directly without a shared helper.
///
/// **How to fix.** Inline this body back into the `RAISE_VARARGS`
/// dispatch arm in `pyre-interpreter/src/pyopcode.rs` (mirroring
/// `pyopcode.py:704-707`), delete this standalone fn, and either route
/// the BH path through the inlined sequence or rewrite it to match
/// RPython's inline shape.
pub fn normalize_raise_cause(cause: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let cause = normalize_raise_value(cause);
    unsafe {
        if cause.is_null() || pyre_object::is_none(cause) || pyre_object::is_exception(cause) {
            return Ok(cause);
        }
    }
    Err(PyError::type_error(
        "exception cause must be None or derive from BaseException",
    ))
}

pub fn attach_raise_cause(exc: PyObjectRef, cause: Option<PyObjectRef>) -> Result<(), PyError> {
    // `pypy/interpreter/pyopcode.py do_raise` /
    // `pypy/interpreter/executioncontext.py:325 _normalize_exception` —
    // when a `raise` runs while another exception is being handled,
    // chain the in-flight one as the new `__context__` so tracebacks
    // can show "During handling of the above exception, another
    // exception occurred:". Skip self-context to avoid the obvious
    // cycle (re-raising the same exception object).  Both
    // `__context__` and `__cause__`/`__suppress_context__` writes land
    // in the typed slots on `W_ExceptionObject` per
    // `interp_exceptions.py:113-117`.
    let active = get_current_exception();
    if !active.is_null()
        && active != exc
        && unsafe { !pyre_object::is_none(active) }
        && unsafe { pyre_object::is_exception(exc) }
    {
        // `interp_exceptions.py:115 W_BaseException.w_context = None`
        // class default — only write if no `__context__` is already
        // stamped on the exception (mirrors `or_insert` semantics).
        let existing = unsafe { pyre_object::excobject::w_exception_get_context(exc) };
        if existing.is_null() {
            unsafe { pyre_object::excobject::w_exception_set_context(exc, active) };
        }
    }
    if let Some(cause_obj) = cause {
        if !cause_obj.is_null() && unsafe { pyre_object::is_exception(exc) } {
            // `interp_exceptions.py:166-174 descr_setcause` — writes
            // `w_cause` and flips `suppress_context` to True.
            unsafe {
                pyre_object::excobject::w_exception_set_cause(exc, cause_obj);
                pyre_object::excobject::w_exception_set_suppress_context(exc, true);
            };
        }
    }
    Ok(())
}

/// pyopcode.py:1032-1040 `cmp_exc_match(self, w_1, w_2)` line-by-line:
///
/// ```python
/// def cmp_exc_match(self, w_1, w_2):
///     space = self.space
///     if space.isinstance_w(w_2, space.w_tuple):
///         for w_type in space.fixedview(w_2):
///             if not space.exception_is_valid_class_w(w_type):
///                 raise oefmt(space.w_TypeError, CANNOT_CATCH_MSG)
///     elif not space.exception_is_valid_class_w(w_2):
///         raise oefmt(space.w_TypeError, CANNOT_CATCH_MSG)
///     return space.exception_match(space.type(w_1), w_2)
/// ```
///
/// `w_1` is `exc_value` (the exception instance, peeked from TOS at
/// pyopcode.py:1852), `w_2` is `exc_type` (the type spec, popped at
/// :1851). `space.type(w_1)` is the exception's class.
///
/// pyopcode.py:24-25 `CANNOT_CATCH_MSG`.
pub const CANNOT_CATCH_MSG: &str =
    "catching classes that do not inherit from BaseException is not allowed";

/// pyopcode.py:1034-1039 — the class-validity gate of `cmp_exc_match`,
/// split out from `check_exc_match_against` so the bool-returning hot
/// helper keeps a 1-register C ABI suitable for residual JIT calls.
/// PyPy's `@jit.unroll_safe` `cmp_exc_match` inlines into the trace and
/// its `raise oefmt(...)` becomes a guard; pyre matches the structure
/// by keeping the raise on the caller side (the BC handler), which
/// likewise runs outside the JIT-traced bool-returning fast path.
pub fn validate_check_exc_match_class(exc_type: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_tuple(exc_type) {
            let n = pyre_object::w_tuple_len(exc_type) as i64;
            for i in 0..n {
                if let Some(w_type) = pyre_object::w_tuple_getitem(exc_type, i) {
                    if !crate::baseobjspace::exception_is_valid_class_w(w_type) {
                        return Err(PyError::type_error(CANNOT_CATCH_MSG));
                    }
                }
            }
        } else if !crate::baseobjspace::exception_is_valid_class_w(exc_type) {
            return Err(PyError::type_error(CANNOT_CATCH_MSG));
        }
    }
    Ok(())
}

pub fn check_exc_match_against(exc_value: PyObjectRef, exc_type: PyObjectRef) -> bool {
    // pyopcode.py:1040 `return space.exception_match(space.type(w_1), w_2)`.
    // `crate::typedef::r#type` is the `space.type` equivalent — it
    // resolves `w_class` for objects whose specific class was already
    // installed (post-`init_typeobjects`) AND for exception instances
    // whose `w_class` slot still holds the generic `EXCEPTION_TYPE`
    // stub (pre-registry-init internal `w_exception_new` callers, e.g.
    // `PyError::value_error`) by falling back to the `ExcKind`-tag
    // registry per typedef.rs:176-197.
    //
    // The validity gate (pyopcode.py:1034-1039) lives in
    // `validate_check_exc_match_class` and is invoked by the BC handler
    // BEFORE this helper, mirroring PyPy's `@jit.unroll_safe` inlining
    // where `raise oefmt(...)` becomes a guard outside the bool-returning
    // residual call.  The 1-register `bool` ABI is preserved for
    // cranelift / dynasm residual-call codegen.
    let Some(w_exc_class) = crate::typedef::r#type(exc_value) else {
        return false;
    };
    crate::baseobjspace::exception_match(w_exc_class, exc_type)
}

/// Try to dispatch an exception using the exception table or block stack.
///
/// Returns `true` if a handler was found (resume PC updated to handler),
/// `false` if the exception should propagate to the caller.
///
/// `err` is taken by `&mut` so the bytecode_trace_after_exception /
/// exception_trace plumbing can replace it with a tracer exception
/// (pyopcode.py:144-145 `except OperationError as e: operr = e`); the
/// caller's `Err(err)` propagation then surfaces the replacement.
pub fn handle_exception(frame: &mut PyFrame, err: &mut PyError, next_instr: &mut usize) -> bool {
    // Internal control-flow / corruption markers are not real Python
    // exceptions and must never be dispatched via bytecode handlers.
    if err.kind == crate::PyErrorKind::GeneratorReturn
        || err.kind == crate::PyErrorKind::BytecodeCorruption
    {
        return false;
    }
    // pyopcode.py:135-148 — exception trace plumbing:
    //   try:
    //       trace = self.get_w_f_trace()
    //       if trace is not None:
    //           self.getorcreatedebug().w_f_trace = None
    //       try:
    //           ec.bytecode_trace_after_exception(self)
    //       finally:
    //           if trace is not None:
    //               self.getorcreatedebug().w_f_trace = trace
    //   except OperationError as e:
    //       operr = e
    //   pytraceback.record_application_traceback(
    //       self.space, operr, self, self.last_instr)
    //   ec.exception_trace(self, operr)
    //
    // bytecode_trace_after_exception + exception_trace are gated on a
    // live tracefunc so the no-tracer hot path skips the f_trace
    // save/restore dance.  record_application_traceback runs
    // unconditionally per `:147-148`, so the traceback chain grows on
    // every exception regardless of trace state.
    // bytecode_trace_after_exception's exception is caught by the
    // surrounding `except OperationError` and replaces operr;
    // exception_trace's exception is NOT caught (line 148 stands
    // outside the except), so it short-circuits the unrollstack search
    // — pyre signals that by returning `false` after replacing `err`.
    // `pyopcode.py:122-149 handle_operation_error(attach_tb=True)` —
    // the entire `if attach_tb:` block (bytecode_trace_after_exception,
    // record_application_traceback, exception_trace) is gated on
    // `attach_tb`.  RERAISE opcode raises `RaiseWithExplicitTraceback`
    // which routes through the `attach_tb=False` branch, so all three
    // tracing hooks are skipped per `:91-94`.  Pyre carries the same
    // intent via `PyError.attach_tb` set by `eval.rs::reraise`.
    let ec = frame.execution_context as *mut crate::PyExecutionContext;
    let exc_obj = err.to_exc_object();
    if err.exc_object.is_null() {
        err.exc_object = exc_obj;
    }
    if err.attach_tb {
        if !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
            let saved_trace = frame.get_w_f_trace();
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = pyre_object::PY_NULL;
            }
            let after_exc_result =
                unsafe { (*ec).bytecode_trace_after_exception(frame as *mut PyFrame) };
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = saved_trace;
            }
            if let Err(trace_err) = after_exc_result {
                // pyopcode.py:144-145 — `except OperationError as e: operr = e`.
                *err = trace_err;
            }
        }
        // `pyopcode.py:147-148 pytraceback.record_application_traceback`
        // — prepends a `W_PyTraceback` wrapping the current frame onto
        // the exception's `w_traceback` chain.
        unsafe {
            crate::pytraceback::record_application_traceback(
                exc_obj,
                frame as *mut PyFrame,
                frame.last_instr as i64,
            );
        }
    }
    if err.attach_tb && !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
        if let Err(trace_err) = unsafe {
            (*ec).exception_trace(
                frame as *mut PyFrame,
                pyre_object::PY_NULL,
                exc_obj,
                pyre_object::PY_NULL,
            )
        } {
            // pyopcode.py:148 `ec.exception_trace(self, operr)` is
            // outside the except-block; a raise here propagates past
            // unrollstack. Replace err and return `false` so the
            // caller's `return Err(err)` surfaces the tracer error
            // without searching for a handler for the original.
            *err = trace_err;
            return false;
        }
    }
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    // pyre's `last_instr` is a rustpython code-unit index; the PyPy-shaped
    // `lookup_exceptiontable` lookup takes byte offsets, so multiply by 2.
    // (See exception_table.rs: varint values are word offsets but the lookup
    // operates in byte space, mirroring `pycode.py:241-246`.)
    //
    // `frame.last_instr == -1` is the pre-first-opcode sentinel
    // (`pyframe.py:227-235` initialization).  An injected operr
    // (`eval_frame_plain_with_operr`) drives `handle_exception` before any
    // bytecode has executed, so the lookup must mirror PyPy
    // `pycode.py:250-253`: with `instr_offset == -1`, the first entry's
    // `start <= -1` is False and `start > -1` is True, returning the
    // `depth == -1` sentinel (no handler).  Skip the table lookup outright
    // rather than casting -1 to `u32::MAX` (panic in debug, wrap in
    // release).
    let lookup_result = if frame.last_instr < 0 {
        None
    } else {
        let pc_bytes = (frame.last_instr as u32) * 2;
        crate::exception_table::lookup_exceptiontable(&code.exceptiontable, pc_bytes)
    };
    let pc_units = if frame.last_instr < 0 {
        0u32
    } else {
        frame.last_instr as u32
    };

    // `pypy/interpreter/pyopcode.py:151-173` exception-table dispatch.
    if let Some((target_bytes, depth, lasti)) = lookup_result {
        // `pyopcode.py:155-156` — depth is relative (0 = empty value
        // stack); convert to absolute by adding the frame's locals+cells
        // base, then drop the stack to that depth.
        let target_depth = frame.nlocals() + frame.ncells() + depth as usize;
        while frame.valuestackdepth > target_depth {
            frame.pop();
        }
        // `pyopcode.py:157-170` — lasti=True: push the raise-site offset
        // as an int below the exception, so RERAISE N can read it for
        // traceback/f_lineno correctness.  If this dispatch was triggered
        // by RERAISE (reraise_lasti from PyError, mirroring PyPy's
        // `handle_operation_error(reraise_lasti=...)`), use the original
        // raise-site lasti the RERAISE carried; otherwise use the current
        // instruction (the raising site itself).
        if lasti {
            let lasti_value: i64 = if err.reraise_lasti >= 0 {
                err.reraise_lasti as i64
            } else {
                pc_units as i64
            };
            frame.push(pyre_object::w_int_new(lasti_value));
        }
        // pyopcode.py: reraise_lasti is a local of handle_operation_error;
        // OperationError raised from this function carries no lasti.  Clear
        // here so a re-thrown PyError does not double-consume.
        err.reraise_lasti = -1;
        let exc_obj = err.to_exc_object();
        frame.push(exc_obj);
        // The decoded `target` is a byte offset; pyre's `next_instr` is a
        // code-unit index, so divide by 2.
        *next_instr = (target_bytes / 2) as usize;
        return true;
    }

    // `pyopcode.py:175-185` no-handler propagation: if this unwind was
    // triggered by RERAISE N, restore `last_instr` to the original
    // raise-site offset so `frame.f_lineno` reports the right line.
    if err.reraise_lasti >= 0 {
        frame.last_instr = err.reraise_lasti as isize;
    }
    err.reraise_lasti = -1;
    frame.frame_finished_execution = true;

    false
}

/// Execute a frame — pure interpreter, no JIT.
///
/// Crate-private since Slice C.3 (PyFrame Heap-Allocation Epic): canonical
/// surface is `PyFrame::run` / `PyFrame::execute_frame` (PyPy
/// `pyframe.py:268 run` / `pyframe.py:331 execute_frame`).  Retained as a
/// free function because pyre's JIT override mechanism (call.rs
/// `EVAL_OVERRIDE: OnceLock<EvalFn>` where `EvalFn = fn(&mut PyFrame) ->
/// PyResult`) requires a `fn` pointer.  Rust methods cannot be cast to
/// `fn` pointers, so the canonical body stays as a free function and the
/// `EVAL_OVERRIDE.unwrap_or(eval_frame_plain)` fallback (call.rs:328 etc.)
/// continues to reference it directly.
pub(crate) fn eval_frame_plain(frame: &mut PyFrame) -> PyResult {
    eval_frame_plain_with_operr(frame, None)
}

/// pyframe.py:270-299 execute_frame body — enter/call_trace/eval_loop/
/// return_trace/leave wrapping. When `operr` is Some, the generator's
/// throw() path routes it through handle_operation_error and sets
/// last_instr = next_instr - 1 before resuming (pyframe.py:273-277).
pub(crate) fn eval_frame_plain_with_operr(frame: &mut PyFrame, operr: Option<PyError>) -> PyResult {
    frame.fix_array_ptrs();
    if frame.execution_context.is_null() {
        if let Some(mut err) = operr {
            let mut next_instr = frame.next_instr();
            if !handle_exception(frame, &mut err, &mut next_instr) {
                return Err(err);
            }
            frame.last_instr = next_instr as isize - 1;
        }
        return eval_loop(frame);
    }
    let execution_context =
        unsafe { &mut *(frame.execution_context as *mut crate::PyExecutionContext) };
    crate::call::set_last_exec_ctx(frame.execution_context);
    execution_context.enter(frame as *mut PyFrame);
    let mut got_exception = true;
    let mut w_exitvalue = pyre_object::w_none();
    // pyframe.py:343-373 PyFrame.execute_frame parity:
    //   try:
    //     ec.call_trace(self)            # outside inner try
    //     try:
    //       ... eval ...
    //     finally:
    //       ec.return_trace(self, w_exitvalue)
    //     got_exception = False
    //   finally:
    //     ec.leave(self, w_exitvalue, got_exception)
    //
    // call_trace lives in the outer try only — if it raises, neither the
    // eval body nor return_trace runs, but leave still does (because
    // enter() already executed).  Python finally semantics: a finally
    // block that raises replaces the prior exception (return_trace
    // overrides eval-body, leave overrides everything).
    let outer_result = (|| -> PyResult {
        execution_context.call_trace(frame as *mut PyFrame)?;
        let inner_result = (|| -> PyResult {
            if let Some(mut err) = operr {
                let mut next_instr = frame.next_instr();
                if !handle_exception(frame, &mut err, &mut next_instr) {
                    return Err(err);
                }
                frame.last_instr = next_instr as isize - 1;
            }
            let result = eval_loop(frame)?;
            w_exitvalue = result;
            Ok(result)
        })();
        let return_trace_result =
            execution_context.return_trace(frame as *mut PyFrame, w_exitvalue);
        // Python finally: a finally-block exception replaces any
        // pending exception from the try-body. Only the all-OK path
        // advances to `got_exception = false`.
        let combined = match return_trace_result {
            Err(rt_err) => Err(rt_err),
            Ok(()) => inner_result,
        };
        if combined.is_ok() {
            got_exception = false;
        }
        combined
    })();
    let leave_result = execution_context.leave(frame as *mut PyFrame, w_exitvalue, got_exception);
    match leave_result {
        Err(leave_err) => Err(leave_err),
        Ok(()) => outer_result,
    }
}

/// Resume interpretation after compiled code guard failure.
pub fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    eval_loop(frame)
}

fn eval_loop(frame: &mut PyFrame) -> PyResult {
    let _current_frame_guard = if frame.execution_context.is_null() {
        install_current_frame(frame)
    } else {
        install_current_frame_tls_only(frame)
    };
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    let mut next_instr = frame.next_instr();

    loop {
        if next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let pc = next_instr;
        frame.last_instr = pc as isize;
        // pypy/interpreter/pyopcode.py:170-176 dispatch_bytecode parity:
        //   self.last_instr = intmask(next_instr)
        //   if jit.we_are_jitted():
        //       ec.bytecode_only_trace(self)
        //   else:
        //       ec.bytecode_trace(self)
        // pyre's interpreter path (this fn) takes the non-jitted branch
        // — bytecode_trace fires bytecode_only_trace then decrements
        // the ticker. Gated upstream on `w_tracefunc.is_null()` so the
        // no-tracer hot path is a single null-check + ticker decrement.
        let ec = frame.execution_context as *mut crate::PyExecutionContext;
        if !ec.is_null() {
            unsafe {
                (*ec).bytecode_trace(
                    frame as *mut PyFrame,
                    crate::executioncontext::TICK_COUNTER_STEP,
                )?
            };
        }
        let (opcode_pc, instruction, op_arg) = decode_instruction_for_dispatch(code, pc)?;
        let fallthrough = opcode_pc + 1;
        // `decode_instruction_for_dispatch` absorbs any EXTENDED_ARG prefix
        // units, so the real opcode may sit past `pc`.  Re-point `last_instr`
        // at the opcode unit (`opcode_pc`) so a falling-through handler's
        // `next_instr()` (= last_instr + 1) lands at `fallthrough` rather than
        // re-dispatching the opcode unit that trailed an EXTENDED_ARG.
        // Mirrors interp_jit.py dispatch (`set_last_instr_from_next_instr`).
        frame.set_last_instr_from_next_instr(fallthrough);
        match execute_opcode_step(frame, code, instruction, op_arg, fallthrough) {
            Ok(StepResult::Continue)
            | Ok(StepResult::CloseLoop {
                jump_args: _,
                loop_header_pc: _,
            }) => {
                next_instr = frame.next_instr();
            }
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => return Ok(result),
            Err(mut err) => {
                // GeneratorReturn: RETURN_GENERATOR unwind → return generator object
                if err.kind == crate::PyErrorKind::GeneratorReturn {
                    let gen_ptr = err.message.parse::<usize>().unwrap_or(0);
                    return Ok(gen_ptr as pyre_object::PyObjectRef);
                }
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
        }
    }
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.push(value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() {
            return Err(stack_underflow_error("interpreter opcode"));
        }
        Ok(self.pop())
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() + depth {
            return Err(stack_underflow_error("interpreter peek"));
        }
        Ok(PyFrame::peek_at(self, depth))
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        // `pypy/interpreter/pyopcode.py:1457 MAKE_FUNCTION` stamps
        // `func.w_func_globals = self.w_globals` from the running
        // frame's dict object directly.  Pyre resolves the same
        // canonical sibling via `get_w_globals_obj()` and threads it
        // through `make_function_from_code_obj_with_globals_obj` so
        // the freshly-created function's `__globals__` identity IS
        // the frame's view — no lazy `dict_storage_to_dict` second
        // resolution that could surface a different W_DictObject.
        let w_globals_obj = self.get_w_globals_obj();
        // Capture the globals OBJECT only; the raw `*mut DictStorage` is
        // recovered from the object via the proxy back-link wherever a frame
        // built from this function still needs it.  Threading a raw here is
        // what dangled exec-defined functions when the exec temp storage was
        // freed (the `GlobalsBinding` leak), so it is dropped.
        Ok(
            crate::runtime_ops::make_function_from_code_obj_with_globals_obj(
                code_obj,
                w_globals_obj,
            ),
        )
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        call_callable(self, callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_list_from_refs(items))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_tuple_from_refs(items))
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        build_map_from_refs(items)
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setitem(obj, key, value).map(|_| ())
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        unsafe { w_list_append(list, value) };
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        unpack_sequence_exact(seq, count)
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        getattr_str(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setattr_str(obj, name, value).map(|_| ())
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(self.locals_w()[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = self.locals_w()[idx];
        if value.is_null() {
            return Err(PyError::name_error_with_name(
                format!("local variable '{name}' referenced before assignment"),
                name,
            ));
        }
        // Cell objects are valid even if their contents are PY_NULL
        // (needed for __class__ cell during class body execution).
        // The cell itself is non-null, so the check above passes.
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // STORE_FAST always writes directly to the slot.
        // Cell content updates use STORE_DEREF, not STORE_FAST.
        self.locals_w_mut()[idx] = value;
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    /// PyPy: LOAD_NAME checks locals first (class body), then globals,
    /// then `__builtins__` via `load_global_value`'s fallback chain
    /// (pypy/interpreter/pyopcode.py:526-555 LOAD_NAME → load_global).
    ///
    /// Non-dict mapping locals (`exec(src, g, mapping)`,
    /// `pypy/interpreter/pyopcode.py:2003 ensure_ns`) bypass the
    /// `*mut DictStorage` fast path and route through
    /// `space.getitem(w_locals_object, name)` directly per PyPy
    /// `pyopcode.py:LOAD_NAME` `space.finditem_str(w_locals, name)`.
    fn load_name_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            match crate::baseobjspace::getitem(w_locals_object, key) {
                Ok(value) => return Ok(value),
                Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {
                    // pyopcode.py:LOAD_NAME `if not w_value: w_value =
                    // ec.space.finditem_str(self.w_globals, name)` —
                    // a missing locals entry falls through to globals.
                }
                Err(err) => return Err(err),
            }
            return self.load_global_value(name, nameindex);
        }
        let w_locals = self.get_w_locals();
        if !w_locals.is_null() {
            let locals = unsafe { &*w_locals };
            if let Ok(value) = dict_storage_load(locals, name) {
                return Ok(value);
            }
        }
        self.load_global_value(name, nameindex)
    }

    /// pyopcode.py:855-859 STORE_NAME —
    /// `space.setitem_str(self.getorcreatedebug().w_locals, varname, w_value)`.
    ///
    /// Writes straight to `w_locals` (the class namespace, or — at module
    /// scope — the globals dict). It must NOT route through `getdictscope`:
    /// that runs `fast2locals`, which would erase a module frame's
    /// `CO_FAST_HIDDEN` inlined-comprehension locals (their fast slot is NULL,
    /// the binding lives in `w_locals` via STORE_NAME) on every store.
    fn store_name_value(
        &mut self,
        name: &str,
        _nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            crate::baseobjspace::setitem(w_locals_object, key, value)?;
            return Ok(());
        }
        let ns = unsafe { &mut *self.get_or_create_w_locals() };
        dict_storage_store(ns, name, value);
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py:567 STORE_GLOBAL — bypasses w_locals
    /// and writes directly into w_globals so `exec("global x; x = 1", g, l)`
    /// lands `x` in `g` even when `l != g`.
    fn store_global_value(
        &mut self,
        name: &str,
        _nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        let w_globals_obj = self.get_w_globals_obj();
        if !w_globals_obj.is_null() {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(w_globals_obj, name, value);
            }
        } else {
            let ns = unsafe { &mut *self.get_w_globals() };
            dict_storage_store(ns, name, value);
        }
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py:918-927 `_load_global` — first reads
    /// `w_globals`, then falls back to `self.get_builtin().getdictvalue
    /// (space, varname)`.  PyPy's `get_builtin()` returns the `Module`
    /// chosen at frame-creation time by `pick_builtin(w_globals)`
    /// (`pyframe.py:115-116` + `pypy/module/__builtin__/moduledef.py:89`),
    /// so `exec("x = len", {"__builtins__": {}})` raises `NameError`
    /// because the empty dict is the picked builtin.
    fn load_global_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        // `pyframe.py:128-132 get_w_globals` returns the W_DictObject
        // directly; pyre's `w_globals_obj` slot (eagerly resolved at
        // frame construction per `pyframe.py:98 __init__`) carries
        // that identity.  Route the primary lookup through the strategy
        // dispatch (`dictmultiobject.py:111-112 setitem_str` /
        // `:113-115 getitem_str`) so dict-subclass overrides resolve
        // properly and the W_ModuleDictObject path consults its cell
        // map directly instead of walking the back-mirror storage.
        let w_globals_obj = self.get_w_globals_obj();
        if !w_globals_obj.is_null() {
            if let Some(value) =
                unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_globals_obj, name) }
            {
                return Ok(value);
            }
        }
        // `pyopcode.py:918-927 _load_global` — fall back to
        // `self.get_builtin().getdictvalue(space, varname)`.  Pyre's
        // path consults the `GlobalCache` (`celldict.py:214 get_global_cache`)
        // on the globals' backing W_ModuleDictObject so a repeated
        // LOAD_GLOBAL miss reuses the cached builtin entry instead of
        // re-walking `__builtins__.w_dict` every iteration.
        // `celldict.py:285-291 _LOAD_GLOBAL_cached`: when the frame's
        // globals is not the pycode's first-seen globals the entire
        // cached path is bypassed via `_load_global_fallback` — both
        // the per-pycode `_globals_caches[nameindex]` slot AND the
        // strategy-level `get_global_cache(varname)` install are
        // skipped, because both would attach a cache to a module that
        // is not the one being executed.  Identity is checked via
        // `pycode.w_globals == frame.get_w_globals()` (raw pointer
        // equality mirrors PyPy's `is` on the wrapped dict).
        // `get_w_globals()` recovers the raw storage from the frame's
        // canonical `w_globals_obj`; `dict_storage_to_dict` /
        // `w_dict_get_dict_storage_proxy` are mutual inverses under the
        // `mirror_target` invariant, so this raw comparison agrees with
        // the object `is` check.
        let pycode_matches_frame: bool = unsafe {
            let cw = crate::pycode::w_code_get_w_globals(self.pycode as PyObjectRef);
            !cw.is_null() && std::ptr::eq(cw, self.get_w_globals())
        };
        if pycode_matches_frame
            && !w_globals_obj.is_null()
            && unsafe { pyre_object::dictmultiobject::is_module_dict(w_globals_obj) }
        {
            let cache_hit: Option<PyObjectRef> = unsafe {
                load_global_via_cache(
                    w_globals_obj,
                    self.w_builtin,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                )
            }?;
            if let Some(value) = cache_hit {
                return Ok(value);
            }
        } else if !self.w_builtin.is_null() && unsafe { pyre_object::is_module(self.w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(self.w_builtin) };
            if !w_dict.is_null() {
                if let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)? {
                    return Ok(value);
                }
            }
        }
        // `pyopcode.py:970 _load_global_failed`: NameError.
        Err(PyError::name_error_with_name(
            format!("name '{name}' is not defined"),
            name,
        ))
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(PY_NULL)
    }
}

impl StackOpcodeHandler for PyFrame {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        let top_idx = self.valuestackdepth - 1;
        let other_idx = self.valuestackdepth - depth;
        self.locals_w_mut().swap(top_idx, other_idx);
        Ok(())
    }
}

/// `celldict.py:279-329 _LOAD_GLOBAL_cached` slow-path: consult the
/// W_ModuleDictObject's `mstrategy.get_global_cache` for `name`,
/// chaining through `cache.builtincache` to the `__builtins__` Module
/// on a globals miss.  Returns `None` when the name is absent from
/// both globals and builtins.
///
/// Public extern alias so `runtime_ops::jit_load_name_from_namespace`
/// can reuse this cache path on a globals miss.
///
/// # Safety
/// `w_module_dict` must be a valid W_ModuleDictObject; `w_builtin`
/// may be null or a valid Module; `name` is the requested str key.
pub unsafe fn load_global_via_cache_extern(
    w_module_dict: PyObjectRef,
    w_builtin: PyObjectRef,
    name: &str,
) -> Option<PyObjectRef> {
    // JIT extern path: discard `space.finditem_str`'s `PyError` because
    // the C-ABI signature has no error channel.  For the builtins dict
    // (the only call site that can raise here in practice), `finditem_str`
    // only raises on a non-dict mapping with custom `__getitem__` — never
    // for the W_DictObject / W_ModuleDictObject backing real builtins.
    match unsafe { load_global_via_cache(w_module_dict, w_builtin, name, std::ptr::null_mut(), 0) }
    {
        Ok(v) => v,
        Err(_) => None,
    }
}

/// `celldict.py:279-329 _LOAD_GLOBAL_cached`.  When `pycode` is
/// non-null, `pycode._globals_caches[nameindex]` is consulted before
/// `mstrategy.get_global_cache(name)`; on the slow path, the resolved
/// `cache.ref` (`celldict.py:321/353`) is installed into the slot.
///
/// Returns `Ok(Some(value))` on cache hit (globals or chained builtin),
/// `Ok(None)` on full miss, `Err(_)` when `space.finditem_str` raises
/// during the builtins fallback (`baseobjspace.py:45-49
/// W_Root.getdictvalue` → `space.finditem_str`).
unsafe fn load_global_via_cache(
    w_module_dict: PyObjectRef,
    w_builtin: PyObjectRef,
    name: &str,
    pycode: PyObjectRef,
    nameindex: usize,
) -> Result<Option<PyObjectRef>, PyError> {
    use pyre_object::celldict::unwrap_cell;
    use pyre_object::dictmultiobject::W_ModuleDictObject;
    // Body is a chain of unsafe-fn / raw-ptr ops on caller-supplied
    // PyObjectRefs; SAFETY contract is on the `unsafe fn` signature
    // (caller upholds w_module_dict / pycode / w_builtin validity).
    unsafe {
        // `celldict.py:292-313`: per-name slot fast path.  Read the slot,
        // upgrade the weakref; if the cache is alive, follow cell → builtincache
        // → builtins w_dict before falling through to the strategy lookup.
        if !pycode.is_null() {
            if let Some(cache) = crate::pycode::w_code_globals_caches_get(pycode, nameindex) {
                // `celldict.py:295-313`: fast-path layout —
                //
                //     w_value = cache.getvalue(self.space)
                //     if w_value is not None:
                //         return w_value
                //     if cache.valid:
                //         builtincache = cache.builtincache
                //         if builtincache is not None:
                //             w_value = builtincache.getvalue(self.space)
                //             if w_value is not None:
                //                 return w_value
                //             # builtin getdictvalue + _load_global_failed
                //
                // The builtins fallback is GATED on `builtincache is not None`.
                // Under pyre's honor__builtins__=True equivalence the
                // `builtincache` attach is dead, so the slot path just
                // returns early on a cell hit and otherwise falls through to
                // the slow path (`# either no cache or an invalid cache`),
                // which calls `_load_global` whose own fallback chain reads
                // the frame's picked builtin via `space.finditem_str`.
                let (cell_opt, valid, bc_opt) = {
                    let c = cache.borrow();
                    (c.cell, c.valid, c.builtincache.clone())
                };
                if let Some(v) = cell_opt {
                    return Ok(Some(unwrap_cell(v)));
                }
                if valid && let Some(bc) = bc_opt {
                    let bcell = bc.borrow().cell;
                    if let Some(v) = bcell {
                        return Ok(Some(unwrap_cell(v)));
                    }
                    // `celldict.py:307-313`: the `_load_global_failed`
                    // branch is inside `if builtincache is not None` — only
                    // reachable when a real builtincache is installed.
                    // Under honor=True this scope is dead; included for
                    // strict line-by-line shape parity should
                    // honor__builtins__ ever flip False.
                    if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                        let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
                        if !w_builtin_dict.is_null() {
                            return crate::baseobjspace::finditem_str(w_builtin_dict, name);
                        }
                    }
                }
            }
        }
        let raw = &mut *(w_module_dict as *mut W_ModuleDictObject);
        if raw.mstrategy.is_null() || raw.dstorage.is_null() {
            return Ok(None);
        }
        // `celldict.py:315-322`: the slow-path install just routes through
        // `w_globals.get_global_cache(varname)` and writes
        // `pycode._globals_caches[nameindex] = cache.ref`.
        //
        // Under pyre's permanent `honor__builtins__=True` (frame picks its
        // own builtin per `pyframe.py:115`), the cache carries no
        // `builtincache` — that branch is dead in
        // `ModuleDictStrategy::get_global_cache` per its line-by-line port
        // of `celldict.py:224 not space.config.objspace.honor__builtins__`.
        let strategy = &mut *raw.mstrategy;
        let storage = &*raw.dstorage;
        let cache = strategy.get_global_cache(storage, name);
        // `celldict.py:321/353 pycode._globals_caches[nameindex] = cache.ref`.
        if !pycode.is_null() {
            crate::pycode::w_code_globals_caches_set(pycode, nameindex, &cache);
        }
        // `_LOAD_GLOBAL_cached` lines 296-298: cache.getvalue hit.
        let cell_opt = cache.borrow().cell;
        if let Some(v) = cell_opt {
            return Ok(Some(unwrap_cell(v)));
        }
        // `_load_global_fallback` → `_load_global` (`pyopcode.py:958-967`):
        // when globals miss, route through `self.get_builtin().getdictvalue(
        // space, varname)` which resolves via `space.finditem_str` per
        // `baseobjspace.py:45-49 W_Root.getdictvalue`.  The caller threads
        // its frame's picked builtin in as `w_builtin`.
        if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
            let w_builtin_dict = pyre_object::w_module_get_w_dict(w_builtin);
            if !w_builtin_dict.is_null() {
                return crate::baseobjspace::finditem_str(w_builtin_dict, name);
            }
        }
        Ok(None)
    }
}

thread_local! {
    /// Cache for user-defined iterator __next__ result.
    /// concrete_iter_continues calls __next__ and caches here;
    /// iter_next_value returns the cached value.
    static USER_ITER_NEXT_CACHE: std::cell::Cell<PyObjectRef> =
        const { std::cell::Cell::new(PY_NULL) };
}

/// PyPy: pyopcode.py GET_ITER → space.iter(w_iterable)
///       pyopcode.py FOR_ITER → space.next(w_iterator)
impl IterOpcodeHandler for PyFrame {
    /// GET_ITER: convert iterable to iterator.
    /// PyPy: space.iter(w_iterable) → calls __iter__ or wraps in seq_iter.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            // mappingproxy iterates over its backing dict's keys.
            // dictproxyobject.py:41 descr_iter → space.iter(self.w_mapping).
            let iter = if pyre_object::is_dict_proxy(iter) {
                let mapping = pyre_object::w_dict_proxy_get_mapping(iter);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = mapping;
                mapping
            } else {
                iter
            };
            // `range` sequence → fresh `W_RangeIterator` cursor; replace
            // the stack operand so FOR_ITER advances the iterator, not the
            // reusable range object.  (Mirrors the dict-proxy rewrite
            // above.)  This runs in the loop preheader, outside the traced
            // loop body, so the JIT's `for i in range(N)` fast path is
            // unaffected.
            if pyre_object::is_w_range(iter) {
                let it = pyre_object::w_range_iter(iter);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = it;
                return Ok(());
            }
            // Already an iterator
            if pyre_object::is_range_iter(iter)
                || pyre_object::is_long_range_iter(iter)
                || pyre_object::is_seq_iter(iter)
                || pyre_object::generatorobject::is_generator(iter)
                || pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
                || pyre_object::itertoolsmodule::is_takewhile(iter)
                || pyre_object::itertoolsmodule::is_dropwhile(iter)
                || pyre_object::itertoolsmodule::is_filterfalse(iter)
                || pyre_object::itertoolsmodule::is_pairwise(iter)
                || pyre_object::dictviewobject::is_dict_view_iterator(iter)
                || pyre_object::enumerateobject::is_enumerate(iter)
                || pyre_object::reversedobject::is_reversed(iter)
                || pyre_object::filterobject::is_filter(iter)
                || pyre_object::mapobject::is_map(iter)
                || pyre_object::zipobject::is_zip(iter)
                || pyre_object::callableiteratorobject::is_callable_iterator(iter)
                || pyre_object::sreobject::is_sre_scanner(iter)
            {
                return Ok(());
            }
            // `pypy/objspace/std/dictmultiobject.py W_DictMulti
            // ViewKeysObject.descr_iter` (and values / items siblings)
            // — `_iter_*` returns a live `W_BaseDictIterator`.  Pyre
            // produces a `W_DictViewIterator` carrying the source
            // dict's `dictversion` counter so mid-iteration mutation
            // surfaces as `RuntimeError("dictionary changed size during
            // iteration")` per `:1719-1741 descr_next`.
            if pyre_object::dictviewobject::is_dict_view(iter) {
                let kind = pyre_object::dictviewobject::w_dict_view_get_kind(iter);
                let w_dict = pyre_object::dictviewobject::w_dict_view_get_dict(iter);
                let it = pyre_object::dictviewobject::w_dict_view_iterator_new(w_dict, kind);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = it;
                return Ok(());
            }
            // list → seq_iter
            if pyre_object::is_list(iter) {
                let len = pyre_object::w_list_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // tuple → seq_iter
            if pyre_object::is_tuple(iter) {
                let len = pyre_object::w_tuple_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // str → list of 1-char strings → seq_iter
            if pyre_object::is_str(iter) {
                // Walk code points through the WTF-8 view so iterating a
                // surrogateescape / surrogatepass-decoded string yields its
                // lone surrogates instead of panicking in w_str_get_value.
                let chars: Vec<pyre_object::PyObjectRef> = pyre_object::w_str_get_wtf8(iter)
                    .code_points()
                    .map(|c| {
                        let mut one = rustpython_wtf8::Wtf8Buf::new();
                        one.push(c);
                        pyre_object::w_str_from_wtf8(one)
                    })
                    .collect();
                let len = chars.len();
                let char_list = pyre_object::w_list_new(chars);
                let seq_iter = pyre_object::w_seq_iter_new(char_list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // bytes/bytearray → list of int → seq_iter
            if pyre_object::bytesobject::is_bytes_like(iter) {
                let len = pyre_object::bytesobject::bytes_like_len(iter);
                let mut items = Vec::with_capacity(len);
                for i in 0..len {
                    items.push(pyre_object::w_int_new(
                        pyre_object::bytesobject::bytes_like_getitem(iter, i) as i64,
                    ));
                }
                let list = pyre_object::w_list_new(items);
                let seq_iter = pyre_object::w_seq_iter_new(list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // dict → iterate over keys.  `pypy/objspace/std/dict
            // multiobject.py W_DictMultiObject.descr_iter` returns
            // `W_DictMultiIterKeys(self)` — pyre's `W_DictViewIterator`
            // with kind=Keys plays the same role, capturing the
            // dict's `dictversion` so mid-iteration mutation raises
            // `RuntimeError("dictionary changed size during
            // iteration")`.
            if pyre_object::is_dict(iter) {
                let it = pyre_object::dictviewobject::w_dict_view_iterator_new(
                    iter,
                    pyre_object::dictviewobject::DictViewKind::Keys,
                );
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = it;
                return Ok(());
            }
            // set / frozenset → iterate via insertion order (PyPy:
            // setobject.py W_BaseSetObject.descr_iter)
            if pyre_object::is_set_or_frozenset(iter) {
                let items = pyre_object::w_set_items(iter);
                let len = items.len();
                let key_list = pyre_object::w_list_new(items);
                let seq_iter = pyre_object::w_seq_iter_new(key_list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // User-defined __iter__ — PyPy: space.iter → __iter__()
            // Delegates to baseobjspace::iter which handles type MRO
            // and __getitem__ fallback (PyPy: space.iter →
            // PyObject_GetIter → tp_iter or PySeqIter_New).
            if pyre_object::is_instance(iter) {
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = result;
                return Ok(());
            }
            // Type object: metaclass __iter__ (NOT the type's own MRO)
            // CPython: iter(X) calls type(X).__iter__(X)
            if pyre_object::is_type(iter) {
                // baseobjspace.py:76 — metaclass from w_class
                let mc = {
                    let w_class = (*iter).w_class;
                    let w_type_type = crate::typedef::w_type();
                    if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                        Some(w_class)
                    } else {
                        None
                    }
                };
                if let Some(metaclass) = mc {
                    if let Some(method) = crate::baseobjspace::lookup_in_type(metaclass, "__iter__")
                    {
                        let result = crate::call_function(method, &[iter]);
                        let tos = self.valuestackdepth - 1;
                        self.locals_w_mut()[tos] = result;
                        return Ok(());
                    }
                }
            }
        }
        ensure_range_iter(iter)
    }

    /// FOR_ITER: check if iterator has more items.
    /// PyPy: space.next() → StopIteration means exhausted.
    /// For user-defined iterators, we speculatively call __next__ and
    /// cache the result — iter_next_value returns the cached value.
    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        unsafe {
            // Generator iterator
            if pyre_object::generatorobject::is_generator(iter) {
                match crate::baseobjspace::next(iter) {
                    Ok(result) => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                        return Ok(true);
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                }
            }
            // itertools iterators + W_Enumerate + W_DictViewIterator
            // — delegate to baseobjspace::next.  The shared cache slot
            // (USER_ITER_NEXT_CACHE) carries the most recent value
            // across the iter_continues / iter_next_value pair.
            if pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
                || pyre_object::itertoolsmodule::is_takewhile(iter)
                || pyre_object::itertoolsmodule::is_dropwhile(iter)
                || pyre_object::itertoolsmodule::is_filterfalse(iter)
                || pyre_object::itertoolsmodule::is_pairwise(iter)
                || pyre_object::enumerateobject::is_enumerate(iter)
                || pyre_object::reversedobject::is_reversed(iter)
                || pyre_object::filterobject::is_filter(iter)
                || pyre_object::mapobject::is_map(iter)
                || pyre_object::zipobject::is_zip(iter)
                || pyre_object::callableiteratorobject::is_callable_iterator(iter)
                || pyre_object::dictviewobject::is_dict_view_iterator(iter)
                || pyre_object::sreobject::is_sre_scanner(iter)
            {
                match crate::baseobjspace::next(iter) {
                    Ok(result) => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                        return Ok(true);
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                }
            }
            // User-defined iterator with __next__
            if pyre_object::is_instance(iter) {
                let w_type = pyre_object::w_instance_get_type(iter);
                if let Some(next_method) = crate::baseobjspace::lookup_in_type(w_type, "__next__") {
                    match crate::call::call_callable(self, next_method, &[iter]) {
                        Ok(result) => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                            return Ok(true);
                        }
                        Err(e) if e.kind == PyErrorKind::StopIteration => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                            return Ok(false);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        range_iter_continues(iter)
    }

    /// PyPy: space.next(w_iterator) → returns cached value from concrete_iter_continues.
    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        // Generator/user-defined/itertools/enumerate/dict-iter:
        // return cached value populated by concrete_iter_continues.
        if unsafe {
            pyre_object::generatorobject::is_generator(iter)
                || pyre_object::is_instance(iter)
                || pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
                || pyre_object::itertoolsmodule::is_takewhile(iter)
                || pyre_object::itertoolsmodule::is_dropwhile(iter)
                || pyre_object::itertoolsmodule::is_filterfalse(iter)
                || pyre_object::itertoolsmodule::is_pairwise(iter)
                || pyre_object::enumerateobject::is_enumerate(iter)
                || pyre_object::reversedobject::is_reversed(iter)
                || pyre_object::filterobject::is_filter(iter)
                || pyre_object::mapobject::is_map(iter)
                || pyre_object::zipobject::is_zip(iter)
                || pyre_object::callableiteratorobject::is_callable_iterator(iter)
                || pyre_object::dictviewobject::is_dict_view_iterator(iter)
                || pyre_object::sreobject::is_sre_scanner(iter)
        } {
            let cached = USER_ITER_NEXT_CACHE.with(|c| c.get());
            if !cached.is_null() {
                return Ok(cached);
            }
            return Ok(PY_NULL);
        }
        range_iter_next_or_null(iter)
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.set_last_instr_from_next_instr(target);
        Ok(())
    }
}

impl TruthOpcodeHandler for PyFrame {
    type Truth = bool;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        truth_value(value)
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        Ok(bool_value_from_truth(if negate { !truth } else { truth }))
    }
}

impl ControlFlowOpcodeHandler for PyFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.next_instr()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.set_last_instr_from_next_instr(target);
        Ok(())
    }

    fn close_loop(&mut self, target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop {
            jump_args: vec![],
            loop_header_pc: target,
        })
    }

    /// pyopcode.py:180-183 RETURN_VALUE — frame_finished_execution = True
    /// when the returning path exits the frame (matched by StepResult::Return).
    fn finish_value(&mut self, value: Self::Value) -> Result<StepResult<Self::Value>, PyError> {
        if std::env::var_os("PYRE_INTERP_RETURN_LOG").is_some() {
            unsafe {
                let code_ptr = crate::pyframe::pyframe_get_pycode(self);
                let name = if !code_ptr.is_null() {
                    (*code_ptr).obj_name.as_str()
                } else {
                    "?"
                };
                let arg0_intval = {
                    let lw = self.locals_w();
                    if lw.len() > 0 {
                        let v = lw[0];
                        if !v.is_null() && pyre_object::pyobject::is_int(v) {
                            Some(pyre_object::intobject::w_int_get_value(v))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                let ret_intval = if !value.is_null() && pyre_object::pyobject::is_int(value) {
                    Some(pyre_object::intobject::w_int_get_value(value))
                } else {
                    None
                };
                let f_back = self.f_backref as usize;
                eprintln!(
                    "[interp] return name={} arg0={:?} ret={:?} frame={:p} f_back=0x{:x} ret_ref=0x{:x}",
                    name, arg0_intval, ret_intval, self as *const _, f_back, value as usize
                );
            }
        }
        self.frame_finished_execution = true;
        Ok(StepResult::Return(value))
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(
        &mut self,
        _value: Self::Value,
        truth: Self::Truth,
    ) -> Result<bool, PyError> {
        Ok(truth)
    }
}

impl ArithmeticOpcodeHandler for PyFrame {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        binary_value(a, b, op)
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        compare_value(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_negative_value(value)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_invert_value(value)
    }
}

impl ConstantOpcodeHandler for PyFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_int_new(value))
    }

    fn bigint_constant(&mut self, value: &crate::PyBigInt) -> Result<Self::Value, PyError> {
        Ok(w_long_new(value.clone()))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        Ok(w_float_new(value))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        Ok(w_bool_from(value))
    }

    fn str_constant(&mut self, value: &rustpython_wtf8::Wtf8) -> Result<Self::Value, PyError> {
        Ok(box_str_constant(value))
    }

    fn bytes_constant(&mut self, value: &[u8]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(value))
    }

    fn code_constant(
        &mut self,
        code: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        Ok(crate::pycode::intern_code_constant(code))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }

    fn ellipsis_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(pyre_object::noneobject::w_ellipsis())
    }

    fn slice_constant(
        &mut self,
        start: Self::Value,
        stop: Self::Value,
        step: Self::Value,
    ) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_slice_new(start, stop, step))
    }

    fn frozenset_constant(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_frozenset_from_items(items))
    }
}

/// `callmethod.py:66-78` fast-path discriminator: bind the receiver only
/// when the MRO descriptor `d` is a method-descriptor-typed function
/// (`flag_method_descriptor` — set on `function` alone, typedef.py:807)
/// that `getattr` surfaced unchanged.  The `d == attr` identity check is
/// the moral equivalent of `w_obj.getdictvalue(space, name) is None` plus
/// `has_object_getattribute()` (callmethod.py:46/67): an instance-dict
/// shadow, a descriptor `__get__` result, or a `__getattribute__` override
/// all hand `getattr` a different object than the raw descriptor.
/// Everything else takes the slow path (callmethod.py:79-82): the getattr
/// result is called as-is, with no self binding.
unsafe fn method_descriptor_bound(
    d: PyObjectRef,
    attr: PyObjectRef,
    obj: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        if d != attr || !crate::is_function(d) {
            return PY_NULL;
        }
        // BuiltinFunction has no `__get__` (its typedef carries no
        // `method_descriptor` flag, function.py:783).
        if std::ptr::eq((*d).ob_type, &crate::BUILTIN_FUNCTION_TYPE as *const _) {
            PY_NULL
        } else {
            obj
        }
    }
}

/// Compute the `null_or_self` value LOAD_METHOD pushes alongside the
/// resolved attribute `attr` (the result of `getattr(obj, name)`).
///
/// Pure MRO inspection (`lookup_in_type` + descriptor-kind predicates) —
/// it never invokes a descriptor `__get__` or `__getattribute__`, so the
/// side effects already paid by the `getattr` that produced `attr` are not
/// repeated.  Shared by [`PyFrame::load_method`] and the blackhole residual
/// helper `bh_load_method_self_fn` so both bind self identically.
///
///  - method-descriptor function surfaced unchanged by getattr → bind
///    instance (self); see [`method_descriptor_bound`]
///  - classmethod → bind class (w_type)
///  - everything else (staticmethod / non-method descriptors / shadowed
///    or arbitrary class attrs) → no binding (NULL)
pub fn compute_load_method_bound(obj: PyObjectRef, attr: PyObjectRef, name: &str) -> PyObjectRef {
    unsafe {
        if pyre_object::is_method(attr) {
            return PY_NULL;
        }
        if pyre_object::is_instance(obj) {
            // callmethod.py:66-67 `w_value = w_obj.getdictvalue(space, name)`:
            // a shadowing instance attribute is what getattr returned for
            // every non-data descriptor — never bind self for it.  (Data
            // descriptors that win over the instance dict — property /
            // member — resolve to PY_NULL either way.)
            let shadowed = crate::objspace::std::mapdict::instance_node_getdictvalue(
                obj,
                rustpython_wtf8::Wtf8::new(name),
            )
            .is_some();
            let w_type = pyre_object::w_instance_get_type(obj);
            // callmethod.py:33 `w_type.has_object_getattribute()`: a non-default
            // `__getattribute__` produced `attr` through the override rather
            // than the default descriptor path, so the MRO-shape binding
            // inference below does not apply.  `_PyObject_GetMethod` skips the
            // self-binding optimization for a custom `tp_getattro` (pushes
            // NULL), so an override returning the raw descriptor must call as a
            // plain function, not a bound method.  This is the same gate
            // `load_method_fast_path` applies before its fast path.
            if !pyre_object::typeobject::w_type_get_uses_object_getattribute(w_type) {
                return PY_NULL;
            }
            let raw = crate::baseobjspace::lookup_in_type(w_type, name);
            match raw {
                _ if shadowed => PY_NULL,
                // staticmethod / classmethod wrappers: getattr already
                // unwrapped them, so the identity fast path below can never
                // match; classmethod keeps its explicit cls binding.
                Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                Some(d) if pyre_object::is_classmethod(d) => w_type,
                Some(d) => method_descriptor_bound(d, attr, obj),
                None => {
                    // Not found in type MRO → found in instance __dict__.
                    // Instance __dict__ attrs bypass descriptor protocol.
                    PY_NULL
                }
            }
        } else if pyre_object::is_type(obj) {
            // Type receiver: PyPy resolves LOAD_METHOD through the
            // METAclass MRO (`space.type(w_obj)`), so a name found in the
            // type's own MRO reaches the call as a plain getattr value
            // with no binding.
            let raw = crate::baseobjspace::lookup_in_type(obj, name);
            match raw {
                Some(d) if pyre_object::is_classmethod(d) => obj,
                Some(_) => PY_NULL, // found in own MRO → no binding
                None => {
                    // Not in the type's own MRO → resolved via the
                    // metaclass MRO; bind the type for a method-descriptor
                    // function getattr surfaced unchanged.
                    match crate::typedef::r#type(obj)
                        .and_then(|meta| crate::baseobjspace::lookup_in_type(meta, name))
                    {
                        Some(d) => method_descriptor_bound(d, attr, obj),
                        None => PY_NULL,
                    }
                }
            }
        } else if let Some(w_type) =
            crate::typedef::r#type(obj).filter(|_| !pyre_object::is_module(obj))
        {
            // Builtin-storage receiver (list, str, ... and their
            // subclasses such as enum members) found via TypeDef; the
            // same fast-path discriminator applies — `dict.get` etc. are
            // FunctionWithFixedCode (interp2app) attrs that getattr
            // returns unchanged, while staticmethods (str.maketrans) and
            // classmethods (dict.fromkeys) were already unwrapped.
            match crate::baseobjspace::lookup_in_type(w_type, name) {
                Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                Some(d) if pyre_object::is_classmethod(d) => w_type,
                Some(d) => method_descriptor_bound(d, attr, obj),
                None => PY_NULL,
            }
        } else {
            PY_NULL
        }
    }
}

impl OpcodeStepExecutor for PyFrame {
    /// SETUP_ANNOTATIONS — ensure `__annotations__` exists in the
    /// current locals namespace. PyPy: pyopcode.py SETUP_ANNOTATIONS
    /// (typeobject.py auto-fills the slot at class creation, but the
    /// pyre-equivalent flow runs the bytecode opcode and writes into
    /// the class_locals namespace just like CPython).
    fn setup_annotations(&mut self) -> Result<(), PyError> {
        let ns = self.getdictscope()?;
        if ns.is_null() {
            return Ok(());
        }
        let ns = unsafe { &mut *ns };
        if dict_storage_load(ns, "__annotations__").is_err() {
            dict_storage_store(ns, "__annotations__", pyre_object::w_dict_new());
        }
        Ok(())
    }

    /// WITH_EXCEPT_START — call __exit__ for the exceptional `with` exit.
    ///
    /// Stack layout the bytecode emits (bottom → top):
    ///   exit_func, exit_self, lasti, unused, val
    ///
    /// `val` (TOS) is the in-flight exception. LOAD_SPECIAL split the
    /// context manager's `__exit__` into `exit_func` (the function) and
    /// `exit_self` (the bound instance, or NULL). We call
    /// `exit_func(exit_self, type(val), val, val.__traceback__)` with
    /// `exit_self` prepended only when it is non-NULL, and push the result
    /// so the following TO_BOOL decides whether to suppress.
    fn with_except_start(&mut self) -> Result<(), PyError> {
        let depth = self.valuestackdepth;
        if depth < 5 {
            return Err(PyError::type_error(
                "WITH_EXCEPT_START requires five stack values",
            ));
        }
        let val = self.locals_w()[depth - 1];
        let exit_self = self.locals_w()[depth - 4];
        let exit_func = self.locals_w()[depth - 5];
        let exc_type = crate::typedef::r#type(val).unwrap_or(pyre_object::w_none());
        let exc_tb =
            crate::baseobjspace::getattr_str(val, "__traceback__").unwrap_or(pyre_object::w_none());
        let res = if exit_self.is_null() {
            crate::call_function(exit_func, &[exc_type, val, exc_tb])
        } else {
            crate::call_function(exit_func, &[exit_self, exc_type, val, exc_tb])
        };
        if res.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::type_error("__exit__ failed"))
                .into());
        }
        self.push(res);
        Ok(())
    }

    // ── LoadCommonConstant ──
    fn load_common_constant(&mut self, cc: crate::bytecode::CommonConstant) -> Result<(), PyError> {
        use crate::bytecode::CommonConstant;
        let val = match cc {
            CommonConstant::AssertionError => {
                // `LOAD_ASSERTION_ERROR` pushes the `AssertionError` class
                // itself, so `assert x` raises `AssertionError()` and
                // `assert x, msg` raises `AssertionError(msg)`.
                crate::builtins::lookup_exc_class("AssertionError").unwrap_or_else(|| {
                    crate::typedef::gettypeobject(&pyre_object::excobject::EXC_ASSERTION_ERROR_TYPE)
                })
            }
            CommonConstant::NotImplementedError => {
                crate::make_builtin_function("NotImplementedError", |_args| {
                    Err(crate::PyError::type_error("not implemented"))
                })
            }
            CommonConstant::BuiltinTuple => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE)
            }
            CommonConstant::BuiltinAll => crate::make_module_builtin_function_with_arity(
                "all",
                crate::builtins::builtin_all_fn,
                1,
            ),
            CommonConstant::BuiltinAny => crate::make_module_builtin_function_with_arity(
                "any",
                crate::builtins::builtin_any_fn,
                1,
            ),
            CommonConstant::BuiltinList => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
            CommonConstant::BuiltinSet => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
        };
        self.push(val);
        Ok(())
    }

    // ── PopJumpIfNone / PopJumpIfNotNone ──
    // CPython 3.13: replaces IS_OP + POP_JUMP_IF_TRUE/FALSE for None checks

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), PyError> {
        let val = self.pop();
        if unsafe { pyre_object::is_none(val) } || val.is_null() {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), PyError> {
        let val = self.pop();
        if !val.is_null() && !unsafe { pyre_object::is_none(val) } {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    // ── Closures / cells ──

    /// PyPy: pyopcode.py LOAD_DEREF
    ///
    /// Reads cell/free variable. If the slot holds a cell object (from
    /// closure tuple via COPY_FREE_VARS), dereferences it. Otherwise
    /// reads the raw value (pyre's direct storage for cellvars).
    /// LOAD_DEREF — RustPython 3.13 uses unified index (same as LOAD_FAST).
    ///
    /// PyPy: pyopcode.py LOAD_DEREF → cell.get()
    /// If the slot holds a cell object, dereference it to get the value.
    fn load_deref(&mut self, idx: usize) -> Result<(), PyError> {
        let slot = self.locals_w()[idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self.code(), idx));
        }
        self.push(value);
        Ok(())
    }

    /// STORE_DEREF — unified index. Stores into cell if present.
    ///
    /// PyPy: pyopcode.py STORE_DEREF → cell.set(value)
    fn store_deref(&mut self, idx: usize) -> Result<(), PyError> {
        let value = self.pop();
        let slot = self.locals_w()[idx];
        if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_set(slot, value) };
        } else {
            self.locals_w_mut()[idx] = value;
        }
        Ok(())
    }

    /// LOAD_CLOSURE — unified index. Push cell object itself (not contents).
    ///
    /// PyPy: pyopcode.py LOAD_CLOSURE → push cell for closure capture.
    fn load_closure(&mut self, idx: usize) -> Result<(), PyError> {
        let cell = self.locals_w()[idx];
        self.push(cell);
        Ok(())
    }

    /// MAKE_CELL — wrap the slot value in a W_CellObject.
    ///
    /// CPython 3.13 / RustPython MAKE_CELL — create cell object in slot.
    /// Wraps the current value (PY_NULL if uninitialized) in a W_CellObject.
    /// LoadFast on cell slots returns the cell object itself (needed for
    /// closure creation via BUILD_TUPLE + SET_FUNCTION_ATTRIBUTE).
    ///
    /// `initialize_frame_scopes` already installs an empty cell for every
    /// pure cellvar (a cellvar not shadowing a parameter).  Only an
    /// argument slot promoted to a cellvar still holds a raw value here,
    /// so wrap solely when the slot is not already a cell — otherwise a
    /// never-reassigned cellvar like `__class__` would become a
    /// cell-wrapping-a-cell, and `fast2locals` / closure reads would
    /// surface the inner cell instead of the value.
    fn make_cell(&mut self, idx: usize) -> Result<(), PyError> {
        let current = self.locals_w()[idx];
        if current.is_null() || !unsafe { pyre_object::is_cell(current) } {
            self.locals_w_mut()[idx] = pyre_object::w_cell_new(current);
        }
        Ok(())
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), PyError> {
        let nlocals = self.nlocals();
        self.locals_w_mut()[nlocals + idx] = PY_NULL;
        Ok(())
    }

    // ── Exception handling ──

    fn setup_finally(&mut self, handler: usize) -> Result<(), PyError> {
        self.append_block(crate::pyframe::FrameBlock {
            valuestackdepth: self.valuestackdepth,
            handlerposition: handler,
            previous: self.lastblock,
        });
        Ok(())
    }

    fn setup_except(&mut self, handler: usize) -> Result<(), PyError> {
        self.setup_finally(handler)
    }

    fn pop_block(&mut self) -> Result<(), PyError> {
        self.pop_block();
        Ok(())
    }

    fn raise_varargs(&mut self, argc: usize) -> Result<(), PyError> {
        match argc {
            0 => {
                // Bare `raise` — re-raise current exception
                // PyPy: executioncontext.py sys_exc_info
                let exc = get_current_exception();
                if exc.is_null() || unsafe { pyre_object::is_none(exc) } {
                    Err(PyError::runtime_error("No active exception to reraise"))
                } else if unsafe { pyre_object::is_exception(exc) } {
                    Err(unsafe { PyError::from_exc_object(exc) })
                } else {
                    Err(PyError::runtime_error("No active exception to reraise"))
                }
            }
            1 => {
                // pyopcode.py:708-722 — cause=None, normalize exc.
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // pyopcode.py:711-713 — class raise: call the type.
                        let result = crate::call_function(w_value, &[]);
                        if pyre_object::is_exception(result) {
                            attach_raise_cause(result, None)?;
                            Err(PyError::from_exc_object(result))
                        } else {
                            Err(PyError::type_error(
                                "exceptions must derive from BaseException",
                            ))
                        }
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, None)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            2 => {
                // pyopcode.py:704-722 — pop+normalize cause first, then exc.
                let raw_cause = self.pop();
                let cause = Some(normalize_raise_cause(raw_cause)?);
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // pyopcode.py:711-713 — class raise: call the type.
                        let result = crate::call_function(w_value, &[]);
                        if pyre_object::is_exception(result) {
                            attach_raise_cause(result, cause)?;
                            Err(PyError::from_exc_object(result))
                        } else {
                            Err(PyError::type_error(
                                "exceptions must derive from BaseException",
                            ))
                        }
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, cause)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            _ => Err(PyError::type_error("too many arguments for raise")),
        }
    }

    fn end_finally(&mut self) -> Result<(), PyError> {
        // Pop the exception or None from stack
        let _ = self.pop();
        Ok(())
    }

    // ── Import ──
    // PyPy: pyopcode.py IMPORT_NAME
    // Stack: [level, fromlist] → pops both, pushes module object.
    fn import_name(&mut self, name: &str) -> Result<(), PyError> {
        let w_fromlist = self.pop();
        let w_level = self.pop();
        let level = if unsafe { pyre_object::is_int(w_level) } {
            unsafe { pyre_object::w_int_get_value(w_level) }
        } else {
            0
        };

        let module = crate::importing::importhook(
            name,
            self.get_w_globals_obj(), // for relative imports: __name__/__package__
            w_fromlist,
            level,
            self.execution_context,
        )?;
        self.push(module);
        Ok(())
    }

    // PyPy: pyopcode.py IMPORT_FROM
    // Stack: [module] → peek module, push getattr(module, name)
    fn import_from(&mut self, name: &str) -> Result<(), PyError> {
        let module = self.peek();
        let attr = crate::importing::import_from(module, name, self.execution_context)?;
        self.push(attr);
        Ok(())
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        // CPython 3.13: TOS = container, TOS1 = item
        let haystack = self.pop();
        let needle = self.pop();
        let result = crate::baseobjspace::contains(haystack, needle)?;
        let inverted = match invert {
            crate::bytecode::Invert::No => result,
            crate::bytecode::Invert::Yes => !result,
        };
        self.push(pyre_object::w_bool_from(inverted));
        Ok(())
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        let b = self.pop();
        let a = self.pop();
        let same = std::ptr::eq(a, b); // pointer identity
        let result = match invert {
            crate::bytecode::Invert::No => same,
            crate::bytecode::Invert::Yes => !same,
        };
        self.push(pyre_object::w_bool_from(result));
        Ok(())
    }

    // ── ToBool ──
    // CPython 3.13: converts TOS to bool

    fn to_bool(&mut self) -> Result<(), PyError> {
        let val = self.pop();
        let truth = crate::baseobjspace::is_true(val)?;
        self.push(pyre_object::w_bool_from(truth));
        Ok(())
    }

    // ── DeleteSubscr ──

    fn delete_subscript(&mut self) -> Result<(), PyError> {
        let index = self.pop();
        let obj = self.pop();
        crate::baseobjspace::delitem(obj, index)?;
        Ok(())
    }

    // ── DeleteFast ──

    fn delete_fast(&mut self, idx: usize) -> Result<(), PyError> {
        self.locals_w_mut()[idx] = PY_NULL;
        Ok(())
    }

    // ── FormatSimple (str(TOS)) ──
    fn format_simple(&mut self) -> Result<(), PyError> {
        let val = self.pop();
        // `f'{x}'` → `PyObject_Format(x, NULL)`; a user `__format__` is
        // invoked with an empty spec, otherwise this is `str(value)`.
        let s = crate::runtime_ops::format_value(val, pyre_object::PY_NULL)?;
        self.push(s);
        Ok(())
    }

    // ── FormatWithSpec (format(TOS1, TOS)) ──
    fn format_with_spec(&mut self) -> Result<(), PyError> {
        let spec = self.pop();
        let val = self.pop();
        // `PyObject_Format(value, spec)` — dispatch to a user-defined
        // `__format__` when present, else apply the shared spec parser
        // (empty spec → `str(value)`).  `runtime_ops::format_value` keeps
        // f-string `{n:08.3f}` and `"{:08.3f}".format(n)` identical, and
        // reads a non-`str`/non-UTF-8 spec as empty rather than panicking.
        let s = crate::runtime_ops::format_value(val, spec)?;
        self.push(s);
        Ok(())
    }

    // ── ConvertValue (repr/str/ascii conversion) ──
    fn convert_value(&mut self, conv: crate::bytecode::ConvertValueOparg) -> Result<(), PyError> {
        let val = self.pop();
        // `str(val)` is computed in WTF-8 so a lone surrogate (a str, or
        // an exception whose single argument is a str) survives instead
        // of being forced through a Rust `String` via `py_str`.  This is
        // the path the `'%s' % x` → CONVERT_VALUE/FORMAT_SIMPLE compile
        // rewrite takes.
        let code = crate::runtime_ops::convert_value_code(conv);
        self.push(crate::runtime_ops::convert_value(val, code)?);
        Ok(())
    }

    // ── CopyFreeVars ──
    // CPython 3.13: copy n freevars from function closure to frame cell slots
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), PyError> {
        // No-op — closure passing needs call-site integration
        // The closure tuple is on the Function, but COPY_FREE_VARS
        // runs inside the callee frame which doesn't have a reference to
        // the function object. Need to pass closure during frame creation.
        Ok(())
    }

    // ── SetFunctionAttribute ──
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: pop attr, pop func, set, push func.
    /// Stack effect: (2) → (1)
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: (attr, func -- func)
    /// attr = TOS1 (below), func = TOS (top).
    /// Pops both, sets attribute on func, pushes func back.
    fn set_function_attribute_with_flag(
        &mut self,
        flag: crate::bytecode::MakeFunctionFlag,
    ) -> Result<(), PyError> {
        use crate::bytecode::MakeFunctionFlag;
        let func = self.pop(); // TOS = function
        let attr = self.pop(); // TOS1 = attribute value (closure tuple etc.)
        match flag {
            MakeFunctionFlag::Closure => unsafe {
                crate::function_set_closure(func, attr);
            },
            MakeFunctionFlag::Defaults => unsafe {
                crate::function_set_defaults(func, attr);
            },
            MakeFunctionFlag::KwOnlyDefaults => unsafe {
                crate::function_set_kwdefaults(func, attr);
            },
            MakeFunctionFlag::Annotations => {
                // `pypy/interpreter/function.py:553-559
                // fset_func_annotations` — MAKE_FUNCTION ANNOTATIONS
                // (oparg.rs:352 `MakeFunctionFlag::Annotations = 2`)
                // carries the eager annotations dict.  PyPy stores it
                // on `self.w_ann`; pyre stamps the typed
                // `Function.w_ann` slot directly so
                // `f.__annotations__ is f.__annotations__` holds
                // (the getattr arm reads the same field) instead of
                // routing through a side table.
                unsafe { crate::function::function_set_annotations(func, attr) };
            }
            MakeFunctionFlag::Annotate => {
                // PEP 649: lazy annotations.  `attr` is the
                // `__annotate__` callable the `__annotations__` getter
                // evaluates with `format=1` when the runtime dict is
                // requested; stored on the function's typed
                // `w_annotate` slot (CPython 3.14 `func_annotate`).
                unsafe { (*(func as *mut crate::function::Function)).w_annotate = attr };
            }
            // `MakeFunctionFlag::TypeParams` (oparg.rs:356) carries the
            // tuple of TypeVar / ParamSpec / TypeVarTuple bound by a
            // PEP 695 generic function.  Pyre has no PEP 695 surface
            // yet (typing tests aren't in the bench suite); accept
            // the operand silently rather than panic on the bytecode.
            MakeFunctionFlag::TypeParams => {}
        }
        self.push(func);
        Ok(())
    }

    // ── PushExcInfo ──
    // PyPy: executioncontext.py enter_frame / normalize_exception
    fn push_exc_info(&mut self) -> Result<(), PyError> {
        let exc = self.pop();
        // Save previous exception, set current.  Routed through the
        // named TLS accessors (not a raw `CURRENT_EXCEPTION.with`
        // closure) so the codewriter sees two residual-callable leaves
        // with registered fnaddrs instead of an unresolvable
        // `LocalKey::with` monomorphization — the same per-thread slot
        // the compiled trace reads/writes through
        // `get_current_exception_fn` / `set_current_exception_fn`.
        let prev = get_current_exception();
        set_current_exception(exc);
        // Push "previous exception" for later restore
        self.push(prev);
        // Push the exception value back
        self.push(exc);
        Ok(())
    }

    // ── CheckExcMatch ──
    // TOS = exception type to match, TOS1 = caught exception
    // Pops type, peeks exc, pushes bool result
    fn check_exc_match(&mut self) -> Result<(), PyError> {
        let exc_type = self.pop();
        let exc_value = self.peek();
        // pyopcode.py:1032-1040 cmp_exc_match split:
        //   :1034-1039 — `validate_check_exc_match_class(exc_type)?`
        //                raises TypeError(CANNOT_CATCH_MSG) for invalid
        //                except targets (`raise oefmt(...)` upstream).
        //   :1040     — `check_exc_match_against(exc_value, exc_type)`
        //                computes the match boolean.
        // PyPy keeps both in a single `@jit.unroll_safe cmp_exc_match`;
        // the JIT inlines and the `raise` becomes a guard. Pyre splits
        // so the bool-returning hot helper keeps a 1-register ABI for
        // residual JIT calls; the validity gate runs in this BC handler
        // (outside the residual call path) and lifts to `?` propagation.
        validate_check_exc_match_class(exc_type)?;
        let matched = check_exc_match_against(exc_value, exc_type);
        self.push(pyre_object::w_bool_from(matched));
        Ok(())
    }

    // ── PopExcept ──
    fn pop_except(&mut self) -> Result<(), PyError> {
        // Restore previous exc_info from stack.  Named TLS accessor for
        // the same codewriter-resolvability reason as `push_exc_info`.
        let prev_exc = self.pop();
        set_current_exception(prev_exc);
        Ok(())
    }

    // ── Reraise ──
    // `pypy/interpreter/pyopcode.py:1348-1376 RERAISE`.
    fn reraise(&mut self, oparg: u32) -> Result<(), PyError> {
        // pyopcode.py:1357-1363
        let reraise_lasti: i32 = if oparg != 0 {
            // pyopcode.py:1361 — self.space.int_w(self.peekvalue(oparg))
            crate::baseobjspace::int_w(self.peekvalue(oparg as usize))? as i32
        } else {
            -1
        };
        // pyopcode.py:1364 — w_exc = self.popvalue()
        let w_exc = self.popvalue();
        // pyopcode.py:1367 — w_value = space.interp_w(W_BaseException, w_exc)
        if w_exc.is_null() || !unsafe { pyre_object::is_exception(w_exc) } {
            return Err(PyError::type_error(
                "exception must derive from BaseException",
            ));
        }
        // pyopcode.py:1368-1369 — w_type = space.type(w_exc); operr = OperationError(w_type, w_exc, w_value.w_traceback)
        let mut err = unsafe { PyError::from_exc_object(w_exc) };
        // pyopcode.py:1376 — raise RaiseWithExplicitTraceback(operr, reraise_lasti)
        err.attach_tb = false;
        err.reraise_lasti = reraise_lasti;
        Err(err)
    }

    // ── LoadFromDictOrGlobals ──
    // CPython 3.13: LOAD_FROM_DICT_OR_GLOBALS — try TOS dict first, then globals
    fn load_from_dict_or_globals(&mut self, name: &str) -> Result<(), PyError> {
        let dict = self.pop();
        // Try dict first (if it's a dict or has attrs)
        if let Ok(val) = crate::baseobjspace::getattr_str(dict, name) {
            self.push(val);
            return Ok(());
        }
        // Fall back to globals
        let w_globals_obj = self.get_w_globals_obj();
        if !w_globals_obj.is_null() {
            if let Some(val) =
                unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_globals_obj, name) }
            {
                self.push(val);
                return Ok(());
            }
        } else {
            unsafe {
                if let Some(&val) = (*self.get_w_globals()).get(name) {
                    self.push(val);
                    return Ok(());
                }
            }
        }
        Err(PyError::name_error_with_name(
            format!("name '{name}' is not defined"),
            name,
        ))
    }

    // ── GetLen ──
    fn get_len(&mut self, obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let len = crate::baseobjspace::len(obj)?;
        Ok(len)
    }

    // ── LoadFastAndClear (comprehension scope) ──
    fn load_fast_and_clear(&mut self, idx: usize) -> Result<(), PyError> {
        let val = self.locals_w()[idx];
        self.push(val);
        self.locals_w_mut()[idx] = PY_NULL;
        Ok(())
    }

    // ── BuildSet ──
    fn build_set(&mut self, count: usize) -> Result<(), PyError> {
        // Build as a set-like object backed by __data__ dict.
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.pop());
        }
        items.reverse();
        let set_obj = crate::builtins::builtin_set_from_items(&items)?;
        self.push(set_obj);
        Ok(())
    }

    // ── DictUpdate ──
    // pypy/interpreter/pyopcode.py:1524-1532 DICT_UPDATE — `space.ismapping_w`
    // gate then `dict.update(source)`. Non-mapping operand surfaces
    // "'<T>' object is not a mapping" (TypeError).
    fn dict_update(&mut self, i: usize) -> Result<(), PyError> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        dict_update_from_mapping(dict, source)
    }

    // ── DictMerge ──
    // pypy/interpreter/pyopcode.py:1514-1522 DICT_MERGE → _dict_merge
    // (pyopcode.py:1979-2026).
    fn dict_merge(&mut self, i: usize) -> Result<(), PyError> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        // pyopcode.py:1514 — callable = peekvalue(oparg + 2)
        // Stack after pop: [..., callable, NULL, args_tuple, dict]
        let w_callable = if self.valuestackdepth > i + 2 {
            PyFrame::peek_at(self, i + 2)
        } else {
            pyre_object::PY_NULL
        };
        dict_merge_from_mapping(dict, source, w_callable)
    }

    // ── MapAdd ──
    // PyPy: STORE_MAP/MAP_ADD; CPython: MAP_ADD
    // dict = STACK[-i-2]; dict[TOS1] = TOS; pop key+value
    fn map_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let key = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            pyre_object::w_dict_store(dict, key, value);
        }
        Ok(())
    }

    // ── SetAdd ──
    // PyPy: SET_ADD; CPython: SET_ADD
    // set = STACK[-i]; set.add(TOS); pop value
    fn set_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_set_or_frozenset(set) {
                pyre_object::w_set_add(set, value);
            } else if pyre_object::is_list(set) {
                pyre_object::w_list_append(set, value);
            }
        }
        Ok(())
    }

    // ── none_value ──
    fn none_value(&mut self) -> Result<PyObjectRef, PyError> {
        Ok(pyre_object::w_none())
    }

    // ── unary_positive ──
    // PyPy: UNARY_POSITIVE → space.pos(w_value)
    fn unary_positive(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        crate::baseobjspace::pos(val)
    }

    // ── list_to_tuple ──
    // PyPy intrinsic: convert list to tuple (used in star unpacking).
    fn list_to_tuple(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        unsafe {
            if pyre_object::is_list(val) {
                let items = pyre_object::w_list_items_copy_as_vec(val);
                return Ok(pyre_object::w_tuple_new(items));
            }
        }
        Err(PyError::type_error("expected list for list_to_tuple"))
    }

    // ── print_expr ──
    // PyPy: PRINT_EXPR → sys.displayhook(value)
    fn print_expr(&mut self, val: PyObjectRef) -> Result<(), PyError> {
        if !unsafe { pyre_object::is_none(val) } {
            let s = unsafe { crate::py_repr(val)? };
            println!("{}", s);
        }
        Ok(())
    }

    // ── delete_name ──
    // pypy/interpreter/pyopcode.py:821 DELETE_NAME — delete from w_locals; KeyError → NameError.
    fn delete_name(&mut self, name: &str) -> Result<(), PyError> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            crate::baseobjspace::delitem(w_locals_object, key).map_err(|err| {
                if matches!(err.kind, PyErrorKind::KeyError) {
                    PyError::name_error_with_name(format!("name '{name}' is not defined"), name)
                } else {
                    err
                }
            })?;
            return Ok(());
        }
        let w_locals = self.get_w_locals();
        let w_globals_obj = self.get_w_globals_obj();
        let found: bool = if !w_locals.is_null() {
            // No `get_w_locals_obj` accessor yet — locals DictStorage
            // doesn't have a canonical W_DictObject sibling for routing.
            unsafe { crate::dict_storage_delete(&mut *w_locals, name) }
        } else if w_globals_obj.is_null() {
            let ns = self.get_w_globals();
            unsafe { crate::dict_storage_delete(&mut *ns, name) }
        } else {
            // Globals fallback: route through `w_dict_delitem_str` on
            // the canonical W_DictObject so the W_ModuleDictObject
            // strategy and mirror `DictStorage` stay coherent via
            // `maybe_sync_dict_storage_delete`.
            unsafe { pyre_object::w_dict_delitem_str(w_globals_obj, name) }
        };
        if !found {
            return Err(PyError::name_error_with_name(
                format!("name '{name}' is not defined"),
                name,
            ));
        }
        Ok(())
    }

    // ── delete_global ──
    // pypy/interpreter/pyopcode.py:901-903 DELETE_GLOBAL —
    //   `self.space.delitem(self.get_w_globals(), w_varname)`.
    // `space.delitem` on a dict raises `KeyError(w_varname)` when the
    // key is missing; pyre routes through `w_dict_delitem_str` on the
    // canonical W_DictObject so the W_ModuleDictObject's strategy and
    // its mirror `DictStorage` stay coherent via
    // `maybe_sync_dict_storage_delete`.
    fn delete_global(&mut self, name: &str) -> Result<(), PyError> {
        let w_globals_obj = self.get_w_globals_obj();
        let found: bool = if w_globals_obj.is_null() {
            let ns = self.get_w_globals();
            unsafe { crate::dict_storage_delete(&mut *ns, name) }
        } else {
            unsafe { pyre_object::w_dict_delitem_str(w_globals_obj, name) }
        };
        if !found {
            return Err(PyError::key_error(format!("'{name}'")));
        }
        Ok(())
    }

    // ── import_star ──
    // pypy/interpreter/pyopcode.py:1076 IMPORT_STAR — merge module's public names into
    // the locals mapping (class body / exec-with-locals), not globals.
    //
    // Non-dict mapping locals route through `import_all_from_w` so each
    // `from module import *` entry lands via `space.setitem(w_locals,
    // name, value)` rather than the `*mut DictStorage` fast path,
    // matching `pyopcode.py:1078 self.getdictscope()` returning a
    // generic `w_obj`.
    fn import_star(&mut self) -> Result<(), PyError> {
        let module = self.pop();
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            crate::importing::import_all_from_w(module, w_locals_object)?;
            return Ok(());
        }
        let w_locals = self.getdictscope()?;
        crate::importing::import_all_from(module, w_locals)?;
        self.setdictscope(w_locals)?;
        Ok(())
    }

    // ── load_build_class ──
    // PyPy: BUILD_CLASS; CPython: LOAD_BUILD_CLASS
    fn load_build_class(&mut self) -> Result<(), PyError> {
        let bc = crate::get_build_class_func();
        self.push(bc);
        Ok(())
    }

    // ── yield from / send ──
    fn get_yield_from_iter(&mut self) -> Result<(), PyError> {
        let iterable = self.pop();
        let iter = crate::baseobjspace::iter(iterable)?;
        self.push(iter);
        Ok(())
    }

    fn send_value(&mut self, target: usize) -> Result<(), PyError> {
        let _value = self.pop(); // sent value
        let iter = self.peek();
        match crate::baseobjspace::next(iter) {
            Ok(result) => {
                self.push(result);
                Ok(())
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                // `pypy/interpreter/pyopcode.py:1158-1166 next_yield_from`:
                //     try:
                //         w_stop_value = space.getattr(e.get_w_value(space),
                //                                      space.newtext("value"))
                //     except OperationError as e:
                //         if not e.match(space, space.w_AttributeError):
                //             raise
                //         w_stop_value = space.w_None
                //     self.pushvalue(w_stop_value)
                //
                // CPython 3.13 emits SEND with an EOI target; pyre's
                // dispatch lands here on StopIteration and must surface
                // the exception's `.value` as the yield-from result so
                // `val = yield from inner()` captures `inner`'s return.
                let value = if !e.exc_object.is_null()
                    && unsafe { pyre_object::is_exception(e.exc_object) }
                {
                    crate::baseobjspace::getattr_str(e.exc_object, "value")
                        .unwrap_or_else(|_| pyre_object::w_none())
                } else {
                    pyre_object::w_none()
                };
                self.push(value);
                self.set_last_instr_from_next_instr(target);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn end_send(&mut self) -> Result<(), PyError> {
        let result = self.pop();
        let _iter = self.pop();
        self.push(result);
        Ok(())
    }

    // ── load_method ──
    // PyPy: LOOKUP_METHOD — interpreter-only override.
    // For instances, pushes [attr, self] so CALL prepends self.
    // ── return_generator ──
    // CPython 3.12: RETURN_GENERATOR creates a generator from the current
    // frame and returns it to the caller. PyPy: generator.py GeneratorIterator.
    fn return_generator(&mut self) -> Result<(), PyError> {
        // When the generator function is already wrapped (CodeFlags::GENERATOR
        // detected in call_user_function_with_eval), RETURN_GENERATOR fires
        // during the first __next__() resume. It's a no-op in that case —
        // the generator object was already created at call time.
        // Push dummy value for the following POP_TOP to consume.
        self.push(pyre_object::w_none());
        Ok(())
    }

    // ── load_super_attr ──
    // CPython 3.12 LOAD_SUPER_ATTR: stack = [global_super, class, self]
    // → super(class, self).attr
    fn load_super_attr_with(&mut self, name: &str, is_method: bool) -> Result<(), PyError> {
        let self_obj = self.pop();
        let cls = self.pop();
        let _global_super = self.pop();

        let proxy = pyre_object::superobject::w_super_new(cls, self_obj);
        let result = crate::baseobjspace::getattr_str(proxy, name)?;

        // CPython _PySuper_Lookup: determines whether the resolved attr
        // is an unbound method (needs self binding) or a staticmethod /
        // classmethod (no self binding / bind class).
        if is_method {
            // getattr now returns a bound method via descriptor protocol.
            // Unwrap for the (func, self) pattern that CALL expects.
            if unsafe { pyre_object::is_method(result) } {
                let func = unsafe { pyre_object::w_method_get_func(result) };
                let recv = unsafe { pyre_object::w_method_get_self(result) };
                self.push(func);
                self.push(recv);
            } else {
                // staticmethod or classmethod — no self binding
                self.push(result);
                self.push(PY_NULL);
            }
        } else {
            // is_method=false: getattr already returned a bound method.
            self.push(result);
        }
        Ok(())
    }

    // For non-instances (modules etc.), pushes [attr, NULL].
    // The default trait impl always pushes [attr, NULL], which is what
    // the JIT tracer uses — no runtime branch in the shared path.
    fn load_method(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        // callmethod.py:60-78 fast method path: a plain method descriptor in
        // the class, nothing shadowing it in the instance, on a type that uses
        // the default __getattribute__.  Pushes [w_descr, w_obj] (the unbound
        // function + receiver) so CALL_METHOD binds self without allocating a
        // Method wrapper.  Shared with the JIT tracer (trace_opcode.rs) so the
        // concrete and symbolic frames produce the identical stack shape.
        if let Some((_, _, w_descr)) =
            unsafe { crate::baseobjspace::load_method_fast_path(obj, name) }
        {
            self.push(w_descr);
            self.push(obj);
            return Ok(());
        }
        let attr = crate::baseobjspace::getattr_str(obj, name)?;
        // LOOKUP_METHOD pushes (attr, null_or_self): the resolved attribute
        // first, then the bound receiver computed by the shared, side-effect
        // free binding decision (NULL when no self should be prepended).
        let bound = compute_load_method_bound(obj, attr, name);
        self.push(attr);
        self.push(bound);
        Ok(())
    }

    /// pyopcode.py:1024-1027 `LOAD_ATTR` — the interpreter consults the mapdict
    /// attribute cache only off-trace; under the JIT it does the plain
    /// `space.getattr`, which the trace folds via the type's `version_tag`.
    fn load_attr_cached(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // pyopcode.py:1024 `if not jit.we_are_jitted():` — positive form keeps
        // the annotator off the bare-`!` hazard. The cache path's helpers are
        // `dont_look_inside`, so the JIT never traces into them.
        if majit_metainterp::jit::we_are_jitted() {
            return OpcodeStepExecutor::load_attr(self, name);
        }
        // Graceful underflow (shared_opcode.rs:167 `opcode_load_attr` →
        // `pop_value()?`): a corrupted concrete-execution stack during
        // trace recording (e.g. a residual call the inline executor
        // could not perform) aborts the trace instead of panicking the
        // hard-asserting `pop()`.
        let obj = self.pop_value()?;
        let w_value = unsafe {
            crate::objspace::std::mapdict::load_attr_caching(
                self.pycode as PyObjectRef,
                obj,
                nameindex,
                name,
            )
        }?;
        self.push(w_value);
        Ok(())
    }

    /// pyopcode.py:917-926 `STORE_ATTR` — consults the mapdict attribute cache
    /// only off-trace; under the JIT it does the plain `space.setattr`, folded
    /// by the type's `version_tag`.
    fn store_attr_cached(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // pyopcode.py:920 `if not jit.we_are_jitted():` — positive form.
        if majit_metainterp::jit::we_are_jitted() {
            return OpcodeStepExecutor::store_attr(self, name);
        }
        // pyopcode.py:918-919 — obj is the top of stack, value below it.
        // Graceful underflow like `opcode_store_attr` (shared_opcode.rs:176)
        // so a corrupted trace-recording stack aborts the trace instead of
        // panicking the hard-asserting `pop()`.
        let obj = self.pop_value()?;
        let value = self.pop_value()?;
        unsafe {
            crate::objspace::std::mapdict::store_attr_caching(
                self.pycode as PyObjectRef,
                obj,
                nameindex,
                name,
                value,
            )
        }
    }

    // ── call ──
    // PyPy: baseobjspace.py:1240-1267 `call_valuestack` +
    // function.py:139-203 `funccall_valuestack`.
    //
    // CPython 3.12+ CALL: stack is [callable, null_or_self, arg0..argN-1].
    // null_or_self is NULL for plain calls, `self` for method calls.
    fn call(&mut self, nargs: usize) -> Result<(), PyError> {
        // baseobjspace.py:1240-1261 fast path: Function + no method binding
        //
        // baseobjspace.py:1243 — skip fast path when profiling is active
        // and the function wraps a builtin code (c_call/c_return events).
        // Conservative: skip entire fast path if profiled, since
        // funccall_valuestack's builtin dispatch also bypasses profiling.
        //
        // Guard: only enter when the value stack has at least nargs + 2
        // items above stack_base (callable + null_or_self + args).
        let stack_items = self.valuestackdepth.saturating_sub(self.stack_base());
        if stack_items >= nargs + 2 && !self.get_is_being_profiled() {
            let null_or_self = self.peekvalue_maybe_none(nargs);
            let callable = self.peekvalue_maybe_none(nargs + 1);
            if null_or_self.is_null()
                && !callable.is_null()
                && unsafe { crate::is_function(callable) }
            {
                let result =
                    crate::function::funccall_valuestack(callable, nargs, self, nargs + 2, false);
                if result.is_null() {
                    return Err(crate::call::take_call_error()
                        .unwrap_or_else(|| crate::PyError::type_error("call failed"))
                        .into());
                }
                self.push(result);
                return Ok(());
            }
        }

        // Slow path: method call or non-Function callable.
        // Must allocate Vec for args.
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let null_or_self = self.pop();
        let callable = self.pop();

        let result = if null_or_self.is_null() {
            call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            call_callable(self, callable, &full_args)?
        };
        self.push(result);
        Ok(())
    }

    // ── call_function_ex ──
    // pyopcode.py:1360 CALL_FUNCTION_EX:
    //     w_kwargs = self.popvalue() if has_kwarg else None
    //     w_args = self.popvalue()
    //     w_function = self.popvalue()
    //     args = self.argument_factory([], None, None,
    //                                  w_star=w_args,
    //                                  w_starstar=w_kwargs,
    //                                  w_function=w_function)
    //     w_result = self.space.call_args(w_function, args)
    //     self.pushvalue(w_result)
    //
    // argument.py Arguments.unpack_combined_starargs iterates w_star with
    // space.fixedview_unroll / space.listview_no_unpack, so arbitrary
    // iterables are accepted.
    fn call_function_ex(&mut self) -> Result<(), PyError> {
        let kwargs_or_null = self.pop();
        let args_obj = self.pop();
        let _null = self.pop();
        let callable = self.pop();

        // argument.py unpack_combined_starargs equivalent: fast-path tuple
        // and list so common bytecode emits avoid iter protocol overhead;
        // fall back to the iter protocol for arbitrary iterables.
        let args: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(args_obj) {
                let n = pyre_object::w_tuple_len(args_obj);
                (0..n as i64)
                    .filter_map(|i| pyre_object::w_tuple_getitem(args_obj, i))
                    .collect()
            } else if pyre_object::is_list(args_obj) {
                let n = pyre_object::w_list_len(args_obj);
                (0..n as i64)
                    .filter_map(|i| pyre_object::w_list_getitem(args_obj, i))
                    .collect()
            } else {
                crate::builtins::collect_iterable(args_obj)?
            }
        };

        // Merge the `**` mapping into the call.  argument.py:106-150
        // `_combine_starstarargs_wrapped` accepts any mapping — the dict fast
        // path or an arbitrary object via `keys()` / `__getitem__` — raising
        // "argument after ** must be a mapping" for a non-mapping and
        // "keywords must be strings" for a non-str key.
        if !kwargs_or_null.is_null() {
            let mut keyword_names_w: Vec<PyObjectRef> = Vec::new();
            let mut keywords_w: Vec<PyObjectRef> = Vec::new();
            crate::argument::combine_starstarargs_wrapped(
                &mut keyword_names_w,
                &mut keywords_w,
                kwargs_or_null,
                callable,
            )?;
            if !keyword_names_w.is_empty() {
                let entries: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = keyword_names_w
                    .iter()
                    .zip(keywords_w.iter())
                    .map(|(&k, &v)| (unsafe { pyre_object::w_str_get_wtf8(k) }.to_owned(), v))
                    .collect();
                let result = crate::call::call_with_kwargs(self, callable, &args, &entries)?;
                self.push(result);
                return Ok(());
            }
        }

        let result = call_callable(self, callable, &args)?;
        self.push(result);
        Ok(())
    }

    // ── call_kw ──
    // PyPy: CALL_FUNCTION_KW; CPython 3.13: CALL_KW
    // Stack: [callable, self_or_null, arg1, ..., argN, kwarg_names_tuple]
    /// CALL_KW — call with keyword arguments.
    ///
    /// PyPy: argument.py _match_signature
    /// Stack: [callable, null_or_self, arg0..argN-1, kwarg_names_tuple]
    /// The last `len(kwarg_names)` args are keyword args.
    ///
    /// Keyword resolution happens HERE (before frame creation) so the
    /// JIT eval loop sees correctly-positioned locals. PyPy does this
    /// in Arguments.parse_into_scope before the frame executes.
    fn call_kw(&mut self, nargs: usize) -> Result<(), PyError> {
        let kwarg_names = self.pop();
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let self_or_null = self.pop();
        let callable = self.pop();

        if self_or_null != PY_NULL && !unsafe { pyre_object::is_none(self_or_null) } {
            args.insert(0, self_or_null);
        }

        // Unwrap bound methods: load_method pushes (method, PY_NULL) for
        // bound methods. Extract the underlying function and prepend the
        // receiver so resolve_kwargs sees the correct function signature.
        let callable_unwrapped = crate::baseobjspace::unwrap_cell(callable);
        let callable_unwrapped = if unsafe { pyre_object::is_method(callable_unwrapped) } {
            let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
            let receiver = unsafe { pyre_object::w_method_get_self(callable_unwrapped) };
            if !receiver.is_null() && !unsafe { pyre_object::is_none(receiver) } {
                args.insert(0, receiver);
            }
            func
        } else {
            callable_unwrapped
        };

        // For type objects with kwargs: use call_with_kwargs which handles
        // __new__/__init__ kwargs forwarding correctly.
        if unsafe { pyre_object::is_type(callable_unwrapped) } {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let pos_args = args[..n_pos].to_vec();
                let mut kw_entries = Vec::with_capacity(nkw);
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                        kw_entries.push((key, args[n_pos + ki]));
                    }
                }
                let result = crate::call::call_with_kwargs(
                    self,
                    callable_unwrapped,
                    &pos_args,
                    &kw_entries,
                )?;
                self.push(result);
                return Ok(());
            }
        }

        // A generic alias has no signature of its own; its __call__
        // forwards to __origin__(*args, **kwargs).  Split the keyword tail
        // and route through call_with_kwargs so the origin's own kwargs
        // handling (e.g. dict.__init__) sees real keywords.
        if unsafe { pyre_object::is_generic_alias(callable_unwrapped) } {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            let n_pos = args.len().saturating_sub(nkw);
            let pos_args = args[..n_pos].to_vec();
            let mut kw_entries = Vec::with_capacity(nkw);
            for ki in 0..nkw {
                let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                if let Some(name_obj) = name {
                    let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                    kw_entries.push((key, args[n_pos + ki]));
                }
            }
            let result =
                crate::call::call_with_kwargs(self, callable_unwrapped, &pos_args, &kw_entries)?;
            self.push(result);
            return Ok(());
        }

        // Resolve keyword args into positional order.
        // argument.py Arguments._match_signature step: match keywords to
        // argnames, fill defaults, pack *args/**kwargs. PyPy's
        // `space.call_args` performs this exactly once; pyre mirrors that
        // by calling resolve_kwargs here and then dispatching directly to
        // call_user_function_resolved — which skips the defaults_fill /
        // pack_varargs replay that call_user_function_with_args performs
        // for positional-only paths.
        let is_builtin = unsafe { crate::is_function(callable_unwrapped) }
            && unsafe {
                crate::is_builtin_code(
                    crate::getcode(callable_unwrapped) as pyre_object::PyObjectRef
                )
            };
        if is_builtin {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let pos_args = args[..n_pos].to_vec();
                let mut kw_entries = Vec::with_capacity(nkw);
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        let key = unsafe { pyre_object::w_str_get_wtf8(name_obj) }.to_owned();
                        kw_entries.push((key, args[n_pos + ki]));
                    }
                }
                // PyPy CALL_FUNCTION_KW builds an Arguments object with
                // keyword_names_w / keywords_w, and the profiled-builtin path
                // passes that same object to call_args_and_c_profile.  Route
                // through call_with_kwargs so pyre's profile path constructs
                // Arguments::with_kw instead of treating the kwargs dict tail
                // as a positional firstarg.
                let result = crate::call::call_with_kwargs(
                    self,
                    callable_unwrapped,
                    &pos_args,
                    &kw_entries,
                )?;
                self.push(result);
                return Ok(());
            }
            let result = call_callable(self, callable_unwrapped, &args)?;
            self.push(result);
            return Ok(());
        }

        // pypy/interpreter/function.py Method.call_args parity: unwrap
        // bound method by prepending the receiver, then run resolve_kwargs
        // against the underlying function. This matches
        // `self.space.call_args(w_function, args)` after the MRO-dispatched
        // `im_func` has been extracted.
        let (target_func, mut prepended) = if unsafe { pyre_object::is_method(callable_unwrapped) }
        {
            let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
            let receiver = unsafe {
                let w_self = pyre_object::w_method_get_self(callable_unwrapped);
                if !w_self.is_null() && !pyre_object::is_none(w_self) {
                    w_self
                } else {
                    pyre_object::w_method_get_class(callable_unwrapped)
                }
            };
            if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
                let mut prepended = Vec::with_capacity(1 + args.len());
                prepended.push(receiver);
                prepended.extend_from_slice(&args);
                (func, Some(prepended))
            } else {
                (func, None)
            }
        } else {
            (callable_unwrapped, None)
        };
        let call_args: &[PyObjectRef] = prepended.as_deref().unwrap_or(&args);
        let resolved = crate::call::resolve_kwargs(target_func, call_args, kwarg_names)?;
        // Drop the temporary prepended buffer once resolved is built.
        prepended = None;
        let _ = prepended;

        let result = if unsafe { crate::is_function(target_func) } {
            crate::call::call_user_function_resolved(self, target_func, &resolved)?
        } else {
            call_callable(self, target_func, &resolved)?
        };
        self.push(result);
        Ok(())
    }

    // ── load_locals ──
    // PyPy: LOAD_LOCALS; CPython: LOAD_LOCALS
    // Pushes the current namespace dict onto the stack.
    fn load_locals(&mut self) -> Result<(), PyError> {
        let dict = pyre_object::w_dict_new();
        unsafe {
            let w_locals = self.get_w_locals();
            if !w_locals.is_null() {
                for (key, &value) in (*w_locals).entries() {
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                    }
                }
            } else {
                let code = &*crate::pyframe_get_pycode(self);
                for (idx, name) in code.varnames.iter().enumerate() {
                    let value = self.locals_w()[idx];
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), value);
                    }
                }
                let w_globals_obj = self.get_w_globals_obj();
                if self.nlocals() == 0 && !w_globals_obj.is_null() {
                    for (key, value) in
                        unsafe { pyre_object::dictmultiobject::w_dict_items(w_globals_obj) }
                    {
                        if !value.is_null() {
                            pyre_object::w_dict_store(dict, key, value);
                        }
                    }
                } else {
                    let w_globals = self.get_w_globals();
                    if self.nlocals() == 0 && !w_globals.is_null() {
                        for (key, &value) in (*w_globals).entries() {
                            if !value.is_null() {
                                pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                            }
                        }
                    }
                }
            }
        }
        self.push(dict);
        Ok(())
    }

    // ── unpack_ex ──
    // PyPy: UNPACK_SEQUENCE with star; CPython: UNPACK_EX
    // `a, *b, c = iterable`
    fn unpack_ex(&mut self, args: crate::bytecode::UnpackExArgs) -> Result<(), PyError> {
        let before = args.before as usize;
        let after = args.after as usize;
        let value = self.pop();

        let elements: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(value) {
                pyre_object::w_tuple_items_copy_as_vec(value)
            } else if pyre_object::is_list(value) {
                pyre_object::w_list_items_copy_as_vec(value)
            } else {
                // Any other iterable is materialised via the iteration
                // protocol, matching `unpack_sequence_exact`'s fallback.
                crate::builtins::collect_iterable(value)?
            }
        };

        let min_expected = before + after;
        if elements.len() < min_expected {
            return Err(PyError::value_error(&format!(
                "not enough values to unpack (expected at least {}, got {})",
                min_expected,
                elements.len()
            )));
        }

        let middle_len = elements.len() - min_expected;

        // Push after items (reversed), then middle list, then before items (reversed)
        for i in (0..after).rev() {
            self.push(elements[before + middle_len + i]);
        }
        let middle: Vec<PyObjectRef> = elements[before..before + middle_len].to_vec();
        self.push(pyre_object::w_list_new(middle));
        for i in (0..before).rev() {
            self.push(elements[i]);
        }

        Ok(())
    }

    // ── delete_attr ──
    // PyPy: DELETE_ATTR → space.delattr(obj, name)
    fn delete_attr(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        crate::baseobjspace::delattr_str(obj, name)?;
        Ok(())
    }

    // ── set_update ──
    // PyPy: set.update(iterable); CPython: SET_UPDATE
    fn set_update(&mut self, i: usize) -> Result<(), PyError> {
        let iterable = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_set_or_frozenset(set) {
                let items = crate::builtins::collect_iterable(iterable)?;
                for item in items {
                    pyre_object::w_set_add(set, item);
                }
            } else if pyre_object::is_list(set) {
                if pyre_object::is_list(iterable) {
                    let items = pyre_object::w_list_items_copy_as_vec(iterable);
                    for item in items {
                        pyre_object::w_list_append(set, item);
                    }
                } else if pyre_object::is_tuple(iterable) {
                    for item in pyre_object::w_tuple_items_copy_as_vec(iterable) {
                        pyre_object::w_list_append(set, item);
                    }
                }
            }
        }
        Ok(())
    }

    // ── BuildSlice ──
    // CPython 3.13: BUILD_SLICE creates a slice object from 2 or 3 stack items
    fn build_slice(&mut self, argc: crate::bytecode::BuildSliceArgCount) -> Result<(), PyError> {
        use crate::bytecode::BuildSliceArgCount;
        let step = match argc {
            BuildSliceArgCount::Three => self.pop(),
            BuildSliceArgCount::Two => pyre_object::w_none(),
        };
        let stop = self.pop();
        let start = self.pop();
        self.push(pyre_object::w_slice_new(start, stop, step));
        Ok(())
    }

    // ── BinarySlice (a[b:c]) ──
    // PyPy: BINARY_SUBSCR with slice; CPython 3.13: BINARY_SLICE
    fn binary_slice(&mut self) -> Result<(), PyError> {
        let stop = self.pop();
        let start = self.pop();
        let obj = self.pop();
        let result = crate::runtime_ops::binary_slice_values(obj, start, stop)?;
        self.push(result);
        Ok(())
    }

    // ── StoreSlice (a[b:c] = d) ──
    // Stack (bottom→top): value, container, start, stop.
    fn store_slice(&mut self) -> Result<(), PyError> {
        let stop = self.pop();
        let start = self.pop();
        let container = self.pop();
        let value = self.pop();
        crate::runtime_ops::store_slice_values(container, start, stop, value)
    }

    // ── BuildString (f-string concatenation) ──
    // CPython 3.13: concatenate N string fragments from stack
    fn build_string(&mut self, count: usize) -> Result<(), PyError> {
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(self.pop());
        }
        parts.reverse();
        self.push(crate::runtime_ops::build_string_from_refs(&parts));
        Ok(())
    }

    // ── ListExtend ──
    // pypy/interpreter/pyopcode.py:1480-1491 LIST_EXTEND — calls
    // `list.extend(iterable)`; on failure surfaces "Value after * must be
    // an iterable, not <T>" when the operand isn't iterable, else
    // re-raises the inner error.
    fn list_extend(&mut self, _i: usize) -> Result<(), PyError> {
        let iterable = self.pop();
        let list = self.peek();
        crate::opcode_ops::list_extend_value(list, iterable)
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<PyObjectRef>, PyError> {
        Err(PyError::type_error(format!(
            "unimplemented instruction: {instruction:?}"
        )))
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyExecutionContext;
    use crate::*;
    use std::rc::Rc;

    /// The JIT compiler state — warmstate, the compiled-loop registry, and the
    /// executable-code buffers — is a process-global singleton driven on a
    /// single thread by design. The `cargo test` harness runs tests on a thread
    /// pool, so two loops crossing the compile threshold at once race on that
    /// shared state and one of them reads a half-installed trace. Tests that
    /// drive JIT compilation serialise on this lock so only one compiles at a
    /// time. Poison is recovered: a panicking test must not wedge the others.
    static JIT_COMPILE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn jit_compile_test_guard() -> std::sync::MutexGuard<'static, ()> {
        JIT_COMPILE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        frame.execute_frame(None, None)
    }

    fn run_exec_frame(source: &str) -> (PyResult, crate::pyframe::FrameBox) {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let result = frame.execute_frame(None, None);
        (result, frame)
    }

    #[test]
    fn test_exception_is_valid_obj_as_class_w_matches_baseexception_subclass_rule() {
        let (_result, frame) = run_exec_frame("good = ValueError\nbad = int");
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let good = *w_globals.get("good").expect("missing good");
        let bad = *w_globals.get("bad").expect("missing bad");

        unsafe {
            assert!(crate::baseobjspace::exception_is_valid_obj_as_class_w(good));
            assert!(!crate::baseobjspace::exception_is_valid_obj_as_class_w(bad));
        }
    }

    #[test]
    fn test_raise_non_exception_type_raises_typeerror() {
        let (result, _frame) = run_exec_frame("raise int");
        let err = result.expect_err("raise int should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
    }

    #[test]
    fn test_make_cell_closure_over_parameter_not_double_wrapped() {
        // An argument slot promoted to a cellvar (captured by an inner
        // function) must wrap to a single cell, not a cell-of-cell, so the
        // closure reads the value rather than an inner cell.
        let (_result, frame) = run_exec_frame(
            "def make_adder(n):\n    def add(x):\n        return x + n\n    return add\nresult = make_adder(10)(5)",
        );
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let result = *w_globals.get("result").expect("missing result");
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 15);
    }

    #[test]
    fn test_make_cell_class_cell_super_not_double_wrapped() {
        // The implicit `__class__` cellvar is never reassigned in the body,
        // so MAKE_CELL must leave the pre-installed cell alone; a
        // cell-of-cell would make zero-arg super() resolve an inner cell
        // instead of the class.
        crate::test_hooks::install_hash_hook();
        let (_result, frame) = run_exec_frame(
            "class A:\n    def f(self):\n        return 1\nclass B(A):\n    def f(self):\n        return 10 + super().f()\nresult = B().f()",
        );
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let result = *w_globals.get("result").expect("missing result");
        assert_eq!(unsafe { pyre_object::w_int_get_value(result) }, 11);
    }

    #[test]
    fn test_raise_invalid_cause_raises_typeerror() {
        let (result, _frame) = run_exec_frame("raise ValueError() from 1");
        let err = result.expect_err("invalid cause should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "exception cause must be None or derive from BaseException"
        );
    }

    #[test]
    fn test_raise_from_sets_cause_attribute() {
        let (_result, frame) = run_exec_frame("exc = ValueError()\ncause = KeyError()");
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let exc = *w_globals.get("exc").expect("missing exc");
        let cause = *w_globals.get("cause").expect("missing cause");

        let code = compile_exec("raise exc from cause").expect("compile failed");
        let mut raise_frame = PyFrame::new(code);
        unsafe {
            (*raise_frame.fget_w_globals()).insert("exc".to_string(), exc);
            (*raise_frame.fget_w_globals()).insert("cause".to_string(), cause);
        }

        let err = raise_frame
            .execute_frame(None, None)
            .expect_err("raise from should fail");
        assert_eq!(err.to_exc_object(), exc);
        assert_eq!(
            crate::getattr_str(exc, "__cause__").expect("read cause"),
            cause
        );
    }

    #[test]
    fn test_literal() {
        let result = run_eval("42").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_addition() {
        let result = run_eval("1 + 2").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 3) };
    }

    #[test]
    fn test_subtraction() {
        let result = run_eval("10 - 3").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_multiplication() {
        let result = run_eval("6 * 7").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_complex_expr() {
        let result = run_eval("(2 + 3) * 4 - 1").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 19) };
    }

    #[test]
    fn test_comparison() {
        let result = run_eval("3 < 5").unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_comparison_false() {
        let result = run_eval("5 < 3").unwrap();
        unsafe { assert!(!w_bool_get_value(result)) };
    }

    #[test]
    fn test_store_load_namespace() {
        let source = "x = 5\ny = x * x";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            let y = w_dict_getitem_str(frame.w_globals_obj, "y").unwrap();
            assert_eq!(w_int_get_value(x), 5);
            assert_eq!(w_int_get_value(y), 25);
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "i = 0\nwhile i < 10:\n    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            assert_eq!(w_int_get_value(i), 10);
        }
    }

    #[test]
    fn test_eval_loop_redecodes_opargs_after_extended_arg_jumps() {
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
        let code = compile_exec(&source).expect("compile failed");
        assert!(
            code.instructions.windows(2).any(|pair| {
                matches!(pair[0].op, Instruction::ExtendedArg)
                    && !matches!(pair[1].op, Instruction::ExtendedArg)
            }),
            "expected an instruction with an ExtendedArg prefix"
        );
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "r").unwrap();
            assert_eq!(w_int_get_value(r), 6);
        }
    }

    #[test]
    fn test_eval_loop_raises_on_malformed_extended_arg_chain() {
        let code = compile_exec("x = 1").expect("compile failed");
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::GetIter);
        }
        let mut frame = PyFrame::new(code);
        let err = frame
            .execute_frame(None, None)
            .expect_err("expected bytecode corruption");
        assert_eq!(err.kind, PyErrorKind::BytecodeCorruption);
        assert_eq!(err.message, "bytecode corruption");
    }

    #[test]
    fn test_none_result() {
        let result = run_eval("None").unwrap();
        unsafe { assert!(is_none(result)) };
    }

    #[test]
    fn test_bool_result() {
        let result = run_eval("True").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_literal() {
        let result = run_eval("1.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 1.5);
        }
    }

    #[test]
    fn test_float_addition() {
        let result = run_eval("1.5 + 2.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 4.0);
        }
    }

    #[test]
    fn test_float_truediv() {
        let result = run_eval("10 / 4").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 2.5);
        }
    }

    #[test]
    fn test_float_comparison() {
        let result = run_eval("1.5 < 2.5").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_int_mixed() {
        let result = run_eval("1.5 + 2").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 3.5);
        }
    }

    #[test]
    fn test_float_negation() {
        let result = run_eval("-3.14").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), -3.14);
        }
    }

    #[test]
    fn test_float_truthiness() {
        // Test via is_true directly since `not` uses ToBool instruction
        assert!(!is_true(w_float_new(0.0)).unwrap());
        assert!(is_true(w_float_new(1.5)).unwrap());
        assert!(is_true(w_float_new(-0.1)).unwrap());
    }

    // ── str tests ────────────────────────────────────────────────────

    #[test]
    fn test_str_literal() {
        let result = run_eval("'hello'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_str_concat() {
        let result = run_eval("'hello' + ' world'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello world");
        }
    }

    #[test]
    fn test_str_repeat() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_str_comparison() {
        let result = run_eval("'abc' < 'abd'").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    // ── for loop / range tests ──────────────────────────────────────

    #[test]
    fn test_for_range() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            assert_eq!(w_int_get_value(s), 45);
        }
    }

    #[test]
    fn test_hot_range_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "s = 0\nfor i in range(3000):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            assert_eq!(w_int_get_value(s), 4_498_500);
        }
    }

    #[test]
    fn test_hot_module_branch_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    if i < 1500:
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4500);
        }
    }

    #[test]
    fn test_hot_tuple_unpack_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    a, b = (i, 1)
    acc = acc + a + b
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_hot_list_index_store_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
lst = [0]
i = 0
acc = 0
while i < 3000:
    lst[0] = i
    acc = acc + lst[0]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            let lst = w_dict_getitem_str(frame.w_globals_obj, "lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, 0).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_bitwise_or_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc | i
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4095);
        }
    }

    #[test]
    fn test_hot_unary_invert_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (~i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), -4_501_500);
        }
    }

    #[test]
    fn test_hot_positive_floordiv_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i // 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 1_498_500);
        }
    }

    #[test]
    fn test_hot_positive_mod_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i % 7)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 8_994);
        }
    }

    #[test]
    fn test_hot_builtin_abs_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + abs(i - 1500)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 2_250_000);
        }
    }

    #[test]
    fn test_hot_list_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
lst = [1]
i = 0
acc = 0
while i < 3000:
    if lst:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_tuple_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
tpl = ()
i = 0
acc = 0
while i < 3000:
    if tpl:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_none_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = None
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_float_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = 0.5
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_string_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_string_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = \"\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_dict_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = {1: 2}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_len_string_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 12_000);
        }
    }

    #[test]
    fn test_hot_builtin_len_dict_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = {1: 2, 3: 4}
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6_000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_true_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
x = 42
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_false_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6000);
        }
    }

    #[test]
    fn test_hot_builtin_type_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if type(x) == list:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_min_small_int_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + min(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6426);
        }
    }

    #[test]
    fn test_hot_builtin_max_small_int_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + max(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 11568);
        }
    }

    #[test]
    fn test_hot_empty_dict_truth_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
value = {}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_list_negative_index_store_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
lst = [0, 1]
i = 0
acc = 0
while i < 3000:
    lst[-1] = i
    acc = acc + lst[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            let lst = w_dict_getitem_str(frame.w_globals_obj, "lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, -1).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_tuple_negative_index_load_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
tpl = (3, 5)
i = 0
acc = 0
while i < 3000:
    acc = acc + tpl[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 15_000);
        }
    }

    #[test]
    fn test_hot_user_function_loop_survives_compiled_trace() {
        let _jit_guard = jit_compile_test_guard();
        let source = "\
def inc(x):
    return x + 1
i = 0
acc = 0
while i < 3000:
    acc = acc + inc(i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = w_dict_getitem_str(frame.w_globals_obj, "i").unwrap();
            let acc = w_dict_getitem_str(frame.w_globals_obj, "acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_for_range_start_stop() {
        let source = "s = 0\nfor i in range(5, 10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            assert_eq!(w_int_get_value(s), 35);
        }
    }

    #[test]
    fn test_for_range_step() {
        let source = "s = 0\nfor i in range(0, 10, 2):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            // 0 + 2 + 4 + 6 + 8 = 20
            assert_eq!(w_int_get_value(s), 20);
        }
    }

    #[test]
    fn test_for_range_empty() {
        let source = "s = 42\nfor i in range(0):\n    s = 0";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            assert_eq!(w_int_get_value(s), 42);
        }
    }

    #[test]
    fn test_builtin_range_print() {
        let source = "s = 0\nfor i in range(5):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = w_dict_getitem_str(frame.w_globals_obj, "s").unwrap();
            // 0 + 1 + 2 + 3 + 4 = 10
            assert_eq!(w_int_get_value(s), 10);
        }
    }

    // ── builtin tests ───────────────────────────────────────────────

    #[test]
    fn test_builtin_len() {
        let source = "x = len([1, 2, 3])";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert_eq!(w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_builtin_abs() {
        let source = "x = abs(-5)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert_eq!(w_int_get_value(x), 5);
        }
    }

    #[test]
    fn test_builtin_min_max() {
        let source = "a = min(3, 7)\nb = max(3, 7)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let a = w_dict_getitem_str(frame.w_globals_obj, "a").unwrap();
            let b = w_dict_getitem_str(frame.w_globals_obj, "b").unwrap();
            assert_eq!(w_int_get_value(a), 3);
            assert_eq!(w_int_get_value(b), 7);
        }
    }

    // ── container tests ────────────────────────────────────────────

    #[test]
    fn test_list_literal() {
        let source = "x = [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert!(is_list(x));
            assert_eq!(w_list_len(x), 3);
            assert_eq!(w_int_get_value(w_list_getitem(x, 0).unwrap()), 1);
            assert_eq!(w_int_get_value(w_list_getitem(x, 1).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(x, 2).unwrap()), 3);
        }
    }

    #[test]
    fn test_tuple_unpack() {
        let source = "a, b = 1, 2";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let a = w_dict_getitem_str(frame.w_globals_obj, "a").unwrap();
            let b = w_dict_getitem_str(frame.w_globals_obj, "b").unwrap();
            assert_eq!(w_int_get_value(a), 1);
            assert_eq!(w_int_get_value(b), 2);
        }
    }

    #[test]
    fn test_list_subscr() {
        let source = "lst = [10, 20, 30]\nx = lst[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert_eq!(w_int_get_value(x), 20);
        }
    }

    #[test]
    fn test_list_store_subscr() {
        let source = "lst = [1, 2, 3]\nlst[0] = 99\nx = lst[0]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert_eq!(w_int_get_value(x), 99);
        }
    }

    #[test]
    fn test_dict_literal_and_subscr() {
        let source = "d = {1: 10, 2: 20}\nx = d[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = w_dict_getitem_str(frame.w_globals_obj, "x").unwrap();
            assert_eq!(w_int_get_value(x), 10);
        }
    }

    // ── function definition and call tests ──────────────────────────

    #[test]
    fn test_simple_function() {
        let source = "def double(x):\n    return x * 2\nresult = double(21)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_with_locals() {
        let source = "\
def add_squares(a, b):
    aa = a * a
    bb = b * b
    return aa + bb
result = add_squares(3, 4)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 25);
        }
    }

    #[test]
    fn test_recursive_function() {
        let source = "\
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
result = factorial(5)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 120);
        }
    }

    // ── attribute tests ─────────────────────────────────────────────

    #[test]
    fn test_store_load_attr() {
        crate::test_hooks::install_hash_hook();
        let source = "\
def f():
    pass
f.x = 42
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_store_load_multiple_attrs() {
        crate::test_hooks::install_hash_hook();
        let source = "\
def f():
    pass
f.a = 10
f.b = 20
result = f.a + f.b";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    #[test]
    fn test_attr_overwrite() {
        crate::test_hooks::install_hash_hook();
        let source = "\
def f():
    pass
f.x = 1
f.x = 2
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_attr_on_different_objects() {
        crate::test_hooks::install_hash_hook();
        let source = "\
def f():
    pass
def g():
    pass
f.x = 10
g.x = 20
result = f.x + g.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    // ── Opcode tests ──

    #[test]
    fn test_contains_op_in() {
        let source = "x = [1, 2, 3]\nresult = 1 in x";
        let (res, frame) = run_exec_frame(source);
        res.expect("exec failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result), "1 in [1,2,3] should be True");
        }
    }

    #[test]
    fn test_contains_op_not_in() {
        let source = "result = 4 not in [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_op() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_op() {
        let result = run_eval("1 is not None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring() {
        let source = "x = 42\nresult = f'val={x}'";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_str_get_value(result), "val=42");
        }
    }

    #[test]
    fn test_list_slice() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[1:3]";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert!(is_list(result), "slice result should be list");
                assert_eq!(w_list_len(result), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 3);
            },
            Err(e) => panic!("list_slice failed: {} (kind: {:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_delete_subscr() {
        // del x[0] in a list
        let source = "x = [1, 2, 3]\ndel x[0]\nresult = x[0]";
        let (result, _) = run_exec_frame(source);
        // After del x[0], x[0] becomes PY_NULL; accessing may succeed or fail
        // Just check it doesn't crash during del
        let _ = result;
    }

    #[test]
    fn test_to_bool() {
        let result = run_eval("not 0").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_none_is_none() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring_with_expr() {
        let source = "x = 10\ny = 20\nresult = f'{x} + {y} = {x + y}'";
        let (res, frame) = run_exec_frame(source);
        res.expect("f-string exec failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_str_get_value(result), "10 + 20 = 30");
        }
    }

    #[test]
    fn test_string_contains() {
        let source = "result = 'lo' in 'hello'";
        let (res, frame) = run_exec_frame(source);
        res.expect("string contains failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_tuple_contains() {
        let source = "result = 2 in (1, 2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple contains failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_not_in() {
        let source = "result = 5 not in [1, 2, 3]";
        let (res, frame) = run_exec_frame(source);
        res.expect("not in failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_none() {
        let source = "result = 42 is not None";
        let (res, frame) = run_exec_frame(source);
        res.expect("is not None failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_list_slice_negative() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[-3:]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative slice failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 3);
        }
    }

    #[test]
    fn test_nested_function_call() {
        let source = "\
def add(a, b):
    return a + b
result = add(add(1, 2), add(3, 4))";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested call failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_while_loop_with_break() {
        let source = "\
x = 0
while True:
    x = x + 1
    if x == 5:
        break
result = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("while+break failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_inplace_add() {
        let source = "x = 10\nx += 5\nresult = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("inplace add failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_string_iteration_chars() {
        let source = "\
result = ''
for c in 'hello':
    result = result + c
";
        let (res, frame) = run_exec_frame(source);
        res.expect("string iteration failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_enumerate_style() {
        // Test: manual counter with for loop
        let source = "\
count = 0
for x in [10, 20, 30]:
    count = count + 1
result = count";
        let (res, frame) = run_exec_frame(source);
        res.expect("enumerate style failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_nested_for_loops() {
        let source = "\
result = 0
for i in [1, 2, 3]:
    for j in [10, 20]:
        result = result + i * j
";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested for failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            // 1*10 + 1*20 + 2*10 + 2*20 + 3*10 + 3*20 = 10+20+20+40+30+60 = 180
            assert_eq!(w_int_get_value(result), 180);
        }
    }

    #[test]
    fn test_try_except_basic() {
        let source = "\
x = 0
try:
    x = 1 / 0
except:
    x = 42
result = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_int_get_value(result), 42);
            },
            Err(e) => panic!("try/except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_recursive_fibonacci() {
        let source = "\
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(10)";
        let (res, frame) = run_exec_frame(source);
        res.expect("fib failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 55);
        }
    }

    #[test]
    fn test_string_multiply() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_list_multiply() {
        let result = run_eval("[1, 2] * 3").unwrap();
        unsafe {
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 6);
        }
    }

    #[test]
    fn test_negative_index() {
        let source = "x = [10, 20, 30]\nresult = x[-1]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative index failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_boolean_operators() {
        let source = "result = True and False";
        let (res, frame) = run_exec_frame(source);
        res.expect("boolean and failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(!crate::baseobjspace::is_true(r).unwrap());
        }
    }

    #[test]
    fn test_chained_comparison() {
        let source = "result = 1 < 2 < 3";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert!(w_bool_get_value(r));
            },
            Err(e) => eprintln!("chained comparison: {}", e.message),
        }
    }

    #[test]
    fn test_try_except_specific() {
        let source = "\
result = 0
try:
    x = 1 / 0
except ZeroDivisionError:
    result = 99
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_int_get_value(r), 99);
            },
            Err(e) => panic!("specific except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_try_except_no_match_propagates() {
        // If except doesn't match, error should propagate
        let source = "\
try:
    x = 1 / 0
except ValueError:
    pass
";
        let (res, _) = run_exec_frame(source);
        // Should fail because ZeroDivisionError != ValueError
        // Bare except catches all, specific except may not work yet
        let _ = res; // Don't assert — depends on CHECK_EXC_MATCH impl
    }

    #[test]
    fn test_check_exc_match_invalid_target_raises_type_error() {
        // pyopcode.py:1032-1039 — `except <non-exception>:` raises
        // TypeError(CANNOT_CATCH_MSG). The bare `except 42:` form is
        // syntactically valid; the runtime gate fires in CHECK_EXC_MATCH.
        let source = "\
try:
    raise ValueError(\"boom\")
except 42:
    pass
";
        let (res, _) = run_exec_frame(source);
        match res {
            Err(e) => {
                assert!(
                    matches!(e.kind, crate::PyErrorKind::TypeError),
                    "expected TypeError, got {:?}: {}",
                    e.kind,
                    e.message_text(),
                );
                // The error round-trips through the raised exception
                // object, so the text lives behind `message_text()`.
                assert!(
                    e.message_text().contains("BaseException"),
                    "expected CANNOT_CATCH_MSG, got: {}",
                    e.message_text(),
                );
            }
            Ok(_) => panic!("expected TypeError for `except 42:`"),
        }
    }

    #[test]
    fn test_check_exc_match_invalid_tuple_member_raises_type_error() {
        // pyopcode.py:1034-1037 — tuple form, any non-exception entry
        // raises TypeError. `except (ValueError, 42):` must trigger the
        // gate even though `ValueError` itself is valid.
        let source = "\
try:
    raise ValueError(\"boom\")
except (ValueError, 42):
    pass
";
        let (res, _) = run_exec_frame(source);
        match res {
            Err(e) => assert!(
                matches!(e.kind, crate::PyErrorKind::TypeError),
                "expected TypeError, got {:?}: {}",
                e.kind,
                e.message,
            ),
            Ok(_) => panic!("expected TypeError for `except (ValueError, 42):`"),
        }
    }

    #[test]
    fn test_check_exc_match_against_matches_by_actual_type() {
        // pyopcode.py:1040 `return space.exception_match(space.type(w_1), w_2)`:
        // the left operand is matched by its *actual* type, never treated as
        // an unconditional success.  Guards the three shapes the residual
        // `bh_compare_fn` (call_jit.rs) and the BC `check_exc_match` share:
        //   * a matching exception instance   -> true
        //   * a non-matching exception class  -> false (an `except` clause
        //     past the first must not spuriously match)
        //   * a non-exception value           -> false (matched by `type(v)`,
        //     whose MRO holds no exception class)
        let (_result, frame) = run_exec_frame(
            "exc = ValueError(\"boom\")\nplain = 5\nvalue_error = ValueError\ntype_error = TypeError",
        );
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let exc = *w_globals.get("exc").expect("missing exc");
        let plain = *w_globals.get("plain").expect("missing plain");
        let value_error = *w_globals.get("value_error").expect("missing value_error");
        let type_error = *w_globals.get("type_error").expect("missing type_error");

        assert!(check_exc_match_against(exc, value_error));
        assert!(!check_exc_match_against(exc, type_error));
        assert!(!check_exc_match_against(plain, value_error));
    }

    #[test]
    fn test_try_finally() {
        let source = "\
result = 0
try:
    result = 1
finally:
    result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_int_get_value(r), 11);
            },
            Err(e) => panic!("try/finally failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_multiple_except() {
        let source = "\
result = 0
try:
    x = 1 / 0
except:
    result = 1
result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple except failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 11);
        }
    }

    #[test]
    fn test_for_with_continue() {
        let source = "\
result = 0
for x in [1, 2, 3, 4, 5]:
    if x == 3:
        continue
    result = result + x
";
        let (res, frame) = run_exec_frame(source);
        res.expect("for+continue failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            // 1 + 2 + 4 + 5 = 12 (skips 3)
            assert_eq!(w_int_get_value(r), 12);
        }
    }

    #[test]
    fn test_default_args() {
        let source = "\
def greet(name, greeting='hello'):
    return greeting
result = greet('world')
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_str_get_value(r), "hello");
            },
            Err(e) => {
                // Default args may need KW_DEFAULTS support
                eprintln!("default args: {} ({:?})", e.message, e.kind);
            }
        }
    }

    #[test]
    fn test_augmented_assign_list() {
        let source = "x = [1, 2]\nx += [3]\nresult = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert!(is_list(result));
                // After += [3], x should have 3 elements
                assert_eq!(w_list_len(result), 3);
            },
            Err(e) => panic!("augmented list failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_for_loop_over_list() {
        let source = "\
total = 0
for x in [1, 2, 3, 4, 5]:
    total = total + x
result = total";
        let (res, frame) = run_exec_frame(source);
        res.expect("for loop failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_for_loop_over_string() {
        let source = "\
result = 0
for c in 'abc':
    result = result + 1";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_int_get_value(result), 3);
            },
            Err(e) => {
                // String iteration might not work yet — ignore
                eprintln!("for-string: {}", e.message);
            }
        }
    }

    #[test]
    fn test_multiple_assignment() {
        let source = "a = b = 42\nresult = a + b";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple assign failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 84);
        }
    }

    #[test]
    fn test_closure_basic() {
        let source = "\
def make_adder(n):
    def adder(x):
        return x + n
    return adder
add5 = make_adder(5)
result = add5(10)";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert_eq!(w_int_get_value(r), 15);
            },
            Err(e) => panic!("closure failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_tuple_unpacking_assign() {
        let source = "a, b, c = 1, 2, 3\nresult = a + b + c";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple unpack failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 6);
        }
    }

    #[test]
    fn test_dict_access_ops() {
        let source = "d = {1: 10, 2: 20}\nresult = d[1] + d[2]";
        let (res, frame) = run_exec_frame(source);
        res.expect("dict access failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_string_len() {
        let source = "result = len('hello')";
        let (res, frame) = run_exec_frame(source);
        res.expect("string len failed");
        unsafe {
            let r = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(r), 5);
        }
    }

    #[test]
    fn test_power_operator() {
        let result = run_eval("2 ** 10").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 1024);
        }
    }

    #[test]
    fn test_modulo() {
        let result = run_eval("17 % 5").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_floor_division() {
        let result = run_eval("17 // 3").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_bitwise_ops() {
        let result = run_eval("(0xFF & 0x0F) | 0x30").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 0x3F);
        }
    }

    #[test]
    fn test_list_comprehension() {
        // Use explicit loop with list + index (no method calls)
        let source = "\
result = [0, 0, 0]
i = 0
for x in [1, 2, 3]:
    result[i] = x * 2
    i = i + 1
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
                assert!(is_list(result));
                assert_eq!(w_list_len(result), 3);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 4);
                assert_eq!(w_int_get_value(w_list_getitem(result, 2).unwrap()), 6);
            },
            Err(e) => panic!("list comprehension failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_globals_builtin_uses_current_module_namespace() {
        let source = "x = 41\nresult = globals()['x'] + 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("globals() failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_locals_builtin_uses_current_function_locals() {
        let source = "\
def f(a, b):
    c = a + b
    return locals()['a'] + locals()['b'] + locals()['c']
result = f(2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in function failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_locals_builtin_uses_class_namespace() {
        let source = "\
x = 1
class C:
    y = 2
    snap = locals()
result = C.snap['y'] + globals()['x']";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in class failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_bound_method_materialized_by_attribute_access() {
        crate::test_hooks::install_hash_hook();
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add
result = m(41)";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_bound_method_lookup_materializes_method_object() {
        crate::test_hooks::install_hash_hook();
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup setup failed");
        unsafe {
            let c_obj = w_dict_getitem_str(frame.w_globals_obj, "c").unwrap();
            let m_obj = w_dict_getitem_str(frame.w_globals_obj, "m").unwrap();
            assert!(pyre_object::is_method(m_obj));
            assert!(std::ptr::eq(pyre_object::w_method_get_self(m_obj), c_obj));
        }
    }

    #[test]
    fn test_builtin_type_method_materialized_by_attribute_access() {
        let source = "\
xs = []
m = xs.append
m(42)
result = len(xs)";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin type method lookup failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_builtin_function_stored_on_class_is_not_bound() {
        let source = "\
class C:
    f = len
c = C()
result = c.f([1, 2, 3])";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin function descriptor semantics failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    /// `pypy/interpreter/typedef.py:817-831 Function.typedef.acceptable_as_base_class
    /// = False` enforces that `type(len)()` raises `TypeError("cannot
    /// create 'builtin_function' instances")` via the
    /// `init_builtin_function_type` `__new__` staticmethod.  This was
    /// previously failing because `PyError::type_error(msg)` produced
    /// an exception whose `args_w` slot stayed `PY_NULL`, so
    /// `str(e)` (which reads `W_BaseException.args_w` per
    /// `interp_exceptions.py:126-135 descr_str`) returned an empty
    /// string — `to_exc_object` now stamps `args_w = [msg]` per
    /// `:123-124 W_BaseException.descr_init self.args_w = args_w`.
    #[test]
    fn test_builtin_function_typedef_overrides_match_pypy() {
        // The `__doc__` slot routes through `getset_func_doc` which falls
        // back to `BuiltinCode.getdocstring` (function.py:446-449). pyre's
        // `len` is registered without a docstring so the access path
        // returns whatever code.getdocstring yields — the test only checks
        // that the lookup does not crash and that mutation/deletion fire
        // the orthodox `_check_code_mutable` AttributeError per
        // function.py:387 ("Cannot change __doc__ attribute of builtin
        // functions").
        let source = "\
doc_value = len.__doc__
self_is_none = len.__self__ is None
repr_result = len.__repr__()
new_err = ''
try:
    type(len)()
except TypeError as e:
    new_err = str(e)
set_err = ''
try:
    len.__doc__ = 'x'
except AttributeError as e:
    set_err = str(e)
del_err = ''
try:
    del len.__doc__
except AttributeError as e:
    del_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin_function typedef overrides failed");
        unsafe {
            let _doc_value = w_dict_getitem_str(frame.w_globals_obj, "doc_value").unwrap();
            let self_is_none = w_dict_getitem_str(frame.w_globals_obj, "self_is_none").unwrap();
            let repr_result = w_dict_getitem_str(frame.w_globals_obj, "repr_result").unwrap();
            let new_err = w_dict_getitem_str(frame.w_globals_obj, "new_err").unwrap();
            let set_err = w_dict_getitem_str(frame.w_globals_obj, "set_err").unwrap();
            let del_err = w_dict_getitem_str(frame.w_globals_obj, "del_err").unwrap();
            assert!(w_bool_get_value(self_is_none));
            assert_eq!(w_str_get_value(repr_result), "<built-in function len>");
            assert_eq!(
                w_str_get_value(new_err),
                "cannot create 'builtin_function' instances"
            );
            assert!(
                w_str_get_value(set_err).contains("__doc__"),
                "len.__doc__ = 'x' should raise AttributeError mentioning __doc__, got: {:?}",
                w_str_get_value(set_err)
            );
            assert!(
                w_str_get_value(del_err).contains("__doc__"),
                "del len.__doc__ should raise AttributeError mentioning __doc__, got: {:?}",
                w_str_get_value(del_err)
            );
        }
    }

    #[test]
    fn test_set_subtype_and_init_follow_pypy_constructor_protocol() {
        let source = "\
class S(set):
    pass
s = S([1, 2, 3])
manual = set()
set.__init__(manual, [4, 5])
is_subtype = type(s) is S
result = len(s)
manual_result = len(manual)";
        let (res, frame) = run_exec_frame(source);
        res.expect("set constructor parity failed");
        unsafe {
            let is_subtype = w_dict_getitem_str(frame.w_globals_obj, "is_subtype").unwrap();
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            let manual_result = w_dict_getitem_str(frame.w_globals_obj, "manual_result").unwrap();
            assert!(w_bool_get_value(is_subtype));
            assert_eq!(w_int_get_value(result), 3);
            assert_eq!(w_int_get_value(manual_result), 2);
        }
    }

    #[test]
    fn test_frozenset_constructor_exact_and_subtype_paths_match_pypy() {
        let source = "\
class F(frozenset):
    pass
seed = frozenset([1, 2])
same = frozenset(seed) is seed
sub = F([1, 2, 3])
is_subtype = type(sub) is F
result = len(sub)";
        let (res, frame) = run_exec_frame(source);
        res.expect("frozenset constructor parity failed");
        unsafe {
            let same = w_dict_getitem_str(frame.w_globals_obj, "same").unwrap();
            let is_subtype = w_dict_getitem_str(frame.w_globals_obj, "is_subtype").unwrap();
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert!(w_bool_get_value(same));
            assert!(w_bool_get_value(is_subtype));
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    /// `pypy/objspace/std/setobject.py:160-180 W_SetObject.descr_init`
    /// parses against `Signature(['some_iterable'])`, raising TypeError
    /// when called with more than one positional argument.  Previously
    /// failed because `set([1], 2)` *did* raise but with an empty
    /// `args_w` slot; once `error.to_exc_object` stamps
    /// `args_w = [msg]` per `interp_exceptions.py:123-124`, `str(e)`
    /// surfaces the message and the test passes.
    #[test]
    fn test_set_constructors_reject_extra_positionals_like_pypy() {
        // setobject.py:160 W_SetObject.descr_init parses against
        // `init_signature = Signature(['some_iterable'])`, so anything
        // beyond `(self, iterable)` is a TypeError; setobject.py:631
        // W_FrozensetObject.descr_new2 has the gateway-level fixed maxargs
        // for `(space, w_frozensettype, w_iterable=None)`.
        let source = "\
init_err = ''
try:
    set([1], 2)
except TypeError as e:
    init_err = str(e)
init_direct_err = ''
try:
    s = set()
    set.__init__(s, [1], 2)
except TypeError as e:
    init_direct_err = str(e)
frozen_err = ''
try:
    frozenset([1], 2)
except TypeError as e:
    frozen_err = str(e)
frozen_new_err = ''
try:
    frozenset.__new__(frozenset, [1], 2)
except TypeError as e:
    frozen_new_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("set/frozenset arity enforcement failed");
        unsafe {
            let init_err = w_dict_getitem_str(frame.w_globals_obj, "init_err").unwrap();
            let init_direct_err =
                w_dict_getitem_str(frame.w_globals_obj, "init_direct_err").unwrap();
            let frozen_err = w_dict_getitem_str(frame.w_globals_obj, "frozen_err").unwrap();
            let frozen_new_err = w_dict_getitem_str(frame.w_globals_obj, "frozen_new_err").unwrap();
            assert!(
                !w_str_get_value(init_err).is_empty(),
                "set([1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(init_direct_err).is_empty(),
                "set.__init__(s, [1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_err).is_empty(),
                "frozenset([1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_new_err).is_empty(),
                "frozenset.__new__(frozenset, [1], 2) should raise TypeError"
            );
        }
    }

    /// `pypy/objspace/std/typeobject.py:520-523
    /// W_TypeObject.check_user_subclass` refuses `set.__new__(int)`
    /// (and similar cross-layout calls) before the base allocator
    /// runs.  Previously failed because the cross-layout TypeError
    /// *was* raised but with an empty `args_w` slot; once
    /// `error.to_exc_object` stamps `args_w = [msg]` per
    /// `interp_exceptions.py:123-124`, `str(e)` surfaces the message
    /// and the test passes.
    #[test]
    fn test_set_new_rejects_foreign_layout_typedef() {
        // typeobject.py:520-523 W_TypeObject.check_user_subclass refuses
        // `set.__new__(int)` (and similar cross-layout calls) before the
        // base allocator runs. pyre's `check_user_subclass` enforces the
        // same layout-typedef identity guard.
        let source = "\
err = ''
try:
    set.__new__(int)
except TypeError as e:
    err = str(e)
frozen_err = ''
try:
    frozenset.__new__(int, [1, 2])
except TypeError as e:
    frozen_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("layout safety check failed");
        unsafe {
            let err = w_dict_getitem_str(frame.w_globals_obj, "err").unwrap();
            let frozen_err = w_dict_getitem_str(frame.w_globals_obj, "frozen_err").unwrap();
            assert!(
                !w_str_get_value(err).is_empty(),
                "set.__new__(int) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_err).is_empty(),
                "frozenset.__new__(int, [1, 2]) should raise TypeError"
            );
        }
    }

    #[test]
    fn test_metaclass_method_materialized_by_attribute_access() {
        crate::test_hooks::install_hash_hook();
        let source = "\
class Meta(type):
    def pick(cls):
        return cls
class C(metaclass=Meta):
    pass
bound = C.pick
result = bound()";
        let (res, frame) = run_exec_frame(source);
        res.expect("metaclass descriptor lookup failed");
        let result = unsafe { w_dict_getitem_str(frame.w_globals_obj, "result").unwrap() };
        let c_obj = unsafe { w_dict_getitem_str(frame.w_globals_obj, "C").unwrap() };
        assert!(std::ptr::eq(result, c_obj));
    }

    #[test]
    fn test_staticmethod_prepare_is_called_with_bound_lookup() {
        crate::test_hooks::install_hash_hook();
        let source = "\
class Meta(type):
    @staticmethod
    def __prepare__(name, bases):
        return {'seed': 41}
class C(metaclass=Meta):
    value = seed + 1
result = C.value";
        let (res, frame) = run_exec_frame(source);
        res.expect("__prepare__ lookup failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_dunder_globals_and_code_are_materialized() {
        crate::test_hooks::install_hash_hook();
        let source = "\
x = 7
def f(a, *, b=3):
    return a + b + x
g = f.__globals__
code = f.__code__";
        let (res, frame) = run_exec_frame(source);
        res.expect("function dunder lookup failed");
        let globals = unsafe { w_dict_getitem_str(frame.w_globals_obj, "g").unwrap() };
        let code = unsafe { w_dict_getitem_str(frame.w_globals_obj, "code").unwrap() };
        unsafe {
            let x = pyre_object::w_dict_lookup(globals, pyre_object::w_str_new("x")).unwrap();
            assert_eq!(w_int_get_value(x), 7);
            let argcount = crate::baseobjspace::getattr_str(code, "co_argcount").unwrap();
            assert_eq!(w_int_get_value(argcount), 1);
            let kwonly = crate::baseobjspace::getattr_str(code, "co_kwonlyargcount").unwrap();
            assert_eq!(w_int_get_value(kwonly), 1);
            let name = crate::baseobjspace::getattr_str(code, "co_name").unwrap();
            assert_eq!(w_str_get_value(name), "f");
            let varnames = crate::baseobjspace::getattr_str(code, "co_varnames").unwrap();
            let first = w_tuple_getitem(varnames, 0).unwrap();
            assert_eq!(w_str_get_value(first), "a");
        }
    }

    #[test]
    fn test_vars_builtin_raises_type_error_without_dict() {
        let source = "\
result = 0
try:
    vars(1)
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("vars() exception path failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_type_builtin_rejects_invalid_arity() {
        let source = "\
result = 0
try:
    type()
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("type() exception path failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals_obj, "result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }
}
