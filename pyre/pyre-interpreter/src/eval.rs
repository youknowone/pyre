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
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs, decode_instruction_forward,
    ensure_range_iter, execute_opcode_step, stack_underflow_error, unpack_sequence_exact,
};
use crate::{locals_w, locals_w_mut};
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

    static PYFRAME_ROOT_AREA: PyFrameRootArea = PyFrameRootArea {
        current_frame: CURRENT_FRAME.with(|frame| frame as *const _),
        last_exec_ctx: crate::call::capture_last_exec_ctx_cell(),
        import_roots: crate::importing::capture_import_root_area(),
        in_flight_exception: IN_FLIGHT_EXCEPTION.with(|cell| cell as *const _),
        bh_last_exception: majit_metainterp::blackhole::BH_LAST_EXC_VALUE
            .with(|cell| cell as *const _),
        guard_exception: majit_metainterp::blackhole::GUARD_EXC_VALUE
            .with(|cell| cell as *const _),
        jit_pending_exception: crate::stack_check::capture_jit_pending_exception_area(),
        pending_call_error: crate::call::capture_pending_call_error_area(),
        parked_call_errors: crate::call::capture_parked_call_errors_area(),
        pending_hash_error: crate::baseobjspace::capture_pending_hash_error_area(),
    };
}

struct PyFrameRootArea {
    current_frame: *const Cell<*mut PyFrame>,
    last_exec_ctx: *const (),
    import_roots: *const (),
    in_flight_exception: *const Cell<PyObjectRef>,
    bh_last_exception: *const Cell<i64>,
    guard_exception: *const Cell<i64>,
    jit_pending_exception: *const Cell<i64>,
    pending_call_error: *const (),
    parked_call_errors: *const (),
    pending_hash_error: *const (),
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

/// Flat TLS read of the per-thread `CURRENT_FRAME` slot — the frame a builtin
/// was called from, since a builtin call creates no Python frame of its own.
/// Null when no frame is installed.
///
/// The slot is runtime-mutable, not a build-time constant, so the JIT
/// residualises the read instead of tracing into it (`@dont_look_inside`, the
/// [`get_current_exception`] shape).
#[majit_macros::dont_look_inside]
pub fn current_frame() -> *mut PyFrame {
    CURRENT_FRAME.with(|current| current.get())
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
    // Barrier for the same reason as `ExecutionContext::enter`: this is a
    // traced `Type::Ref` store and `frame` can be an old-generation frame
    // taking a young predecessor.
    pyre_object::gc_hook::try_gc_write_barrier(frame as *mut PyFrame as *mut u8);
    majit_gc::bh_probe_note_store(
        frame as *mut PyFrame as usize,
        crate::pyframe::PYFRAME_F_BACKREF_OFFSET,
        3,
    );
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

/// Re-anchor a caller frame across a callee call that may run a moving minor
/// collection.
///
/// The interpreter holds the running frame as a raw `&mut PyFrame`, which the
/// GC does not treat as a managed reference.  When the callee allocates enough
/// to trigger a minor collection, the caller frame is relocated and that raw
/// pointer is left aimed at the abandoned nursery copy.  A field write through
/// it afterwards — the `CALL` result push and its `valuestackdepth` bump —
/// then lands on dead memory, so the live frame keeps a stack depth one slot
/// short and the next opcode reads the wrong operands.
///
/// Pushing the frame onto the shadow stack lets the root walker forward it in
/// place during the collection; `live()` reads the forwarded pointer back.
/// This mirrors the JIT eval layer's `FrameRoot`.
pub struct FrameAnchor {
    depth: usize,
    /// The shadow stack is per-thread, so a depth taken on one thread names a
    /// different slot on another. The marker is what keeps an anchor from
    /// being sent or shared across threads now that the type is public.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl FrameAnchor {
    pub fn new(frame: &mut PyFrame) -> Self {
        unsafe { Self::from_raw(frame as *mut PyFrame) }
    }

    /// Anchor a frame the caller holds as a raw pointer.
    ///
    /// `executioncontext`, the frame typedef and the introspection builtins
    /// carry `*mut PyFrame` rather than a reference, because the same frame
    /// stays reachable through the `f_backref` chain it was read off.  Minting
    /// a `&mut` for the length of the anchor would claim an exclusivity none of
    /// them has.
    ///
    /// # Safety
    /// `frame` must be null or name a live `PyFrame`.  A null is an
    /// anticipated input for the residual helpers, which take the caller frame
    /// as a raw operand the emit site may not have; the root walker skips a
    /// null slot, so anchoring one costs a push and answers null from
    /// [`Self::live`].
    pub unsafe fn from_raw(frame: *mut PyFrame) -> Self {
        let depth = majit_gc::shadow_stack::push(majit_ir::GcRef(frame as usize));
        Self {
            depth,
            _not_send: std::marker::PhantomData,
        }
    }

    pub fn live(&self) -> *mut PyFrame {
        majit_gc::shadow_stack::get(self.depth).0 as *mut PyFrame
    }
}

impl Drop for FrameAnchor {
    fn drop(&mut self) {
        majit_gc::shadow_stack::try_pop_to(self.depth);
    }
}

/// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
/// expose every live slot of `PyFrame.locals_cells_stack_w` on the active
/// f_backref chain as a GC root.
///
/// pyre's JIT-compiled code allocates W_IntObject / result boxes into the
/// nursery (`NewWithVtable` → `gc_alloc_typed_nursery_shim`). When the
/// nursery fills and a minor collection runs, only registered roots are
/// forwarded — an unforwarded nursery ref is left addressing a corpse.
/// `Nursery::reset` only rewinds the free pointer on native (it zero-fills
/// on wasm32, and writes the 0xAA poison only when that debug mode is on),
/// so the corpse keeps its forwarding header until something is allocated
/// over it and the stale ref reads whichever of the two it finds. The
/// interpreter stores live
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
        // The code object caches its own globals dict (`PyCode.w_globals`).
        // Reuse the same walk for managed custom tracing and bootstrap code.
        walk_raw_code_roots(func.code as PyObjectRef, visitor);
        visitor(&mut *(&mut func.closure as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.defs_w as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_kw_defs as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_module as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_func_globals_obj as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_builtins as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_ann as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_annotate as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_func_dict as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_typeparams as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_doc as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_qualname as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_objclass as *mut PyObjectRef as *mut majit_ir::GcRef));
        visitor(&mut *(&mut func.w_text_signature as *mut PyObjectRef as *mut majit_ir::GcRef));
        // BuiltinFunction.w_moduleobj is an ordinary movable module reference.
        // Builtin functions are immortal, so only this raw-root walker can
        // forward the slot during a collection.
        visitor(&mut *(&mut func.w_moduleobj as *mut PyObjectRef as *mut majit_ir::GcRef));
    }
}

/// Forward a `PyCode`'s cached globals, realized constants, qualname and
/// mapdict-method entries. This is both the managed wrapper's custom trace and
/// the explicit walk for bootstrap wrappers outside the collector. No-op for
/// non-code values.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn walk_raw_code_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe fn walk(
        value: PyObjectRef,
        visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
        visited: &mut Vec<usize>,
    ) {
        unsafe {
            if value.is_null() || !crate::pycode::is_code(value) {
                return;
            }
            let identity = value as usize;
            // PyPy's GC traces the PyCode constant list transitively and its
            // mark state prevents revisiting shared/cyclic nodes. PyCode
            // wrappers may be reached recursively without a collector mark
            // check, so mirror that small identity set while walking constants.
            if visited.contains(&identity) {
                return;
            }
            visited.push(identity);
            let code = &mut *(value as *mut crate::pycode::PyCode);
            visitor(&mut *(&mut code.w_globals as *mut PyObjectRef as *mut majit_ir::GcRef));
            // The realized `co_qualname` is an ordinary movable string object
            // shared by every function built from this code.
            visitor(&mut *(&mut code.w_qualname as *mut PyObjectRef as *mut majit_ir::GcRef));
            // `co_name` is realized and retained the same way.
            visitor(&mut *(&mut code.w_name as *mut PyObjectRef as *mut majit_ir::GcRef));
            if !code.co_consts_w.is_null() {
                for slot in (&*code.co_consts_w).iter() {
                    let mut child = slot.load(std::sync::atomic::Ordering::Acquire);
                    if child.is_null() {
                        continue;
                    }
                    visitor(&mut *(&mut child as *mut PyObjectRef as *mut majit_ir::GcRef));
                    slot.store(child, std::sync::atomic::Ordering::Release);
                    // Recurse through nested co_consts_w, matching PyPy's
                    // transitively traced PyCode list.
                    walk(child, visitor, visited);
                }
            }
            // mapdict.py:1418 CacheEntry.w_method is the cache's sole GC
            // reference.  PyPy traces it as part of the live PyCode; do the
            // same here now that managed code wrappers reach this walker from
            // their custom trace (the registry walk remains for bootstrap
            // prebuilt wrappers).
            if !code.mapdict_caches.is_null() {
                for entry in (&mut *code.mapdict_caches).iter_mut().flatten() {
                    visitor(
                        &mut *(&mut entry.w_method as *mut PyObjectRef as *mut majit_ir::GcRef),
                    );
                }
            }
        }
    }

    unsafe {
        if value.is_null() || !crate::pycode::is_code(value) {
            return;
        }
        walk(value, visitor, &mut Vec::new());
    }
}

/// Mark the GC-managed children of a `W_BaseException`.
///
/// Exception allocation is split: an ordinary exception goes to the non-moving
/// oldgen via `try_gc_alloc_stable_raw`, while the immortal singletons and the
/// GC-allocation-failure fallback use `malloc_typed` (`interp_exceptions.rs`
/// `new_exception`). For the `malloc_typed` family the collector never traces
/// the exception at all — the root visitor's `is_managed_heap_object` guard
/// short-circuits and `mark_object` is never reached — and for the oldgen
/// family a minor reaches the children only through the remembered set. Its
/// `args_w` tuple, `w_errno` / `w_strerror` / `w_filename` ints/strings,
/// `w_traceback` / `w_context` / `w_cause`, `w_dict`, … are ordinary GC-managed
/// objects, so when an exception is the only holder of those children (a caught
/// `except X as e` bound to a frame local) a collection sweeps them and a
/// later `e.args` / `e.errno` reads freed memory. Visit every
/// `W_BASE_EXCEPTION_GC_PTR_OFFSETS` slot in place, the same shape
/// `walk_raw_function_roots` / `walk_raw_getset_roots` use for Box/`malloc_typed`
/// -held children. No-op for non-exception values.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn walk_raw_exception_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicate (see `walk_raw_getset_roots`): `!is_exception`
        // over a cross-crate bool is `UnaryNotUnknownOperand` to the annotator.
        if pyre_object::interp_exceptions::is_exception(value) {
            for &offset in pyre_object::interp_exceptions::W_BASE_EXCEPTION_GC_PTR_OFFSETS.iter() {
                let slot = (value as usize + offset) as *mut PyObjectRef;
                visitor(&mut *(slot as *mut majit_ir::GcRef));
            }
        }
    }
}

/// Maximum recursion depth for `walk_raw_immortal_roots`.  Immortal objects
/// chain a few levels at most (`map(f, filter(p, iter(seq)))`, a view's dict,
/// a tuple of iterators); this bounds the walk against a malformed cycle — the
/// only cycle protection.
const IMMORTAL_WALK_MAX_DEPTH: u32 = 8;

/// Force-forward the managed children of any `malloc_typed`-immortal REGISTERED
/// pyre object reachable from a root slot (a frame value-stack/locals slot, or
/// an explicit `gc_roots::pin_root` shadow-stack slot).  Such objects are
/// outside the GC arenas, so the marker skips them and their registered
/// `gc_ptr_offsets` trace never fires; a managed child held solely through one
/// is freed by a collection (a bare `for x in <expr>:` whose iterator sits on
/// the value stack, a `d.keys()` view, an immortal iterator's source list, an
/// `_pickle.Unpickler` pinned across `load`).  Drive off the per-type offset
/// registry so every present and future immortal type is covered.  Runs on both
/// minor and major collections via the collection-kind-agnostic `visitor`.
///
/// # Safety
/// `value` must be a valid `PyObjectRef` or null, and `visitor` must accept the
/// forwarded child slots for the duration of the walk.
pub unsafe fn walk_raw_immortal_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe { walk_immortal_rec(value, visitor, 0) };
}

unsafe fn walk_immortal_rec(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    depth: u32,
) {
    unsafe {
        if value.is_null() || depth >= IMMORTAL_WALK_MAX_DEPTH {
            return;
        }
        // A tagged immediate is an int with no managed children; short-circuit
        // before any predicate below dereferences its `ob_type`. Gated on
        // `CAN_BE_TAGGED` (default false).
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(value) {
            return;
        }

        // (A) A `malloc_typed`-immortal REGISTERED pyre object: the marker
        // ignores it, so follow its registered `gc_ptr_offsets` in place and
        // recurse.  Managed registered objects (`try_gc_owns_object`) are
        // traced by the marker already, so leave their offsets to `None` and
        // fall through — re-walking a managed graph here is redundant.
        // Positive predicate (no `!` over a cross-crate bool): compute the
        // offsets only for a non-owned object.
        let immortal_offsets = if pyre_object::gc_hook::try_gc_owns_object(value as *mut u8) {
            None
        } else {
            pyre_object::gc_hook::offsets_for_pytype((*value).ob_type)
        };
        if let Some(offsets) = immortal_offsets {
            for &off in offsets {
                let slot = (value as usize + off) as *mut PyObjectRef;
                visitor(&mut *(slot as *mut majit_ir::GcRef));
                // Re-read the slot AFTER the visitor so a relocated child is
                // the one recursed into.
                walk_immortal_rec(*slot, visitor, depth + 1);
            }
            return;
        }

        // (B) A managed container reached AS A CHILD of an immortal (depth>=1):
        // its immortal elements would otherwise stay untraced-through (the
        // marker forwards element slots but does not recurse into an immortal
        // element's own children).  Enumerate element slots in place and
        // recurse.  Top-level (depth 0) managed containers are left to the
        // marker — matches prior scope.
        if depth >= 1 {
            if pyre_object::is_list(value) {
                if let Some((ptr, n)) = pyre_object::listobject::w_list_object_items_ptr_len(value)
                {
                    for i in 0..n {
                        let s = ptr.add(i) as *mut PyObjectRef;
                        visitor(&mut *(s as *mut majit_ir::GcRef));
                        walk_immortal_rec(*s, visitor, depth + 1);
                    }
                }
            } else if pyre_object::is_tuple(value) {
                // Dispatch on the concrete tuple layout (general vs specialised
                // arity-2): a specialised `_oo` stores objects inline, so a
                // general-tuple block read would mis-cast its inline slots.
                pyre_object::tupleobject::w_tuple_walk_gc_refs(
                    value,
                    &mut |s: *mut PyObjectRef| {
                        visitor(&mut *(s as *mut majit_ir::GcRef));
                        walk_immortal_rec(*s, visitor, depth + 1);
                    },
                );
            } else if pyre_object::is_dict(value) {
                let strat = pyre_object::dictmultiobject::w_dict_get_strategy(value);
                strat.walk_gc_refs(value, &mut |s: *mut PyObjectRef| {
                    visitor(&mut *(s as *mut majit_ir::GcRef));
                    walk_immortal_rec(*s, visitor, depth + 1);
                });
            } else if pyre_object::setobject::is_set_or_frozenset(value) {
                // `set` and `frozenset` share the `W_SetObject` layout and the
                // marker's `set_object_custom_trace`, so one walker covers both.
                pyre_object::setobject::w_set_walk_gc_refs(value, &mut |s: *mut PyObjectRef| {
                    visitor(&mut *(s as *mut majit_ir::GcRef));
                    walk_immortal_rec(*s, visitor, depth + 1);
                });
            }
        }
    }
}

/// Mark the GC-reachable children of a `getset_descriptor`
/// (`GetSetProperty`).  The descriptor itself is Box-immortal
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
        if pyre_object::typedef::is_getset_property(value) {
            let d = &mut *(value as *mut pyre_object::typedef::GetSetProperty);
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

/// Mark the GC-reachable children of the function a `staticmethod` /
/// `classmethod` wraps.  A builtin type dict binds the wrapper, not the
/// function, so `walk_raw_function_roots` applied to the dict value stops at
/// the wrapper and never descends to `w_function`.  That function is
/// Box-immortal — it never moves and so is never traced — while the metadata
/// `TypeCache.build` stamps onto it (`w_qualname`, `w_objclass`) and its
/// lazily allocated `w_func_dict` are ordinary young objects.  Without this
/// walk a minor collection leaves those slots pointing into vacated nursery
/// memory (`str.maketrans`, `dict.fromkeys`).  No-op for other values.
unsafe fn walk_raw_wrapped_function_roots(
    value: PyObjectRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    unsafe {
        if value.is_null() {
            return;
        }
        // Positive predicates (see `walk_raw_getset_roots`): `!is_staticmethod`
        // over a cross-crate bool is `UnaryNotUnknownOperand` to the annotator.
        if pyre_object::function::is_staticmethod(value) {
            walk_raw_function_roots(
                pyre_object::function::w_staticmethod_get_func(value),
                visitor,
            );
        }
        if pyre_object::function::is_classmethod(value) {
            walk_raw_function_roots(
                pyre_object::function::w_classmethod_get_func(value),
                visitor,
            );
        }
    }
}

/// Box-immortal builtin types never have their `W_TYPE_GC_TYPE_ID` custom
/// trace fired, but their namespaces, `bases`, and `weak_subclasses` can hold
/// young GC objects after startup.  Walk every registered builtin type (and
/// the rare pre-GC heap-type fallback) as pinned roots so those children stay
/// live and their slots are forwarded.  GC-managed heap types reach the same
/// fields through their own `W_TYPE_GC_TYPE_ID` custom trace.
unsafe fn walk_builtin_type_dicts_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    unsafe {
        for addr in pyre_object::typeobject::snapshot_builtin_type_roots() {
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
                let t = &mut *(w_type as *mut pyre_object::typeobject::W_TypeObject);
                forward(&mut t.w_name);
                forward(&mut t.w_qualname);
                // Heap and builtin types both hold a managed W_DictObject.
                // Forward the field itself; the dict's custom trace walks its
                // keys and values during a major collection. During a minor,
                // an already-old dict is not recursively rescanned merely
                // because this root slot is visited, so explicitly descend
                // through the strategy storage as the translated prebuilt
                // object's trace function does.
                let dict_slot = &mut t.dict as *mut *mut u8 as *mut PyObjectRef;
                forward(&mut *dict_slot);
                pyre_object::dictmultiobject::w_dict_walk_gc_refs(*dict_slot, &mut |slot| {
                    forward(slot)
                });
                // `weak_subclasses` holds `w_weakref_new` (`try_gc_alloc`)
                // young WEAKREF GcStructs whose only strong root is this
                // off-GC list; forward each slot in place so the WEAKREF
                // survives collection (its `weakptr` payload is invalidated
                // separately by the collector's weakref scan).  Without this,
                // the first collection reclaims the weakref and the base's
                // `weak_subclasses[i]` dangles — a UAF on the next
                // `mutated()` / `w_type_get_subclasses` deref.  The
                // `W_TYPE_GC_TYPE_ID` custom trace performs the same walk for
                // a heap type.
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
                // `mro_w` is a stable type-9 GcArray block (`alloc_mro_block_gc`)
                // whose only strong root is this immortal type. The custom
                // trace `type_object_custom_trace` never fires for a Box-immortal
                // owner, so forward the block field slot here: the collector
                // marks the tid-9 block and its varsize walker forwards
                // items[0..len]. Omitting this lets the first major collection
                // sweep the block — a UAF on the next MRO read. Guard on GC
                // ownership so the `std::alloc` bootstrap fallback (not owned)
                // is left in place.
                if !t.mro_w.is_null()
                    && pyre_object::gc_hook::try_gc_owns_object(t.mro_w as *mut u8)
                {
                    forward(&mut *(std::ptr::addr_of_mut!(t.mro_w) as *mut PyObjectRef));
                }
            }
        }
    }
}

/// Whether `PYRE_INTERP_RETURN_LOG` asks the RETURN_VALUE handler to trace
/// every returning frame.
///
/// Read once: `finish_value` runs on every interpreted return, and `getenv`
/// takes a lock and scans the environment array, so an uncached read puts that
/// scan on the return path of every Python-level call.
#[cfg(not(feature = "sandbox"))]
fn interp_return_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_INTERP_RETURN_LOG").is_some())
}

/// Whether the incminimark-parity minor-collection skip of clean prebuilt
/// structures is enabled (`PYRE_GC_PREBUILT_REMEMBER=0` opts out, restoring
/// the rescan-everything-every-minor behavior).
fn gc_prebuilt_remember_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        #[cfg(not(feature = "sandbox"))]
        {
            std::env::var("PYRE_GC_PREBUILT_REMEMBER").as_deref() != Ok("0")
        }
        // The host env is off-limits under sandbox; keep the parity default
        // (the prebuilt-remember minor-collection skip enabled).
        #[cfg(feature = "sandbox")]
        {
            true
        }
    })
}

/// Whether `PYRE_RERAISE_DIAG` asks `RERAISE` to describe an operand that is
/// not an exception before it raises the `TypeError`.
///
/// Read once: the check sits on the raise path, which is cold, but the flag is
/// also consulted from a `OnceLock` for the same reason as
/// [`interp_return_log_enabled`] — a `getenv` scans the environment array under
/// a lock.
#[cfg(not(feature = "sandbox"))]
fn reraise_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_RERAISE_DIAG").is_some())
}

#[cfg(feature = "sandbox")]
fn reraise_diag_enabled() -> bool {
    false
}

/// Describe a `RERAISE` operand that failed the `W_BaseException` check.
///
/// The operand is whatever `PUSH_EXC_INFO` pushed, and the compiler emits the
/// pair together, so a non-exception here means the frame's value-stack slot
/// was corrupted in between — never a program error. Report enough to tell the
/// two shapes apart: a NULL slot (a live slot that was cleared) versus a live
/// pointer whose object is no longer an exception (a stale or swept address).
fn reraise_bad_operand_diag(
    w_exc: PyObjectRef,
    oparg: u32,
    code_name: &str,
    depth: usize,
    below: &[PyObjectRef],
) {
    let describe = |v: PyObjectRef| -> String {
        if v.is_null() {
            return "NULL".to_string();
        }
        let ob_type = unsafe { (*v).ob_type };
        let name = if ob_type.is_null() {
            "<null ob_type>"
        } else {
            unsafe { (*ob_type).name }
        };
        let is_exc = unsafe { pyre_object::is_exception(v) };
        format!(
            "0x{:x}:{name}{}",
            v as usize,
            if is_exc { "(EXC)" } else { "" }
        )
    };
    let operand = if w_exc.is_null() {
        "NULL".to_string()
    } else {
        let addr = w_exc as usize;
        let owned = pyre_object::gc_hook::try_gc_owns_object(w_exc as *mut u8);
        let words: Vec<String> = (0..4)
            .map(|i| {
                format!("0x{:x}", unsafe {
                    *((addr + i * std::mem::size_of::<usize>()) as *const usize)
                })
            })
            .collect();
        format!(
            "{} gc_owned={owned} words=[{}]",
            describe(w_exc),
            words.join(", ")
        )
    };
    let stack: Vec<String> = below.iter().map(|&v| describe(v)).collect();
    // Through the seam, not `eprintln!`: under sandbox the interpreter reaches
    // fd 2 only via `ops::write`, and the compile-out fence rejects a direct
    // `std::io::_eprint` here.
    crate::host_seam::emit_stderr(
        format!(
            "[reraise] code={code_name} oparg={oparg} depth={depth} operand={operand} \
             below=[{}]\n",
            stack.join(", ")
        )
        .as_bytes(),
    );
}

pub fn capture_pyframe_root_area() -> *const () {
    PYFRAME_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// Advance the root walk one link along the `f_backref` chain.
///
/// `f_backref` holds a `jit.virtual_ref`, which at interpreter level is the
/// caller frame pointer itself; once the JIT virtualizes an inlined callee the
/// slot instead holds a `JitVirtualRef`, and reading that as a `PyFrame` would
/// interpret its `virtual_token` word as frame fields.  Hop through the vref
/// instead.  A still-virtual vref ends the walk: the frames it stands for have
/// no heap image to visit, and `virtualref.py force_virtual_if_necessary`
/// cannot run here because materializing one allocates.
#[inline]
unsafe fn chain_next_frame(f_backref: *mut PyFrame) -> *mut PyFrame {
    crate::executioncontext::vref_referent(f_backref)
}

/// Forward the frame named by a `JitVirtualRef` stored in a frame-shaped slot.
///
/// Frame backrefs and `executioncontext.rs`'s `vref_referent` both allow such a slot to
/// hold a vref in place of a `*mut PyFrame`. Its leading word is the
/// `JIT_VIRTUAL_REF_VTABLE` magic, not a PyObject `ob_type`, so callers must
/// skip PyObject-shaped walks when this returns true. The vref is old-gen and
/// non-moving, but its `forced` frame may be young.
/// `virtualref.py:94-98 is_virtual_ref(gcref)` supplies the predicate.
#[inline]
unsafe fn forward_virtual_ref_forced(
    value: *mut u8,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) -> bool {
    unsafe {
        if !majit_metainterp::virtualref::ptr_is_virtual_ref(value as *const u8) {
            return false;
        }
        let vref = value as *mut majit_metainterp::virtualref::JitVirtualRef;
        visitor(&mut *(&mut (*vref).forced as *mut *mut u8 as *mut majit_ir::GcRef));
        true
    }
}

/// Visit one `locals_cells_stack_w` slot as a GC root.
///
/// A slot holding a `JitVirtualRef` must skip the PyObject-shaped raw walks:
/// its leading word is the vtable magic rather than an `ob_type`.
unsafe fn walk_frame_value_slot(
    slot_ptr: *mut majit_ir::GcRef,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    // SAFETY: `slot_ptr` points to live `GcRef` storage whose allocation
    // outlives this walk. The visitor reads, conditionally forwards, and
    // stores back a `GcRef` (same layout as `*mut PyObject`).
    visitor(unsafe { &mut *slot_ptr });
    // A caught exception bound to a local (`except X as e`) is
    // `malloc_typed`-immortal, so the visitor above is a no-op
    // for it and its GC-managed children (`args_w`, `w_errno`,
    // …) are never traced. Forward them in place. Read the slot
    // AFTER the visitor so a relocated value is the live one.
    unsafe {
        let value = (*slot_ptr).0 as PyObjectRef;
        if forward_virtual_ref_forced(value as *mut u8, visitor) {
            return;
        }
        walk_raw_exception_roots(value, visitor);
        walk_raw_immortal_roots(value, visitor);
    }
}

/// How much of `locals_cells_stack_w` is reachable state.
///
/// `valuestackdepth` is an absolute index that starts at `stack_base()`
/// (`pyframe.rs`) and `pop` refuses to go below it, so for a running
/// frame it already covers the locals/cells prefix. `PyFrame::descr_clear`
/// is the one writer that breaks that: it rebinds every cell slot to a fresh
/// `w_cell_new` and then sets `valuestackdepth = 0`. Clamping to the raw
/// depth would walk zero slots for such a frame and drop cells that the
/// array still points at, so the prefix is the floor. The array is
/// `PY_NULL`-filled at allocation, so widening never reads uninitialised
/// slots.
unsafe fn walk_depth(f: &PyFrame, arr: &FixedObjectArray) -> usize {
    f.valuestackdepth.max(f.stack_base()).min(arr.len())
}

/// Walk one captured thread's active frame and interpreter root state.
///
/// # Safety
/// `data` must come from [`capture_pyframe_root_area`], and the owning thread
/// must be quiesced.
pub unsafe fn walk_pyframe_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let area = unsafe { &*(data as *const PyFrameRootArea) };
    // These are genuinely thread-local exception carriers, but every stopped
    // mutator's carrier must be visited during free-threaded collection.
    // PyPy reaches the equivalent values through each ExecutionContext /
    // thread state; keeping them in this registered per-mutator root area
    // preserves that ownership instead of resolving TLS on only the thread
    // which happened to initiate collection.
    unsafe {
        walk_in_flight_exception_area(area.in_flight_exception, visitor);
        walk_raw_exception_cell_area(area.bh_last_exception, visitor);
        walk_raw_exception_cell_area(area.guard_exception, visitor);
        walk_raw_exception_cell_area(area.jit_pending_exception, visitor);
        crate::call::walk_pending_call_error_area(area.pending_call_error, visitor);
        crate::call::walk_parked_call_errors_area(area.parked_call_errors, visitor);
        crate::baseobjspace::walk_pending_hash_error_area(area.pending_hash_error, visitor);
    }
    // incminimark.py:339-355 prebuilt-object scanning parity: a minor
    // collection scans an old/prebuilt object only when the write barrier
    // recorded a store into it since the previous minor collection
    // (`old_objects_pointing_to_young`); a major collection always traces
    // `prebuilt_root_objects`.  The Box-immortal structures walked below
    // (module dicts / cells, heap-type namespace dicts, method caches,
    // function fields) are pyre's prebuilt family; their mutation helpers
    // set `mark_prebuilt_roots_dirty`, so a clean bit during a minor
    // collection means no young pointer can be inside and the walks are
    // skipped.  Live-frame slots are real stack roots and are always walked.
    let is_minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    let scan_prebuilt = !is_minor
        || pyre_object::gc_roots::prebuilt_roots_dirty()
        || !gc_prebuilt_remember_enabled();
    let cf = unsafe { &*area.current_frame };
    {
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
        let ambient_ec = unsafe {
            (&*(area.last_exec_ctx as *const Cell<*const PyExecutionContext>)).get()
                as *mut PyExecutionContext
        };
        let mut visit_ec_slots = |ec: *mut PyExecutionContext| {
            if ec.is_null() {
                return;
            }
            let top_slot = unsafe { &mut (*ec).topframeref as *mut *mut PyFrame };
            visitor(unsafe { &mut *(top_slot as *mut majit_ir::GcRef) });
            unsafe { (*ec).walk_builtin_roots(visitor) };
            // `sys_exc_value` holds the active handler exception, which
            // is nursery-allocated and may move; forward it so the EC
            // slot is updated on a minor collection (the value-stack
            // copy alone is not authoritative for later EC reads).
            let exc_slot = unsafe { &mut (*ec).sys_exc_value as *mut PyObjectRef };
            visitor(unsafe { &mut *(exc_slot as *mut majit_ir::GcRef) });
            let async_exc_slot = unsafe { &mut (*ec).w_async_exception_type as *mut PyObjectRef };
            visitor(unsafe { &mut *(async_exc_slot as *mut majit_ir::GcRef) });
            // Exceptions may use the off-GC malloc_typed fallback.  In that
            // case forwarding the carrier above is a no-op, so trace its raw
            // GC-managed children just as `walk_in_flight_exception` does.
            // This is the post-PUSH_EXC_INFO owner of the same exception and
            // must preserve its traceback/frame graph identically.
            unsafe { walk_raw_exception_roots((*ec).sys_exc_value, visitor) };
            // `executioncontext.py:55` current_gen_or_coroutine is the head
            // of the running-generator chain.  The generator custom trace
            // follows each node's `previous_gen_or_coroutine` edge.
            let current_gen_slot =
                unsafe { &mut (*ec).current_gen_or_coroutine as *mut PyObjectRef };
            visitor(unsafe { &mut *(current_gen_slot as *mut majit_ir::GcRef) });
            let contextvar_slot = unsafe { &mut (*ec).contextvar_context as *mut PyObjectRef };
            visitor(unsafe { &mut *(contextvar_slot as *mut majit_ir::GcRef) });
            for hook in unsafe {
                [
                    &mut (*ec).w_asyncgen_firstiter_fn as *mut PyObjectRef,
                    &mut (*ec).w_asyncgen_finalizer_fn as *mut PyObjectRef,
                ]
            } {
                visitor(unsafe { &mut *(hook as *mut majit_ir::GcRef) });
            }
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
            // The walk covers `[0, walk_depth)` — the locals/cells prefix
            // plus the live operand stack. Slots above it are popped
            // capacity, not roots. Non-ref slots are filtered by
            // `is_nursery_object_start` inside the collector, so a slot
            // holding a non-object word is harmless for the bump-pointer
            // nursery.
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
                // The visitor above forwarded the slot, which for a vref leaves
                // the vref itself put — it is old-gen, so mark-sweep does not
                // move it — while the frame its `forced` slot names may be
                // young and relocating. `forced` is an interior slot the
                // collector reaches later, off the gray stack, but
                // `chain_next_frame` reads it during THIS scan, so forward it
                // here as well; otherwise the walk steps into a frame copy the
                // collector is about to abandon, and the roots only this walker
                // knows about (caught exceptions, module-dict cells, the
                // prebuilt families) get forwarded into dead memory. A direct
                // `f_backref` needs no such hop — the visitor already left it
                // naming the live copy.
                let f_backref = *f_back_slot;
                forward_virtual_ref_forced(f_backref as *mut u8, visitor);

                // pyframe.py:102 `self.pycode` — the running code object.
                // Visited as a root so a code object reachable only via
                // `frame.pycode` (e.g. `exec`'d code with no owning
                // Function) stays alive now that code objects are GC-managed.
                let pycode_slot = &mut (*(frame)).pycode as *mut *const ();
                visitor(&mut *(pycode_slot as *mut majit_ir::GcRef));
                // Forward the running code object's cached globals dict.  For
                // `exec`'d code with a movable (non-module) globals dict and no
                // owning Function, this frame is the only root that reaches it.
                walk_raw_code_roots((*(frame)).pycode as PyObjectRef, visitor);

                // PyFrame is normally a GC object in PyPy, so its GCREF
                // fields are traced before consumers dereference them.
                // pyre also has stdalloc-backed frames, so the frame root
                // walker must expose those fields explicitly.
                let locals_slot =
                    &mut (*(frame)).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
                visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
                // pyframe.py:75-76/276-279: translated PyPy stores the
                // generator owner in `f_generator_wref`; the `_nowref`
                // fallback exists only when translation has no weakrefs.
                // This field is therefore a non-owning back-reference in
                // pyre and must not keep the generator alive through an
                // escaped `cr_frame`/`gi_frame`.
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
                let w_builtin = (*frame).w_builtin;
                if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
                    // Module is a Box-immortal carrier, but its module dict is
                    // now a non-moving GC object. Mark the header through the
                    // owning field so its custom trace can retain the values.
                    let w_dict_slot = &mut (*(w_builtin as *mut pyre_object::module::Module)).w_dict
                        as *mut PyObjectRef;
                    visitor(&mut *(w_dict_slot as *mut majit_ir::GcRef));
                }
                // pyframe.py:49 `self.w_globals` is the dict OBJECT. Forward
                // the field before following the dict's own storage.
                let w_globals_obj_slot = &mut (*frame).w_globals as *mut PyObjectRef;
                visitor(&mut *(w_globals_obj_slot as *mut majit_ir::GcRef));
                // pyframe.py:147 `debugdata.w_locals` (the frame's locals
                // mapping object) and `w_f_trace` carry GCREFs that survive
                // the frame; forward both slots.  The locals mapping holds its
                // own bindings (module globals, class namespace, function
                // `locals()` dict, or an `exec` mapping), so forwarding the
                // object pointer keeps the whole namespace reachable.
                if !(*frame).debugdata.is_null() {
                    if pyre_object::gc_hook::try_gc_owns_object((*frame).debugdata as *mut u8) {
                        let debugdata_slot =
                            &mut (*frame).debugdata as *mut *mut crate::pyframe::FrameDebugData;
                        visitor(&mut *(debugdata_slot as *mut majit_ir::GcRef));
                    }
                    let d = &mut *(*frame).debugdata;
                    let w_locals_slot = &mut d.w_locals as *mut PyObjectRef;
                    visitor(&mut *(w_locals_slot as *mut majit_ir::GcRef));
                    let w_extra_locals_slot = &mut d.w_extra_locals as *mut PyObjectRef;
                    visitor(&mut *(w_extra_locals_slot as *mut majit_ir::GcRef));
                    let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
                    visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
                    let hidden_operationerr_slot = &mut d.hidden_operationerr as *mut PyObjectRef;
                    visitor(&mut *(hidden_operationerr_slot as *mut majit_ir::GcRef));
                }
                if !(*frame).lastblock.is_null()
                    && pyre_object::gc_hook::try_gc_owns_object((*frame).lastblock as *mut u8)
                {
                    let lastblock_slot =
                        &mut (*frame).lastblock as *mut *mut crate::pyframe::FrameBlock;
                    visitor(&mut *(lastblock_slot as *mut majit_ir::GcRef));
                }
                let live_obj = (*frame).w_globals;
                // For a W_ModuleDictObject the LOAD_GLOBAL read path consults the
                // authoritative `dstorage` cell map / `object_storage` /
                // strategy caches. Forward those movable
                // values here so a relocated global is not read back stale.
                // No-op for non-module dicts.  The picked builtin Module's
                // dict is consulted on a globals miss (`_load_global`
                // fallback), so forward it too.
                if scan_prebuilt {
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
                let next_frame = chain_next_frame((*frame).f_backref);
                if f.locals_cells_stack_w.is_null() {
                    (std::ptr::null_mut::<PyObjectRef>(), 0, next_frame)
                } else {
                    let arr = &*f.locals_cells_stack_w;
                    (
                        arr.items_ptr() as *mut PyObjectRef,
                        walk_depth(f, arr),
                        next_frame,
                    )
                }
            };
            if !arr_ptr.is_null() && depth > 0 {
                for i in 0..depth {
                    let slot_ptr = unsafe { arr_ptr.add(i) } as *mut majit_ir::GcRef;
                    unsafe { walk_frame_value_slot(slot_ptr, visitor) };
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
        if scan_prebuilt {
            unsafe {
                let mut forward = |slot: &mut PyObjectRef| {
                    visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
                    walk_raw_function_roots(*slot, visitor);
                    // getset descriptors are Box-immortal (custom trace never
                    // fires), so their collectable `fget`/`fset`/`fdel`
                    // functions must be marked reachable here or the getter
                    // dangles after a collection.
                    walk_raw_getset_roots(*slot, visitor);
                    walk_raw_wrapped_function_roots(*slot, visitor);
                };
                crate::importing::walk_import_roots_area(area.import_roots, &mut forward);
                // The `_mapdict_caches` LOAD_METHOD `w_method` slots
                // (mapdict.py:1418) and the stamped `w_globals` slots
                // (`pycode.py:159-165 frame_stores_global`) used to ride here
                // as per-thread areas.  Their holder is an immortal code
                // object that outlives the stamping thread, so they moved to
                // process-global walkers registered in `pyre-jit::eval`
                // (`mapdict_method_cache_root_walker`,
                // `w_globals_stamped_code_root_walker`).
            }
        }
    }
}

/// The interpreter's process-global off-GC slots, as one root source.
///
/// Frame roots are not here: `PyFrame.locals_cells_stack_w` and the
/// thread-local exception carriers ride the per-mutator `PyFrameRootArea`
/// (`walk_pyframe_root_area`), which reaches every registered thread rather
/// than only the collecting one. What is left is genuinely process-global —
/// app-level interphook handles, the `threading` module's own roots, and the
/// faulthandler's — so it registers once for the process.
fn walk_interpreter_global_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    walk_global_prebuilt_roots(visitor);
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    {
        // The cast reinterprets a mirror's link slot as a GC reference in
        // place, so the two spellings of a machine word have to stay
        // interchangeable for the forwarding write to land on the slot.
        const _: () = assert!(
            std::mem::size_of::<PyObjectRef>() == std::mem::size_of::<majit_ir::GcRef>()
                && std::mem::align_of::<PyObjectRef>() == std::mem::align_of::<majit_ir::GcRef>()
        );
        let mut forward = |slot: &mut PyObjectRef| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        };
        crate::cpyext::walk_gc_roots(&mut forward);
    }
    crate::executioncontext::walk_space_user_del_action_roots(visitor);
    #[cfg(not(target_arch = "wasm32"))]
    crate::module::signal::interp_signal::walk_check_signal_action_roots(visitor);
    crate::module::gc::hook::walk_hook_roots(visitor);
    crate::module::sys::vm::walk_monitoring_tool_roots(visitor);
    crate::module::thread::walk_thread_roots(visitor);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    crate::module::faulthandler::handler::walk_faulthandler_roots(visitor);
}

/// Install the interpreter's process-global GC root walker with the majit-gc
/// collector.
///
/// Called once at process startup from the JIT driver / pyrex main.
/// Stored in a process-global fn-pointer cell (#396); calling again with
/// the same fn pointer is idempotent.
pub fn register_interpreter_global_root_walker() {
    majit_gc::shadow_stack::register_extra_root_walker(walk_interpreter_global_roots);
}

thread_local! {
    /// The exception currently being raised / propagated up the Rust call
    /// stack. Between the raising frame's `record_application_traceback` and
    /// the frame that finally catches it, the exception is held only in the
    /// Rust `PyError` value in flight — not on any frame's value stack or in
    /// `ec.sys_exc_value` yet. `W_BaseException` is std::alloc-backed (outside
    /// the managed nursery/old-gen), so the collector never reaches it as a
    /// root and never traces its slots; a dispatch-loop safepoint running the
    /// non-moving old-gen major would sweep its old-gen traceback chain from
    /// under it. Mirrors `tstate->current_exception`: keep the in-flight
    /// exception's traceback chain a GC root until the exception is caught.
    static IN_FLIGHT_EXCEPTION: Cell<PyObjectRef> = const { Cell::new(pyre_object::PY_NULL) };
}

/// Publish the exception now being raised / propagated so the GC root walker
/// keeps it (and its traceback chain) alive across a collection. Called from
/// `record_application_traceback`, the single chokepoint every raising frame
/// passes through.
///
/// Writes the runtime-mutable `IN_FLIGHT_EXCEPTION` thread-local, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the `gc_interp::at_outermost_activation`
/// shape). One single-word argument, `()` result, and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn set_in_flight_exception(exc: PyObjectRef) {
    IN_FLIGHT_EXCEPTION.with(|c| c.set(exc));
}

#[allow(dead_code)]
fn walk_in_flight_exception(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    IN_FLIGHT_EXCEPTION.with(|c| {
        unsafe { walk_in_flight_exception_area(c as *const _, visitor) };
    });
}

unsafe fn walk_in_flight_exception_area(
    c: *const Cell<PyObjectRef>,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let c = unsafe { &*c };
    let exc = c.get();
    if exc.is_null() {
        return;
    }
    // Forward the exception OBJECT slot itself. A GC-managed exception
    // (oldgen-stable `w_exception_new_empty`) is marked here and the collector
    // then recurses into its slots via offset tracing.
    let slot = c.as_ptr();
    unsafe { visitor(&mut *(slot as *mut majit_ir::GcRef)) };
    // Off-GC fallback exceptions are malloc_typed, so explicitly trace their
    // raw GC-managed children after forwarding the carrier itself.
    let exc = unsafe { *(slot as *const PyObjectRef) };
    unsafe { walk_raw_exception_roots(exc, visitor) };
}

/// Forward one raw `i64` exception carrier cell and trace the exception's
/// GC-managed children. Shared by every such carrier
/// (`BH_LAST_EXC_VALUE`, `GUARD_EXC_VALUE`, `TL_JIT_PENDING_EXCEPTION`) so they
/// cannot drift.
pub(crate) unsafe fn walk_raw_exception_cell_area(
    c: *const Cell<i64>,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let c = unsafe { &*c };
    if c.get() == 0 {
        return;
    }
    let slot = c.as_ptr();
    unsafe { visitor(&mut *(slot as *mut majit_ir::GcRef)) };
    let exc = c.get() as PyObjectRef;
    unsafe { walk_raw_exception_roots(exc, visitor) };
}

fn walk_global_prebuilt_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // `reduce_protocol`'s app-level interphook handles are process-global
    // off-GC slots.  RPython stores them in the space's ordinary object graph;
    // expose the equivalent slots on every collection so both minor moves and
    // major marking preserve their functions and private globals namespace.
    crate::reduce_protocol::walk_handle_roots(visitor);
    // The aiter/anext app-level handles are the same off-GC-slot case.
    crate::async_operation::walk_handle_roots(visitor);
    // `_compat_pickle`'s fix_imports tables are `space.fromcache(State)` off-GC
    // slots; forward them on every collection so a minor move updates the cached
    // mapping pointers. Placed with the ungated handle roots (not the gated
    // prebuilt block) because the state is published lazily without
    // `mark_prebuilt_roots_dirty`, so its possibly-young dicts must be forwarded
    // on the first collection regardless of the prebuilt-remember bit.
    {
        let mut fwd = |slot: &mut PyObjectRef| {
            visitor(unsafe { &mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef) });
        };
        crate::module::_pickle::walk_pickle_state_gc(&mut fwd);
        // `space.fromcache(AuditHolder)`'s hooks are installed by running app
        // code, so a hook callable is young when it lands and cannot wait for
        // the prebuilt-roots scan below.
        crate::module::sys::vm::walk_audit_hooks_gc(&mut fwd);
        // `space.fromcache(CodecState)` publishes a young list and two young
        // dicts on first use and marks no prebuilt-roots dirty bit, so it
        // belongs with these two rather than behind the gate below.
        crate::module::_codecs::walk_codec_state_gc(&mut fwd);
    }
    let is_minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    let scan_prebuilt = !is_minor
        || pyre_object::gc_roots::prebuilt_roots_dirty()
        || !gc_prebuilt_remember_enabled();
    if !scan_prebuilt {
        return;
    }
    // PyPy's GC reaches standalone code objects through the ordinary object
    // graph. Pyre's bootstrap wrappers need the equivalent process-global
    // owner before walking module/type caches below.
    crate::pycode::walk_prebuilt_code_roots(visitor);
    unsafe {
        let mut forward = |slot: &mut PyObjectRef| {
            visitor(&mut *(slot as *mut PyObjectRef as *mut majit_ir::GcRef));
            walk_raw_function_roots(*slot, visitor);
            walk_raw_getset_roots(*slot, visitor);
            walk_raw_wrapped_function_roots(*slot, visitor);
        };
        walk_builtin_type_dicts_gc(&mut forward);
        // `typeobject.py:76-101 MethodCache` is an ordinary GC-managed
        // old/prebuilt object upstream.  A cache fill takes the write barrier;
        // pyre's off-GC equivalent calls `mark_prebuilt_roots_dirty`, so scan
        // it with the same remembered prebuilt family.  MiniMark promotes a
        // nursery survivor directly to oldgen in this minor, so no clean-minor
        // rescan is needed after the dirty bit is cleared.
        crate::baseobjspace::walk_method_cache_gc(&mut forward);
        // interp_posix.ApplevelForkCallbacks is another object-space cache.
        #[cfg(not(target_arch = "wasm32"))]
        crate::module::posix::interp_posix::walk_fork_callback_roots(&mut forward);
        // `space.sys.modules` and its authoritative dictionary belong to the
        // process/interpreter import state.  Keep them in the global
        // non-stack-root walk so incminimark's end-of-marking rescan sees
        // modules and bindings installed after the initial root snapshot.
        crate::importing::walk_process_import_roots(&mut forward);
    }
    if is_minor {
        pyre_object::gc_roots::clear_prebuilt_roots_dirty();
    }
}

/// Forward the GC slots a SUSPENDED generator's frame owns.
///
/// A suspended generator's frame is off the active `CURRENT_FRAME` /
/// `f_backref` chain that [`walk_pyframe_roots`] traverses, so its
/// locals/cells/valuestack and the generator's own slots are never
/// reached during root scanning.  The generator object's custom trace
/// (`pyre-jit` `generator_object_custom_trace`) calls this while marking
/// so the suspended frame's live references survive a collection.
///
/// Only the slots unique to the suspended frame are forwarded here.
/// The globals/builtin dict VALUES are not walked: a module dict is
/// rooted globally by `walk_module_dicts_gc`, and a GC-managed `exec`
/// globals dict is reached transitively once its (forwarded) object
/// pointer is marked — its own trace walks the values.  This deliberately
/// avoids the globals-proxy / module-dict-cell walk that
/// [`walk_pyframe_roots`] performs during root scanning, keeping the
/// marking-phase visit to plain slot forwarding.
pub fn walk_suspended_generator_frame(
    frame: *mut PyFrame,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    if frame.is_null() {
        return;
    }
    unsafe {
        let pycode_slot = &mut (*frame).pycode as *mut *const ();
        visitor(&mut *(pycode_slot as *mut majit_ir::GcRef));

        // The locals/cells/valuestack array pointer, then each element
        // slot — walked exactly as the per-frame body of
        // `walk_pyframe_roots` (the array pointer plus the full
        // fixed-length payload).
        let locals_slot =
            &mut (*frame).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
        visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
        if !(*frame).locals_cells_stack_w.is_null() {
            let arr = &*(*frame).locals_cells_stack_w;
            let base = arr.items_ptr() as *mut PyObjectRef;
            let len = walk_depth(&*frame, arr);
            for i in 0..len {
                visitor(&mut *(base.add(i) as *mut majit_ir::GcRef));
            }
        }

        // `f_generator_nowref` is the raw counterpart of PyPy's translated
        // `f_generator_wref`, not a frame-owned GC edge (pyframe.py:75-76,
        // 276-279).  The generator owns this suspended frame in the other
        // direction.
        let yielding_slot = &mut (*frame).w_yielding_from as *mut PyObjectRef;
        visitor(&mut *(yielding_slot as *mut majit_ir::GcRef));

        // Forward the globals/builtin object pointers; their dict VALUES are
        // not walked here — a module dict is rooted globally by
        // `walk_module_dicts_gc`, and a GC-managed `exec` globals dict is
        // reached transitively through its own trace.
        let w_globals_obj_slot = &mut (*frame).w_globals as *mut PyObjectRef;
        visitor(&mut *(w_globals_obj_slot as *mut majit_ir::GcRef));
        let w_builtin_slot = &mut (*frame).w_builtin as *mut PyObjectRef;
        visitor(&mut *(w_builtin_slot as *mut majit_ir::GcRef));
        let w_builtin = (*frame).w_builtin;
        if !w_builtin.is_null() && pyre_object::is_module(w_builtin) {
            let w_dict_slot =
                &mut (*(w_builtin as *mut pyre_object::module::Module)).w_dict as *mut PyObjectRef;
            visitor(&mut *(w_dict_slot as *mut majit_ir::GcRef));
        }

        if !(*frame).debugdata.is_null() {
            if pyre_object::gc_hook::try_gc_owns_object((*frame).debugdata as *mut u8) {
                let debugdata_slot =
                    &mut (*frame).debugdata as *mut *mut crate::pyframe::FrameDebugData;
                visitor(&mut *(debugdata_slot as *mut majit_ir::GcRef));
            }
            let d = &mut *(*frame).debugdata;
            let w_locals_slot = &mut d.w_locals as *mut PyObjectRef;
            visitor(&mut *(w_locals_slot as *mut majit_ir::GcRef));
            let w_extra_locals_slot = &mut d.w_extra_locals as *mut PyObjectRef;
            visitor(&mut *(w_extra_locals_slot as *mut majit_ir::GcRef));
            let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
            visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
            let hidden_operationerr_slot = &mut d.hidden_operationerr as *mut PyObjectRef;
            visitor(&mut *(hidden_operationerr_slot as *mut majit_ir::GcRef));
        }
        if !(*frame).lastblock.is_null()
            && pyre_object::gc_hook::try_gc_owns_object((*frame).lastblock as *mut u8)
        {
            let lastblock_slot = &mut (*frame).lastblock as *mut *mut crate::pyframe::FrameBlock;
            visitor(&mut *(lastblock_slot as *mut majit_ir::GcRef));
        }
    }
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

/// `executioncontext.py:219-233 sys_exc_info` — return the topmost handled
/// exception, including one parked on the running generator/coroutine chain.
/// The bytecode PUSH_EXC_INFO machinery deliberately continues to use
/// [`get_current_exception`] for the direct EC slot; app-level `sys.exception`
/// and bare raise use this logical view instead.
#[majit_macros::dont_look_inside]
pub fn get_sys_exception() -> PyObjectRef {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return PY_NULL;
    }
    unsafe { (*ec).sys_exc_info() }
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

/// `pyopcode.py:764-766` — `raise Class` instantiates the class, and
/// `normalize_exception` then validates the result.  `space.call_function`
/// propagates the constructor's own error in RPython; pyre's returns
/// `PY_NULL` with the error parked in the pending-call slot, so an unchecked
/// null would both lose that error and report the raise as
/// "exceptions must derive from BaseException".
///
/// # Safety
/// `w_type` must be a live exception class (`exception_is_valid_obj_as_class_w`).
unsafe fn instantiate_raised_class(w_type: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let result = unsafe { crate::call_function(w_type, &[]) };
    if result.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::type_error("exceptions must derive from BaseException")));
    }
    if !unsafe { pyre_object::is_exception(result) } {
        return Err(crate::error::exception_from_call_type_error(w_type, result));
    }
    Ok(result)
}

/// Instantiate the `from` cause of a `raise X from Y` when it is an exception
/// class, and answer it unchanged otherwise.
///
/// Whether the result derives from `BaseException` is deliberately not asked
/// here.  `pyopcode.py:757-760` instantiates and does nothing else; the check
/// is `error.py:376-385 set_cause`, which runs after the raised value has been
/// popped and normalized.  Asking it early lets `raise Cls from 42` answer the
/// cause's TypeError without ever running `Cls()`, and lets it pre-empt the
/// raised value's own "exceptions must derive from BaseException".
/// [`attach_raise_cause`] is where it is asked.
///
/// # TODO: inline back into RAISE_VARARGS
///
/// **Deviation.** RPython performs this inline inside `RAISE_VARARGS`
/// (`pypy/interpreter/pyopcode.py:757-760`, `space.call_function(w_cause)`
/// when `w_cause` is an exception class) without a named helper. Pyre
/// extracts the step into a standalone helper so the JIT raise BH path
/// (`pyre-jit/src/call_jit.rs`), the tracer (`pyre-jit-trace`) and the
/// interpreter raise path share one instantiation.
///
/// **When to fix.** When `bh_normalize_raise_varargs_with_frame` is removed or
/// rewritten — e.g. when the JIT BH path can dispatch the same inlined
/// `RAISE_VARARGS` sequence directly without a shared helper.
///
/// **How to fix.** Inline this body back into the `RAISE_VARARGS`
/// dispatch arm in `pyre-interpreter/src/pyopcode.rs` (mirroring
/// `pyopcode.py:704-707`), delete this standalone fn, and either route
/// the BH path through the inlined sequence or rewrite it to match
/// RPython's inline shape.
pub fn normalize_raise_cause(cause: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if !unsafe { crate::baseobjspace::exception_is_valid_obj_as_class_w(cause) } {
        return Ok(cause);
    }
    // `space.call_function(w_cause)` propagates the constructor's own error;
    // pyre's answers `PY_NULL` with the error parked in the pending-call slot,
    // so an unchecked null would lose it and report the raise as having no
    // cause at all.
    let result = unsafe { crate::call_function(cause, &[]) };
    if result.is_null() {
        return Err(crate::call::take_call_error().unwrap_or_else(|| {
            PyError::type_error("exception causes must derive from BaseException")
        }));
    }
    Ok(result)
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
    // in the typed slots on `W_BaseException` per
    // `interp_exceptions.py:113-117`.
    // `ExecutionContext.sys_exc_info()` is the logical handled-exception
    // view: when a generator is resumed from inside its caller's `except`,
    // `push_gen_or_coroutine` parks that caller exception on the generator
    // chain.  It must still become the context of a new exception raised by
    // the generator. `get_sys_exception` is residual, so this does not expose
    // the virtualizable frame to the tracer.
    crate::error::chain_context(exc, get_sys_exception());
    if let Some(cause_obj) = cause
        && !cause_obj.is_null()
    {
        // `error.py:376-385 set_cause` checks the cause here, after the raised
        // value has been normalized: `raise Cls from 42` therefore runs `Cls()`
        // first, and a raised value that is not an exception answers its own
        // TypeError rather than this one.  The wording is
        // `_exception_getclass(space, w_cause, "exception causes")`, not the one
        // `descr_setcause` uses for `e.__cause__ = x`.
        let valid =
            unsafe { pyre_object::is_none(cause_obj) || pyre_object::is_exception(cause_obj) };
        if !valid {
            return Err(PyError::type_error(
                "exception causes must derive from BaseException",
            ));
        }
        if unsafe { pyre_object::is_exception(exc) } {
            // `interp_exceptions.py:166-174 descr_setcause` — writes
            // `w_cause` and flips `suppress_context` to True.
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_cause(exc, cause_obj);
                pyre_object::interp_exceptions::w_exception_set_suppress_context(exc, true);
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
                if let Some(w_type) = pyre_object::w_tuple_getitem(exc_type, i)
                    && !crate::baseobjspace::exception_is_valid_class_w(w_type)
                {
                    return Err(PyError::type_error(CANNOT_CATCH_MSG));
                }
            }
        } else if !crate::baseobjspace::exception_is_valid_class_w(exc_type) {
            return Err(PyError::type_error(CANNOT_CATCH_MSG));
        }
    }
    Ok(())
}

fn validate_check_eg_match_class(exc_type: PyObjectRef) -> Result<(), PyError> {
    validate_check_exc_match_class(exc_type)?;
    let base_group = crate::builtins::lookup_exc_class("BaseExceptionGroup").unwrap();
    unsafe {
        if pyre_object::is_tuple(exc_type) {
            let n = pyre_object::w_tuple_len(exc_type) as i64;
            for i in 0..n {
                if let Some(w_type) = pyre_object::w_tuple_getitem(exc_type, i)
                    && crate::baseobjspace::issubclass(w_type, base_group)?
                {
                    return Err(PyError::type_error(
                        "catching ExceptionGroup with except* is not allowed. Use except instead.",
                    ));
                }
            }
        } else if crate::baseobjspace::issubclass(exc_type, base_group)? {
            return Err(PyError::type_error(
                "catching ExceptionGroup with except* is not allowed. Use except instead.",
            ));
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
    // registry (`lookup_exc_class_for_kind`).
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
    crate::baseobjspace::exception_match(w_exc_class.as_ptr(), exc_type)
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
    handle_exception_with_context(frame, err, next_instr, ContextSource::GeneratorChain)
}

/// Where the implicit `__context__` of `err` comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    /// `executioncontext.py:219-233 sys_exc_info` — the logical handled
    /// exception, walking to the generator that parked one. An exception the
    /// frame itself produces is raised while the caller's handler is live, so
    /// it takes that handler's exception.
    GeneratorChain,
    /// The flat EC slot alone, for an exception *thrown into* a resumed
    /// generator. `push_gen_or_coroutine` has already swapped the generator's
    /// own handled exception into the slot; the caller's belongs to the caller,
    /// which merely delivered this exception rather than raising it under a
    /// live handler. A generator holding none of its own therefore keeps a null
    /// context.
    ResumedFrameOnly,
}

/// [`handle_exception`] with an explicit context source
/// (`pyframe.py:303-306` records the context of a thrown-in
/// `SApplicationException` before the handler search).
pub fn handle_exception_with_context(
    frame: &mut PyFrame,
    err: &mut PyError,
    next_instr: &mut usize,
    context_source: ContextSource,
) -> bool {
    // An internal corruption marker is not a real Python exception and must
    // never be dispatched via bytecode handlers.
    if err.kind == crate::PyErrorKind::BytecodeCorruption {
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
    // Everything below allocates before it touches the frame again:
    // `to_exc_object` materialises the exception, `chain_context` builds the
    // `__context__` link, the trace hooks run arbitrary Python and
    // `record_application_traceback` allocates a `PyTraceback`.  A frame the
    // JIT built is a nursery object (`emit_new_pyframe_inline_with_params`),
    // so `frame` names the abandoned copy after any of them collect.  Anchor it
    // once and re-read at each point the frame is next used.
    let frame_anchor = FrameAnchor::new(frame);
    let exc_obj = err.to_exc_object();
    if err.exc_object.is_null() {
        err.exc_object = exc_obj;
    }
    // Implicit __context__ chaining: any exception raised while another is being
    // handled records that active exception as its __context__, not only an
    // explicit `raise`.
    //
    // `error.py:410-420 record_context` records it once and then marks the
    // OperationError, so the frames the SAME error merely unwinds through never
    // re-derive it.  `PyError::context_recorded` is that mark, and it rides the
    // error outward because the dispatch loop moves the same value into
    // `Err(err)` on propagation.  The mark is set below whether or not an active
    // exception was found, mirroring the `finally`.
    if !err.context_recorded {
        let active = match context_source {
            ContextSource::GeneratorChain => get_sys_exception(),
            ContextSource::ResumedFrameOnly => get_current_exception(),
        };
        crate::error::chain_context(err.exc_object, active);
        err.context_recorded = true;
    }
    let frame = unsafe { &mut *frame_anchor.live() };
    if err.attach_tb {
        if !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
            // The materialized exception is old-gen managed but lives only in the
            // `PyError` local; publish it as the in-flight root before the trace hook
            // runs arbitrary Python that can allocate and drive a major collection to
            // sweep an unrooted (white) exception. `record_application_traceback`
            // re-publishes the possibly-replaced operr below.
            set_in_flight_exception(err.exc_object);
            let saved_trace = frame.get_w_f_trace();
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = pyre_object::PY_NULL;
            }
            let after_exc_result =
                unsafe { (*ec).bytecode_trace_after_exception(frame as *mut PyFrame) };
            // The hook ran application code; restore the slot on the frame that
            // survived it.
            let frame = unsafe { &mut *frame_anchor.live() };
            if !saved_trace.is_null() {
                frame.getorcreatedebug(-1).w_f_trace = saved_trace;
            }
            if let Err(trace_err) = after_exc_result {
                // pyopcode.py:144-145 — `except OperationError as e: operr = e`.
                *err = trace_err;
            }
        }
        // pyopcode.py:144-149 — after `except OperationError as e: operr = e`,
        // record/trace the (possibly tracer-replaced) operr, not the exception
        // captured before the trace hook ran.  Re-derive from `err`: an
        // unreplaced err returns the cached object; a replaced err
        // materialises the replacement.  Cache it so record and trace share
        // one object.
        let operr_obj = err.to_exc_object();
        if err.exc_object.is_null() {
            err.exc_object = operr_obj;
        }
        // `pyopcode.py:147-148 pytraceback.record_application_traceback`
        // — prepends a `PyTraceback` wrapping the current frame onto
        // the exception's `w_traceback` chain.
        // `w_pytraceback_new` copies this pointer into the node it allocates,
        // so it has to be the address the frame has now.
        let frame = unsafe { &mut *frame_anchor.live() };
        unsafe {
            crate::pytraceback::record_application_traceback(
                operr_obj,
                frame as *mut PyFrame,
                frame.last_instr as i64,
            );
        }
    }
    if err.attach_tb && !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
        // `exception_trace` fabricates an `OperationError` whose
        // `normalize_exception` follows the `raise inst` shape
        // (error.py:238-245): the raised instance must sit in the
        // `w_type` slot with a null value so the `(inst, None)` path
        // derives the class.  Passing the instance as `w_value` with a
        // null `w_type` makes `normalize_exception` take `w_inst = w_type`
        // (null) and raise "exceptions must derive from BaseException".
        let operr_obj = err.to_exc_object();
        // `executioncontext.py:362` hands the tracer
        // `operr.get_w_traceback(space)` — the slot read with its mark.
        let w_tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(operr_obj) };
        unsafe { crate::pytraceback::mark_traceback_escaped(w_tb) };
        let frame = unsafe { &mut *frame_anchor.live() };
        if let Err(trace_err) = unsafe {
            (*ec).exception_trace(frame as *mut PyFrame, operr_obj, pyre_object::PY_NULL, w_tb)
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
    // `attach_tb=False` (RaiseWithExplicitTraceback) suppresses the traceback
    // record for the frame that performed the re-raise only.  Once that frame's
    // record/trace decision is made, the flag is cleared so that if no handler
    // is found here and the exception propagates to the caller, that outer frame
    // records its own traceback entry — mirroring the special-exception being
    // unwrapped to a plain OperationError after one frame.
    err.attach_tb = true;
    // `record_application_traceback` and `exception_trace` above both allocate;
    // `pycode` is read off the frame, so a stale frame yields a stale code
    // object as well as a stale value stack.
    let frame = unsafe { &mut *frame_anchor.live() };
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    // pyre's `last_instr` is a rustpython code-unit index; the PyPy-shaped
    // `lookup_exceptiontable` lookup takes byte offsets, so multiply by 2.
    // (See pycode.rs: varint values are word offsets but the lookup
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
        crate::pycode::lookup_exceptiontable(&code.exceptiontable, pc_bytes)
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
        // Same shape as `opcode_build_list`: materialise, then push. The push
        // has to land on the frame the materialisation left live.
        let frame = unsafe { &mut *frame_anchor.live() };
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
    frame.set_frame_finished_execution(true);

    false
}

/// Execute a frame — pure interpreter, no JIT.
///
/// Crate-private: the canonical
/// surface is `PyFrame::run` / `PyFrame::execute_frame` (PyPy
/// `pyframe.py:268 run` / `pyframe.py:331 execute_frame`).  Retained as a
/// free function because pyre's JIT override mechanism (call.rs
/// `EVAL_OVERRIDE: OnceLock<EvalFn>` where `EvalFn = fn(&mut PyFrame) ->
/// PyResult`) requires a `fn` pointer.  Rust methods cannot be cast to
/// `fn` pointers, so the canonical body stays as a free function and the
/// `EVAL_OVERRIDE.unwrap_or(eval_frame_plain)` fallback (`call.rs`'s `get_eval_fn`)
/// continues to reference it directly.
pub(crate) fn eval_frame_plain(frame: &mut PyFrame) -> PyResult {
    frame.execute_frame(None, None)
}

/// pyframe.py:270-299 execute_frame body — enter/call_trace/eval_loop/
/// return_trace/leave wrapping. When `operr` is Some, the generator's
/// throw() path routes it through handle_operation_error and sets
/// last_instr = next_instr - 1 before resuming (pyframe.py:273-277).
#[allow(dead_code)]
pub(crate) fn eval_frame_plain_with_operr(frame: &mut PyFrame, operr: Option<PyError>) -> PyResult {
    frame.execute_frame(None, operr)
}

enum FrameResume {
    Yielded(PyObjectRef),
    Dispatch(Option<PyError>),
}

/// pyframe.py:285-315 `resume_execute_frame`.  A suspended `yield from` is
/// resumed only after `execute_frame` has entered the outer frame.  Clearing
/// `w_yielding_from` before the delegate call gives the running generator the
/// same transient `gi_yieldfrom is None` state as PyPy.
fn prepare_frame_resume(
    frame: &mut PyFrame,
    w_inputvalue: Option<PyObjectRef>,
    operr: Option<PyError>,
    throw_args: Option<([PyObjectRef; 3], usize)>,
) -> Result<FrameResume, PyError> {
    let mut pending_operr = operr;
    if !frame.w_yielding_from.is_null() {
        let w_yf = frame.w_yielding_from;
        frame.w_yielding_from = pyre_object::PY_NULL;
        let w_arg = w_inputvalue.unwrap_or_else(pyre_object::w_none);
        match crate::baseobjspace::resume_yield_from(
            frame,
            w_yf,
            w_arg,
            pending_operr.take(),
            throw_args,
        ) {
            Ok(Some(value)) => return Ok(FrameResume::Yielded(value)),
            // The delegate's StopIteration value is already on the outer
            // frame stack and its SEND completion target is installed.
            Ok(None) => return Ok(FrameResume::Dispatch(None)),
            Err(err) => pending_operr = Some(err),
        }
    }
    if pending_operr.is_none()
        && let Some(w_arg_or_err) = w_inputvalue
    {
        let _ = frame.resume_execute_frame(w_arg_or_err)?;
    }
    Ok(FrameResume::Dispatch(pending_operr))
}

pub(crate) fn eval_frame_plain_with_resume(
    frame: &mut PyFrame,
    w_inputvalue: Option<PyObjectRef>,
    operr: Option<PyError>,
    throw_args: Option<([PyObjectRef; 3], usize)>,
) -> PyResult {
    // Spend one unit of the recursion budget on this frame's activation and
    // give it back when it returns.  Every Python frame costs the same unit —
    // module body, called function, `exec`ed code, resumed generator — so the
    // depth `stack_check()` reads is the number of live Python frames.
    let _recursion_depth = crate::call::enter_recursive_frame(frame);
    frame.fix_array_ptrs();
    if frame.execution_context.is_null() {
        match prepare_frame_resume(frame, w_inputvalue, operr, throw_args)? {
            FrameResume::Yielded(value) => return Ok(value),
            FrameResume::Dispatch(Some(mut err)) => {
                let mut next_instr = frame.next_instr();
                if !handle_exception_with_context(
                    frame,
                    &mut err,
                    &mut next_instr,
                    ContextSource::ResumedFrameOnly,
                ) {
                    return Err(err);
                }
                frame.last_instr = next_instr as isize - 1;
            }
            FrameResume::Dispatch(None) => {}
        }
        return eval_loop(frame);
    }
    let execution_context =
        unsafe { &mut *(frame.execution_context as *mut crate::PyExecutionContext) };
    // executioncontext.py / threadlocals.py parity: the current
    // ExecutionContext is owned by the OS-thread locals and is installed by
    // thread bootstrap.  Entering an (including inlined) frame must not
    // replace that thread-owned slot from a frame field; doing so lets a
    // collapsed/stale translated frame identity poison every subsequent
    // `space.getexecutioncontext()` lookup.
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
            match prepare_frame_resume(frame, w_inputvalue, operr, throw_args)? {
                FrameResume::Yielded(value) => {
                    w_exitvalue = value;
                    return Ok(value);
                }
                FrameResume::Dispatch(Some(mut err)) => {
                    let mut next_instr = frame.next_instr();
                    if !handle_exception_with_context(
                        frame,
                        &mut err,
                        &mut next_instr,
                        ContextSource::ResumedFrameOnly,
                    ) {
                        return Err(err);
                    }
                    frame.last_instr = next_instr as isize - 1;
                }
                FrameResume::Dispatch(None) => {}
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
    // Bump the monotonic frame eval-loop entry odometer: a user Python frame
    // is about to run bytecode.  The FBW FOR_ITER Option-C guard snapshots
    // this around a residual call to detect a body effect that ran through
    // user code (a side-effecting getter / dunder / module top level).
    crate::call::bump_frame_entry_count();
    // Count this interpreter activation so the JIT eval loop's GC safepoint
    // fires only at the outermost activation (PYRE_GC_INTERP root-completeness):
    // a nested `eval_loop_jit` running under this one observes depth > 1 and
    // skips collection. No-op when the flag is off.
    let _eval_activation = pyre_object::gc_interp::EvalActivationGuard::enter();
    if _eval_activation.armed() {
        // Publish the process-stable configuration in the breaker word once
        // per activation, before the first dispatch. Compiled back-edges mask
        // this bit out.
        majit_ir::eval_breaker_word::set_gc_interp();
    }
    let _current_frame_guard = if frame.execution_context.is_null() {
        install_current_frame(frame)
    } else {
        install_current_frame_tls_only(frame)
    };
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    let mut next_instr = frame.next_instr();

    loop {
        // PyPy's ActionFlag is one process breaker.  Keep pyre's free-threaded
        // finalization and STW extensions on the same already-established
        // breaker word, so the ordinary dispatch pays one relaxed load rather
        // than polling two process-global atomics independently.
        let dispatch_breaker = majit_ir::eval_breaker_word::load();
        if dispatch_breaker & majit_ir::eval_breaker_word::EB_FINALIZING != 0 {
            crate::module::thread::park_if_finalizing();
        }
        // Interpreter-path GC safepoint (PYRE_GC_INTERP), mirroring the JIT
        // eval loop. Between opcodes the only live refs are in the frame,
        // reachable through the installed `current_frame` root walker; no
        // bytecode handler holds a Rust-stack temporary here. A no-op unless
        // the flag is on and enough interpreter objects have accumulated.
        // Without it, a JIT-off run reclaims interpreter-routed old-gen
        // allocations only at explicit `gc.collect`, so RSS grows unbounded.
        if dispatch_breaker
            & (majit_ir::eval_breaker_word::EB_GC_INTERP | majit_ir::eval_breaker_word::EB_GC)
            != 0
        {
            pyre_object::gc_interp::safepoint();
        }
        // Free-threaded stop-the-world rendezvous.  Worker threads deliberately
        // execute this plain evaluator (their JitDriver state is thread-owned),
        // so they must poll the same process breaker as compiled/JIT-warm
        // loops; otherwise a non-allocating Python loop can prevent collection
        // and fork/finalization STW forever.
        if dispatch_breaker & majit_ir::eval_breaker_word::EB_STW != 0 {
            majit_gc::gc_sync::safepoint_poll();
        }

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
            if frame.take_failed_attr_before_opcode() {
                unsafe { (*ec).run_failed_attr_finalizers() };
            }
            let trace_result = unsafe {
                (*ec).bytecode_trace(
                    frame as *mut PyFrame,
                    crate::executioncontext::TICK_COUNTER_STEP,
                )
            };
            // pypy/interpreter/pyopcode.py:71-97 `handle_bytecode` wraps
            // `dispatch_bytecode` (which runs `bytecode_trace` at :203) in the
            // same `except OperationError`/`KeyboardInterrupt` that routes an
            // opcode error through `handle_operation_error`.  An exception a
            // signal handler delivers from `bytecode_trace` (e.g.
            // `CheckSignalAction` raising `KeyboardInterrupt`) must therefore
            // search this frame's exception blocks at `last_instr`, exactly
            // like the opcode error path below — not unwind the frame.
            // Propagating with `?` skipped that block search, so a `try`
            // around the interrupted instruction was bypassed and the
            // exception surfaced one frame up.
            if let Err(mut err) = trace_result {
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
            // A trace callback may perform a debugger line-jump by setting
            // `frame.f_lineno` (`PyFrame::fset_f_lineno` → `last_instr =
            // best_addr`).  Honour it: if a tracer is installed and it
            // moved `last_instr` off the instruction we were about to
            // dispatch, resume from the jump target instead of `pc`.  The
            // `gettrace` null-check keeps this off the no-tracer hot path.
            if unsafe { !(*ec).gettrace().is_null() } && frame.last_instr as usize != pc {
                next_instr = frame.last_instr as usize;
                continue;
            }
        }
        let (opcode_pc, instruction, op_arg) = decode_instruction_forward(code, pc)?;
        let fallthrough = opcode_pc + 1;
        // `decode_instruction_forward` absorbs any EXTENDED_ARG prefix
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
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
        }
    }
}

/// CPython 3.14's failed-attribute refcount boundary: once `LOAD_ATTR` has
/// popped a finalizable receiver, an otherwise-unreferenced temporary runs its
/// finalizer before the surrounding exception handler continues. This is
/// observable in `test_io.test_error_through_destructor` for both native and
/// `_pyio` streams. A reachability pass, rather than an IO/type shortcut,
/// decides whether the receiver is actually dead.
#[majit_macros::dont_look_inside]
pub(crate) fn finalize_failed_attr_receiver_now(obj: PyObjectRef) -> bool {
    crate::typedef::r#type(obj).is_some_and(|w_type| unsafe {
        crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__del__").is_some()
    })
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    type Anchor = FrameAnchor;

    fn anchor(&mut self) -> Self::Anchor {
        FrameAnchor::new(self)
    }

    fn push_anchored(anchor: &Self::Anchor, value: Self::Value) -> Result<(), PyError> {
        // A JIT-created frame lives in the nursery and the allocating step may
        // have relocated it; push onto the forwarded live frame.
        unsafe { &mut *anchor.live() }.push(value);
        Ok(())
    }

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        self.push(value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.stack_base() {
            return Err(stack_underflow_error("interpreter opcode"));
        }
        Ok(self.pop())
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        // The operand stack starts at `stack_base()` (`co_nlocals` + cell +
        // free slots), matching `_stack_start()`; guarding against `nlocals()`
        // alone would let an underflow slip into the cell/free region.
        // `valuestackdepth` is a `usize` field (seeded unsigned) whereas
        // `stack_base() + depth` seeds signed; cast both to `i64` (lowered as
        // `intmask`, identity on non-negative counts) so the guard compares
        // within one signedness instead of tripping the rtyper's
        // signed-vs-unsigned refusal.
        if (self.valuestackdepth as i64) <= (self.stack_base() + depth) as i64 {
            return Err(stack_underflow_error("interpreter peek"));
        }
        Ok(PyFrame::peek_at(self, depth))
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        // `pypy/interpreter/pyopcode.py:1457 MAKE_FUNCTION` stamps
        // `func.w_func_globals = self.w_globals` from the running
        // frame's dict object directly.  Pyre resolves the same
        // canonical sibling via `get_w_globals()` and threads it
        // through `make_function_from_code_obj_with_globals_obj` so
        // the freshly-created function's `__globals__` identity IS
        // the frame's view, with no second resolution that could surface a
        // different dict object.
        let w_globals = self.get_w_globals();
        // Capture the globals OBJECT only; the raw `*mut DictStorage` is
        // recovered from the object via the proxy back-link wherever a frame
        // built from this function still needs it.  Threading a raw here is
        // what dangled exec-defined functions when the exec temp storage was
        // freed (the `GlobalsBinding` leak), so it is dropped.
        Ok(crate::runtime_ops::make_function_from_code_obj_with_globals_obj(code_obj, w_globals))
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
        // The receiver is already off the value stack, so this local is the
        // only thing keeping it reachable while `getattr_str` runs arbitrary
        // Python — and the failing branch below still has to read its type.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        roots.pin_root(obj);
        getattr_str(obj, name).map_err(|error| {
            if finalize_failed_attr_receiver_now(roots.get(obj_slot)) {
                // Keep this write on the live virtualizable red frame.  The
                // opaque helper only answers the semantic type question;
                // writing `frame.flags` inside it would race the JIT's
                // register-resident copy of the same field.
                self.defer_failed_attr_until_pop_except();
            }
            error
        })
    }

    fn load_special_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        crate::baseobjspace::load_special_resolve(obj, name)
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
        Ok(locals_w!(self)[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = locals_w!(self)[idx];
        if value.is_null() {
            return Err(PyError::unbound_local_error(format!(
                "cannot access local variable '{name}' where it is not associated with a value"
            )));
        }
        // Cell objects are valid even if their contents are PY_NULL
        // (needed for __class__ cell during class body execution).
        // The cell itself is non-null, so the check above passes.
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // STORE_FAST always writes directly to the slot.
        // Cell content updates use STORE_DEREF, not STORE_FAST.
        self.set_locals_w(idx, value);
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
    /// `space.getitem(w_locals, name)` directly per PyPy
    /// `pyopcode.py:LOAD_NAME` `space.finditem_str(w_locals, name)`.
    fn load_name_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        let w_locals = self.get_w_locals();
        if !w_locals.is_null() {
            // At module scope `initialize_frame_scopes` binds `w_locals` to the
            // very same object as `w_globals`, so the locals probe here is a
            // redundant copy of the globals lookup `load_global_value` runs
            // next: same dict, same builtins fallback, identical result. Skip
            // it when they are identical — both to avoid the double lookup and,
            // critically, to avoid materializing a throwaway `w_str` key on
            // every module-loop LOAD_NAME (`load_global_value` already probes
            // the globals dict borrow-based via `getitem_str` + the cell cache).
            let w_globals = self.get_w_globals();
            if !std::ptr::eq(w_locals, w_globals) {
                // pyopcode.py:967-968 `w_value = space.finditem_str(
                // self.getorcreatedebug().w_locals, varname)`. The probe is
                // `finditem_str`, not `getitem`: an exact dict answers from its
                // strategy with the borrowed key, so neither the wrapped key nor
                // a KeyError is built. A class body reads every one of its names
                // through here, and the names it does not bind itself — a module
                // global, a builtin — miss on every pass, which is the miss that
                // shortcut exists for. `finditem_str` keeps a dict subclass on
                // the generic path, where a `__getitem__` or `__missing__`
                // override still runs.
                // `pyopcode.py:965 w_varname = self.getname_w(nameindex)`:
                // the generic arm names its key through `co_names_w`, so a
                // namespace that cannot use the borrowed-key shortcut — a
                // `dict` subclass, a non-dict mapping — wraps no key of its own
                // per execution either.
                if let Some(value) = crate::baseobjspace::finditem_str_named(
                    w_locals,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                )? {
                    return Ok(value);
                }
                // pyopcode.py:972 — a missing locals entry falls through to
                // `LOAD_GLOBAL_cached`.
            }
            return self.load_global_value(name, nameindex);
        }
        // No locals mapping bound (degenerate): fall through to globals.
        self.load_global_value(name, nameindex)
    }

    /// pyopcode.py:855-859 STORE_NAME —
    /// `space.setitem_str(self.getorcreatedebug().w_locals, varname, w_value)`.
    ///
    /// `nameindex` addresses the `co_names_w` slot naming the key on the
    /// non-dict-mapping arm below; callers holding no index pass
    /// [`crate::pyopcode::NO_NAMEINDEX`].
    ///
    /// Writes straight to `w_locals` (the class namespace, or — at module
    /// scope — the globals dict). It must NOT route through `getdictscope`:
    /// that runs `fast2locals`, which would erase a module frame's
    /// `CO_FAST_HIDDEN` inlined-comprehension locals (their fast slot is NULL,
    /// the binding lives in `w_locals` via STORE_NAME) on every store.
    fn store_name_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        let w_locals = self.get_or_create_w_locals();
        let hash = crate::baseobjspace::named_key_hash(name, self.pycode as PyObjectRef, nameindex);
        if store_name_into_dict(w_locals, name, hash, value) {
            return Ok(());
        }
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        crate::baseobjspace::setitem(w_locals, key, value)?;
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py:567 STORE_GLOBAL — bypasses w_locals
    /// and writes directly into w_globals so `exec("global x; x = 1", g, l)`
    /// lands `x` in `g` even when `l != g`.
    ///
    /// `nameindex` names the key through `co_names_w` on the dict-subclass arm;
    /// callers holding no index pass [`crate::pyopcode::NO_NAMEINDEX`].
    fn store_global_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        let w_globals = self.get_w_globals();
        let hash = crate::baseobjspace::named_key_hash(name, self.pycode as PyObjectRef, nameindex);
        if !w_globals.is_null() && !store_name_into_dict(w_globals, name, hash, value) {
            let key = unsafe {
                crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
            };
            crate::baseobjspace::setitem(w_globals, key, value)?;
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
        // `pyopcode.py:958-960 _load_global_fallback` uses
        // `space.finditem_str(self.get_w_globals(), varname)`.  finditem_str
        // takes a borrowed-string fast path for real W_DictObject /
        // W_ModuleDictObject layouts and dispatches a dict subclass through
        // the general mapping object, so a raising key `__eq__` propagates
        // instead of being swallowed as a miss.
        let w_globals = self.get_w_globals();
        if !w_globals.is_null()
            && let Some(value) = crate::baseobjspace::finditem_str_named(
                w_globals,
                name,
                self.pycode as PyObjectRef,
                nameindex,
            )?
        {
            return Ok(value);
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
        // is not the one being executed.  Identity is `pycode.w_globals
        // is self.get_w_globals_storage()` — the wrapped dict OBJECT on both
        // sides (`w_code_get_w_globals` vs the frame's `w_globals`).
        // `celldict.py:287-291 _LOAD_GLOBAL_cached`: under the JIT the
        // whole `GlobalCache` chase is bypassed via `_load_global_fallback`
        // → `_load_global` (`pyopcode.py:958-967 space.finditem_str`), so
        // only the builtin `finditem_str` fallback below runs.  Positive
        // form (`load_attr_cached`) keeps the annotator off
        // the bare-`!` hazard; `we_are_jitted()` folds to `ConstBool(true)`
        // so the cache arm and its `Arc<Mutex<GlobalCache>>` chase are
        // dead-code-eliminated on the lifted graph.
        let use_cache = if majit_metainterp::jit::we_are_jitted() {
            false
        } else {
            unsafe {
                let cwo = crate::pycode::w_code_get_w_globals(self.pycode as PyObjectRef);
                !cwo.is_null()
                    && std::ptr::eq(cwo, w_globals)
                    && !w_globals.is_null()
                    && pyre_object::dictmultiobject::is_module_dict(w_globals)
            }
        };
        // `_load_global` (pyopcode.py) reads the builtins through the METHOD
        // `self.get_builtin()`, not the `builtin` field.  Under the default
        // `honor__builtins__=False` the constructor never assigns that field
        // (pyframe.py) and the method answers `space.builtin` without
        // consulting the frame at all, so the field being unset must not be
        // observable here.  A frame the JIT materialized from a virtual is
        // exactly such a frame: reading the raw field made the builtins leg
        // silently find nothing there, raising `NameError` for a builtin that
        // plainly exists.
        let w_builtin = self.get_builtin();
        if use_cache {
            let cache_hit: Option<PyObjectRef> = unsafe {
                load_global_via_cache(
                    w_globals,
                    w_builtin,
                    name,
                    self.pycode as PyObjectRef,
                    nameindex,
                )
            }?;
            if let Some(value) = cache_hit {
                return Ok(value);
            }
        } else if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if !w_dict.is_null()
                && let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)?
            {
                return Ok(value);
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
        // `pyopcode.py:1844-1852 SWAP`, peek/settop element-wise.  A
        // `<[T]>::swap` call hands the locals array to a callee, which the
        // codewriter can only residualize; the element-wise spelling stays
        // native array operations, and it is what the preceding comment
        // already claimed this function did.
        //
        // `assert oparg >= 2` is upstream's own precondition, and it is what
        // makes both indices known non-negative to the annotator.
        assert!(depth >= 2);
        // Both halves go through the `maybe_none` read: a stack slot holds
        // NULL while an inlined comprehension runs, where
        // `LOAD_FAST_AND_CLEAR` pushed an unbound local and cleared its slot.
        let w_top = self.peekvalue_maybe_none(0);
        let w_other = self.peekvalue_maybe_none(depth - 1);
        // The first store overwrites the only stack slot still holding
        // `w_top`.  `FixedObjectArray::set_ref` roots its receiver and the
        // value it was handed, and nothing else, across a barrier that can
        // wait behind a foreign collection — so an unrooted `w_top` would be
        // stale by the second store.  The GC transform carries every live
        // variable on the shadow stack across that point; spell the same
        // thing out here and read `w_top` back.
        let roots = pyre_object::gc_roots::push_roots();
        let top_slot = roots.base();
        roots.pin_root(w_top);
        self.settopvalue(w_other, 0);
        self.settopvalue(roots.get(top_slot), depth - 1);
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

/// `pyopcode.py:855-859 space.setitem_str(w_ns, varname, w_value)` on a real
/// `W_DictObject` / `W_ModuleDictObject`: stores by borrowed `&str` through the
/// strategy without materializing a throwaway `w_str` (an overwrite reuses the
/// stored key; only a new name allocates one).  This is the raw mapping store,
/// not `__setitem__`, exactly as the object-keyed `setitem` resolves a dict.
///
/// Answers `false` when `w_ns` is not a dict, leaving the caller on the
/// object-keyed path: a dict subclass is an ordinary instance in pyre and must
/// keep its mapping identity and any `__setitem__` override, and a non-dict
/// mapping (`exec(src, g, mapping)`) has no strategy to store into.
///
/// `hash` is `name`'s digest when the caller holds it — the memo
/// `rstr.py:402-412 ll_strhash` keeps in the shared `co_names_w` string
/// (`crate::baseobjspace::named_key_hash`), so a stored name is hashed once per
/// string rather than once per opcode.  Zero leaves the strategy to hash the
/// borrowed bytes.
fn store_name_into_dict(w_ns: PyObjectRef, name: &str, hash: i64, value: PyObjectRef) -> bool {
    if !unsafe { pyre_object::is_dict(w_ns) } {
        return false;
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_hashed(w_ns, name, hash, value);
    }
    true
}

/// STORE_NAME for a caller holding the wrapped name rather than a `co_names_w`
/// index — the JIT's `bh_store_name_fn`, whose `w_name` operand is the trace's
/// own `box_str_constant`.
///
/// Naming the key from that constant is what keeps the mapping arm from minting
/// an immortal `w_str` per execution, the job `co_names_w` (`pycode.py:127-129`)
/// does for the interpreter; the residual carries no index to reach a slot with.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn store_name_value_w(
    frame: &mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    let name = unsafe { pyre_object::unicodeobject::w_str_get_value(w_name) };
    // The trace's own `box_str_constant` names the key, so it is the same
    // string object on every execution — `rstr.py:402-412 ll_strhash`'s memo
    // makes it hashed once rather than once per store.
    let hash = unsafe { pyre_object::unicodeobject::w_str_hash_memoized(w_name) };
    let w_locals = frame.get_or_create_w_locals();
    if store_name_into_dict(w_locals, name, hash, value) {
        return Ok(());
    }
    crate::baseobjspace::setitem(w_locals, w_name, value)?;
    Ok(())
}

/// DELETE_NAME counterpart of [`store_name_value_w`], for the JIT's
/// `bh_delete_name_fn`.  The trace already holds the key object, so it deletes
/// through that rather than naming the key again.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn delete_name_w(frame: &mut PyFrame, w_name: PyObjectRef) -> Result<(), PyError> {
    let w_locals = frame.get_or_create_w_locals();
    crate::baseobjspace::delitem(w_locals, w_name).map_err(|err| {
        if matches!(err.kind, PyErrorKind::KeyError) {
            let name = unsafe { pyre_object::unicodeobject::w_str_get_value(w_name) };
            PyError::name_error_with_name(format!("name '{name}' is not defined"), name)
        } else {
            err
        }
    })
}

/// STORE_GLOBAL counterpart of [`store_name_value_w`], for the JIT's
/// `bh_store_global_fn`.  `pyopcode.py:567` writes straight into `w_globals`.
///
/// # Safety
/// `w_name` must point to a valid `str` object.
pub unsafe fn store_global_value_w(
    frame: &mut PyFrame,
    w_name: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    let name = unsafe { pyre_object::unicodeobject::w_str_get_value(w_name) };
    let hash = unsafe { pyre_object::unicodeobject::w_str_hash_memoized(w_name) };
    let w_globals = frame.get_w_globals();
    if !w_globals.is_null() && !store_name_into_dict(w_globals, name, hash, value) {
        crate::baseobjspace::setitem(w_globals, w_name, value)?;
    }
    Ok(())
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
///
/// The `GlobalCache` chase walks an `Arc<Mutex<GlobalCache>>` cache
/// structure the tracer cannot model (its `deref` reads past the
/// refcount / borrow-flag header, not a value-model identity).
/// `celldict.py:287-291 _LOAD_GLOBAL_cached` bypasses the whole cache
/// under `jit.we_are_jitted()`, resolving the builtin through
/// `space.finditem_str` instead, so the cache is off-trace plumbing;
/// residualize it (`@jit.dont_look_inside`) like the sibling
/// `load_attr_caching`.
#[majit_macros::dont_look_inside]
unsafe fn load_global_via_cache(
    w_module_dict: PyObjectRef,
    w_builtin: PyObjectRef,
    name: &str,
    pycode: PyObjectRef,
    nameindex: usize,
) -> Result<Option<PyObjectRef>, PyError> {
    use pyre_object::celldict::unwrap_cell;
    use pyre_object::dictmultiobject::{DictOperationGuard, W_ModuleDictObject};
    let module_guard = DictOperationGuard::new(w_module_dict, &[w_builtin, pycode]);
    let w_module_dict = module_guard.root(0);
    let w_builtin = module_guard.root(1);
    let pycode = module_guard.root(2);
    // Body is a chain of unsafe-fn / raw-ptr ops on caller-supplied
    // PyObjectRefs; SAFETY contract is on the `unsafe fn` signature
    // (caller upholds w_module_dict / pycode / w_builtin validity).
    unsafe {
        // `celldict.py:292-313`: per-name slot fast path.  Read the slot,
        // upgrade the weakref; if the cache is alive, follow cell → builtincache
        // → builtins w_dict before falling through to the strategy lookup.
        if !pycode.is_null()
            && let Some(cache) = crate::pycode::w_code_globals_caches_get(pycode, nameindex)
        {
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
                let c = cache.lock().unwrap();
                (c.cell, c.valid, c.builtincache.clone())
            };
            if let Some(v) = cell_opt {
                return Ok(Some(unwrap_cell(v)));
            }
            if valid && let Some(bc) = bc_opt {
                let bcell = bc.lock().unwrap().cell;
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
        let cell_opt = cache.lock().unwrap().cell;
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

/// PyPy: pyopcode.py GET_ITER → space.iter(w_iterable)
///       pyopcode.py FOR_ITER → space.next(w_iterator)
impl IterOpcodeHandler for PyFrame {
    fn iter_value(&mut self, iterable: Self::Value) -> Result<Self::Value, PyError> {
        crate::baseobjspace::iter(iterable)
    }

    /// GET_ITER: convert iterable to iterator.
    /// PyPy: space.iter(w_iterable) → calls __iter__ or wraps in seq_iter.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            // mappingproxy iterates over its backing dict's keys.
            // dictproxyobject.py:41 descr_iter → space.iter(self.w_mapping).
            // The backing mapping is a `dict`, which moves, and publishing it
            // to the stack slot runs the frame array's write barrier, whose
            // slow path can wait behind a foreign collection
            // (`FixedObjectArray::set_ref` reloads both operands across it for
            // that reason).  Pin the mapping on the read, so both the store and
            // the operand this dispatches on come out of the root slot rather
            // than the word that was live before the barrier.  The bracket ends
            // with the branch: past it the mapping is the frame's own stack
            // slot, which the collector updates.
            let iter = if pyre_object::is_dict_proxy(iter) {
                let roots = pyre_object::gc_roots::push_roots();
                let mapping_slot = roots.base();
                roots.pin_root(pyre_object::w_dict_proxy_get_mapping(iter));
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, roots.get(mapping_slot));
                roots.get(mapping_slot)
            } else {
                iter
            };
            // `range` sequence → fresh `W_IntRangeIterator` cursor; replace
            // the stack operand so FOR_ITER advances the iterator, not the
            // reusable range object.  (Mirrors the dict-proxy rewrite
            // above.)  This runs in the loop preheader, outside the traced
            // loop body, so the JIT's `for i in range(N)` fast path is
            // unaffected.
            if pyre_object::is_w_range(iter) {
                let it = pyre_object::w_range_iter(iter);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // Already an iterator
            if pyre_object::is_range_iter(iter)
                || pyre_object::is_long_range_iter(iter)
                || pyre_object::is_seq_iter(iter)
                || pyre_object::is_list_iter(iter)
                || pyre_object::is_list_reverse_iter(iter)
                || pyre_object::is_tuple_iter(iter)
                || pyre_object::is_set_iterator(iter)
                || pyre_object::generator::is_generator(iter)
                || pyre_object::interp_itertools::is_repeat(iter)
                || pyre_object::interp_itertools::is_count(iter)
                || pyre_object::interp_itertools::is_takewhile(iter)
                || pyre_object::interp_itertools::is_dropwhile(iter)
                || pyre_object::interp_itertools::is_filterfalse(iter)
                || pyre_object::interp_itertools::is_pairwise(iter)
                || pyre_object::interp_itertools::is_cycle(iter)
                || pyre_object::interp_itertools::is_chain(iter)
                || pyre_object::dictmultiobject::is_dict_view_iterator(iter)
                || pyre_object::functional::is_enumerate(iter)
                || pyre_object::functional::is_reversed(iter)
                || pyre_object::functional::is_filter(iter)
                || pyre_object::functional::is_map(iter)
                || pyre_object::functional::is_zip(iter)
                || pyre_object::operation::is_callable_iterator(iter)
                || pyre_object::interp_sre::is_sre_scanner(iter)
                || crate::module::r#struct::is_unpack_iter(iter)
            {
                return Ok(());
            }
            // `pypy/objspace/std/dictmultiobject.py`
            // `W_DictViewKeysObject.descr_iter` (and values / items
            // siblings) returns a live `W_BaseDictMultiIterObject`. Pyre
            // produces a `W_BaseDictMultiIterObject` carrying the source
            // dict's `dictversion` counter so mid-iteration mutation
            // surfaces as `RuntimeError("dictionary changed size during
            // iteration")` per `:1719-1741 descr_next`.
            if pyre_object::dictmultiobject::is_dict_view(iter) {
                let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(iter);
                let w_dict = pyre_object::dictmultiobject::w_dict_view_get_dict(iter);
                let it = pyre_object::dictmultiobject::w_dict_view_iterator_new(w_dict, kind);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // list → W_FastListIterObject for an exact list; a subclass may override
            // `__iter__`, so route it through `space.iter`.
            if pyre_object::is_list(iter) {
                if pyre_object::is_exact_list(iter) {
                    let seq_iter = pyre_object::w_list_iter_new(iter);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, seq_iter);
                    return Ok(());
                }
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
                return Ok(());
            }
            // tuple → seq_iter for an exact tuple; a subclass may override
            // `__iter__`, so route it through `space.iter`.
            if pyre_object::is_tuple(iter) {
                if pyre_object::is_exact_tuple(iter) {
                    let seq_iter = pyre_object::w_tuple_iter_new(iter);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, seq_iter);
                    return Ok(());
                }
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
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
                        pyre_object::w_str_from_wtf8_managed(one)
                    })
                    .collect();
                let len = chars.len();
                let char_list = pyre_object::w_list_new(chars);
                let seq_iter = pyre_object::w_seq_iter_new(char_list, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // bytes/bytearray → seq_iter cursor over the bytes themselves, so
            // GET_ITER stays O(1) and a bytearray mutated mid-loop is observed
            // (baseobjspace::iter takes the same shape).
            if pyre_object::bytesobject::is_bytes_like(iter) {
                let len = pyre_object::bytesobject::bytes_like_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // dict → iterate over keys.
            // `pypy/objspace/std/dictmultiobject.py:W_DictMultiObject.descr_iter` returns
            // `W_DictMultiIterKeysObject` — pyre's
            // `W_BaseDictMultiIterObject` with kind=Keys plays the same
            // role, capturing the
            // dict's `dictversion` so mid-iteration mutation raises
            // `RuntimeError("dictionary changed size during
            // iteration")`.
            if pyre_object::is_dict(iter) {
                let it = pyre_object::dictmultiobject::w_dict_view_iterator_new(
                    iter,
                    pyre_object::dictmultiobject::DictViewKind::Keys,
                );
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, it);
                return Ok(());
            }
            // setobject.py W_BaseSetObject.descr_iter returns a live
            // W_SetIterObject over the set implementation.  Do not snapshot
            // the elements into a list: IteratorImplementation.next must see
            // a size change and make RuntimeError sticky.
            if pyre_object::is_set_or_frozenset(iter) {
                let set_iter = pyre_object::w_set_iter_new(iter);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, set_iter);
                return Ok(());
            }
            // array.array → seq_iter cursor (interp_array.py descr_iter
            // returns space.newseqiter(self)).
            if pyre_object::interp_array::is_array(iter) {
                let len = pyre_object::interp_array::w_array_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, seq_iter);
                return Ok(());
            }
            // User-defined __iter__ — PyPy: space.iter → __iter__().
            // Instances, plus typed-payload builtins (e.g. deque) whose
            // type registers `__iter__` on its MRO.  Delegates to
            // baseobjspace::iter which handles type MRO and __getitem__
            // fallback (PyPy: space.iter → PyObject_GetIter → tp_iter or
            // PySeqIter_New).  Already-iterator payloads returned above, so
            // this only sees non-iterator containers.
            if pyre_object::is_instance(iter)
                || crate::typedef::r#type(iter).is_some_and(|t| {
                    crate::baseobjspace::lookup_in_type_where(t.as_ptr(), "__iter__").is_some()
                })
            {
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.set_locals_w(tos, result);
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
                if let Some(metaclass) = mc
                    && let Some(method) = crate::baseobjspace::lookup_in_type(metaclass, "__iter__")
                {
                    let result = crate::call_function(method, &[iter]);
                    let tos = self.valuestackdepth - 1;
                    self.set_locals_w(tos, result);
                    return Ok(());
                }
            }
        }
        ensure_range_iter(iter)
    }

    /// FOR_ITER: advance the iterator one step.
    /// PyPy: space.next() → StopIteration means exhausted.
    fn iter_next(&mut self, iter: Self::Value) -> Result<Option<Self::Value>, PyError> {
        // baseobjspace::next walks the iterator protocol and raises
        // StopIteration for exhaustion.  All iterator kinds dispatch uniformly
        // through space.next here (pyopcode.py:1289 `w_nextitem =
        // self.space.next(w_iterator)`); the JIT specialises range/long-range/
        // seq by inlining this dispatch during tracing (trace_opcode.rs
        // iter_next), not by branching the interpreter opcode implementation.
        match crate::baseobjspace::next(iter) {
            Ok(result) => Ok(Some(result)),
            Err(e) if e.matches_stop_iteration() => Ok(None),
            Err(e) => Err(e),
        }
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
        #[cfg(not(feature = "sandbox"))]
        if interp_return_log_enabled() {
            unsafe {
                let code_ptr = crate::pyframe::pyframe_get_pycode(self);
                let name = if !code_ptr.is_null() {
                    (*code_ptr).obj_name.as_str()
                } else {
                    "?"
                };
                let arg0_intval = {
                    let lw = locals_w!(self);
                    if !lw.is_empty() {
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
        self.set_frame_finished_execution(true);
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

    fn complex_constant(&mut self, re: f64, im: f64) -> Result<Self::Value, PyError> {
        Ok(pyre_object::complexobject::w_complex_new(re, im))
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
        // Reached only for a code constant nested inside a container constant
        // (e.g. a tuple element), which has no top-level `co_consts_w` slot;
        // realize a wrapper directly.  Top-level `LOAD_CONST` of a code constant
        // goes through `constant_at` below.
        Ok(crate::pycode::box_code_constant(code))
    }

    fn constant_at(
        &mut self,
        index: crate::bytecode::oparg::ConstIdx,
        _enclosing: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        // `pyopcode.py:498-499 getconstant_w(index) -> co_consts_w[index]`:
        // return the one object `self.pycode` holds at `index`.
        Ok(unsafe {
            crate::pycode::w_code_const(self.pycode as pyre_object::PyObjectRef, usize::from(index))
        })
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }

    fn ellipsis_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(pyre_object::special::w_ellipsis())
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
        if d != attr || !crate::is_function_carrier(d) {
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
            //
            // `is_type` reports the physical layout every type object shares,
            // not the metaclass, so read the metaclass and require it to be
            // `type`.  The shape inferred below is what
            // `type.__getattribute__` produces; a custom metaclass can
            // override `__getattribute__` or define a data descriptor of the
            // same name, and either one produced `attr` in place of the
            // class's own MRO entry — binding `cls` onto that value would
            // pass the class to something that never asked for it.
            let metatype_is_type = crate::typedef::r#type(obj)
                .is_some_and(|meta| std::ptr::eq(meta.as_ptr(), crate::typedef::w_type()));
            if !metatype_is_type {
                return PY_NULL;
            }
            let raw = crate::baseobjspace::lookup_in_type(obj, name);
            match raw {
                Some(d) if pyre_object::is_classmethod(d) => obj,
                Some(_) => PY_NULL, // found in own MRO → no binding
                None => {
                    // Not in the type's own MRO → resolved via the
                    // metaclass MRO; bind the type for a method-descriptor
                    // function getattr surfaced unchanged.
                    match crate::typedef::r#type(obj)
                        .and_then(|meta| crate::baseobjspace::lookup_in_type(meta.as_ptr(), name))
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
            match crate::baseobjspace::lookup_in_type(w_type.as_ptr(), name) {
                Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                Some(d) if pyre_object::is_classmethod(d) => w_type.as_ptr(),
                Some(d) => method_descriptor_bound(d, attr, obj),
                None => PY_NULL,
            }
        } else {
            PY_NULL
        }
    }
}

/// Shared WITH_EXCEPT_START call semantics for the interpreter and generated
/// JitCode residual.  Keeping the call here preserves the bytecode operation's
/// exact `__exit__(type, value, traceback)` argument construction on both paths.
pub fn with_except_start_values(
    exit_func: PyObjectRef,
    exit_self: PyObjectRef,
    val: PyObjectRef,
) -> PyObjectRef {
    let exc_type = crate::typedef::r#type(val).map_or(pyre_object::w_none(), |p| p.as_ptr());
    // pyopcode.py:1358-1362 reads W_BaseException.w_traceback directly while
    // building the `__exit__(type, value, traceback)` arguments.
    let exc_tb = if unsafe { pyre_object::is_exception(val) } {
        let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(val) };
        if tb.is_null() {
            pyre_object::w_none()
        } else {
            tb
        }
    } else {
        pyre_object::w_none()
    };
    if exit_self.is_null() {
        crate::call_function(exit_func, &[exc_type, val, exc_tb])
    } else {
        crate::call_function(exit_func, &[exit_self, exc_type, val, exc_tb])
    }
}

impl OpcodeStepExecutor for PyFrame {
    fn pop_top(&mut self) -> Result<(), PyError> {
        let _ = self.pop_value()?;
        self.failed_attr_after_stack_pop();
        Ok(())
    }

    /// SETUP_ANNOTATIONS — ensure `__annotations__` exists in the
    /// current locals namespace. PyPy: pyopcode.py SETUP_ANNOTATIONS
    /// (typeobject.py auto-fills the slot at class creation, but the
    /// pyre-equivalent flow runs the bytecode opcode and writes into
    /// the class_locals namespace just like CPython).
    fn setup_annotations(&mut self) -> Result<(), PyError> {
        // `if not self.space.finditem_str(w_locals, '__annotations__')`:
        // probe by item lookup, not membership — a custom mapping's
        // `__contains__` can disagree with `__getitem__`/KeyError.
        let w_locals = self.get_or_create_w_locals();
        if crate::baseobjspace::finditem_str(w_locals, "__annotations__")?.is_none() {
            let key = unsafe { pyre_object::w_str_new("__annotations__") };
            crate::baseobjspace::setitem(w_locals, key, pyre_object::w_dict_new())?;
        }
        Ok(())
    }

    fn cleanup_throw(&mut self) -> Result<(), PyError> {
        let w_exc = self.pop_value()?;
        let mut err = unsafe { PyError::from_exc_object(w_exc) };
        if !err.matches_stop_iteration() {
            // CPython 3.14 `CLEANUP_THROW` installs the existing exception and
            // jumps straight to `exception_unwind`; unlike the ordinary
            // opcode-error path it does not prepend another traceback entry.
            // This is the same explicit-reraise shape as PyPy's
            // `RaiseWithExplicitTraceback`, represented in pyre by
            // `attach_tb = false`.
            err.attach_tb = false;
            return Err(err);
        }

        self.pop_value()?;
        self.pop_value()?;
        let value =
            if !err.exc_object.is_null() && unsafe { pyre_object::is_exception(err.exc_object) } {
                crate::baseobjspace::getattr_str(err.exc_object, "value")
                    .unwrap_or_else(|_| pyre_object::w_none())
            } else {
                pyre_object::w_none()
            };
        self.push(value);
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
        // Indices first: a subscript evaluates its receiver before its index
        // expression, so arithmetic inside the brackets puts the subtraction's
        // overflow check between the `locals_cells_stack_w` read and its use.
        let (i_val, i_self, i_func) = (depth - 1, depth - 4, depth - 5);
        let val = locals_w!(self)[i_val];
        let exit_self = locals_w!(self)[i_self];
        let exit_func = locals_w!(self)[i_func];
        let anchor = FrameAnchor::new(self);
        let res = with_except_start_values(exit_func, exit_self, val);
        if res.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::type_error("__exit__ failed")));
        }
        Self::push_anchored(&anchor, res)
    }

    // ── LoadCommonConstant ──
    fn load_common_constant(&mut self, cc: crate::bytecode::CommonConstant) -> Result<(), PyError> {
        // `LOAD_ASSERTION_ERROR` pushes the `AssertionError` class itself,
        // so `assert x` raises `AssertionError()` and `assert x, msg`
        // raises `AssertionError(msg)`.  The resolution is shared with the
        // JIT residual via `opcode_ops::load_common_constant_value`.
        let val = crate::opcode_ops::load_common_constant_value(cc);
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
        let slot = locals_w!(self)[idx];
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
        // CPython 3.14 compile.c keeps an explicit class-body `__class__`
        // assignment in the namespace (`STORE_NAME`) while the implicit cell
        // remains available for methods and is filled by `type.__new__`.
        // The pinned RustPython compiler emits STORE_DEREF for that assignment;
        // normalize its semantics at the opcode boundary.
        if crate::pyframe::class_scope_class_deref_is_name(self.code(), idx) {
            return self.store_name_value("__class__", crate::pyopcode::NO_NAMEINDEX, value);
        }
        let slot = locals_w!(self)[idx];
        if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_set(slot, value) };
        } else {
            self.set_locals_w(idx, value);
        }
        Ok(())
    }

    /// LOAD_CLOSURE — unified index. Push cell object itself (not contents).
    ///
    /// PyPy: pyopcode.py LOAD_CLOSURE → push cell for closure capture.
    fn load_closure(&mut self, idx: usize) -> Result<(), PyError> {
        let cell = locals_w!(self)[idx];
        self.push(cell);
        Ok(())
    }

    /// MAKE_CELL — wrap the slot value in a Cell.
    ///
    /// CPython 3.13 / RustPython MAKE_CELL — create cell object in slot.
    /// Wraps the current value (PY_NULL if uninitialized) in a Cell.
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
        let current = locals_w!(self)[idx];
        if current.is_null() || !unsafe { pyre_object::is_cell(current) } {
            let cell = pyre_object::w_cell_new(current);
            self.set_locals_w(idx, cell);
        }
        Ok(())
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), PyError> {
        if crate::pyframe::class_scope_class_deref_is_name(self.code(), idx) {
            return self.delete_name("__class__", crate::pyopcode::NO_NAMEINDEX);
        }
        // `pyopcode.py:580 DELETE_DEREF`: fetch the cell, raise if empty, then
        // `cell.set(None)` — clear the cell *contents* (PY_NULL is the empty
        // marker), not the slot pointer that holds the cell.  The cell lives at
        // `locals_w!(self)[idx]`, the same slot `load_deref`/`store_deref` use.
        let slot = locals_w!(self)[idx];
        let is_cell = !slot.is_null() && unsafe { pyre_object::is_cell(slot) };
        let contents = if is_cell {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if contents == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self.code(), idx));
        }
        if is_cell {
            unsafe { pyre_object::w_cell_set(slot, PY_NULL) };
        } else {
            self.set_locals_w(idx, PY_NULL);
        }
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
                let exc = get_sys_exception();
                if exc.is_null() || unsafe { pyre_object::is_none(exc) } {
                    Err(PyError::runtime_error("No active exception to reraise"))
                } else if unsafe { pyre_object::is_exception(exc) } {
                    // RAISE_VARARGS(nbargs=0) re-raises the active exception via
                    // RaiseWithExplicitTraceback, i.e. without recording a fresh
                    // traceback for this frame.  Preserving the exception's existing
                    // w_traceback keeps its identity stable (mirroring the RERAISE
                    // opcode below), which the except* metadata check
                    // (_is_same_exception_metadata) relies on.
                    let mut err = unsafe { PyError::from_exc_object(exc) };
                    err.attach_tb = false;
                    Err(err)
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
                        let result = instantiate_raised_class(w_value)?;
                        attach_raise_cause(result, None)?;
                        Err(PyError::from_exc_object(result))
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
                // `normalize_raise_cause` returns `Ok(null)` for a null/absent
                // cause; report absence as `None` so the value is never
                // `Some(null)` (mirrors the JIT paths call_jit.rs / trace_opcode.rs).
                let c = normalize_raise_cause(raw_cause)?;
                let cause = if c.is_null() { None } else { Some(c) };
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // Root the normalized `cause` across the class
                        // instantiation: it is a fresh oldgen exception held only
                        // in this Rust local, invisible to the precise collector
                        // until `attach_raise_cause` reads it, and `call_function`
                        // below can drive a collection.
                        let _roots = pyre_object::gc_roots::push_roots();
                        if let Some(c) = cause {
                            pyre_object::gc_roots::pin_root(c);
                        }
                        // pyopcode.py:711-713 — class raise: call the type.
                        let result = instantiate_raised_class(w_value)?;
                        attach_raise_cause(result, cause)?;
                        Err(PyError::from_exc_object(result))
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
    fn import_module(&mut self, name: &str) -> Result<PyObjectRef, PyError> {
        if let Some(m) = crate::importing::get_sys_module(name) {
            return Ok(m);
        }
        crate::importing::importhook(
            name,
            self.get_w_globals(),
            pyre_object::w_none(),
            0,
            self.execution_context,
        )
    }

    fn build_template_op(&mut self) -> Result<(), PyError> {
        // Stack: [strings, interpolations] (two tuples the compiler split).
        let interpolations = self.pop();
        let strings = self.pop();
        let anchor = FrameAnchor::new(self);
        let module = self.import_module("_template")?;
        let func = getattr_str(module, "_build_template")?;
        let result = call_callable(self, func, &[strings, interpolations])?;
        Self::push_anchored(&anchor, result)
    }

    fn build_interpolation_op(
        &mut self,
        conversion: u32,
        has_format_spec: bool,
    ) -> Result<(), PyError> {
        // Stack: [value, expression, format_spec?] — format_spec present only
        // when the oparg low bit is set, else it defaults to the empty string.
        let format_spec = if has_format_spec {
            self.pop()
        } else {
            pyre_object::w_str_new("")
        };
        let expression = self.pop();
        let value = self.pop();
        let conversion_obj = pyre_object::w_int_new(conversion as i64);
        let anchor = FrameAnchor::new(self);
        let module = self.import_module("_template")?;
        let func = getattr_str(module, "_build_interpolation")?;
        let result = call_callable(
            self,
            func,
            &[value, expression, conversion_obj, format_spec],
        )?;
        Self::push_anchored(&anchor, result)
    }

    fn import_name(&mut self, name: &str) -> Result<(), PyError> {
        let w_fromlist = self.pop();
        let w_flag = self.pop();
        let anchor = FrameAnchor::new(self);
        let w_obj = crate::importing::import_name(self, name, w_fromlist, w_flag)?;
        Self::push_anchored(&anchor, w_obj)
    }

    // PyPy: pyopcode.py IMPORT_FROM
    // Stack: [module] → peek module, push getattr(module, name)
    fn import_from(&mut self, name: &str) -> Result<(), PyError> {
        let module = self.peek();
        let ec = self.execution_context;
        let anchor = FrameAnchor::new(self);
        let attr = crate::importing::import_from(module, name, ec)?;
        Self::push_anchored(&anchor, attr)
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        // CPython 3.13: TOS = container, TOS1 = item
        let haystack = self.pop();
        let needle = self.pop();
        let anchor = FrameAnchor::new(self);
        let result = crate::baseobjspace::contains(haystack, needle)?;
        let inverted = match invert {
            crate::bytecode::Invert::No => result,
            crate::bytecode::Invert::Yes => !result,
        };
        Self::push_anchored(&anchor, pyre_object::w_bool_from(inverted))
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), PyError> {
        let b = self.pop();
        let a = self.pop();
        // `COMPARE_OP 'is'` → `space.is_w` (descroperation.py): plain
        // `int`s are identical by value (`W_IntObject.is_w`), everything
        // else by pointer.
        let same = crate::baseobjspace::is_w(a, b);
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
        let anchor = FrameAnchor::new(self);
        let truth = crate::baseobjspace::is_true(val)?;
        Self::push_anchored(&anchor, pyre_object::w_bool_from(truth))
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
        if locals_w!(self)[idx].is_null() {
            let code = unsafe { &*crate::pyframe_get_pycode(self) };
            let name = if idx < code.varnames.len() {
                code.varnames[idx].as_str()
            } else {
                ""
            };
            return Err(PyError::unbound_local_error(format!(
                "cannot access local variable '{name}' where it is not associated with a value"
            )));
        }
        self.set_locals_w(idx, PY_NULL);
        Ok(())
    }

    // ── FormatSimple (str(TOS)) ──
    fn format_simple(&mut self) -> Result<(), PyError> {
        let val = self.pop();
        // `f'{x}'` → `PyObject_Format(x, NULL)`; a user `__format__` is
        // invoked with an empty spec, otherwise this is `str(value)`.
        let anchor = FrameAnchor::new(self);
        let s = crate::runtime_ops::format_value(val, pyre_object::PY_NULL)?;
        Self::push_anchored(&anchor, s)
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
        let anchor = FrameAnchor::new(self);
        let s = crate::runtime_ops::format_value(val, spec)?;
        Self::push_anchored(&anchor, s)
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
        let anchor = FrameAnchor::new(self);
        let converted = crate::runtime_ops::convert_value(val, code)?;
        Self::push_anchored(&anchor, converted)
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
                // CPython 3.14's annotations compile scope is qualified as
                // `<target>.__annotate__`.  The external compiler currently
                // supplies only the enclosing scope, so finish the compiler
                // metadata at the bytecode operation that binds the annotate
                // function to its target.
                let roots = pyre_object::gc_roots::push_roots();
                let func_slot = roots.base();
                roots.pin_root(func);
                let attr_slot = func_slot + 1;
                roots.pin_root(attr);
                let mut qualname =
                    unsafe { crate::function::function_get_qualname(roots.get(func_slot)) };
                qualname.push_str(".__annotate__");
                let w_qualname = pyre_object::w_str_from_wtf8(qualname);
                unsafe {
                    crate::function::function_set_qualname(roots.get(attr_slot), w_qualname);
                    crate::function::function_set_annotate_unchecked(
                        roots.get(func_slot),
                        roots.get(attr_slot),
                    );
                }
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
        // `PUSH_EXC_INFO` transfers ownership from the propagating `PyError`
        // to the execution context.  Its `sys_exc_value` slot and raw
        // exception children are walked above, so the temporary propagation
        // root must no longer retain a completed handler's traceback.
        set_in_flight_exception(pyre_object::PY_NULL);
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

    fn check_eg_match(&mut self) -> Result<(), PyError> {
        let exc_type = self.pop();
        validate_check_eg_match_class(exc_type)?;
        let exc_value = self.pop();
        let anchor = FrameAnchor::new(self);
        let (matching, rest) = if unsafe { pyre_object::is_none(exc_value) } {
            (pyre_object::w_none(), pyre_object::w_none())
        } else {
            crate::builtins::exception_group_match(exc_value, exc_type)?
        };
        Self::push_anchored(&anchor, rest)?;
        Self::push_anchored(&anchor, matching)?;
        if !unsafe { pyre_object::is_none(matching) } {
            set_current_exception(matching);
        }
        Ok(())
    }

    fn prep_reraise_star(
        &mut self,
        orig: Self::Value,
        exceptions: Self::Value,
    ) -> Result<Self::Value, PyError> {
        crate::builtins::exception_group_prep_reraise_star(orig, exceptions)
    }

    // ── PopExcept ──
    fn pop_except(&mut self) -> Result<(), PyError> {
        // Restore previous exc_info from stack.  Named TLS accessor for
        // the same codewriter-resolvability reason as `push_exc_info`.
        let prev_exc = self.pop();
        set_current_exception(prev_exc);
        self.failed_attr_after_pop_except();
        Ok(())
    }

    // ── Reraise ──
    // `pypy/interpreter/pyopcode.py:1348-1376 RERAISE`.
    //
    // `RERAISE` reads the operand `PUSH_EXC_INFO` left on the value stack, so
    // this opcode only sees a non-exception when the frame's stack slot itself
    // went bad between the two. That is unreachable for a correct interpreter,
    // and the intermittent `test_asyncio` failures reach it — hence the
    // diagnostic below rather than a bare `TypeError`.
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
            if reraise_diag_enabled() {
                // Snapshot what is left below the popped operand. If the real
                // exception is sitting one or two slots away, the defect is a
                // stack-depth miscount; if it is nowhere on the stack, the slot
                // held a reference the collector reclaimed and reissued.
                let depth = self.valuestackdepth;
                let below: Vec<PyObjectRef> = (0..6)
                    .take_while(|i| *i < depth)
                    .map(|i| self.peekvalue_maybe_none(i))
                    .collect();
                let code_ptr = unsafe { crate::pyframe::pyframe_get_pycode(self) };
                let code_name = if code_ptr.is_null() {
                    "?"
                } else {
                    unsafe { (*code_ptr).obj_name.as_str() }
                };
                reraise_bad_operand_diag(w_exc, oparg, code_name, depth, &below);
            }
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
    fn load_from_dict_or_globals(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        let mapping = self.pop();
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        let anchor = FrameAnchor::new(self);
        match crate::baseobjspace::getitem(mapping, key) {
            Ok(value) => {
                return Self::push_anchored(&anchor, value);
            }
            Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {}
            Err(err) => return Err(err),
        }

        let value = unsafe { &mut *anchor.live() }.load_global_value(name, nameindex)?;
        Self::push_anchored(&anchor, value)
    }

    // ── LoadFromDictOrDeref ──
    // CPython 3.13: LOAD_FROM_DICT_OR_DEREF — used by the PEP 695 type-param
    // scope.  Pop the namespace mapping (TOS), try `mapping[name]`, then fall
    // back to the cell / free variable at `idx`.
    fn load_from_dict_or_deref(&mut self, idx: usize, name: &str) -> Result<(), PyError> {
        let mapping = self.pop();
        // A localsplus name has no `co_names_w` slot to realize into, so nothing
        // bounds how often this runs; interning is what keeps an immortal
        // string per execution from being an immortal string per execution.
        let key = pyre_object::unicodeobject::intern_str_value(name);
        let anchor = FrameAnchor::new(self);
        match crate::baseobjspace::getitem(mapping, key) {
            Ok(value) => {
                return Self::push_anchored(&anchor, value);
            }
            Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {}
            Err(err) => return Err(err),
        }
        // `getitem` may have run a `__getitem__`; everything below reads the
        // frame's own code object, globals and cells.
        let self_ = unsafe { &mut *anchor.live() };
        // CPython 3.14 addresses the outer `__class__` freevar in the
        // class-cell collision. The pinned compiler encodes the implicit
        // method cell instead; if no outer cell exists, it emits this opcode
        // where CPython uses LOAD_NAME, so preserve globals/builtins fallback.
        let deref_idx = if crate::pyframe::class_scope_class_deref_is_name(self_.code(), idx) {
            match crate::pyframe::class_scope_outer_class_freevar(self_.code()) {
                Some(free_idx) => free_idx,
                None => {
                    // This deref name has no `co_names` index, so do the
                    // uncached LOAD_NAME fallback directly rather than
                    // corrupting an unrelated per-code LOAD_GLOBAL cache slot.
                    let w_globals = self_.get_w_globals();
                    if !w_globals.is_null()
                        && let Some(value) = unsafe {
                            pyre_object::dictmultiobject::w_dict_getitem_str(w_globals, name)
                        }
                    {
                        return Self::push_anchored(&anchor, value);
                    }
                    // `self.get_builtin()`, not the `builtin` field — see
                    // `load_global_value`.
                    let w_builtin = self_.get_builtin();
                    if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
                        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
                        if !w_dict.is_null()
                            && let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)?
                        {
                            return Self::push_anchored(&anchor, value);
                        }
                    }
                    return Err(PyError::name_error_with_name(
                        format!("name '{name}' is not defined"),
                        name,
                    ));
                }
            }
        } else {
            idx
        };
        let self_ = unsafe { &mut *anchor.live() };
        let slot = locals_w!(self_)[deref_idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(crate::pyframe::deref_unbound_error(self_.code(), deref_idx));
        }
        Self::push_anchored(&anchor, value)
    }

    // ── GetLen ──
    fn get_len(&mut self, obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let len = crate::baseobjspace::len(obj)?;
        Ok(len)
    }

    // ── Pattern matching (PEP 634) ──
    // Each opcode is the stack shuffle around an `opcode_ops` value-level
    // helper; the JIT residuals (`bh_match_*_fn`) call the same helpers.
    fn match_mapping(&mut self) -> Result<(), PyError> {
        let subject = PyFrame::peek_at(self, 0);
        self.push(crate::opcode_ops::match_mapping_value(subject));
        Ok(())
    }

    fn match_sequence(&mut self) -> Result<(), PyError> {
        let subject = PyFrame::peek_at(self, 0);
        self.push(crate::opcode_ops::match_sequence_value(subject));
        Ok(())
    }

    // MATCH_KEYS: STACK[-1] = keys tuple, STACK[-2] = subject (neither popped).
    fn match_keys(&mut self) -> Result<(), PyError> {
        let keys = PyFrame::peek_at(self, 0);
        let subject = PyFrame::peek_at(self, 1);
        let anchor = FrameAnchor::new(self);
        let result = crate::opcode_ops::match_keys_value(subject, keys)?;
        Self::push_anchored(&anchor, result)
    }

    // MATCH_CLASS count: STACK[-1] = keyword attr-name tuple, STACK[-2] = class,
    // STACK[-3] = subject (all popped). Push the extracted-attrs tuple on a
    // match, else None. `count` is the number of positional sub-patterns.
    fn match_class(&mut self, count: usize) -> Result<(), PyError> {
        let kwd_attrs = self.pop();
        let cls = self.pop();
        let subject = self.pop();
        let anchor = FrameAnchor::new(self);
        let result = crate::opcode_ops::match_class_value(subject, cls, kwd_attrs, count)?;
        Self::push_anchored(&anchor, result)
    }

    // ── LoadFastAndClear (comprehension scope) ──
    fn load_fast_and_clear(&mut self, idx: usize) -> Result<(), PyError> {
        let val = locals_w!(self)[idx];
        self.push(val);
        self.set_locals_w(idx, PY_NULL);
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
        let anchor = FrameAnchor::new(self);
        let set_obj = crate::builtins::builtin_set_from_items(&items)?;
        Self::push_anchored(&anchor, set_obj)
    }

    // ── DictUpdate ──
    // pypy/interpreter/pyopcode.py:1524-1532 DICT_UPDATE — `space.ismapping_w`
    // gate then `dict.update(source)`. Non-mapping operand surfaces
    // "'<T>' object is not a mapping" (TypeError).
    fn dict_update(&mut self, i: usize) -> Result<(), PyError> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::dict_update_value(dict, source)
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
        crate::opcode_ops::dict_merge_value(dict, source, w_callable)
    }

    // ── MapAdd ──
    // PyPy: STORE_MAP/MAP_ADD; CPython: MAP_ADD
    // dict = STACK[-i-2]; dict[TOS1] = TOS; pop key+value
    fn map_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let key = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::map_add_value(dict, key, value)
    }

    // ── SetAdd ──
    // PyPy: SET_ADD; CPython: SET_ADD
    // set = STACK[-i]; set.add(TOS); pop value
    fn set_add(&mut self, i: usize) -> Result<(), PyError> {
        let value = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        crate::opcode_ops::set_add_value(set, value)
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
        crate::opcode_ops::list_to_tuple_value(val)
    }

    fn async_gen_wrap(&mut self, val: PyObjectRef) -> Result<PyObjectRef, PyError> {
        Ok(pyre_object::generator::w_async_gen_value_wrapper_new(val))
    }

    // ── print_expr ──
    // PRINT_EXPR → sys.displayhook(value). Routing through the live hook lets
    // a rebound displayhook (doctest, IDLE) and a redirected sys.stdout take
    // effect instead of writing straight to the native stream.
    fn print_expr(&mut self, val: PyObjectRef) -> Result<(), PyError> {
        if let Some(sys_mod) = crate::importing::get_sys_module("sys") {
            match crate::baseobjspace::getattr_str(sys_mod, "displayhook") {
                Ok(hook) => {
                    let r = crate::call_function(hook, &[val]);
                    if r.is_null() {
                        return Err(crate::call::take_call_error().unwrap_or_else(|| {
                            PyError::runtime_error("displayhook raised an exception")
                        }));
                    }
                    return Ok(());
                }
                Err(e) if e.kind == PyErrorKind::AttributeError => {
                    return Err(PyError::runtime_error("lost sys.displayhook"));
                }
                Err(e) => return Err(e),
            }
        }
        // No `sys` yet (early bootstrap) — native repr print.
        if !unsafe { pyre_object::is_none(val) } {
            let s = unsafe { crate::display::py_repr_wtf8(val)? };
            crate::host_seam::emit_stdout(crate::display::wtf8_format!(s, "\n").as_bytes());
        }
        Ok(())
    }

    // ── delete_name ──
    // pypy/interpreter/pyopcode.py:821 DELETE_NAME — delete from w_locals; KeyError → NameError.
    fn delete_name(&mut self, name: &str, nameindex: usize) -> Result<(), PyError> {
        // `space.delitem(w_locals, w_name)`; at module scope `w_locals` is the
        // globals dict, so a module DELETE_NAME routes through the canonical
        // W_DictObject too.  KeyError → NameError.
        let w_locals = self.get_or_create_w_locals();
        let key = unsafe {
            crate::pycode::w_code_getname_w_or_new(self.pycode as PyObjectRef, nameindex, name)
        };
        crate::baseobjspace::delitem(w_locals, key).map_err(|err| {
            if matches!(err.kind, PyErrorKind::KeyError) {
                PyError::name_error_with_name(format!("name '{name}' is not defined"), name)
            } else {
                err
            }
        })?;
        Ok(())
    }

    // ── delete_global ──
    // pypy/interpreter/pyopcode.py:901-903 DELETE_GLOBAL —
    //   `self.space.delitem(self.get_w_globals_storage(), w_varname)`.
    // CPython/PyPy dict deletion uses the dict's intrinsic strategy and does
    // not invoke a dict subclass's Python-level __delitem__.  Resolve pyre's
    // composed dict-subclass backing first, then use that same strategy path.
    fn delete_global(&mut self, name: &str) -> Result<(), PyError> {
        let w_globals = self.get_w_globals();
        let backing = unsafe { crate::type_methods::resolve_dict_backing(w_globals) };
        let found = !backing.is_null() && unsafe { pyre_object::w_dict_delitem_str(backing, name) };
        if !found {
            return Err(PyError::key_error(format!("'{name}'")));
        }
        Ok(())
    }

    // ── import_star ──
    // IMPORT_STAR — merge the module's public names into the locals
    // mapping (class body / exec-with-locals), not globals:
    //     w_locals = self.getdictscope()
    //     import_all_from(self.space, w_module, w_locals)
    //     self.setdictscope(w_locals)
    // `getdictscope` runs fast2locals so the mapping reflects the live
    // fast locals; `import_all_from_w` lands each `from module import *`
    // entry via `space.setitem(w_locals, name, value)` rather than the
    // `*mut DictStorage` fast path; `setdictscope` runs locals2fast to
    // write the merged mapping back into the frame's fast locals.
    fn import_star(&mut self) -> Result<(), PyError> {
        let module = self.pop();
        let w_locals = self.getdictscope()?;
        crate::importing::import_all_from_w(module, w_locals)?;
        self.setdictscope(w_locals)?;
        Ok(())
    }

    // ── load_build_class ──
    // PyPy pyopcode.py:866-870 LOAD_BUILD_CLASS reads
    // `self.get_builtin().getdictvalue('__build_class__')`.  Python 3.14
    // reports a NameError when the selected builtin mapping has no entry.
    fn load_build_class(&mut self) -> Result<(), PyError> {
        let anchor = FrameAnchor::new(self);
        let bc = self.load_build_class_value()?;
        Self::push_anchored(&anchor, bc)
    }

    fn load_build_class_value(&mut self) -> Result<PyObjectRef, PyError> {
        let w_builtin = self.get_builtin();
        let bc = if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
            if w_dict.is_null() {
                None
            } else {
                crate::baseobjspace::finditem_str(w_dict, "__build_class__")?
            }
        } else {
            None
        };
        let Some(bc) = bc else {
            return Err(PyError::name_error_with_name(
                "__build_class__ not found",
                "__build_class__",
            ));
        };
        Ok(bc)
    }

    // ── yield from / send ──
    fn get_yield_from_iter(&mut self) -> Result<(), PyError> {
        let iterable = self.pop();
        let anchor = FrameAnchor::new(self);
        // CPython 3.14 `GET_YIELD_FROM_ITER` / PyPy's coroutine-aware
        // `YIELD_FROM`: exact generators already are their iterator.  A
        // native coroutine is also sent to directly, but only when the
        // current frame is itself a coroutine or was marked by
        // `types.coroutine` with CO_ITERABLE_COROUTINE.  Calling ordinary
        // `iter()` here loses both halves of that distinction because native
        // coroutine objects intentionally expose no public `__iter__`.
        let iter = unsafe {
            if pyre_object::generator::is_coroutine(iterable) {
                let flags = self.code().flags;
                if !flags
                    .intersects(crate::CodeFlags::COROUTINE | crate::CodeFlags::ITERABLE_COROUTINE)
                {
                    return Err(PyError::type_error(
                        "cannot 'yield from' a coroutine object in a non-coroutine generator",
                    ));
                }
                iterable
            } else if pyre_object::generator::is_generator(iterable) {
                iterable
            } else {
                crate::baseobjspace::iter(iterable)?
            }
        };
        Self::push_anchored(&anchor, iter)
    }

    fn send_value(&mut self, target: usize) -> Result<(), PyError> {
        let value = self.pop();
        let iter = self.peek();
        let anchor = FrameAnchor::new(self);
        let result = if unsafe { pyre_object::is_none(value) } {
            // generator.py / pyopcode.py `next_yield_from`: coroutine
            // objects are not public iterators, but the interpreter's SEND
            // machinery resumes both GeneratorIterator and Coroutine through
            // their shared `send_ex(None)` path.
            if unsafe { pyre_object::generator::is_generator_or_coroutine(iter) } {
                crate::baseobjspace::generator_next_method(&[iter])
            } else {
                crate::baseobjspace::next(iter)
            }
        } else {
            let send = crate::baseobjspace::getattr_str(iter, "send")?;
            crate::call::call_function_impl_result(send, &[value])
        };
        match result {
            Ok(result) => {
                let frame = unsafe { &mut *anchor.live() };
                frame.w_yielding_from = iter;
                if pyre_object::gc_hook::try_gc_owns_object(frame as *mut PyFrame as *mut u8) {
                    pyre_object::gc_hook::try_gc_write_barrier(frame as *mut PyFrame as *mut u8);
                }
                Self::push_anchored(&anchor, result)
            }
            Err(e) if e.matches_stop_iteration() => {
                let frame = unsafe { &mut *anchor.live() };
                if std::ptr::eq(frame.w_yielding_from, iter) {
                    frame.w_yielding_from = pyre_object::PY_NULL;
                }
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
                Self::push_anchored(&anchor, value)?;
                unsafe { &mut *anchor.live() }.set_last_instr_from_next_instr(target);
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

    fn get_awaitable(&mut self, context: u32) -> Result<(), PyError> {
        // pyopcode.py:1599 GET_AWAITABLE.
        let w_iterable = self.pop();
        let anchor = FrameAnchor::new(self);
        let w_iter = crate::baseobjspace::get_awaitable_iter(w_iterable, context)?;
        // pyopcode.py:1604 guards a coroutine that is already being awaited
        // (`w_iter.get_delegate() is not None`) with RuntimeError.  pyre's
        // generator object has no delegate / `w_yielded_from` field, so the
        // reentrant-await case is instead caught at SEND by the generator
        // `running` flag.
        Self::push_anchored(&anchor, w_iter)
    }

    fn get_aiter(&mut self) -> Result<(), PyError> {
        let obj = self.pop();
        let anchor = FrameAnchor::new(self);
        let method =
            unsafe { crate::baseobjspace::lookup_special(obj, "__aiter__")? }.ok_or_else(|| {
                crate::PyError::type_error(format!(
                    "'async for' requires an object with __aiter__ method, got {}",
                    crate::type_methods::arg_type_name(obj)
                ))
            })?;
        let iter = crate::call::call_function_impl_result(method, &[])?;
        if unsafe { crate::baseobjspace::lookup_special(iter, "__anext__")? }.is_none() {
            return Err(crate::PyError::type_error(format!(
                "'async for' received an object from __aiter__ that does not implement __anext__: {}",
                crate::type_methods::arg_type_name(iter)
            )));
        }
        Self::push_anchored(&anchor, iter)
    }

    fn get_anext(&mut self) -> Result<(), PyError> {
        let iter = self.peek();
        let anchor = FrameAnchor::new(self);
        let method = unsafe { crate::baseobjspace::lookup_special(iter, "__anext__")? }
            .ok_or_else(|| {
                crate::PyError::type_error(format!(
                    "'async for' requires an iterator with __anext__ method, got {}",
                    crate::type_methods::arg_type_name(iter)
                ))
            })?;
        let next = crate::call::call_function_impl_result(method, &[])?;
        let awaitable = crate::baseobjspace::get_awaitable_iter(next, 0).map_err(|mut cause| {
            // CPython 3.14 `_PyEval_GetANext` uses
            // `_PyErr_FormatFromCause` for *every* failure produced while
            // converting `__anext__`'s result to an awaitable.  In
            // particular, an exception raised by `result.__await__()` is the
            // explicit cause of this TypeError; only an exception raised by
            // `__anext__` itself propagates unchanged above.
            let message = format!(
                "'async for' received an invalid object from __anext__: {}",
                crate::type_methods::arg_type_name(next)
            );
            let cause_obj = cause.to_exc_object();
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(cause_obj);
            let cause_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let mut error = crate::PyError::type_error(message);
            let error_obj = error.to_exc_object();
            let cause_obj = pyre_object::gc_roots::shadow_stack_get(cause_slot);
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_context(error_obj, cause_obj);
                pyre_object::interp_exceptions::w_exception_set_cause(error_obj, cause_obj);
                pyre_object::interp_exceptions::w_exception_set_suppress_context(error_obj, true);
                crate::PyError::from_exc_object(error_obj)
            }
        })?;
        Self::push_anchored(&anchor, awaitable)
    }

    fn end_async_for(&mut self) -> Result<(), PyError> {
        let exc = self.pop();
        let err = unsafe { crate::PyError::from_exc_object(exc) };
        if err.kind == crate::PyErrorKind::StopAsyncIteration {
            // The 3.14 exception table preserves the surrounding handler's
            // previous-exception entry below the asynchronous iterator; the
            // StopAsyncIteration edge itself pushes only `exc`.
            self.pop(); // asynchronous iterator
            Ok(())
        } else {
            Err(err)
        }
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
    fn load_super_attr_with(
        &mut self,
        name: &str,
        is_method: bool,
        is_two_arg: bool,
    ) -> Result<(), PyError> {
        let self_obj = self.pop();
        let cls = self.pop();
        let global_super = self.pop();
        let anchor = FrameAnchor::new(self);

        // CPython 3.14 `LOAD_SUPER_ATTR`: the callable loaded from globals is
        // authoritative (it may shadow builtins.super).  Bit 1 distinguishes
        // `super(type, obj)` from zero-argument `super()`; `cls` / `self_obj`
        // are stack operands for the fast builtin case but are not arguments
        // to a shadowing zero-arg callable.
        let proxy = if is_two_arg {
            crate::call::call_function_impl_result(global_super, &[cls, self_obj])?
        } else {
            crate::call::call_function_impl_result(global_super, &[])?
        };
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
                Self::push_anchored(&anchor, func)?;
                Self::push_anchored(&anchor, recv)?;
            } else {
                // staticmethod or classmethod — no self binding
                Self::push_anchored(&anchor, result)?;
                Self::push_anchored(&anchor, PY_NULL)?;
            }
        } else {
            // is_method=false: getattr already returned a bound method.
            Self::push_anchored(&anchor, result)?;
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
        // `__getattribute__` allocates: the receiver is popped, so nothing but
        // this local still reaches it, and the same collection relocates a
        // JIT-created frame.  Pin the receiver and push onto the forwarded
        // live frame.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        roots.pin_root(obj);
        let anchor = FrameAnchor::new(self);
        let attr = crate::baseobjspace::getattr_str(obj, name)?;
        let obj = roots.get(obj_slot);
        // LOOKUP_METHOD pushes (attr, null_or_self): the resolved attribute
        // first, then the bound receiver computed by the shared, side-effect
        // free binding decision (NULL when no self should be prepended).
        let bound = compute_load_method_bound(obj, attr, name);
        let live = unsafe { &mut *anchor.live() };
        live.push(attr);
        live.push(bound);
        Ok(())
    }

    fn load_special(&mut self, name: &str) -> Result<(), PyError> {
        let obj = self.pop();
        // The descriptor `__get__` allocates and can relocate a JIT-created
        // frame; push onto the forwarded live frame.
        let anchor = FrameAnchor::new(self);
        let bound = crate::baseobjspace::load_special_resolve(obj, name)?;
        let live = unsafe { &mut *anchor.live() };
        live.push(bound);
        live.push(PY_NULL);
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
        // Graceful underflow (`shared_opcode.rs`'s `opcode_load_attr` →
        // `pop_value()?`): a corrupted concrete-execution stack during
        // trace recording (e.g. a residual call the inline executor
        // could not perform) aborts the trace instead of panicking the
        // hard-asserting `pop()`.
        let obj = self.pop_value()?;
        // Popped, so nothing but this local still reaches the receiver across
        // the cache lookup, which runs arbitrary Python on a cache miss.
        let roots = pyre_object::gc_roots::push_roots();
        let obj_slot = roots.base();
        roots.pin_root(obj);
        let pycode = self.pycode as PyObjectRef;
        let anchor = FrameAnchor::new(self);
        let w_value = unsafe {
            crate::objspace::std::mapdict::load_attr_caching(pycode, obj, nameindex, name)
        }
        .map_err(|error| {
            if finalize_failed_attr_receiver_now(roots.get(obj_slot)) {
                unsafe { &mut *anchor.live() }.defer_failed_attr_until_pop_except();
            }
            error
        })?;
        Self::push_anchored(&anchor, w_value)
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
        // Graceful underflow like `opcode_store_attr` (`shared_opcode.rs`)
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
        // baseobjspace.py:1243-1266 fast path: Function, including the
        // CALL_METHOD form.  callmethod.py:85-94 counts a non-null `self` as
        // one extra argument while `dropvalues` remains the physical
        // `[callable, null_or_self, explicit args...]` width.  This is what
        // lets the translated interpreter expose an ordinary `_flat_pycall`
        // to the meta-tracer for `obj.method(...)`, just like PyPy.
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
            let mut null_or_self = self.peekvalue_maybe_none(nargs);
            let mut callable = self.peekvalue_maybe_none(nargs + 1);
            // baseobjspace.py:1254-1259: `_Method` is not a generic callable
            // here.  Reuse its null/self stack slot for `w_instance`, unwrap
            // `w_function`, and continue through the identical Function
            // valuestack path.  Module aliases such as `random.gauss =
            // _inst.gauss` depend on this just as direct `obj.method()` calls
            // do; allocating an Arguments Vec for every alias call diverges
            // from PyPy's meta-traced interpreter shape.
            if !callable.is_null()
                && null_or_self.is_null()
                && unsafe { pyre_object::is_method(callable) }
            {
                let receiver = unsafe { pyre_object::w_method_get_self(callable) };
                let function = unsafe { pyre_object::w_method_get_func(callable) };
                if !receiver.is_null()
                    && !function.is_null()
                    && unsafe { crate::is_function(function) }
                {
                    self.settopvalue(receiver, nargs);
                    null_or_self = receiver;
                    callable = function;
                }
            }
            if !callable.is_null() && unsafe { crate::is_function(callable) } {
                let methodcall = !null_or_self.is_null();
                let call_nargs = nargs + usize::from(methodcall);
                let anchor = FrameAnchor::new(self);
                let result = crate::function::funccall_valuestack(
                    callable,
                    call_nargs,
                    self,
                    nargs + 2,
                    methodcall,
                );
                if result.is_null() {
                    return Err(crate::call::take_call_error()
                        .unwrap_or_else(|| crate::PyError::type_error("call failed")));
                }
                // baseobjspace.py:1256 self.pushvalue(w_result). The callee may
                // have relocated this frame via a minor collection, so push
                // onto the forwarded live frame, not the pre-call pointer.
                unsafe { &mut *anchor.live() }.push(result);
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

        let anchor = FrameAnchor::new(self);
        let result = if null_or_self.is_null() {
            call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            call_callable(self, callable, &full_args)?
        };
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
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
        let self_or_null = self.pop();
        let callable = self.pop();
        let anchor = FrameAnchor::new(self);
        let result =
            crate::call::call_function_ex(self, callable, self_or_null, args_obj, kwargs_or_null)?;
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
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

        let anchor = FrameAnchor::new(self);
        let result = crate::call::call_kw(self, callable, self_or_null, &args, kwarg_names)?;
        // The callee may have relocated this frame via a minor collection;
        // push onto the forwarded live frame, not the pre-call pointer.
        unsafe { &mut *anchor.live() }.push(result);
        Ok(())
    }

    // ── load_locals ──
    // PyPy: LOAD_LOCALS; CPython: LOAD_LOCALS
    // Pushes the current namespace dict onto the stack.
    fn load_locals(&mut self) -> Result<(), PyError> {
        // pyopcode.py:793-794:
        //   self.pushvalue(self.getorcreatedebug().w_locals)
        // In particular this must preserve a metaclass's custom `__prepare__`
        // mapping.  Rebuilding a plain dict loses mapping-subclass semantics;
        // treating a dict subclass as a non-dict can also accidentally copy
        // globals into the class namespace and make LOAD_FROM_DICT_OR_DEREF
        // select a global over its closure cell.
        let anchor = FrameAnchor::new(self);
        let w_locals = self.get_or_create_w_locals();
        Self::push_anchored(&anchor, w_locals)
    }

    // ── unpack_ex ──
    // PyPy: UNPACK_SEQUENCE with star; CPython: UNPACK_EX
    // `a, *b, c = iterable`
    fn unpack_ex(&mut self, args: crate::bytecode::UnpackExArgs) -> Result<(), PyError> {
        let before = args.before as usize;
        let after = args.after as usize;
        let value = self.pop();
        // `unpack_ex_slots` returns the `before + 1 + after` slots in TOS
        // order (head items, starred list, tail items); push bottom-first so
        // the first head item ends on top.
        let anchor = FrameAnchor::new(self);
        let slots = crate::runtime_ops::unpack_ex_slots(before, after, value)?;
        for item in slots.into_iter().rev() {
            Self::push_anchored(&anchor, item)?;
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
        crate::opcode_ops::set_update_value(set, iterable)
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
        let anchor = FrameAnchor::new(self);
        let result = crate::runtime_ops::binary_slice_values(obj, start, stop)?;
        Self::push_anchored(&anchor, result)
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
    use crate::*;

    // Language-level behaviour belongs in `pyre/extra_tests/snippets`, which
    // runs every script under CPython as well as both backends. What is left
    // here either reaches a Rust entry point no Python program can call, or
    // pins a value pyre and CPython 3.14 disagree about — a snippet asserting
    // the latter would have to fail on one of its own runners. Each test says
    // which of the two it is.

    fn run_exec_frame(source: &str) -> (PyResult, crate::pyframe::FrameBox) {
        // Module globals are now a celldict whose str keys hash through the
        // `hash_w` trampoline (production installs it before the first frame
        // via `init_jit_hooks`); mirror that here so frame construction can
        // seed the builtins.
        crate::test_hooks::install_hash_hook();
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let result = frame.execute_frame(None, None);
        (result, frame)
    }

    // A frame value slot may hold a `JitVirtualRef`, whose leading magic word
    // would be read as a type pointer by the PyObject-shaped raw walks. Keep
    // the virtual-ref branch returning after it visits the `forced` field.
    #[test]
    fn test_frame_value_slot_holding_a_virtual_ref_skips_the_pyobject_walks() {
        use majit_metainterp::virtualref::{JIT_VIRTUAL_REF_VTABLE, JitVirtualRef, ObjectHeader};

        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: std::ptr::null_mut(),
            forced: 0x1000usize as *mut u8,
        };
        let mut slot = majit_ir::GcRef(&mut vref as *mut JitVirtualRef as usize);
        let slot_address = &mut slot as *mut majit_ir::GcRef as usize;
        let forced_address = std::ptr::addr_of_mut!(vref.forced) as usize;
        let mut visited = Vec::new();

        unsafe {
            walk_frame_value_slot(&mut slot, &mut |root| {
                visited.push(root as *mut majit_ir::GcRef as usize);
            });
        }

        assert_eq!(visited, vec![slot_address, forced_address]);
    }

    // `exception_is_valid_obj_as_class_w` is a Rust predicate the raise path
    // consults; no Python expression evaluates it on its own.
    #[test]
    fn test_exception_is_valid_obj_as_class_w_matches_baseexception_subclass_rule() {
        let (_result, frame) = run_exec_frame("good = ValueError\nbad = int");
        let w_globals = frame.get_w_globals();
        let good =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "good") }.expect("missing good");
        let bad =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "bad") }.expect("missing bad");

        unsafe {
            assert!(crate::baseobjspace::exception_is_valid_obj_as_class_w(good));
            assert!(!crate::baseobjspace::exception_is_valid_obj_as_class_w(bad));
        }
    }

    // A lazy `PyError` is a Rust-side value; Python only ever sees the
    // instance `to_exc_object()` already materialised.
    #[test]
    fn test_to_exc_object_memoizes_lazy_exception() {
        // Bring up the managed heap / builtin exception types by running a
        // real frame first; `to_exc_object()` allocates the instance.
        let _ = run_exec_frame("pass");
        // A raw-message NameError is lazy: `exc_object` stays null until the
        // first `to_exc_object()` materialises it. The write-once memo
        // (`get_w_value`, error.py:349) must then return that same instance on
        // every later call instead of allocating a fresh one.
        let mut err = PyError::name_error_with_name("name 'x' is not defined", "x");
        assert!(err.exc_object.is_null(), "raw-message error starts lazy");
        let first = err.to_exc_object();
        assert!(!first.is_null());
        assert_eq!(err.exc_object, first, "materialised instance is memoised");
        let second = err.to_exc_object();
        assert_eq!(first, second, "second call returns the memoised instance");
    }

    // The corrupted chain is written with `replace_op` after compilation; no
    // Python source the compiler accepts produces this bytecode.
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
        assert_eq!(err.message_text(), "bytecode corruption");
    }

    // `check_exc_match_against` is called here directly: the residual
    // `bh_compare_fn` reaches it on a path an `except` clause cannot select.
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
        let w_globals = frame.get_w_globals();
        let exc =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "exc") }.expect("missing exc");
        let plain =
            unsafe { pyre_object::w_dict_getitem_str(w_globals, "plain") }.expect("missing plain");
        let value_error = unsafe { pyre_object::w_dict_getitem_str(w_globals, "value_error") }
            .expect("missing value_error");
        let type_error = unsafe { pyre_object::w_dict_getitem_str(w_globals, "type_error") }
            .expect("missing type_error");

        assert!(check_exc_match_against(exc, value_error));
        assert!(!check_exc_match_against(exc, type_error));
        assert!(!check_exc_match_against(plain, value_error));
    }

    #[test]
    fn test_pyerror_matches_stop_iteration_uses_exception_mro() {
        let (result, frame) =
            run_exec_frame("class VS(ValueError, StopIteration):\n    pass\nexc = VS('done')");
        result.expect("exception subclass setup failed");
        let exc = unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), "exc") }
            .expect("missing exc");
        let err = unsafe { PyError::from_exc_object(exc) };

        assert_eq!(err.kind, PyErrorKind::ValueError);
        assert!(err.matches_stop_iteration());
        assert!(PyError::stop_iteration().matches_stop_iteration());
        assert!(!PyError::value_error("not exhausted").matches_stop_iteration());
    }

    // pyre materialises the rich-compare and `__iter__` rows in
    // `UnionType.__dict__`; CPython 3.14 leaves them to the slot table, so
    // `required <= UT.__dict__.keys()` is false there. Divergent by design,
    // hence not a snippet.
    #[test]
    fn test_union_type_exposes_cpython_314_richcompare_surface() {
        let source = r#"
UT = type(int | str)
required = {
    '__getattribute__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
    '__name__', '__qualname__', '__origin__', '__iter__', '__doc__',
}
assert required <= UT.__dict__.keys()
u = int | str
assert UT.__getattribute__(u, '__args__') == (int, str)
assert u.__name__ == 'Union'
assert u.__qualname__ == 'Union'
assert u.__origin__ is UT
assert u.__module__ == 'typing'
assert UT.__dict__['__iter__'] is None
try:
    iter(u)
except TypeError:
    pass
else:
    raise AssertionError('UnionType must disable __getitem__ iteration fallback')
assert UT.__ne__(u, int | str) is False
assert UT.__ne__(u, int | bytes) is True
assert UT.__ne__(u, int) is NotImplemented
assert UT.__lt__(u, int | bytes) is NotImplemented
assert UT.__le__(u, int | bytes) is NotImplemented
assert UT.__gt__(u, int | bytes) is NotImplemented
assert UT.__ge__(u, int | bytes) is NotImplemented
"#;
        let (result, _frame) = run_exec_frame(source);
        result.expect("UnionType explicit rich-compare surface failed");
    }

    // pyre spells the type in this AttributeError with its module prefix
    // (`__main__.make_type.<locals>.X`); CPython 3.14 prints the bare
    // qualname. Divergent by design, hence not a snippet.
    #[test]
    fn test_empty_member_slot_error_uses_fully_qualified_type_name() {
        let source = "\
def make_type():
    class X:
        __slots__ = 'a'
    return X
X = make_type()
try:
    X().a
except AttributeError as exc:
    result = str(exc)";
        let (res, frame) = run_exec_frame(source);
        res.expect("member slot AttributeError regression");
        unsafe {
            let value = w_dict_getitem_str(frame.w_globals, "result").unwrap();
            assert_eq!(
                w_str_get_wtf8(value).as_str(),
                Ok("'__main__.make_type.<locals>.X' object has no attribute 'a'")
            );
        }
    }

    #[test]
    // `sort_seen` pins what `list.__sizeof__()` reports while `sort` holds the
    // receiver: pyre answers 56, CPython 3.14 answers 32 because it detaches
    // `ob_item` for the duration. Every other clause here matches CPython, but
    // that one keeps the whole fixture off the snippet runner.
    fn test_list_cpython_allocation_and_sizeof() {
        let source = "\
def allocation(value):
    return (value.__sizeof__() - type(value).__basicsize__) // 8

value = []
append_allocations = []
for item in range(20):
    value.append(item)
    append_allocations.append(allocation(value))

sources = (
    list([1, 2, 3, 4, 5]),
    list((1, 2, 3, 4, 5)),
    list(range(5)),
    list(item for item in range(5)),
    list(dict.fromkeys(range(5))),
    list(set(range(5))),
    list(dict.fromkeys(range(5)).keys()),
    list(dict.fromkeys(range(5)).values()),
    list(dict.fromkeys(range(5)).items()),
)

class HintTwenty:
    def __iter__(self):
        return iter(range(3))
    def __len__(self):
        return 20

sorted_value = [3, 2, 1]
sort_seen = []
def sort_key(item):
    sort_seen.append((len(sorted_value), sorted_value.__sizeof__()))
    return item
sorted_value.sort(key=sort_key)

noop_sorted = [3, 2, 1]
def noop_key(item):
    noop_sorted.clear()
    del noop_sorted[:]
    noop_sorted[:] = []
    return item
noop_sorted.sort(key=noop_key)

mucked_sorted = [3, 2, 1]
def mucked_key(item):
    if item == 3:
        mucked_sorted.append(9)
        mucked_sorted.pop()
    return item
try:
    mucked_sorted.sort(key=mucked_key)
except ValueError as exc:
    mucked_detected = str(exc) == 'list modified during sort'
else:
    mucked_detected = False

class Sub(list):
    pass
sub = Sub(range(3))

class HugeHint:
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration
    def __length_hint__(self):
        return (1 << 63) - 1

nonempty = [1]
try:
    nonempty.extend(HugeHint())
except MemoryError:
    ignored_overflowing_hint = False
else:
    ignored_overflowing_hint = nonempty == [1]
try:
    list(HugeHint())
except MemoryError:
    empty_huge_hint_fails = True
else:
    empty_huge_hint_fails = False

result = (
    append_allocations == [4, 4, 4, 4, 8, 8, 8, 8,
                           16, 16, 16, 16, 16, 16, 16, 16,
                           24, 24, 24, 24]
    and [allocation(item) for item in sources] == [6, 6, 6, 8, 8, 8, 8, 8, 8]
    and allocation(list(HintTwenty())) == 8
    and sort_seen == [(0, 56), (0, 56), (0, 56)]
    and sorted_value == [1, 2, 3]
    and allocation(sorted_value) == 4
    and noop_sorted == [1, 2, 3]
    and mucked_sorted == [1, 2, 3]
    and mucked_detected
    and ignored_overflowing_hint
    and empty_huge_hint_fails
    and sub.__sizeof__() == Sub.__basicsize__ + 4 * 8
)
";
        let (res, frame) = run_exec_frame(source);
        res.expect("CPython list allocation metadata failed");
        unsafe {
            let result = w_dict_getitem_str(frame.w_globals, "result").unwrap();
            assert!(crate::baseobjspace::is_true(result).unwrap());
        }
    }
}
