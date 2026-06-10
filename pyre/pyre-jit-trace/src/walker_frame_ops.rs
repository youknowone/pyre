//! `WalkerFrameOps` trait — abstraction over the small surface of
//! `MIFrame` methods that the strategy-aware STORE_SUBSCR specialization
//! emits.  The trait lets the shared strategy helpers
//! (`generated_list_setitem_by_strategy`,
//! `generated_list_setslice_same_len_by_strategy`,
//! `generated_store_subscr_value`, and the `store_subscr_value` body in
//! `trace_opcode.rs`) run against either `MIFrame` or the walker
//! `WalkContext`, so both dispatch paths emit the same
//! `guard_class`+`SETARRAYITEM_GC`-family shape.
//!
//! ## Trait shape — `self`-only signatures, `ctx` reached via accessor
//!
//! Earlier draft passed `ctx: &mut TraceCtx` as an explicit method
//! argument alongside `&mut self`.  That shape forces the caller to
//! split-borrow `self.trace_ctx` from `self`, which is illegal whenever
//! the same `TraceCtx` is reached transitively through both `self` and
//! the explicit arg (always true for `MIFrame`'s `self.ctx: *mut TraceCtx`
//! and `WalkContext`'s `self.trace_ctx: &'frame mut TraceCtx`).
//!
//! The current shape — `&mut self` only, `ctx` reached through
//! `self.ctx_mut()` / `self.ctx()` accessor methods — lets each impl
//! choose how to materialise the borrow (`MIFrame` does `unsafe { &mut
//! *self.ctx }`; `WalkContext` returns its `trace_ctx` field).  The
//! accessor is scoped to a single statement at each callsite so the
//! borrow ends before the next `self`-mut-call, satisfying the borrow
//! checker without nested-borrow gymnastics.
//!
//! ## Why a trait (not free functions)
//!
//! Five of the six methods are pure compositions over `ctx_mut` +
//! `generate_guard` and live as default impls below.  The lone
//! load-bearing method is `generate_guard`, which on `MIFrame` walks
//! `parent_frames`, `flush_to_frame_for_guard`,
//! `get_list_of_active_boxes`, `build_framestack_snapshot`, etc.  The
//! walker has the same semantic responsibility (capture a multi-frame
//! resume snapshot before recording the guard op) but reaches it
//! through `walker_capture_snapshot_for_last_guard` and the
//! dispatch-time `WalkContext` register banks instead of `MIFrame`'s
//! state.  Two distinct implementations are unavoidable; a trait makes
//! the two impls interchangeable at the `generated_*` call sites.
//!
//! ## Trait scope
//!
//! Only the methods reached by the STORE_SUBSCR specialization closure
//! are members.  Other `MIFrame` methods (`guard_nonnull`,
//! `trace_dynamic_list_index`, ...) stay where they are until a walker
//! specialization needs the same surface.
//!
//! `MIFrame` and `WalkContext` both delegate `generate_guard` to their
//! existing infrastructure (`MIFrame::generate_guard` and
//! `walker_capture_snapshot_for_last_guard`, respectively).  The
//! generated strategy helpers are generic over this trait, so the
//! residual-call walker specialization can reuse the same IR shape as the
//! trait-dispatch path.

use majit_ir::{OpCode, OpRef, Type};
use majit_metainterp::TraceCtx;
use pyre_object::PyType;

/// Surface used by the strategy-aware STORE_SUBSCR specialization.  See
/// module doc for the rationale.
pub trait WalkerFrameOps {
    /// Mut access to the embedded `TraceCtx`.  Each impl materialises
    /// the borrow from its own internal storage (`MIFrame`'s
    /// `self.ctx: *mut TraceCtx` raw pointer deref; `WalkContext`'s
    /// `self.trace_ctx: &'frame mut TraceCtx` field).
    fn ctx_mut(&mut self) -> &mut TraceCtx;

    /// Immutable counterpart of [`ctx_mut`].  Used by `value_type`'s
    /// `get_opref_type` lookup which doesn't need write access.
    fn ctx(&self) -> &TraceCtx;

    /// `pyjitpl.py:177-220` `Box.type` parity — return the OpRef's
    /// intrinsic type (Const kind, recorded result_type, or PtrInfo
    /// virtualized).  Default impl reads `ctx.get_opref_type`; impls
    /// override only if they carry a faster cached lookup.
    fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        self.ctx().get_opref_type(value).unwrap_or(Type::Ref)
    }

    /// `pyjitpl.py:2558-2602` `generate_guard` parity — flush a pending
    /// quasi-immut guard, capture multi-frame resume snapshot, then
    /// record the guard op with its snapshot.  The single load-bearing
    /// method; impls diverge between `MIFrame` (trait dispatch frame
    /// state) and `WalkContext` (walker register banks + dispatch
    /// snapshot helper).
    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]);

    /// `pyjitpl.py:3508-3514` `implement_guard_value` parity — pick the
    /// const factory (`const_ref` for Type::Ref, `const_int`
    /// otherwise), record `GUARD_VALUE`, then update the heapcache's
    /// box replacement so downstream reads see the proved constant.
    fn implement_guard_value(&mut self, value: OpRef, expected: i64) {
        let ty = self.value_type(value);
        let expected_ref = match ty {
            Type::Ref => self.ctx_mut().const_ref(expected),
            _ => self.ctx_mut().const_int(expected),
        };
        self.generate_guard(OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py:3512 `replace_box` parity.
        self.ctx_mut()
            .heap_cache_mut()
            .replace_box(value, expected_ref);
    }

    /// `pyjitpl.py:1518-1523` `opimpl_guard_class` parity.  Skips the
    /// guard when the heapcache already knows the class or the OpRef
    /// is constant (the runtime type is already pinned).  Otherwise
    /// records `GUARD_NONNULL_CLASS` with `expected_type_const` and
    /// updates the heapcache's class+nullity record.
    fn guard_class(&mut self, obj: OpRef, expected_type: *const PyType) {
        if self.ctx().heap_cache().is_class_known(obj) {
            return;
        }
        if obj.is_constant() {
            // The const-arm `flush_guard_not_invalidated` in
            // `MIFrame::guard_class` (trace_opcode.rs:4681) is skipped
            // here.  It only fires when a prior quasi-immut field read
            // set `pending_guard_not_invalidated_pc`; the
            // `store_subscr_value` precondition (concrete obj/key/value
            // are direct stack reads, not quasi-immut loads) excludes
            // that path on the walker leg.
            self.ctx_mut()
                .heap_cache_mut()
                .class_now_known(obj, expected_type as usize as i64);
            return;
        }
        let expected_type_const = self.ctx_mut().const_int(expected_type as usize as i64);
        self.generate_guard(OpCode::GuardNonnullClass, &[obj, expected_type_const]);
        // heapcache.py:470-473 `class_now_known` parity.
        self.ctx_mut()
            .heap_cache_mut()
            .class_now_known(obj, expected_type as usize as i64);
    }

    /// `intobject.py` `int_intval`-pattern guard — class-guard the obj
    /// as `INT_TYPE`, read its `intval` field, then `implement_guard_value`
    /// the unboxed payload against `expected`.
    fn guard_int_object_value(&mut self, int_obj: OpRef, expected: i64) {
        self.guard_class(int_obj, &pyre_object::pyobject::INT_TYPE as *const PyType);
        let actual_value = crate::state::opimpl_getfield_gc_i(
            self.ctx_mut(),
            int_obj,
            crate::descr::int_intval_descr(),
        );
        self.implement_guard_value(actual_value, expected);
    }

    /// `listobject.py` strategy field guard — `getfield_gc_i(strategy)`
    /// then `implement_guard_value`.  Skips runtime W_ListObject layout
    /// reasoning by reading the strategy id directly from its descr.
    fn guard_list_strategy(&mut self, obj: OpRef, expected: i64) {
        let strategy = crate::state::opimpl_getfield_gc_i(
            self.ctx_mut(),
            obj,
            crate::descr::list_strategy_descr(),
        );
        self.implement_guard_value(strategy, expected);
    }
}

// `MIFrame` impl — delegates `generate_guard` to the existing
// `MIFrame::generate_guard` method so the trait-dispatch leg keeps its
// current `flush_guard_not_invalidated` / `parent_frames` / `orgpc`
// plumbing untouched.  The `ctx_mut` / `ctx` accessors materialise the
// borrow from `self.ctx: *mut TraceCtx` via unsafe deref.  The deref is
// sound because `MIFrame`'s lifetime invariant
// (`pyjitpl.py:177 MIFrame.metainterp`) guarantees `self.ctx` always
// points at the live `TraceCtx` owned by the enclosing dispatch frame.
impl WalkerFrameOps for crate::state::MIFrame {
    fn ctx_mut(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    fn ctx(&self) -> &TraceCtx {
        unsafe { &*self.ctx }
    }

    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        // Re-borrow `ctx` from the raw pointer (rather than
        // `self.ctx_mut()` which holds `&mut self`) so the
        // `&mut MIFrame` borrow Rust's `MIFrame::generate_guard` needs
        // is disjoint from the `&mut TraceCtx` arg it also needs.
        // `self.ctx` is `*mut TraceCtx` so dereferencing it produces a
        // borrow whose lifetime is independent of `self`'s.
        let ctx_ptr: *mut TraceCtx = self.ctx;
        let ctx_ref: &mut TraceCtx = unsafe { &mut *ctx_ptr };
        crate::state::MIFrame::generate_guard(self, ctx_ref, opcode, args)
    }
}

// `WalkContext` impl — `generate_guard` delegates to the existing
// `walker_capture_snapshot_for_last_guard` snapshot helper after
// recording the guard op via `ctx.record_guard`.  The walker has no
// quasi-immut field reads in its STORE_SUBSCR path (concrete obj/key/value
// are direct register reads, not field loads), so the
// `flush_guard_not_invalidated` step that `MIFrame::generate_guard`
// performs is intentionally omitted here.
impl<'frame, 'static_a: 'frame> WalkerFrameOps
    for crate::jitcode_dispatch::WalkContext<'frame, 'static_a>
{
    fn ctx_mut(&mut self) -> &mut TraceCtx {
        self.trace_ctx
    }

    fn ctx(&self) -> &TraceCtx {
        self.trace_ctx
    }

    fn generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        // `pyjitpl.py:2558-2560 generate_guard` const-skip parity.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // Record the guard op first; then capture the snapshot via the
        // walker-side helper that reads `outer_active_boxes`,
        // `outer_jitcode_index`, `entry_py_pc` from `self` (the
        // `WalkContext`).  Re-borrow `trace_ctx` from `self` in a scope
        // narrow enough that `walker_capture_snapshot_for_last_guard`'s
        // `&mut WalkContext` arg sees a fresh exclusive borrow.
        self.trace_ctx.record_guard(opcode, args, 0);
        crate::jitcode_dispatch::walker_capture_snapshot_for_last_guard(self, 0);
    }
}
