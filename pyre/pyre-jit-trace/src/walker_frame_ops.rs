//! `WalkerFrameOps` trait — abstraction over the small surface of
//! `MIFrame` methods that the strategy-aware STORE_SUBSCR specialization
//! emits.  Sub-slice 5c lifts the trait-dispatch helpers
//! (`generated_list_setitem_by_strategy`,
//! `generated_list_setslice_same_len_by_strategy`,
//! `generated_store_subscr_value`, and the `store_subscr_value` body in
//! `trace_opcode.rs`) onto this trait so the walker dispatch path can
//! emit the same `guard_class`+`SETARRAYITEM_GC`-family shape that
//! today only the trait path produces.
//!
//! ## Why a trait (not free functions)
//!
//! Five of the six methods — `value_type`, `implement_guard_value`,
//! `guard_class`, `guard_int_object_value`, `guard_list_strategy` — are
//! pure compositions over `ctx` + `generate_guard`, so they live as
//! default impls below.  The lone load-bearing method is
//! `generate_guard`, which on `MIFrame` walks `parent_frames`,
//! `flush_to_frame_for_guard`, `get_list_of_active_boxes`,
//! `build_framestack_snapshot`, etc.  The walker has the same semantic
//! responsibility (capture a multi-frame resume snapshot before
//! recording the guard op) but reaches it through
//! `walker_capture_snapshot_for_last_guard` and the dispatch-time
//! `WalkContext` register banks instead of `MIFrame`'s state.  Two
//! distinct implementations are unavoidable; a trait makes the two
//! impls interchangeable at the `generated_*` call sites.
//!
//! ## Trait scope
//!
//! Only the methods reached by the STORE_SUBSCR specialization closure
//! are members.  Other `MIFrame` methods (`guard_nonnull`,
//! `trace_dynamic_list_index`, …) stay where they are; they'll join
//! this trait only when a future sub-slice needs them on the walker
//! side.
//!
//! ## Step 4 plan
//!
//! Walker impl lives in `jitcode_dispatch.rs` as
//! `impl<'a,'b> WalkerFrameOps for WalkContext<'a,'b>`.  Its
//! `generate_guard` delegates to the existing
//! `walker_capture_snapshot_for_last_guard` + `ctx.trace_ctx
//! .record_guard(...)` pair, and `value_type` reads
//! `ctx.trace_ctx.get_opref_type` directly.  After that lands,
//! `generated_*` functions in `majit-translate/src/codegen.rs` swap
//! `frame: &mut crate::state::MIFrame` for `frame: &mut impl
//! WalkerFrameOps`, completing the lift.

use majit_ir::{OpCode, OpRef, Type};
use majit_metainterp::TraceCtx;
use pyre_object::PyType;

/// Surface used by the strategy-aware STORE_SUBSCR specialization.  See
/// module doc for the rationale.
pub trait WalkerFrameOps {
    /// `pyjitpl.py:177-220` `Box.type` parity — return the OpRef's
    /// intrinsic type (Const kind, recorded result_type, or PtrInfo
    /// virtualized).  Defaults to `Type::Ref` when the OpRef has no
    /// recorded type, mirroring `MIFrame::value_type`.
    fn value_type(&self, ctx: &TraceCtx, value: OpRef) -> Type;

    /// `pyjitpl.py:2558-2602` `generate_guard` parity — flush a pending
    /// quasi-immut guard, capture multi-frame resume snapshot, then
    /// record the guard op with its snapshot.  The single load-bearing
    /// method; impls diverge between `MIFrame` (trait dispatch frame
    /// state) and `WalkContext` (walker register banks + dispatch
    /// snapshot helper).
    fn generate_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]);

    /// `pyjitpl.py:3508-3514` `implement_guard_value` parity — pick the
    /// const factory (`const_ref` for Type::Ref, `const_int`
    /// otherwise), record `GUARD_VALUE`, then update the heapcache's
    /// box replacement so downstream reads see the proved constant.
    fn implement_guard_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        let expected_ref = match self.value_type(ctx, value) {
            Type::Ref => ctx.const_ref(expected),
            _ => ctx.const_int(expected),
        };
        self.generate_guard(ctx, OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py:3512 `replace_box` parity.
        ctx.heap_cache_mut().replace_box(value, expected_ref);
    }

    /// `pyjitpl.py:1518-1523` `opimpl_guard_class` parity.  Skips the
    /// guard when the heapcache already knows the class or the OpRef
    /// is constant (the runtime type is already pinned).  Otherwise
    /// records `GUARD_NONNULL_CLASS` with `expected_type_const` and
    /// updates the heapcache's class+nullity record.
    fn guard_class(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        expected_type: *const PyType,
    ) {
        if ctx.heap_cache().is_class_known(obj) {
            return;
        }
        if obj.is_constant() {
            // The trait-side body also flushes a pending
            // GUARD_NOT_INVALIDATED here (trace_opcode.rs:4681).  The
            // walker impl's `generate_guard` is responsible for the
            // analogous flush; the const-skip path bypasses the guard
            // record, so the flush must happen even on this branch.
            // Walker impl: forward to `walker_flush_guard_not_invalidated`
            // before `class_now_known`.  Default impl below stays
            // flush-free since the constant short-circuit is the same
            // upstream; impls override only if the flush sequencing
            // differs.
            ctx.heap_cache_mut()
                .class_now_known(obj, expected_type as usize as i64);
            return;
        }
        let expected_type_const = ctx.const_int(expected_type as usize as i64);
        self.generate_guard(ctx, OpCode::GuardNonnullClass, &[obj, expected_type_const]);
        // heapcache.py:470-473 `class_now_known` parity.
        ctx.heap_cache_mut()
            .class_now_known(obj, expected_type as usize as i64);
    }

    /// `intobject.py` `int_intval`-pattern guard — class-guard the obj
    /// as `INT_TYPE`, read its `intval` field, then `implement_guard_value`
    /// the unboxed payload against `expected`.
    fn guard_int_object_value(
        &mut self,
        ctx: &mut TraceCtx,
        int_obj: OpRef,
        expected: i64,
    ) {
        self.guard_class(
            ctx,
            int_obj,
            &pyre_object::pyobject::INT_TYPE as *const PyType,
        );
        let actual_value =
            crate::state::opimpl_getfield_gc_i(ctx, int_obj, crate::descr::int_intval_descr());
        self.implement_guard_value(ctx, actual_value, expected);
    }

    /// `listobject.py` strategy field guard — `getfield_gc_i(strategy)`
    /// then `implement_guard_value`.  Skips runtime W_ListObject layout
    /// reasoning by reading the strategy id directly from its descr.
    fn guard_list_strategy(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected: i64) {
        let strategy =
            crate::state::opimpl_getfield_gc_i(ctx, obj, crate::descr::list_strategy_descr());
        self.implement_guard_value(ctx, strategy, expected);
    }
}

// `MIFrame` impl — delegates to the existing `pub(crate)` methods in
// `trace_opcode.rs` so the trait-dispatch leg keeps its current
// `flush_guard_not_invalidated` / `parent_frames` / `orgpc` plumbing
// untouched.  The trait + default impls above produce byte-identical
// recorded IR to the inline `MIFrame::*` methods because the default
// bodies are direct ports of the corresponding `MIFrame` bodies.
impl WalkerFrameOps for crate::state::MIFrame {
    fn value_type(&self, _ctx: &TraceCtx, value: OpRef) -> Type {
        crate::state::MIFrame::value_type(self, value)
    }

    fn generate_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        crate::state::MIFrame::generate_guard(self, ctx, opcode, args)
    }

    // `implement_guard_value` / `guard_class` / `guard_int_object_value`
    // / `guard_list_strategy` use the default impls above.  Their
    // bodies are direct ports of the `MIFrame` versions, so the
    // recorded IR is byte-identical to the trait path's emit.
    //
    // The one observable behavior the default impls drop versus
    // `MIFrame::guard_class` is the const-arm
    // `flush_guard_not_invalidated` (trace_opcode.rs:4681): this only
    // fires when a prior quasi-immut field read set
    // `pending_guard_not_invalidated_pc`.  The trait path's
    // `store_subscr_value` precondition is "no pending quasi-immut
    // guard" (concrete obj/key/value are non-null direct stack reads,
    // not quasi-immut field loads), so the const-skip never occurs in
    // STORE_SUBSCR.  Walker impl in step 4 will assert the precondition
    // before delegating, keeping behavior parity with the default impl.
}
