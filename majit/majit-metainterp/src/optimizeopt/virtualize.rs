/// Virtualize optimization pass: remove heap allocations for non-escaping objects.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualize.py.
///
/// Tracks "virtual" objects — allocations that never escape the trace.
/// Instead of emitting the allocation, fields are tracked in the optimizer.
/// If a virtual escapes (e.g., passed to a call or stored in a non-virtual),
/// it gets "forced" (materialized by emitting the allocation + setfield ops).
use std::sync::Arc;

use majit_ir::operand::Operand;
use majit_ir::{Descr, DescrRef, FieldDescr, OopSpecIndex, Op, OpCode, OpRef, Type, Value};

use crate::optimizeopt::info::{
    ArrayStructInfo, PtrInfo, VirtualArrayInfo, VirtualInfo, VirtualStructInfo,
    VirtualizableFieldState,
};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Optimizer-level config for virtualizable frame tracking.
///
/// Byte offsets of frame fields that should be tracked symbolically.
/// The optimizer absorbs SetfieldRaw/GetfieldRaw on these fields and
/// carries their values in guard fail_args instead of emitting memory ops.
#[derive(Clone, Debug)]
pub(crate) struct VirtualizableConfig {
    /// Byte offsets of static (scalar) frame fields (e.g. next_instr, stack_depth).
    pub static_field_offsets: Vec<usize>,
    /// Types of static (scalar) frame fields, parallel to `static_field_offsets`.
    #[allow(dead_code)]
    pub static_field_types: Vec<Type>,
    /// virtualizable.py:71-72 `static_field_descrs`.
    ///
    /// Standard virtualizable traces must keep using the real cached
    /// field descriptors built by `VirtualizableInfo`, not synthetic
    /// slot-only placeholders. `OptVirtualize::init_virtualizable`
    /// copies these into `VirtualizableFieldState.field_descrs` so the
    /// force path later emits upstream-shaped SetfieldRaw ops whose
    /// FieldDescr carries `parent_descr`.
    pub static_field_descrs: Vec<DescrRef>,
    /// Byte offsets of array pointer frame fields (e.g. locals_w, value_stack_w).
    pub array_field_offsets: Vec<usize>,
    /// Item types of array fields, parallel to `array_field_offsets`.
    #[allow(dead_code)]
    pub array_item_types: Vec<Type>,
    /// virtualizable.py:73-74 `array_field_descrs`.
    ///
    /// Same role as `static_field_descrs`, but for the array-pointer
    /// fields on the virtualizable object.
    pub array_field_descrs: Vec<DescrRef>,
    /// Number of input slots between `OpRef::input_arg_ref(0)` (frame) and the first vable
    /// scalar slot. Equals `JitDriverStaticData::num_reds() - 1` after the
    /// frame is excluded — typically `NUM_EXTRA_REDS` from the
    /// virtualizable!{} macro (e.g. `1` for pyre's `extra_reds = { ec: Ref }`).
    /// `0` means the legacy `[frame, vable_scalars..., array_items...]`
    /// layout; nonzero shifts every input-derived OpRef by that count.
    /// Mirrors `interp_jit.py reds = ['frame', 'ec']` — the non-vable
    /// extra reds occupy `InputArg` slots `1..1+vable_input_offset`.
    pub vable_input_offset: usize,
    /// Flat input-arg slot holding the virtualizable identity at loop entry.
    ///
    /// `Some(0)` is the legacy `[frame, vable_scalars.., array_items..]` layout,
    /// where the frame leads. The macro state-field JIT mints its inputargs in
    /// `[int scalars.., fixed-array cells.., identity]` order instead
    /// (`majit-macros/src/jit_interp/codegen_state.rs` `create_sym`), so the
    /// identity sits past the scalars and the cells; that position comes from
    /// `VirtualizableInfo::identity_live_index`, which the macro emits only when
    /// the state declares no fixed array. With one present the position is
    /// `num_scalars + sum(fixed array lengths)` — runtime `Vec` lengths, not a
    /// macro-expansion constant — so nothing is declared and this is `None`.
    ///
    /// `None` means no position was declared, and the tracker then
    /// DECLINES to track this virtualizable. It must not fall back to slot 0,
    /// because that does not fail loudly on the state-field layout: it finds the
    /// first int scalar and installs `PtrInfo::Virtualizable` on it, since
    /// `inputarg_type` keys on the raw index and never on the `InputArg` variant
    /// tag, so a `Ref`-tagged probe resolves to the `Int` slot's host. The
    /// preamble then exports that scalar as a `Ref` leaf and the loop-close jump
    /// hands back the `Int` it really is — the `expected=Ref actual=Int` cross
    /// rejected at `virtualstate.rs` `enum_forced_boxes_for_entry`, i.e.
    /// `VirtualStatesCantMatch`, and no trace ever compiles. It only bites when
    /// that scalar reaches the Jump as its own inputarg; a scalar reassigned
    /// every iteration passes the recomputed value instead and hides it.
    /// Declining costs the virtualizable optimization; guessing costs every
    /// trace. What declining actually costs on this layout was measured on
    /// `tests/jit_interp_fixed_array_identity_slot.rs` and is nil: resolving the
    /// slot and declining it give the same compile count and the same trace size
    /// (6 ops recorded, 5 after optimization), because `is_standard_ref` never
    /// returns true there and no vable array access is ever resolved for
    /// mirroring.
    ///
    /// The decline is not total. On the bridge path (`ctx.building_bridge`)
    /// `identity_input_ref` returns `Some(input_arg_ref(ctx.inputarg_base))`
    /// before it consults this field at all, so a tracker is still installed
    /// there with this set to `None`.
    pub identity_input_index: Option<usize>,
}

/// JitVirtualRef field slot indices.
///
/// RPython virtualref.py: JitVirtualRef has two fields (virtual_token, forced).
/// The typeptr/vtable at offset 0 is handled by NEW_WITH_VTABLE, not stored as
/// a tracked field. Indices are dense (0-based), matching RPython's
/// `heaptracker.all_fielddescrs()` which excludes typeptr.
pub(crate) const VREF_VIRTUAL_TOKEN_FIELD_INDEX: u32 = 0;
pub(crate) const VREF_FORCED_FIELD_INDEX: u32 = 1;
/// Size descriptor index for the JitVirtualRef struct.
const VREF_SIZE_DESCR_INDEX: u32 = 0x7F10;

/// TODO: Virtualizable field tracking in the optimizer — pyre-only, being
/// retired.
///
/// RPython does NOT track virtualizable field values in the optimizer.
/// Field tracking happens during tracing (`pyjitpl.py:virtualizable_boxes`),
/// not in the optimization pipeline. The optimizer only removes
/// `COND_CALL(OS_JIT_FORCE_VIRTUALIZABLE)` when the target is virtual.
///
/// The tracing layer now has that model — `TraceCtx::virtualizable_boxes` is
/// the live shadow, and both loop-close arms carry it into the JUMP — so the
/// array-element half of this tracker is gone: `init` no longer seeds element
/// state from the trace-entry input args, because the standard-path read
/// answers from the shadow and records no op for a fold to match.
///
/// What is still here, and what retiring the rest costs:
///
/// - the identity `PtrInfo::Virtualizable` install. Not a deviation to
///   remove — it is what keeps the base from being forced, which is the one
///   virtualizable job upstream's optimizer does have.
/// - the STATIC field map (`VirtualizableFieldState.fields`), still seeded
///   from input args. Retiring it needs the same argument the array half
///   got: that no recorded op reads a static vable field on the standard
///   path. That has not been established.
/// - `is_standard_ref` / `mirror_setarrayitem` / `invalidate_array`, which
///   exist to keep the static map honest and follow it.
pub(crate) struct VirtualizableTracker {
    config: VirtualizableConfig,
    needs_setup: bool,
}

impl VirtualizableTracker {
    fn new(config: VirtualizableConfig) -> Self {
        VirtualizableTracker {
            config,
            needs_setup: false,
        }
    }

    fn setup(&mut self) {
        self.needs_setup = true;
    }

    /// The input-arg slot the virtualizable identity occupies, or `None` when
    /// no sound slot is known and the tracker must decline.
    ///
    /// The answer is always expressed in this run's own OpRef namespace:
    /// inputarg `#i` is raw OpRef `inputarg_base + i`. Phase 1 runs at base 0;
    /// Phase 2 and bridges are shifted above the parent trace's high water
    /// mark.
    ///
    /// A bridge keeps the identity at slot 0: its input args are rebuilt from
    /// the deadframe frame-first, which is also what `init`'s seeding gate and
    /// `is_standard_ref` assume, so the front end's resolved offset does not
    /// describe them — the bridge's layout is established by the deadframe.
    /// Every other run — loop, preamble, unrolled Phase 2 body — carries the
    /// front end's own layout, so it consults `identity_input_index` and
    /// declines with it.
    ///
    /// The discriminator is `ctx.building_bridge`, not
    /// `inputarg_base != 0`. Phase 2 shifts the base as well, so keying on the
    /// base sends an unrolled loop body down the bridge path and installs
    /// `PtrInfo::Virtualizable` on whatever inputarg 0 happens to be. On the
    /// banked layout that is an Int scalar, and the loop-close Jump then fails
    /// to match the Ref-typed preview (`VirtualStatesCantMatch`).
    fn identity_input_ref(&self, ctx: &OptContext) -> Option<OpRef> {
        if ctx.building_bridge {
            return Some(OpRef::input_arg_ref(ctx.inputarg_base));
        }
        Some(OpRef::input_arg_ref(
            ctx.inputarg_base + self.config.identity_input_index? as u32,
        ))
    }

    /// Apply deferred virtualizable setup if needed.
    fn ensure_setup(&mut self, ctx: &mut OptContext) {
        if self.needs_setup {
            self.needs_setup = false;
            // No sound identity slot — decline rather than install
            // `PtrInfo::Virtualizable` on whatever inputarg 0 happens to be.
            let Some(identity_ref) = self.identity_input_ref(ctx) else {
                return;
            };
            let first_check = ctx
                .get_box_replacement_operand_opt(identity_ref)
                .as_ref()
                .is_some_and(|b| ctx.has_ptr_info(b));
            if !first_check {
                self.init(ctx);
                let second_check = ctx
                    .get_box_replacement_operand_opt(identity_ref)
                    .as_ref()
                    .is_some_and(|b| ctx.has_ptr_info(b));
                if !second_check {
                    {
                        let b = ctx.materialize_operand_at(identity_ref);
                        ctx.set_ptr_info(
                            &b,
                            PtrInfo::Virtualizable(VirtualizableFieldState {
                                fields: vec![],
                                field_descrs: vec![],
                                arrays: vec![],
                                heap_fields: vec![],
                                last_guard_pos: -1,
                            }),
                        );
                    }
                }
            }
        }
    }

    /// Seed virtualizable state from existing trace inputs.
    fn init(&mut self, ctx: &mut OptContext) {
        if ctx.num_inputs() <= 1 {
            return;
        }
        // Same decline as `ensure_setup`: without a known identity slot there is
        // nothing to hang the seeded `VirtualizableFieldState` on.
        let Some(identity_ref) = self.identity_input_ref(ctx) else {
            return;
        };

        let mut state = VirtualizableFieldState {
            fields: vec![],
            field_descrs: vec![],
            arrays: vec![],
            heap_fields: vec![],
            last_guard_pos: -1,
        };
        // Input-layout seeding applies only to the initial loop/preamble entry
        // (`inputarg_base == 0`), whose inputargs carry the
        // `[frame, vable_scalars..., array_items...]` layout. The gate is the
        // base and not `building_bridge` because the seeding below addresses
        // slots with unshifted raw indices (`input_arg_typed(flat_input_idx)`),
        // which name this run's inputargs only when the base is 0.
        //
        // A bridge inherits only the failing guard's live boxes as inputargs
        // (frame first, then the surviving reds), not the unpacked vable
        // scalar/array slots; it re-establishes the virtualizable fields from
        // the explicit SetfieldGc/SetarrayitemGc reconstruction ops the resume
        // path records into the bridge body. So the bridge frame is seeded as
        // an empty Virtualizable and populated by those ops. An unrolled
        // Phase 2 body is also shifted and skips the seeding, but keeps the
        // resolved identity slot; see `identity_input_ref`.
        let base = ctx.inputarg_base;
        if base == 0 {
            let mut flat_input_idx = 1usize + self.config.vable_input_offset;

            // RPython `info.AbstractStructPtrInfo._fields` is keyed by
            // `fielddescr.get_index()` (descr.py `index_in_parent`,
            // populated by `cpu.fielddescrof(VTYPE, name)`).  Mirror that
            // here so runtime queries via
            // `op.descr.as_field_descr()?.index_in_parent() as u32` find the
            // slot the init step seeded.
            //
            // `virtualizable.py:71-72 build_field_descr` assigns
            // `index_in_parent = 1 + i` for static fields and
            // `1 + num_static + j` for array-pointer fields; mirror that
            // schedule for the synthetic fallback used by tests that pass
            // empty `static_field_descrs` / `array_field_descrs`.
            let num_static = self.config.static_field_offsets.len();
            for (field_idx_in_vinfo, &_offset) in
                self.config.static_field_offsets.iter().enumerate()
            {
                if flat_input_idx >= ctx.num_inputs() {
                    break;
                }
                let descr_for_slot = self
                    .config
                    .static_field_descrs
                    .get(field_idx_in_vinfo)
                    .cloned();
                let field_idx = descr_for_slot
                    .as_ref()
                    .and_then(|d| d.as_field_descr())
                    .map(|fd| fd.index_in_parent() as u32)
                    .unwrap_or((1 + field_idx_in_vinfo) as u32);
                let slot_tp = ctx
                    .inputarg_type_at(flat_input_idx)
                    .unwrap_or(majit_ir::Type::Ref);
                let input_ref = OpRef::input_arg_typed(flat_input_idx as u32, slot_tp);
                let input_op = ctx.materialize_operand_at(input_ref);
                set_field(&mut state.fields, field_idx, input_op);
                if let Some(descr) = descr_for_slot {
                    set_field_descr(&mut state.field_descrs, field_idx, descr);
                }
                flat_input_idx += 1;
            }

            // Array elements are deliberately not seeded. Every layout carries
            // them into the loop JUMP through the tracer's live
            // `virtualizable_boxes` shadow — the macro state-field JIT via
            // `JitState::collect_jump_args_with_boxes`, PyFrame via
            // `jitcode_dispatch::append_virtualizable_boxes` — and the tracer
            // updates that shadow through `set_virtualizable_entry_at` without
            // recording an op, so a seeded entry box is invisible to
            // `mirror_setarrayitem` and goes stale. The standard-path read
            // records no op either (`TraceCtx::vable_getarrayitem_*_checked`
            // answers from the shadow, pyjitpl.py:1170-1184), so there is
            // nothing for a seeded element to fold against in the first place.
            // Measured before removal: check.py dynasm 434/434 with zero
            // jit-stats counters moved.
        }

        let b = ctx.materialize_operand_at(identity_ref);
        ctx.set_ptr_info(&b, PtrInfo::Virtualizable(state));
    }

    fn is_standard_ref(&self, b: &Operand, ctx: &OptContext) -> bool {
        // pyjitpl.py `standard_box is box` — box identity against the
        // standard virtualizable frame, then virtualizable check. The slot is
        // resolved by `identity_input_ref`, so this reads the same answer
        // `ensure_setup` installed on: a bridge's frame-first slot 0, or the
        // front end's declared `identity_input_index`, both in this run's own
        // OpRef namespace. A run with no known identity slot declines:
        // nothing is the standard ref.
        match self
            .identity_input_ref(ctx)
            .and_then(|r| ctx.get_box_replacement_operand_opt(r))
        {
            Some(std) => b.same_box(&std) && ctx.is_virtualizable(b),
            None => false,
        }
    }

    fn array_idx_for_offset(&self, offset: usize) -> Option<u32> {
        self.config
            .array_field_offsets
            .iter()
            .position(|&off| off == offset)
            .map(|idx| idx as u32)
    }

    fn resolve_array_source(
        &self,
        array_box: &Operand,
        ctx: &mut OptContext,
    ) -> Option<(Operand, u32)> {
        let producer = ctx.get_producing_op(array_box)?;
        // The array-pointer field of the virtualizable frame is read with the
        // raw field descr on the loop hot path (GetfieldRaw*), but with the
        // virtualizable field descr during a bridge's frame reconstruction
        // (GetfieldGc*). Both name the same array-pointer slot; the offset
        // gate in `array_idx_for_offset` below filters to configured array
        // fields, so accept either read form.
        if !matches!(
            producer.opcode,
            OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
        ) {
            return None;
        }
        // Terminal box of the GetfieldRaw receiver — the virtualizable frame.
        let frame_box = ctx.resolve_operand_operand(&producer.arg(0));
        let is_standard = ctx
            .resolve_operand_operand_opt(&producer.arg(0))
            .is_some_and(|b| self.is_standard_ref(&b, ctx));
        if !is_standard {
            return None;
        }
        // `virtualize.py` reads `op.getdescr().offset` directly to resolve
        // raw-field byte offsets; mirror that via `FieldDescr::offset()`.
        let offset = producer
            .getdescr()
            .and_then(|d| d.as_field_descr().map(|fd| fd.offset()))?;
        let array_idx = self.array_idx_for_offset(offset)?;
        Some((frame_box, array_idx))
    }

    /// Mirror a setarrayitem write to the virtualizable array state.
    fn mirror_setarrayitem(
        &self,
        array_box: &Operand,
        index: i64,
        value_ref: OpRef,
        ctx: &mut OptContext,
    ) {
        if let Some((frame_box, array_idx)) = self.resolve_array_source(array_box, ctx) {
            let elem_idx = index as usize;
            let value_op = ctx.materialize_operand_at(value_ref);
            ctx.with_ptr_info_mut(&frame_box, |info| {
                if let PtrInfo::Virtualizable(vstate) = info {
                    set_array_element(&mut vstate.arrays, array_idx, elem_idx, value_op.clone());
                }
            });
        }
    }

    /// Invalidate every tracked slot of the standard-virtualizable array
    /// `array_box` resolves to.  Mirrors `force_lazy_setarrayitem(descr,
    /// indexb, can_cache=False)` (heap.py:580-586): a variable-index
    /// SETARRAYITEM_GC may overwrite any element, so every const-index slot
    /// a later read could fold against must be dropped before the write.
    fn invalidate_array(&self, array_box: &Operand, ctx: &mut OptContext) {
        if let Some((frame_ref, array_idx)) = self.resolve_array_source(array_box, ctx) {
            ctx.with_ptr_info_mut(&frame_ref, |info| {
                if let PtrInfo::Virtualizable(vstate) = info {
                    vstate.arrays.retain(|(i, _)| *i != array_idx);
                }
            });
        }
    }

    /// Read counterpart to [`mirror_setarrayitem`]: returns the tracked
    /// value box for `array_box[index]` on the standard virtualizable
    /// array state (seeded from the inputarg layout, updated by
    /// `mirror_setarrayitem`), or `None` when `array_box` is not the
    /// standard virtualizable array field or the slot is untracked.
    fn tracked_array_element(
        &self,
        array_box: &Operand,
        index: i64,
        ctx: &mut OptContext,
    ) -> Option<OpRef> {
        if index < 0 {
            return None;
        }
        let (frame_box, array_idx) = self.resolve_array_source(array_box, ctx)?;
        let elem_idx = index as usize;
        match ctx.peek_ptr_info(&frame_box)? {
            PtrInfo::Virtualizable(vstate) => {
                get_array_element(&vstate.arrays, array_idx, elem_idx)
            }
            _ => None,
        }
    }
}

/// The virtualize optimization pass.
pub struct OptVirtualize {
    /// TODO: pyre-specific virtualizable field tracker.
    /// See `VirtualizableTracker` doc comment for convergence path.
    vable: Option<VirtualizableTracker>,
    /// optimizer.py REMOVED + virtualize.py:67-75,180,247:
    last_emitted_was_removed: bool,
    /// virtualize.py:48
    last_guard_not_forced_2: Option<Op>,
    /// virtualize.py:81 / 84
    finish_guard_op: Option<Op>,
    /// `virtualize.py:140` `vrefinfo =
    /// self.optimizer.metainterp_sd.virtualref_info` parity — the
    /// cached `VirtualRefInfo` whose `descr_forced` /
    /// `descr_virtual_token` / `descr` Arcs `optimize_virtual_ref` and
    /// `optimize_virtual_ref_finish` stamp onto SETFIELD_GC ops.
    /// Cloned cheaply (3 `Arc`s); production passes the live
    /// `MetaInterp.virtualref_info`, tests use `Default`.
    vrefinfo: crate::virtualref::VirtualRefInfo,
}

impl OptVirtualize {
    pub fn new() -> Self {
        OptVirtualize {
            vable: None,
            last_emitted_was_removed: false,
            last_guard_not_forced_2: None,
            finish_guard_op: None,
            vrefinfo: crate::virtualref::VirtualRefInfo::new(),
        }
    }

    /// Create with virtualizable config for frame field tracking.
    pub(crate) fn with_virtualizable(config: VirtualizableConfig) -> Self {
        OptVirtualize {
            vable: Some(VirtualizableTracker::new(config)),
            last_emitted_was_removed: false,
            last_guard_not_forced_2: None,
            finish_guard_op: None,
            vrefinfo: crate::virtualref::VirtualRefInfo::new(),
        }
    }

    /// `virtualize.py:140` parity: install the live `VirtualRefInfo`
    /// from `MetaInterp.virtualref_info` so emit sites read the cached
    /// `vrefinfo.descr_*` Arcs through this field instead of
    /// reconstructing them on demand.
    pub fn with_vrefinfo(mut self, vrefinfo: crate::virtualref::VirtualRefInfo) -> Self {
        self.vrefinfo = vrefinfo;
        self
    }

    // ── PtrInfo accessors (delegated to ctx) ──

    fn is_virtual(opref: OpRef, ctx: &OptContext) -> bool {
        ctx.get_box_replacement_operand_opt(opref)
            .as_ref()
            .is_some_and(|b| ctx.is_virtual(b))
    }

    fn is_standard_virtualizable_ref(&self, b: &Operand, ctx: &OptContext) -> bool {
        self.vable
            .as_ref()
            .is_some_and(|vt| vt.is_standard_ref(b, ctx))
    }

    /// virtualize.py make_virtual_raw_slice
    ///
    /// ```text
    /// def make_virtual_raw_slice(self, offset, parent, source_op):
    ///     opinfo = info.RawSlicePtrInfo(offset, parent)
    ///     newop = self.replace_op_with(source_op, source_op.getopnum(),
    ///                                args=[source_op.getarg(0), ConstInt(offset)])
    ///     newop.set_forwarded(opinfo)
    ///     return opinfo
    /// ```
    ///
    /// `parent` is the *immediate* predecessor (a `RawBufferPtrInfo` or
    /// another `RawSlicePtrInfo`) — RPython stores the PtrInfo object
    /// directly; majit stores its `OpRef` and resolves through
    /// `ctx.get_ptr_info`. Slice offsets are NOT flattened at creation;
    /// `info.RawSlicePtrInfo.getitem_raw` recursively delegates via
    /// `self.parent.getitem_raw(self.offset + offset, ...)`, so the
    /// equivalent walk happens at access time in `resolve_raw_slice`.
    fn make_virtual_raw_slice(
        &mut self,
        offset: i64,
        parent: OpRef,
        source_op: &Op,
        source_op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) {
        let opinfo = crate::optimizeopt::info::RawSlicePtrInfo {
            offset,
            parent: ctx.materialize_operand_at(parent),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        };
        let b = Operand::from_bound_op(source_op_rc);
        ctx.set_ptr_info(&b, PtrInfo::VirtualRawSlice(opinfo));
    }

    /// virtualize.py make_virtual_raw_memory
    ///
    /// Create a RawBufferPtrInfo for a RAW_MALLOC_VARSIZE_CHAR
    /// result. `func` comes from source_op.getarg(0); size is the
    /// constant-folded allocation length.
    fn make_virtual_raw_memory(
        &mut self,
        size: usize,
        func: i64,
        source_op: &Op,
        source_op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) {
        let opinfo =
            crate::optimizeopt::info::RawBufferPtrInfo::new(func, size, source_op.getdescr());
        let b = Operand::from_bound_op(source_op_rc);
        ctx.set_ptr_info(&b, PtrInfo::VirtualRawBuffer(opinfo));
    }

    /// Resolve a slice/buffer alias chain to the underlying parent OpRef and
    /// the cumulative byte offset. Returns `(parent, total_offset)` when the
    /// chain ends in a `VirtualRawBuffer`, or `None` otherwise.
    fn resolve_raw_slice(opref: OpRef, ctx: &OptContext) -> Option<(OpRef, i64)> {
        let mut current = opref;
        let mut total_offset: i64 = 0;
        loop {
            let current_box = ctx.get_box_replacement_operand_opt(current);
            match current_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
                Some(PtrInfo::VirtualRawSlice(slice)) => {
                    // info.py RawSlicePtrInfo.getitem_raw recurses
                    // into `self.parent.getitem_raw(self.offset + offset,
                    // ...)`; RPython int has no overflow so a chain of
                    // signed addends is always representable. In Rust we
                    // bail on i64 overflow rather than wrap.
                    total_offset = total_offset.checked_add(slice.offset)?;
                    current = slice.parent.to_opref();
                }
                Some(PtrInfo::VirtualRawBuffer(_)) => return Some((current, total_offset)),
                _ => return None,
            }
        }
    }

    // ── Per-opcode handlers ──

    fn optimize_new_with_vtable(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let descr = op.getdescr().expect("NEW_WITH_VTABLE needs descr");
        // virtualize.py `known_class = ConstInt(op.getdescr().get_vtable())`
        // — no null filter; ConstInt(0) flows downstream as the
        // known_class. info.py ConstPtrInfo.get_known_class
        // handles the nonnull check inside, so the upstream contract
        // is "always carry the vtable value; let consumers interpret
        // null as 'no known class' at read time".
        let known_class = descr.as_size_descr().map(|sd| sd.vtable() as i64);
        let vinfo = VirtualInfo {
            descr,
            known_class,
            ob_type_descr: None,
            fields: Vec::new(),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        };
        let b = Operand::from_bound_op(op_rc);
        ctx.set_ptr_info(&b, PtrInfo::Virtual(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let descr = op.getdescr().expect("NEW needs descr");
        let vinfo = VirtualStructInfo {
            descr,
            fields: Vec::new(),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        };
        let b = Operand::from_bound_op(op_rc);
        ctx.set_ptr_info(&b, PtrInfo::VirtualStruct(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new_array(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let size_ref = op.arg(0).to_opref();
        if let Some(size) = ctx
            .resolve_operand_operand_opt(&op.arg(0))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            // virtualize.py `if not info.reasonable_array_index(size):`
            // — defined at info.py:487-492 with upper bound 150000.
            if crate::optimizeopt::info::reasonable_array_index(size) {
                let descr = op.getdescr().expect("NEW_ARRAY needs descr");
                // virtualize.py:30-32: arraydescr.is_array_of_structs()
                let is_struct = descr
                    .as_array_descr()
                    .is_some_and(|ad| ad.is_array_of_structs());
                if is_struct {
                    // virtualize.py:31: assert clear
                    debug_assert!(matches!(op.opcode, OpCode::NewArrayClear));
                    // info.py:645: lgt = len(descr.get_all_fielddescrs())
                    let fielddescrs: Vec<DescrRef> = descr
                        .as_array_descr()
                        .and_then(|ad| ad.get_all_interiorfielddescrs())
                        .map(|fds| fds.to_vec())
                        .unwrap_or_default();
                    let lgt = fielddescrs.len();
                    // info.py:648: self._items = [None] * (size * lgt)
                    let element_fields = (0..size as usize)
                        .map(|_| (0..lgt as u32).map(|j| (j, Operand::None)).collect())
                        .collect();
                    let vinfo = ArrayStructInfo {
                        descr,
                        fielddescrs,
                        element_fields,
                        last_guard_pos: -1,
                        avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                    };
                    let b = Operand::from_bound_op(op_rc);
                    ctx.set_ptr_info(&b, PtrInfo::VirtualArrayStruct(vinfo));
                } else {
                    // virtualize.py `make_varray` passes
                    // `optimizer.new_const_item(arraydescr)` into
                    // `ArrayPtrInfo`; info.py `_init_items` fills a
                    // NEW_ARRAY_CLEAR with that typed zero/null constant.
                    // NEW_ARRAY deliberately keeps `None` for unreadable,
                    // uninitialized elements.
                    let clear = matches!(op.opcode, OpCode::NewArrayClear);
                    let items = if clear {
                        let item_type = descr
                            .as_array_descr()
                            .expect("non-struct NEW_ARRAY descr must be an ArrayDescr")
                            .item_type();
                        let default_ref = match item_type {
                            Type::Int | Type::Void => ctx.make_constant_int(0),
                            Type::Ref => ctx.make_constant_ref(majit_ir::GcRef::NULL),
                            Type::Float => ctx.make_constant_float(0.0),
                        };
                        vec![ctx.materialize_operand_at(default_ref); size as usize]
                    } else {
                        vec![Operand::None; size as usize]
                    };
                    let vinfo = VirtualArrayInfo {
                        descr,
                        clear,
                        items,
                        last_guard_pos: -1,
                        avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
                    };
                    let b = Operand::from_bound_op(op_rc);
                    ctx.set_ptr_info(&b, PtrInfo::VirtualArray(vinfo));
                }
                return OptimizationResult::Remove;
            }
        }
        // virtualize.py:220 `self.pure_from_args(rop.ARRAYLEN_GC, [op],
        // arg, descr=op.getdescr())` — array descr discriminates the
        // pure-cache key so the reverse ARRAYLEN→size fold doesn't
        // collide across distinct array types.
        if let Some(descr) = op.getdescr() {
            ctx.register_pure_from_args1_with_descr(
                OpCode::ArraylenGc,
                op.pos.get(),
                size_ref,
                descr,
            );
        } else {
            ctx.register_pure_from_args1(OpCode::ArraylenGc, op.pos.get(), size_ref);
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_NEW_ARRAY_CLEAR.
    /// RPython forwards to `optimize_NEW_ARRAY(op, clear=True)`; the
    /// OpCode discriminator in majit already encodes `clear` semantics
    /// (optimize_new_array consults `OpCode::NewArrayClear`),
    /// so this wrapper has no behavioral effect. Kept as a structural
    /// mirror of the upstream dispatch table.
    #[allow(dead_code)]
    fn optimize_new_array_clear(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        self.optimize_new_array(op, op_rc, ctx)
    }

    fn optimize_setfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let struct_box = ctx.resolve_operand_operand_opt(&op.arg(0));
        let value_ref = ctx.resolve_operand_operand(&op.arg(1)).to_opref();
        let setfield_descr_arc = op
            .getdescr()
            .expect("optimize_setfield_gc: field op without FieldDescr");
        let field_descr = setfield_descr_arc
            .as_field_descr()
            .expect("optimize_setfield_gc: field op without FieldDescr");
        let field_idx = field_descr.index_in_parent() as u32;
        let is_typeptr = field_descr.is_typeptr();
        let is_raw_op = matches!(op.opcode, OpCode::SetfieldRaw);
        // Pre-extract constant value before mutable borrow of ptr_info.
        // Class pointer may be stored as Value::Int OR Value::Ref.
        let value_as_constant: Option<usize> = ctx
            .get_box_replacement_operand_opt(value_ref)
            .and_then(|b| ctx.get_constant_box(&b))
            .and_then(|v| match v {
                majit_ir::Value::Int(i) => Some(i as usize),
                majit_ir::Value::Ref(gc) => Some(gc.as_usize()),
                _ => None,
            });

        // RPython virtualize.py:200-202: virtual SetfieldGc always updates
        // the field, even for imported virtual heads. Body computation must
        // be able to update virtual fields (e.g., i.intval = i + step).

        if is_raw_op
            && struct_box
                .as_ref()
                .is_some_and(|b| self.is_standard_virtualizable_ref(b, ctx))
        {
            return OptimizationResult::PassOn;
        }

        // RPython: if struct is NOT virtual, PassOn to OptHeap which stores
        // it as a lazy_set. The virtual value is NOT forced — OptHeap delays
        // it until guard emission (force_lazy_sets_for_guard) or JUMP.

        let descr_for_vstate = Some(setfield_descr_arc.clone());
        let value_op = ctx.materialize_operand_at(value_ref);
        let early = struct_box
            .as_ref()
            .and_then(|b| ctx.with_ptr_info_mut(b, |info| {
                if !info.is_virtual() {
                    return None;
                }
                if !is_typeptr {
                    let parent_descr = field_descr.get_parent_descr().expect(
                        "optimize_setfield_gc: non-typeptr FieldDescr.get_parent_descr() returned None",
                    );
                    info.init_fields(parent_descr.clone(), field_idx as usize);
                }
                match info {
                    PtrInfo::Virtual(vinfo) => {
                        // info.py AbstractStructPtrInfo.setfield:
                        //   self._fields[fielddescr.get_index()] = op.
                        // heaptracker.py all_fielddescrs() excludes typeptr:
                        //   if name == 'typeptr': continue # dealt otherwise
                        // → _fields never contains typeptr. In pyre, typeptr
                        // setfield is filtered at trace recording time
                        // (jtransform.py:908-911 parity in helpers.rs), so this
                        // branch should not observe a typeptr op. Defensively
                        // capture known_class if a typeptr setfield still arrives.
                        if is_typeptr {
                            if vinfo.known_class.is_none()
                                && let Some(class_val) = value_as_constant {
                                    vinfo.known_class = Some(class_val as i64);
                                }
                            return Some(OptimizationResult::Remove);
                        }
                        set_field(&mut vinfo.fields, field_idx, value_op.clone());
                        if let Some(err) =
                            field_slot_disagreement(&vinfo.descr, field_idx, field_descr)
                        {
                            panic!("Virtual setfield: {err}");
                        }
                        Some(OptimizationResult::Remove)
                    }
                    PtrInfo::VirtualStruct(vinfo) => {
                        set_field(&mut vinfo.fields, field_idx, value_op.clone());
                        if let Some(err) =
                            field_slot_disagreement(&vinfo.descr, field_idx, field_descr)
                        {
                            panic!("VirtualStruct setfield: {err}");
                        }
                        Some(OptimizationResult::Remove)
                    }
                    PtrInfo::Virtualizable(vstate) => {
                        set_field(&mut vstate.fields, field_idx, value_op.clone());
                        // Store original descr for force path
                        if let Some(d) = descr_for_vstate {
                            set_field_descr(&mut vstate.field_descrs, field_idx, d);
                        }
                        Some(OptimizationResult::Remove)
                    }
                    _ => None,
                }
            }))
            .flatten();
        if let Some(result) = early {
            return result;
        }
        // RPython: virtual value is NOT forced in optimize_SETFIELD_GC.
        // It's forced by _emit_operation (optimizer.py) at final emit.
        // In majit, this is handled by emit_operation or force_all_lazy_sets.
        // virtualize.py:204: self.make_nonnull(op.getarg(0))
        if !struct_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = struct_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    fn optimize_getfield_gc(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let struct_box = ctx.resolve_operand_operand_opt(&op.arg(0));
        let field_descr_arc = op
            .getdescr()
            .expect("optimize_getfield_gc: field op without FieldDescr");
        let field_descr = field_descr_arc
            .as_field_descr()
            .expect("optimize_getfield_gc: descr is not a FieldDescr");
        let field_idx = field_descr.index_in_parent() as u32;
        let is_typeptr = field_descr.is_typeptr();
        let is_raw_op = matches!(
            op.opcode,
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF
        );
        let is_standard_vable_ref = struct_box
            .as_ref()
            .is_some_and(|b| self.is_standard_virtualizable_ref(b, ctx));

        // The standard virtualizable's array-pointer field has no virtual-field
        // value to fold to (its contents are tracked as array elements, not a
        // scalar field), so the read passes through unchanged — both the raw
        // hot-path form and the GetfieldGc* form a bridge's reconstruction
        // emits. The frame is a real object (the trace's first inputarg), so
        // reading its array-pointer field does not force it.
        let reads_vable_array_field = is_standard_vable_ref
            && self
                .vable
                .as_ref()
                .is_some_and(|vt| vt.array_idx_for_offset(field_descr.offset()).is_some());
        if (is_raw_op && is_standard_vable_ref) || reads_vable_array_field {
            return OptimizationResult::PassOn;
        }

        // info.py `getfield` opens with the same
        // `init_fields(fielddescr.get_parent_descr(), fielddescr.get_index())`
        // that `setfield` does, so upstream's read is what grows `_fields` and
        // swaps in the more precise descr (info.py:184-188) when the index
        // belongs to a subclass the allocation's descr does not cover. pyre
        // keys fields by slot instead of indexing an array, so the read needed
        // nothing to answer and the call was dropped; `vinfo.descr` then stayed
        // at whatever the allocation set. That descr is what
        // `field_slot_identifies` below reads, so the upgrade has to happen for
        // the slot it checks to be the slot upstream would have used.
        //
        // Only for a virtual: `virtualize.py` reaches `opinfo.getfield`
        // under `opinfo.is_virtual()`, and a non-virtual info's descr is
        // `OptHeap`'s to move (`optimizer.py:484`). The header reads are
        // excluded by the `is_typeptr` / `is_w_class` guards below -- they do
        // not resolve through the field list at all.
        if !is_raw_op
            && !field_descr.is_header_field()
            && let (Some(b), Some(parent_descr)) =
                (struct_box.as_ref(), field_descr.get_parent_descr())
        {
            ctx.with_ptr_info_mut(b, |info| {
                if info.is_virtual() {
                    info.init_fields(parent_descr, field_idx as usize);
                }
            });
        }

        if let Some(info) = struct_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            // info.py getfield: return _fields[fielddescr.get_index()].
            // For Virtual, ob_type (typeptr) is not in fields — fold from
            // known_class (info.py get_known_class).
            if let PtrInfo::Virtual(ref vinfo) = info
                && is_typeptr
                // A stored class of 0 means the allocation's vtable address was unavailable at
                // build time, so the value reads as no known class while the flag stays valid.
                && let Some(class_val) = vinfo.known_class.filter(|&c| c != 0)
            {
                let b = ctx.materialize_operand_at(op.pos.get());
                ctx.make_constant_box(&b, majit_ir::Value::Int(class_val));
                return OptimizationResult::Remove;
            }
            // Pyre object-model: `w_class` (PyObject offset 8) is a header
            // field carrying Python-level class identity, not a value field.
            // Its shared descr has `index_in_parent == 0`, which would
            // collide with the first value field (e.g. `W_IntObject.intval`,
            // an `Int`) and forward `Ref ← Int` in `make_equal_to`. Resolve
            // it from the virtual's class identity: a stored `w_class` field
            // when the layout tracks one (specialised tuples set it
            // explicitly), otherwise the canonical class object for the size
            // descr's type (builtins built by `new_with_vtable` inherit the
            // type's `get_instantiate`).
            if field_descr.is_w_class()
                && let PtrInfo::Virtual(ref vinfo) = info
            {
                let stored = vinfo
                    .descr
                    .as_size_descr()
                    .and_then(|sd| sd.class_word_index_in_parent().map(|idx| idx as u32))
                    .and_then(|widx| get_field(&vinfo.fields, widx));
                if let Some(val_ref) = stored {
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_val = ctx.get_box_replacement_operand(val_ref);
                    ctx.make_equal_to(&b_old, &b_val);
                    return OptimizationResult::Remove;
                }
                if let Some(w_class) = vinfo
                    .descr
                    .as_size_descr()
                    .and_then(|sd| sd.w_class_obj())
                    .filter(|&w| w != 0)
                {
                    let b = ctx.materialize_operand_at(op.pos.get());
                    ctx.make_constant_box(
                        &b,
                        majit_ir::Value::Ref(majit_ir::GcRef(w_class as usize)),
                    );
                    return OptimizationResult::Remove;
                }
                // Class identity unresolved: leave the read in place so
                // the virtual is forced and the real `w_class` is read,
                // rather than mis-indexing a value field.
                return OptimizationResult::PassOn;
            }
            // Once the allocation is forced the read is NOT resolved — not
            // here and not in `optimize_getfield`. `virtualize.py:184-195
            // optimize_GETFIELD_GC_*` folds only under
            // `opinfo.is_virtual()` and otherwise emits; upstream has no
            // counterpart to resolve afterwards because
            // `jtransform.py handle_getfield_typeptr` deletes the
            // read at codewriter time, so no typeptr getfield ever reaches
            // the optimizer.
            //
            // Answering from the layout's canonical class here is unsound:
            // `force_box` empties the forced instance's field list
            // (`info.rs`'s `force_box_impl`, so heap's `do_setfield` cannot
            // MUST_ALIAS-elide the materializing SETFIELD_GC) and routes
            // the header write into `OptHeap`, where it sits in
            // `CachedField::lazy_set`. A retag to a user subclass would be
            // discarded and the base class answered instead. Resolving it
            // in `OptHeap` does not close the hole either: reads and writes
            // of this header carry different descr spellings, and the
            // caches are keyed by `Arc::as_ptr` while `structinfo_setfield`
            // slots by `field_slot_index`, so the two do not meet. Removing
            // that split is the prerequisite for folding this read at all.
            // `optimize_setfield_gc` panics on a slot its descr does not
            // identify, but a spelling that only ever appears on reads never
            // reaches that check, and this is the side that resolves it.
            //
            // It does reach here.  `PYFRAME_VABLE_TOKEN_FIELD_DESCR`
            // (`pyre-jit-trace descr.rs`) describes `PyFrame.vable_token` at
            // its byte offset with a placeholder `index_in_parent: 0` and no
            // parent, because the positional census that assigns the real
            // indices deliberately does not list the field -- upstream carries
            // it as `rvirtualizable.py:29`'s appended `('vable_token',
            // llmemory.GCREF)` and pyre registers it as an extra GC edge so
            // `clear_gc_fields` zeroes it.  Slot 0 of that layout is
            // `PyFrame.locals_cells_stack_w`, so `field_idx` addressed the
            // locals array and `get_field` forwarded a live array pointer as
            // the frame's token; `emit_force_virtualizable` reads that token
            // with GETFIELD_GC_R to decide whether the frame is JIT-owned, and
            // a non-null pointer reads as owned on a frame that has no token.
            //
            // A field the positional list does not hold cannot have been stored
            // under its own identity either, so this is exactly
            // `virtualize.py:188`'s state: the trace never stored it and the
            // read answers the zeroed allocation.  Skip the slot lookup and
            // take the zero fold below -- which for `vable_token` is the
            // correct value, a virtual frame having never been forced.
            //
            // Not gated on `debug_assertions`: the resolution it guards runs in
            // release, so the guard has to.  `Virtualizable` is not covered --
            // its fields come from the state-field JIT's own descr set, which
            // `vstate.descr` does not index.
            let slot_identifies_field = match &info {
                PtrInfo::Virtual(vinfo) => {
                    field_slot_identifies(&vinfo.descr, field_idx, field_descr)
                }
                PtrInfo::VirtualStruct(vinfo) => {
                    field_slot_identifies(&vinfo.descr, field_idx, field_descr)
                }
                _ => true,
            };
            let slot_resolvable =
                slot_identifies_field || is_raw_op || field_descr.is_header_field();
            if !slot_resolvable && crate::majit_log_enabled() {
                // What the skip is worth: a populated slot is the value
                // `get_field` would have forwarded for a field not in it.
                let populated = match &info {
                    PtrInfo::Virtual(vinfo) => get_field(&vinfo.fields, field_idx).is_some(),
                    PtrInfo::VirtualStruct(vinfo) => get_field(&vinfo.fields, field_idx).is_some(),
                    _ => false,
                };
                eprintln!(
                    "[jit][getfield-slot-unlisted] field {:?} at offset {} does not hold slot \
                     {field_idx} of the virtual's descr (slot populated: {populated}); folding \
                     to the zeroed allocation",
                    field_descr.field_name(),
                    field_descr.offset(),
                );
            }
            let field_val = match &info {
                _ if !slot_resolvable => None,
                PtrInfo::Virtual(vinfo) => get_field(&vinfo.fields, field_idx),
                PtrInfo::VirtualStruct(vinfo) => get_field(&vinfo.fields, field_idx),
                PtrInfo::Virtualizable(vstate) => vstate
                    .fields
                    .iter()
                    .find(|(idx, _)| *idx == field_idx)
                    .filter(|(_, b)| {
                        virtualizable_slot_identifies(
                            vstate,
                            field_idx,
                            field_descr,
                            b,
                            op.result_type(),
                        )
                    })
                    .map(|(_, b)| b.to_opref()),
                _ => None,
            };
            if let Some(val_ref) = field_val {
                let b_old = Operand::from_bound_op(op_rc);
                let b_val = ctx
                    .get_box_replacement_operand_opt(val_ref)
                    .unwrap_or_else(|| ctx.materialize_operand_at(val_ref));
                ctx.make_equal_to(&b_old, &b_val);
                return OptimizationResult::Remove;
            }
            // heaptracker.py:66 typeptr exclusion: typeptr is excluded from
            // virtual fields but can be resolved from the SizeDescr vtable.
            // RPython doesn't need this because GUARD_CLASS reads the class
            // directly from the object, not via a separate field read.
            let is_typeptr = op.with_field_descr(|fd| fd.is_typeptr()).unwrap_or(false);
            if field_val.is_none()
                && matches!(op.opcode, majit_ir::OpCode::GetfieldGcI)
                && is_typeptr
            {
                let vtable = match &info {
                    PtrInfo::Virtual(vinfo) => vinfo
                        .descr
                        .as_size_descr()
                        .map(|sd| sd.vtable())
                        .filter(|&v| v != 0),
                    PtrInfo::VirtualStruct(vinfo) => vinfo
                        .descr
                        .as_size_descr()
                        .map(|sd| sd.vtable())
                        .filter(|&v| v != 0),
                    _ => None,
                };
                if let Some(vtable) = vtable {
                    let b = ctx.materialize_operand_at(op.pos.get());
                    ctx.make_constant_box(&b, Value::Int(vtable as i64));
                    return OptimizationResult::Remove;
                }
            }
            // virtualize.py:188-189: a field the trace never stored reads the
            // zeroed allocation, so `fieldop is None` folds to
            // `optimizer.new_const(fielddescr)` and the read is dropped.
            // Without the fold the load survives to the arg-forcing pass,
            // which materializes the very virtual it reads: an exception
            // whose traceback slot is read before it is written
            // (`pytraceback.rs`'s `record_application_traceback`) escapes with
            // its args list and the traceback node behind it.
            //
            // Reaching here means `field_val` was `None` and neither header
            // arm answered.  `virtualstate.py:171-174` tolerates a `None`
            // fieldstate and `info.py _force_elements` emits no
            // SETFIELD for a `None` field, so upstream itself depends on the
            // allocation being zeroed -- the fold does not add an assumption
            // the rest of the optimizer lacks.
            //
            // `w_class` and `typeptr` are excluded: both are header fields
            // resolved from class identity above, and neither is ever zero on
            // a live object, so folding them to null/0 would answer a read
            // the allocation does not satisfy. Raw field reads are excluded
            // because upstream defines this handler for GETFIELD_GC_{I,R,F}
            // only.
            let folds_to_zero = !is_raw_op
                && !field_descr.is_header_field()
                && matches!(info, PtrInfo::Virtual(_) | PtrInfo::VirtualStruct(_));
            if folds_to_zero {
                // optimizer.py new_const: CONST_NULL for a pointer
                // field, CONST_ZERO_FLOAT for a float field, else CONST_0.
                let zero = match op.opcode {
                    majit_ir::OpCode::GetfieldGcR => Value::Ref(majit_ir::GcRef::NULL),
                    majit_ir::OpCode::GetfieldGcF => Value::Float(0.0),
                    _ => Value::Int(0),
                };
                let b = ctx.materialize_operand_at(op.pos.get());
                ctx.make_constant_box(&b, zero);
                return OptimizationResult::Remove;
            }
        }
        // virtualize.py:192: self.make_nonnull(op.getarg(0))
        // optimizer.py:437-448: only set NonNull if no existing PtrInfo.
        if !struct_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = struct_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    fn optimize_setarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_box = ctx.resolve_operand_operand_opt(&op.arg(0));
        let value_ref = ctx.resolve_operand_operand(&op.arg(2)).to_opref();

        if let Some(index) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            let idx = index as usize;
            let value_op = ctx.materialize_operand_at(value_ref);
            let did_virtual_write = array_box
                .as_ref()
                .and_then(|b| {
                    ctx.with_ptr_info_mut(b, |info| {
                        if let PtrInfo::VirtualArray(vinfo) = info
                            && idx < vinfo.items.len()
                        {
                            vinfo.items[idx] = value_op.clone();
                            return true;
                        }
                        false
                    })
                })
                .unwrap_or(false);
            if did_virtual_write {
                return OptimizationResult::Remove;
            }
            if let (Some(vt), Some(ab)) = (self.vable.as_ref(), array_box.as_ref()) {
                // Mirror into the virtualizable's tracked array state so a
                // later const-index read folds (read-after-write). The op is
                // KEPT (PassOn → OptHeap lazy set): the real frame array must
                // still be written. Whether the virtual rhs is force-boxed at
                // the export flush is decided in heap.rs `emit_lazy_setfield`,
                // which defers writes whose target is the standard
                // virtualizable (the value then flows virtual through the JUMP).
                vt.mirror_setarrayitem(ab, index, value_ref, ctx);
            }
        } else if let (Some(vt), Some(ab)) = (self.vable.as_ref(), array_box.as_ref()) {
            // Non-constant index: a variable-index write may overwrite any
            // const-index slot, so invalidate the whole tracked array before
            // a later const-index read in `optimize_getarrayitem_gc` can fold
            // to a now-stale value.  `force_lazy_setarrayitem(can_cache=False)`
            // (heap.py:751 variable-index branch -> heap.py:580-586).
            vt.invalidate_array(ab, ctx);
        }
        // virtualize.py:307: self.make_nonnull(op.getarg(0))
        if !array_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = array_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_GETARRAYITEM_GC_I (aliased to R/F and PURE variants)
    fn optimize_getarrayitem_gc(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_box = ctx.resolve_operand_operand_opt(&op.arg(0));

        if let Some(info) = array_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
            && let PtrInfo::VirtualArray(vinfo) = info
            && let Some(index) = ctx
                .resolve_operand_operand_opt(&op.arg(1))
                .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            // info.py: getitem returns None for
            // negative, out-of-range, or uninitialized slots.
            // virtualize.py:282-284: None → InvalidLoop.
            if index < 0 || (index as usize) >= vinfo.items.len() {
                return OptimizationResult::InvalidLoop("virtual array getitem index out of range");
            }
            let item_ref = vinfo.items[index as usize].to_opref();
            if item_ref.is_none() {
                return OptimizationResult::InvalidLoop(
                    "virtual array getitem from uninitialized slot",
                );
            }
            let b_old = Operand::from_bound_op(op_rc);
            let b_item = ctx
                .get_box_replacement_operand_opt(item_ref)
                .unwrap_or_else(|| ctx.materialize_operand_at(item_ref));
            ctx.make_equal_to(&b_old, &b_item);
            return OptimizationResult::Remove;
        }
        // Standard virtualizable array read-after-write: the value-stack /
        // array field of the standard virtualizable frame is tracked in
        // `vstate.arrays` (seeded from the inputarg layout, updated by
        // `mirror_setarrayitem`).  Fold a constant-index read to the
        // tracked box, symmetric with the static-field fold in
        // `optimize_getfield_gc`.  Read-only: the matching setarrayitem is
        // left emitted, so no heap write is dropped.
        if let Some(index) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
            && let Some(item_ref) = self
                .vable
                .as_ref()
                .zip(array_box.as_ref())
                .and_then(|(vt, ab)| vt.tracked_array_element(ab, index, ctx))
        {
            let b_old = ctx.materialize_operand_at(op.pos.get());
            let b_item = ctx.materialize_operand_at(item_ref);
            ctx.make_equal_to(&b_old, &b_item);
            return OptimizationResult::Remove;
        }
        // virtualize.py:287: self.make_nonnull(op.getarg(0))
        if !array_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = array_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_ARRAYLEN_GC
    fn optimize_arraylen_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_box = ctx.resolve_operand_operand_opt(&op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) =
            array_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
        {
            let len = vinfo.items.len() as i64;
            let b = ctx.materialize_operand_at(op.pos.get());
            ctx.make_constant_box(&b, Value::Int(len));
            return OptimizationResult::Remove;
        }
        // virtualize.py:273: self.make_nonnull(op.getarg(0))
        if !array_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = array_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_GETINTERIORFIELD_GC_I (aliased to R/F)
    fn optimize_getinteriorfield_gc(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_box = ctx.resolve_operand_operand_opt(&op.arg(0));
        // `info.py getinteriorfield_virtual` indexes the per-element
        // field list by `fielddescr.get_index()`.  Strip the surrounding
        // `InteriorFieldDescr` first (`descr.py InteriorFieldDescr.
        // __init__` stores the inner `fielddescr`).
        let field_idx = op
            .getdescr()
            .and_then(|d| {
                d.as_interior_field_descr()
                    .map(|ifd| ifd.field_descr().index_in_parent() as u32)
            })
            .expect("optimize_getinteriorfield_gc: op without InteriorFieldDescr");

        if let Some(PtrInfo::VirtualArrayStruct(vinfo)) =
            array_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
            && let Some(index) = ctx
                .resolve_operand_operand_opt(&op.arg(1))
                .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            // info.py _compute_index: negative or out-of-range → -1
            // info.py getinteriorfield_virtual: -1 → None
            // virtualize.py:394-396: None → InvalidLoop
            if index < 0 || (index as usize) >= vinfo.element_fields.len() {
                return OptimizationResult::InvalidLoop(
                    "virtual interior field index out of range",
                );
            }
            let fld = get_field(&vinfo.element_fields[index as usize], field_idx);
            if fld.is_none() {
                return OptimizationResult::InvalidLoop(
                    "virtual interior field from uninitialized slot",
                );
            }
            let fld = fld.unwrap();
            let b_old = Operand::from_bound_op(op_rc);
            let b_fld = ctx
                .get_box_replacement_operand_opt(fld)
                .unwrap_or_else(|| ctx.materialize_operand_at(fld));
            ctx.make_equal_to(&b_old, &b_fld);
            return OptimizationResult::Remove;
        }
        // virtualize.py:399: self.make_nonnull(op.getarg(0))
        if !array_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = array_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_SETINTERIORFIELD_GC
    fn optimize_setinteriorfield_gc(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_box = ctx.resolve_operand_operand_opt(&op.arg(0));
        let value_ref = ctx.resolve_operand_operand(&op.arg(2)).to_opref();
        // `info.py setinteriorfield_virtual` indexes the per-element
        // field list by `fielddescr.get_index()`.  Same shape as the GET
        // counterpart — strip the outer `InteriorFieldDescr` first.
        let field_idx = op
            .getdescr()
            .and_then(|d| {
                d.as_interior_field_descr()
                    .map(|ifd| ifd.field_descr().index_in_parent() as u32)
            })
            .expect("optimize_setinteriorfield_gc: op without InteriorFieldDescr");

        if let Some(index) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            let elem_idx = index as usize;
            let value_op = ctx.materialize_operand_at(value_ref);
            let did_write = array_box
                .as_ref()
                .and_then(|b| {
                    ctx.with_ptr_info_mut(b, |info| {
                        if let PtrInfo::VirtualArrayStruct(vinfo) = info
                            && elem_idx < vinfo.element_fields.len()
                        {
                            set_field(
                                &mut vinfo.element_fields[elem_idx],
                                field_idx,
                                value_op.clone(),
                            );
                            return true;
                        }
                        false
                    })
                })
                .unwrap_or(false);
            if did_write {
                return OptimizationResult::Remove;
            }
        }
        // virtualize.py:413: self.make_nonnull(op.getarg(0))
        if !array_box.as_ref().is_some_and(|b| ctx.has_ptr_info(b))
            && let Some(b) = array_box.as_ref()
        {
            ctx.set_ptr_info(b, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// virtualize.py optimize_INT_ADD
    ///
    /// ```text
    /// def optimize_INT_ADD(self, op):
    ///     opinfo = getrawptrinfo(op.getarg(0))
    ///     offsetbox = self.get_constant_box(op.getarg(1))
    ///     if opinfo and opinfo.is_virtual() and offsetbox is not None:
    ///         offset = offsetbox.getint()
    ///         if (isinstance(opinfo, info.RawBufferPtrInfo) or
    ///             isinstance(opinfo, info.RawSlicePtrInfo)):
    ///             self.make_virtual_raw_slice(offset, opinfo, op)
    ///             return
    ///     return self.emit(op)
    /// ```
    ///
    /// `parent` is the immediate predecessor's PtrInfo (RPython) — in
    /// majit we pass the immediate predecessor's `OpRef`. The slice does
    /// NOT flatten the offset chain at creation time; subsequent
    /// raw_load/store walk the chain via `resolve_raw_slice` and
    /// accumulate offsets. This matches `info.RawSlicePtrInfo.getitem_raw`,
    /// which delegates to `self.parent.getitem_raw(self.offset + offset, ...)`.
    fn optimize_int_add(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if op.num_args() < 2 {
            return OptimizationResult::PassOn;
        }
        let arg0 = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        let Some(offset) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b| ctx.get_constant_int_box(&b))
        else {
            return OptimizationResult::PassOn;
        };
        let info = ctx.peek_ptr_info(&op.arg(0).get_box_replacement(false));
        match info {
            Some(PtrInfo::VirtualRawBuffer(_)) | Some(PtrInfo::VirtualRawSlice(_)) => {
                self.make_virtual_raw_slice(offset, arg0, op, op_rc, ctx);
                OptimizationResult::Remove
            }
            _ => OptimizationResult::PassOn,
        }
    }

    fn optimize_raw_load(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let buf_ref = op.arg(0).to_opref();
        let offset_ref = op.arg(1).to_opref();

        if let Some(offset) = ctx
            .get_box_replacement_operand_opt(offset_ref)
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            // virtualize.py:358-371: walk through RawSlicePtrInfo to the
            // underlying VirtualRawBuffer, accumulating any slice offset.
            let (parent, base_offset) = match Self::resolve_raw_slice(buf_ref, ctx) {
                Some((p, o)) => (p, o),
                None if matches!(
                    ctx.get_box_replacement_operand_opt(buf_ref)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b)),
                    Some(PtrInfo::VirtualRawBuffer(_))
                ) =>
                {
                    (buf_ref, 0)
                }
                None => return OptimizationResult::PassOn,
            };
            let parent_box = ctx.get_box_replacement_operand_opt(parent);
            if let Some(PtrInfo::VirtualRawBuffer(vinfo)) =
                parent_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
            {
                // virtualize.py:362-365: `getitem_raw(offset, ...)` —
                // unbounded signed int arithmetic upstream; in Rust,
                // bail on i64 overflow rather than wrap into a stale
                // matching offset.
                let Some(lookup_offset) = base_offset.checked_add(offset) else {
                    return OptimizationResult::PassOn;
                };
                let Some(descr) = op.getdescr() else {
                    return OptimizationResult::PassOn;
                };
                let Some(ad) = descr.as_array_descr() else {
                    return OptimizationResult::PassOn;
                };
                // rawbuffer.py: read_value(offset, length, descr)
                if let Ok(val_ref) = vinfo.read_value(lookup_offset, ad.item_size(), &descr) {
                    let b_old = Operand::from_bound_op(op_rc);
                    let b_val = ctx
                        .get_box_replacement_operand_opt(val_ref)
                        .unwrap_or_else(|| ctx.materialize_operand_at(val_ref));
                    ctx.make_equal_to(&b_old, &b_val);
                    return OptimizationResult::Remove;
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_raw_store(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let buf_ref = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        let offset_ref = op.arg(1).to_opref();
        let value_ref = ctx.resolve_operand_operand(&op.arg(2)).to_opref();

        if let Some(offset) = ctx
            .get_box_replacement_operand_opt(offset_ref)
            .and_then(|b_| ctx.get_constant_int_box(&b_))
        {
            // virtualize.py:374-385: same slice→parent walk as raw_load.
            let (parent, base_offset) = match Self::resolve_raw_slice(buf_ref, ctx) {
                Some((p, o)) => (p, o),
                None if matches!(
                    ctx.get_box_replacement_operand_opt(buf_ref)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b)),
                    Some(PtrInfo::VirtualRawBuffer(_))
                ) =>
                {
                    (buf_ref, 0)
                }
                None => return OptimizationResult::PassOn,
            };
            // virtualize.py:378: `setitem_raw(offset, ...)` — unbounded
            // signed int upstream; bail on i64 overflow rather than
            // wrap into a colliding offset.
            let Some(store_offset) = base_offset.checked_add(offset) else {
                return OptimizationResult::PassOn;
            };
            let Some(descr) = op.getdescr() else {
                return OptimizationResult::PassOn;
            };
            let Some(ad) = descr.as_array_descr() else {
                return OptimizationResult::PassOn;
            };
            // virtualize.py:374-381: try setitem_raw → return (remove);
            // except InvalidRawOperation → pass → emit(op)
            let item_size = ad.item_size();
            let outcome = ctx.get_box_replacement_operand_opt(parent).and_then(|b| {
                ctx.with_ptr_info_mut(&b, |info| {
                    if let PtrInfo::VirtualRawBuffer(vinfo) = info {
                        Some(
                            vinfo
                                .write_value(store_offset, item_size, descr.clone(), value_ref)
                                .is_ok(),
                        )
                    } else {
                        None
                    }
                })
            });
            match outcome {
                Some(Some(true)) => return OptimizationResult::Remove,
                Some(Some(false)) => return OptimizationResult::PassOn,
                _ => {}
            }
        }
        OptimizationResult::PassOn
    }

    /// `virtualize.py optimize_GETARRAYITEM_RAW_I` (aliased to `_F`):
    ///
    /// ```python
    /// def optimize_GETARRAYITEM_RAW_I(self, op):
    ///     opinfo = getrawptrinfo(op.getarg(0))
    ///     if opinfo and opinfo.is_virtual():
    ///         indexbox = self.get_constant_box(op.getarg(1))
    ///         if indexbox is not None:
    ///             offset, itemsize, descr = self._unpack_arrayitem_raw_op(op, indexbox)
    ///             try:
    ///                 itemvalue = opinfo.getitem_raw(offset, itemsize, descr)
    ///             except InvalidRawOperation:
    ///                 pass
    ///             else:
    ///                 self.make_equal_to(op, itemvalue)
    ///                 return
    ///     self.make_nonnull(op.getarg(0))
    ///     return self.emit(op)
    /// ```
    ///
    /// `_unpack_arrayitem_raw_op` (`virtualize.py`) is inlined: it
    /// just unpacks the array_descr to `(basesize + itemsize*index,
    /// itemsize, descr)` so factoring it out wouldn't share with anything.
    /// Slice walk via `resolve_raw_slice` is the pyre equivalent of
    /// `RawSlicePtrInfo.getitem_raw` (`info.py`) recursing through
    /// `self.parent.getitem_raw(self.offset + offset, ...)`.
    fn optimize_getarrayitem_raw(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_ref = ctx.resolve_operand_operand(&op.arg(0)).to_opref();

        if let Some(index) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
            && let Some(descr) = op.getdescr()
            && let Some(ad) = descr.as_array_descr()
        {
            // resume.py:1544 / `materialize_virtual_from_rd` in
            // pyre/pyre-jit/src/eval.rs
            // `assert not descr.is_array_of_pointers()` at
            // setrawbuffer_item. Upstream's `_I/_F`-only
            // surface guarantees this; pyre carries the
            // assertion through the materialisation path,
            // so a virtualisation handler that admits a
            // pointer descr would panic at resume time.
            // Reject pointer descrs at entry instead.
            if ad.is_array_of_pointers() {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            }
            // virtualize.py _unpack_arrayitem_raw_op:
            // `offset = basesize + (itemsize*index)`. RPython
            // int is unbounded so this is always
            // representable; in Rust we emulate that by using
            // checked arithmetic and falling through (== "no
            // optimisation") on i64 overflow rather than
            // wrapping into a stale offset that could match a
            // sibling write. `itemsize`/`basesize` come from
            // `unpack_arraydescr_size` (RPython unbounded
            // int); `usize → i64` via `try_from` so a
            // pathological descr that exceeds `i64::MAX`
            // bails rather than wrapping into a negative.
            let itemsize_u = ad.item_size();
            let basesize_u = ad.base_size();
            let (Ok(basesize), Ok(itemsize)) =
                (i64::try_from(basesize_u), i64::try_from(itemsize_u))
            else {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            };
            let Some(item_offset) = itemsize
                .checked_mul(index)
                .and_then(|m| basesize.checked_add(m))
            else {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            };
            let resolved = match Self::resolve_raw_slice(array_ref, ctx) {
                Some((p, o)) => Some((p, o)),
                None if matches!(
                    ctx.get_box_replacement_operand_opt(array_ref)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b)),
                    Some(PtrInfo::VirtualRawBuffer(_))
                ) =>
                {
                    Some((array_ref, 0))
                }
                None => None,
            };
            if let Some((parent, base_offset)) = resolved {
                let parent_box = ctx.get_box_replacement_operand_opt(parent);
                if let Some(PtrInfo::VirtualRawBuffer(vinfo)) =
                    parent_box.as_ref().and_then(|b| ctx.peek_ptr_info(b))
                {
                    // rawbuffer.py:89/120 store offsets as
                    // signed: `self.offsets[i] > offset` is a
                    // signed compare. A negative
                    // `lookup_offset` is a valid lookup key
                    // and matches an entry written at the
                    // same negative offset.
                    let Some(lookup_offset) = base_offset.checked_add(item_offset) else {
                        if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                            ctx.make_nonnull(&array_box);
                        }
                        return OptimizationResult::PassOn;
                    };
                    // rawbuffer.py read_value ↔ getitem_raw +
                    // InvalidRawOperation: an `Err` here matches
                    // the upstream `except InvalidRawOperation:
                    // pass` arm — fall through to
                    // make_nonnull + emit.
                    if let Ok(val_ref) = vinfo.read_value(lookup_offset, itemsize_u, &descr) {
                        let b_old = Operand::from_bound_op(op_rc);
                        let b_val = ctx.get_box_replacement_operand(val_ref);
                        ctx.make_equal_to(&b_old, &b_val);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // virtualize.py:332: self.make_nonnull(op.getarg(0)) — for raw
        // arrays this is a no-op because the helper skips `op.type == 'i'`
        // (raw pointer); kept literal so the upstream callsite stays
        // 1:1 with the source.
        if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
            ctx.make_nonnull(&array_box);
        }
        OptimizationResult::PassOn
    }

    /// `virtualize.py optimize_SETARRAYITEM_RAW`:
    ///
    /// ```python
    /// def optimize_SETARRAYITEM_RAW(self, op):
    ///     opinfo = getrawptrinfo(op.getarg(0))
    ///     if opinfo and opinfo.is_virtual():
    ///         indexbox = self.get_constant_box(op.getarg(1))
    ///         if indexbox is not None:
    ///             offset, itemsize, descr = self._unpack_arrayitem_raw_op(op, indexbox)
    ///             itemop = get_box_replacement(op.getarg(2))
    ///             try:
    ///                 opinfo.setitem_raw(offset, itemsize, descr, itemop)
    ///                 return
    ///             except InvalidRawOperation:
    ///                 pass
    ///     self.make_nonnull(op.getarg(0))
    ///     return self.emit(op)
    /// ```
    fn optimize_setarrayitem_raw(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        let value_ref = ctx.resolve_operand_operand(&op.arg(2)).to_opref();

        if let Some(index) = ctx
            .resolve_operand_operand_opt(&op.arg(1))
            .and_then(|b_| ctx.get_constant_int_box(&b_))
            && let Some(descr) = op.getdescr()
            && let Some(ad) = descr.as_array_descr()
        {
            // resume.py:1544 / `materialize_virtual_from_rd` in
            // pyre/pyre-jit/src/eval.rs
            // `assert not descr.is_array_of_pointers()`. A
            // pointer descr stored into the virtual rawbuffer's
            // `descrs[]` would panic at resume materialisation,
            // so reject it at entry. Upstream's `_I/_F`-only
            // surface guarantees this never reaches the
            // optimiser.
            if ad.is_array_of_pointers() {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            }
            // virtualize.py _unpack_arrayitem_raw_op:
            // `offset = basesize + (itemsize*index)`. RPython
            // int is unbounded so this is always
            // representable; bail on i64 overflow rather than
            // wrap into a colliding offset. `usize → i64` via
            // `try_from` for descr sizes that exceed
            // `i64::MAX` (no upstream analogue but defensive
            // against unbounded-int → i64 narrowing).
            let itemsize_u = ad.item_size();
            let basesize_u = ad.base_size();
            let (Ok(basesize), Ok(itemsize)) =
                (i64::try_from(basesize_u), i64::try_from(itemsize_u))
            else {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            };
            let Some(item_offset) = itemsize
                .checked_mul(index)
                .and_then(|m| basesize.checked_add(m))
            else {
                if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                    ctx.make_nonnull(&array_box);
                }
                return OptimizationResult::PassOn;
            };
            let resolved = match Self::resolve_raw_slice(array_ref, ctx) {
                Some((p, o)) => Some((p, o)),
                None if matches!(
                    ctx.get_box_replacement_operand_opt(array_ref)
                        .as_ref()
                        .and_then(|b| ctx.peek_ptr_info(b)),
                    Some(PtrInfo::VirtualRawBuffer(_))
                ) =>
                {
                    Some((array_ref, 0))
                }
                None => None,
            };
            if let Some((parent, base_offset)) = resolved {
                // rawbuffer.py:89 keeps `offsets` sorted by
                // signed compare; a negative store_offset is
                // a legitimate write key.
                let Some(store_offset) = base_offset.checked_add(item_offset) else {
                    if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
                        ctx.make_nonnull(&array_box);
                    }
                    return OptimizationResult::PassOn;
                };
                let outcome = ctx.get_box_replacement_operand_opt(parent).and_then(|b| {
                    ctx.with_ptr_info_mut(&b, |info| {
                        if let PtrInfo::VirtualRawBuffer(vinfo) = info {
                            Some(
                                vinfo
                                    .write_value(store_offset, itemsize_u, descr.clone(), value_ref)
                                    .is_ok(),
                            )
                        } else {
                            None
                        }
                    })
                });
                // rawbuffer.py write_value ↔ setitem_raw +
                // InvalidRawOperation: an `Err` here matches the
                // upstream `except InvalidRawOperation: pass` and
                // falls through to make_nonnull + emit.
                if let Some(Some(true)) = outcome {
                    return OptimizationResult::Remove;
                }
            }
        }
        // virtualize.py:348: self.make_nonnull(op.getarg(0)) — no-op for
        // raw pointers via the helper's `op.type == 'i'` skip; kept
        // literal for callsite parity.
        if let Some(array_box) = ctx.get_box_replacement_operand_opt(array_ref) {
            ctx.make_nonnull(&array_box);
        }
        OptimizationResult::PassOn
    }

    /// Handle VirtualRefR / VirtualRefI.
    ///
    /// virtualize.py optimize_VIRTUAL_REF
    ///
    /// Replace the VIRTUAL_REF operation with a virtual object of type
    /// JitVirtualRef (via make_virtual → InstancePtrInfo / PtrInfo::Virtual).
    /// Two tracked fields:
    /// - virtual_token: set to a ForceToken op
    /// - forced: set to CONST_NULL
    ///   The typeptr/vtable at offset 0 is handled by NEW_WITH_VTABLE when
    ///   the vref is forced — not stored as a tracked virtual field.
    fn optimize_virtual_ref(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // `virtualize.py:140` `vrefinfo = ... metainterp_sd.virtualref_info`
        // / `virtualize.py:123` `vrefinfo.descr` parity.
        let vref_descr: DescrRef = self.vrefinfo.descr.clone();

        // virtualize.py:127: token = ResOperation(rop.FORCE_TOKEN, [])
        let token_op = Op::new(OpCode::ForceToken, &[]);
        let token_ref = ctx.emit_extra(ctx.current_pass_idx, token_op);
        if let Some(b) = ctx.get_box_replacement_operand_opt(token_ref) {
            ctx.set_ptr_info(&b, PtrInfo::nonnull());
        }

        // virtualize.py:129: vrefvalue.setfield(descr_forced, newop, CONST_NULL)
        let null_ref = ctx.emit_constant_ref(majit_ir::GcRef::NULL);

        // virtualize.py: make_virtual(c_cls, newop, vref_descr)
        // → InstancePtrInfo(descr, known_class, is_virtual=True)
        let known_class = Some(crate::virtualref::JIT_VIRTUAL_REF_VTABLE as i64);
        let fields = vec![
            (
                VREF_VIRTUAL_TOKEN_FIELD_INDEX,
                ctx.materialize_operand_at(token_ref),
            ),
            (
                VREF_FORCED_FIELD_INDEX,
                ctx.materialize_operand_at(null_ref),
            ),
        ];
        // info.py:175-188 stores no fielddescr side-list; the SizeDescr
        // (VRefSizeDescr.all_fielddescrs) is the authoritative view.
        let vinfo = VirtualInfo {
            descr: vref_descr,
            known_class,
            ob_type_descr: None,
            fields,
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        };
        let b = Operand::from_bound_op(op_rc);
        ctx.set_ptr_info(&b, PtrInfo::Virtual(vinfo));

        OptimizationResult::Remove
    }

    /// virtualize.py optimize_VIRTUAL_REF_FINISH.
    ///
    /// ```python
    /// def optimize_VIRTUAL_REF_FINISH(self, op):
    ///     vrefinfo = self.optimizer.metainterp_sd.virtualref_info
    ///     seo = self.optimizer.send_extra_operation
    ///
    ///     # - set 'forced' to point to the real object
    ///     objbox = op.getarg(1)
    ///     if not CONST_NULL.same_constant(objbox):
    ///         seo(ResOperation(rop.SETFIELD_GC, op.getarglist(),
    ///                          descr=vrefinfo.descr_forced))
    ///
    ///     # - set 'virtual_token' to TOKEN_NONE (== NULL)
    ///     args = [op.getarg(0), CONST_NULL]
    ///     seo(ResOperation(rop.SETFIELD_GC, args,
    ///                      descr=vrefinfo.descr_virtual_token))
    /// ```
    ///
    /// Two uses:
    /// 1. Normal case: `objbox` is `CONST_NULL` — the frame is being left
    ///    normally. Just clear the vref.virtual_token.
    /// 2. Forced case: `objbox` is the real virtual object — the vref was
    ///    already forced during tracing, so store it into vref.forced.
    ///
    /// majit note: RPython routes the emitted SETFIELD_GCs back through
    /// `send_extra_operation`, which re-enters the virtualize pass and
    /// lets `optimize_setfield_gc` absorb the writes into the vref's
    /// virtual fields if it is still virtual. majit's `emit_extra` skips
    /// the current (virtualize) pass, so the absorption is done in-place
    /// here on the VirtualStruct half and the setfield_gc emit path is
    /// taken only when the vref has already escaped.
    fn optimize_virtual_ref_finish(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let vref_ref = ctx.resolve_operand_operand(&op.arg(0)).to_opref();
        let obj_ref = ctx.resolve_operand_operand(&op.arg(1)).to_opref();

        // virtualize.py: `CONST_NULL.same_constant(objbox)` — only a
        // Ref-typed null constant matches; a plain ConstInt(0) does not.
        // `get_box_replacement` resolves const-namespace OpRefs to their
        // on-demand `Forwarded::Const` and walks the chain terminal;
        // `is_const_null` reads `const_value()` and tolerates an unbound
        // terminal (non-const -> false), so the null check is read-only.
        let obj_box = ctx
            .get_box_replacement_operand_opt(obj_ref)
            .unwrap_or_else(|| ctx.materialize_operand_at(obj_ref));
        let obj_is_null = ctx.is_const_null(&obj_box);

        // If vref is still virtual, update the virtual struct fields directly
        // (majit in-place absorption: `emit_extra` skips the current pass, so
        // there is no `send_extra_operation` re-entry to absorb the writes).
        // virtualize.py:150-153: set 'forced' to point to the real object
        // (skipped when objbox is CONST_NULL).
        let vref_box = ctx.get_box_replacement_operand_opt(vref_ref);
        let obj_op = ctx.materialize_operand_at(obj_ref);
        let did_forced_write = vref_box
            .as_ref()
            .and_then(|b| {
                ctx.with_ptr_info_mut(b, |info| {
                    if !info.is_virtual() {
                        return false;
                    }
                    if let PtrInfo::Virtual(vinfo) = info {
                        if !obj_is_null {
                            set_field(&mut vinfo.fields, VREF_FORCED_FIELD_INDEX, obj_op.clone());
                        }
                        return true;
                    }
                    false
                })
            })
            .unwrap_or(false);
        if did_forced_write {
            // virtualize.py:155-158: set 'virtual_token' to CONST_NULL.
            // emit_constant_ref needs a ctx reborrow, hence two sequential
            // with_ptr_info_mut calls.
            let null_ref = ctx.emit_constant_ref(majit_ir::GcRef(0));
            let null_op = ctx.materialize_operand_at(null_ref);
            if let Some(b) = vref_box.as_ref() {
                ctx.with_ptr_info_mut(b, |info| {
                    if let PtrInfo::Virtual(vinfo) = info {
                        set_field(
                            &mut vinfo.fields,
                            VREF_VIRTUAL_TOKEN_FIELD_INDEX,
                            null_op.clone(),
                        );
                    }
                });
            }
            return OptimizationResult::Remove;
        }

        // vref is not virtual (was forced/escaped): emit SETFIELD_GC ops.

        // virtualize.py:150-153: set 'forced' to the real object via
        // `vrefinfo.descr_forced` (the cached `cpu.fielddescrof(...)`
        // Arc from `virtualref.py:42`).
        if !obj_is_null {
            let arg_vref = ctx.materialize_operand_at(vref_ref);
            let arg_obj = ctx.materialize_operand_at(obj_ref);
            let mut set_forced = Op::new(OpCode::SetfieldGc, &[arg_vref.clone(), arg_obj.clone()]);
            set_forced.setdescr(self.vrefinfo.descr_forced.clone());
            ctx.emit_extra(ctx.current_pass_idx, set_forced);
        }

        // virtualize.py:155-158: set 'virtual_token' to CONST_NULL via
        // `vrefinfo.descr_virtual_token` (`virtualref.py:40-41`).
        let null_ref = ctx.emit_constant_ref(majit_ir::GcRef(0));
        let arg_vref = ctx.materialize_operand_at(vref_ref);
        let arg_null = ctx.materialize_operand_at(null_ref);
        let mut set_token = Op::new(OpCode::SetfieldGc, &[arg_vref.clone(), arg_null.clone()]);
        set_token.setdescr(self.vrefinfo.descr_virtual_token.clone());
        ctx.emit_extra(ctx.current_pass_idx, set_token);

        OptimizationResult::Remove
    }

    /// virtualize.py _optimize_JIT_FORCE_VIRTUAL
    ///
    /// ```python
    /// def _optimize_JIT_FORCE_VIRTUAL(self, op):
    ///     vref = getptrinfo(op.getarg(1))
    ///     vrefinfo = self.optimizer.metainterp_sd.virtualref_info
    ///     if vref and vref.is_virtual():
    ///         tokenop = vref.getfield(vrefinfo.descr_virtual_token, None)
    ///         if tokenop is None:
    ///             return False
    ///         tokeninfo = getptrinfo(tokenop)
    ///         if (tokeninfo is not None and tokeninfo.is_constant() and
    ///                 not tokeninfo.is_nonnull()):
    ///             forcedop = vref.getfield(vrefinfo.descr_forced, None)
    ///             forcedinfo = getptrinfo(forcedop)
    ///             if forcedinfo is not None and not forcedinfo.is_null():
    ///                 self.make_equal_to(op, forcedop)
    ///                 self.last_emitted_operation = REMOVED
    ///                 return True
    ///     return False
    /// ```
    ///
    /// Returns true when the call was eliminated by aliasing `op` to the
    /// already-forced object stored in the vref's `forced` field. The narrow
    /// condition is critical: the vref must be virtual, its `virtual_token`
    /// field must hold a constant null (set by VirtualRefFinish on the normal
    /// frame-leave path), and its `forced` field must point at a non-null
    /// object (set by VirtualRefFinish in the forced-during-tracing path).
    fn optimize_jit_force_virtual(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> bool {
        if op.num_args() < 2 {
            return false;
        }
        // vref = getptrinfo(op.getarg(1)); if vref and vref.is_virtual():
        let (token_ref, forced_ref) = match ctx.peek_ptr_info(&op.arg(1).get_box_replacement(false))
        {
            Some(PtrInfo::Virtual(vinfo)) => {
                // tokenop = vref.getfield(vrefinfo.descr_virtual_token, None)
                // if tokenop is None: return False
                let tok = match get_field(&vinfo.fields, VREF_VIRTUAL_TOKEN_FIELD_INDEX) {
                    Some(r) => r,
                    None => return false,
                };
                // forcedop = vref.getfield(vrefinfo.descr_forced, None)
                let forced = get_field(&vinfo.fields, VREF_FORCED_FIELD_INDEX);
                (tok, forced)
            }
            _ => return false,
        };
        // tokeninfo = getptrinfo(tokenop)
        // if tokeninfo is not None and tokeninfo.is_constant() and not tokeninfo.is_nonnull():
        // The token field is `llmemory.GCREF` upstream
        // (`virtualref.py:17 _virtualref_descrs`); pyre stores it as a
        // `Type::Ref` slot whose constant null is `Value::Ref(GcRef(0))`
        // (see `optimize_virtual_ref_finish`).
        let token_is_constant_null = matches!(
            ctx.get_box_replacement_operand_opt(token_ref).and_then(|b| ctx.get_constant_box(&b)),
            Some(Value::Ref(r)) if r.0 == 0
        );
        if !token_is_constant_null {
            return false;
        }
        // forcedinfo = getptrinfo(forcedop)
        // if forcedinfo is not None and not forcedinfo.is_null():
        let forced_ref = match forced_ref {
            Some(r) if r != OpRef::NONE => r,
            _ => return false,
        };
        // One chain walk; the position view falls back to the source.
        let forced_box = ctx.get_box_replacement_operand_opt(forced_ref);
        let forced_resolved = forced_box.as_ref().map_or(forced_ref, |b| b.to_opref());
        let forced_ok = match forced_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
            Some(info) => !info.is_null(),
            None => false,
        };
        if !forced_ok {
            return false;
        }
        // self.make_equal_to(op, forcedop)
        // `forced_resolved` is the chain terminal of a forced virtual, which
        // is always an emitted producer, so it resolves without minting.
        let b_old = Operand::from_bound_op(op_rc);
        let b_forced = ctx
            .get_box_replacement_operand_opt(forced_resolved)
            .expect("forced virtual terminal must resolve to a bound operand");
        ctx.make_equal_to(&b_old, &b_forced);
        // self.last_emitted_operation = REMOVED
        self.last_emitted_was_removed = true;
        true
    }
}

impl Default for OptVirtualize {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptVirtualize {
    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if let Some(ref mut vt) = self.vable {
            vt.ensure_setup(ctx);
        }
        // optimizer.py base emit/emit_result reset last_emitted_operation
        // to the current op on every emit. RPython's `last_emitted is REMOVED`
        // check therefore reads the prior op's outcome — model that by
        // snapshotting the flag at entry and resetting it. Removal paths
        // (_optimize_JIT_FORCE_VIRTUAL, do_RAW_MALLOC_VARSIZE_CHAR) set the
        // flag back to true before returning Remove. virtualize.py:67-75
        // optimize_GUARD_NO_EXCEPTION / optimize_GUARD_NOT_FORCED read the
        // snapshot.
        let prior_emitted_was_removed = self.last_emitted_was_removed;
        self.last_emitted_was_removed = false;
        match op.opcode {
            // virtualize.py: optimize_NEW_WITH_VTABLE → make_virtual.
            // InstancePtrInfo(descr, known_class, is_virtual=True)
            OpCode::NewWithVtable => self.optimize_new_with_vtable(op, op_rc, ctx),
            OpCode::New => self.optimize_new(op, op_rc, ctx),
            OpCode::NewArray | OpCode::NewArrayClear => self.optimize_new_array(op, op_rc, ctx),

            // Field access on potentially-virtual objects
            OpCode::SetfieldGc | OpCode::SetfieldRaw => self.optimize_setfield_gc(op, ctx),
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF => self.optimize_getfield_gc(op, op_rc, ctx),

            // virtualize.py optimize_SETARRAYITEM_GC vs
            // virtualize.py optimize_SETARRAYITEM_RAW — upstream
            // splits these because the former calls
            // `opinfo.setitem(...)` against `VirtualArray` while the
            // latter calls `opinfo.setitem_raw(...)` against
            // `VirtualRawBuffer/Slice` and catches `InvalidRawOperation`.
            OpCode::SetarrayitemGc => self.optimize_setarrayitem_gc(op, ctx),
            OpCode::SetarrayitemRaw => self.optimize_setarrayitem_raw(op, ctx),
            // virtualize.py:289-296 — GETARRAYITEM_GC_R/F + the PURE
            // variants alias `optimize_GETARRAYITEM_GC_I` (the upstream
            // comment notes the operations are not completely
            // equivalent — `GETARRAYITEM_GC_PURE` is `is_always_pure()`
            // — but the OptVirtualize dispatch is the same).
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF => self.optimize_getarrayitem_gc(op, op_rc, ctx),
            // virtualize.py optimize_GETARRAYITEM_RAW_I (aliased
            // to _F at virtualize.py:334). Upstream's
            // `GETARRAYITEM_RAW` family is `_I/_F` only — RPython
            // resoperation has no `_R` variant.
            //
            // pyre's IR also has `OpCode::GetarrayitemRawR` (raw
            // arrays of GC refs). It is NOT routed through this
            // optimisation: a folded read against `VirtualRawBuffer`
            // would let a pointer descr enter the buffer's
            // `descrs[]`, which `setrawbuffer_item`
            // (`materialize_virtual_from_rd` in
            // pyre/pyre-jit/src/eval.rs) explicitly rejects with
            // `assert !is_array_of_pointers()` at resume
            // materialisation. `_R` therefore falls through the
            // catchall arm to plain emit, mirroring upstream's
            // "no fold for `_R`" surface.
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawF => {
                self.optimize_getarrayitem_raw(op, op_rc, ctx)
            }

            // Array length
            OpCode::ArraylenGc => self.optimize_arraylen_gc(op, ctx),

            // Interior field access on potentially-virtual array-of-structs
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => self.optimize_getinteriorfield_gc(op, op_rc, ctx),
            OpCode::SetinteriorfieldGc => self.optimize_setinteriorfield_gc(op, ctx),

            // virtualize.py optimize_INT_ADD: rawbuf + const → slice
            OpCode::IntAdd => self.optimize_int_add(op, op_rc, ctx),

            // Raw memory access on potentially-virtual raw buffers (and slices)
            OpCode::RawLoadI | OpCode::RawLoadF => self.optimize_raw_load(op, op_rc, ctx),
            OpCode::RawStore => self.optimize_raw_store(op, ctx),

            // RPython virtualize.py does NOT define optimize_GUARD_CLASS,
            // GUARD_NONNULL, GUARD_NONNULL_CLASS, or GUARD_VALUE — these
            // are handled exclusively by rewrite.py. Flow the guards
            // through to the next pass so OptRewrite sees them.
            // emit_guard_operation (mod.rs) calls store_final_boxes_in_guard
            // + force_box on fail_args at emit time, so virtualize does not
            // need to pre-process guard fail_args here.

            // VirtualRef: replace with a virtual struct tracking token + forced fields
            OpCode::VirtualRefR | OpCode::VirtualRefI => self.optimize_virtual_ref(op, op_rc, ctx),
            // VirtualRefFinish: finalize the virtual ref
            OpCode::VirtualRefFinish => self.optimize_virtual_ref_finish(op, ctx),

            // virtualize.py optimize_GUARD_NO_EXCEPTION
            //   if self.last_emitted_operation is REMOVED:
            //       return
            //   return self.emit(op)
            OpCode::GuardNoException => {
                if prior_emitted_was_removed {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // virtualize.py optimize_GUARD_NOT_FORCED
            //   if self.last_emitted_operation is REMOVED:
            //       return
            //   return self.emit(op)
            OpCode::GuardNotForced => {
                if prior_emitted_was_removed {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // virtualize.py optimize_GUARD_NOT_FORCED_2
            //   self._last_guard_not_forced_2 = op
            // The op is NOT emitted here; it is stashed and re-inserted just
            // before the FINISH op in postprocess_FINISH below.
            OpCode::GuardNotForced2 => {
                self.last_guard_not_forced_2 = Some(op.clone());
                OptimizationResult::Remove
            }

            // virtualize.py optimize_CALL_MAY_FORCE_I/R/F/N
            //   if oopspecindex == EffectInfo.OS_JIT_FORCE_VIRTUAL:
            //       if self._optimize_JIT_FORCE_VIRTUAL(op):
            //           return
            //   return self.emit(op)
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(descr) = __descr_arc_descr.as_ref()
                    && let Some(cd) = descr.as_call_descr()
                {
                    let ei = cd.get_extra_info();
                    if ei.oopspecindex == OopSpecIndex::JitForceVirtual
                        && self.optimize_jit_force_virtual(op, op_rc, ctx)
                    {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            // virtualize.py optimize_FINISH / postprocess_FINISH
            //
            //   def optimize_FINISH(self, op):
            //       self._finish_guard_op = self._last_guard_not_forced_2
            //       return self.emit(op)
            //
            //   def postprocess_FINISH(self, op):
            //       guard_op = self._finish_guard_op
            //       if guard_op is not None:
            //           guard_op = self.optimizer.store_final_boxes_in_guard(
            //               guard_op, [])
            //           i = len(self.optimizer._newoperations) - 1
            //           assert i >= 0
            //           self.optimizer._newoperations.insert(i, guard_op)
            //
            // The stash here is only half the port: the guard has to be
            // finalized and inserted AFTER the FINISH is emitted, because
            // `emit(op)` force_box's the FINISH args and
            // `store_final_boxes_in_guard` has to see a return box that was
            // virtual as already materialized.  Finalizing it on the way
            // through the pipeline instead would encode that same box as still
            // virtual — a consistent image, but not upstream's image.
            //
            // `propagate_postprocess` below runs at the right moment but is a
            // method on a PASS, and both the finalization and the
            // `collect_optimizer_knowledge_for_resume` that feeds it are
            // Optimizer-side.  So it hands the guard to the Optimizer through
            // `ctx.pending_finish_guard_postprocess`, the same shape
            // `pending_guard_class_postprocess` uses, and
            // `drain_pending_finish_guard_postprocess` does the insert.
            OpCode::Finish => {
                self.finish_guard_op = self.last_guard_not_forced_2.take();
                OptimizationResult::PassOn
            }

            // virtualize.py: optimize_COND_CALL — if the call is
            // OS_JIT_FORCE_VIRTUALIZABLE and the target is virtual, remove.
            OpCode::CondCallN => {
                let __descr_arc_descr = op.getdescr();
                if let Some(descr) = __descr_arc_descr.as_ref()
                    && let Some(cd) = descr.as_call_descr()
                {
                    let ei = cd.get_extra_info();
                    if ei.oopspecindex == OopSpecIndex::JitForceVirtualizable
                        && op.num_args() >= 3
                        && Self::is_virtual(op.arg(2).to_opref(), ctx)
                    {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            // virtualize.py optimize_CALL_N (aliased to CALL_R / CALL_I)
            //
            //   def optimize_CALL_N(self, op):
            //       effectinfo = op.getdescr().get_extra_info()
            //       if effectinfo.oopspecindex == EffectInfo.OS_RAW_MALLOC_VARSIZE_CHAR:
            //           return self.do_RAW_MALLOC_VARSIZE_CHAR(op)
            //       elif effectinfo.oopspecindex == EffectInfo.OS_RAW_FREE:
            //           return self.do_RAW_FREE(op)
            //       elif effectinfo.oopspecindex == EffectInfo.OS_JIT_FORCE_VIRTUALIZABLE:
            //           # we might end up having CALL here instead of COND_CALL
            //           info = getptrinfo(op.getarg(1))
            //           if info and info.is_virtual():
            //               return
            //       else:
            //           return self.emit(op)
            //
            // The Python control flow is significant: when oopspecindex is
            // JIT_FORCE_VIRTUALIZABLE, the function falls off without
            // emitting regardless of whether the inner is_virtual check
            // succeeds — the elif chain blocks the else: emit branch.
            //
            // CALL_F is NOT in the alias list (RPython virtualize.py defines
            // only optimize_CALL_N/R/I) — float-typed calls flow through the
            // base Optimization.emit and only get virtual-arg forcing in the
            // standard force_box path.
            OpCode::CallN | OpCode::CallR | OpCode::CallI => {
                let __descr_arc_descr = op.getdescr();
                if let Some(descr) = __descr_arc_descr.as_ref()
                    && let Some(cd) = descr.as_call_descr()
                {
                    let ei = cd.get_extra_info();
                    // virtualize.py do_RAW_MALLOC_VARSIZE_CHAR
                    if ei.oopspecindex == OopSpecIndex::RawMallocVarsizeChar {
                        // virtualize.py do_RAW_MALLOC_VARSIZE_CHAR:
                        //   sizebox = self.get_constant_box(op.getarg(1))
                        //   if sizebox is None:
                        //       return self.emit(op)
                        //   self.make_virtual_raw_memory(sizebox.getint(), op)
                        //   self.last_emitted_operation = REMOVED
                        if op.num_args() >= 2
                            && let Some(size) =
                                ctx.get_constant_int_box(&op.arg(1).get_box_replacement(false))
                        {
                            // virtualize.py:53 func = source_op.getarg(0).getint()
                            let func =
                                op.arg(0).get_box_replacement(false).const_int().expect(
                                    "virtualize.py:53 source_op.getarg(0) must be ConstInt",
                                );
                            self.make_virtual_raw_memory(size as usize, func, op, op_rc, ctx);
                            self.last_emitted_was_removed = true;
                            return OptimizationResult::Remove;
                        }
                        return OptimizationResult::PassOn;
                    }
                    // virtualize.py do_RAW_FREE
                    if ei.oopspecindex == OopSpecIndex::RawFree {
                        // virtualize.py do_RAW_FREE:
                        //   opinfo = getrawptrinfo(op.getarg(1))
                        //   if opinfo and opinfo.is_virtual():
                        //       return
                        //   return self.emit(op)
                        if op.num_args() >= 2 && Self::is_virtual(op.arg(1).to_opref(), ctx) {
                            return OptimizationResult::Remove;
                        }
                        return OptimizationResult::PassOn;
                    }
                    // virtualize.py:232-236 OS_JIT_FORCE_VIRTUALIZABLE
                    //   info = getptrinfo(op.getarg(1))
                    //   if info and info.is_virtual():
                    //       return
                    //   # falls off (no else branch matches) → REMOVED
                    if ei.oopspecindex == OopSpecIndex::JitForceVirtualizable {
                        return OptimizationResult::Remove;
                    }
                }
                // virtualize.py:237-238 else: return self.emit(op)
                OptimizationResult::PassOn
            }

            // RecordKnownResult + CallPure must pass through to OptPure
            // for @elidable constant folding. Must appear BEFORE is_call()
            // since they are in the CALL opcode range.
            OpCode::RecordKnownResult => OptimizationResult::PassOn,
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                OptimizationResult::PassOn
            }

            // Calls / escaping operations — force all virtual args
            _ if op.opcode.is_call() => OptimizationResult::PassOn,

            // RPython virtualize.py has no optimize_JUMP. JUMP is held
            // out of the pass pipeline (flush=False at optimizer.py)
            // or sent through via send_extra_operation in flush=True, which
            // dispatches to the standard emit path — no virtualize-specific
            // handler. Falling through to the default PassOn matches RPython.

            // RECORD_EXACT_CLASS / RECORD_EXACT_VALUE_I / RECORD_EXACT_VALUE_R:
            // Handled by OptRewrite (rewrite.py), not virtualize.py.
            // PassOn forwards them to rewrite which runs before virtualize
            // in the default pipeline — these should already be consumed
            // before reaching this pass. Keep as PassOn for robustness.

            // virtualize.py dispatch_opt = make_dispatcher_method(
            //     OptVirtualize, 'optimize_', default=OptVirtualize.emit)
            // The default for unhandled opcodes is the base Optimization.emit
            // which forwards to the next pass without touching args. Forcing
            // virtual args and fail_args happens at the terminal Optimizer
            // emit step (optimizer.py _emit_operation /
            // emit_guard_operation), which majit mirrors in
            // OptContext::emit / emit_guard_operation.
            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {
        self.last_emitted_was_removed = false;
        self.last_guard_not_forced_2 = None;
        if let Some(ref mut vt) = self.vable {
            vt.setup();
        }
        self.finish_guard_op = None;
    }

    // virtualize.py postprocess_FINISH
    //
    //   def postprocess_FINISH(self, op):
    //       guard_op = self._finish_guard_op
    //       if guard_op is not None:
    //           guard_op = self.optimizer.store_final_boxes_in_guard(guard_op, [])
    //           i = len(self.optimizer._newoperations) - 1
    //           assert i >= 0
    //           self.optimizer._newoperations.insert(i, guard_op)
    //
    // The two Optimizer-side halves are in
    // `Optimizer::drain_pending_finish_guard_postprocess`, which this hands
    // the guard to.
    fn have_postprocess_op(&self, opcode: OpCode) -> bool {
        matches!(opcode, OpCode::Finish)
    }

    fn propagate_postprocess(&mut self, op: &Op, ctx: &mut OptContext) {
        if op.opcode != OpCode::Finish {
            return;
        }
        if let Some(guard_op) = self.finish_guard_op.take() {
            debug_assert!(
                ctx.pending_finish_guard_postprocess.is_none(),
                "postprocess_FINISH queued multiple guards"
            );
            ctx.pending_finish_guard_postprocess = Some(guard_op);
        }
    }

    fn name(&self) -> &'static str {
        "virtualize"
    }

    fn set_vrefinfo(&mut self, vrefinfo: crate::virtualref::VirtualRefInfo) {
        self.vrefinfo = vrefinfo;
    }
}

// PtrInfo helpers (is_nonnull, is_virtual, etc.) are in info.rs.

// ── Field list helpers ──

/// The postcondition of a virtual `set_field`: slot `field_idx` of the struct
/// descr this PtrInfo carries is the field that supplied the index.
///
/// `info.py:206` writes `self._fields[fielddescr.get_index()]`, and
/// `info.py _force_elements` reads it back as
/// `for i, fielddescr in enumerate(descr.get_all_fielddescrs()): fld =
/// self._fields[i]` — two descrs, one index. Upstream cannot
/// disagree, because `heaptracker.py all_fielddescrs` and `:97-109
/// get_fielddescr_index_in` are one declaration-order walk. Nor can the two
/// descrs be a mismatched pair: `rclass.py:549` declares a subclass as
/// `MkStruct(name, ('super', rbase.object_type), *llfields)`, so the inherited
/// fields are walked first and a field's index is the same in a class as in
/// every subclass of it. `info.py:184-188` spends exactly that guarantee when
/// it swaps `self.descr` for "a more precise descr" and keeps the index.
///
/// pyre reaches one such list from two producers that rank fields differently —
/// the codewriter walks declarations (`codewriter/assembler.rs
/// bh_all_field_specs_for_struct_into`), `jitcode/assembler.rs
/// register_struct_layout` sorts by byte offset — so the pairing above is a
/// postcondition to state rather than a property of the construction. Measured
/// over 120928 virtual setfields (1172 programs) it holds everywhere, including
/// the 2286 that did index a descr other than the field's own parent; the
/// reachable failure needs a reordered list AND that cross-descr step at once,
/// which nothing in the corpus produces. A census over what ran cannot promise
/// what the producers can emit, so check it here — the same argument
/// `jitcode/assembler.rs field_descr_position_disagreement` already makes for
/// the other end of this pipe.
///
/// Returns the disagreement as a message so the caller's panic names both the
/// slot and the field. Gated on [`jit_strict_mode`], not `cfg!(debug_assertions)`:
/// the producers this states a postcondition over are exercised by the release
/// corpus far more widely than by the unit tests, and a check that is inert
/// exactly where the wide corpus runs covers the narrow half only. Off in plain
/// release so production keeps graceful degradation. Written as a function
/// rather than a `debug_assert!` because the message needs the same walk the
/// predicate does. [`field_slot_identifies`] is the same walk without the
/// message, for the read side, which has to answer in release too.
///
/// [`jit_strict_mode`]: crate::jit_strict_mode
fn field_slot_disagreement(
    descr: &DescrRef,
    field_idx: u32,
    field: &dyn FieldDescr,
) -> Option<String> {
    if !crate::jit_strict_mode() {
        return None;
    }
    let fields = descr.as_size_descr()?.all_fielddescrs();
    let Some(slot) = fields.get(field_idx as usize) else {
        return Some(format!(
            "field slot {field_idx} is outside its own descr's field list (len {}, descr \
             index {}); the slot is past the end of the struct this PtrInfo describes",
            fields.len(),
            descr.index(),
        ));
    };
    if !slot_holds_field(slot.as_ref(), field) {
        return Some(format!(
            "field {:?} at offset {} claims slot {field_idx} of descr index {}, but that \
             slot holds {:?} at offset {}",
            field.field_name(),
            field.offset(),
            descr.index(),
            slot.field_name(),
            slot.offset(),
        ));
    }
    None
}

/// Whether `slot` and `field` name the same field.
///
/// Both halves must agree. The name is the better key but is not always
/// carried — the flattened inline aggregates (`ob_header`, an enum's `__pos_0`)
/// reach here under the documented empty-name fallback — so the name is
/// compared only when both sides have one, and the offset is compared always.
/// Neither alone is sufficient: a name can be absent, and a flattened layout
/// puts an aggregate and its first leaf at one address
/// (`heaptracker.py:68-69`).
pub(crate) fn slot_holds_field(slot: &dyn FieldDescr, field: &dyn FieldDescr) -> bool {
    let named_apart = !field.field_name().is_empty()
        && !slot.field_name().is_empty()
        && slot.field_name() != field.field_name();
    !named_apart && slot.offset() == field.offset()
}

/// Whether `field_idx` addresses `field` in the struct `descr` describes.
///
/// The read side's release-live half of [`field_slot_disagreement`]. The write
/// side can panic on a disagreement because a wrong store is unrecoverable; a
/// read has a correct answer available — a field the slot list does not hold
/// was never stored under its own identity, so `virtualize.py:188`'s zeroed
/// allocation is what it reads — so it answers that instead of aborting, and
/// has to be able to answer it in a release build.
///
/// A descr that is not a size descr answers `true`: the caller has no field
/// list to check against, which is the state every pre-existing read was
/// resolved in and not a disagreement this can see.
fn field_slot_identifies(descr: &DescrRef, field_idx: u32, field: &dyn FieldDescr) -> bool {
    let Some(size_descr) = descr.as_size_descr() else {
        return true;
    };
    size_descr
        .all_fielddescrs()
        .get(field_idx as usize)
        .is_some_and(|slot| slot_holds_field(slot.as_ref(), field))
}

/// Whether `field_idx` addresses `field` in a virtualizable's slot list.
///
/// The virtualizable half of [`field_slot_identifies`], which cannot serve
/// here. `VirtualizableFieldState.fields` is keyed in `VirtualizableInfo::
/// static_fields` order while the read computes `index_in_parent`
/// (`majit-ir ptr_info.rs` spells both), so `vstate.descr`'s field list does
/// not index that slot and cannot say what it holds. What the state does carry
/// is the descr each store recorded, so the check runs against that; a slot no
/// store described falls back to the value's type, which is what separates an
/// `Int` field read from a slot holding a `Ref`.
///
/// A slot that does not identify the field leaves the read unfolded. That is
/// the answer a virtualizable needs: `folds_to_zero` covers only `Virtual` and
/// `VirtualStruct`, so the caller emits the real load instead of reading a
/// zeroed allocation off a live frame.
fn virtualizable_slot_identifies(
    vstate: &VirtualizableFieldState,
    field_idx: u32,
    field: &dyn FieldDescr,
    stored: &Operand,
    result_type: Type,
) -> bool {
    match get_field_descr(&vstate.field_descrs, field_idx) {
        Some(slot) => slot
            .as_field_descr()
            .is_some_and(|slot| slot_holds_field(slot, field)),
        None => stored.type_() == result_type,
    }
}

fn set_field(fields: &mut Vec<(u32, Operand)>, field_idx: u32, value: Operand) {
    for entry in fields.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = value.clone();
            return;
        }
    }
    fields.push((field_idx, value));
}

fn set_field_descr(field_descrs: &mut Vec<(u32, DescrRef)>, field_idx: u32, descr: DescrRef) {
    for entry in field_descrs.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = descr;
            return;
        }
    }
    field_descrs.push((field_idx, descr));
}

fn get_field_descr(field_descrs: &[(u32, DescrRef)], field_idx: u32) -> Option<DescrRef> {
    field_descrs
        .iter()
        .find(|(idx, _)| *idx == field_idx)
        .map(|(_, descr)| descr.clone())
}

fn get_field(fields: &[(u32, Operand)], field_idx: u32) -> Option<OpRef> {
    fields
        .iter()
        .find(|(idx, _)| *idx == field_idx)
        .map(|(_, b)| b.to_opref())
}

#[derive(Debug)]
struct VRefFieldDescr {
    index: u32,
    offset: usize,
    field_type: Type,
}

impl Descr for VRefFieldDescr {
    fn index(&self) -> u32 {
        self.index
    }

    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for VRefFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }

    fn field_size(&self) -> usize {
        // `virtualref.py:17-20` declares both fields as pointer types, so
        // source translation gives them the target pointer width.
        std::mem::size_of::<*mut u8>()
    }

    fn field_type(&self) -> majit_ir::Type {
        self.field_type
    }

    fn index_in_parent(&self) -> usize {
        self.index as usize
    }

    fn get_parent_descr(&self) -> Option<DescrRef> {
        Some(vref_size_descr())
    }
}

/// `virtualref.py:40-42` parity: process-static Arc cache for the
/// `descr_virtual_token` / `descr_forced` field descrs.  PyPy stores
/// these on `VirtualRefInfo` (one instance per `cpu`); pyre's single
/// `MetaInterp` per process collapses to the same identity by caching
/// at module level.  Every `make_vref_field_descr(VREF_*)` call and
/// every `VREF_ALL_FIELDDESCRS` index returns the same Arc — the
/// `Arc::ptr_eq` identity `history.py:125` demands for `descr is
/// other_descr` comparisons.
static VREF_DESCR_VIRTUAL_TOKEN: std::sync::LazyLock<Arc<VRefFieldDescr>> =
    std::sync::LazyLock::new(|| build_vref_field_descr(VREF_VIRTUAL_TOKEN_FIELD_INDEX));

static VREF_DESCR_FORCED: std::sync::LazyLock<Arc<VRefFieldDescr>> =
    std::sync::LazyLock::new(|| build_vref_field_descr(VREF_FORCED_FIELD_INDEX));

/// `virtualref.py:32-33` parity: process-static Arc cache for the
/// `descr = cpu.sizeof(JIT_VIRTUAL_REF)` slot.
static VREF_SIZE_DESCR: std::sync::LazyLock<Arc<VRefSizeDescr>> =
    std::sync::LazyLock::new(|| Arc::new(VRefSizeDescr));

#[allow(dead_code)]
fn make_vref_field_descr(index: u32) -> DescrRef {
    make_vref_field_descr_typed(index)
}

/// `virtualref.py` parity helper for `VirtualRefInfo::new()`:
/// returns the same cached `DescrRef` `make_vref_field_descr` hands
/// out, so the descrs stored on `VirtualRefInfo.descr_virtual_token`
/// / `descr_forced` share identity with the Arcs the
/// `optimize_virtual_ref_finish` emit sites stamp onto SETFIELD_GC
/// ops.  Without this shared identity, `Arc::ptr_eq` checks (e.g.
/// the heap pass's stale-set canonicalization) would split into two
/// equivalence classes per field.
pub(crate) fn make_vref_field_descr_pub(index: u32) -> DescrRef {
    make_vref_field_descr_typed(index)
}

fn make_vref_field_descr_typed(index: u32) -> Arc<VRefFieldDescr> {
    match index {
        VREF_VIRTUAL_TOKEN_FIELD_INDEX => VREF_DESCR_VIRTUAL_TOKEN.clone(),
        VREF_FORCED_FIELD_INDEX => VREF_DESCR_FORCED.clone(),
        _ => panic!("invalid JitVirtualRef field slot {index}"),
    }
}

pub(crate) fn vref_size_descr() -> DescrRef {
    VREF_SIZE_DESCR.clone() as DescrRef
}

/// One-shot constructor used only by the `LazyLock` initializers above
/// — never call this directly; always go through
/// `make_vref_field_descr_typed` so cached identity is preserved.
fn build_vref_field_descr(index: u32) -> Arc<VRefFieldDescr> {
    let (offset, field_type) = match index {
        // `virtualref.py:17` registers `virtual_token` and `forced` both
        // as `llmemory.GCREF` slots; the rtyper writes them through
        // `setfield_gc_r`.  Pyre's slot type must match so
        // `optimize_virtual_ref_finish`'s `Value::Ref(GcRef(0))` write
        // and `optimize_jit_force_virtual`'s constant-null read agree
        // on the value tag.
        //
        // `virtualref.py:17-20` makes these fields part of the translated
        // JitVirtualRef structure; use that structure's target layout rather
        // than assuming native 64-bit pointer offsets.
        VREF_VIRTUAL_TOKEN_FIELD_INDEX => (
            std::mem::offset_of!(crate::virtualref::JitVirtualRef, virtual_token),
            Type::Ref,
        ),
        VREF_FORCED_FIELD_INDEX => (
            std::mem::offset_of!(crate::virtualref::JitVirtualRef, forced),
            Type::Ref,
        ),
        _ => panic!("invalid JitVirtualRef field slot {index}"),
    };
    Arc::new(VRefFieldDescr {
        index,
        offset,
        field_type,
    })
}

/// `virtualref.py:17-20` target-layout size descriptor for JitVirtualRef.
#[derive(Debug)]
struct VRefSizeDescr;

/// virtualref.py:17 registers JitVirtualRef with two fields:
/// `virtual_token` (slot 0) and `forced` (slot 1). Mirror that here so
/// `SizeDescr::all_fielddescrs()` returns the descriptor-order pair —
/// `info::all_fielddescrs_from_descr` consumes this view at force-box
/// and visitor-dispatch sites (`info.rs`).
static VREF_ALL_FIELDDESCRS: std::sync::LazyLock<Vec<Arc<dyn majit_ir::FieldDescr>>> =
    std::sync::LazyLock::new(|| {
        vec![
            make_vref_field_descr_typed(VREF_VIRTUAL_TOKEN_FIELD_INDEX)
                as Arc<dyn majit_ir::FieldDescr>,
            make_vref_field_descr_typed(VREF_FORCED_FIELD_INDEX) as Arc<dyn majit_ir::FieldDescr>,
        ]
    });

impl Descr for VRefSizeDescr {
    fn index(&self) -> u32 {
        VREF_SIZE_DESCR_INDEX
    }
    fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
        Some(self)
    }
}

impl majit_ir::SizeDescr for VRefSizeDescr {
    fn size(&self) -> usize {
        std::mem::size_of::<crate::virtualref::JitVirtualRef>()
    }
    fn type_id(&self) -> u32 {
        crate::virtualref::vref_gc_type_id()
    }
    fn is_object(&self) -> bool {
        true
    }
    fn vtable(&self) -> usize {
        // virtualref.py:94-98: jit_virtual_ref_const_class — the vtable
        // identity used by is_virtual_ref(). Pyre stores this as the
        // JIT_VIRTUAL_REF_VTABLE magic value at offset 0
        // (super_.typeptr). NEW_WITH_VTABLE writes it at allocation
        // time, matching RPython's gc.new_with_vtable().
        crate::virtualref::JIT_VIRTUAL_REF_VTABLE
    }
    fn is_immutable(&self) -> bool {
        false
    }
    fn all_fielddescrs(&self) -> &[Arc<dyn majit_ir::FieldDescr>] {
        &VREF_ALL_FIELDDESCRS
    }
}

/// Lookup helper for `PtrInfo::Virtualizable.arrays` — returns the OpRef
/// stored at `arrays[arr_idx][elem_idx]` if present and non-NONE.
fn get_array_element(
    arrays: &[(u32, Vec<Operand>)],
    arr_idx: u32,
    elem_idx: usize,
) -> Option<OpRef> {
    arrays
        .iter()
        .find(|(i, _)| *i == arr_idx)
        .and_then(|(_, e)| e.get(elem_idx).map(|b| b.to_opref()))
        .filter(|r| !r.is_none())
}

/// Write helper for `PtrInfo::Virtualizable.arrays` — grows the inner Vec
/// with `OpRef::NONE` placeholders as needed, then stores `value` at
/// `arr_idx`/`elem_idx`.
fn set_array_element(
    arrays: &mut Vec<(u32, Vec<Operand>)>,
    arr_idx: u32,
    elem_idx: usize,
    value: Operand,
) {
    if let Some((_, elems)) = arrays.iter_mut().find(|(i, _)| *i == arr_idx) {
        if elem_idx >= elems.len() {
            elems.resize(elem_idx + 1, Operand::None);
        }
        elems[elem_idx] = value;
    } else {
        let mut elems = vec![Operand::None; elem_idx + 1];
        elems[elem_idx] = value;
        arrays.push((arr_idx, elems));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::info::RawBufferPtrInfo;
    use crate::optimizeopt::optimizer::Optimizer;
    use std::sync::Arc;

    // ── Test descriptors ──

    #[derive(Debug)]
    struct TestSizeDescr {
        idx: u32,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.idx
        }
        fn is_immutable(&self) -> bool {
            false
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        idx: u32,
    }

    #[derive(Debug)]
    struct TestParentSizeDescr {
        idx: u32,
        #[allow(dead_code)]
        field_type: majit_ir::Type,
        all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    }

    #[derive(Debug)]
    struct TestParentFieldDescr {
        idx: u32,
        field_type: majit_ir::Type,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_size_descr(self.idx, majit_ir::Type::Int))
        }
        fn index_in_parent(&self) -> usize {
            self.idx as usize
        }
        fn offset(&self) -> usize {
            self.idx as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    /// Ref-typed counterpart to `TestFieldDescr`. Identical semantics
    /// except `field_type() == Type::Ref`; used by test fixtures that
    /// need a Ref-valued field (e.g. a `next` pointer in a linked
    /// node). Both implementations override `get_parent_descr` to
    /// return a fresh parent-backed SizeDescr each call so stale
    /// hand-written descriptors still obey the optimizer's
    /// "non-typeptr fields always know their parent" contract.
    #[derive(Debug)]
    struct TestRefFieldDescr {
        idx: u32,
    }

    impl Descr for TestRefFieldDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestRefFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_size_descr(self.idx, majit_ir::Type::Ref))
        }
        fn offset(&self) -> usize {
            self.idx as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Ref
        }
        fn index_in_parent(&self) -> usize {
            self.idx as usize
        }
    }

    #[derive(Debug)]
    struct TestFloatFieldDescr {
        idx: u32,
    }

    impl Descr for TestFloatFieldDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFloatFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_size_descr(self.idx, majit_ir::Type::Float))
        }
        fn offset(&self) -> usize {
            self.idx as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Float
        }
        fn index_in_parent(&self) -> usize {
            self.idx as usize
        }
    }

    impl Descr for TestParentSizeDescr {
        fn index(&self) -> u32 {
            0xFFFF_0000 | self.idx
        }
        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for TestParentSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            0xFFFF_0000 | self.idx
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.all_fielddescrs
        }
    }

    impl Descr for TestParentFieldDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestParentFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            None
        }
        fn index_in_parent(&self) -> usize {
            self.idx as usize
        }
        fn offset(&self) -> usize {
            self.idx as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            self.field_type
        }
    }

    /// A field descr that CLAIMS a slot it does not occupy: `index_in_parent`
    /// answers `claimed_idx` while `offset` answers a genuinely different
    /// field's address, so the parent's `all_fielddescrs()[claimed_idx]` and
    /// this descr do not name the same field.
    ///
    /// This is descriptor census's "in-range but naming a different slot" mint reduced to
    /// two descrs. It is deliberately NOT out-of-range: `get_field` searches a
    /// `Vec<(u32, Operand)>` by key and cannot index past its end, so the
    /// out-of-range rows reach `force_box_impl`'s `.get(idx).expect(..)`, a
    /// different consumer with a different (loud) failure.
    #[derive(Debug)]
    struct MisindexedFieldDescr {
        claimed_idx: u32,
        real_offset: usize,
    }

    impl Descr for MisindexedFieldDescr {
        fn index(&self) -> u32 {
            0xDEAD_0000 | self.claimed_idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for MisindexedFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(test_parent_size_descr(
                self.claimed_idx,
                majit_ir::Type::Int,
            ))
        }
        fn index_in_parent(&self) -> usize {
            self.claimed_idx as usize
        }
        fn offset(&self) -> usize {
            self.real_offset
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    fn misindexed_field_descr(claimed_idx: u32, real_offset: usize) -> DescrRef {
        Arc::new(MisindexedFieldDescr {
            claimed_idx,
            real_offset,
        })
    }

    /// A field descr whose parent carries NO layout: `get_parent_descr()`
    /// answers a descr for which `as_size_descr()` is `None`.
    ///
    /// This is the one shape that lets a POPULATED slot coexist with
    /// `cur_len == 0`, which is what `init_fields`' first arm needs to be
    /// observable on the read path. Two short-circuits do it, both on the same
    /// `as_size_descr()`:
    ///
    /// - `init_fields` opens with
    ///   `let Some(size_descr) = descr.as_size_descr() else { return; }`
    ///   (`ptr_info.rs`), so the setfield's own `init_fields` leaves the
    ///   virtual's descr exactly as the allocation set it. Any
    ///   size-descr-parented field descr would instead take the `cur_len == 0`
    ///   arm right there and close the window before the read is reached.
    /// - `field_slot_disagreement` opens with `descr.as_size_descr()?`
    ///   (this file), so the write is not refused and no panic fires.
    #[derive(Debug)]
    struct NarrowParentFieldDescr {
        idx: u32,
    }

    impl Descr for NarrowParentFieldDescr {
        fn index(&self) -> u32 {
            0xBEEF_0000 | self.idx
        }
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for NarrowParentFieldDescr {
        fn get_parent_descr(&self) -> Option<DescrRef> {
            // Deliberately NOT a SizeDescr: `TestArrayDescr` implements only
            // `Descr::index`, so `as_size_descr()` falls through to the trait
            // default and answers `None`. The `0xA000` tag keeps this parent's
            // descr index clear of the bare `idx` that `size_descr`/
            // `field_descr` mint, so nothing keyed on index can confuse them.
            Some(Arc::new(TestArrayDescr {
                idx: 0xA000 | self.idx,
            }))
        }
        fn index_in_parent(&self) -> usize {
            self.idx as usize
        }
        fn offset(&self) -> usize {
            self.idx as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    fn narrow_parent_field_descr(idx: u32) -> DescrRef {
        Arc::new(NarrowParentFieldDescr { idx })
    }

    fn test_parent_size_descr(idx: u32, field_type: majit_ir::Type) -> DescrRef {
        let all_fielddescrs: Vec<Arc<dyn FieldDescr>> = (0..=idx)
            .map(|field_idx| {
                Arc::new(TestParentFieldDescr {
                    idx: field_idx,
                    field_type: if field_idx == idx {
                        field_type
                    } else {
                        majit_ir::Type::Int
                    },
                }) as Arc<dyn FieldDescr>
            })
            .collect();
        Arc::new(TestParentSizeDescr {
            idx,
            field_type,
            all_fielddescrs,
        })
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        idx: u32,
    }

    impl Descr for TestArrayDescr {
        fn index(&self) -> u32 {
            self.idx
        }
    }

    fn size_descr(idx: u32) -> DescrRef {
        Arc::new(TestSizeDescr { idx })
    }

    /// An allocation descr that is NOT a `SizeDescr` — `as_size_descr()`
    /// answers `None`, so a virtual allocated with it starts at `cur_len == 0`
    /// via `init_fields`' `.map(..).unwrap_or(0)` (`ptr_info.rs`).
    fn non_size_descr(idx: u32) -> DescrRef {
        Arc::new(TestArrayDescr { idx })
    }

    fn field_descr(idx: u32) -> DescrRef {
        Arc::new(TestFieldDescr { idx })
    }

    fn ref_field_descr(idx: u32) -> DescrRef {
        // ensure_ptr_info_arg0 (`optimizeopt/mod.rs`) requires field descrs flowing
        // into GETFIELD/SETFIELD to carry a parent_descr backreference per
        // optimizer.py:478. TestRefFieldDescr mirrors TestFieldDescr but
        // for Ref-typed slots, returning a fresh parent SizeDescr on each
        // `get_parent_descr()` call so the test doesn't need to keep a
        // Weak parent alive across the test body.
        Arc::new(TestRefFieldDescr { idx })
    }

    fn float_field_descr(idx: u32) -> DescrRef {
        Arc::new(TestFloatFieldDescr { idx })
    }

    fn array_descr(idx: u32) -> DescrRef {
        Arc::new(TestArrayDescr { idx })
    }

    /// Test helper: build a `FieldDescr` with explicit `offset` and
    /// `index_in_parent` for the virtualizable-field test sites.  Mirrors
    /// the shape `cpu.fielddescrof(VTYPE, name)` produces (descr.py:218-239
    /// `get_field_descr`) — pyre's `init` keys `VirtualizableFieldState.fields`
    /// by `fielddescr.get_index()` (info.py:203-206), so the synthetic
    /// fallback at `init` assigns `1 + field_idx_in_vinfo` for static slots
    /// and `1 + num_static + array_idx` for array slots.
    fn test_vable_field_descr(offset: usize, field_type: Type, index_in_parent: usize) -> DescrRef {
        let field_size = match field_type {
            Type::Int | Type::Ref | Type::Float => 8,
            Type::Void => 0,
        };
        let flag = majit_ir::ArrayFlag::from_field_type(field_type);
        let mut fd = majit_ir::SimpleFieldDescr::new(0, offset, field_size, field_type, false)
            .with_flag(flag);
        fd.index_in_parent = index_in_parent;
        Arc::new(fd) as DescrRef
    }

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            // Type-tag op.pos so `opref_type` priority 0
            // (`opref.ty()`) resolves via the variant tag without
            // falling through to the inputarg-slot fallback (which
            // collides with low op-position raws).
            op.pos
                .set(OpRef::op_typed(i as u32, op.opcode.result_type()));
        }
    }

    use super::super::seed_guard_snapshots_with;

    fn seed_virtualize_guard_snapshots(ops: &[Op]) -> (Vec<Op>, crate::optimizeopt::SnapshotBoxes) {
        // These direct optimizer tests do not build MIFrame objects.  Their
        // guard bracket list is the explicit active-box snapshot input that
        // RPython would get from capture_resumedata(); store_final_boxes then
        // overwrites guard.fail_args with the numbered liveboxes.
        seed_guard_snapshots_with(ops, |guard| {
            guard
                .getfailargs()
                .map(|fail_args| fail_args.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default()
        })
    }

    /// Canonicalize an op's args the way the production driver does in
    /// `Optimizer::propagate_forward` (optimizer.py setarg loop):
    /// resolve each arg through the box environment so the op carries the
    /// canonical operand that the handlers read via
    /// `op.arg(i).get_box_replacement(false)`. Tests that drive a pass's
    /// `propagate_forward` directly bypass that loop, so they must
    /// canonicalize explicitly before invoking the handler.
    fn resolve_op_args(op: &mut Op, ctx: &mut OptContext) {
        for i in 0..op.num_args() {
            // Mirror the production driver's `None` arm
            // (`optimizer.rs`'s `propagate_from_pass_range`):
            // an unbound operand whose root is not a sentinel is minted and
            // registered via `materialize_operand_at`, then walked to its
            // terminal — cloning the orig arg would skip canonicalization.
            let canonical = match ctx.resolve_operand_operand_opt(&op.arg(i)) {
                Some(b) => b,
                None => {
                    let argref = op.arg(i).to_opref();
                    if argref.is_none() {
                        op.arg(i).clone()
                    } else {
                        ctx.materialize_operand_at(argref)
                            .get_box_replacement(false)
                    }
                }
            };
            op.setarg(i, canonical);
        }
    }

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        run_pass_typed(ops, &[])
    }

    /// Like `run_pass`, but declares specific OpRef slots as Int-typed.
    /// Use for tests whose anonymous high-numbered Boxes feed int-typed
    /// setfield values — otherwise the MUST_ALIAS replay through
    /// `make_equal_to` would cross-type-forward an Int-typed `getfield_gc_i`
    /// result into the Ref-seeded value slot and trip the Box.type
    /// invariant guard on `make_equal_to`.
    fn run_pass_typed(ops: &[Op], int_slots: &[u32]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVirtualize::new()));
        // See `run_heap_opt` in heap.rs for the rationale behind the
        // 1024 Ref seed: tests use anonymous high-numbered OpRefs as
        // stand-in Box arguments, and the preamble exporter needs an
        // intrinsic type per renamed inputarg.
        let mut types = vec![Type::Ref; 1024];
        for &idx in int_slots {
            types[idx as usize] = Type::Int;
        }
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&types);
        let (ops, snapshots) = seed_virtualize_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::new(), 1024)
    }

    fn run_default_pipeline(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::default_pipeline();
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&vec![Type::Ref; 1024]);
        let (ops, snapshots) = seed_virtualize_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::new(), 1024)
    }

    fn run_default_pipeline_typed(ops: &[Op], int_slots: &[u32], float_slots: &[u32]) -> Vec<Op> {
        let mut opt = Optimizer::default_pipeline();
        let mut types = vec![Type::Ref; 1024];
        for &idx in int_slots {
            types[idx as usize] = Type::Int;
        }
        for &idx in float_slots {
            types[idx as usize] = Type::Float;
        }
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&types);
        let (ops, snapshots) = seed_virtualize_guard_snapshots(ops);
        opt.snapshot_boxes = snapshots;
        opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::ConstMap::new(), 1024)
    }

    fn run_pass_with_constants(ops: &[Op], constants: &[(OpRef, Value)]) -> Vec<Op> {
        let (ops, snapshots) = seed_virtualize_guard_snapshots(ops);
        let mut ctx = OptContext::new(ops.len());
        ctx.snapshot_boxes = snapshots;
        for &(opref, ref val) in constants {
            let b = ctx.materialize_operand_at(opref);
            ctx.make_constant_box(&b, *val);
        }

        let mut pass = OptVirtualize::new();
        pass.setup();

        for op in &ops {
            // Resolve forwarded arguments
            let mut resolved_op = op.clone();
            // optimizer.py:651-652 setarg loop parity. `resolve_op_args`
            // binds each arg to its canonical box (oparser object-identity),
            // materialising and registering a bound box for any unbound
            // position so no position-only `Operand::Box` is minted.
            resolve_op_args(&mut resolved_op, &mut ctx);

            let resolved_rc = std::rc::Rc::new(resolved_op.clone());
            ctx.bind_input_resops(std::slice::from_ref(&resolved_rc));
            match pass.propagate_forward(&resolved_op, &resolved_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        ctx.new_operations
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect()
    }

    #[test]
    fn test_standard_virtualizable_force_is_noop_in_optimizer() {
        // Verify that Optimizer::force_box skips Virtualizable PtrInfo
        // without destroying the tracked field state.
        // opencoder.py:259 inputarg_from_tp — vable is the sole Ref inputarg.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref]);
        let vable_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.set_ptr_info(
            &vable_box,
            PtrInfo::Virtualizable(VirtualizableFieldState {
                fields: vec![],
                field_descrs: vec![],
                arrays: vec![(0, vec![Operand::None])],
                heap_fields: vec![],
                last_guard_pos: -1,
            }),
        );

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVirtualize::with_virtualizable(
            VirtualizableConfig {
                static_field_offsets: vec![],
                static_field_types: vec![],
                static_field_descrs: vec![],
                array_field_offsets: vec![8],
                array_item_types: vec![Type::Ref],
                array_field_descrs: vec![],
                vable_input_offset: 0,
                identity_input_index: Some(0),
            },
        )));
        let forced = opt.force_box(OpRef::input_arg_ref(0), &mut ctx);
        assert_eq!(forced, OpRef::input_arg_ref(0));
        assert!(
            ctx.new_operations.is_empty(),
            "standard virtualizable should not be forced to raw heap ops by optimizer"
        );
        let v_box = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_ref(0))
            .expect("standard virtualizable operand populated");
        assert!(
            ctx.is_virtualizable(&v_box),
            "Virtualizable PtrInfo must survive force_box"
        );
    }

    #[test]
    fn test_standard_virtualizable_raw_first_read_is_not_cached() {
        // opencoder.py:259 inputarg_from_tp — vable is the sole Ref inputarg.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![8],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        // array slot at byte offset 8; array_idx_for_offset reads the
        // FieldDescr's `offset()` directly so the index_in_parent value is
        // immaterial — pass `1` (= `1 + num_static + array_idx` with
        // num_static=0) for consistency with `init`.
        let field_descr = test_vable_field_descr(8, Type::Int, 1);
        let arr_descr = array_descr(20);
        let b = ctx.materialize_operand_at(OpRef::int_op(50));
        ctx.make_constant_box(&b, Value::Int(0));

        let get_array_ptr = Op::with_descr(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            field_descr,
        );
        let get_item = Op::with_descr(
            OpCode::GetarrayitemRawI,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
            ],
            arr_descr.clone(),
        );
        let get_item_again = Op::with_descr(
            OpCode::GetarrayitemRawI,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
            ],
            arr_descr,
        );

        let mut ops = vec![get_array_ptr, get_item, get_item_again];
        assign_positions(&mut ops);
        // Route raw array reads through the GetfieldRawI result so
        // resolve_array_source() sees the producing OpRef, not the bare vable
        // inputarg.
        // Bind the array-element reads to the GetfieldRawI producer's bound
        // result box (oparser object-identity); GetfieldRawI (ops[0]) is
        // Int-typed so its result position is `OpRef::int_op(0)`.
        let array_ptr_box =
            crate::history::test_support::rooted_resop_operand(Type::Int, ops[0].pos.get().raw());
        ops[1].setarg(0, array_ptr_box.clone());
        ops[2].setarg(0, array_ptr_box);

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity. `resolve_op_args`
            // binds each arg to its canonical box (oparser object-identity),
            // materialising and registering a bound box for any unbound
            // position so no position-only `Operand::Box` is minted.
            resolve_op_args(&mut resolved, &mut ctx);
            match pass.propagate_forward(&resolved, &std::rc::Rc::new(resolved.clone()), &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|op| op.opcode == OpCode::GetarrayitemRawI)
            .count();
        assert_eq!(
            get_count, 2,
            "standard virtualizable path should not absorb raw array reads into optimizer-owned state"
        );
    }

    #[test]
    fn test_standard_virtualizable_call_does_not_force_frame_to_raw_storeback() {
        // opencoder.py:259 inputarg_from_tp — vable Ref + an opaque Int call
        // arg at slot 1 (slot 100 stays an outside-of-inputargs free opref).
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8, 16],
            static_field_types: vec![Type::Int, Type::Int],
            static_field_descrs: vec![],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let mut call = Op::new(
            OpCode::CallMayForceI,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 1),
            ],
        );
        call.setdescr(majit_ir::descr::make_call_descr(
            vec![Type::Int, Type::Int, Type::Int],
            Type::Int,
            majit_ir::EffectInfo::default(),
        ));

        // RPython parity: virtualize.py's default for calls is emit(op)
        // which forwards to the next pass without forcing. Forcing happens
        // in _emit_operation (Optimizer level). OptVirtualize returns PassOn.
        let result = pass.propagate_forward(&call, &std::rc::Rc::new(call.clone()), &mut ctx);
        assert!(
            matches!(result, OptimizationResult::PassOn),
            "call should PassOn (forcing happens at Optimizer::emit_operation level)"
        );
        assert!(
            ctx.new_operations
                .iter()
                .all(|op| op.opcode != OpCode::SetfieldRaw),
            "standard virtualizable call should not force frame writeback"
        );
    }

    #[test]
    fn test_standard_virtualizable_raw_getfield_is_not_absorbed_by_optimizer() {
        // opencoder.py:259 inputarg_from_tp — vable is the sole Ref inputarg
        // here; slot 10 (the GetfieldRawI result) lives above the inputarg
        // range and is not seeded.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            static_field_descrs: vec![],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let mut get = Op::new(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
        );
        get.setdescr(test_vable_field_descr(8, Type::Int, 1));
        get.pos.set(OpRef::int_op(10));

        let result = pass.propagate_forward(&get, &std::rc::Rc::new(get.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_raw_setfield_is_not_absorbed_by_optimizer() {
        // opencoder.py:259 inputarg_from_tp — vable Ref + Int value inputarg.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            static_field_descrs: vec![],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let mut set = Op::new(
            OpCode::SetfieldRaw,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 1),
            ],
        );
        set.setdescr(test_vable_field_descr(8, Type::Int, 1));

        let result = pass.propagate_forward(&set, &std::rc::Rc::new(set.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    /// A config that declares an array field carries no lengths at all now:
    /// the element seeding that needed them is gone, so the shape that used
    /// to trip the length assertion is just an ordinary config. It still has
    /// to install the identity `PtrInfo::Virtualizable`, which is the only
    /// thing `ensure_setup` still owes a virtualizable with arrays.
    #[test]
    fn an_array_declaring_config_needs_no_lengths_and_still_installs_the_identity() {
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![48, 56],
            array_item_types: vec![Type::Ref, Type::Ref],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();
        if let Some(ref mut vt) = pass.vable {
            vt.ensure_setup(&mut ctx);
        }
        let identity = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_ref(0))
            .expect("the identity inputarg must materialize");
        assert!(
            ctx.is_virtualizable(&identity),
            "ensure_setup must still mark the identity virtualizable so the base is not forced",
        );
    }

    #[test]
    fn test_standard_virtualizable_init_uses_parent_backed_field_descrs() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        let parent = majit_ir::make_size_descr_full(900, 16, 1);
        info.set_parent_descr(parent);
        let config = info.to_optimizer_config();
        let real_descr = info.static_field_descr(0);

        // opencoder.py:259 inputarg_from_tp — vable Ref + the `pc` static Int
        // field's flat-input slot 1 (init_virtualizable consumes
        // `config.static_field_offsets.len()` slots after the vable).
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(config);
        pass.setup();
        if let Some(ref mut vt) = pass.vable {
            vt.ensure_setup(&mut ctx);
        }

        let vbox = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_ref(0))
            .expect("standard virtualizable operand populated");
        let Some(PtrInfo::Virtualizable(vstate)) = ctx.peek_ptr_info(&vbox) else {
            panic!("expected standard virtualizable ptr info on OpRef::input_arg_ref(0)");
        };
        // `info.AbstractStructPtrInfo._fields` is keyed by
        // `fielddescr.get_index()`; `virtualizable.py:71-72
        // build_field_descr` assigns `index_in_parent = 1 + i` to the
        // i-th static field, so the `pc` slot lands at index 1.
        let key = real_descr
            .as_field_descr()
            .expect("virtualizable static_field_descr is a FieldDescr")
            .index_in_parent() as u32;
        let seeded = get_field_descr(&vstate.field_descrs, key)
            .expect("virtualizable init should seed field descr");
        assert_eq!(
            majit_ir::descr::descr_identity(&seeded),
            majit_ir::descr::descr_identity(&real_descr)
        );
        assert!(
            seeded
                .as_field_descr()
                .and_then(|fd| fd.get_parent_descr())
                .is_some(),
            "standard virtualizable config must carry real fielddescr.parent_descr",
        );
    }

    #[test]
    fn test_standard_virtualizable_raw_getarrayitem_is_not_absorbed_by_optimizer() {
        // opencoder.py:259 inputarg_from_tp — vable Ref + Int array index.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let mut get_field = Op::new(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
        );
        get_field.setdescr(test_vable_field_descr(24, Type::Int, 1));
        get_field.pos.set(OpRef::int_op(10));
        resolve_op_args(&mut get_field, &mut ctx);
        assert!(matches!(
            pass.propagate_forward(&get_field, &std::rc::Rc::new(get_field.clone()), &mut ctx),
            OptimizationResult::PassOn
        ));
        ctx.emit(get_field);

        let mut get_item = Op::new(
            OpCode::GetarrayitemRawI,
            &[
                crate::history::test_support::rooted_resop_operand(Type::Int, 10),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 1),
            ],
        );
        get_item.setdescr(array_descr(24));
        let result =
            pass.propagate_forward(&get_item, &std::rc::Rc::new(get_item.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_raw_setarrayitem_is_not_absorbed_by_optimizer() {
        // opencoder.py:259 inputarg_from_tp — vable Ref + Int array index.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref, Type::Int]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let mut get_field = Op::new(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
        );
        get_field.setdescr(test_vable_field_descr(24, Type::Int, 1));
        get_field.pos.set(OpRef::int_op(10));
        resolve_op_args(&mut get_field, &mut ctx);
        assert!(matches!(
            pass.propagate_forward(&get_field, &std::rc::Rc::new(get_field.clone()), &mut ctx),
            OptimizationResult::PassOn
        ));
        ctx.emit(get_field);

        let mut set_item = Op::new(
            OpCode::SetarrayitemRaw,
            &[
                crate::history::test_support::rooted_resop_operand(Type::Int, 10),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 1),
                crate::history::test_support::rooted_resop_operand(Type::Int, 2),
            ],
        );
        set_item.setdescr(array_descr(24));
        let result =
            pass.propagate_forward(&set_item, &std::rc::Rc::new(set_item.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_gc_getarrayitem_folds_after_setarrayitem() {
        // GETARRAYITEM_GC on the standard virtualizable array field folds
        // to the value written by a prior SETARRAYITEM_GC at the same
        // const index (read-after-write), symmetric with the static-field
        // fold.  The setarrayitem stays emitted (the heap write is kept);
        // only the redundant read is removed.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![8],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let field_descr = test_vable_field_descr(8, Type::Int, 1);
        let arr_descr = array_descr(20);
        // const array index 0 and a stored value.
        let b = ctx.materialize_operand_at(OpRef::int_op(50));
        ctx.make_constant_box(&b, Value::Int(0));
        let b = ctx.materialize_operand_at(OpRef::int_op(51));
        ctx.make_constant_box(&b, Value::Int(42));

        let get_array_ptr = Op::with_descr(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            field_descr,
        );
        let set_item = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
                crate::history::test_support::rooted_resop_operand(Type::Int, 51),
            ],
            arr_descr.clone(),
        );
        let get_item = Op::with_descr(
            OpCode::GetarrayitemGcI,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
            ],
            arr_descr,
        );

        let mut ops = vec![get_array_ptr, set_item, get_item];
        assign_positions(&mut ops);
        // Route the array element ops through the GetfieldRawI result so
        // resolve_array_source() sees the producing OpRef, not the bare
        // vable inputarg. GetfieldRawI (ops[0]) is Int-typed so its result
        // position is `OpRef::int_op(0)`.
        let array_ptr_box =
            crate::history::test_support::rooted_resop_operand(Type::Int, ops[0].pos.get().raw());
        ops[1].setarg(0, array_ptr_box.clone());
        ops[2].setarg(0, array_ptr_box);

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity. `resolve_op_args`
            // binds each arg to its canonical box (oparser object-identity),
            // materialising and registering a bound box for any unbound
            // position so no position-only `Operand::Box` is minted.
            resolve_op_args(&mut resolved, &mut ctx);
            match pass.propagate_forward(&resolved, &std::rc::Rc::new(resolved.clone()), &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => panic!("unexpected InvalidLoop in test"),
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|op| op.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 0,
            "GETARRAYITEM_GC on standard vable array should fold after a same-index SETARRAYITEM_GC"
        );
        let set_count = ctx
            .new_operations
            .iter()
            .filter(|op| op.opcode == OpCode::SetarrayitemGc)
            .count();
        assert_eq!(
            set_count, 1,
            "the heap write must be preserved (read-only fold)"
        );
    }

    #[test]
    fn test_standard_virtualizable_variable_index_setarrayitem_invalidates_array_fold() {
        // A SETARRAYITEM_GC with a NON-constant index may overwrite any slot,
        // so a subsequent const-index GETARRAYITEM_GC must NOT fold to a value
        // a prior const-index write tracked.  The variable-index write
        // invalidates the tracked array (force_lazy_setarrayitem,
        // can_cache=False).  Companion of the read-after-write fold test above.
        let mut ctx = OptContext::with_inputarg_types(8, &[Type::Ref]);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            static_field_descrs: vec![],
            array_field_offsets: vec![8],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        pass.setup();

        let field_descr = test_vable_field_descr(8, Type::Int, 1);
        let arr_descr = array_descr(20);
        // const index 0 + two stored values; int_op(60) is a NON-constant
        // index (never made constant) for the variable-index write.
        let b = ctx.materialize_operand_at(OpRef::int_op(50));
        ctx.make_constant_box(&b, Value::Int(0));
        let b = ctx.materialize_operand_at(OpRef::int_op(51));
        ctx.make_constant_box(&b, Value::Int(42));
        let b = ctx.materialize_operand_at(OpRef::int_op(52));
        ctx.make_constant_box(&b, Value::Int(99));

        let get_array_ptr = Op::with_descr(
            OpCode::GetfieldRawI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            field_descr,
        );
        // stack[0] = 42 (const index → tracked)
        let set_item_const = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
                crate::history::test_support::rooted_resop_operand(Type::Int, 51),
            ],
            arr_descr.clone(),
        );
        // stack[i] = 99 (variable index → must invalidate the tracked array)
        let set_item_var = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 60),
                crate::history::test_support::rooted_resop_operand(Type::Int, 52),
            ],
            arr_descr.clone(),
        );
        // stack[0] (const index read — must NOT fold after the variable write)
        let get_item = Op::with_descr(
            OpCode::GetarrayitemGcI,
            &[
                crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 50),
            ],
            arr_descr,
        );

        let mut ops = vec![get_array_ptr, set_item_const, set_item_var, get_item];
        assign_positions(&mut ops);
        // GetfieldRawI (ops[0]) is Int-typed so its result position is
        // `OpRef::int_op(0)`; bind the element ops to its result box.
        let array_ptr_box =
            crate::history::test_support::rooted_resop_operand(Type::Int, ops[0].pos.get().raw());
        ops[1].setarg(0, array_ptr_box.clone());
        ops[2].setarg(0, array_ptr_box.clone());
        ops[3].setarg(0, array_ptr_box);

        for op in &ops {
            let mut resolved = op.clone();
            // optimizer.py:651-652 setarg loop parity. `resolve_op_args`
            // binds each arg to its canonical box (oparser object-identity),
            // materialising and registering a bound box for any unbound
            // position so no position-only `Operand::Box` is minted.
            resolve_op_args(&mut resolved, &mut ctx);
            match pass.propagate_forward(&resolved, &std::rc::Rc::new(resolved.clone()), &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop(_) => panic!("unexpected InvalidLoop in test"),
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|op| op.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETARRAYITEM_GC must NOT fold after a variable-index SETARRAYITEM_GC \
             invalidated the tracked array"
        );
        let set_count = ctx
            .new_operations
            .iter()
            .filter(|op| op.opcode == OpCode::SetarrayitemGc)
            .count();
        assert_eq!(set_count, 2, "both heap writes must be preserved");
    }

    #[test]
    fn test_standard_virtualizable_loop_keeps_original_input_arity() {
        let mut opt = Optimizer::default_pipeline_with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            static_field_descrs: vec![],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_field_descrs: vec![],
            vable_input_offset: 0,
            identity_input_index: Some(0),
        });
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        let mut ops = vec![
            Op::new(
                OpCode::Label,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 1),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 2),
                ],
            ),
            Op::new(
                OpCode::GuardTrue,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    1,
                )],
            ),
            Op::new(
                OpCode::Jump,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 1),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 2),
                ],
            ),
        ];
        ops[1].setfailargs(Default::default());
        assign_positions(&mut ops);

        let (ops, snapshots) = seed_virtualize_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        // The optimizer above is configured WITH a virtualizable, so every
        // guard snapshot must carry a vable section — `pyjitpl.py:3326-3330`
        // makes `virtualizable_boxes` non-empty for the whole life of such a
        // trace, and `resume.py:236-239` (armed via
        // `minimum_virtualizable_size`) asserts it. Identity first
        // (`opencoder.py:718-726`), then this config's one static field and
        // its one array item.
        opt.snapshot_vable_boxes = vec![Some(vec![
            crate::resume::SnapshotBox::typed(OpRef::input_arg_typed(0, Type::Ref), Type::Ref),
            crate::resume::SnapshotBox::typed(OpRef::input_arg_typed(1, Type::Int), Type::Int),
            crate::resume::SnapshotBox::typed(OpRef::input_arg_typed(2, Type::Int), Type::Int),
        ])];
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);
        let jump = result
            .iter()
            .find(|op| op.opcode == OpCode::Jump)
            .expect("optimized loop should keep a jump");

        assert_eq!(opt.final_num_inputs(), 3);
        assert_eq!(jump.num_args(), 3);
    }

    // ── Tests ──

    #[test]
    fn test_new_with_vtable_removed() {
        // NEW_WITH_VTABLE should be removed (not emitted) — it becomes virtual
        let mut ops = vec![Op::with_descr(OpCode::NewWithVtable, &[], size_descr(1))];
        assign_positions(&mut ops);
        let result = run_pass(&ops);
        assert!(result.is_empty(), "NEW_WITH_VTABLE should be removed");
    }

    #[test]
    fn test_typeptr_read_with_zero_vtable_is_not_folded() {
        let sd: DescrRef = majit_ir::make_size_descr_with_vtable(1, 8, 0, 0);
        let typeptr_descr: DescrRef = Arc::new(majit_ir::SimpleFieldDescr::new_with_name(
            0,
            0,
            8,
            Type::Int,
            false,
            majit_ir::ArrayFlag::Signed,
            "object.typeptr".to_string(),
            "typeptr".to_string(),
        ));
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                typeptr_descr,
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::GetfieldGcI),
            "a zero-vtable typeptr read must remain instead of folding to constant 0: {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ob_type_read_with_zero_vtable_is_not_folded() {
        let sd: DescrRef = majit_ir::make_size_descr_with_vtable(1, 8, 0, 0);
        let typeptr_descr: DescrRef = Arc::new(majit_ir::SimpleFieldDescr::new_with_name(
            0,
            0,
            8,
            Type::Ref,
            false,
            majit_ir::ArrayFlag::Pointer,
            "PyObject.ob_type".to_string(),
            "ob_type".to_string(),
        ));
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd),
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                typeptr_descr,
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        assert!(
            result.iter().any(|op| op.opcode == OpCode::GetfieldGcR),
            "a zero-vtable ob_type read must remain instead of folding to null: {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_new_removed() {
        let mut ops = vec![Op::with_descr(OpCode::New, &[], size_descr(1))];
        assign_positions(&mut ops);
        let result = run_pass(&ops);
        assert!(result.is_empty(), "NEW should be removed");
    }

    #[test]
    fn test_fresh_virtual_unwritten_fields_are_typed_zero() {
        // virtualize.py optimize_GETFIELD_GC_I: GC allocations are
        // zero-filled, so an unset virtual field folds through
        // optimizer.new_const(fielddescr) without forcing the allocation.
        for (get_opcode, field_descr) in [
            (OpCode::GetfieldGcI, field_descr(0)),
            (OpCode::GetfieldGcR, ref_field_descr(0)),
            (OpCode::GetfieldGcF, float_field_descr(0)),
        ] {
            let mut ops = vec![
                Op::with_descr(OpCode::NewWithVtable, &[], size_descr(1)),
                Op::with_descr(
                    get_opcode,
                    &[crate::history::test_support::rooted_resop_operand(
                        Type::Ref,
                        0,
                    )],
                    field_descr,
                ),
            ];
            assign_positions(&mut ops);
            let result = run_pass(&ops);
            assert!(
                result.is_empty(),
                "{get_opcode:?} of an unset fresh field forced the virtual: {:?}",
                result.iter().map(|op| op.opcode).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_setfield_getfield_on_virtual() {
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // i1 = getfield_gc_i(p0, descr=field1)
        //
        // After optimization: all removed, i1 forwards to i10.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                fd.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass_typed(&ops, &[100]);
        assert!(
            result.is_empty(),
            "all ops should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    /// ```text
    /// p0 = new_with_vtable(descr=size1)
    /// setfield_gc(p0, i100, descr=field10)          # slot 10 <- i100
    /// i1 = getfield_gc_i(p0, descr=misindexed)      # claims a slot it does not hold
    /// i2 = int_mul(i1, i200)                        # survives, so i1 is observable
    /// ```
    ///
    /// The setfield fixes the virtual's descr to `field_descr(10)`'s parent
    /// (`optimize_setfield_gc` -> `init_fields`), whose slot list is `0..=10`
    /// with `offset == idx * 8`. So `real_offset` is what decides LISTED vs
    /// UNLISTED, and it is a parameter rather than a constant precisely so a
    /// reader can check that for themselves: `read_slot * 8` makes the claimed
    /// slot genuinely hold the field, anything else makes it a false claim.
    /// Both legs below pass 24 — slot 3's address — so any `read_slot` other
    /// than 3 is unlisted.
    ///
    /// `IntMul` is not in `OptVirtualize`'s dispatch table, so it survives the
    /// pass with its argument resolved — that argument is the only place the
    /// read's answer is observable, because the read itself is `Remove`d under
    /// BOTH behaviours and an op-count assertion cannot tell them apart.
    fn slot_read_trace(read_slot: u32, real_offset: usize) -> Vec<Op> {
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], size_descr(1)),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                field_descr(10),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                misindexed_field_descr(read_slot, real_offset),
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 2),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ),
        ];
        assign_positions(&mut ops);
        ops
    }

    /// The answer the surviving `IntMul` received for the folded read, or
    /// `None` when the read forwarded a non-constant operand.
    fn folded_read_answer(result: &[Op]) -> Option<Value> {
        assert_eq!(
            result.len(),
            1,
            "expected only the IntMul to survive; got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        assert_eq!(result[0].opcode, OpCode::IntMul);
        result[0].arg(0).const_value()
    }

    /// THE POSITIVE LEG. A read whose claimed slot IS populated but does
    /// NOT hold the field being read must answer the zeroed allocation, not
    /// the value the other field stored there.
    ///
    /// The correct answer is `Value::Int(0)`: `field_slot_identifies` fails,
    /// `slot_resolvable` is false, `field_val` is `None`, and the read falls
    /// through to `virtualize.py:188-189`'s zero-fold — the guard's own log
    /// line says "folding to the zeroed allocation".
    ///
    /// Before `field_slot_identifies` there was no read-side slot check, so
    /// `get_field` found slot 10 populated and forwarded ANOTHER FIELD'S VALUE.
    /// The fold is not the harm; the unguarded forward was.
    #[test]
    fn test_unlisted_slot_read_of_a_populated_slot_answers_the_zeroed_allocation() {
        let ops = slot_read_trace(10, 24);
        let result = run_pass_typed(&ops, &[100, 200]);
        assert_eq!(
            folded_read_answer(&result),
            Some(Value::Int(0)),
            "a read whose slot does not hold it must answer the zeroed \
             allocation; forwarding the populated slot hands back a different \
             field's value"
        );
    }

    /// THE NEGATIVE CONTROL, and it must pass with or without the read-side
    /// check. Same trace, same descrs, same populated slot 10 — only the
    /// claimed slot moves to 5, which nothing ever stored.
    ///
    /// Both behaviours reach the zero-fold here: with a slot check because the
    /// slot does not hold the field, without one because `get_field` finds
    /// slot 5 empty. A test that exercised only this case would pass on
    /// both branches and prove nothing about either.
    #[test]
    fn test_unlisted_slot_read_of_an_unpopulated_slot_is_branch_invariant() {
        let ops = slot_read_trace(5, 24);
        let result = run_pass_typed(&ops, &[100, 200]);
        assert_eq!(
            folded_read_answer(&result),
            Some(Value::Int(0)),
            "an unwritten slot folds to the zeroed allocation on either side \
             of the read-side slot check; this leg discriminates nothing and \
             exists to prove the positive leg is not measuring the fold"
        );
    }

    /// THE THIRD LEG — the one that makes the read path's `init_fields`
    /// (in `optimize_getfield_gc`) actually replace the descr, which neither
    /// leg above does.
    ///
    /// Both legs above allocate with `size_descr(1)` and then `setfield` a
    /// `field_descr(10)` whose parent IS a `SizeDescr`, so the SETFIELD's own
    /// `init_fields` takes the `cur_len == 0` arm (`ptr_info.rs`) and leaves
    /// `cur_len == 11`. Every slot they read is below that, so the read-side
    /// call is a no-op on both — ablating it (`if false &&`) left both green.
    ///
    /// This leg keeps `cur_len == 0` alive until the READ by denying the
    /// setfield a `SizeDescr` parent (`NarrowParentFieldDescr`), which is the
    /// only shape that lets a POPULATED slot coexist with `cur_len == 0`:
    ///
    /// ```text
    /// p0 = new_with_vtable(descr=non_size)      # as_size_descr() == None  => cur_len 0
    /// setfield_gc(p0, i100, descr=narrow(3))    # slot 3 <- i100; init_fields returns
    ///                                           #   early, disagreement short-circuits
    /// i1 = getfield_gc_i(p0, descr=misindexed(3, 56))   # claims slot 3, holds slot 7
    /// i2 = int_mul(i1, i200)                    # survives, so i1 is observable
    /// ```
    ///
    /// At the read, `cur_len == 0` fires `init_fields`' first arm and installs
    /// the field's own 4-slot parent. `field_slot_identifies` can then see that
    /// slot 3 sits at offset 24 while the descr claims offset 56, refuses the
    /// resolution, and the read folds to the zeroed allocation.
    ///
    /// WITHOUT the read-side `init_fields` the descr stays the non-size one,
    /// and `field_slot_identifies` FAILS OPEN — its
    /// `let Some(size_descr) = descr.as_size_descr() else { return true; }`
    /// returns `true` for a descr with no field list at all. `get_field` then
    /// finds slot 3 populated and forwards it, so `folded_read_answer` reads
    /// `None` (a non-constant operand) instead of `Some(Int(0))`. That is the
    /// two-sided ablation: delete the read-side `init_fields` block in
    /// `optimize_getfield_gc` and this assertion must go RED, where the two
    /// legs above stay green.
    #[test]
    fn test_read_path_init_fields_upgrades_a_zero_length_descr_before_the_slot_check() {
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], non_size_descr(0x0D)),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                narrow_parent_field_descr(3),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                // Claims slot 3 (offset 24) but answers slot 7's address.
                misindexed_field_descr(3, 56),
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 2),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ),
        ];
        assign_positions(&mut ops);
        let result = run_pass_typed(&ops, &[100, 200]);
        assert_eq!(
            folded_read_answer(&result),
            Some(Value::Int(0)),
            "the read-side init_fields must install the field's parent before \
             field_slot_identifies runs; without it the descr carries no field \
             list, the slot check fails open, and the populated slot 3 is \
             forwarded instead of folded"
        );
    }

    #[test]
    fn test_setfield_initializes_parent_backed_fielddescrs() {
        let group = majit_ir::descr::make_simple_descr_group(
            1,
            24,
            1,
            0,
            &[majit_ir::descr::SimpleFieldDescrSpec {
                is_class_word: Some(false),
                index: 10,
                field_key: "Node.value".to_string(),
                name: "Node.value".to_string(),
                offset: 16,
                field_size: 8,
                field_type: Type::Int,
                is_immutable: false,
                is_quasi_immutable: false,
                flag: majit_ir::ArrayFlag::Signed,
                virtualizable: false,
                index_in_parent: 0,
            }],
        );
        let sd = group.size_descr.clone() as DescrRef;
        let fd = group.field_descrs[0].clone() as DescrRef;

        let mut ctx = OptContext::new(2);
        let mut pass = OptVirtualize::new();
        pass.setup();

        let mut new_op = Op::with_descr(OpCode::NewWithVtable, &[], sd);
        new_op.pos.set(OpRef::ref_op(0));
        let new_op_rc = std::rc::Rc::new(new_op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&new_op_rc));
        assert!(matches!(
            pass.propagate_forward(&new_op, &new_op_rc, &mut ctx),
            OptimizationResult::Remove
        ));

        let mut set_op = Op::with_descr(
            OpCode::SetfieldGc,
            &[
                crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                crate::history::test_support::rooted_resop_operand(Type::Int, 100),
            ],
            fd,
        );
        set_op.pos.set(OpRef::int_op(1));
        resolve_op_args(&mut set_op, &mut ctx);
        assert!(matches!(
            pass.propagate_forward(&set_op, &std::rc::Rc::new(set_op.clone()), &mut ctx),
            OptimizationResult::Remove
        ));

        let inputarg_box = ctx
            .get_box_replacement_operand_opt(OpRef::ref_op(0))
            .expect("inputarg operand populated");
        let info = ctx
            .peek_ptr_info(&inputarg_box)
            .expect("virtual info missing");
        let PtrInfo::Virtual(vinfo) = info else {
            panic!("expected Virtual ptr info, got {info:?}");
        };
        assert_eq!(
            vinfo
                .fields
                .iter()
                .map(|(i, b)| (*i, b.to_opref()))
                .collect::<Vec<_>>(),
            vec![(0, OpRef::int_op(100))]
        );
        // info.py:188 keeps no cached fielddescr list — `descr.get_all_fielddescrs()`
        // is the authoritative view. Round-trip the size descr the same way
        // production consumers (info.rs all_fielddescrs_from_descr) do.
        let fielddescrs = vinfo
            .descr
            .as_size_descr()
            .expect("Virtual carries a SizeDescr")
            .all_fielddescrs();
        assert_eq!(fielddescrs.len(), 1);
        assert_eq!(fielddescrs[0].index_in_parent(), 0);
    }

    #[test]
    fn test_virtual_escaping_at_call() {
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // call_n(p0)   <- p0 escapes here, should force allocation
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // i10 is a loop inputarg (Int) in a real trace; the forced setfield
        // re-resolves it, so bind it as a typed inputarg producer.
        let result = run_pass_typed(&ops, &[100]);

        // Expect: new_with_vtable, setfield_gc, call_n
        assert!(
            result.len() >= 2,
            "expected forced allocation + call; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        // The first emitted op should be the forced NEW_WITH_VTABLE
        assert_eq!(result[0].opcode, OpCode::NewWithVtable);
        // There should be a SETFIELD_GC for the field
        let setfield_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .count();
        assert!(
            setfield_count >= 1,
            "expected at least one SETFIELD_GC for forced field"
        );
        // The last op should be the CALL_N
        assert_eq!(result.last().unwrap().opcode, OpCode::CallN);
    }

    #[test]
    fn test_new_array_virtual() {
        // i0 = <constant 3>
        // p1 = new_array(i0, descr=array1)
        // setarrayitem_gc(p1, i_idx0, i_val42, descr=array1)
        // i2 = getarrayitem_gc_i(p1, i_idx0, descr=array1)
        //
        // All removed, i2 forwards to i_val42.
        let ad = array_descr(20);

        // OpRef::int_op(50) = constant 3 (array size)
        // OpRef::int_op(51) = constant 0 (index)
        // OpRef::int_op(52) = value to store (arbitrary opref)

        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArray,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    50,
                )],
                ad.clone(),
            ), // pos=0
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 52),
                ],
                ad.clone(),
            ), // pos=1
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                ],
                ad.clone(),
            ), // pos=2
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::int_op(50), Value::Int(3)),
            (OpRef::int_op(51), Value::Int(0)),
        ];

        let result = run_pass_with_constants(&ops, &constants);
        assert!(
            result.is_empty(),
            "all array ops on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_new_array_clear_unwritten_item_is_typed_zero() {
        // virtualize.py:27-35 + info.py:507-514: NEW_ARRAY_CLEAR seeds every
        // virtual slot with optimizer.new_const_item(arraydescr), so reading
        // an unwritten integer item folds to zero instead of raising
        // InvalidLoop("reading uninitialized virtual array items").
        let ad: DescrRef = Arc::new(majit_ir::descr::SimpleArrayDescr::with_flag(
            21,
            0,
            8,
            21,
            Type::Int,
            majit_ir::ArrayFlag::Signed,
        ));
        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArrayClear,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    50,
                )],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::GetarrayitemGcI,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                ],
                ad,
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass_with_constants(
            &ops,
            &[
                (OpRef::int_op(50), Value::Int(3)),
                (OpRef::int_op(51), Value::Int(1)),
            ],
        );
        assert!(
            result.is_empty(),
            "clear-array read should fold to its typed zero; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    /// rpython/jit/metainterp/optimizeopt/test/test_util.py:351-360:
    /// `complexarray = GcArray(Struct('complex', ('real', Float),
    /// ('imag', Float)))` with `complexarraydescr = cpu.arraydescrof(...)`,
    /// `complexrealdescr = cpu.interiorfielddescrof(complexarray, "real")`,
    /// `compleximagdescr = cpu.interiorfielddescrof(complexarray, "imag")`.
    /// Returns `(complexarraydescr, complexrealdescr, compleximagdescr)`.
    fn complex_array_descrs() -> (DescrRef, DescrRef, DescrRef) {
        // base_size 0, item_size 16 (two 8-byte floats); FLAG_STRUCT marks
        // `is_array_of_structs()` (descr.py).
        let arr = Arc::new(majit_ir::descr::SimpleArrayDescr::with_flag(
            90,
            0,
            16,
            90,
            Type::Float,
            majit_ir::ArrayFlag::Struct,
        ));
        let real_fd: Arc<dyn majit_ir::descr::FieldDescr> = {
            let mut fd = majit_ir::SimpleFieldDescr::new(0, 0, 8, Type::Float, false);
            fd.index_in_parent = 0;
            Arc::new(fd)
        };
        let imag_fd: Arc<dyn majit_ir::descr::FieldDescr> = {
            let mut fd = majit_ir::SimpleFieldDescr::new(0, 8, 8, Type::Float, false);
            fd.index_in_parent = 1;
            Arc::new(fd)
        };
        let real: DescrRef = Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
            0,
            arr.clone(),
            real_fd,
        ));
        let imag: DescrRef = Arc::new(majit_ir::descr::SimpleInteriorFieldDescr::new(
            1,
            arr.clone(),
            imag_fd,
        ));
        // descr.py get_array_descr sets arraydescr.all_interiorfielddescrs.
        arr.set_all_interiorfielddescrs(vec![real.clone(), imag.clone()]);
        (arr as DescrRef, real, imag)
    }

    #[test]
    fn test_new_array_struct_virtual() {
        // virtualize.py:30-32 array-of-structs NEW_ARRAY_CLEAR virtualization,
        // mirroring the virtual roundtrip exercised by
        // rpython/jit/metainterp/optimizeopt/test/test_optimizebasic.py:2526
        // test_dirty_array_of_structs_field_after_force:
        //   p1 = new_array_clear(1, descr=complexarraydescr)
        //   setinteriorfield_gc(p1, 0, f_real, descr=complexrealdescr)
        //   setinteriorfield_gc(p1, 0, f_imag, descr=compleximagdescr)
        //   f2 = getinteriorfield_gc_f(p1, 0, descr=complexrealdescr)
        // The array stays virtual; `f2` forwards to `f_real`; all ops removed.
        let (arr, real, imag) = complex_array_descrs();

        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArrayClear,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    50,
                )],
                arr.clone(),
            ),
            Op::with_descr(
                OpCode::SetinteriorfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                    crate::history::test_support::rooted_resop_operand(Type::Float, 60),
                ],
                real.clone(),
            ),
            Op::with_descr(
                OpCode::SetinteriorfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                    crate::history::test_support::rooted_resop_operand(Type::Float, 61),
                ],
                imag.clone(),
            ),
            Op::with_descr(
                OpCode::GetinteriorfieldGcF,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                ],
                real.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::int_op(50), Value::Int(1)),
            (OpRef::int_op(51), Value::Int(0)),
        ];

        let result = run_pass_with_constants(&ops, &constants);
        assert!(
            result.is_empty(),
            "all interiorfield ops on virtual array-of-struct should be removed; \
             got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_new_array_struct_forced_at_call() {
        // info.py ArrayStructInfo._force_elements: when the virtual
        // array-of-structs escapes (here via call_n), it is reconstructed as
        // NEW_ARRAY_CLEAR + one SETINTERIORFIELD_GC per stored field, emitted
        // before the escaping op.
        //   p0 = new_array_clear(1, descr=complexarraydescr)
        //   setinteriorfield_gc(p0, 0, f_real, descr=complexrealdescr)
        //   setinteriorfield_gc(p0, 0, f_imag, descr=compleximagdescr)
        //   call_n(p0)   <- p0 escapes, force it
        let (arr, real, imag) = complex_array_descrs();

        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArrayClear,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    50,
                )],
                arr.clone(),
            ),
            Op::with_descr(
                OpCode::SetinteriorfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                    crate::history::test_support::rooted_inputarg_operand(Type::Float, 60),
                ],
                real.clone(),
            ),
            Op::with_descr(
                OpCode::SetinteriorfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 51),
                    crate::history::test_support::rooted_inputarg_operand(Type::Float, 61),
                ],
                imag.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // Forcing-on-escape is driven by the full `Optimizer`, not the
        // single-op `run_pass_with_constants` loop.  Mirror `run_pass_typed`
        // but seed the size/index constants (position-keyed, optimizer.rs
        // :2058-2064) and mark the float value slots 60/61 so they don't
        // collide with the Ref-typed inputarg seeding.
        let (ops, snapshots) = seed_virtualize_guard_snapshots(&ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVirtualize::new()));
        let mut types = vec![Type::Ref; 1024];
        types[60] = Type::Float;
        types[61] = Type::Float;
        opt.trace_inputargs = majit_ir::OpRef::inputarg_refs(&types);
        opt.snapshot_boxes = snapshots;
        let mut constants = majit_ir::ConstMap::new();
        constants.insert(50u32, Value::Int(1));
        constants.insert(51u32, Value::Int(0));
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // Forced reconstruction: NEW_ARRAY_CLEAR, 2× SETINTERIORFIELD_GC, CALL_N.
        assert_eq!(
            result.first().map(|o| o.opcode),
            Some(OpCode::NewArrayClear),
            "forced array-of-struct should re-emit NEW_ARRAY_CLEAR first; got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        let setinterior = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetinteriorfieldGc)
            .count();
        assert_eq!(
            setinterior,
            2,
            "both stored interior fields should be re-emitted; got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        assert_eq!(
            result.last().map(|o| o.opcode),
            Some(OpCode::CallN),
            "escaping call_n must come after the reconstruction; got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_arraylen_gc_on_virtual() {
        // Virtual array of length 5 -> arraylen_gc returns constant 5
        let ad = array_descr(20);

        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArray,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    50,
                )],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::ArraylenGc,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                ad.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::int_op(50), Value::Int(5))];

        let result = run_pass_with_constants(&ops, &constants);
        // Both NEW_ARRAY and ARRAYLEN_GC should be removed
        assert!(
            result.is_empty(),
            "arraylen on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_guard_class_on_virtual() {
        // p0 = new_with_vtable(descr=size_with_vtable(42))
        // guard_class(p0, ConstClass(42))   <- removed, class matches
        //
        // rpython/jit/metainterp/optimizeopt/virtualize.py does not
        // define `optimize_GUARD_CLASS`. rewrite.py
        // `optimize_GUARD_CLASS` calls `info.get_known_class(cpu)` on
        // the virtual's InstancePtrInfo and removes the guard when the
        // stored class matches. Run the full default pipeline so
        // OptRewrite sees the guard after OptVirtualize produced the
        // virtual.
        let sd: DescrRef = majit_ir::make_size_descr_with_vtable(1, 8, 0, 42);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::new(
                OpCode::GuardClass,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = seed_virtualize_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        constants.insert(200u32, majit_ir::Value::Int(42)); // expected class ptr matches vtable
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);
        // Both NEW_WITH_VTABLE (virtual) and GuardClass (redundant) removed
        assert!(
            result.is_empty(),
            "guard_class on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_guard_nonnull_on_virtual() {
        // p0 = new_with_vtable(descr=size1)
        // guard_nonnull(p0)   <- should be removed, virtual is always non-null
        let sd = size_descr(1);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::new(
                OpCode::GuardNonnull,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        // On this branch, inline guard numbering causes the guard emit to
        // trigger a lazy setfield flush, producing one extra op (NewWithVtable).
        assert_eq!(
            result.len(),
            2,
            "guard_nonnull on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_nested_virtuals() {
        // p0 = new_with_vtable(descr=size1)        -- outer
        // p1 = new_with_vtable(descr=size2)        -- inner
        // setfield_gc(p0, p1, descr=field_ref)     -- outer.field = inner
        // setfield_gc(p1, i_val, descr=field_int)  -- inner.field = i_val
        // call_n(p0)                                -- force outer, which forces inner
        let sd1 = size_descr(1);
        let sd2 = size_descr(2);
        let fd_ref = field_descr(10);
        let fd_int = field_descr(20);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd1.clone()), // pos=0
            Op::with_descr(OpCode::NewWithVtable, &[], sd2.clone()), // pos=1
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                ],
                fd_ref.clone(),
            ), // pos=2
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd_int.clone(),
            ), // pos=3
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ), // pos=4
        ];
        assign_positions(&mut ops);

        // i_val is a loop inputarg (Int); the forced inner setfield re-resolves
        // it, so bind it as a typed inputarg producer.
        let result = run_pass_typed(&ops, &[100]);

        // When p0 is forced, p1 (nested in p0's field) should also be forced.
        // Expect: new_with_vtable(inner), setfield_gc(inner), new_with_vtable(outer), setfield_gc(outer), call_n
        let new_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::NewWithVtable)
            .count();
        assert_eq!(
            new_count, 2,
            "both virtuals should be forced; got {new_count} NEW_WITH_VTABLE ops"
        );

        let setfield_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 2,
            "both fields should be set; got {setfield_count} SETFIELD_GC ops"
        );

        assert_eq!(
            result.last().unwrap().opcode,
            OpCode::CallN,
            "last op should be the CALL_N"
        );
    }

    #[test]
    fn test_virtual_struct_new() {
        // p0 = new(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // i1 = getfield_gc_i(p0, descr=field1)
        // -> all removed
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                fd.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass_typed(&ops, &[100]);
        assert!(
            result.is_empty(),
            "all struct ops should be removed; got {} ops",
            result.len()
        );
    }

    #[test]
    fn test_virtual_struct_forced_at_call() {
        // p0 = new(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // call_n(p0)
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // i10 is a loop inputarg (Int); the forced setfield re-resolves it, so
        // bind it as a typed inputarg producer.
        let result = run_pass_typed(&ops, &[100]);

        // Forced: NEW, SETFIELD_GC, CALL_N
        assert_eq!(result[0].opcode, OpCode::New);
        let has_setfield = result.iter().any(|o| o.opcode == OpCode::SetfieldGc);
        assert!(has_setfield, "should have SETFIELD_GC");
        assert_eq!(result.last().unwrap().opcode, OpCode::CallN);
    }

    #[test]
    fn test_default_pipeline_forced_virtual_keeps_field_store_before_call() {
        // info.py _force_elements clears the non-virtual field slot
        // before emitting SETFIELD_GC. Otherwise OptHeap can see the newly
        // forced PtrInfo as already containing the value and remove the
        // materialization store before an escaping call.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd,
            ),
            Op::new(
                OpCode::CallR,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 200),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
            ),
            Op::new(
                OpCode::Finish,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Int,
                    2,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // The field value, call argument and finish operand are loop inputargs
        // (Int) re-resolved by the escaping call; bind them as typed inputargs.
        let result = run_default_pipeline_typed(&ops, &[2, 100, 200], &[]);
        let setfield_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetfieldGc)
            .expect("forced virtual must emit SETFIELD_GC for its field");
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallR)
            .expect("escaping call must remain");
        assert!(
            setfield_pos < call_pos,
            "SETFIELD_GC must materialize the virtual field before the call; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_default_pipeline_lazy_setfield_flushed_before_residual_call_descr() {
        // heap.py `force_from_effectinfo`: a residual CALL
        // whose descr lacks per-call write analysis must still flush
        // any lazy_set on the cached fields it could touch. PyPy
        // `effectinfo.py effectinfo_from_writeanalyze` force-promotes
        // analyzer-absent EIs to `EF_RANDOM_EFFECTS` (`MOST_GENERAL`,
        // `effectinfo.py:271-273`). `emit_residual_call` /
        // `handle_side_effects` then see `call_has_random_effects` and
        // route through `clean_caches`,
        // so the per-cached-field flush runs and `setfield_gc` survives
        // in front of the call. The test threads `MOST_GENERAL` directly
        // rather than through `default_effect_info()`, which returns the
        // same constant, so the assertion stays pinned to the shape
        // under test.
        let sd = size_descr(2);
        let fd = field_descr(11);
        let call_descr = crate::call_descr::make_call_descr_with_effect(
            &[Type::Ref],
            Type::Ref,
            majit_ir::EffectInfo::MOST_GENERAL,
        );

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd,
            ),
            Op::with_descr(
                OpCode::CallR,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 200),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
                call_descr,
            ),
            Op::new(
                OpCode::Finish,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Int,
                    2,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // The field value, call argument and finish operand are loop inputargs
        // (Int) re-resolved by the residual call; bind them as typed inputargs.
        let result = run_default_pipeline_typed(&ops, &[2, 100, 200], &[]);
        let setfield_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetfieldGc)
            .expect("descrful CallR must not absorb the lazy SETFIELD_GC");
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallR)
            .expect("descrful CallR must survive optimization");
        assert!(
            setfield_pos < call_pos,
            "SETFIELD_GC must flush before a residual CALL whose descr has \
             no per-call write analysis; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_default_pipeline_escaping_call_arg_flushes_materialization_store() {
        let sd = size_descr(3);
        let fd = field_descr(12);
        let call_descr = crate::call_descr::make_call_descr_with_effect(
            &[Type::Ref],
            Type::Ref,
            majit_ir::EffectInfo::default(),
        );

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd,
            ),
            Op::with_descr(
                OpCode::CallR,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 200),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
                call_descr,
            ),
            Op::new(
                OpCode::Finish,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Int,
                    2,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // The field value, call argument and finish operand are loop inputargs
        // (Int) re-resolved by the escaping call; bind them as typed inputargs.
        let result = run_default_pipeline_typed(&ops, &[2, 100, 200], &[]);
        let setfield_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::SetfieldGc)
            .expect("escaping call argument must flush materialization store");
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallR)
            .expect("escaping call must remain");
        assert!(
            setfield_pos < call_pos,
            "SETFIELD_GC must initialize the escaping argument before CALL_R; got {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_default_pipeline_escaping_call_arg_flush_is_selective() {
        let fd = field_descr(12);
        let call_descr = crate::call_descr::make_call_descr_with_effect(
            &[Type::Ref],
            Type::Ref,
            majit_ir::EffectInfo::default(),
        );

        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 1),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 101),
                ],
                fd,
            ),
            Op::with_descr(
                OpCode::CallR,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 200),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
                call_descr,
            ),
            Op::new(
                OpCode::Finish,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Int,
                    2,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // The field values, call argument and finish operand are loop inputargs
        // (Int) re-resolved by the escaping call; bind them as typed inputargs.
        let result = run_default_pipeline_typed(&ops, &[2, 100, 101, 200], &[]);
        let call_pos = result
            .iter()
            .position(|op| op.opcode == OpCode::CallR)
            .expect("escaping call must remain");
        let arg0_setfield_pos = result
            .iter()
            .position(|op| {
                op.opcode == OpCode::SetfieldGc
                    && op.getarglist().first().map(|a| a.to_opref())
                        == Some(OpRef::input_arg_ref(0))
            })
            .expect("escaping argument store must be emitted");
        let arg1_setfield_pos = result
            .iter()
            .position(|op| {
                op.opcode == OpCode::SetfieldGc
                    && op.getarglist().first().map(|a| a.to_opref())
                        == Some(OpRef::input_arg_ref(1))
            })
            .expect("unrelated store must still be emitted by the final flush");

        assert!(
            arg0_setfield_pos < call_pos,
            "store for the escaping call argument must be before the call: {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
        assert!(
            arg1_setfield_pos > call_pos,
            "unrelated lazy store must remain pending until after the call: {:?}",
            result.iter().map(|op| op.opcode).collect::<Vec<_>>()
        );
    }

    // Note: forced struct field forwarding is handled by heap.rs caching,
    // not by virtualize.rs PtrInfo tracking. After force_box, the object
    // is materialized and heap.py caches field values independently.

    #[test]
    fn test_setfield_getfield_different_fields() {
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field_a)
        // setfield_gc(p0, i20, descr=field_b)
        // i1 = getfield_gc_i(p0, descr=field_a) -> i10
        // i2 = getfield_gc_i(p0, descr=field_b) -> i20
        let sd = size_descr(1);
        let fd_a = field_descr(10);
        let fd_b = field_descr(20);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                fd_a.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                fd_b.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                fd_a.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                fd_b.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass_typed(&ops, &[100, 200]);
        assert!(
            result.is_empty(),
            "all ops on virtual should be removed; got {} ops",
            result.len()
        );
    }

    #[test]
    fn test_setfield_overwrite() {
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // setfield_gc(p0, i20, descr=field1)   <- overwrites
        // call_n(p0)                            <- force
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 100),
                ],
                fd.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 200),
                ],
                fd.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // The setfield values (i10/i20) are loop inputargs in a real trace;
        // model them as Int-typed inputargs so the driver's inputarg seed
        // binds a canonical producer for each (no position-only re-resolution).
        let result = run_pass_typed(&ops, &[100, 200]);

        // Only one SETFIELD_GC should be emitted (the last value)
        let setfield_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 1,
            "overwritten field should produce only 1 SETFIELD_GC; got {setfield_count}"
        );
    }

    #[test]
    fn test_guard_class_twice() {
        // guard_class(p0, cls)   <- emitted (records known class)
        // guard_class(p0, cls)   <- removed (class already known)
        //
        // rewrite.py `postprocess_GUARD_CLASS` records the
        // class via `make_constant_class`, and the second
        // `optimize_GUARD_CLASS` (rewrite.py) sees the recorded
        // known class and removes itself. virtualize.py doesn't handle
        // GUARD_CLASS at all; run the full default pipeline.
        let mut ops = vec![
            Op::new(
                OpCode::GuardClass,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ),
            Op::new(
                OpCode::GuardClass,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = seed_virtualize_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        constants.insert(200u32, majit_ir::Value::Int(42)); // class ptr constant
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);
        assert_eq!(
            result.len(),
            1,
            "second guard_class should be removed; got {} ops",
            result.len()
        );
        assert_eq!(result[0].opcode, OpCode::GuardClass);
    }

    #[test]
    fn test_non_virtual_passthrough() {
        // Operations on non-virtual objects should pass through unchanged.
        // The struct base is the non-virtual Ref inputarg 0 (a real,
        // driver-bound box); a no-producer ResOp position would not resolve
        // to a bound box, which the canonical-arg handler requires.
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                fd.clone(),
            ),
            Op::with_descr(
                OpCode::GetfieldGcI,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
                fd.clone(),
            ),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        assert_eq!(result.len(), 2, "non-virtual ops should pass through");
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI);
    }

    // ── VirtualRef tests ──

    #[test]
    fn test_virtual_ref_non_escaping() {
        // vref = virtual_ref_r(obj, token)   <- becomes virtual struct
        // virtual_ref_finish(vref, CONST_NULL) <- absorbed into virtual, removed
        //
        // Expected output: only ForceToken (emitted by optimizer) + SameAsR for the null constant
        let mut ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=0
            Op::new(
                OpCode::VirtualRefFinish,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 102),
                ],
            ), // pos=1
        ];
        assign_positions(&mut ops);

        // OpRef::int_op(102) = CONST_NULL (Ref-typed null, matching producer `const_null()`).
        let constants = vec![(OpRef::int_op(102), Value::Ref(majit_ir::GcRef(0)))];
        let result = run_pass_with_constants(&ops, &constants);

        // VirtualRefR should be removed (virtual), VirtualRefFinish should be removed.
        // Only the ForceToken and null constant ops remain.
        let has_virtual_ref = result
            .iter()
            .any(|o| matches!(o.opcode, OpCode::VirtualRefR | OpCode::VirtualRefI));
        assert!(
            !has_virtual_ref,
            "VirtualRef should not appear in output; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        let has_finish = result.iter().any(|o| o.opcode == OpCode::VirtualRefFinish);
        assert!(
            !has_finish,
            "VirtualRefFinish should not appear in output; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_virtual_ref_escapes_at_call() {
        // vref = virtual_ref_r(obj, token)   <- becomes virtual struct
        // call_n(vref)                        <- vref escapes, force it
        //
        // Expected: NEW (forced struct) + SETFIELD_GC (fields) + CALL_N
        let mut ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=0
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ), // pos=1
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // The virtual ref should be forced (New or NewWithVtable emitted)
        let has_alloc = result
            .iter()
            .any(|o| matches!(o.opcode, OpCode::New | OpCode::NewWithVtable));
        assert!(
            has_alloc,
            "forced vref should emit allocation; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        assert_eq!(
            result.last().unwrap().opcode,
            OpCode::CallN,
            "last op should be CALL_N"
        );
    }

    #[test]
    fn test_virtual_ref_finish_with_forced_obj() {
        // vref = virtual_ref_r(obj, token)
        // virtual_ref_finish(vref, real_obj)   <- real_obj is non-null
        //
        // When the vref is still virtual and finish has a non-null obj,
        // the forced field is updated in the virtual struct.
        // No ops should be emitted for the VirtualRefFinish itself.
        let mut ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=0
            Op::new(
                OpCode::VirtualRefFinish,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ), // pos=1, non-null
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        let has_finish = result.iter().any(|o| o.opcode == OpCode::VirtualRefFinish);
        assert!(
            !has_finish,
            "VirtualRefFinish should be removed; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_virtual_ref_does_not_force_underlying_obj() {
        // p0 = new_with_vtable(descr=size1)   <- virtual
        // vref = virtual_ref_r(p0, token)     <- virtual (RPython: InstancePtrInfo)
        // call_n(vref)                         <- forces vref, NOT p0
        //
        // The key property: forcing the vref should NOT force the wrapped
        // object p0. The vref's `forced` field is set to CONST_NULL
        // by optimize_virtual_ref, so p0 is not referenced in the vref fields.
        // p0 only appears in the original VirtualRefR args, which are discarded.
        let sd = size_descr(1);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()), // pos=0
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=1
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    1,
                )],
            ), // pos=2
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // RPython parity: vref is a Virtual (InstancePtrInfo) forced as
        // NewWithVtable. The only NewWithVtable should be the vref itself;
        // p0 (the wrapped object) must NOT be forced.
        let new_vtable_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::NewWithVtable)
            .count();
        assert_eq!(
            new_vtable_count,
            1,
            "only the vref should be forced as NewWithVtable, not p0; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        // No New ops — the vref is no longer a VirtualStruct
        let new_count = result.iter().filter(|o| o.opcode == OpCode::New).count();
        assert_eq!(
            new_count,
            0,
            "no New should be emitted; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_virtual_ref_finish_on_escaped_vref() {
        // vref = virtual_ref_r(obj, token)
        // call_n(vref)                         <- forces vref
        // virtual_ref_finish(vref, real_obj)   <- vref is now non-virtual
        //
        // VirtualRefFinish on a non-virtual vref should emit SETFIELD_GC ops.
        let mut ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=0
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
            ), // pos=1
            Op::new(
                OpCode::VirtualRefFinish,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
            ), // pos=2
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // After the call, vref is forced. VirtualRefFinish should emit
        // SETFIELD_GC for `forced` and `virtual_token` fields.
        let setfield_after_call = result
            .iter()
            .skip_while(|o| o.opcode != OpCode::CallN)
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .count();
        assert!(
            setfield_after_call >= 2,
            "VirtualRefFinish on escaped vref should emit SETFIELD_GCs; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_virtual_ref_getfield_on_virtual_vref() {
        // vref = virtual_ref_r(obj, token)
        // p0 = getfield_gc_r(vref, descr=vref_forced_field)
        //
        // The vref is virtual, so getfield should return the virtual field value.
        let forced_descr = ref_field_descr(super::VREF_FORCED_FIELD_INDEX);

        let mut ops = vec![
            Op::new(
                OpCode::VirtualRefR,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
            ), // pos=0
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                forced_descr,
            ), // pos=1
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // The getfield should be removed (the forced field is a known constant 0)
        let has_getfield = result.iter().any(|o| o.opcode == OpCode::GetfieldGcR);
        assert!(
            !has_getfield,
            "getfield on virtual vref should be removed; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── VirtualRawBuffer optimization tests (RPython: test_rawmem.py parity) ──

    /// cpu.arraydescrof(rffi.CArray(lltype.Signed)) — 8-byte signed int array.
    fn raw_arraydescr() -> majit_ir::DescrRef {
        majit_ir::descr::make_array_descr(0, 8, majit_ir::Type::Int)
    }

    fn run_pass_with_raw_buffer(
        ops: &[Op],
        constants: &[(OpRef, Value)],
        raw_bufs: &[(OpRef, usize)],
    ) -> Vec<Op> {
        let mut ctx = OptContext::new(ops.len());
        for &(opref, ref val) in constants {
            let b = ctx.materialize_operand_at(opref);
            ctx.make_constant_box(&b, *val);
        }

        let mut pass = OptVirtualize::new();
        pass.setup();

        // Pre-populate VirtualRawBuffer info for specified OpRefs
        for &(opref, size) in raw_bufs {
            let b = ctx.materialize_operand_at(opref);
            ctx.set_ptr_info(
                &b,
                PtrInfo::VirtualRawBuffer(RawBufferPtrInfo::new(0, size, None)),
            );
        }

        for op in ops {
            let mut resolved_op = op.clone();
            // optimizer.py:651-652 setarg loop parity. `resolve_op_args`
            // binds each arg to its canonical box (oparser object-identity),
            // materialising and registering a bound box for any unbound
            // position so no position-only `Operand::Box` is minted.
            resolve_op_args(&mut resolved_op, &mut ctx);

            let resolved_rc = std::rc::Rc::new(resolved_op.clone());
            ctx.bind_input_resops(std::slice::from_ref(&resolved_rc));
            match pass.propagate_forward(&resolved_op, &resolved_rc, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) | OptimizationResult::Restart(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop(_) => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        ctx.new_operations
            .into_iter()
            .map(|rc| (*rc).clone())
            .collect()
    }

    #[test]
    fn test_raw_store_then_load_same_offset_forwarded() {
        // Mirrors RPython's test_raw_storage_int: store a value, then
        // load from the same offset on a virtual buffer.
        // raw_store(buf, offset=0, val, descr=arraydescr)
        // i1 = raw_load_i(buf, offset=0, descr=arraydescr)
        // -> i1 should be forwarded to val, both ops removed.
        let ad = raw_arraydescr();
        let mut ops = vec![
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawLoadI,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                ad,
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::int_op(100), Value::Int(0))]; // offset = 0
        let raw_bufs = vec![(OpRef::input_arg_ref(0), 32)];

        let result = run_pass_with_raw_buffer(&ops, &constants, &raw_bufs);
        assert!(
            result.is_empty(),
            "raw_store + raw_load at same offset on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_raw_ops_different_offsets_no_interference() {
        // Store two values at different offsets on a virtual raw buffer.
        // Load from each offset separately: each should get its own value.
        // raw_store(buf, offset=0, val_a, descr=arraydescr)
        // raw_store(buf, offset=8, val_b, descr=arraydescr)
        // i1 = raw_load_i(buf, offset=0, descr=arraydescr)  -> val_a
        // i2 = raw_load_i(buf, offset=8, descr=arraydescr)  -> val_b
        let ad = raw_arraydescr();
        let mut ops = vec![
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 201),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawLoadI,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawLoadI,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
                ad,
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![
            (OpRef::int_op(100), Value::Int(0)),
            (OpRef::int_op(101), Value::Int(8)),
        ];
        let raw_bufs = vec![(OpRef::input_arg_ref(0), 32)];

        let result = run_pass_with_raw_buffer(&ops, &constants, &raw_bufs);
        assert!(
            result.is_empty(),
            "all raw ops on virtual buffer should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_raw_store_overwrite_same_offset() {
        // Store twice at the same offset, then load.
        // raw_store(buf, 0, val_a, descr=arraydescr)
        // raw_store(buf, 0, val_b, descr=arraydescr)   <- overwrites
        // i1 = raw_load_i(buf, 0, descr=arraydescr)    -> val_b
        let ad = raw_arraydescr();
        let mut ops = vec![
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 201),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawLoadI,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                ad,
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::int_op(100), Value::Int(0))];
        let raw_bufs = vec![(OpRef::input_arg_ref(0), 32)];

        let result = run_pass_with_raw_buffer(&ops, &constants, &raw_bufs);
        // All removed: stores absorbed into virtual, load forwarded.
        assert!(
            result.is_empty(),
            "overwritten raw_store + load should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_raw_load_on_non_virtual_passes_through() {
        // When the buffer is NOT virtual, raw_load should pass through unchanged.
        let ad = raw_arraydescr();
        let mut ops = vec![
            Op::with_descr(
                OpCode::RawStore,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 50),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 200),
                ],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::RawLoadI,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Int, 50),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                ad,
            ),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef::int_op(100), Value::Int(0))];
        // No raw_bufs — OpRef::int_op(50) is NOT a virtual buffer.
        let result = run_pass_with_raw_buffer(&ops, &constants, &[]);
        assert_eq!(
            result.len(),
            2,
            "non-virtual raw ops should pass through; got {} ops",
            result.len()
        );
        assert_eq!(result[0].opcode, OpCode::RawStore);
        assert_eq!(result[1].opcode, OpCode::RawLoadI);
    }

    #[test]
    fn test_call_forced_virtual_immutable_getfield() {
        // RPython test_optimizeopt.py:test_forced_virtual_pure_getfield
        //
        // [p0]
        // p1 = new_with_vtable(descr=nodesize3)
        // setfield_gc(p1, p0, descr=valuedescr3)   <- immutable field
        // call_n(p1)
        // p2 = getfield_gc_r(p1, descr=valuedescr3)
        // call_n(p2)
        // jump(p0)
        //
        // Expected:
        // [p0]
        // p1 = new_with_vtable(descr=nodesize3)
        // setfield_gc(p1, p0, descr=valuedescr3)
        // call_n(p1)
        // call_n(p0)
        // jump(p0)
        let group = majit_ir::descr::make_simple_descr_group(
            1,
            16,
            1,
            0,
            &[majit_ir::descr::SimpleFieldDescrSpec {
                is_class_word: Some(false),
                index: 10,
                field_key: "Node.value".to_string(),
                name: "Node.value".to_string(),
                offset: 0,
                field_size: 8,
                field_type: Type::Ref,
                is_immutable: true,
                is_quasi_immutable: false,
                flag: majit_ir::ArrayFlag::Unsigned,
                virtualizable: false,
                index_in_parent: 0,
            }],
        );
        let sd = group.size_descr.clone() as DescrRef;
        let fd = group.field_descrs[0].clone() as DescrRef;
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 100),
                ],
                fd.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
            ),
            Op::with_descr(
                OpCode::GetfieldGcR,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                fd.clone(),
            ),
            Op::new(
                OpCode::CallN,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    3,
                )],
            ),
            Op::new(
                OpCode::Jump,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    100,
                )],
            ),
        ];
        assign_positions(&mut ops);

        // p0 is the loop inputarg; bind it as a Ref inputarg producer so the
        // forced immutable getfield forwards to it without re-resolving a
        // position-only box.
        let result = run_default_pipeline(&ops);
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetfieldGcR),
            "forced immutable getfield should be removed; got {opcodes:?}"
        );
        assert_eq!(
            opcodes,
            vec![
                OpCode::NewWithVtable,
                OpCode::SetfieldGc,
                OpCode::CallN,
                OpCode::CallN,
                OpCode::Jump,
            ]
        );
        assert_eq!(result[3].arg(0).to_opref(), OpRef::input_arg_ref(100));
        assert_eq!(result[4].arg(0).to_opref(), OpRef::input_arg_ref(100));
    }

    #[test]
    fn test_jump_forces_virtual_value_lazy_setfield() {
        // At the trace end, flush() forces all lazy sets (heap.py
        // flush → force_all_lazy_sets). force_lazy_set (heap.py)
        // re-sends the SetfieldGc with emit=False, which routes it past
        // OptHeap through the rest of the chain — NOT back into the lazy
        // cache — and the final emission forces boxes in its args
        // (optimizer.py _emit_operation force_box). A virtual
        // stored into a non-virtual escapes: the New materializes and the
        // store is emitted before the Jump.
        //
        // [p0]
        // p1 = new(descr=node)
        // setfield_gc(p0, p1, descr=next)
        // jump(p0)
        let node_sd = size_descr(1);
        let next_fd = ref_field_descr(11);
        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], node_sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                ],
                next_fd.clone(),
            ),
            Op::new(
                OpCode::Jump,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
            ),
        ];
        ops[0].pos.set(OpRef::ref_op(1));
        ops[1].pos.set(OpRef::void_op(2));
        ops[2].pos.set(OpRef::void_op(3));
        let mut opt = Optimizer::default_pipeline();
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        let opcodes: Vec<_> = result.iter().map(|op| op.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::New, OpCode::SetfieldGc, OpCode::Jump],
            "virtual New stored into a non-virtual escapes at flush; got {result:?}"
        );
    }

    // A residual call receiving a fresh virtual must force its pending float
    // field store before the escape.
    #[test]
    fn test_callr_preserves_float_field_store_on_escaping_fresh_object() {
        let float_sd = size_descr(1);
        let float_fd = float_field_descr(10);
        let call_descr: DescrRef = Arc::new(majit_ir::SimpleCallDescr::new(
            77,
            vec![Type::Ref],
            Type::Ref,
            false,
            8,
            majit_ir::EffectInfo::default(),
        ));

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], float_sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Float, 100),
                ],
                float_fd,
            ),
            Op::with_descr(
                OpCode::CallR,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Ref,
                    0,
                )],
                call_descr,
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        // The float field value is a loop inputarg (Float) re-resolved by the
        // escaping call; bind it as a typed inputarg producer.
        let result = run_default_pipeline_typed(&ops, &[], &[100]);
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                OpCode::NewWithVtable,
                OpCode::SetfieldGc,
                OpCode::CallR,
                OpCode::Jump,
            ],
            "escaping fresh float object must keep its floatval store before the call; got {result:?}"
        );
    }

    #[test]
    fn test_finish_forces_virtual_refs_to_emitted_allocations() {
        let node_sd = size_descr(1);
        let value_fd = field_descr(10);
        let next_fd = ref_field_descr(11);

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], node_sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 2),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 100),
                ],
                value_fd.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 2),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
                next_fd.clone(),
            ),
            Op::with_descr(OpCode::New, &[], node_sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 5),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 101),
                ],
                value_fd.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 5),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 2),
                ],
                next_fd.clone(),
            ),
            Op::new(
                OpCode::Finish,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 5),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 2),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 1),
                    crate::history::test_support::rooted_inputarg_operand(Type::Ref, 0),
                ],
            ),
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed((idx + 2) as u32, op.opcode.result_type()));
        }

        let mut opt = Optimizer::default_pipeline();
        let mut constants: majit_ir::ConstMap<majit_ir::Value> = majit_ir::ConstMap::new();
        constants.insert(100u32, majit_ir::Value::Int(7));
        constants.insert(101u32, majit_ir::Value::Int(11));
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: Vec<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos.get())
            .collect();
        assert_eq!(
            new_positions.len(),
            2,
            "expected two forced allocations; got {result:?}"
        );

        for set_op in result.iter().filter(|op| op.opcode == OpCode::SetfieldGc) {
            assert!(
                new_positions.contains(&set_op.arg(0).to_opref()),
                "SetfieldGc target must be one of the emitted News; got {:?} in {:?}",
                set_op.arg(0),
                result
            );
        }

        let finish = result
            .iter()
            .find(|op| op.opcode == OpCode::Finish)
            .expect("optimized trace should keep Finish");
        assert!(
            new_positions.contains(&finish.arg(0).to_opref()),
            "first Finish ref should be a forced allocation; got {:?} in {:?}",
            finish.arg(0),
            result
        );
        assert!(
            new_positions.contains(&finish.arg(1).to_opref()),
            "second Finish ref should be a forced allocation; got {:?} in {:?}",
            finish.arg(1),
            result
        );
        assert!(
            !constants.contains_key(&finish.arg(0).to_opref().raw()),
            "forced allocation ref must not collide with an exported int constant"
        );
        assert!(
            !constants.contains_key(&finish.arg(1).to_opref().raw()),
            "forced allocation ref must not collide with an exported int constant"
        );
    }

    #[test]
    fn test_guard_fail_args_virtual_not_forced() {
        // resume.py parity: virtual objects in guard fail_args should NOT be
        // forced (no allocation emitted). rd_numb with TAGVIRTUAL is set.
        //
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // guard_true(i20) [p0]
        //
        // Expected: no NEW_WITH_VTABLE emitted. Guard has rd_numb and
        // rd_virtuals; liveboxes contain TAGBOX field values only.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                20,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()), // pos=0
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 10),
                ],
                fd.clone(),
            ), // pos=1
            guard,                                                  // pos=2
        ];
        assign_positions(&mut ops);

        // i10 (field value) and i20 (guard cond) are loop inputargs; bind them
        // as typed inputarg producers so resume numbering resolves the live
        // field value to a canonical box. p0 (the virtual) stays an in-trace
        // ResOp producer, encoded into rd_virtuals rather than liveboxes.
        let result = run_pass_typed(&ops, &[10, 20]);

        // The virtual should NOT be forced — no NEW_WITH_VTABLE emitted
        let new_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::NewWithVtable)
            .count();
        assert_eq!(
            new_count,
            0,
            "virtual in guard fail_args should NOT be forced; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // resume.py parity: liveboxes_from_env contains TAGBOX entries
        // for the virtual's field values; the virtual itself is encoded via
        // TAGVIRTUAL into rd_virtuals (no slot in liveboxes).
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(10)),
            "virtual's int field (input_arg_int(10)) should appear in liveboxes; got {:?}",
            fa
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "virtual structure should be encoded into rd_virtuals tree"
        );
    }

    #[test]
    fn test_guard_fail_args_mixed_virtual_and_non_virtual() {
        // Guard with both virtual and non-virtual fail_args.
        //
        // p0 = new(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // guard_true(i20) [i30, p0, i40]
        //
        // RPython resume.py parity: liveboxes is TAGBOX-only — virtual
        // p0 is encoded into rd_virtuals; the surviving liveboxes are the
        // concrete TAGBOX boxes (OpRef::int_op(30), OpRef::int_op(40), and the virtual's
        // field value OpRef::int_op(10)).
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                20,
            )],
        );
        guard.setfailargs(
            vec![
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 30),
                crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 40),
            ]
            .into(),
        );

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()), // pos=0
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 10),
                ],
                fd.clone(),
            ), // pos=1
            guard,                                        // pos=2
        ];
        assign_positions(&mut ops);

        // i10 (field value), i30/i40 (non-virtual liveboxes) and i20 (guard
        // cond) are loop inputargs; bind them as typed inputarg producers so
        // resume numbering resolves each live box to a canonical box. p0 (the
        // virtual) stays an in-trace ResOp producer, encoded into rd_virtuals.
        let result = run_pass_typed(&ops, &[10, 20, 30, 40]);

        // No allocation emitted
        let new_count = result
            .iter()
            .filter(|o| matches!(o.opcode, OpCode::New | OpCode::NewWithVtable))
            .count();
        assert_eq!(new_count, 0, "virtual should not be forced");

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // resume.py parity: liveboxes is TAGBOX-only.
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(30)),
            "non-virtual input_arg_int(30) should remain in liveboxes; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(40)),
            "non-virtual input_arg_int(40) should remain in liveboxes; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(10)),
            "virtual's field (input_arg_int(10)) should appear in liveboxes; got {:?}",
            fa
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "virtual structure should be encoded into rd_virtuals tree"
        );
    }

    #[test]
    fn test_guard_fail_args_no_virtual_no_rd_numb() {
        // Guard with no virtuals in fail_args should not have rd_numb.
        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                10,
            )],
        );
        guard.setfailargs(
            vec![
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 20),
                crate::history::test_support::rooted_inputarg_operand(Type::Int, 30),
            ]
            .into(),
        );
        let mut ops = vec![guard];
        assign_positions(&mut ops);

        // The guard condition and live fail_args are loop inputargs; bind them
        // as typed inputarg producers so the resume numbering resolves each to
        // a canonical box instead of a position-only fallback.
        let result = run_pass_typed(&ops, &[10, 20, 30]);
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        // No virtuals — fail_args should remain as-is with concrete values.
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "no virtuals => all fail_args should be concrete"
        );
    }

    #[test]
    fn test_guard_fail_args_virtual_struct_not_forced() {
        // VirtualStruct (New) in guard fail_args should also use resume data.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                20,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 10),
                ],
                fd.clone(),
            ),
            guard,
        ];
        assign_positions(&mut ops);

        // i10 (field value) and i20 (guard cond) are loop inputargs; bind them
        // as typed inputarg producers so resume numbering resolves the live
        // field value to a canonical box. p0 (the virtual) stays an in-trace
        // ResOp producer, encoded into rd_virtuals rather than liveboxes.
        let result = run_pass_typed(&ops, &[10, 20]);

        let new_count = result
            .iter()
            .filter(|o| matches!(o.opcode, OpCode::New | OpCode::NewWithVtable))
            .count();
        assert_eq!(new_count, 0, "virtual struct should not be forced");

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );
        // resume.py parity: liveboxes is TAGBOX-only.
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(10)),
            "virtual struct's int field should appear in liveboxes; got {:?}",
            fa
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "virtual struct should be encoded into rd_virtuals tree"
        );
    }

    #[test]
    fn test_guard_fail_args_virtual_with_multiple_fields() {
        // Virtual with two fields in guard fail_args.
        let sd = size_descr(1);
        let fd_a = field_descr(10);
        let fd_b = field_descr(20);

        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                30,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 10),
                ],
                fd_a.clone(),
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 20),
                ],
                fd_b.clone(),
            ),
            guard,
        ];
        assign_positions(&mut ops);

        // The two field values and the guard cond are loop inputargs; bind them
        // as typed inputarg producers so resume numbering resolves each live
        // field value to a canonical box. p0 (the virtual) stays an in-trace
        // ResOp producer, encoded into rd_virtuals rather than liveboxes.
        let result = run_pass_typed(&ops, &[10, 20, 30]);

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // resume.py parity: liveboxes is TAGBOX-only.
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fa
        );
        // Both field values must appear in liveboxes.
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(10)),
            "first field value (input_arg_int(10)) should appear in liveboxes; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(20)),
            "second field value (input_arg_int(20)) should appear in liveboxes; got {:?}",
            fa
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "virtual structure should be encoded into rd_virtuals tree"
        );
    }

    #[test]
    fn test_guard_fail_args_nested_virtual_field_encodes_into_rd_virtuals() {
        // Nested virtual: outer.field = inner_virtual (Ref), inner.field = OpRef::int_op(40) (Int).
        // RPython resume.py:_number_virtuals (resume.py _number_virtuals;
        // visitor_walk_recursive at resume.py:426) recursively encodes nested
        // virtuals as TAGVIRTUAL inside rd_virtuals; no New/NewWithVtable is
        // materialized at numbering time.  Liveboxes only carry the leaf
        // TAGBOX values.
        let outer_sd = size_descr(1);
        let inner_sd = size_descr(2);
        let outer_fd = ref_field_descr(10);
        let inner_fd = field_descr(20);

        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                30,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], outer_sd),
            Op::with_descr(OpCode::New, &[], inner_sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 40),
                ],
                inner_fd,
            ),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                ],
                outer_fd,
            ),
            guard,
        ];
        assign_positions(&mut ops);

        // The leaf int field value (i40) and guard cond (i30) are loop
        // inputargs; bind them as typed inputarg producers so resume numbering
        // resolves the leaf livebox to a canonical box. The outer/inner virtuals
        // stay in-trace ResOp producers, encoded into rd_virtuals.
        let result = run_pass_typed(&ops, &[30, 40]);
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        // No concrete allocations emitted — both virtuals stay TAGVIRTUAL.
        assert_eq!(
            result
                .iter()
                .filter(|op| matches!(op.opcode, OpCode::New | OpCode::NewWithVtable))
                .count(),
            0,
            "nested virtuals should stay virtual; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb after RPython numbering"
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "rd_virtuals should encode the nested virtual tree"
        );

        // Liveboxes are TAGBOX-only — only the leaf int OpRef::int_op(40) survives.
        let fa = guard_op.getfailargs().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fa
        );
        assert!(
            fa.iter().any(|a| a.to_opref() == OpRef::input_arg_int(40)),
            "leaf int field (input_arg_int(40)) should appear in liveboxes; got {:?}",
            fa
        );
    }

    #[test]
    fn test_guard_fail_args_virtual_array_encodes_into_rd_virtuals() {
        // Virtual array: NewArray(len=1), set item 0 = OpRef::int_op(12).
        // RPython resume.py:_number_virtuals encodes the array virtually;
        // the array's elements are added to liveboxes as TAGBOX, the array
        // identity stays TAGVIRTUAL inside rd_virtuals.
        let ad = array_descr(30);
        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_resop_operand(
                Type::Int,
                20,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArray,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    10,
                )],
                ad.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 11),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 12),
                ],
                ad,
            ),
            guard,
        ];
        assign_positions(&mut ops);

        let constants = &[
            (OpRef::int_op(10), Value::Int(1)),
            (OpRef::int_op(11), Value::Int(0)),
            (OpRef::int_op(12), Value::Int(99)),
        ];
        let result = run_pass_with_constants(&ops, constants);

        // No concrete NewArray allocation — virtual array stays virtual.
        assert_eq!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::NewArray)
                .count(),
            0,
            "virtual array should stay virtual; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.resolved_rd_numb().is_some(),
            "guard should have rd_numb after RPython numbering"
        );
        assert!(
            guard_op.resolved_rd_virtuals().is_some(),
            "rd_virtuals should encode the virtual array"
        );
    }

    #[test]
    fn test_guard_fail_args_virtual_array_with_nested_virtual_item() {
        // RPython resume.py:_number_virtuals recursively numbers a virtual
        // stored in a virtual array item.  The array and the item remain
        // TAGVIRTUAL; only the item's scalar payload is a TAGBOX live value.
        let array_sd = array_descr(40);
        let item_sd = size_descr(41);
        let item_value_fd = field_descr(42);
        let mut guard = Op::new(
            OpCode::GuardTrue,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Int,
                30,
            )],
        );
        guard.setfailargs(
            vec![crate::history::test_support::rooted_resop_operand(
                Type::Ref,
                0,
            )]
            .into(),
        );
        let mut ops = vec![
            Op::with_descr(
                OpCode::NewArray,
                &[crate::history::test_support::rooted_resop_operand(
                    Type::Int,
                    10,
                )],
                array_sd.clone(),
            ),
            Op::with_descr(OpCode::NewWithVtable, &[], item_sd),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                    crate::history::test_support::rooted_inputarg_operand(Type::Int, 40),
                ],
                item_value_fd,
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 0),
                    crate::history::test_support::rooted_resop_operand(Type::Int, 11),
                    crate::history::test_support::rooted_resop_operand(Type::Ref, 1),
                ],
                array_sd,
            ),
            guard,
        ];
        assign_positions(&mut ops);

        let result = run_pass_with_constants(
            &ops,
            &[
                (OpRef::int_op(10), Value::Int(1)),
                (OpRef::int_op(11), Value::Int(0)),
            ],
        );
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");
        assert_eq!(
            result
                .iter()
                .filter(|op| matches!(op.opcode, OpCode::NewArray | OpCode::NewWithVtable))
                .count(),
            0,
            "nested virtual array item should remain virtual; got {result:?}"
        );
        let failargs = guard_op.getfailargs().unwrap();
        assert!(
            failargs
                .iter()
                .any(|arg| arg.to_opref() == OpRef::input_arg_int(40)),
            "nested item payload should be a live TAGBOX; got {failargs:?}"
        );
        let virtuals = guard_op
            .resolved_rd_virtuals()
            .expect("rd_virtuals should encode the array and nested item");
        // A count alone also passes when the two are numbered as UNRELATED
        // virtuals. What the bridge decoder follows is the link: it reads the
        // array's item slot, resolves its TAGVIRTUAL number back into
        // `rd_virtuals`, and materializes the entry it lands on. Assert that
        // path, locating each end by what it is rather than by slot order.
        use majit_ir::resumedata::{TAGBOX, TAGVIRTUAL, UNINITIALIZED_TAG, untag};

        let array_idx = {
            let found = virtuals
                .iter()
                .enumerate()
                .filter(|(_, v)| {
                    matches!(
                        v.as_ref(),
                        majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
                            | majit_ir::RdVirtualInfo::VArrayInfoNotClear { .. }
                    )
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            assert_eq!(
                found.len(),
                1,
                "exactly one VArrayInfo entry expected; got {virtuals:?}"
            );
            found[0]
        };
        let (array_arraydescr, array_items) = match virtuals[array_idx].as_ref() {
            majit_ir::RdVirtualInfo::VArrayInfoClear {
                arraydescr,
                fieldnums,
                ..
            }
            | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                arraydescr,
                fieldnums,
                ..
            } => (arraydescr.clone(), fieldnums.clone()),
            _ => unreachable!("filtered by the matches! above"),
        };
        // The materializer dereferences the arraydescr unconditionally.
        assert!(
            array_arraydescr.is_some(),
            "VArrayInfo must carry its live arraydescr"
        );
        assert_eq!(
            array_items.len(),
            1,
            "NEW_ARRAY(len=1) leaves exactly one item slot; got {array_items:?}"
        );
        let (item_num, item_tagbits) = untag(array_items[0]);
        assert_eq!(
            item_tagbits,
            TAGVIRTUAL,
            "array item 0 must stay TAGVIRTUAL rather than force-boxing to a \
             failarg; got {:?}",
            untag(array_items[0])
        );
        // resume.py assign_number_to_virtual numbers nested virtuals
        // negatively and resume.py getvirtual_ptr resolves them by
        // Python negative indexing.
        let item_idx = if item_num < 0 {
            (virtuals.len() as i32 + item_num) as usize
        } else {
            item_num as usize
        };
        assert_ne!(
            item_idx, array_idx,
            "the item's TAGVIRTUAL must name a slot other than the array's"
        );
        let majit_ir::RdVirtualInfo::VirtualInfo {
            descr: item_descr,
            fielddescrs: item_fielddescrs,
            fieldnums: item_fieldnums,
            ..
        } = virtuals[item_idx].as_ref()
        else {
            panic!(
                "the array item's TAGVIRTUAL must resolve to the nested \
                 NEW_WITH_VTABLE VirtualInfo; got {:?}",
                virtuals[item_idx]
            )
        };
        // A `None` size descr silently drops the whole nested materialization.
        assert!(
            item_descr.is_some(),
            "nested VirtualInfo must carry its live SizeDescr"
        );
        assert_eq!(
            item_fielddescrs.len(),
            item_fieldnums.len(),
            "fielddescrs and fieldnums must stay 1:1"
        );
        // resume.py setfields skips UNINITIALIZED, so the item's only
        // live slot is the one the SETFIELD_GC wrote.
        let live = item_fieldnums
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, n)| n != UNINITIALIZED_TAG)
            .collect::<Vec<_>>();
        assert_eq!(
            live.len(),
            1,
            "nested item has exactly one live field slot; got {live:?}"
        );
        let (live_slot, live_tag) = live[0];
        assert_eq!(
            item_fielddescrs[live_slot].index, 42,
            "the live slot must be the field_descr(42) the SETFIELD_GC named; \
             got {:?}",
            item_fielddescrs[live_slot]
        );
        let (leaf_num, leaf_tagbits) = untag(live_tag);
        assert_eq!(
            leaf_tagbits,
            TAGBOX,
            "the nested item's leaf must be a TAGBOX failarg; got {:?}",
            untag(live_tag)
        );
        // resume.py decode_box indexes liveboxes with the same
        // negative-index rule.
        let leaf_slot = if leaf_num < 0 {
            leaf_num + failargs.len() as i32
        } else {
            leaf_num
        } as usize;
        assert_eq!(
            failargs[leaf_slot].to_opref(),
            OpRef::input_arg_int(40),
            "the leaf TAGBOX must resolve to the SETFIELD_GC value i40; \
             got {failargs:?}"
        );
    }
}
