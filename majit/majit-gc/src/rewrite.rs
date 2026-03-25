/// GC rewriter — transforms high-level allocation and store operations
/// into GC-aware lower-level IR before code generation.
///
/// Converts:
/// - NEW / NEW_WITH_VTABLE -> CALL_MALLOC_NURSERY + tid initialization
///   (consecutive fixed-size allocations are batched into a single
///    CALL_MALLOC_NURSERY with NURSERY_PTR_INCREMENT for subsequent objects)
/// - NEW_ARRAY / NEW_ARRAY_CLEAR -> CALL_MALLOC_NURSERY_VARSIZE
/// - SETFIELD_GC with a Ref-typed value -> COND_CALL_GC_WB + SETFIELD_GC
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
use std::collections::{HashMap, HashSet};

use majit_ir::Type;
use majit_ir::descr::{DescrRef, FieldDescr};
use majit_ir::resoperation::{Op, OpCode, OpRef};

use crate::{GcRewriter, WriteBarrierDescr};

/// Alignment for nursery allocations (8 bytes).
const NURSERY_ALIGN: usize = 8;

/// Align `size` up to `NURSERY_ALIGN`.
fn round_up(size: usize) -> usize {
    (size + NURSERY_ALIGN - 1) & !(NURSERY_ALIGN - 1)
}

/// GC rewriter implementation.
///
/// Walks a list of IR operations and rewrites allocation / store ops
/// into backend-friendly forms (CALL_MALLOC_NURSERY, COND_CALL_GC_WB, etc.).
pub struct GcRewriterImpl {
    /// Nursery free pointer address (for inline allocation).
    pub nursery_free_addr: usize,
    /// Nursery top pointer address.
    pub nursery_top_addr: usize,
    /// Maximum object size for nursery allocation.
    pub max_nursery_size: usize,
    /// Write barrier descriptor.
    pub wb_descr: WriteBarrierDescr,
    /// JitFrame info for call_assembler rewriting.
    /// rewrite.py:665 — handle_call_assembler needs frame layout info.
    pub jitframe_info: Option<JitFrameDescrs>,
}

/// JitFrame field descriptors for handle_call_assembler.
///
/// rewrite.py:666 — `descrs = self.gc_ll_descr.getframedescrs(self.cpu)`
#[derive(Clone)]
pub struct JitFrameDescrs {
    /// Address of `jit_create_self_recursive_callee_frame_1_raw_int`.
    pub create_fn_addr: usize,
    /// Address of `jit_drop_callee_frame`.
    pub drop_fn_addr: usize,
    /// GC type id for JitFrame (from gc.register_type).
    pub jitframe_tid: u32,
    /// JitFrame fixed header size (bytes).
    pub jitframe_fixed_size: usize,
    /// Byte offsets of JitFrame fields (from majit_meta::jitframe).
    pub jf_frame_info_ofs: i32,
    pub jf_descr_ofs: i32,
    pub jf_force_descr_ofs: i32,
    pub jf_savedata_ofs: i32,
    pub jf_guard_exc_ofs: i32,
    pub jf_forward_ofs: i32,
    /// Offset from JitFrame start to jf_frame array.
    pub jf_frame_ofs: usize,
    /// BASEITEMOFS: offset of first item within the jf_frame array.
    pub jf_frame_baseitemofs: usize,
    /// LENGTHOFS: offset of length field within the jf_frame array.
    pub jf_frame_lengthofs: usize,
    /// SIGN_SIZE: size of one jf_frame slot.
    pub sign_size: usize,
}

/// A deferred ZERO_ARRAY emission that can be elided or shortened when
/// subsequent SETARRAYITEM writes cover parts of the array.
struct PendingZero {
    /// The OpRef of the array being zeroed.
    array_ref: OpRef,
    /// Start index of the zero range.
    start: usize,
    /// Length (number of items) of the zero range.
    length: usize,
    /// The original op (used for descriptor propagation).
    original_op: Op,
}

/// Per-rewrite mutable state (not stored on the struct so that
/// `rewrite_for_gc` can take `&self`).
struct RewriteState {
    /// Output operation list.
    out: Vec<Op>,
    /// Next position index for emitted result ops that do not have an
    /// explicit source position to preserve.
    next_pos: u32,
    /// Constant pool (from optimizer) — maps OpRef key → i64 value.
    constants: HashMap<u32, i64>,

    // ── Nursery batching ──
    /// The index in `out` of the current CALL_MALLOC_NURSERY op, if any.
    /// We try to merge consecutive small allocations into one bump.
    pending_malloc_idx: Option<usize>,
    /// Total size accumulated in the current CALL_MALLOC_NURSERY.
    pending_malloc_total: usize,
    /// Size of the *previous* individual allocation (used for
    /// NURSERY_PTR_INCREMENT offset).
    previous_size: usize,
    /// OpRef of the last object produced by the current nursery batch.
    last_malloced_ref: OpRef,

    // ── Write barrier tracking ──
    /// Set of OpRef indices whose write barrier has already been emitted
    /// (freshly allocated objects, or objects we already issued a WB for).
    /// Cleared whenever we emit an operation that can trigger a collection.
    wb_applied: HashSet<u32>,
    /// Forwarding map from original result OpRefs to rewritten result OpRefs.
    forwarding: HashMap<u32, OpRef>,
    /// One-shot marker for an already-emitted array write barrier.
    ///
    /// Re-running the rewriter over an already-rewritten trace should not
    /// duplicate a `COND_CALL_GC_WB_ARRAY` immediately followed by the
    /// corresponding `SETARRAYITEM_GC`.
    pending_array_wb: Option<u32>,

    // ── Pending zero tracking ──
    /// Deferred ZERO_ARRAY ops that may be optimized away if subsequent
    /// SETARRAYITEM writes cover the entire range.
    pending_zeros: Vec<PendingZero>,
    /// Tracks which array indices have been explicitly SET since the
    /// pending zero was recorded. Keyed by array OpRef index.
    initialized_indices: HashMap<u32, HashSet<usize>>,
}

impl RewriteState {
    fn new(hint: usize, next_pos: u32) -> Self {
        RewriteState {
            out: Vec::with_capacity(hint + hint / 4),
            next_pos,
            constants: HashMap::new(),
            pending_malloc_idx: None,
            pending_malloc_total: 0,
            previous_size: 0,
            last_malloced_ref: OpRef::NONE,
            wb_applied: HashSet::new(),
            forwarding: HashMap::new(),
            pending_array_wb: None,
            pending_zeros: Vec::new(),
            initialized_indices: HashMap::new(),
        }
    }

    fn with_constants(hint: usize, next_pos: u32, constants: HashMap<u32, i64>) -> Self {
        let mut s = Self::new(hint, next_pos);
        s.constants = constants;
        s
    }

    /// Resolve a constant value from the constant pool.
    fn resolve_constant(&self, key: u32) -> Option<i64> {
        self.constants.get(&key).copied()
    }

    /// Emit an op. Void ops do not consume a result id.
    fn emit(&mut self, mut op: Op) -> OpRef {
        let pos = if op.result_type() == Type::Void {
            OpRef::NONE
        } else {
            let pos = OpRef(self.next_pos);
            self.next_pos += 1;
            pos
        };
        op.pos = pos;
        self.out.push(op);
        pos
    }

    /// Emit a result-producing op, preserving the provided position when the
    /// source trace already assigned one.
    fn emit_result(&mut self, mut op: Op, preferred_pos: OpRef) -> OpRef {
        let pos = if preferred_pos.is_none() {
            let pos = OpRef(self.next_pos);
            self.next_pos += 1;
            pos
        } else {
            preferred_pos
        };
        op.pos = pos;
        self.out.push(op);
        pos
    }

    /// Mark that we are about to emit something that can trigger GC.
    /// This invalidates nursery batching and write barrier memoisation.
    fn emitting_can_collect(&mut self) {
        self.pending_malloc_idx = None;
        self.wb_applied.clear();
        self.pending_array_wb = None;
    }

    fn remember_wb(&mut self, r: OpRef) {
        self.wb_applied.insert(r.0);
    }

    fn wb_already_applied(&self, r: OpRef) -> bool {
        self.wb_applied.contains(&r.0)
    }

    fn resolve(&self, r: OpRef) -> OpRef {
        if r.is_none() {
            return r;
        }
        self.forwarding.get(&r.0).copied().unwrap_or(r)
    }

    fn rewrite_op(&self, op: &Op) -> Op {
        let mut rewritten = op.clone();
        for arg in rewritten.args.iter_mut() {
            *arg = self.resolve(*arg);
        }
        if let Some(fail_args) = rewritten.fail_args.as_mut() {
            for arg in fail_args.iter_mut() {
                *arg = self.resolve(*arg);
            }
        }
        rewritten.pos = OpRef::NONE;
        rewritten
    }

    fn record_result_mapping(&mut self, old_pos: OpRef, new_pos: OpRef) {
        if !old_pos.is_none() {
            self.forwarding.insert(old_pos.0, new_pos);
        }
    }

    fn emit_rewritten_from(&mut self, original: &Op, rewritten: Op) -> OpRef {
        let result = if original.result_type() == Type::Void {
            self.emit(rewritten)
        } else {
            self.emit_result(rewritten, original.pos)
        };
        if original.result_type() != Type::Void {
            self.record_result_mapping(original.pos, result);
        }
        result
    }

    fn take_pending_array_wb(&mut self, obj: OpRef) -> bool {
        let matched = self.pending_array_wb == Some(obj.0);
        self.pending_array_wb = None;
        matched
    }

    /// Record that a SETARRAYITEM wrote to `array_ref[index]`,
    /// so the pending zero for that slot can be skipped.
    fn record_setarrayitem_index(&mut self, array_ref: OpRef, index: usize) {
        if self
            .pending_zeros
            .iter()
            .any(|pz| pz.array_ref == array_ref)
        {
            self.initialized_indices
                .entry(array_ref.0)
                .or_default()
                .insert(index);
        }
    }

    /// Flush all pending ZERO_ARRAY ops, emitting optimized versions
    /// that skip indices already covered by SETARRAYITEM writes.
    fn flush_pending_zeros(&mut self) {
        let pending = std::mem::take(&mut self.pending_zeros);
        let inited = std::mem::take(&mut self.initialized_indices);

        for pz in pending {
            let written = inited.get(&pz.array_ref.0);

            // Find contiguous zero ranges by excluding written indices.
            let mut i = pz.start;
            let end = pz.start + pz.length;

            while i < end {
                // Skip indices already initialized by SETARRAYITEM.
                if written.is_some_and(|s| s.contains(&i)) {
                    i += 1;
                    continue;
                }

                // Start of a zero run.
                let run_start = i;
                while i < end && !written.is_some_and(|s| s.contains(&i)) {
                    i += 1;
                }
                let run_len = i - run_start;

                // Emit ZERO_ARRAY(array, start_const, length_const, 0, descr).
                let mut zero_op = Op::new(
                    OpCode::ZeroArray,
                    &[
                        pz.array_ref,
                        OpRef(run_start as u32),
                        OpRef(run_len as u32),
                        OpRef(0),
                        OpRef(0),
                    ],
                );
                zero_op.descr = pz.original_op.descr.clone();
                self.emit(zero_op);
            }
        }
    }
}

impl GcRewriterImpl {
    /// Can we use the nursery for this allocation size?
    fn can_use_nursery(&self, size: usize) -> bool {
        size <= self.max_nursery_size
    }

    // ────────────────────────────────────────────────────────
    // NEW / NEW_WITH_VTABLE  → CALL_MALLOC_NURSERY + tid init
    // ────────────────────────────────────────────────────────

    fn handle_new(&self, op: &Op, st: &mut RewriteState) {
        let descr = op
            .descr
            .as_ref()
            .expect("NEW must have a SizeDescr")
            .as_size_descr()
            .expect("NEW descr must be SizeDescr");

        let size = round_up(descr.size());
        let type_id = descr.type_id();

        let obj_ref = self.gen_malloc_nursery(size, op.pos, st);
        st.record_result_mapping(op.pos, obj_ref);

        // Initialize the tid header field.
        self.gen_initialize_tid(obj_ref, type_id, st);

        // For NEW_WITH_VTABLE, also set the vtable pointer.
        if op.opcode == OpCode::NewWithVtable {
            let vtable = descr.vtable();
            if vtable != 0 {
                self.gen_initialize_vtable(obj_ref, vtable, st);
            }
        }
    }

    // ────────────────────────────────────────────────────────
    // NEW_ARRAY / NEW_ARRAY_CLEAR  → CALL_MALLOC_NURSERY_VARSIZE
    // ────────────────────────────────────────────────────────

    fn handle_new_array(&self, op: &Op, st: &mut RewriteState) {
        let is_clear = op.opcode == OpCode::NewArrayClear;

        let descr_ref = op
            .descr
            .as_ref()
            .expect("NEW_ARRAY must have an ArrayDescr");
        let descr = descr_ref
            .as_array_descr()
            .expect("NEW_ARRAY descr must be ArrayDescr");

        let _item_size = descr.item_size();
        let v_length = st.resolve(op.arg(0)); // the length operand

        // Emit CALL_MALLOC_NURSERY_VARSIZE(itemsize_const, length).
        // This is a potentially-collecting call, so flush state first.
        st.emitting_can_collect();

        let mut varsize_op = Op::new(OpCode::CallMallocNurseryVarsize, &[v_length]);
        varsize_op.descr = op.descr.clone();
        let result = st.emit_result(varsize_op, op.pos);
        st.record_result_mapping(op.pos, result);

        // Initialize the array length field.
        if let Some(len_descr) = descr.len_descr() {
            self.gen_initialize_len(result, v_length, descr_ref.clone(), len_descr, st);
        }

        // For NEW_ARRAY_CLEAR, defer ZERO_ARRAY emission so that
        // subsequent SETARRAYITEM writes can eliminate redundant zeroing.
        if is_clear && !v_length.is_none() {
            let length_val = v_length.0 as usize;
            if length_val > 0 {
                st.pending_zeros.push(PendingZero {
                    array_ref: result,
                    start: 0,
                    length: length_val,
                    original_op: op.clone(),
                });
            }
        }

        // Don't add to wb_applied: large young arrays may need card
        // marking, so the GC still relies on write barriers.
    }

    // ────────────────────────────────────────────────────────
    // SETFIELD_GC  → maybe COND_CALL_GC_WB + SETFIELD_GC
    // ────────────────────────────────────────────────────────

    fn handle_setfield_gc(&self, op: &Op, st: &mut RewriteState) {
        let rewritten = st.rewrite_op(op);
        let obj = rewritten.arg(0);

        // Determine whether the stored value is a Ref type from the field descriptor.
        let is_ref_store = op
            .descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.is_pointer_field())
            .unwrap_or(false);

        if is_ref_store && !st.wb_already_applied(obj) {
            // Emit COND_CALL_GC_WB(obj) before the store.
            let wb_op = Op::new(OpCode::CondCallGcWb, &[obj]);
            st.emit(wb_op);
            st.remember_wb(obj);
        }

        // Emit the original SETFIELD_GC unchanged.
        st.emit(rewritten);
    }

    // ────────────────────────────────────────────────────────
    // SETARRAYITEM_GC  → maybe COND_CALL_GC_WB_ARRAY + SETARRAYITEM_GC
    // ────────────────────────────────────────────────────────

    fn handle_setarrayitem_gc(&self, op: &Op, st: &mut RewriteState) {
        let rewritten = st.rewrite_op(op);
        let obj = rewritten.arg(0);
        // arg(1) = index, arg(2) = value

        // Track the index for pending zero optimization.
        let index_ref = rewritten.arg(1);
        if !index_ref.is_none() {
            st.record_setarrayitem_index(obj, index_ref.0 as usize);
        }

        let is_ref_store = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| ad.is_array_of_pointers())
            .unwrap_or(false);

        if st.take_pending_array_wb(obj) {
            st.emit(rewritten);
            return;
        }

        if is_ref_store && !st.wb_already_applied(obj) {
            // For arrays we emit the array variant which also passes the index.
            let index = rewritten.arg(1);
            let wb_op = Op::new(OpCode::CondCallGcWbArray, &[obj, index]);
            st.emit(wb_op);
            // Array WB is not enough to skip future barriers (card marking).
            // So we intentionally do NOT remember_wb here.
        }

        st.emit(rewritten);
    }

    // ────────────────────────────────────────────────────────
    // gen_malloc_nursery: batched bump-pointer allocation
    // ────────────────────────────────────────────────────────

    /// Try to emit (or extend) a CALL_MALLOC_NURSERY for `size` bytes.
    /// Returns the OpRef of the newly allocated object.
    fn gen_malloc_nursery(&self, size: usize, result_pos: OpRef, st: &mut RewriteState) -> OpRef {
        let size = round_up(size);

        if !self.can_use_nursery(size) {
            // Fallback: emit a plain CALL_MALLOC_NURSERY that the backend
            // will turn into a slow-path call. We still flush state.
            st.emitting_can_collect();
            let op = Op::new(OpCode::CallMallocNursery, &[OpRef(size as u32)]);
            let r = st.emit_result(op, result_pos);
            st.remember_wb(r);
            return r;
        }

        // Try to merge with a previous CALL_MALLOC_NURSERY.
        if let Some(prev_idx) = st.pending_malloc_idx {
            let new_total = st.pending_malloc_total + size;
            if self.can_use_nursery(new_total) {
                // Update the existing CALL_MALLOC_NURSERY's size arg.
                st.out[prev_idx].args[0] = OpRef(new_total as u32);
                st.pending_malloc_total = new_total;

                // Emit a NURSERY_PTR_INCREMENT to compute the address of
                // this object within the same nursery bump.
                let incr_op = Op::new(
                    OpCode::NurseryPtrIncrement,
                    &[st.last_malloced_ref, OpRef(st.previous_size as u32)],
                );
                let r = st.emit_result(incr_op, result_pos);
                st.previous_size = size;
                st.last_malloced_ref = r;
                st.remember_wb(r);
                return r;
            }
        }

        // Emit a fresh CALL_MALLOC_NURSERY.
        st.emitting_can_collect();
        let op = Op::new(OpCode::CallMallocNursery, &[OpRef(size as u32)]);
        let r = st.emit_result(op, result_pos);
        st.pending_malloc_idx = Some(st.out.len() - 1);
        st.pending_malloc_total = size;
        st.previous_size = size;
        st.last_malloced_ref = r;
        st.remember_wb(r);
        r
    }

    // ────────────────────────────────────────────────────────
    // Helpers for header initialisation
    // ────────────────────────────────────────────────────────

    /// Emit a GcStore to write the type-id into the object header.
    fn gen_initialize_tid(&self, obj: OpRef, tid: u32, st: &mut RewriteState) {
        // We encode the tid in a GcStore(obj, 0, tid) — offset 0 means the
        // header word which sits at (obj_ptr - HEADER_SIZE) in the actual
        // layout, but the backend interprets tid-init specially.
        let store = Op::new(OpCode::GcStore, &[obj, OpRef(0), OpRef(tid)]);
        st.emit(store);
    }

    /// Emit a GcStore to write the vtable pointer.
    fn gen_initialize_vtable(&self, obj: OpRef, vtable: usize, st: &mut RewriteState) {
        let store = Op::new(OpCode::GcStore, &[obj, OpRef(1), OpRef(vtable as u32)]);
        st.emit(store);
    }

    /// Emit a GcStore to initialise the array length field.
    fn gen_initialize_len(
        &self,
        obj: OpRef,
        length: OpRef,
        array_descr: DescrRef,
        _len_descr: &dyn FieldDescr,
        st: &mut RewriteState,
    ) {
        let mut store = Op::new(OpCode::GcStore, &[obj, OpRef(2), length]);
        store.descr = Some(array_descr);
        st.emit(store);
    }

    // ── rewrite.py:665-695 handle_call_assembler helpers ──────────

    /// Check if a CallR op is a call to the create_frame helper.
    fn is_create_frame_call(&self, op: &Op, st: &RewriteState) -> bool {
        let Some(ref descrs) = self.jitframe_info else {
            return false;
        };
        let func_addr_key = op.arg(0).0;
        let func_addr = st.resolve_constant(func_addr_key);
        func_addr == Some(descrs.create_fn_addr as i64)
    }

    /// Check if a CallN op is a call to the drop_frame helper.
    fn is_drop_frame_call(&self, op: &Op, st: &RewriteState) -> bool {
        let Some(ref descrs) = self.jitframe_info else {
            return false;
        };
        let func_addr_key = op.arg(0).0;
        let func_addr = st.resolve_constant(func_addr_key);
        func_addr == Some(descrs.drop_fn_addr as i64)
    }

    /// rewrite.py:665-695 — replace CallR(create_frame) with nursery
    /// allocation + JitFrame field initialization.
    ///
    /// Emits:
    ///   v_frame = CallMallocNurseryVarsizeFrame(size_const)
    ///   GcStore(v_frame, tid_ofs, tid)         # gen_initialize_tid
    ///   GcStore(v_frame, jf_descr, 0)          # zero GCREF fields
    ///   GcStore(v_frame, jf_force_descr, 0)
    ///   GcStore(v_frame, jf_savedata, 0)
    ///   GcStore(v_frame, jf_guard_exc, 0)
    ///   GcStore(v_frame, jf_forward, 0)
    ///   GcStore(v_frame, jf_frame_info, info_ptr)
    fn handle_create_frame(&self, op: &Op, st: &mut RewriteState) {
        let descrs = self.jitframe_info.as_ref().unwrap();

        // For now, use a fixed allocation size.  A full implementation
        // would read jfi_frame_size from the JitFrameInfo pointer (the
        // second CallR arg is the caller frame; the target's frame_info
        // is looked up via the compiled loop token).  For the current
        // self-recursive pattern the size is constant.
        //
        // TODO: emit GetfieldRawI to read jfi_frame_size at runtime
        // (rewrite.py:627-633).
        let alloc_size = descrs.jitframe_fixed_size + descrs.sign_size * (1 + 4); // length + 4 slots (conservative)
        let alloc_size = (alloc_size + 7) & !7; // align

        // CallMallocNurseryVarsizeFrame(size)
        // rewrite.py:634 — gen_malloc_nursery_varsize_frame
        st.emitting_can_collect();
        let size_ref = OpRef(alloc_size as u32);
        let malloc_op = Op::new(OpCode::CallMallocNurseryVarsizeFrame, &[size_ref]);
        let frame = st.emit_result(malloc_op, op.pos);
        st.remember_wb(frame);

        // gen_initialize_tid (rewrite.py:635)
        self.gen_initialize_tid(frame, descrs.jitframe_tid, st);

        // Zero GCREF fields (rewrite.py:641-650)
        let zero = OpRef(0);
        for &ofs in &[
            descrs.jf_descr_ofs,
            descrs.jf_force_descr_ofs,
            descrs.jf_savedata_ofs,
            descrs.jf_guard_exc_ofs,
            descrs.jf_forward_ofs,
        ] {
            st.emit(Op::new(OpCode::GcStore, &[frame, OpRef(ofs as u32), zero]));
        }

        // Set jf_frame_info (rewrite.py:671)
        // The caller_frame arg (op.arg(1)) carries the frame_info
        // via its code pointer.  For self-recursive calls the info
        // can be resolved statically; for now store 0 (the Cranelift
        // backend fills it at execution time via the dispatch entry).
        st.emit(Op::new(
            OpCode::GcStore,
            &[frame, OpRef(descrs.jf_frame_info_ofs as u32), zero],
        ));

        // Map this op's result to the nursery-allocated frame.
        st.record_result_mapping(op.pos, frame);
    }
}

impl GcRewriter for GcRewriterImpl {
    fn rewrite_for_gc(&self, ops: &[Op]) -> Vec<Op> {
        self.rewrite_for_gc_with_constants(ops, &HashMap::new())
    }

    fn rewrite_for_gc_with_constants(&self, ops: &[Op], constants: &HashMap<u32, i64>) -> Vec<Op> {
        let next_pos = ops
            .iter()
            .filter_map(|op| (!op.pos.is_none()).then_some(op.pos.0))
            .max()
            .map_or(0, |max_pos| max_pos.saturating_add(1));
        let mut st = RewriteState::with_constants(ops.len(), next_pos, constants.clone());

        for op in ops {
            if !matches!(
                op.opcode,
                OpCode::SetarrayitemGc | OpCode::CondCallGcWbArray | OpCode::DebugMergePoint
            ) {
                st.pending_array_wb = None;
            }

            match op.opcode {
                // Skip debug merge points (they carry no semantics).
                OpCode::DebugMergePoint => continue,

                // ── Allocation ──
                OpCode::New | OpCode::NewWithVtable => {
                    self.handle_new(op, &mut st);
                }
                OpCode::NewArray | OpCode::NewArrayClear => {
                    self.handle_new_array(op, &mut st);
                }
                // Newstr / Newunicode are array-like allocations.
                OpCode::Newstr | OpCode::Newunicode => {
                    self.handle_new_array(op, &mut st);
                }

                // ── Stores that may need a write barrier ──
                OpCode::SetfieldGc => {
                    self.handle_setfield_gc(op, &mut st);
                }
                OpCode::SetarrayitemGc => {
                    self.handle_setarrayitem_gc(op, &mut st);
                }

                // ── call_assembler: rewrite.py:414 handle_call_assembler ──
                OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN => {
                    st.flush_pending_zeros();
                    st.emitting_can_collect();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── rewrite.py:665 handle_call_assembler parity ──
                // Infrastructure for replacing CallR(create_frame) with
                // nursery alloc and eliding CallN(drop_frame).
                // Disabled until nursery allocation is fully wired:
                // - CallMallocNurseryVarsizeFrame must be lowered by Cranelift
                // - Guard failure must materialize PyFrame from JitFrame
                // - Arena put must be kept while arena is still used
                //
                // OpCode::CallR if self.is_create_frame_call(op, &st) => {
                //     self.handle_create_frame(op, &mut st);
                // }
                // OpCode::CallN if self.is_drop_frame_call(op, &st) => {
                //     // elide — GC collects
                // }

                // ── Operations that can trigger GC ──
                _ if op.opcode.can_malloc() => {
                    st.flush_pending_zeros();
                    st.emitting_can_collect();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Guards flush pending zeros but do not invalidate WB tracking. ──
                _ if op.opcode.is_guard() => {
                    st.flush_pending_zeros();
                    let rewritten = st.rewrite_op(op);
                    st.emit(rewritten);
                }

                // ── Everything else: pass through unchanged. ──
                OpCode::CondCallGcWb => {
                    let rewritten = st.rewrite_op(op);
                    let obj = rewritten.arg(0);
                    st.emit(rewritten);
                    st.remember_wb(obj);
                }
                OpCode::CondCallGcWbArray => {
                    let rewritten = st.rewrite_op(op);
                    let obj = rewritten.arg(0);
                    st.emit(rewritten);
                    st.pending_array_wb = Some(obj.0);
                }
                // ── Final ops (Jump, Finish) flush pending zeros before emit. ──
                _ if op.opcode.is_final() => {
                    st.flush_pending_zeros();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Everything else: pass through unchanged. ──
                _ => {
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }
            }
        }

        // Flush any remaining pending zeros at end of trace.
        st.flush_pending_zeros();

        st.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use majit_ir::descr::{ArrayDescr, Descr, DescrRef, SizeDescr};
    use majit_ir::value::Type;

    // ── Minimal concrete descriptor implementations for testing ──

    #[derive(Debug)]
    struct TestSizeDescr {
        size: usize,
        type_id: u32,
        vtable: usize,
    }

    impl Descr for TestSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            self.size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.vtable != 0
        }
        fn vtable(&self) -> usize {
            self.vtable
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
    }

    fn make_rewriter() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0,
                jit_wb_card_page_shift: 0,
            },
        }
    }

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: DescrRef) -> Op {
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = OpRef(pos);
        op
    }

    fn size_descr(size: usize, type_id: u32) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
        })
    }

    fn vtable_descr(size: usize, type_id: u32, vtable: usize) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable,
        })
    }

    fn ref_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 0,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn int_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 8,
            field_size: 8,
            field_type: Type::Int,
        })
    }

    fn array_descr_ref() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: 16,
            item_size: 8,
            type_id: 5,
            item_type: Type::Ref,
        })
    }

    fn array_descr_int() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: 16,
            item_size: 4,
            type_id: 6,
            item_type: Type::Int,
        })
    }

    // ── Test 1: NEW → CALL_MALLOC_NURSERY + tid init ──

    #[test]
    fn test_new_rewrite() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(OpCode::New, &[], size_descr(32, 7))];

        let result = rw.rewrite_for_gc(&ops);

        // Expect: CallMallocNursery, GcStore (tid)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        assert_eq!(result[0].args[0].0, 32); // size (already aligned)
        assert_eq!(result[1].opcode, OpCode::GcStore); // tid init
        assert_eq!(result[1].args[2].0, 7); // type_id = 7
    }

    // ── Test 2: NEW_ARRAY → CALL_MALLOC_NURSERY_VARSIZE ──

    #[test]
    fn test_new_array_rewrite() {
        let rw = make_rewriter();
        let length_ref = OpRef(100); // some prior op producing the length
        let ops = vec![Op::with_descr(
            OpCode::NewArray,
            &[length_ref],
            array_descr_int(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Expect: CallMallocNurseryVarsize
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
        );
        let varsize = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
            .unwrap();
        assert_eq!(varsize.args[0], length_ref);
    }

    // ── Test 3: SETFIELD_GC with Ref value → write barrier inserted ──

    #[test]
    fn test_setfield_gc_ref_needs_wb() {
        let rw = make_rewriter();
        let obj = OpRef(0);
        let val = OpRef(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[obj, val],
            ref_field_descr(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Expect: CondCallGcWb(obj), SetfieldGc(obj, val)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[0].args[0], obj);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
    }

    // ── Test 4: SETFIELD_GC with Int value → no write barrier ──

    #[test]
    fn test_setfield_gc_int_no_wb() {
        let rw = make_rewriter();
        let obj = OpRef(0);
        let val = OpRef(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[obj, val],
            int_field_descr(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // Only the original SetfieldGc, no write barrier.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }

    // ── Test 5: Non-GC ops pass through unchanged ──

    #[test]
    fn test_passthrough() {
        let rw = make_rewriter();
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(2)]),
            Op::new(OpCode::Jump, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 6: Multiple allocations are batched ──

    #[test]
    fn test_batched_allocations() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::with_descr(OpCode::New, &[], size_descr(32, 2)),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // First NEW: CallMallocNursery + GcStore(tid)
        // Second NEW: NurseryPtrIncrement + GcStore(tid)
        // And the original CallMallocNursery should have been updated
        // to the combined size.
        assert!(result.iter().any(|o| o.opcode == OpCode::CallMallocNursery));
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::NurseryPtrIncrement)
        );

        let malloc = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNursery)
            .unwrap();
        // Combined size: round_up(24) + round_up(32) = 24 + 32 = 56
        assert_eq!(malloc.args[0].0, 56);

        let incr = result
            .iter()
            .find(|o| o.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        // Offset = previous_size = round_up(24) = 24
        assert_eq!(incr.args[1].0, 24);

        // Both should have tid initialisation.
        let tid_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(tid_stores.len(), 2);
        assert_eq!(tid_stores[0].args[2].0, 1); // first type_id
        assert_eq!(tid_stores[1].args[2].0, 2); // second type_id
    }

    // ── Test 7: A collecting operation between two NEWs prevents batching ──

    #[test]
    fn test_call_breaks_batch() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::new(OpCode::CallN, &[OpRef(99)]),
            Op::with_descr(OpCode::New, &[], size_descr(24, 2)),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // There should be two separate CallMallocNursery ops
        // (the CallN in between flushes the batch).
        let malloc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallMallocNursery)
            .count();
        assert_eq!(malloc_count, 2);
    }

    // ── Test 8: WB not duplicated for same object ──

    #[test]
    fn test_wb_not_duplicated() {
        let rw = make_rewriter();
        let obj = OpRef(0);
        let val1 = OpRef(1);
        let val2 = OpRef(2);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[obj, val1], ref_field_descr()),
            Op::with_descr(OpCode::SetfieldGc, &[obj, val2], ref_field_descr()),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Only one CondCallGcWb, then two SetfieldGc.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 1);
        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::SetfieldGc)
                .count(),
            2
        );
    }

    // ── Test 9: Freshly allocated object skips WB ──

    #[test]
    fn test_fresh_alloc_skips_wb() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(32, 1)),
            // The freshly allocated object (pos 0) is used as the target of a store.
            // We build the SetfieldGc referencing pos=0 from the CallMallocNursery result.
        ];

        let _result = rw.rewrite_for_gc(&ops);

        // Now rewrite a SetfieldGc that stores a ref into the new object.
        let ops2 = vec![
            Op::with_descr(OpCode::New, &[], size_descr(32, 1)),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef(0), OpRef(99)], // arg(0) = pos of the alloc = 0
                ref_field_descr(),
            ),
        ];

        let result2 = rw.rewrite_for_gc(&ops2);

        // The CallMallocNursery result at pos=0 is in wb_applied,
        // so the SetfieldGc at arg(0)=OpRef(0) should NOT get a WB.
        // Expected: CallMallocNursery, GcStore(tid), SetfieldGc
        // No CondCallGcWb because OpRef(0) was remembered.
        let wb_count = result2
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 0);
    }

    // ── Test 10: NEW_WITH_VTABLE also writes vtable ──

    #[test]
    fn test_new_with_vtable() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(
            OpCode::NewWithVtable,
            &[],
            vtable_descr(48, 3, 0xDEAD),
        )];

        let result = rw.rewrite_for_gc(&ops);

        // CallMallocNursery + GcStore(tid) + GcStore(vtable)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        // tid init
        assert_eq!(result[1].opcode, OpCode::GcStore);
        assert_eq!(result[1].args[2].0, 3);
        // vtable init
        assert_eq!(result[2].opcode, OpCode::GcStore);
        assert_eq!(result[2].args[2].0, 0xDEAD_u32 as u32);
    }

    // ── Test 11: SETARRAYITEM_GC with Ref triggers array WB ──

    #[test]
    fn test_setarrayitem_gc_ref_wb() {
        let rw = make_rewriter();
        let obj = OpRef(0);
        let idx = OpRef(1);
        let val = OpRef(2);
        let ops = vec![Op::with_descr(
            OpCode::SetarrayitemGc,
            &[obj, idx, val],
            array_descr_ref(),
        )];

        let result = rw.rewrite_for_gc(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CondCallGcWbArray);
        assert_eq!(result[0].args[0], obj);
        assert_eq!(result[0].args[1], idx);
        assert_eq!(result[1].opcode, OpCode::SetarrayitemGc);
    }

    // ── Test 12: Collecting op clears WB memoisation ──

    #[test]
    fn test_collecting_op_clears_wb() {
        let rw = make_rewriter();
        let obj = OpRef(0);
        let val = OpRef(1);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[obj, val], ref_field_descr()),
            // This call can collect, clearing the WB set.
            Op::new(OpCode::CallN, &[OpRef(99)]),
            Op::with_descr(OpCode::SetfieldGc, &[obj, val], ref_field_descr()),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Two CondCallGcWb — the second one is needed because the CallN cleared the set.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 2);
    }

    #[test]
    fn test_explicit_result_positions_are_preserved_through_rewrite() {
        let rw = make_rewriter();
        let ops = vec![
            mk_op_with_descr(OpCode::New, &[], 2, size_descr(24, 1)),
            mk_op_with_descr(OpCode::New, &[], 3, size_descr(16, 2)),
            mk_op(OpCode::Finish, &[OpRef(3)], 4),
        ];

        let result = rw.rewrite_for_gc(&ops);

        let first_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::CallMallocNursery)
            .unwrap();
        let second_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        let finish = result.last().unwrap();

        assert_eq!(first_alloc.pos, OpRef(2));
        assert_eq!(second_alloc.pos, OpRef(3));
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.args[0], OpRef(3));
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::GcStore)
                .all(|op| op.pos.is_none())
        );
    }

    #[test]
    fn test_rewrite_is_idempotent_for_existing_array_wb_sequence() {
        let rw = make_rewriter();
        let once = vec![
            mk_op(OpCode::CondCallGcWbArray, &[OpRef(5), OpRef(1)], 7),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(5), OpRef(1), OpRef(6)],
                8,
                array_descr_ref(),
            ),
        ];

        let twice = rw.rewrite_for_gc(&once);

        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::CondCallGcWbArray)
                .count(),
            1
        );
        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::SetarrayitemGc)
                .count(),
            1
        );
    }

    // ── Pending zero flush tests ──

    #[test]
    fn test_pending_zero_fully_initialized() {
        // NEW_ARRAY_CLEAR(3) + SET[0] + SET[1] + SET[2] → no ZERO_ARRAY emitted.
        let rw = make_rewriter();
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[OpRef(3)], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(0), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(2), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // All indices were SET, so no ZERO_ARRAY should be emitted.
        let zero_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .count();
        assert_eq!(
            zero_count, 0,
            "fully initialized array should produce no ZERO_ARRAY"
        );
    }

    #[test]
    fn test_pending_zero_partially_initialized() {
        // NEW_ARRAY_CLEAR(4) + SET[0] + SET[1] → ZERO_ARRAY(arr, 2, 2)
        let rw = make_rewriter();
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[OpRef(4)], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(0), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Indices 0 and 1 were SET. Indices 2 and 3 still need zeroing.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "should emit exactly one ZERO_ARRAY");
        assert_eq!(zeros[0].args[1], OpRef(2), "zero start should be 2");
        assert_eq!(zeros[0].args[2], OpRef(2), "zero length should be 2");
    }

    #[test]
    fn test_pending_zero_flushed_at_guard() {
        // Guard forces pending zero flush even if no indices were SET.
        let rw = make_rewriter();
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[OpRef(3)], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::new(OpCode::GuardTrue, &[OpRef(50)]),
            Op::new(OpCode::Finish, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // The ZERO_ARRAY should appear before the guard.
        let zero_idx = result.iter().position(|o| o.opcode == OpCode::ZeroArray);
        let guard_idx = result.iter().position(|o| o.opcode == OpCode::GuardTrue);
        assert!(zero_idx.is_some(), "ZERO_ARRAY should be emitted");
        assert!(guard_idx.is_some(), "GuardTrue should be present");
        assert!(
            zero_idx.unwrap() < guard_idx.unwrap(),
            "ZERO_ARRAY should come before GuardTrue"
        );

        let zero = result
            .iter()
            .find(|o| o.opcode == OpCode::ZeroArray)
            .unwrap();
        assert_eq!(zero.args[1], OpRef(0), "zero start should be 0");
        assert_eq!(zero.args[2], OpRef(3), "zero length should be 3");
    }

    #[test]
    fn test_pending_zero_gap_in_middle() {
        // NEW_ARRAY_CLEAR(5) + SET[0] + SET[2] + SET[4] → two ZERO_ARRAY runs.
        let rw = make_rewriter();
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[OpRef(5)], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(0), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(2), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(4), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let result = rw.rewrite_for_gc(&ops);

        // Indices 1 and 3 need zeroing → two separate ZERO_ARRAY runs.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(
            zeros.len(),
            2,
            "should emit two ZERO_ARRAY runs for gaps at 1 and 3"
        );
        assert_eq!(zeros[0].args[1], OpRef(1));
        assert_eq!(zeros[0].args[2], OpRef(1));
        assert_eq!(zeros[1].args[1], OpRef(3));
        assert_eq!(zeros[1].args[2], OpRef(1));
    }

    #[test]
    fn test_pending_zero_no_clear() {
        // Plain NEW_ARRAY (not CLEAR) should NOT produce any ZERO_ARRAY.
        let rw = make_rewriter();
        let mut new_array = Op::with_descr(OpCode::NewArray, &[OpRef(3)], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_for_gc(&ops);

        let zero_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .count();
        assert_eq!(
            zero_count, 0,
            "plain NEW_ARRAY should not produce ZERO_ARRAY"
        );
    }
}
