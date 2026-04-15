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

/// regalloc.py:871 — compiled_loop_token._ll_initial_locs + frame_info.
///
/// Per-callee metadata needed by handle_call_assembler to allocate and
/// fill the callee jitframe.
#[derive(Clone, Debug)]
pub struct CallAssemblerCalleeLocs {
    /// regalloc.py:869 — byte offsets of each inputarg within jf_frame,
    /// relative to BASEITEMOFS (i.e. index_list[i] = loc.value - base_ofs).
    pub _ll_initial_locs: Vec<i32>,
    /// Total jitframe depth (in slots).
    /// regalloc.py:861 — needed for gen_malloc_frame allocation size.
    pub frame_depth: usize,
    /// rewrite.py:669 — ptr2int(loop_token.compiled_loop_token.frame_info).
    /// Raw address of the callee's JitFrameInfo struct.
    pub frame_info_ptr: usize,
    /// pyjitpl.py:3605 — jd.index_of_virtualizable.
    /// Index into the original arglist of the virtualizable box.
    /// -1 if no virtualizable.
    pub index_of_virtualizable: i32,
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
    /// Type annotations for constant OpRefs. RPython's ConstPtr/ConstInt
    /// boxes carry `.type` intrinsically; here we pass constant types
    /// from the optimizer so the rewriter can check `v.type == 'r'` on
    /// constant values (rewrite.py:930).
    pub constant_types: HashMap<u32, Type>,
    /// rewrite.py:673 — lookup compiled_loop_token._ll_initial_locs
    /// by target token number. Provided by the backend.
    pub call_assembler_callee_locs:
        Option<Box<dyn Fn(u64) -> Option<CallAssemblerCalleeLocs> + Send>>,
}

/// JitFrame field descriptors for handle_call_assembler.
///
/// rewrite.py:666 — `descrs = self.gc_ll_descr.getframedescrs(self.cpu)`
#[derive(Clone)]
pub struct JitFrameDescrs {
    /// Addresses of all `jit_create_*_callee_frame_*` variants.
    pub create_fn_addrs: Vec<usize>,
    /// Address of `jit_drop_callee_frame`.
    pub drop_fn_addr: usize,
    /// GC type id for JitFrame (from gc.register_type).
    pub jitframe_tid: u32,
    /// JitFrame fixed header size (bytes).
    pub jitframe_fixed_size: usize,
    /// Byte offsets of JitFrame fields (from majit_metainterp::jitframe).
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
    // ── PyFrame layout (for nursery allocation transition) ──
    /// Total PyFrame size for nursery allocation.
    pub pyframe_alloc_size: usize,
    /// Byte offset of `code` in PyFrame.
    pub pyframe_code_ofs: usize,
    /// Byte offset of `namespace` in PyFrame.
    pub pyframe_namespace_ofs: usize,
    /// Byte offset of `next_instr` in PyFrame.
    pub pyframe_next_instr_ofs: usize,
    /// Byte offset of `vable_token` in PyFrame.
    pub pyframe_vable_token_ofs: usize,
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
    /// rewrite.py:930 parity — structural equivalent of Box.type.
    /// Maps OpRef.0 → Type for all known values (both op results and
    /// constants). In RPython each Box carries its own `.type` attribute;
    /// here we mirror that with an explicit lookup table because OpRef
    /// is a plain u32 without embedded type information.
    result_types: HashMap<u32, Type>,

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
    /// rewrite.py:41-45: _write_barrier_applied — set of OpRef indices
    /// whose write barrier has already been emitted (freshly allocated
    /// objects, or objects we already issued a WB for). Cleared whenever
    /// we emit an operation that can trigger a collection or on LABEL.
    wb_applied: HashSet<u32>,
    /// Forwarding map from original result OpRefs to rewritten result OpRefs.
    forwarding: HashMap<u32, OpRef>,
    /// One-shot marker for an already-emitted array write barrier.
    ///
    /// Re-running the rewriter over an already-rewritten trace should not
    /// duplicate a `COND_CALL_GC_WB_ARRAY` immediately followed by the
    /// corresponding `SETARRAYITEM_GC`.
    pending_array_wb: Option<u32>,

    // ── Array length tracking (rewrite.py:59 _known_lengths) ──
    /// Maps array OpRef → known length. Populated when NEW_ARRAY has a
    /// constant length operand (rewrite.py:551). Cleared on LABEL
    /// (rewrite.py:1005) and emitting_an_operation_that_can_collect.
    known_lengths: HashMap<u32, usize>,

    // ── Pending zero tracking ──
    /// Deferred ZERO_ARRAY ops that may be optimized away if subsequent
    /// SETARRAYITEM writes cover the entire range.
    pending_zeros: Vec<PendingZero>,
    /// Tracks which array indices have been explicitly SET since the
    /// pending zero was recorded. Keyed by array OpRef index.
    initialized_indices: HashMap<u32, HashSet<usize>>,

    // ── Pending vtable initializations ──
    /// rewrite.py:479-484 handle_malloc_operation parity: for a batch of
    /// NEW_WITH_VTABLE ops being coalesced into one nursery bump, defer
    /// the vtable GcStore until after ALL allocations in the batch have
    /// been emitted. This matches RPython's "all allocations first, then
    /// all initializations" pattern and keeps gen_initialize_tid stores
    /// contiguous for store-buffer friendliness.
    pending_vtable_inits: Vec<(OpRef, usize)>,
}

impl RewriteState {
    fn new(hint: usize, next_pos: u32) -> Self {
        RewriteState {
            out: Vec::with_capacity(hint + hint / 4),
            next_pos,
            constants: HashMap::new(),
            result_types: HashMap::new(),
            pending_malloc_idx: None,
            pending_malloc_total: 0,
            previous_size: 0,
            last_malloced_ref: OpRef::NONE,
            wb_applied: HashSet::new(),
            forwarding: HashMap::new(),
            pending_array_wb: None,
            known_lengths: HashMap::new(),
            pending_zeros: Vec::new(),
            initialized_indices: HashMap::new(),
            pending_vtable_inits: Vec::new(),
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

    /// Create or reuse a constant OpRef in the constants pool.
    /// Returns an OpRef key that resolves to `value` at compile time.
    fn const_int(&mut self, value: i64) -> OpRef {
        // Check if this constant already exists
        for (&key, &val) in &self.constants {
            if val == value {
                return OpRef(key);
            }
        }
        // Allocate a new constant key (use high range to avoid collision)
        let key = 10_000 + self.constants.len() as u32;
        // Skip keys already in use
        let mut key = key;
        while self.constants.contains_key(&key) {
            key += 1;
        }
        self.constants.insert(key, value);
        OpRef(key)
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

    /// rewrite.py:699-711 emitting_an_operation_that_can_collect
    fn emitting_an_operation_that_can_collect(&mut self) {
        self.pending_malloc_idx = None;
        self.wb_applied.clear();
        self.pending_array_wb = None;
        self.emit_pending_zeros();
    }

    fn remember_wb(&mut self, r: OpRef) {
        self.wb_applied.insert(r.0);
    }

    /// rewrite.py:66-67: remember_known_length
    fn remember_known_length(&mut self, op: OpRef, length: usize) {
        self.known_lengths.insert(op.0, length);
    }

    /// rewrite.py:81-82: known_length(op, default)
    fn known_length(&self, op: OpRef, default: usize) -> usize {
        self.known_lengths.get(&op.0).copied().unwrap_or(default)
    }

    /// rewrite.py:714: write_barrier_applied(op)
    fn wb_already_applied(&self, r: OpRef) -> bool {
        self.wb_applied.contains(&r.0)
    }

    /// rewrite.py:930 parity: `v.type` — get the type of an OpRef.
    fn result_type_of(&self, r: OpRef) -> Option<Type> {
        self.result_types.get(&r.0).copied()
    }

    /// rewrite.py:930 parity: `isinstance(v, ConstPtr) and not needs_write_barrier(v.value)`.
    /// A null ConstPtr never needs a write barrier.
    fn is_null_constant(&self, r: OpRef) -> bool {
        if let Some(&val) = self.constants.get(&r.0) {
            val == 0
        } else {
            false
        }
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
    fn emit_pending_zeros(&mut self) {
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

        // rewrite.py:474-484 handle_malloc_operation parity:
        // descr.size in RPython already includes the GC header (the
        // OBJECT type is built with `size = sizeof(header) + sizeof(fields)`).
        // pyre's PyreSizeDescr reports `obj_size` as the bare struct size
        // (e.g. `size_of::<W_IntObject>() == 16`) WITHOUT the GC header,
        // so we add it here so that CallMallocNursery sees the same
        // "object-with-header" total that the cranelift backend expects
        // (it strips the header back off before passing to the alloc
        // shim's payload size).
        let size = round_up(descr.size() + crate::header::GcHeader::SIZE);
        let type_id = descr.type_id();

        let obj_ref = self.gen_malloc_nursery(size, op.pos, st);
        st.record_result_mapping(op.pos, obj_ref);

        // Initialize the tid header field.
        self.gen_initialize_tid(obj_ref, type_id, st);

        // rewrite.py:479-484 handle_malloc_operation parity:
        //   elif opnum == rop.NEW_WITH_VTABLE:
        //       ...
        //       if self.gc_ll_descr.fielddescr_vtable is not None:
        //           self.emit_setfield(op, ConstInt(descr.get_vtable()),
        //                              descr=self.gc_ll_descr.fielddescr_vtable)
        //
        // The vtable setfield is defered to pending_vtable_inits so that a
        // batch of consecutive NEW_WITH_VTABLE ops produces contiguous
        // allocation+tid-init sequences followed by the vtable stores,
        // matching RPython's "alloc1 tid1 alloc2 tid2 ... setfields..."
        // output pattern (rewrite.py processes ops linearly, and the
        // vtable emit_setfield happens at the end of handle_malloc_operation
        // but batched allocations interleave only in the nursery merge).
        if op.opcode == OpCode::NewWithVtable {
            let vtable = descr.vtable();
            if vtable != 0 {
                st.pending_vtable_inits.push((obj_ref, vtable));
            }
        }
    }

    /// Flush pending vtable initializations for a batched nursery allocation.
    /// Emits a GcStore for each (obj, vtable) pair accumulated during the
    /// current batch of consecutive NEW_WITH_VTABLE ops. Called when the
    /// batch terminates (next non-allocation op, or end of trace).
    fn flush_pending_vtable_inits(&self, st: &mut RewriteState) {
        if st.pending_vtable_inits.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut st.pending_vtable_inits);
        for (obj, vtable) in pending {
            self.gen_initialize_vtable(obj, vtable, st);
        }
    }

    // ────────────────────────────────────────────────────────
    // NEW_ARRAY / NEW_ARRAY_CLEAR  → CALL_MALLOC_NURSERY_VARSIZE
    // ────────────────────────────────────────────────────────

    /// rewrite.py:848 gen_malloc_nursery_varsize parity.
    /// kind: FLAG_ARRAY=0, FLAG_STR=1, FLAG_UNICODE=2.
    fn handle_new_array(&self, op: &Op, st: &mut RewriteState, kind: i64) {
        let is_clear = op.opcode == OpCode::NewArrayClear;

        let descr_ref = op
            .descr
            .as_ref()
            .expect("NEW_ARRAY must have an ArrayDescr");
        let descr = descr_ref
            .as_array_descr()
            .expect("NEW_ARRAY descr must be ArrayDescr");

        let item_size = descr.item_size();
        let v_length = st.resolve(op.arg(0)); // the length operand

        // rewrite.py:857-860: CALL_MALLOC_NURSERY_VARSIZE([ConstInt(kind), ConstInt(itemsize), v_length])
        st.emitting_an_operation_that_can_collect();

        let kind_ref = st.const_int(kind);
        let itemsize_ref = st.const_int(item_size as i64);
        let mut varsize_op = Op::new(
            OpCode::CallMallocNurseryVarsize,
            &[kind_ref, itemsize_ref, v_length],
        );
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

        // rewrite.py:551: if isinstance(v_length, ConstInt):
        //     self.remember_known_length(op, v_length.getint())
        if let Some(num_elem) = st.resolve_constant(v_length.0) {
            st.remember_known_length(result, num_elem as usize);
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

        // Flush only THIS obj's pending vtable init before its first
        // user setfield. This matches RPython's linear trace processing
        // where NEW_WITH_VTABLE's emit_setfield runs before any following
        // user setfield on the same object. Other objects' vtable inits
        // stay pending and are flushed either on their own first setfield
        // or at the end of the trace.
        if !st.pending_vtable_inits.is_empty() {
            if let Some(pos) = st.pending_vtable_inits.iter().position(|(o, _)| *o == obj) {
                let (o, vtable) = st.pending_vtable_inits.remove(pos);
                self.gen_initialize_vtable(o, vtable, st);
            }
        }

        // rewrite.py:930-931: check the stored VALUE's type.
        //   v = op.getarg(1)
        //   if (v.type == 'r' and (not isinstance(v, ConstPtr) or
        //       rgc.needs_write_barrier(v.value))):
        //
        // Gate on field descriptor: if the field is not a pointer field,
        // the GC won't trace it, so no WB is needed regardless of value
        // type. In RPython val.type=='r' implies the field is GCREF;
        // here ForceToken (Ref) stores to an Int-typed field (offset 128),
        // a pyre-specific divergence.
        let field_is_ptr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.is_pointer_field())
            .unwrap_or(false);
        let val = rewritten.arg(1);
        let val_is_ref = if field_is_ptr {
            match st.result_type_of(val) {
                Some(tp) => tp == Type::Ref,
                None => true, // field is ptr → assume value is Ref
            }
        } else {
            false
        };
        let is_null_const = val_is_ref && st.is_null_constant(val);

        if val_is_ref && !is_null_const && !st.wb_already_applied(obj) {
            // Emit COND_CALL_GC_WB(obj) before the store.
            let wb_op = Op::new(OpCode::CondCallGcWb, &[obj]);
            st.emit(wb_op);
            st.remember_wb(obj);
        }

        // Emit the original SETFIELD_GC unchanged.
        st.emit(rewritten);
    }

    // ────────────────────────────────────────────────────────
    // SETARRAYITEM_GC  → maybe COND_CALL_GC_WB{_ARRAY} + SETARRAYITEM_GC
    // rewrite.py:936-946 handle_write_barrier_setarrayitem
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

        if st.take_pending_array_wb(obj) {
            st.emit(rewritten);
            return;
        }

        // rewrite.py:938-941: check the stored VALUE's type (arg2).
        //   val = op.getarg(0)  [the base object]
        //   if not self.write_barrier_applied(val):
        //       v = op.getarg(2)
        //       if (v.type == 'r' and (not isinstance(v, ConstPtr) or
        //           rgc.needs_write_barrier(v.value))):
        //           self.gen_write_barrier_array(val, op.getarg(1))
        if !st.wb_already_applied(obj) {
            let v = rewritten.arg(2);
            let val_is_ref = match st.result_type_of(v) {
                Some(tp) => tp == Type::Ref,
                None => op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map(|ad| ad.is_array_of_pointers())
                    .unwrap_or(false),
            };
            let is_null_const = val_is_ref && st.is_null_constant(v);

            if val_is_ref && !is_null_const {
                self.gen_write_barrier_array(obj, rewritten.arg(1), st);
            }
        }

        st.emit(rewritten);
    }

    // ────────────────────────────────────────────────────────
    // rewrite.py:955-973 gen_write_barrier_array
    // ────────────────────────────────────────────────────────

    fn gen_write_barrier_array(&self, v_base: OpRef, v_index: OpRef, st: &mut RewriteState) {
        if self.wb_descr.jit_wb_cards_set != 0 {
            // If we know statically the length of 'v_base', and it is not
            // too big, then produce a regular write_barrier. If it's
            // unknown or too big, produce a write_barrier_from_array.
            const LARGE: usize = 130;
            let length = st.known_length(v_base, LARGE);
            if length >= LARGE {
                // Unknown or too big: produce COND_CALL_GC_WB_ARRAY.
                let wb_op = Op::new(OpCode::CondCallGcWbArray, &[v_base, v_index]);
                st.emit(wb_op);
                // rewrite.py:970: a WB_ARRAY is not enough to prevent
                // any future write barriers, so don't remember_wb!
                return;
            }
        }
        // Fall-back: produce a regular write_barrier.
        let wb_op = Op::new(OpCode::CondCallGcWb, &[v_base]);
        st.emit(wb_op);
        st.remember_wb(v_base);
    }

    // ────────────────────────────────────────────────────────
    // gen_malloc_nursery: batched bump-pointer allocation
    // ────────────────────────────────────────────────────────

    /// Try to emit (or extend) a CALL_MALLOC_NURSERY for `size` bytes.
    /// Returns the OpRef of the newly allocated object.
    fn gen_malloc_nursery(&self, size: usize, result_pos: OpRef, st: &mut RewriteState) -> OpRef {
        let size = round_up(size);

        if !self.can_use_nursery(size) {
            st.emitting_an_operation_that_can_collect();
            let size_ref = st.const_int(size as i64);
            let op = Op::new(OpCode::CallMallocNursery, &[size_ref]);
            let r = st.emit_result(op, result_pos);
            st.remember_wb(r);
            return r;
        }

        // rewrite.py:893-898 merge with previous CALL_MALLOC_NURSERY
        if let Some(prev_idx) = st.pending_malloc_idx {
            let new_total = st.pending_malloc_total + size;
            if self.can_use_nursery(new_total) {
                let new_total_ref = st.const_int(new_total as i64);
                st.out[prev_idx].args[0] = new_total_ref;
                st.pending_malloc_total = new_total;

                // rewrite.py:896: NURSERY_PTR_INCREMENT(last, ConstInt(previous_size))
                let prev_size_ref = st.const_int(st.previous_size as i64);
                let incr_op = Op::new(
                    OpCode::NurseryPtrIncrement,
                    &[st.last_malloced_ref, prev_size_ref],
                );
                let r = st.emit_result(incr_op, result_pos);
                st.previous_size = size;
                st.last_malloced_ref = r;
                st.remember_wb(r);
                return r;
            }
        }

        // rewrite.py:903: CALL_MALLOC_NURSERY(ConstInt(size))
        st.emitting_an_operation_that_can_collect();
        let size_ref = st.const_int(size as i64);
        let op = Op::new(OpCode::CallMallocNursery, &[size_ref]);
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

    /// rewrite.py:914-918 gen_initialize_tid parity.
    ///
    /// RPython: emit_setfield(obj, ConstInt(tid), descr=fielddescr_tid)
    ///   → emit_gc_store_or_indexed(ptr, ConstInt(0), ConstInt(tid), size=WORD, factor=1, ofs=tid_ofs)
    ///   → GC_STORE(ptr, ConstInt(tid_ofs), ConstInt(tid), ConstInt(WORD))
    ///
    /// The tid field is in the GcHeader which sits before the object pointer.
    /// offset = -(GcHeader::SIZE) from the object pointer.
    fn gen_initialize_tid(&self, obj: OpRef, tid: u32, st: &mut RewriteState) {
        let ofs = st.const_int(-(crate::header::GcHeader::SIZE as i64));
        let tid_val = st.const_int(tid as i64);
        let size = st.const_int(std::mem::size_of::<usize>() as i64);
        let store = Op::new(OpCode::GcStore, &[obj, ofs, tid_val, size]);
        st.emit(store);
    }

    /// rewrite.py:479-484 gen_initialize_vtable parity.
    ///
    /// RPython: emit_setfield(obj, ConstInt(vtable), descr=fielddescr_vtable)
    /// The vtable pointer is at offset 0 from the object pointer.
    fn gen_initialize_vtable(&self, obj: OpRef, vtable: usize, st: &mut RewriteState) {
        let ofs = st.const_int(0);
        let vtable_ref = st.const_int(vtable as i64);
        let size = st.const_int(std::mem::size_of::<usize>() as i64);
        let store = Op::new(OpCode::GcStore, &[obj, ofs, vtable_ref, size]);
        st.emit(store);
    }

    /// rewrite.py:550-554 gen_initialize_len parity.
    ///
    /// RPython: emit_setfield(obj, length, descr=lendescr)
    /// The length field offset comes from the array descriptor.
    fn gen_initialize_len(
        &self,
        obj: OpRef,
        length: OpRef,
        array_descr: DescrRef,
        len_descr: &dyn FieldDescr,
        st: &mut RewriteState,
    ) {
        let ofs = st.const_int(len_descr.offset() as i64);
        let size = st.const_int(len_descr.field_size() as i64);
        let mut store = Op::new(OpCode::GcStore, &[obj, ofs, length, size]);
        store.descr = Some(array_descr);
        st.emit(store);
    }

    // ── rewrite.py:665-695 handle_call_assembler helpers ──────────

    /// Check if a CallR op is a call to any create_frame helper variant.
    fn is_create_frame_call(&self, op: &Op, st: &RewriteState) -> bool {
        let Some(ref descrs) = self.jitframe_info else {
            return false;
        };
        let func_addr_key = op.arg(0).0;
        let Some(func_addr) = st.resolve_constant(func_addr_key) else {
            return false;
        };
        descrs
            .create_fn_addrs
            .iter()
            .any(|&addr| func_addr == addr as i64)
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
    /// rewrite.py:665-695 parity: replace CallR(create_frame) with
    /// nursery bump allocation + field initialization.
    ///
    /// Allocates a PyFrame-layout block from the nursery (matching
    /// the field offsets that compiled code reads via raw pointers).
    /// Copies `code` and `namespace` from the caller frame, zeroes
    /// `next_instr` and `vable_token`.
    /// rewrite.py:613-695 gen_malloc_frame + handle_call_assembler parity.
    ///
    /// Allocates a PyFrame-sized block from the nursery, initializes the
    /// GC type ID header, zeroes all GC-traced pointer fields, and copies
    /// caller's `code` and `namespace`.
    fn handle_create_frame(&self, op: &Op, st: &mut RewriteState) {
        let descrs = self.jitframe_info.as_ref().unwrap();

        // rewrite.py:634 — gen_malloc_nursery_varsize_frame
        let alloc_size = (descrs.pyframe_alloc_size + 7) & !7;
        st.emitting_an_operation_that_can_collect();
        let malloc_op = Op::new(
            OpCode::CallMallocNurseryVarsizeFrame,
            &[OpRef(alloc_size as u32)],
        );
        let frame = st.emit_result(malloc_op, op.pos);
        st.remember_wb(frame);

        // rewrite.py:635 — gen_initialize_tid(frame, arraydescr.tid)
        // NOTE: pyre's PyFrame is not yet GC-managed, so the GcHeader
        // type-id write would touch memory before the PyFrame payload
        // that is not guaranteed to be accessible from the Cranelift
        // lowering path (offset 0 in gen_initialize_tid maps to
        // header-relative, but arena slots already have a zeroed
        // gc_header prepended). This call is deferred until the backend
        // GcStore slot-0 lowering is verified to use the correct
        // negative offset from the payload pointer.
        // self.gen_initialize_tid(frame, descrs.jitframe_tid, st);

        // rewrite.py:641-650 — zero all GCREF fields.
        // For JitFrame: jf_savedata, jf_force_descr, jf_descr, jf_guard_exc, jf_forward.
        // For PyFrame: next_instr, vable_token (plus execution_context, locals_cells_stack_w
        // which are written by caller before use).
        let zero = st.const_int(0);
        let word_size = st.const_int(8);
        let ni_ofs = st.const_int(descrs.pyframe_next_instr_ofs as i64);
        let vt_ofs = st.const_int(descrs.pyframe_vable_token_ofs as i64);
        st.emit(Op::new(OpCode::GcStore, &[frame, ni_ofs, zero, word_size]));
        st.emit(Op::new(OpCode::GcStore, &[frame, vt_ofs, zero, word_size]));

        // rewrite.py:655-658 — Copy `code` from caller frame (self-recursive: same code).
        let caller_frame = st.resolve(op.arg(1));
        let code_ofs = st.const_int(descrs.pyframe_code_ofs as i64);
        let ns_ofs = st.const_int(descrs.pyframe_namespace_ofs as i64);

        let caller_code = st.emit(Op::new(
            OpCode::GcLoadI,
            &[caller_frame, code_ofs, word_size],
        ));
        st.emit(Op::new(
            OpCode::GcStore,
            &[frame, code_ofs, caller_code, word_size],
        ));

        // Copy `namespace` from caller frame.
        let caller_ns = st.emit(Op::new(OpCode::GcLoadI, &[caller_frame, ns_ofs, word_size]));
        st.emit(Op::new(
            OpCode::GcStore,
            &[frame, ns_ofs, caller_ns, word_size],
        ));

        st.record_result_mapping(op.pos, frame);
    }

    /// rewrite.py:665-695 handle_call_assembler:
    ///   1. gen_malloc_frame — allocate callee jitframe from nursery
    ///   2. gen_initialize_tid + zero GC fields
    ///   3. store each arg at _ll_initial_locs[i] offset
    ///   4. replace multi-arg CALL_ASSEMBLER with single-arg [frame]
    fn handle_call_assembler(&self, op: &Op, st: &mut RewriteState) {
        let descrs = self.jitframe_info.as_ref().unwrap();
        let lookup = self.call_assembler_callee_locs.as_ref().unwrap();

        // rewrite.py:667-668 — loop_token = op.getdescr(); JitCellToken
        let call_descr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .expect("CallAssembler op must have a CallDescr");
        let token = call_descr
            .call_target_token()
            .expect("CallAssembler descr must have a target token");

        // rewrite.py:673 — index_list = loop_token.compiled_loop_token._ll_initial_locs
        // RPython: compiled_loop_token is pre-allocated with the token;
        // frame_info pointer is stable. Self-recursive calls go through
        // register_pending_call_assembler_target() BEFORE tracing emits
        // any CALL_ASSEMBLER op referencing this token.
        let callee_locs = lookup(token)
            .expect("pending CALL_ASSEMBLER target must be registered before rewriter runs");

        // rewrite.py:627-653 — gen_malloc_frame(llfi)
        // RPython reads jfi_frame_size from frame_info AT RUNTIME so
        // the allocation size is correct even for self-recursive calls
        // (where frame_info is pre-allocated with [0,0] and updated
        // after compilation).
        let llfi = st.const_int(callee_locs.frame_info_ptr as i64);
        let word_size = st.const_int(8);

        // rewrite.py:628-632 — GC_LOAD_I(frame_info, jfi_frame_size_ofs, 8)
        let jfi_frame_size_ofs = st.const_int(std::mem::size_of::<isize>() as i64);
        let size = st.emit(Op::new(
            OpCode::GcLoadI,
            &[llfi, jfi_frame_size_ofs, word_size],
        ));
        // rewrite.py:634 — gen_malloc_nursery_varsize_frame(size)
        st.emitting_an_operation_that_can_collect();
        let malloc_op = Op::new(OpCode::CallMallocNurseryVarsizeFrame, &[size]);
        let frame = st.emit_result(malloc_op, OpRef::NONE);
        st.remember_write_barrier(frame);

        // rewrite.py:635 — gen_initialize_tid(frame, descrs.arraydescr.tid)
        self.gen_initialize_tid(frame, descrs.jitframe_tid, st);

        // rewrite.py:636-650 — zero GC fields
        let zero = st.const_int(0);
        for &ofs in &[
            descrs.jf_descr_ofs,
            descrs.jf_force_descr_ofs,
            descrs.jf_savedata_ofs,
            descrs.jf_guard_exc_ofs,
            descrs.jf_forward_ofs,
        ] {
            let ofs_ref = st.const_int(ofs as i64);
            st.emit(Op::new(OpCode::GcStore, &[frame, ofs_ref, zero, word_size]));
        }

        // rewrite.py:639-640,651-652 — load depth, set length
        let jfi_frame_depth_ofs = st.const_int(0);
        let length = st.emit(Op::new(
            OpCode::GcLoadI,
            &[llfi, jfi_frame_depth_ofs, word_size],
        ));
        let len_ofs = st.const_int(descrs.jf_frame_lengthofs as i64);
        st.emit(Op::new(
            OpCode::GcStore,
            &[frame, len_ofs, length, word_size],
        ));

        // rewrite.py:671 — emit_setfield(frame, ConstInt(llfi), descr=descrs.jf_frame_info)
        let fi_ofs = st.const_int(descrs.jf_frame_info_ofs as i64);
        st.emit(Op::new(OpCode::GcStore, &[frame, fi_ofs, llfi, word_size]));

        // rewrite.py:672-683 — store each arg at _ll_initial_locs[i]
        let arglist: Vec<OpRef> = op.args.iter().map(|&a| st.resolve(a)).collect();
        let index_list = &callee_locs._ll_initial_locs;
        for (i, &arg) in arglist.iter().enumerate() {
            // rewrite.py:678 — array_offset = index_list[i]
            // rewrite.py:681 — offset = basesize + array_offset
            let offset = descrs.jf_frame_baseitemofs as i32 + index_list[i];
            let ofs_ref = st.const_int(offset as i64);
            st.emit(Op::new(OpCode::GcStore, &[frame, ofs_ref, arg, word_size]));
        }

        // rewrite.py:685-695 — replace multi-arg with [frame] or
        // [frame, arglist[index_of_virtualizable]]
        let new_args = if callee_locs.index_of_virtualizable >= 0 {
            let vable_idx = callee_locs.index_of_virtualizable as usize;
            vec![frame, arglist[vable_idx]]
        } else {
            vec![frame]
        };
        let mut call_asm = Op::new(op.opcode, &new_args);
        call_asm.descr = op.descr.clone();
        call_asm.fail_args = op
            .fail_args
            .as_ref()
            .map(|fa| fa.iter().map(|&a| st.resolve(a)).collect());
        st.emit_rewritten_from(op, call_asm);
    }
}

impl GcRewriterImpl {
    /// rewrite.py:988-1001 remove_bridge_exception: check a common
    /// case where SaveExcClass + SaveException + RestoreException
    /// appear at the start of a bridge and are unused. Strip them.
    fn remove_bridge_exception(ops: &[Op]) -> Vec<Op> {
        let mut start = 0;
        if ops
            .first()
            .map_or(false, |op| op.opcode == OpCode::IncrementDebugCounter)
        {
            start = 1;
        }
        if ops.len() >= start + 3
            && ops[start].opcode == OpCode::SaveExcClass
            && ops[start + 1].opcode == OpCode::SaveException
            && ops[start + 2].opcode == OpCode::RestoreException
        {
            let mut result = Vec::with_capacity(ops.len() - 3);
            result.extend_from_slice(&ops[..start]);
            result.extend_from_slice(&ops[start + 3..]);
            return result;
        }
        ops.to_vec()
    }
}

impl GcRewriter for GcRewriterImpl {
    fn rewrite_for_gc(&self, ops: &[Op]) -> Vec<Op> {
        self.rewrite_for_gc_with_constants(ops, &HashMap::new()).0
    }

    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[Op],
        constants: &HashMap<u32, i64>,
    ) -> (Vec<Op>, HashMap<u32, i64>) {
        // rewrite.py:988-1001 remove_bridge_exception: strip a
        // SaveExcClass+SaveException+RestoreException prefix that is
        // a no-op (common in bridges).
        let ops = Self::remove_bridge_exception(ops);

        let next_pos = ops
            .iter()
            .filter_map(|op| (!op.pos.is_none()).then_some(op.pos.0))
            .max()
            .map_or(0, |max_pos| max_pos.saturating_add(1));
        let mut st = RewriteState::with_constants(ops.len(), next_pos, constants.clone());
        // Build result_types map from input ops — structural equivalent
        // of RPython's Box.type attribute (rewrite.py:930 `v.type`).
        for op in &ops {
            let rt = op.result_type();
            if rt != Type::Void && !op.pos.is_none() {
                st.result_types.insert(op.pos.0, rt);
            }
        }
        // Merge constant types — RPython's ConstPtr/ConstInt carry type
        // intrinsically; here we inject them into the same map.
        for (&k, &tp) in &self.constant_types {
            st.result_types.entry(k).or_insert(tp);
        }
        for op in &ops {
            if !matches!(
                op.opcode,
                OpCode::SetarrayitemGc | OpCode::CondCallGcWbArray | OpCode::DebugMergePoint
            ) {
                st.pending_array_wb = None;
            }

            // Flush pending vtable inits before final ops (Jump/Finish)
            // and before any potentially-collecting / escaping op. User
            // SetfieldGc stores are allowed to precede the vtable init
            // because the object's ob_type slot is only observed by GC
            // tracing (which needs tid, not ob_type) and by guard failures
            // (which reconstruct via known_class). This matches RPython's
            // ordering where trace op processing runs sequentially.
            let is_allocation_or_setfield = matches!(
                op.opcode,
                OpCode::New | OpCode::NewWithVtable | OpCode::SetfieldGc | OpCode::DebugMergePoint
            );
            if !is_allocation_or_setfield && !st.pending_vtable_inits.is_empty() {
                self.flush_pending_vtable_inits(&mut st);
            }

            match op.opcode {
                // Skip debug merge points (they carry no semantics).
                OpCode::DebugMergePoint => continue,

                // rewrite.py:1003-1006 emit_label
                OpCode::Label => {
                    st.emitting_an_operation_that_can_collect();
                    st.known_lengths.clear();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Allocation ──
                OpCode::New | OpCode::NewWithVtable => {
                    self.handle_new(op, &mut st);
                }
                // rewrite.py:485-494
                OpCode::NewArray | OpCode::NewArrayClear => {
                    self.handle_new_array(op, &mut st, 0); // FLAG_ARRAY
                }
                OpCode::Newstr => {
                    self.handle_new_array(op, &mut st, 1); // FLAG_STR
                }
                OpCode::Newunicode => {
                    self.handle_new_array(op, &mut st, 2); // FLAG_UNICODE
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
                    st.emit_pending_zeros();
                    st.emitting_an_operation_that_can_collect();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── rewrite.py:665 handle_call_assembler parity ──
                // Replace CallR(create_frame) with nursery bump alloc.
                // Elide CallN(drop_frame) — GC collects nursery objects.
                // Force path (jit_force_callee_frame) reads code/namespace
                // via raw offsets, creates a proper PyFrame for interpreter.
                OpCode::CallR if self.is_create_frame_call(op, &st) => {
                    self.handle_create_frame(op, &mut st);
                }
                OpCode::CallN if self.is_drop_frame_call(op, &st) => {
                    // rewrite.py:665-695: no explicit free — GC collects.
                }

                // ── Operations that can trigger GC ──
                _ if op.opcode.can_malloc() => {
                    st.emit_pending_zeros();
                    st.emitting_an_operation_that_can_collect();
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── Guards flush pending zeros but do not invalidate WB tracking. ──
                _ if op.opcode.is_guard() => {
                    st.emit_pending_zeros();
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
                    st.emit_pending_zeros();
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

        // Flush any remaining pending zeros and vtable inits at end of trace.
        self.flush_pending_vtable_inits(&mut st);
        st.emit_pending_zeros();

        (st.out, st.constants)
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
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0,
            },
            jitframe_info: None,
            constant_types: HashMap::new(),
            call_assembler_callee_locs: None,
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

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // Expect: CallMallocNursery, GcStore (tid)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        // rewrite.py:474-484 parity: size arg is ConstInt(descr.size + GcHeader::SIZE).
        let size_val = constants[&result[0].args[0].0];
        assert_eq!(size_val, (32 + crate::header::GcHeader::SIZE) as i64);
        assert_eq!(result[1].opcode, OpCode::GcStore); // tid init
        let tid_val = constants[&result[1].args[2].0];
        assert_eq!(tid_val, 7); // type_id = 7
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
        // rewrite.py:858: [ConstInt(kind), ConstInt(itemsize), v_length]
        assert_eq!(varsize.args[2], length_ref);
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

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

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
        // rewrite.py:893-895: combined size = round_up(24+8) + round_up(32+8) = 32 + 40 = 72
        let header = crate::header::GcHeader::SIZE as usize;
        let expected_size = round_up(24 + header) as i64 + round_up(32 + header) as i64;
        assert_eq!(constants[&malloc.args[0].0], expected_size);

        let incr = result
            .iter()
            .find(|o| o.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        // rewrite.py:898: ConstInt(previous_size) = round_up(24 + GcHeader::SIZE) = 32
        assert_eq!(constants[&incr.args[1].0], round_up(24 + header) as i64);

        // Both should have tid initialisation.
        let tid_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(tid_stores.len(), 2);
        assert_eq!(constants[&tid_stores[0].args[2].0], 1); // first type_id
        assert_eq!(constants[&tid_stores[1].args[2].0], 2); // second type_id
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

        let (result, constants) = rw.rewrite_for_gc_with_constants(&ops, &HashMap::new());

        // CallMallocNursery + GcStore(tid) + GcStore(vtable)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        assert_eq!(result[1].opcode, OpCode::GcStore);
        let tid_ref = result[1].args[2];
        assert_eq!(constants[&tid_ref.0], 3);
        assert_eq!(result[2].opcode, OpCode::GcStore);
        let vtable_ref = result[2].args[2];
        assert_eq!(constants[&vtable_ref.0], 0xDEAD_i64);
    }

    // ── Test 11: SETARRAYITEM_GC with Ref — no card marking → regular WB ──

    #[test]
    fn test_setarrayitem_gc_ref_wb() {
        // make_rewriter() has jit_wb_cards_set = 0 (card marking disabled).
        // rewrite.py:955-973: without card marking, gen_write_barrier_array
        // falls back to gen_write_barrier → COND_CALL_GC_WB.
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
        assert_eq!(result[0].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[0].args[0], obj);
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

    // ── Test: gen_write_barrier_array LARGE threshold ──

    fn make_rewriter_with_cards() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0x40,
                jit_wb_card_page_shift: 7,
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0x40,
            },
            jitframe_info: None,
            constant_types: HashMap::new(),
            call_assembler_callee_locs: None,
        }
    }

    #[test]
    fn test_setarrayitem_gc_small_array_uses_regular_wb() {
        // rewrite.py:961-964: known length < LARGE → regular WB.
        let rw = make_rewriter_with_cards();
        // NEW_ARRAY with const length 10 (< 130).
        // Use a constant-pool OpRef for the length.
        let len_ref = OpRef(10_000); // constant key
        let mut constants = HashMap::new();
        constants.insert(10_000, 10_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_ref());
        new_array.pos = OpRef(0);
        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let (result, _) = rw.rewrite_for_gc_with_constants(&ops, &constants);
        // Small array → regular COND_CALL_GC_WB, not WB_ARRAY.
        let wb = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(wb, 1, "small array should get regular WB");
        assert_eq!(wb_arr, 0, "small array should NOT get WB_ARRAY");
    }

    #[test]
    fn test_setarrayitem_gc_large_array_uses_wb_array() {
        // rewrite.py:965-970: known length >= LARGE → WB_ARRAY.
        let rw = make_rewriter_with_cards();
        // NEW_ARRAY with const length 200 (>= 130).
        let len_ref = OpRef(10_000);
        let mut constants = HashMap::new();
        constants.insert(10_000, 200_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_ref());
        new_array.pos = OpRef(0);
        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let (result, _) = rw.rewrite_for_gc_with_constants(&ops, &constants);
        let wb = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(wb, 0, "large array should NOT get regular WB");
        assert_eq!(wb_arr, 1, "large array should get WB_ARRAY");
    }

    #[test]
    fn test_setarrayitem_gc_unknown_length_uses_wb_array() {
        // rewrite.py:962: unknown length defaults to LARGE → WB_ARRAY.
        let rw = make_rewriter_with_cards();
        // No NEW_ARRAY, so v_base has unknown length.
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_for_gc(&ops);
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(wb_arr, 1, "unknown length should get WB_ARRAY");
    }

    #[test]
    fn test_setarrayitem_gc_int_value_no_wb() {
        // rewrite.py:940: v.type != 'r' → no write barrier at all.
        let rw = make_rewriter_with_cards();
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_for_gc(&ops);
        let any_wb = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb || o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(any_wb, 0, "int store should not produce any WB");
    }
}
