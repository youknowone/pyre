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
    /// Type annotations for constant OpRefs. RPython's ConstPtr/ConstInt
    /// boxes carry `.type` intrinsically; here we pass constant types
    /// from the optimizer so the rewriter can check `v.type == 'r'` on
    /// constant values (rewrite.py:930).
    pub constant_types: HashMap<u32, Type>,
}

/// JitFrame field descriptors for handle_call_assembler.
///
/// rewrite.py:666 — `descrs = self.gc_ll_descr.getframedescrs(self.cpu)`
#[derive(Clone)]
pub struct JitFrameDescrs {
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
}

/// A deferred ZERO_ARRAY emission that can be elided or shortened when
/// subsequent SETARRAYITEM writes cover parts of the array.
/// rewrite.py:70-73 setarrayitems_occurred / remember_setarrayitem_occurrences
/// Tracks which array item indices have been set since the last ZERO_ARRAY.
/// Keyed by array OpRef, value is set of item indices.

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
    /// rewrite.py:41-45: _write_barrier_applied
    _write_barrier_applied: HashSet<u32>,
    /// Forwarding map from original result OpRefs to rewritten result OpRefs.
    forwarding: HashMap<u32, OpRef>,
    /// One-shot marker for an already-emitted array write barrier.
    ///
    /// Re-running the rewriter over an already-rewritten trace should not
    /// duplicate a `COND_CALL_GC_WB_ARRAY` immediately followed by the
    /// corresponding `SETARRAYITEM_GC`.
    pending_array_wb: Option<u32>,

    // ── Array length tracking (rewrite.py:59 _known_lengths) ──
    _known_lengths: HashMap<u32, usize>,

    // ── Constant addition tracking (rewrite.py:64 _constant_additions) ──
    _constant_additions: HashMap<u32, (OpRef, i64)>,

    // ── Pending zero tracking (rewrite.py:34-36) ──
    /// rewrite.py:34 last_zero_arrays
    last_zero_arrays: Vec<usize>,
    /// rewrite.py:35 _setarrayitems_occurred
    setarrayitems_occurred: HashMap<u32, HashSet<usize>>,
    /// rewrite.py:61 _delayed_zero_setfields
    _delayed_zero_setfields: HashMap<u32, HashSet<usize>>,

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
            _write_barrier_applied: HashSet::new(),
            forwarding: HashMap::new(),
            pending_array_wb: None,
            _known_lengths: HashMap::new(),
            _constant_additions: HashMap::new(),
            last_zero_arrays: Vec::new(),
            setarrayitems_occurred: HashMap::new(),
            _delayed_zero_setfields: HashMap::new(),
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
        self._write_barrier_applied.clear();
        self.pending_array_wb = None;
        self.emit_pending_zeros();
        self._constant_additions.clear();
    }

    /// rewrite.py:716-717: remember_write_barrier
    fn remember_write_barrier(&mut self, r: OpRef) {
        self._write_barrier_applied.insert(r.0);
    }

    /// rewrite.py:66-67: remember_known_length
    fn remember_known_length(&mut self, op: OpRef, length: usize) {
        self._known_lengths.insert(op.0, length);
    }

    /// rewrite.py:81-82: known_length(op, default)
    fn known_length(&self, op: OpRef, default: usize) -> usize {
        self._known_lengths.get(&op.0).copied().unwrap_or(default)
    }

    /// rewrite.py:713-714: write_barrier_applied(op)
    fn write_barrier_applied(&self, r: OpRef) -> bool {
        self._write_barrier_applied.contains(&r.0)
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

    /// rewrite.py:69-76 remember_setarrayitem_occurred
    fn remember_setarrayitem_occurred(&mut self, array_ref: OpRef, index: usize) {
        let resolved = self.resolve(array_ref);
        self.setarrayitems_occurred
            .entry(resolved.0)
            .or_default()
            .insert(index);
    }

    /// rewrite.py:78-79 setarrayitems_occurred
    fn setarrayitems_occurred(&self, array_ref: OpRef) -> Option<&HashSet<usize>> {
        let resolved = self.resolve(array_ref);
        self.setarrayitems_occurred.get(&resolved.0)
    }

    /// rewrite.py:719-766 emit_pending_zeros
    ///
    /// Rewrite ZERO_ARRAY ops in `last_zero_arrays` in-place: shrink the
    /// start/length to skip indices covered by SETARRAYITEM writes.
    /// Non-constant-length ZERO_ARRAYs are not in the list and are left
    /// as-is (they were already emitted verbatim).
    fn emit_pending_zeros(&mut self) {
        // rewrite.py:723-757: rewrite last_zero_arrays in-place.
        let indices = std::mem::take(&mut self.last_zero_arrays);
        for out_idx in indices {
            let op = &self.out[out_idx];
            debug_assert_eq!(op.opcode, OpCode::ZeroArray);
            let descr = op.descr.clone();
            let item_size = descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .map(|ad| ad.item_size())
                .unwrap_or(1);
            let arr = op.arg(0);

            let intset = self.setarrayitems_occurred(arr);
            if intset.is_none() {
                // rewrite.py:731-743: no SETARRAYITEM occurred — apply scale.
                let start = self.resolve_constant(op.arg(1).0).unwrap_or(0) as usize;
                let length = self.resolve_constant(op.arg(2).0).unwrap_or(0) as usize;
                let scaled_op = Op::with_descr(
                    OpCode::ZeroArray,
                    &[
                        arr,
                        self.const_int((start * item_size) as i64),
                        self.const_int((length * item_size) as i64),
                        self.const_int(1),
                        self.const_int(1),
                    ],
                    descr.unwrap(),
                );
                self.out[out_idx] = scaled_op;
                continue;
            }

            // rewrite.py:744-756: trim start and stop by written indices.
            let intset = intset.unwrap();
            let mut start: usize = 0;
            while intset.contains(&start) {
                start += 1;
            }
            let stop_raw = self.resolve_constant(op.arg(2).0).unwrap_or(0) as usize;
            let mut stop = stop_raw;
            debug_assert!(start <= stop);
            while stop > start && intset.contains(&(stop - 1)) {
                stop -= 1;
            }
            let scaled_op = Op::with_descr(
                OpCode::ZeroArray,
                &[
                    arr,
                    self.const_int((start * item_size) as i64),
                    self.const_int(((stop - start) * item_size) as i64),
                    self.const_int(1),
                    self.const_int(1),
                ],
                descr.unwrap(),
            );
            self.out[out_idx] = scaled_op;
        }
        self.setarrayitems_occurred.clear();
        // rewrite.py:760-766: write NULL-pointer-writing ops still pending.
        let delayed = std::mem::take(&mut self._delayed_zero_setfields);
        for (obj_key, offsets) in delayed {
            let obj = self.resolve(OpRef(obj_key));
            for ofs in offsets {
                let c_ofs = self.const_int(ofs as i64);
                let c_zero = self.const_int(0);
                let c_word = self.const_int(8); // WORD
                self.emit(Op::new(OpCode::GcStore, &[obj, c_ofs, c_zero, c_word]));
            }
        }
    }

    /// rewrite.py:84-91: delayed_zero_setfields(op)
    fn delayed_zero_setfields(&mut self, op: OpRef) -> &mut HashSet<usize> {
        let resolved = self.resolve(op);
        self._delayed_zero_setfields.entry(resolved.0).or_default()
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

    /// rewrite.py:498-504: clear_gc_fields
    fn clear_gc_fields(&self, op: &Op, result: OpRef, st: &mut RewriteState) {
        // In RPython, malloc_zero_filled = True for incminimark.
        // Our nursery alloc also zero-fills, so this is typically a no-op.
        // But structurally we track which GC fields need zeroing.
        let descr = op.descr.as_ref().and_then(|d| d.as_size_descr());
        if let Some(sd) = descr {
            let gc_descs = sd.gc_field_descrs();
            if !gc_descs.is_empty() {
                let d = st.delayed_zero_setfields(result);
                for fd in gc_descs {
                    d.insert(fd.offset());
                }
            }
        }
    }

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

        // rewrite.py:544: self.clear_gc_fields(descr, op)
        self.clear_gc_fields(op, obj_ref, st);
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

        // rewrite.py:588-611 handle_clear_array_contents
        // rewrite.py:588-611 handle_clear_array_contents
        if is_clear && !v_length.is_none() {
            let is_const_zero = st.resolve_constant(v_length.0).map_or(false, |v| v == 0);
            if !is_const_zero {
                let item_size = descr.item_size();
                let c_zero = st.const_int(0);

                // rewrite.py:599-602: compute v_length_scaled and scale
                let (v_length_scaled, scale) = if st.resolve_constant(v_length.0).is_some() {
                    // ConstInt: keep original, scale applied in emit_pending_zeros.
                    (v_length, item_size)
                } else {
                    // Non-constant: rewrite.py:601 _emit_mul_if_factor_offset_not_supported
                    // cpu_simplify_scale: emit INT_MUL(v_length, scale), return scale=1.
                    let scale_const = st.const_int(item_size as i64);
                    let mul_op = Op::new(OpCode::IntMul, &[v_length, scale_const]);
                    let v_length_scaled = st.emit_result(mul_op, OpRef(st.next_pos));
                    (v_length_scaled, 1)
                };

                // rewrite.py:607-608: args = [v_arr, c_zero, v_length_scaled, scale, v_scale]
                let v_scale = st.const_int(scale as i64);
                let mut zero_op = Op::with_descr(
                    OpCode::ZeroArray,
                    &[
                        result,
                        c_zero,
                        v_length_scaled,
                        st.const_int(scale as i64),
                        v_scale,
                    ],
                    descr_ref.clone(),
                );
                zero_op.pos = OpRef(st.next_pos);
                let zero_idx = st.out.len();
                st.emit(zero_op);
                // rewrite.py:610-611: only add to last_zero_arrays if ConstInt
                if st.resolve_constant(v_length.0).is_some() {
                    st.last_zero_arrays.push(zero_idx);
                }
            }
        }

        // rewrite.py:551: if isinstance(v_length, ConstInt):
        //     self.remember_known_length(op, v_length.getint())
        if let Some(num_elem) = st.resolve_constant(v_length.0) {
            st.remember_known_length(result, num_elem as usize);
        }

        // Don't add to _write_barrier_applied: large young arrays may need card
        // marking, so the GC still relies on write barriers.
    }

    // ────────────────────────────────────────────────────────
    // SETFIELD_GC  → maybe COND_CALL_GC_WB + SETFIELD_GC
    // ────────────────────────────────────────────────────────

    /// rewrite.py:506-512: consider_setfield_gc
    fn consider_setfield_gc(&self, op: &Op, st: &mut RewriteState) {
        let obj = st.resolve(op.arg(0));
        if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_field_descr()) {
            let ofs = fd.offset();
            if let Some(d) = st._delayed_zero_setfields.get_mut(&obj.0) {
                d.remove(&ofs);
            }
        }
    }

    /// rewrite.py:926-934: handle_write_barrier_setfield
    fn handle_write_barrier_setfield(&self, op: &Op, st: &mut RewriteState) {
        let rewritten = st.rewrite_op(op);
        let val = rewritten.arg(0);

        // Flush pending vtable init for this object.
        if !st.pending_vtable_inits.is_empty() {
            if let Some(pos) = st.pending_vtable_inits.iter().position(|(o, _)| *o == val) {
                let (o, vtable) = st.pending_vtable_inits.remove(pos);
                self.gen_initialize_vtable(o, vtable, st);
            }
        }

        // rewrite.py:928-931
        if !st.write_barrier_applied(val) {
            let v = rewritten.arg(1);
            let field_is_ptr = op
                .descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .map(|fd| fd.is_pointer_field())
                .unwrap_or(false);
            let val_is_ref = if field_is_ptr {
                match st.result_type_of(v) {
                    Some(tp) => tp == Type::Ref,
                    None => true,
                }
            } else {
                false
            };
            if val_is_ref && (!st.is_null_constant(v)) {
                self.gen_write_barrier(val, st);
            }
        }
        st.emit(rewritten);
    }

    /// rewrite.py:514-518: consider_setarrayitem_gc
    fn consider_setarrayitem_gc(&self, op: &Op, st: &mut RewriteState) {
        let array_box = st.resolve(op.arg(0));
        let index_ref = op.arg(1);
        if let Some(index_val) = st.resolve_constant(index_ref.0) {
            st.remember_setarrayitem_occurred(array_box, index_val as usize);
        }
    }

    /// rewrite.py:936-944: handle_write_barrier_setarrayitem
    fn handle_write_barrier_setarrayitem(&self, op: &Op, st: &mut RewriteState) {
        let rewritten = st.rewrite_op(op);
        let val = rewritten.arg(0);

        if st.take_pending_array_wb(val) {
            st.emit(rewritten);
            return;
        }

        // rewrite.py:938-942
        if !st.write_barrier_applied(val) {
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
            if val_is_ref && !st.is_null_constant(v) {
                self.gen_write_barrier_array(val, rewritten.arg(1), st);
            }
        }
        st.emit(rewritten);
    }

    /// rewrite.py:948-953: gen_write_barrier
    fn gen_write_barrier(&self, v_base: OpRef, st: &mut RewriteState) {
        let wb_op = Op::new(OpCode::CondCallGcWb, &[v_base]);
        st.emit(wb_op);
        st.remember_write_barrier(v_base);
    }

    /// rewrite.py:955-973: gen_write_barrier_array
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
                // any future write barriers, so don't remember_write_barrier!
                return;
            }
        }
        // rewrite.py:973: fall-back case: produce a write_barrier
        self.gen_write_barrier(v_base, st);
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
            st.remember_write_barrier(r);
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
                st.remember_write_barrier(r);
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
        st.remember_write_barrier(r);
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

    /// rewrite.py:1008-1031: record_int_add_or_sub
    fn record_int_add_or_sub(op: &Op, is_subtraction: bool, st: &mut RewriteState) {
        let arg1 = op.arg(1);
        let (constant, base_box) = if let Some(c) = st.resolve_constant(arg1.0) {
            let c = if is_subtraction { -c } else { c };
            (c, st.resolve(op.arg(0)))
        } else {
            if is_subtraction {
                return;
            }
            let arg0 = op.arg(0);
            if let Some(c) = st.resolve_constant(arg0.0) {
                (c, st.resolve(arg1))
            } else {
                return;
            }
        };
        let (final_box, final_constant) =
            if let Some(&(older_box, extra_offset)) = st._constant_additions.get(&base_box.0) {
                (older_box, constant + extra_offset)
            } else {
                (base_box, constant)
            };
        if !op.pos.is_none() {
            st._constant_additions
                .insert(op.pos.0, (final_box, final_constant));
        }
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
                    st._known_lengths.clear();
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

                // rewrite.py:384-387: record INT_ADD or INT_SUB with a constant
                OpCode::IntAdd | OpCode::IntAddOvf => {
                    Self::record_int_add_or_sub(op, false, &mut st);
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    Self::record_int_add_or_sub(op, true, &mut st);
                    let rewritten = st.rewrite_op(op);
                    st.emit_rewritten_from(op, rewritten);
                }

                // ── rewrite.py:394-404: write barriers ──
                OpCode::SetfieldGc => {
                    self.consider_setfield_gc(op, &mut st);
                    self.handle_write_barrier_setfield(op, &mut st);
                    continue;
                }
                OpCode::SetarrayitemGc => {
                    self.consider_setarrayitem_gc(op, &mut st);
                    self.handle_write_barrier_setarrayitem(op, &mut st);
                    continue;
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
                    st.remember_write_barrier(obj);
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

        // The CallMallocNursery result at pos=0 is in _write_barrier_applied,
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
        // NEW_ARRAY_CLEAR(3) + SET[0] + SET[1] + SET[2] → ZERO_ARRAY(start=0,len=0)
        let rw = make_rewriter();
        let len_ref = OpRef(10_000);
        let mut constants = HashMap::new();
        constants.insert(10_000, 3_i64);
        constants.insert(10_001, 0_i64);
        constants.insert(10_002, 1_i64);
        constants.insert(10_003, 2_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_001), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_002), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_003), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, _) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        // All indices SET → emit_pending_zeros trims to length 0 (no-op).
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        // The ZERO_ARRAY is still emitted but with length=0 (a no-op).
        assert_eq!(zeros.len(), 1, "ZERO_ARRAY is present but a no-op");
    }

    #[test]
    fn test_pending_zero_partially_initialized() {
        // NEW_ARRAY_CLEAR(4) + SET[0] + SET[1] → ZERO_ARRAY trimmed to start=2
        let rw = make_rewriter();
        let len_ref = OpRef(10_000);
        let mut constants = HashMap::new();
        constants.insert(10_000, 4_i64);
        constants.insert(10_001, 0_i64);
        constants.insert(10_002, 1_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_001), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_002), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, new_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        // Indices 0,1 SET. ZERO_ARRAY trimmed to start=2*itemsize, len=2*itemsize.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "should emit exactly one ZERO_ARRAY");
        // start and length are scaled by item_size (8 bytes for i64 array).
        let start_val = new_consts
            .get(&zeros[0].args[1].0)
            .or(constants.get(&zeros[0].args[1].0))
            .copied()
            .unwrap_or(-1);
        let len_val = new_consts
            .get(&zeros[0].args[2].0)
            .or(constants.get(&zeros[0].args[2].0))
            .copied()
            .unwrap_or(-1);
        assert_eq!(start_val, 8, "zero start should be 2*4=8");
        assert_eq!(len_val, 8, "zero length should be 2*4=8");
    }

    #[test]
    fn test_pending_zero_flushed_at_guard() {
        // Guard forces pending zero flush even if no indices were SET.
        // RPython: ZERO_ARRAY is emitted at NEW_ARRAY_CLEAR time (always),
        // then optimized in emit_pending_zeros at flush points.
        let rw = make_rewriter();
        let len_ref = OpRef(10_000);
        let mut constants = HashMap::new();
        constants.insert(10_000, 3_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::new(OpCode::GuardTrue, &[OpRef(50)]),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, new_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        let zero_idx = result.iter().position(|o| o.opcode == OpCode::ZeroArray);
        let guard_idx = result.iter().position(|o| o.opcode == OpCode::GuardTrue);
        assert!(zero_idx.is_some(), "ZERO_ARRAY should be emitted");
        assert!(guard_idx.is_some(), "GuardTrue should be present");
        // ZERO_ARRAY emitted at NEW_ARRAY_CLEAR time, before guard.
        assert!(
            zero_idx.unwrap() < guard_idx.unwrap(),
            "ZERO_ARRAY should come before GuardTrue"
        );

        let zero = result
            .iter()
            .find(|o| o.opcode == OpCode::ZeroArray)
            .unwrap();
        let start_val = new_consts
            .get(&zero.args[1].0)
            .or(constants.get(&zero.args[1].0))
            .copied()
            .unwrap_or(-1);
        let len_val = new_consts
            .get(&zero.args[2].0)
            .or(constants.get(&zero.args[2].0))
            .copied()
            .unwrap_or(-1);
        assert_eq!(start_val, 0, "zero start should be 0");
        assert_eq!(len_val, 12, "zero length should be 3*4=12");
    }

    #[test]
    fn test_pending_zero_gap_in_middle() {
        // NEW_ARRAY_CLEAR(5) + SET[0] + SET[2] + SET[4] → ZERO_ARRAY trimmed.
        // RPython produces ONE ZERO_ARRAY (trimmed start and stop), not two runs.
        let rw = make_rewriter();
        let len_ref = OpRef(10_000);
        let mut constants = HashMap::new();
        constants.insert(10_000, 5_i64);
        constants.insert(10_001, 0_i64);
        constants.insert(10_002, 2_i64);
        constants.insert(10_003, 4_i64);
        let mut new_array = Op::with_descr(OpCode::NewArrayClear, &[len_ref], array_descr_int());
        new_array.pos = OpRef(0);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_001), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_002), OpRef(100)],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(10_003), OpRef(100)],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, new_consts) = rw.rewrite_for_gc_with_constants(&ops, &constants);

        // RPython trims: start=1 (first non-SET), stop=4 (last non-SET+1).
        // Length = (4-1)*8 = 24, start = 1*8 = 8.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "RPython emits one ZERO_ARRAY, trimmed");
        let start_val = new_consts
            .get(&zeros[0].args[1].0)
            .or(constants.get(&zeros[0].args[1].0))
            .copied()
            .unwrap_or(-1);
        let len_val = new_consts
            .get(&zeros[0].args[2].0)
            .or(constants.get(&zeros[0].args[2].0))
            .copied()
            .unwrap_or(-1);
        assert_eq!(start_val, 4, "zero start should be 1*4=4");
        assert_eq!(len_val, 12, "zero length should be (4-1)*4=12");
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
