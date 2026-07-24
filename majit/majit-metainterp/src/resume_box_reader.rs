//! Bridge-entry virtual materialization — the trace-emit (box-reader) flavour
//! of the resume-data reader.
//!
//! resume.py runs virtual rematerialization through a single
//! `AbstractVirtualInfo.allocate(decoder, index)` (resume.py:618-767) that is
//! polymorphic over the `decoder`: `ResumeDataDirectReader` (blackhole — real
//! `cpu.bh_new`, resume.py:1437-1442) writes memory, while
//! `ResumeDataBoxReader` (tracing/bridge — `metainterp.execute_new_with_vtable`,
//! resume.py:1111-1115) records NEW/SETFIELD ops into the trace. The direct
//! flavour lives in `resume.rs` (`VirtualInfoBlackholeExt::allocate`); this
//! module is the box-reader flavour, shared by every `JitState` consumer whose
//! `setup_bridge_sym` must re-materialize the guard's virtuals and replay its
//! deferred heap writes at bridge entry.

use majit_ir::OpRef;

/// Default `VRawBuffer` array-descr factory for consumers that never
/// materialize raw-buffer virtuals (e.g. a `#[jit_interp]` interpreter with no
/// ctypes/array virtuals). Mints a fresh `SimpleArrayDescr` via majit-ir,
/// ignoring `len_offset` (raw stores read only item size, and no length field
/// is consulted). A consumer that DOES virtualize raw buffers must inject its
/// own gccache-keyed factory instead.
pub fn default_bridge_array_descr(
    base_size: usize,
    item_size: usize,
    _len_offset: Option<usize>,
    item_type: majit_ir::Type,
    signed: bool,
) -> majit_ir::DescrRef {
    majit_ir::descr::make_array_descr_signed(base_size, item_size, item_type, signed)
}

/// resume.py:874-899 VirtualCache — per-virtual-number `OpRef` banks the box
/// reader probes before allocating, plus the concrete `GcRef`/int shadows a
/// consumer may seed for branch-fold parity. `mint_raw_array_descr` is the
/// consumer-provided array-descr factory for `VRawBuffer` materialization (the
/// only virtual kind whose descr is minted rather than looked up in a parent
/// `SizeDescr`); descr identity is the consumer's gccache concern, so it is
/// injected rather than fixed in core.
pub struct BridgeVirtualCache {
    virtuals_ptr_cache: Vec<Option<OpRef>>,
    virtuals_int_cache: Vec<Option<OpRef>>,
    concrete_ptr_cache: Vec<Option<majit_ir::GcRef>>,
    concrete_int_cache: Vec<Option<i64>>,
    mint_raw_array_descr:
        fn(usize, usize, Option<usize>, majit_ir::Type, bool) -> majit_ir::DescrRef,
}

impl BridgeVirtualCache {
    pub fn new(
        size: usize,
        mint_raw_array_descr: fn(
            usize,
            usize,
            Option<usize>,
            majit_ir::Type,
            bool,
        ) -> majit_ir::DescrRef,
    ) -> Self {
        Self {
            virtuals_ptr_cache: vec![None; size],
            virtuals_int_cache: vec![None; size],
            concrete_ptr_cache: vec![None; size],
            concrete_int_cache: vec![None; size],
            mint_raw_array_descr,
        }
    }

    pub fn get_any(&self, i: usize) -> Option<OpRef> {
        self.virtuals_ptr_cache
            .get(i)
            .copied()
            .flatten()
            .or_else(|| self.virtuals_int_cache.get(i).copied().flatten())
    }

    pub fn set_ptr(&mut self, i: usize, v: OpRef) {
        self.virtuals_ptr_cache[i] = Some(v);
    }

    pub fn set_int(&mut self, i: usize, v: OpRef) {
        self.virtuals_int_cache[i] = Some(v);
    }

    pub fn get_concrete_ptr(&self, i: usize) -> Option<majit_ir::GcRef> {
        self.concrete_ptr_cache.get(i).copied().flatten()
    }

    pub fn set_concrete_ptr(&mut self, i: usize, v: majit_ir::GcRef) {
        if i < self.concrete_ptr_cache.len() {
            self.concrete_ptr_cache[i] = Some(v);
        }
    }

    pub fn get_concrete_int(&self, i: usize) -> Option<i64> {
        self.concrete_int_cache.get(i).copied().flatten()
    }

    pub fn set_concrete_int(&mut self, i: usize, v: i64) {
        if i < self.concrete_int_cache.len() {
            self.concrete_int_cache[i] = Some(v);
        }
    }
}

fn emit_stroruni_oopspec_call(
    ctx: &mut crate::TraceCtx,
    oopspec: majit_ir::effectinfo::OopSpecIndex,
    args: &[OpRef],
) -> OpRef {
    let cic = ctx
        .callinfocollection
        .as_ref()
        .expect(
            "TraceCtx.callinfocollection missing — bridge-virtual VStr/VUni \
             Concat/Slice materialization requires pyjitpl to populate it \
             (resume.py:1143-1188)",
        )
        .clone();
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr.expect("callinfo_for_oopspec missing entry for VStr/VUni oopspec");
    let func_const = ctx.const_int(func as i64);
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(func_const);
    call_args.extend_from_slice(args);
    // resume.py:1143-1160 `execute_and_record_varargs(CALL_R, ...)`.
    ctx.profiler()
        .count_ops(majit_ir::OpCode::CallR, crate::counters::OPS);
    ctx.profiler()
        .count_ops(majit_ir::OpCode::CallR, crate::counters::RECORDED_OPS);
    ctx.record_op_with_descr(majit_ir::OpCode::CallR, &call_args, calldescr.clone())
}

/// resume.py:1556-1564 decode_box parity for fieldnums (i16 tagged): decode one
/// tagged array/field value into its bridge `OpRef` (typed InputArg for TAGBOX,
/// const for TAGINT/TAGCONST, recursively materialized virtual for TAGVIRTUAL).
pub fn decode_fieldnum(
    ctx: &mut crate::TraceCtx,
    tagged: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};
    // resume.py:1245 `decode_box` dispatches purely on the tag bits;
    // it has no UNINITIALIZED case. The UNINITIALIZED skip lives in
    // the callers (e.g. VArrayStructInfo.allocate, resume.py:629),
    // so this decoder mirrors `decode_box` exactly — an UNINITIALIZED
    // tag reaching here falls into the TAGCONST arm and fails loud on
    // the out-of-range const index, matching upstream's IndexError.
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            // resume.py:1247-1264 decode_box parity:
            //   if num < 0: num += len(liveboxes)
            //   return self.liveboxes[num]
            // The returned Box object carries `box.type` intrinsically
            // (history.py:220). For the bridge tracer, those liveboxes
            // are the bridge's `InputArg{Int,Ref,Float}` slots, so we
            // mint the typed `OpRef::input_arg_typed` variant matching
            // `fail_arg_types[idx]` rather than a bare untyped raw
            // OpRef — variant-aware Eq (resoperation.rs:290) requires
            // the optimizer/heap-cache key to be the same typed variant
            // the bridge inputarg list produces.
            let idx = if val < 0 {
                val + resume_data.num_failargs
            } else {
                val
            };
            // resume.py:1261 `box = self.liveboxes[num]` — direct
            // indexing, IndexError on out-of-range. Encoder /
            // decoder asymmetry is a bug, not a silent fallback;
            // mirror the upstream fail-loud contract.
            let tp = *resume_data
                .fail_arg_types
                .get(idx as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "decode_fieldnum TAGBOX out-of-range: idx={} num_failargs={} \
                         fail_arg_types.len()={} (encoder/decoder mismatch — see \
                         resume.py:1245-1264 decode_box)",
                        idx,
                        resume_data.num_failargs,
                        resume_data.fail_arg_types.len()
                    )
                });
            OpRef::input_arg_typed(idx as u32, tp)
        }
        TAGINT => ctx.const_int(val as i64),
        TAGCONST => {
            // resume.py:1247-1251 decode_box parity:
            //   if tag == TAGCONST:
            //       if tagged_eq(tagged, NULLREF):
            //           box = CONST_NULL
            //       else:
            //           box = self.consts[num - TAG_CONST_OFFSET]
            if tagged == majit_ir::resumedata::NULLREF {
                return ctx.const_null();
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py:1251 `box = self.consts[num - TAG_CONST_OFFSET]`
            // — direct indexing, fail-fast on out-of-range (mirrors
            // Python IndexError; never silently substitutes).
            // compile.py:853 `ResumeGuardDescr` storage — read off
            // the shared Arc so the bridge tracer observes the
            // same pool the GC walker updates.
            let storage = resume_data
                .storage
                .as_ref()
                .expect("resume_data.storage missing");
            let c = storage.rd_consts()[ci];
            match c.get_type() {
                majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            }
        }
        TAGVIRTUAL => {
            // resume.py:278-284 nested virtuals are numbered negatively;
            // getvirtual resolves them via Python negative list indexing
            // into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            materialize_bridge_virtual(ctx, vidx, rd_virtuals, resume_data, cache)
        }
        _ => OpRef::NONE,
    }
}

pub fn materialize_bridge_virtual(
    ctx: &mut crate::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py:874-899 VirtualCache: list caches indexed by virtual number
    // (ptr and int banks). This bridge helper is still OpRef-typed, so it
    // probes both banks before allocating.
    if let Some(cached) = cache.get_any(vidx) {
        return cached;
    }

    // resume.py:947 assert self.virtuals_cache is not None — a TAGVIRTUAL in
    // the stream guarantees rd_virtuals is present; None is an encoder bug.
    let virtuals = rd_virtuals.expect("materialize_bridge_virtual: rd_virtuals is None");
    // resume.py:951 self.rd_virtuals[index] — direct indexing, IndexError on
    // an out-of-range virtual number is a bug, not a silent NONE fallback.
    let entry = &virtuals[vidx];

    // resume.py:612-760 dispatch by virtual kind.
    // RPython: rd_virtuals[index].allocate(self, index) — polymorphic on
    // the AbstractVirtualInfo subclass. Rust equivalent: match on
    // RdVirtualInfo enum variant.

    /// resume.py:591-603 AbstractVirtualStructInfo.setfields helper.
    /// Walks fielddescrs in lock-step with fieldnums, decoding each
    /// fieldnum and emitting SETFIELD_GC.
    fn setfields(
        ctx: &mut crate::TraceCtx,
        struct_op: OpRef,
        fielddescrs: &[majit_ir::FieldDescrInfo],
        fieldnums: &[i16],
        parent_descr: majit_ir::DescrRef,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        resume_data: &crate::jit_state::ResumeDataResult,
        cache: &mut BridgeVirtualCache,
    ) {
        // resume.py:597-603 setfields — range(len(fielddescrs)), index
        // fieldnums[i]. The len-equality assert (resume.py:606) is in
        // debug_prints, not this allocate path: a short fieldnums raises
        // IndexError here, a longer one is ignored.
        for i in 0..fielddescrs.len() {
            let fd_info = &fielddescrs[i];
            let fnum = fieldnums[i];
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                continue;
            }
            // resume.py:597-603 self.setfields → decoder.setfield(struct,
            // fieldnum, fielddescr): reuse the parent SizeDescr's live
            // FieldDescr (canonical immutable / quasi-immutable / ei_index)
            // rather than reconstructing a partial copy. The descr is keyed
            // by index_in_parent (small sequential), not the 268M-hash
            // stable_field_index.
            let field_descr =
                majit_ir::descr::field_descr_from_parent_by_offset(&parent_descr, fd_info.offset);
            // resume.py:1111-1122 materializer operations use
            // `execute_and_record`.
            ctx.profiler()
                .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            // Bridge virtual rematerialisation — `upd.setfield(valuebox)`
            // parity: cache stores the Box identity (`value` OpRef).
            // Cache-hit readers resolve the intrinsic value via
            // `box_value(cached)` at hit time (covering const pool,
            // standard-virtualizable shadow, and the frontend object's
            // `value` field) — non-Const operands whose runtime concrete
            // was stamped at the original record site (or threaded from
            // the parent guard's fail_args via `set_opref_concrete`)
            // surface through that `value` field; unstamped operands
            // return `None` so the downstream sanity check skips.
            ctx.heapcache_setfield_cached(struct_op, fd_info.index, value);
        }
    }

    match entry.as_ref() {
        // resume.py:612-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(size_descr) = descr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:619 decoder.allocate_with_vtable(descr=self.descr)
            ctx.profiler()
                .count_ops(OpCode::NewWithVtable, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::NewWithVtable, crate::counters::RECORDED_OPS);
            let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:620 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:621 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                size_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:628-637 VStructInfo.allocate
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(struct_descr) = typedescr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:635 decoder.allocate_struct(self.typedescr)
            ctx.profiler().count_ops(OpCode::New, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::New, crate::counters::RECORDED_OPS);
            let new_op = ctx.record_op_with_descr(OpCode::New, &[], struct_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:636 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:637 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                struct_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:649-671 AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            let kind = *kind;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py:653 decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            // resume.py:645 AbstractVArrayInfo.__init__ asserts arraydescr is
            // not None; resume.py:652 allocate reads self.arraydescr directly.
            let array_descr = arraydescr.clone().expect("VArrayInfo: arraydescr is None");
            ctx.profiler().count_ops(alloc_opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(alloc_opcode, crate::counters::RECORDED_OPS);
            let new_op = ctx.record_op_with_descr(alloc_opcode, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:654 decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:656-670 element loop: dispatch by arraydescr kind
            // NB. the check for the kind of array elements is moved out of the loop
            let set_opcode = match kind {
                0 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_pointers()
                2 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_floats() — TODO: SetarrayitemRaw/Float
                _ => OpCode::SetarrayitemGc, // int
            };
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if value.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                // resume.py:660/665/670 setarrayitem_{ref,float,int}
                ctx.profiler().count_ops(set_opcode, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(set_opcode, crate::counters::RECORDED_OPS);
                ctx.record_op_with_descr(
                    set_opcode,
                    &[new_op, idx_ref, value],
                    array_descr.clone(),
                );
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef::from_raw({})",
                    vidx,
                    clear,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py:749: array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            ctx.profiler()
                .count_ops(OpCode::NewArrayClear, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::NewArrayClear, crate::counters::RECORDED_OPS);
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                    if value.is_none() {
                        continue;
                    }
                    let idx_ref = ctx.const_int(i as i64);
                    // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                    ctx.record_op_with_descr(
                        OpCode::SetinteriorfieldGc,
                        &[new_op, idx_ref, value],
                        fielddescrs[j].clone(),
                    );
                }
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
            // resume.py:1124-1132: ResumeDataBoxReader.allocate_raw_buffer →
            //   execute_and_record_varargs(rop.CALL_I, [ConstInt(func), ConstInt(size)], calldescr)
            let func_ref = ctx.const_int(*func);
            let size_ref = ctx.const_int(*size as i64);
            // resume.py:1124-1126: calldescr comes from the shared
            // callinfocollection, not a freshly minted synthetic descr. The
            // func is NOT taken from callinfo_for_oopspec (resume.py:1127-1130:
            // several malloc variants share the oopspec), so only the calldescr
            // is read from the CIC and *func from the VRawBufferInfo is kept.
            let cic = ctx
                .callinfocollection
                .as_ref()
                .expect(
                    "TraceCtx.callinfocollection missing — bridge-virtual \
                     VRawBufferInfo materialization requires pyjitpl to populate \
                     it (resume.py:1124-1126)",
                )
                .clone();
            // resume.py:1126: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py:444-447) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1131-1132: execute_and_record_varargs(CALL_I,
            //   [func, size], calldescr). A missing entry surfaces here, as the
            // calldescr is consumed by the CALL_I op, not as a separate lookup
            // assertion.
            ctx.profiler()
                .count_ops(OpCode::CallI, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::CallI, crate::counters::RECORDED_OPS);
            let buffer = ctx.record_op_with_descr(
                OpCode::CallI,
                &[func_ref, size_ref],
                calldescr
                    .cloned()
                    .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)"),
            );
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets), indexing
            // self.descrs[i] and self.fieldnums[i] by the same i — a short
            // descrs/fieldnums raises IndexError here (encoder bug), a longer
            // one is ignored. No len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                // resume.py:701-708 VRawBufferStateInfo.allocate_int passes
                // fieldnums[i] straight to setrawbuffer_item with no
                // UNINITIALIZED skip (unlike VArrayStructInfo) — a raw buffer
                // is fully written by the encoder.
                // resume.py:1232: itembox = self.decode_box(fieldnum, kind).
                // `decode_box` always returns a box (no UNINITIALIZED case),
                // so the store is unconditional, matching setrawbuffer_item.
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                // resume.py:1225-1234: setrawbuffer_item (direct reader).
                // Dispatches pointer/float/int via arraydescr — all types allowed.
                let di = &descrs[i];
                let tp = match di.item_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr = (cache.mint_raw_array_descr)(
                    di.base_size,
                    di.item_size,
                    di.len_offset,
                    tp,
                    di.is_signed,
                );
                let offset_ref = ctx.const_int(off as i64);
                ctx.profiler()
                    .count_ops(OpCode::RawStore, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(OpCode::RawStore, crate::counters::RECORDED_OPS);
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef::from_raw({})",
                    vidx,
                    func,
                    size,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            // resume.py:725: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base_buffer = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            // resume.py:726: buffer = decoder.int_add_const(base_buffer, self.offset)
            let offset_ref = ctx.const_int(*offset as i64);
            ctx.profiler()
                .count_ops(OpCode::IntAdd, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::IntAdd, crate::counters::RECORDED_OPS);
            let buffer = ctx.record_op(OpCode::IntAdd, &[base_buffer, offset_ref]);
            // resume.py:727: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawSliceInfo(offset={}) → OpRef::from_raw({})",
                    vidx,
                    offset,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:766-775 VStrPlainInfo.allocate / resume.py:820-829
        // VUniPlainInfo.allocate — `ResumeDataBoxReader.allocate_string /
        // allocate_unicode` followed by `string_setitem` / `unicode_setitem`
        // per character.
        //
        //     length = len(self.fieldnums)
        //     string = decoder.allocate_string(length)        # NEWSTR
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i in range(length):
        //         charnum = self.fieldnums[i]
        //         if not tagged_eq(charnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, charnum)  # STRSETITEM
        //     return string
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len();
            let length_ref = ctx.const_int(length as i64);
            let (alloc_opcode, set_opcode) = if is_unicode {
                (OpCode::Newunicode, OpCode::Unicodesetitem)
            } else {
                (OpCode::Newstr, OpCode::Strsetitem)
            };
            // resume.py:769: string = decoder.allocate_string(length)
            ctx.profiler().count_ops(alloc_opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(alloc_opcode, crate::counters::RECORDED_OPS);
            let string = ctx.record_op(alloc_opcode, &[length_ref]);
            // resume.py:770: decoder.virtuals_cache.set_ptr(index, string)
            cache.set_ptr(vidx, string);
            // resume.py:771-774: string_setitem for each filled char.
            for (i, &charnum) in fieldnums.iter().enumerate() {
                if charnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1138-1141 ResumeDataBoxReader.string_setitem:
                //   charbox = self.decode_box(charnum, INT)
                //   execute_and_record(rop.STRSETITEM, string, ConstInt(index), charbox)
                let charbox = decode_fieldnum(ctx, charnum, rd_virtuals, resume_data, cache);
                if charbox.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                ctx.profiler().count_ops(set_opcode, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(set_opcode, crate::counters::RECORDED_OPS);
                ctx.record_op(set_opcode, &[string, idx_ref, charbox]);
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}PlainInfo(length={}) → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    length,
                    string.raw(),
                );
            }
            string
        }
        // resume.py:785-793 VStrConcatInfo.allocate / resume.py:840-848
        // VUniConcatInfo.allocate:
        //
        //     left, right = self.fieldnums
        //     string = decoder.concat_strings(left, right)   # CALL_R(OS_STR_CONCAT)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.concat_strings` at resume.py:1143-1149:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_CONCAT)
        //     str1box = self.decode_box(str1num, REF)
        //     str2box = self.decode_box(str2num, REF)
        //     execute_and_record_varargs(CALL_R, [ConstInt(func), str1box, str2box], calldescr)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                2,
                "VStr/VUniConcatInfo must have exactly 2 fieldnums (left, right)"
            );
            let left = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let right = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[left, right]);
            cache.set_ptr(vidx, string);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}ConcatInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:805-809 VStrSliceInfo.allocate / resume.py:860-864
        // VUniSliceInfo.allocate:
        //
        //     largerstr, start, length = self.fieldnums
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.slice_string` at resume.py:1151-1160 /
        // `slice_unicode` at resume.py:1179-1188:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_SLICE)
        //     strbox = self.decode_box(strnum, REF)
        //     startbox = self.decode_box(startnum, INT)
        //     lengthbox = self.decode_box(lengthnum, INT)
        //     stopbox = execute_and_record(INT_ADD, startbox, lengthbox)
        //     execute_and_record_varargs(CALL_R,
        //         [ConstInt(func), strbox, startbox, stopbox], calldescr)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                3,
                "VStr/VUniSliceInfo must have exactly 3 fieldnums (largerstr, start, length)"
            );
            let largerstr = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let start = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let length = decode_fieldnum(ctx, fieldnums[2], rd_virtuals, resume_data, cache);
            // resume.py:1157-1158 / :1185-1186: stopbox = INT_ADD(startbox, lengthbox)
            ctx.profiler()
                .count_ops(OpCode::IntAdd, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::IntAdd, crate::counters::RECORDED_OPS);
            let stop = ctx.record_op(OpCode::IntAdd, &[start, length]);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[largerstr, start, stop]);
            cache.set_ptr(vidx, string);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}SliceInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:951/954 getvirtual_ptr direct-indexes rd_virtuals[index];
        // the function preamble already documents this as fail-loud ("not a
        // silent NONE fallback"). An Empty hole here means the resume stream
        // tagged a virtual index that was never assigned a real virtual
        // (encoder/decoder asymmetry) — surface it rather than poison the
        // operand chain with OpRef::NONE.
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_bridge_virtual: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

/// resume.py:1245-1264 `decode_box` symbolic parity for an already-decoded
/// [`RebuiltValue`]: mint the bridge `OpRef` (typed `InputArg` for a live box,
/// const for a pooled const, recursively materialized virtual for a virtual).
pub fn rebuilt_value_to_opref(
    ctx: &mut crate::TraceCtx,
    v: &majit_ir::resumedata::RebuiltValue,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::resumedata::RebuiltValue;
    match v {
        RebuiltValue::Box(n, tp) => OpRef::input_arg_typed(*n as u32, *tp),
        RebuiltValue::Const(c) => match c.get_type() {
            majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
            majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
            _ => ctx.const_int(c.getint()),
        },
        RebuiltValue::Virtual(vidx) => {
            materialize_bridge_virtual(ctx, *vidx, rd_virtuals, resume_data, cache)
        }
        RebuiltValue::Unassigned => OpRef::NONE,
    }
}

/// resume.py:993-1007 `_prepare_pendingfields` op-emission: replay one deferred
/// heap write as a bridge-entry `SETFIELD_GC` / `SETARRAYITEM_GC` and seed the
/// heapcache so a later same-slot get folds against it. `item_index < 0` marks
/// a struct field; `>= 0` an array element.
pub fn emit_pending_field_op(
    ctx: &mut crate::TraceCtx,
    target_op: OpRef,
    value_op: OpRef,
    item_index: i32,
    descr: &majit_ir::DescrRef,
) {
    use majit_ir::OpCode;
    if item_index < 0 {
        // resume.py `_prepare_pendingfields` replays through
        // `execute_and_record`.
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[target_op, value_op], descr.clone());
        ctx.heapcache_setfield_cached(target_op, descr.index(), value_op);
    } else {
        let index_op = ctx.const_int(item_index as i64);
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[target_op, index_op, value_op],
            descr.clone(),
        );
        ctx.heapcache_setarrayitem(target_op, index_op, descr.index(), value_op);
    }
}

/// resume.py:993-1007 `_prepare_pendingfields` (box-reader flavour): replay the
/// guard's deferred heap writes as bridge-entry `SETFIELD_GC` /
/// `SETARRAYITEM_GC` ops so the compiled bridge observes the same heap state the
/// blackhole would rebuild. Symbolic-only — a consumer that also seeds concrete
/// shadows or routes a magic descr (e.g. an exception channel) wraps this or
/// re-implements the loop with those extensions.
pub fn replay_pending_fields(
    ctx: &mut crate::TraceCtx,
    resume_data: &crate::jit_state::ResumeDataResult,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    cache: &mut BridgeVirtualCache,
) {
    let __diag = std::env::var_os("AHEUI_BRIDGE_DIAG").is_some();
    let Some(storage) = resume_data.storage.as_ref() else {
        if __diag {
            eprintln!("[replay] storage=None (no pendingfields replayed)");
        }
        return;
    };
    if __diag {
        eprintln!(
            "[replay] storage=Some pendingfields={} rd_virtuals={} num_failargs={}",
            storage.rd_pendingfields.len(),
            storage.rd_virtuals.len(),
            resume_data.num_failargs,
        );
    }
    let num_virtuals = storage.rd_virtuals.len();
    for pending in &storage.rd_pendingfields {
        let Some(descr) = pending.descr.as_ref() else {
            if __diag {
                eprintln!(
                    "[replay]   pending item_index={} descr=None SKIP",
                    pending.item_index
                );
            }
            continue;
        };
        // resume.py:1002-1005 both operands use the same tagged decoder as frame
        // boxes.
        let rd_consts = storage.rd_consts();
        let target = majit_ir::resumedata::decode_tagged_value(
            pending.target_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            num_virtuals,
        );
        let value = majit_ir::resumedata::decode_tagged_value(
            pending.value_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            num_virtuals,
        );
        if __diag {
            eprintln!(
                "[replay]   pending item_index={} target_tag={} value_tag={} target={:?} value={:?}",
                pending.item_index,
                pending.target_tagged,
                pending.value_tagged,
                std::mem::discriminant(&target),
                std::mem::discriminant(&value),
            );
        }
        let target_op = rebuilt_value_to_opref(ctx, &target, rd_virtuals, resume_data, cache);
        let value_op = rebuilt_value_to_opref(ctx, &value, rd_virtuals, resume_data, cache);
        if target_op.is_none() || value_op.is_none() {
            if __diag {
                eprintln!(
                    "[replay]   -> SKIP (target_op none={} value_op none={})",
                    target_op.is_none(),
                    value_op.is_none()
                );
            }
            continue;
        }
        if __diag {
            eprintln!(
                "[replay]   -> emit_pending_field_op target_op={} value_op={}",
                target_op.raw(),
                value_op.raw()
            );
        }
        emit_pending_field_op(ctx, target_op, value_op, pending.item_index, descr);
    }
}
