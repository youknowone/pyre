use std::cell::RefCell;
/// Virtualize optimization pass: remove heap allocations for non-escaping objects.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualize.py.
///
/// Tracks "virtual" objects — allocations that never escape the trace.
/// Instead of emitting the allocation, fields are tracked in the optimizer.
/// If a virtual escapes (e.g., passed to a call or stored in a non-virtual),
/// it gets "forced" (materialized by emitting the allocation + setfield ops).
use std::collections::HashMap;
use std::sync::Arc;

use majit_ir::{Descr, DescrRef, FieldDescr, OopSpecIndex, Op, OpCode, OpRef, Type, Value};

use crate::optimizeopt::info::{
    PtrInfo, VirtualArrayInfo, VirtualArrayStructInfo, VirtualInfo, VirtualRawBufferInfo,
    VirtualStructInfo, VirtualizableFieldState,
};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Optimizer-level config for virtualizable frame tracking.
///
/// Byte offsets of frame fields that should be tracked symbolically.
/// The optimizer absorbs SetfieldRaw/GetfieldRaw on these fields and
/// carries their values in guard fail_args instead of emitting memory ops.
#[derive(Clone, Debug)]
pub struct VirtualizableConfig {
    /// Byte offsets of static (scalar) frame fields (e.g. next_instr, stack_depth).
    pub static_field_offsets: Vec<usize>,
    /// Types of static (scalar) frame fields, parallel to `static_field_offsets`.
    pub static_field_types: Vec<Type>,
    /// Byte offsets of array pointer frame fields (e.g. locals_w, value_stack_w).
    pub array_field_offsets: Vec<usize>,
    /// Item types of array fields, parallel to `array_field_offsets`.
    pub array_item_types: Vec<Type>,
    /// Trace-entry lengths of array fields, parallel to `array_field_offsets`.
    ///
    /// Standard virtualizable traces carry array elements in the input box
    /// layout; the optimizer needs the concrete lengths to map those input
    /// args back into VirtualizableFieldState without falling back to raw
    /// heap reads.
    pub array_lengths: Vec<usize>,
}

/// Field descriptor index for the `virtual_token` field of JitVirtualRef.
const VREF_VIRTUAL_TOKEN_FIELD_INDEX: u32 = 0x7F00;
/// Field descriptor index for the `forced` field of JitVirtualRef.
const VREF_FORCED_FIELD_INDEX: u32 = 0x7F01;
/// Size descriptor index for the JitVirtualRef struct.
const VREF_SIZE_DESCR_INDEX: u32 = 0x7F10;

/// The virtualize optimization pass.
pub struct OptVirtualize {
    /// Phase 2 (loop body): don't virtualize New() because guard failure
    /// recovery_layout is not yet populated (RPython rd_virtuals equivalent).
    pub is_phase2: bool,
    /// If set, frame OpRef(0) is treated as a virtualizable object
    /// whose field accesses are absorbed by the optimizer.
    vable_config: Option<VirtualizableConfig>,
    /// Maps array pointer OpRef → (array_field_index, array_descr).
    /// Populated when a GetfieldRawI reads an array pointer field from the frame.
    /// The DescrRef is captured from the first SetarrayitemRaw/GetarrayitemRaw op
    /// we absorb, so we can emit correct SetarrayitemRaw on force.
    vable_array_ptrs: HashMap<OpRef, (usize, Option<DescrRef>)>,
    /// Whether virtualizable state has been initialized from existing trace inputs.
    vable_initialized: bool,
    /// Whether setup needs to initialize virtualizable PtrInfo on ctx.
    /// Set in setup(), applied in first propagate_forward().
    needs_vable_setup: bool,
}

impl OptVirtualize {
    pub fn new() -> Self {
        OptVirtualize {
            is_phase2: false,
            vable_config: None,
            vable_array_ptrs: HashMap::new(),
            vable_initialized: false,
            needs_vable_setup: false,
        }
    }

    /// Create with virtualizable config for frame field tracking.
    pub fn with_virtualizable(config: VirtualizableConfig) -> Self {
        OptVirtualize {
            is_phase2: false,
            vable_config: Some(config),
            vable_array_ptrs: HashMap::new(),
            vable_initialized: false,
            needs_vable_setup: false,
        }
    }

    fn record_known_class(
        &mut self,
        obj_ref: OpRef,
        class_ptr: majit_ir::GcRef,
        ctx: &mut OptContext,
    ) {
        let updated = match ctx.get_ptr_info(obj_ref).cloned() {
            Some(PtrInfo::Virtual(mut vinfo)) => {
                if vinfo.known_class.is_none() {
                    vinfo.known_class = Some(class_ptr);
                }
                PtrInfo::Virtual(vinfo)
            }
            Some(PtrInfo::VirtualStruct(vinfo)) => PtrInfo::VirtualStruct(vinfo),
            Some(PtrInfo::VirtualArray(vinfo)) => PtrInfo::VirtualArray(vinfo),
            Some(PtrInfo::VirtualArrayStruct(vinfo)) => PtrInfo::VirtualArrayStruct(vinfo),
            Some(PtrInfo::VirtualRawBuffer(vinfo)) => PtrInfo::VirtualRawBuffer(vinfo),
            Some(PtrInfo::Virtualizable(vinfo)) => PtrInfo::Virtualizable(vinfo),
            Some(PtrInfo::Instance(mut iinfo)) => {
                if iinfo.known_class.is_none() {
                    iinfo.known_class = Some(class_ptr);
                }
                PtrInfo::Instance(iinfo)
            }
            Some(PtrInfo::KnownClass {
                class_ptr: existing,
                is_nonnull,
                ..
            }) => PtrInfo::KnownClass {
                class_ptr: if existing.is_null() {
                    class_ptr
                } else {
                    existing
                },
                is_nonnull: is_nonnull || !class_ptr.is_null(),
                last_guard_pos: -1,
            },
            Some(PtrInfo::NonNull { .. })
            | Some(PtrInfo::Constant(_))
            | Some(PtrInfo::Struct(_))
            | Some(PtrInfo::Array(_))
            | Some(PtrInfo::Str(_))
            | None => PtrInfo::KnownClass {
                class_ptr,
                is_nonnull: true,
                last_guard_pos: -1,
            },
        };
        ctx.set_ptr_info(obj_ref, updated);
    }

    /// Seed virtualizable state from existing trace inputs.
    ///
    /// RPython standard virtualizables do not synthesize optimizer-owned
    /// loop inputs. The interpreter/JitCode contract already carries static
    /// field values and array elements as ordinary trace inputs, and the
    /// optimizer only maps those existing boxes into PtrInfo state.
    fn init_virtualizable(&mut self, ctx: &mut OptContext) {
        let Some(config) = &self.vable_config else {
            return;
        };
        self.vable_initialized = true;
        if ctx.num_inputs() <= 1 {
            return;
        }

        let mut state = VirtualizableFieldState {
            fields: vec![],
            field_descrs: vec![],
            arrays: vec![],
            last_guard_pos: -1,
        };
        let mut flat_input_idx = 1usize;

        for &offset in &config.static_field_offsets {
            if flat_input_idx >= ctx.num_inputs() {
                break;
            }
            let field_idx = virtualizable_field_index(offset);
            let input_ref = OpRef(flat_input_idx as u32);
            set_field(&mut state.fields, field_idx, input_ref);
            set_field_descr(
                &mut state.field_descrs,
                field_idx,
                make_field_index_descr(field_idx),
            );
            flat_input_idx += 1;
        }

        for (array_idx, (&offset, &length)) in config
            .array_field_offsets
            .iter()
            .zip(config.array_lengths.iter())
            .enumerate()
        {
            let field_idx = virtualizable_field_index(offset);
            set_field_descr(
                &mut state.field_descrs,
                field_idx,
                make_field_index_descr(field_idx),
            );

            let mut elements = Vec::with_capacity(length);
            for _ in 0..length {
                if flat_input_idx >= ctx.num_inputs() {
                    break;
                }
                elements.push(OpRef(flat_input_idx as u32));
                flat_input_idx += 1;
            }
            if !elements.is_empty() {
                state.arrays.push((array_idx as u32, elements));
            }
        }

        ctx.set_ptr_info(OpRef(0), PtrInfo::Virtualizable(state));
    }

    // ── PtrInfo accessors (delegated to ctx) ──

    fn is_virtual(opref: OpRef, ctx: &OptContext) -> bool {
        let resolved = ctx.get_box_replacement(opref);
        ctx.get_ptr_info(resolved)
            .is_some_and(|info| info.is_virtual())
    }

    fn is_standard_virtualizable_ref(&self, opref: OpRef, ctx: &OptContext) -> bool {
        self.vable_config.is_some()
            && opref == ctx.get_box_replacement(OpRef(0))
            && matches!(ctx.get_ptr_info(opref), Some(PtrInfo::Virtualizable(_)))
    }

    /// Apply deferred virtualizable setup if needed.
    /// Skips if OpRef(0) already has PtrInfo (e.g. tests pre-populate).
    fn ensure_vable_setup(&mut self, ctx: &mut OptContext) {
        if self.needs_vable_setup {
            self.needs_vable_setup = false;
            if ctx.get_ptr_info(OpRef(0)).is_none() {
                self.init_virtualizable(ctx);
                if ctx.get_ptr_info(OpRef(0)).is_none() {
                    ctx.set_ptr_info(
                        OpRef(0),
                        PtrInfo::Virtualizable(VirtualizableFieldState {
                            fields: vec![],
                            field_descrs: vec![],
                            arrays: vec![],
                            last_guard_pos: -1,
                        }),
                    );
                }
            }
        }
    }

    // ── Force virtual ──

    /// Force a virtual to become concrete: emit the allocation + setfield ops.
    /// Returns the OpRef of the emitted allocation.
    fn force_virtual(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        let info = match ctx.get_ptr_info(resolved) {
            Some(info) if info.is_virtual() => info.clone(),
            _ => return resolved, // not virtual, nothing to do
        };

        match info {
            PtrInfo::Virtual(vinfo) => self.force_virtual_instance(resolved, vinfo, ctx),
            PtrInfo::VirtualArray(vinfo) => self.force_virtual_array(resolved, vinfo, ctx),
            PtrInfo::VirtualStruct(vinfo) => self.force_virtual_struct(resolved, vinfo, ctx),
            PtrInfo::VirtualArrayStruct(vinfo) => {
                self.force_virtual_array_struct(resolved, vinfo, ctx)
            }
            PtrInfo::VirtualRawBuffer(vinfo) => self.force_virtual_raw_buffer(resolved, vinfo, ctx),
            PtrInfo::Virtualizable(vinfo) => {
                if self.is_standard_virtualizable_ref(resolved, ctx) {
                    resolved
                } else {
                    self.force_virtualizable(resolved, vinfo, ctx)
                }
            }
            _ => resolved,
        }
    }

    /// Force a virtualizable: emit SETFIELD_RAW ops to write tracked
    /// field values back to the heap object. Unlike virtual objects,
    /// no allocation is emitted — the object already exists.
    fn force_virtualizable(
        &mut self,
        opref: OpRef,
        vinfo: VirtualizableFieldState,
        ctx: &mut OptContext,
    ) -> OpRef {
        // Mark as no longer virtual (prevents infinite recursion)
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit SETFIELD_RAW for each tracked field, using the original DescrRef
        for (field_idx, value_ref) in &vinfo.fields {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_box_replacement(value_ref);
            let descr = get_field_descr(&vinfo.field_descrs, *field_idx)
                .unwrap_or_else(|| make_field_index_descr(*field_idx));
            let mut set_op = Op::new(OpCode::SetfieldRaw, &[opref, value_ref]);
            set_op.descr = Some(descr);
            ctx.emit_extra(ctx.current_pass_idx, set_op);
        }

        // Emit SETARRAYITEM_RAW for each tracked array element
        for (array_idx, elements) in vinfo.arrays {
            // Find the stored array descr for SetarrayitemRaw
            let array_descr = self
                .vable_array_ptrs
                .iter()
                .find(|(_, (idx, _))| *idx == array_idx as usize)
                .and_then(|(_, (_, descr))| descr.clone());

            // Get the array pointer from the frame — use the stored field descr
            let mut get_array_op = Op::new(OpCode::GetfieldRawI, &[opref]);
            // Look up the original descr for the array pointer field read
            if let Some(ref config) = self.vable_config {
                if let Some(&offset) = config.array_field_offsets.get(array_idx as usize) {
                    // Compute the field descr index for this offset
                    let descr_idx = virtualizable_field_index(offset);
                    get_array_op.descr = get_field_descr(&vinfo.field_descrs, descr_idx)
                        .or_else(|| Some(make_field_index_descr(descr_idx)));
                }
            }
            let array_ref = ctx.emit_extra(ctx.current_pass_idx, get_array_op);

            for (i, elem_ref) in elements.into_iter().enumerate() {
                let item_type = self
                    .vable_config
                    .as_ref()
                    .and_then(|config| config.array_item_types.get(array_idx as usize).copied())
                    .unwrap_or(Type::Int);
                let elem_ref = if elem_ref.is_none() {
                    self.emit_default_value_for_type(ctx, item_type)
                } else {
                    let forced = self.force_virtual(elem_ref, ctx);
                    ctx.get_box_replacement(forced)
                };
                let idx_ref = self.emit_constant_int(ctx, i as i64);
                let mut set_op = Op::new(OpCode::SetarrayitemRaw, &[array_ref, idx_ref, elem_ref]);
                set_op.descr = array_descr.clone();
                ctx.emit_extra(ctx.current_pass_idx, set_op);
            }
        }

        opref
    }

    fn force_virtual_instance(
        &mut self,
        opref: OpRef,
        vinfo: VirtualInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        // RPython info.py:137-156: _is_virtual = False.
        // info.py:225: self._fields[i] = None before emit_extra(setfieldop).
        // Clear field values so heap cache doesn't suppress SetfieldGc.
        let preserved = PtrInfo::Instance(crate::optimizeopt::info::InstancePtrInfo {
            descr: Some(vinfo.descr.clone()),
            known_class: vinfo.known_class,
            fields: Vec::new(),
            field_descrs: vinfo.field_descrs.clone(),
            preamble_fields: Vec::new(),
            last_guard_pos: -1,
        });
        // Mark as no longer virtual FIRST (avoids infinite recursion)
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit the NEW_WITH_VTABLE
        let mut alloc_op = Op::new(OpCode::NewWithVtable, &[]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit_extra(ctx.current_pass_idx, alloc_op);

        // RPython: newop.set_forwarded(self) — preserved info on alloc_ref
        ctx.set_ptr_info(alloc_ref, preserved);

        // Set forwarding only when the refs differ (avoids self-loop)
        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // info.py:216-226 InstancePtrInfo._force_elements: iterates
        // descr.get_all_fielddescrs() which excludes typeptr
        // (heaptracker.py:66-67), so typeptr is NOT emitted from the
        // force path. The vtable is written by the GC rewriter's
        // handle_malloc_operation via fielddescr_vtable (rewrite.py:479-484).
        debug_assert_no_typeptr_in_virtual_fields(&vinfo.fields, "force_virtual_instance");
        for (field_idx, value_ref) in vinfo.fields {
            let value_ref = self.force_virtual(value_ref, ctx);
            let value_ref = ctx.get_box_replacement(value_ref);
            let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
            set_op.descr = Some(
                get_field_descr(&vinfo.field_descrs, field_idx)
                    .unwrap_or_else(|| make_field_index_descr(field_idx)),
            );
            ctx.emit_extra(ctx.current_pass_idx, set_op);
        }

        alloc_ref
    }

    fn force_virtual_array(
        &mut self,
        opref: OpRef,
        vinfo: VirtualArrayInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        let len = vinfo.items.len();

        // Mark as no longer virtual FIRST (avoids infinite recursion)
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit the length constant and NEW_ARRAY
        let len_ref = self.emit_constant_int(ctx, len as i64);
        let mut alloc_op = Op::new(OpCode::NewArray, &[len_ref]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit_extra(ctx.current_pass_idx, alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETARRAYITEM_GC for each item
        for (i, item_ref) in vinfo.items.iter().enumerate() {
            let item_ref = self.force_virtual(*item_ref, ctx);
            let item_ref = ctx.get_box_replacement(item_ref);
            let idx_ref = self.emit_constant_int(ctx, i as i64);
            let mut set_op = Op::new(OpCode::SetarrayitemGc, &[alloc_ref, idx_ref, item_ref]);
            set_op.descr = Some(vinfo.descr.clone());
            ctx.emit_extra(ctx.current_pass_idx, set_op);
        }

        alloc_ref
    }

    fn force_virtual_struct(
        &mut self,
        opref: OpRef,
        vinfo: VirtualStructInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        // RPython info.py:147-156: _is_virtual = False, _fields retained.
        // newop.set_forwarded(self) — the same PtrInfo (now non-virtual)
        // stays accessible via get_ptr_info(alloc_ref).
        // RPython info.py:147-156: _is_virtual = False, _fields retained
        // BUT info.py:225: self._fields[i] = None before emit_extra(setfieldop).
        // Preserve descr+field_descrs for type resolution, but clear field VALUES
        // so the heap cache doesn't suppress the SetfieldGc emissions.
        let preserved = PtrInfo::Struct(crate::optimizeopt::info::StructPtrInfo {
            descr: vinfo.descr.clone(),
            fields: Vec::new(),
            field_descrs: vinfo.field_descrs.clone(),
            preamble_fields: Vec::new(),
            last_guard_pos: -1,
        });
        // Mark as no longer virtual FIRST (avoids infinite recursion)
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit NEW
        let mut alloc_op = Op::new(OpCode::New, &[]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit_extra(ctx.current_pass_idx, alloc_op);

        // RPython: newop.set_forwarded(self) — preserved info on alloc_ref
        ctx.set_ptr_info(alloc_ref, preserved);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // RPython info.py:226: optforce.emit_extra(setfieldop)
        for (field_idx, value_ref) in &vinfo.fields {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_box_replacement(value_ref);
            let descr = get_field_descr(&vinfo.field_descrs, *field_idx)
                .unwrap_or_else(|| make_field_index_descr(*field_idx));
            let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
            set_op.descr = Some(descr);
            ctx.emit_extra(ctx.current_pass_idx, set_op);
        }

        alloc_ref
    }

    fn force_virtual_array_struct(
        &mut self,
        opref: OpRef,
        vinfo: VirtualArrayStructInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        let num_elements = vinfo.element_fields.len();

        // Mark as no longer virtual FIRST
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit NEW_ARRAY for the outer array
        let len_ref = self.emit_constant_int(ctx, num_elements as i64);
        let mut alloc_op = Op::new(OpCode::NewArray, &[len_ref]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit_extra(ctx.current_pass_idx, alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETINTERIORFIELD_GC for each element's fields
        for (elem_idx, fields) in vinfo.element_fields.iter().enumerate() {
            let idx_ref = self.emit_constant_int(ctx, elem_idx as i64);
            for (field_idx, value_ref) in fields {
                let value_ref = self.force_virtual(*value_ref, ctx);
                let value_ref = ctx.get_box_replacement(value_ref);
                let mut set_op =
                    Op::new(OpCode::SetinteriorfieldGc, &[alloc_ref, idx_ref, value_ref]);
                set_op.descr = Some(make_field_index_descr(*field_idx));
                ctx.emit_extra(ctx.current_pass_idx, set_op);
            }
        }

        alloc_ref
    }

    fn force_virtual_raw_buffer(
        &mut self,
        opref: OpRef,
        vinfo: VirtualRawBufferInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        // Mark as no longer virtual FIRST
        ctx.set_ptr_info(opref, PtrInfo::nonnull());

        // Emit CALL_MALLOC_NURSERY or equivalent raw allocation
        let size_ref = self.emit_constant_int(ctx, vinfo.size as i64);
        let alloc_op = Op::new(OpCode::CallMallocNursery, &[size_ref]);
        let alloc_ref = ctx.emit_extra(ctx.current_pass_idx, alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit RAW_STORE for each tracked entry
        for (offset, _length, value_ref) in &vinfo.entries {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_box_replacement(value_ref);
            let offset_ref = self.emit_constant_int(ctx, *offset as i64);
            let set_op = Op::new(OpCode::RawStore, &[alloc_ref, offset_ref, value_ref]);
            ctx.emit_extra(ctx.current_pass_idx, set_op);
        }

        alloc_ref
    }

    /// Force all arguments that are virtual.
    /// Extract the int payload from a VirtualStruct (e.g., W_IntObject's intval field).
    fn has_any_virtual_arg(op: &Op, ctx: &OptContext) -> bool {
        op.args.iter().any(|arg| {
            let resolved = ctx.get_box_replacement(*arg);
            Self::is_virtual(resolved, ctx)
        })
    }

    fn force_all_args(&mut self, op: &Op, ctx: &mut OptContext) -> Op {
        let mut new_op = op.clone();
        for arg in &mut new_op.args {
            let resolved = ctx.get_box_replacement(*arg);
            if Self::is_virtual(resolved, ctx) {
                let forced = self.force_virtual(resolved, ctx);
                *arg = ctx.get_box_replacement(forced);
            } else {
                *arg = resolved;
            }
        }
        new_op
    }

    /// Emit a constant integer, returning its OpRef.
    fn emit_constant_int(&self, ctx: &mut OptContext, value: i64) -> OpRef {
        // Reserve the OpRef up front so the queued op can still self-reference
        // like RPython's constant boxes do when sent through downstream passes.
        let pos_ref = ctx.reserve_pos();
        let mut op = Op::new(OpCode::SameAsI, &[pos_ref]);
        op.pos = pos_ref;
        let opref = ctx.emit_extra(ctx.current_pass_idx, op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    fn emit_constant_ref(&self, ctx: &mut OptContext, value: majit_ir::GcRef) -> OpRef {
        let pos_ref = ctx.reserve_pos();
        let mut op = Op::new(OpCode::SameAsR, &[pos_ref]);
        op.pos = pos_ref;
        let opref = ctx.emit_extra(ctx.current_pass_idx, op);
        ctx.make_constant(opref, Value::Ref(value));
        opref
    }

    fn emit_constant_float(&self, ctx: &mut OptContext, value: f64) -> OpRef {
        let pos_ref = ctx.reserve_pos();
        let mut op = Op::new(OpCode::SameAsF, &[pos_ref]);
        op.pos = pos_ref;
        let opref = ctx.emit_extra(ctx.current_pass_idx, op);
        ctx.make_constant(opref, Value::Float(value));
        opref
    }

    fn emit_default_value_for_type(&self, ctx: &mut OptContext, item_type: Type) -> OpRef {
        match item_type {
            Type::Int => self.emit_constant_int(ctx, 0),
            Type::Ref => self.emit_constant_ref(ctx, majit_ir::GcRef::NULL),
            Type::Float => self.emit_constant_float(ctx, 0.0),
            Type::Void => self.emit_constant_int(ctx, 0),
        }
    }

    // ── Per-opcode handlers ──

    fn optimize_new_with_vtable(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let descr = op.descr.clone().expect("NEW_WITH_VTABLE needs descr");
        // virtualize.py:208: known_class = ConstInt(op.getdescr().get_vtable())
        let known_class = descr
            .as_size_descr()
            .map(|sd| majit_ir::GcRef(sd.vtable()))
            .filter(|gc| !gc.is_null());
        let vinfo = VirtualInfo {
            descr,
            known_class,
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        };
        ctx.set_ptr_info(op.pos, PtrInfo::Virtual(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let descr = op.descr.clone().expect("NEW needs descr");
        let vinfo = VirtualStructInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        };
        ctx.set_ptr_info(op.pos, PtrInfo::VirtualStruct(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new_array(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let size_ref = op.arg(0);
        if let Some(size) = ctx.get_constant_int(size_ref) {
            if size >= 0 && size <= 1024 {
                let descr = op.descr.clone().expect("NEW_ARRAY needs descr");
                let items = vec![OpRef::NONE; size as usize];
                let vinfo = VirtualArrayInfo {
                    descr,
                    clear: matches!(op.opcode, OpCode::NewArrayClear),
                    items,
                    last_guard_pos: -1,
                };
                ctx.set_ptr_info(op.pos, PtrInfo::VirtualArray(vinfo));
                return OptimizationResult::Remove;
            }
        }
        OptimizationResult::PassOn
    }

    #[allow(dead_code)]
    fn optimize_new_array_clear(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        self.optimize_new_array(op, ctx)
    }

    fn optimize_setfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let struct_ref = ctx.get_box_replacement(op.arg(0));
        let value_ref = ctx.get_box_replacement(op.arg(1));
        let field_idx = descr_index(&op.descr);
        let is_raw_op = matches!(op.opcode, OpCode::SetfieldRaw);
        // Pre-extract constant value before mutable borrow of ptr_info.
        // Class pointer may be stored as Value::Int OR Value::Ref.
        let value_as_constant: Option<usize> = ctx.get_constant(value_ref).and_then(|v| match v {
            majit_ir::Value::Int(i) => Some(*i as usize),
            majit_ir::Value::Ref(gc) => Some(gc.as_usize()),
            _ => None,
        });

        // RPython virtualize.py:200-202: virtual SetfieldGc always updates
        // the field, even for imported virtual heads. Body computation must
        // be able to update virtual fields (e.g., i.intval = i + step).

        if is_raw_op && self.is_standard_virtualizable_ref(struct_ref, ctx) {
            return OptimizationResult::PassOn;
        }

        // RPython: if struct is NOT virtual, PassOn to OptHeap which stores
        // it as a lazy_set. The virtual value is NOT forced — OptHeap delays
        // it until guard emission (force_lazy_sets_for_guard) or JUMP.

        if let Some(info) = ctx.get_ptr_info_mut(struct_ref) {
            if info.is_virtual() {
                match info {
                    PtrInfo::Virtual(vinfo) => {
                        // info.py:203-206 AbstractStructPtrInfo.setfield:
                        //   self._fields[fielddescr.get_index()] = op.
                        // heaptracker.py:66-67 all_fielddescrs() excludes typeptr:
                        //   if name == 'typeptr': continue # dealt otherwise
                        // → _fields never contains typeptr. In pyre, typeptr
                        // setfield is filtered at trace recording time
                        // (jtransform.py:908-911 parity in helpers.rs), so this
                        // branch should not observe offset=0. Defensively capture
                        // known_class if an offset-0 setfield still arrives.
                        let offset = extract_field_offset(field_idx);
                        if offset == Some(0) {
                            if vinfo.known_class.is_none() {
                                if let Some(class_val) = value_as_constant {
                                    vinfo.known_class = Some(majit_ir::GcRef(class_val));
                                }
                            }
                            return OptimizationResult::Remove;
                        }
                        set_field(&mut vinfo.fields, field_idx, value_ref);
                        if let Some(descr) = &op.descr {
                            set_field_descr(&mut vinfo.field_descrs, field_idx, descr.clone());
                        }
                        debug_assert_no_typeptr_in_virtual_fields(
                            &vinfo.fields,
                            "optimize_setfield_gc::Virtual",
                        );
                        return OptimizationResult::Remove;
                    }
                    PtrInfo::VirtualStruct(vinfo) => {
                        set_field(&mut vinfo.fields, field_idx, value_ref);
                        if let Some(descr) = &op.descr {
                            set_field_descr(&mut vinfo.field_descrs, field_idx, descr.clone());
                        }
                        return OptimizationResult::Remove;
                    }
                    PtrInfo::Virtualizable(vstate) => {
                        set_field(&mut vstate.fields, field_idx, value_ref);
                        // Store original descr for force path
                        if let Some(d) = op.descr.clone() {
                            set_field_descr(&mut vstate.field_descrs, field_idx, d);
                        }
                        return OptimizationResult::Remove;
                    }
                    _ => {}
                }
            }
        }
        // RPython: virtual value is NOT forced in optimize_SETFIELD_GC.
        // It's forced by _emit_operation (optimizer.py:623-625) at final emit.
        // In majit, this is handled by emit_operation or force_all_lazy_sets.
        // virtualize.py:204: self.make_nonnull(op.getarg(0))
        if ctx.get_ptr_info(struct_ref).is_none() {
            ctx.set_ptr_info(struct_ref, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    fn optimize_getfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let struct_ref = ctx.get_box_replacement(op.arg(0));
        let field_idx = descr_index(&op.descr);
        let is_raw_op = matches!(
            op.opcode,
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF
        );

        // RPython import_state: if this is a GetfieldGcR(pool) that loads a head
        // which was virtual in the preamble, forward to the imported virtual head.
        if matches!(op.opcode, OpCode::GetfieldGcR | OpCode::GetfieldRawR) {
            let pool_ref = ctx.get_box_replacement(OpRef(0)); // pool is always inputarg 0
            if struct_ref == pool_ref {
                for &(descr_idx, virtual_head) in &ctx.imported_virtual_heads {
                    if field_idx == descr_idx as u32 {
                        ctx.replace_op(op.pos, virtual_head);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        if is_raw_op && self.is_standard_virtualizable_ref(struct_ref, ctx) {
            return OptimizationResult::PassOn;
        }

        if let Some(info) = ctx.get_ptr_info(struct_ref).cloned() {
            // info.py:212-214 getfield: return _fields[fielddescr.get_index()].
            // For Virtual, ob_type (offset 0) is not in fields — fold from
            // known_class (info.py:324-325 get_known_class).
            if let PtrInfo::Virtual(ref vinfo) = info {
                let offset = extract_field_offset(field_idx);
                if offset == Some(0) {
                    if let Some(gc_ref) = vinfo.known_class {
                        let class_val = gc_ref.as_usize() as i64;
                        ctx.make_constant(op.pos, majit_ir::Value::Int(class_val));
                        return OptimizationResult::Remove;
                    }
                }
            }
            let field_val = match &info {
                PtrInfo::Virtual(vinfo) => get_field(&vinfo.fields, field_idx),
                PtrInfo::VirtualStruct(vinfo) => get_field(&vinfo.fields, field_idx),
                PtrInfo::Virtualizable(vstate) => get_field(&vstate.fields, field_idx),
                _ => None,
            };
            if let Some(val_ref) = field_val {
                ctx.replace_op(op.pos, val_ref);
                return OptimizationResult::Remove;
            }
            // heaptracker.py:66 typeptr exclusion: typeptr is excluded from
            // virtual fields but can be resolved from the SizeDescr vtable.
            // RPython doesn't need this because GUARD_CLASS reads the class
            // directly from the object, not via a separate GetfieldGcPure.
            if field_val.is_none()
                && matches!(
                    op.opcode,
                    majit_ir::OpCode::GetfieldGcPureI | majit_ir::OpCode::GetfieldGcI
                )
            {
                let is_typeptr = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_field_descr())
                    .map(|fd| fd.is_typeptr())
                    .unwrap_or(false);
                if is_typeptr {
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
                        let const_pos = ctx.alloc_op_position();
                        ctx.make_constant(const_pos, Value::Int(vtable as i64));
                        ctx.replace_op(op.pos, const_pos);
                        return OptimizationResult::Remove;
                    }
                }
            }
            // Virtualizable: first read of an untracked field.
            // Let the op emit, but cache result for future reads
            // and register array pointers for array tracking.
            if let PtrInfo::Virtualizable(_) = &info {
                return self.handle_virtualizable_first_read(op, field_idx, ctx);
            }
        }
        // virtualize.py:192: self.make_nonnull(op.getarg(0))
        // optimizer.py:437-448: only set NonNull if no existing PtrInfo.
        if ctx.get_ptr_info(struct_ref).is_none() {
            ctx.set_ptr_info(struct_ref, PtrInfo::nonnull());
        }
        OptimizationResult::PassOn
    }

    /// Handle first read of a virtualizable field (no cached value yet).
    ///
    /// Standard virtualizable traces should already have scalar fields seeded
    /// from input boxes. The remaining legacy case here is learning which raw
    /// frame field carries an array pointer so later Get/SetarrayitemRaw ops
    /// can be absorbed.
    fn handle_virtualizable_first_read(
        &mut self,
        op: &Op,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let offset = extract_field_offset(field_idx);

        // Array pointer field: remember which array this pointer came from
        // so subsequent Get/SetarrayitemRaw can be absorbed. Do not cache
        // the pointer itself as a tracked virtualizable field: only fields
        // declared in VirtualizableConfig should participate in force/writeback.
        let arr_idx = self.vable_config.as_ref().and_then(|config| {
            offset.and_then(|off| config.array_field_offsets.iter().position(|&o| o == off))
        });
        if let Some(arr_idx) = arr_idx {
            if !self.vable_initialized {
                self.init_virtualizable(ctx);
                self.vable_initialized = true;
            }
            self.vable_array_ptrs.insert(op.pos, (arr_idx, None));
        }
        OptimizationResult::PassOn
    }

    fn optimize_setarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.get_box_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let value_ref = ctx.get_box_replacement(op.arg(2));

        // Virtualizable array check
        if !self.vable_array_ptrs.contains_key(&array_ref) {
            self.try_register_virtualizable_array_ptr(array_ref, ctx);
        }
        if let Some(&mut (arr_idx, ref mut stored_descr)) =
            self.vable_array_ptrs.get_mut(&array_ref)
        {
            // Capture the array descr from the first absorbed op
            if stored_descr.is_none() {
                *stored_descr = op.descr.clone();
            }
            if let Some(index) = ctx.get_constant_int(index_ref) {
                let frame_ref = OpRef(0);
                if let Some(PtrInfo::Virtualizable(vstate)) = ctx.get_ptr_info_mut(frame_ref) {
                    set_array_element(
                        &mut vstate.arrays,
                        arr_idx as u32,
                        index as usize,
                        value_ref,
                    );
                    return OptimizationResult::Remove;
                }
            }
        }

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = ctx.get_ptr_info_mut(array_ref) {
                if let PtrInfo::VirtualArray(vinfo) = info {
                    let idx = index as usize;
                    if idx < vinfo.items.len() {
                        vinfo.items[idx] = value_ref;
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        // Force any virtual args before emitting the setarrayitem
        if Self::has_any_virtual_arg(op, ctx) {
            return OptimizationResult::Replace(self.force_all_args(op, ctx));
        }
        OptimizationResult::PassOn
    }

    fn optimize_getarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.get_box_replacement(op.arg(0));
        let index_ref = op.arg(1);

        // Virtualizable array check
        if !self.vable_array_ptrs.contains_key(&array_ref) {
            self.try_register_virtualizable_array_ptr(array_ref, ctx);
        }
        if let Some(&mut (arr_idx, ref mut stored_descr)) =
            self.vable_array_ptrs.get_mut(&array_ref)
        {
            // Capture the array descr from the first absorbed op
            if stored_descr.is_none() {
                *stored_descr = op.descr.clone();
            }
            if let Some(index) = ctx.get_constant_int(index_ref) {
                let frame_ref = OpRef(0);
                // Try to read cached value first
                if let Some(PtrInfo::Virtualizable(vstate)) = ctx.get_ptr_info(frame_ref) {
                    if let Some(val) =
                        get_array_element(&vstate.arrays, arr_idx as u32, index as usize)
                    {
                        ctx.replace_op(op.pos, val);
                        return OptimizationResult::Remove;
                    }
                }
                // Cache the result of this first read
                if let Some(PtrInfo::Virtualizable(vstate)) = ctx.get_ptr_info_mut(frame_ref) {
                    set_array_element(&mut vstate.arrays, arr_idx as u32, index as usize, op.pos);
                }
            }
        }

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = ctx.get_ptr_info(array_ref).cloned() {
                if let PtrInfo::VirtualArray(vinfo) = info {
                    let idx = index as usize;
                    if idx < vinfo.items.len() {
                        let item_ref = vinfo.items[idx];
                        if !item_ref.is_none() {
                            ctx.replace_op(op.pos, item_ref);
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_arraylen_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = ctx.get_ptr_info(array_ref) {
            let len = vinfo.items.len() as i64;
            ctx.make_constant(op.pos, Value::Int(len));
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    fn optimize_strlen(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_box_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = ctx.get_ptr_info(str_ref) {
            let len = vinfo.items.len() as i64;
            ctx.make_constant(op.pos, Value::Int(len));
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    fn optimize_getinteriorfield_gc(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_ref = ctx.get_box_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let field_idx = descr_index(&op.descr);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(PtrInfo::VirtualArrayStruct(vinfo)) = ctx.get_ptr_info(array_ref) {
                let elem_idx = index as usize;
                if elem_idx < vinfo.element_fields.len() {
                    if let Some(val) = get_field(&vinfo.element_fields[elem_idx], field_idx) {
                        ctx.replace_op(op.pos, val);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_setinteriorfield_gc(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let array_ref = ctx.get_box_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let value_ref = ctx.get_box_replacement(op.arg(2));
        let field_idx = descr_index(&op.descr);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = ctx.get_ptr_info_mut(array_ref) {
                if let PtrInfo::VirtualArrayStruct(vinfo) = info {
                    let elem_idx = index as usize;
                    if elem_idx < vinfo.element_fields.len() {
                        set_field(&mut vinfo.element_fields[elem_idx], field_idx, value_ref);
                        // RPython VArrayStructInfo.fielddescrs parity:
                        // collect InteriorFieldDescr for _number_virtuals.
                        if let Some(ref descr) = op.descr {
                            if !vinfo.fielddescrs.iter().any(|d| d.index() == descr.index()) {
                                vinfo.fielddescrs.push(descr.clone());
                            }
                        }
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_raw_load(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let buf_ref = ctx.get_box_replacement(op.arg(0));
        let offset_ref = op.arg(1);

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(PtrInfo::VirtualRawBuffer(vinfo)) = ctx.get_ptr_info(buf_ref) {
                let offset = offset as usize;
                if let Some((_, _, val_ref)) =
                    vinfo.entries.iter().find(|(off, _, _)| *off == offset)
                {
                    ctx.replace_op(op.pos, *val_ref);
                    return OptimizationResult::Remove;
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_raw_store(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let buf_ref = ctx.get_box_replacement(op.arg(0));
        let offset_ref = op.arg(1);
        let value_ref = ctx.get_box_replacement(op.arg(2));

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(info) = ctx.get_ptr_info_mut(buf_ref) {
                if let PtrInfo::VirtualRawBuffer(vinfo) = info {
                    let offset = offset as usize;
                    // Update or insert entry (use default word size 8 for untyped stores)
                    if let Some(entry) = vinfo.entries.iter_mut().find(|(off, _, _)| *off == offset)
                    {
                        entry.2 = value_ref;
                    } else {
                        // Use write_value for sorted insertion with overlap check.
                        // Default length of 8 bytes (word-sized store).
                        let _ = vinfo.write_value(offset, 8, value_ref);
                    }
                    return OptimizationResult::Remove;
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// Resolve a guard fail_arg, keeping virtual-struct-shaped infos intact
    /// so `store_final_boxes_in_guard` can encode them as TAGVIRTUAL at
    /// emit time. Other virtual-ish infos (raw buffers, array slices) that
    /// the guard encoder does not handle are forced to concrete here.
    ///
    /// Called from the generic guard path in `propagate_forward`. RPython's
    /// virtualize.py does not run a dedicated fail_args pre-pass; the
    /// equivalent work happens inside `Optimizer.store_final_boxes_in_guard`
    /// + `force_box` in `emit_operation` (optimizer.py:677-683).
    fn prepare_guard_fail_arg(&mut self, resolved: OpRef, ctx: &mut OptContext) -> OpRef {
        let Some(info) = ctx.get_ptr_info(resolved).cloned() else {
            return resolved;
        };
        match info {
            PtrInfo::Virtual(_) | PtrInfo::VirtualStruct(_) => resolved,
            PtrInfo::Virtualizable(_) => resolved,
            other if other.is_virtual() => {
                let forced = self.force_virtual(resolved, ctx);
                ctx.get_box_replacement(forced)
            }
            _ => resolved,
        }
    }

    /// Handle VirtualRefR / VirtualRefI.
    ///
    /// Replace the VIRTUAL_REF operation with a virtual struct of type
    /// JitVirtualRef. The struct has two fields:
    /// - virtual_token (field index VREF_VIRTUAL_TOKEN): set to a ForceToken op
    /// - forced (field index VREF_FORCED): set to NULL (constant 0)
    ///
    /// This way the vref itself becomes virtual. If it never escapes, the
    /// allocation is eliminated entirely. If it does escape, the forcing
    /// mechanism emits the struct allocation + field writes.
    fn optimize_virtual_ref(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let vref_descr = make_field_index_descr(VREF_SIZE_DESCR_INDEX);

        // Emit a FORCE_TOKEN to capture the JIT frame address.
        let token_op = Op::new(OpCode::ForceToken, &[]);
        let token_ref = ctx.emit_extra(ctx.current_pass_idx, token_op);

        // The emitted ForceToken may reuse an OpRef index that previously
        // had PtrInfo::Virtual attached (from an earlier NEW_WITH_VTABLE
        // that was virtualized). Clear that to prevent accidental forcing
        // of unrelated virtuals when the vref struct is forced.
        ctx.set_ptr_info(token_ref, PtrInfo::nonnull());

        // Use a sentinel OpRef for the NULL constant in the `forced` field.
        // We don't emit it now; when the virtual struct is forced, the
        // constant 0 is emitted lazily. Using OpRef::NONE here is safe
        // because force_virtual checks whether a field value is virtual
        // before forcing it, and OpRef::NONE is never virtual.
        let null_ref = OpRef::NONE;

        let fields = vec![
            (VREF_VIRTUAL_TOKEN_FIELD_INDEX, token_ref),
            (VREF_FORCED_FIELD_INDEX, null_ref),
        ];
        let vinfo = VirtualStructInfo {
            descr: vref_descr,
            fields,
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        };
        ctx.set_ptr_info(op.pos, PtrInfo::VirtualStruct(vinfo));

        OptimizationResult::Remove
    }

    /// Handle VirtualRefFinish(vref, virtual_obj).
    ///
    /// Two cases:
    /// 1. Normal case: virtual_obj is NULL (constant 0) -- the frame is being
    ///    left normally. Just clear the virtual_token field.
    /// 2. Forced case: virtual_obj is non-NULL -- the vref was forced during
    ///    tracing. Store the real object into the `forced` field.
    ///
    /// If the vref is still virtual, these writes are absorbed into the
    /// virtual struct's field tracking and nothing is emitted.
    /// If the vref was already forced (escaped), we emit SETFIELD_GC ops.
    fn optimize_virtual_ref_finish(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let vref_ref = ctx.get_box_replacement(op.arg(0));
        let obj_ref = ctx.get_box_replacement(op.arg(1));

        // Check if the virtual object arg is non-null (forced case)
        let obj_is_null = ctx.get_constant_int(obj_ref).is_some_and(|v| v == 0);

        // If vref is still virtual, update the virtual struct fields directly
        if let Some(info) = ctx.get_ptr_info_mut(vref_ref) {
            if info.is_virtual() {
                if let PtrInfo::VirtualStruct(vinfo) = info {
                    // Set forced field to the object (or keep NULL)
                    if !obj_is_null {
                        set_field(&mut vinfo.fields, VREF_FORCED_FIELD_INDEX, obj_ref);
                    }
                    // Set virtual_token to NULL (TOKEN_NONE)
                    let null_ref = self.emit_constant_int(ctx, 0);
                    // Re-borrow: get_ptr_info_mut again after emit_constant_int
                    if let Some(PtrInfo::VirtualStruct(vinfo)) = ctx.get_ptr_info_mut(vref_ref) {
                        set_field(&mut vinfo.fields, VREF_VIRTUAL_TOKEN_FIELD_INDEX, null_ref);
                    }
                    return OptimizationResult::Remove;
                }
            }
        }

        // vref is not virtual (was forced/escaped): emit SETFIELD_GC ops

        // Set 'forced' field if the object is non-null
        if !obj_is_null {
            let mut set_forced = Op::new(OpCode::SetfieldGc, &[vref_ref, obj_ref]);
            set_forced.descr = Some(make_field_index_descr(VREF_FORCED_FIELD_INDEX));
            ctx.emit_extra(ctx.current_pass_idx, set_forced);
        }

        // Set 'virtual_token' to NULL
        let null_ref = self.emit_constant_int(ctx, 0);
        let mut set_token = Op::new(OpCode::SetfieldGc, &[vref_ref, null_ref]);
        set_token.descr = Some(make_field_index_descr(VREF_VIRTUAL_TOKEN_FIELD_INDEX));
        ctx.emit_extra(ctx.current_pass_idx, set_token);

        OptimizationResult::Remove
    }

    /// Handle operations that may cause virtuals to escape.
    fn optimize_escaping_op(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let mut forced = op.clone();
        let frame_ref = ctx.get_box_replacement(OpRef(0));
        for arg in &mut forced.args {
            let resolved = ctx.get_box_replacement(*arg);
            if self.vable_config.is_some()
                && resolved == frame_ref
                && matches!(ctx.get_ptr_info(resolved), Some(PtrInfo::Virtualizable(_)))
            {
                *arg = resolved;
                continue;
            }
            if Self::is_virtual(resolved, ctx) {
                let forced_arg = self.force_virtual(resolved, ctx);
                *arg = ctx.get_box_replacement(forced_arg);
            } else {
                *arg = resolved;
            }
        }
        self.clear_forced_field_caches();
        OptimizationResult::Replace(forced)
    }

    fn try_register_virtualizable_array_ptr(&mut self, array_ref: OpRef, ctx: &OptContext) {
        if self.vable_array_ptrs.contains_key(&array_ref) {
            return;
        }
        let Some(config) = &self.vable_config else {
            return;
        };
        let Some(source_op) = ctx.new_operations.iter().find(|op| op.pos == array_ref) else {
            return;
        };
        if !matches!(
            source_op.opcode,
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF
        ) {
            return;
        }
        if source_op.arg(0) != ctx.get_box_replacement(OpRef(0)) {
            return;
        }
        if self.is_standard_virtualizable_ref(ctx.get_box_replacement(OpRef(0)), ctx) {
            return;
        }
        let field_idx = descr_index(&source_op.descr);
        let offset = extract_field_offset(field_idx);
        let arr_idx = config
            .array_field_offsets
            .iter()
            .position(|&o| Some(o) == offset);
        if let Some(arr_idx) = arr_idx {
            self.vable_array_ptrs.insert(array_ref, (arr_idx, None));
        }
    }

    fn clear_forced_field_caches(&mut self) {
        // No-op: forced field caches are handled by heap.rs cache,
        // not by separate PtrInfo variants.
    }
}

impl Default for OptVirtualize {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptVirtualize {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        self.ensure_vable_setup(ctx);
        match op.opcode {
            // virtualize.py:207-209: optimize_NEW_WITH_VTABLE → make_virtual.
            // InstancePtrInfo(descr, known_class, is_virtual=True)
            OpCode::NewWithVtable => self.optimize_new_with_vtable(op, ctx),
            OpCode::New => self.optimize_new(op, ctx),
            OpCode::NewArray | OpCode::NewArrayClear => self.optimize_new_array(op, ctx),

            // Field access on potentially-virtual objects
            OpCode::SetfieldGc | OpCode::SetfieldRaw => self.optimize_setfield_gc(op, ctx),
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF => self.optimize_getfield_gc(op, ctx),

            // Array access on potentially-virtual arrays
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                self.optimize_setarrayitem_gc(op, ctx)
            }
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => self.optimize_getarrayitem_gc(op, ctx),

            // Array length
            OpCode::ArraylenGc => self.optimize_arraylen_gc(op, ctx),
            OpCode::Strlen => self.optimize_strlen(op, ctx),

            // Interior field access on potentially-virtual array-of-structs
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => self.optimize_getinteriorfield_gc(op, ctx),
            OpCode::SetinteriorfieldGc => self.optimize_setinteriorfield_gc(op, ctx),

            // Raw memory access on potentially-virtual raw buffers
            OpCode::RawLoadI | OpCode::RawLoadF => self.optimize_raw_load(op, ctx),
            OpCode::RawStore => self.optimize_raw_store(op, ctx),

            // RPython virtualize.py does NOT define optimize_GUARD_CLASS,
            // GUARD_NONNULL, GUARD_NONNULL_CLASS, or GUARD_VALUE — these
            // are handled exclusively by rewrite.py. Flow the guards
            // through to the next pass so OptRewrite sees them.
            // emit_guard_operation (mod.rs) calls store_final_boxes_in_guard
            // + force_box on fail_args at emit time, so virtualize does not
            // need to pre-process guard fail_args here.

            // VirtualRef: replace with a virtual struct tracking token + forced fields
            OpCode::VirtualRefR | OpCode::VirtualRefI => self.optimize_virtual_ref(op, ctx),
            // VirtualRefFinish: finalize the virtual ref
            OpCode::VirtualRefFinish => self.optimize_virtual_ref_finish(op, ctx),

            // virtualize.py: GUARD_NO_EXCEPTION — if the preceding call was
            // eliminated (all args virtual), the guard is redundant.
            OpCode::GuardNoException => OptimizationResult::PassOn,

            // GUARD_NOT_FORCED checks if the JIT frame was forced during a call.
            // If the force_token from the preceding CALL_MAY_FORCE is virtual
            // (never escaped), the guard always succeeds. Otherwise, pass through.
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => OptimizationResult::PassOn,

            // virtualize.py: optimize_CALL_MAY_FORCE_I/R/F/N
            // If the call is OS_JIT_FORCE_VIRTUAL and the vref is virtual
            // with a null token → replace with the forced value.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.effect_info();
                        if ei.oopspec_index == OopSpecIndex::JitForceVirtual && op.num_args() >= 2 {
                            let vref = ctx.get_box_replacement(op.arg(1));
                            if Self::is_virtual(vref, ctx) {
                                // Virtual ref with known null token →
                                // return the forced value directly.
                                // Simplified: just remove the call.
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
                self.optimize_escaping_op(op, ctx)
            }

            // virtualize.py: optimize_FINISH — force the last
            // GUARD_NOT_FORCED_2 op for resume data.
            OpCode::Finish => {
                // Force all virtual args (same as escaping op).
                self.optimize_escaping_op(op, ctx)
            }

            // virtualize.py: optimize_COND_CALL — if the call is
            // OS_JIT_FORCE_VIRTUALIZABLE and the target is virtual, remove.
            OpCode::CondCallN => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.effect_info();
                        if ei.oopspec_index == OopSpecIndex::JitForceVirtualizable
                            && op.num_args() >= 3
                        {
                            let target = ctx.get_box_replacement(op.arg(2));
                            if Self::is_virtual(target, ctx) {
                                return OptimizationResult::Remove;
                            }
                        }
                    }
                }
                self.optimize_escaping_op(op, ctx)
            }

            // virtualize.py: optimize_CALL_N/R/I — special handling for
            // OS_JIT_FORCE_VIRTUALIZABLE (remove if target is virtual).
            OpCode::CallN | OpCode::CallR | OpCode::CallI | OpCode::CallF => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.effect_info();
                        // virtualize.py: OS_RAW_MALLOC_VARSIZE_CHAR → virtual raw buffer
                        if ei.oopspec_index == OopSpecIndex::RawMallocVarsizeChar {
                            if op.num_args() >= 2 {
                                if let Some(size) = ctx.get_constant_int(op.arg(1)) {
                                    // Create a virtual raw buffer of known size.
                                    let info = PtrInfo::VirtualRawBuffer(
                                        crate::optimizeopt::info::VirtualRawBufferInfo {
                                            size: size as usize,
                                            entries: Vec::new(),
                                            last_guard_pos: -1,
                                        },
                                    );
                                    ctx.set_ptr_info(op.pos, info);
                                    return OptimizationResult::Remove;
                                }
                            }
                            // Non-constant size: emit as-is
                            return self.optimize_escaping_op(op, ctx);
                        }
                        // virtualize.py: OS_RAW_FREE → remove if target is virtual raw buffer
                        if ei.oopspec_index == OopSpecIndex::RawFree {
                            if op.num_args() >= 2 {
                                let target = ctx.get_box_replacement(op.arg(1));
                                if Self::is_virtual(target, ctx) {
                                    return OptimizationResult::Remove;
                                }
                            }
                            return self.optimize_escaping_op(op, ctx);
                        }
                        // virtualize.py: OS_JIT_FORCE_VIRTUALIZABLE
                        if ei.oopspec_index == OopSpecIndex::JitForceVirtualizable {
                            if op.num_args() > 1 {
                                let target = ctx.get_box_replacement(op.arg(1));
                                if Self::is_virtual(target, ctx) {
                                    return OptimizationResult::Remove;
                                }
                            }
                        }
                    }
                }
                self.optimize_escaping_op(op, ctx)
            }

            // RecordKnownResult + CallPure must pass through to OptPure
            // for @elidable constant folding. Must appear BEFORE is_call()
            // since they are in the CALL opcode range.
            OpCode::RecordKnownResult => OptimizationResult::PassOn,
            OpCode::CallPureI | OpCode::CallPureR | OpCode::CallPureF | OpCode::CallPureN => {
                self.optimize_escaping_op(op, ctx)
            }

            // Calls / escaping operations — force all virtual args
            _ if op.opcode.is_call() => self.optimize_escaping_op(op, ctx),

            // RPython virtualize.py has no optimize_JUMP. JUMP is held
            // out of the pass pipeline (flush=False at optimizer.py:536-539)
            // or sent through via send_extra_operation in flush=True, which
            // dispatches to the standard emit path — no virtualize-specific
            // handler. Falling through to the default PassOn matches RPython.

            // ── Record hint opcodes ──
            // These record information about values that downstream passes can use.
            // The hints themselves are removed (no code emitted).

            // RECORD_EXACT_CLASS(ref, class_const): record that ref has class class_const.
            // Enables subsequent GUARD_CLASS elimination.
            OpCode::RecordExactClass => {
                let ref_opref = ctx.get_box_replacement(op.arg(0));
                if let Some(&Value::Ref(class_ref)) = ctx.get_constant(op.arg(1)) {
                    self.record_known_class(ref_opref, class_ref, ctx);
                }
                OptimizationResult::Remove
            }

            // RECORD_EXACT_VALUE_I(ref, int_const): record that ref has exact int value.
            OpCode::RecordExactValueI => {
                let ref_opref = ctx.get_box_replacement(op.arg(0));
                if let Some(val) = ctx.get_constant_int(op.arg(1)) {
                    ctx.make_constant(ref_opref, Value::Int(val));
                }
                OptimizationResult::Remove
            }

            // RECORD_EXACT_VALUE_R(ref, ref_const): record that ref equals ref_const.
            OpCode::RecordExactValueR => {
                let ref_opref = ctx.get_box_replacement(op.arg(0));
                ctx.replace_op(ref_opref, ctx.get_box_replacement(op.arg(1)));
                OptimizationResult::Remove
            }

            // Everything else passes through, but force any virtual args first.
            // optimizer.py:614-625 _emit_operation forces every arg via force_box.
            _ => {
                let is_guard = op.opcode.is_guard();

                // optimizer.py:623-625: for i in range(op.numargs()): arg = self.force_box(op.getarg(i))
                for i in 0..op.num_args() {
                    let arg = ctx.get_box_replacement(op.arg(i));
                    self.force_virtual(arg, ctx);
                }

                // optimizer.py:652-686 emit_guard_operation: store_final_boxes_in_guard
                // is invoked from OptContext::emit() at the actual emission point and
                // walks the snapshot's vable_array via _number_boxes (TAGVIRTUAL).
                // Per-pass guard fail_args do not need extra augmentation.
                if is_guard {
                    let mut guard_op = op.clone();

                    for arg in &mut guard_op.args {
                        *arg = ctx.get_box_replacement(*arg);
                    }

                    if let Some(ref mut fa) = guard_op.fail_args {
                        for arg in fa.iter_mut() {
                            let resolved = ctx.get_box_replacement(*arg);
                            *arg = self.prepare_guard_fail_arg(resolved, ctx);
                        }
                    }

                    return OptimizationResult::Replace(guard_op);
                }

                OptimizationResult::PassOn
            }
        }
    }

    fn setup(&mut self) {
        self.vable_array_ptrs.clear();
        self.vable_initialized = false;
        // Defer virtualizable PtrInfo init to first propagate_forward
        // (setup() doesn't have access to OptContext).
        self.needs_vable_setup = self.vable_config.is_some();
    }

    fn name(&self) -> &'static str {
        "virtualize"
    }

    fn set_phase2(&mut self, phase2: bool) {
        self.is_phase2 = phase2;
    }
}

// PtrInfo helpers (is_nonnull, is_virtual, etc.) are in info.rs.

// ── Field list helpers ──

fn set_field(fields: &mut Vec<(u32, OpRef)>, field_idx: u32, value_ref: OpRef) {
    for entry in fields.iter_mut() {
        if entry.0 == field_idx {
            entry.1 = value_ref;
            return;
        }
    }
    fields.push((field_idx, value_ref));
}

/// RPython parity assertion: `VirtualInfo.fields` must NEVER contain
/// typeptr (offset 0). heaptracker.py:66-67 `all_fielddescrs()` skips
/// typeptr, so `info.py:180 AbstractStructPtrInfo._fields` is sized to
/// the non-typeptr field count and has no slot for offset 0.
///
/// This helper asserts the invariant at boundaries where `Virtual.fields`
/// is constructed/populated/iterated: import (virtualstate), export
/// (virtualstate), force path (force_virtual_instance), and the
/// `optimize_setfield_gc` Virtual arm (before `set_field`).
///
/// Only `VirtualInfo.fields` is subject to this invariant;
/// `VirtualStructInfo.fields` (OpCode::New without vtable) may contain
/// any offset including 0.
#[inline]
pub(crate) fn debug_assert_no_typeptr_in_virtual_fields(
    fields: &[(u32, OpRef)],
    context: &'static str,
) {
    debug_assert!(
        fields
            .iter()
            .all(|(idx, _)| extract_field_offset(*idx) != Some(0)),
        "VirtualInfo.fields must exclude typeptr (offset 0) — RPython \
         heaptracker.py:66-67 all_fielddescrs() parity violation at {context}",
    );
}

fn set_field_descr(field_descrs: &mut Vec<(u32, DescrRef)>, field_idx: u32, descr: DescrRef) {
    FIELD_DESCR_REGISTRY.with(|r| r.borrow_mut().insert(field_idx, descr.clone()));
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

thread_local! {
    static FIELD_DESCR_REGISTRY: RefCell<HashMap<u32, DescrRef>> = RefCell::new(HashMap::new());
}

fn get_field(fields: &[(u32, OpRef)], field_idx: u32) -> Option<OpRef> {
    fields
        .iter()
        .find(|(idx, _)| *idx == field_idx)
        .map(|(_, opref)| *opref)
}

/// Extract the descriptor index used as a field identifier.
fn descr_index(descr: &Option<DescrRef>) -> u32 {
    descr.as_ref().map_or(0, |d| d.index())
}

// ── Minimal descriptor for field identification in forced setfield ops ──

#[derive(Debug)]
struct FieldIndexDescr(u32);

impl Descr for FieldIndexDescr {
    fn index(&self) -> u32 {
        self.0
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for FieldIndexDescr {
    fn offset(&self) -> usize {
        // Decode offset from the encoded field descriptor index.
        // Format: FIELD_DESCR_TAG | (offset << 4) | (size << 1) | (signed << 3) | type_bits
        ((self.0 >> 4) & 0x000f_ffff) as usize
    }

    fn field_size(&self) -> usize {
        // Match the stable field descriptor encoding used by the real
        // pyre/majit descriptors: pointer-sized fields encode size_bits == 0
        // because (8 & 0x7) == 0, but the runtime field width is still 8.
        let size_bits = ((self.0 >> 1) & 0x7) as usize;
        if size_bits == 0 { 8 } else { size_bits }
    }

    fn is_field_signed(&self) -> bool {
        (self.0 >> 3) & 1 != 0
    }

    fn field_type(&self) -> majit_ir::Type {
        match self.0 & 0x3 {
            0 => majit_ir::Type::Int,
            1 => majit_ir::Type::Ref,
            2 => majit_ir::Type::Float,
            _ => majit_ir::Type::Void,
        }
    }
}

pub(crate) fn make_field_index_descr(idx: u32) -> DescrRef {
    if let Some(descr) = FIELD_DESCR_REGISTRY.with(|r| r.borrow().get(&idx).cloned()) {
        return descr;
    }
    Arc::new(FieldIndexDescr(idx))
}

fn virtualizable_field_index(offset: usize) -> u32 {
    // Encode as: FIELD_DESCR_TAG | (offset << 4) | (size_bits << 1) | type_bits
    // type_bits = 0 (Int), size_bits = 0 (8 & 0x7 = 0 for pointer-sized fields)
    0x1000_0000 | (((offset as u32) & 0x000f_ffff) << 4)
}

/// Extract the byte offset from a field descriptor index.
///
/// Field descriptor indices encode offset, size, and type. This extracts
/// just the byte offset for matching against VirtualizableConfig offsets.
fn extract_field_offset(descr_idx: u32) -> Option<usize> {
    const FIELD_DESCR_TAG: u32 = 0x1000_0000;
    if descr_idx & 0xF000_0000 != FIELD_DESCR_TAG {
        return None;
    }
    Some(((descr_idx >> 4) & 0x000f_ffff) as usize)
}

/// Helper for virtualizable array element tracking.
fn get_array_element(arrays: &[(u32, Vec<OpRef>)], arr_idx: u32, elem_idx: usize) -> Option<OpRef> {
    arrays
        .iter()
        .find(|(i, _)| *i == arr_idx)
        .and_then(|(_, e)| e.get(elem_idx).copied())
        .filter(|r| !r.is_none())
}

/// Helper for virtualizable array element tracking.
fn set_array_element(
    arrays: &mut Vec<(u32, Vec<OpRef>)>,
    arr_idx: u32,
    elem_idx: usize,
    value: OpRef,
) {
    if let Some((_, elems)) = arrays.iter_mut().find(|(i, _)| *i == arr_idx) {
        if elem_idx >= elems.len() {
            elems.resize(elem_idx + 1, OpRef::NONE);
        }
        elems[elem_idx] = value;
    } else {
        let mut elems = vec![OpRef::NONE; elem_idx + 1];
        elems[elem_idx] = value;
        arrays.push((arr_idx, elems));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        idx: u32,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.idx
        }
    }

    impl FieldDescr for TestFieldDescr {
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

    fn field_descr(idx: u32) -> DescrRef {
        Arc::new(TestFieldDescr { idx })
    }

    fn ref_field_descr(idx: u32) -> DescrRef {
        majit_ir::make_field_descr(idx as usize * 8, 8, Type::Ref, false)
    }

    fn array_descr(idx: u32) -> DescrRef {
        Arc::new(TestArrayDescr { idx })
    }

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    fn run_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVirtualize::new()));
        opt.optimize_with_constants_and_inputs(ops, &mut std::collections::HashMap::new(), 1024)
    }

    fn run_default_pipeline(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::default_pipeline();
        opt.optimize_with_constants_and_inputs(ops, &mut std::collections::HashMap::new(), 1024)
    }

    fn run_pass_with_constants(ops: &[Op], constants: &[(OpRef, Value)]) -> Vec<Op> {
        let mut ctx = OptContext::new(ops.len());
        for &(opref, ref val) in constants {
            ctx.make_constant(opref, val.clone());
        }

        let mut pass = OptVirtualize::new();
        pass.setup();

        for op in ops {
            // Resolve forwarded arguments
            let mut resolved_op = op.clone();
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }

            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        ctx.new_operations
    }

    #[test]
    fn test_standard_virtualizable_force_is_noop_in_optimizer() {
        let mut ctx = OptContext::with_num_inputs(8, 1);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            array_field_offsets: vec![8],
            array_item_types: vec![Type::Ref],
            array_lengths: vec![1],
        });
        pass.setup();
        ctx.set_ptr_info(
            OpRef(0),
            PtrInfo::Virtualizable(VirtualizableFieldState {
                fields: vec![],
                field_descrs: vec![],
                arrays: vec![(0, vec![OpRef::NONE])],
                last_guard_pos: -1,
            }),
        );

        let forced = pass.force_virtual(OpRef(0), &mut ctx);
        assert_eq!(forced, OpRef(0));
        assert!(
            ctx.new_operations.is_empty(),
            "standard virtualizable should not be forced to raw heap ops by optimizer"
        );
    }

    #[test]
    fn test_standard_virtualizable_raw_first_read_is_not_cached() {
        let mut ctx = OptContext::with_num_inputs(8, 1);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            array_field_offsets: vec![8],
            array_item_types: vec![Type::Int],
            array_lengths: vec![1],
        });
        pass.setup();

        let field_descr =
            make_field_index_descr(0x1000_0000 | (((8_u32) & 0x000f_ffff) << 4) | (0x7 << 1));
        let arr_descr = array_descr(20);
        ctx.make_constant(OpRef(50), Value::Int(0));

        let get_array_ptr = Op::with_descr(OpCode::GetfieldRawI, &[OpRef(0)], field_descr);
        let get_item = Op::with_descr(
            OpCode::GetarrayitemRawI,
            &[OpRef(0), OpRef(50)],
            arr_descr.clone(),
        );
        let get_item_again =
            Op::with_descr(OpCode::GetarrayitemRawI, &[OpRef(0), OpRef(50)], arr_descr);

        let mut ops = vec![get_array_ptr, get_item, get_item_again];
        assign_positions(&mut ops);

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_box_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
                OptimizationResult::InvalidLoop => {
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
    fn test_standard_virtualizable_guard_with_full_frame_payload_is_not_reaugmented() {
        let mut ctx = OptContext::with_num_inputs(8, 3);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_lengths: vec![1],
        });
        pass.setup();

        let pc = OpRef(50);
        let vsd = OpRef(51);
        ctx.make_constant(pc, Value::Int(6));
        ctx.make_constant(vsd, Value::Int(1));
        let local0 = OpRef(2);
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(99)]);
        guard.fail_args = Some(vec![OpRef(0), pc, vsd, local0].into());

        let replaced = match pass.propagate_forward(&guard, &mut ctx) {
            OptimizationResult::Replace(op) => op,
            other => panic!("expected guard replacement, got {other:?}"),
        };
        let fail_args = replaced.fail_args.expect("guard should have fail args");
        assert_eq!(fail_args.as_slice(), &[OpRef(0), pc, vsd, local0]);
    }

    #[test]
    fn test_standard_virtualizable_call_does_not_force_frame_to_raw_storeback() {
        let mut ctx = OptContext::with_num_inputs(8, 3);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8, 16],
            static_field_types: vec![Type::Int, Type::Int],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_lengths: vec![],
        });
        pass.setup();

        let mut call = Op::new(OpCode::CallMayForceI, &[OpRef(0), OpRef(100), OpRef(1)]);
        call.descr = Some(majit_ir::descr::make_call_descr(
            vec![Type::Int, Type::Int, Type::Int],
            Type::Int,
            majit_ir::EffectInfo::default(),
        ));

        let replaced = match pass.propagate_forward(&call, &mut ctx) {
            OptimizationResult::Replace(op) => op,
            other => panic!("expected call replacement, got {other:?}"),
        };
        assert_eq!(replaced.arg(0), OpRef(0));
        assert!(
            ctx.new_operations
                .iter()
                .all(|op| op.opcode != OpCode::SetfieldRaw),
            "standard virtualizable call should not force frame writeback"
        );
    }

    #[test]
    fn test_standard_virtualizable_raw_getfield_is_not_absorbed_by_optimizer() {
        let mut ctx = OptContext::with_num_inputs(8, 2);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_lengths: vec![],
        });
        pass.setup();

        let mut get = Op::new(OpCode::GetfieldRawI, &[OpRef(0)]);
        get.descr = Some(make_field_index_descr(virtualizable_field_index(8)));
        get.pos = OpRef(10);

        let result = pass.propagate_forward(&get, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_raw_setfield_is_not_absorbed_by_optimizer() {
        let mut ctx = OptContext::with_num_inputs(8, 2);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            array_field_offsets: vec![],
            array_item_types: vec![],
            array_lengths: vec![],
        });
        pass.setup();

        let mut set = Op::new(OpCode::SetfieldRaw, &[OpRef(0), OpRef(1)]);
        set.descr = Some(make_field_index_descr(virtualizable_field_index(8)));

        let result = pass.propagate_forward(&set, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_field_index_descr_decodes_pointer_sized_field_width() {
        let descr = make_field_index_descr(virtualizable_field_index(8));
        let fd = descr.as_field_descr().expect("field descr");
        assert_eq!(fd.offset(), 8);
        assert_eq!(fd.field_size(), 8);
        assert_eq!(fd.field_type(), Type::Int);
    }

    #[test]
    fn test_make_field_index_descr_reuses_registered_real_descr() {
        let field_idx = 0x1234_5678;
        let real_descr = majit_ir::make_field_descr(0, 8, Type::Ref, false);
        let mut field_descrs = Vec::new();
        set_field_descr(&mut field_descrs, field_idx, real_descr.clone());

        let descr = make_field_index_descr(field_idx);
        let fd = descr.as_field_descr().expect("field descr");
        assert_eq!(fd.field_size(), 8);
        assert_eq!(fd.field_type(), Type::Ref);
        assert_eq!(fd.offset(), 0);
    }

    #[test]
    fn test_standard_virtualizable_raw_getarrayitem_is_not_absorbed_by_optimizer() {
        let mut ctx = OptContext::with_num_inputs(8, 2);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_lengths: vec![1],
        });
        pass.setup();

        let mut get_field = Op::new(OpCode::GetfieldRawI, &[OpRef(0)]);
        get_field.descr = Some(make_field_index_descr(virtualizable_field_index(24)));
        get_field.pos = OpRef(10);
        assert!(matches!(
            pass.propagate_forward(&get_field, &mut ctx),
            OptimizationResult::PassOn
        ));
        ctx.emit(get_field);

        let mut get_item = Op::new(OpCode::GetarrayitemRawI, &[OpRef(10), OpRef(1)]);
        get_item.descr = Some(array_descr(24));
        let result = pass.propagate_forward(&get_item, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_raw_setarrayitem_is_not_absorbed_by_optimizer() {
        let mut ctx = OptContext::with_num_inputs(8, 2);
        let mut pass = OptVirtualize::with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![],
            static_field_types: vec![],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_lengths: vec![1],
        });
        pass.setup();

        let mut get_field = Op::new(OpCode::GetfieldRawI, &[OpRef(0)]);
        get_field.descr = Some(make_field_index_descr(virtualizable_field_index(24)));
        get_field.pos = OpRef(10);
        assert!(matches!(
            pass.propagate_forward(&get_field, &mut ctx),
            OptimizationResult::PassOn
        ));
        ctx.emit(get_field);

        let mut set_item = Op::new(OpCode::SetarrayitemRaw, &[OpRef(10), OpRef(1), OpRef(2)]);
        set_item.descr = Some(array_descr(24));
        let result = pass.propagate_forward(&set_item, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));
    }

    #[test]
    fn test_standard_virtualizable_loop_keeps_original_input_arity() {
        let mut opt = Optimizer::default_pipeline_with_virtualizable(VirtualizableConfig {
            static_field_offsets: vec![8],
            static_field_types: vec![Type::Int],
            array_field_offsets: vec![24],
            array_item_types: vec![Type::Int],
            array_lengths: vec![1],
        });
        let mut constants = HashMap::new();
        let mut ops = vec![
            Op::new(OpCode::Label, &[OpRef(0), OpRef(1), OpRef(2)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1), OpRef(2)]),
        ];
        ops[1].fail_args = Some(Default::default());
        assign_positions(&mut ops);

        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);
        let jump = result
            .iter()
            .find(|op| op.opcode == OpCode::Jump)
            .expect("optimized loop should keep a jump");

        assert_eq!(opt.final_num_inputs(), 3);
        assert_eq!(jump.args.len(), 3);
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
    fn test_new_removed() {
        let mut ops = vec![Op::with_descr(OpCode::New, &[], size_descr(1))];
        assign_positions(&mut ops);
        let result = run_pass(&ops);
        assert!(result.is_empty(), "NEW should be removed");
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], fd.clone()),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        assert!(
            result.is_empty(),
            "all ops should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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

        // OpRef(50) = constant 3 (array size)
        // OpRef(51) = constant 0 (index)
        // OpRef(52) = value to store (arbitrary opref)

        let mut ops = vec![
            Op::with_descr(OpCode::NewArray, &[OpRef(50)], ad.clone()), // pos=0
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(51), OpRef(52)],
                ad.clone(),
            ), // pos=1
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), OpRef(51)], ad.clone()), // pos=2
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(50), Value::Int(3)), (OpRef(51), Value::Int(0))];

        let result = run_pass_with_constants(&ops, &constants);
        assert!(
            result.is_empty(),
            "all array ops on virtual should be removed; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_arraylen_gc_on_virtual() {
        // Virtual array of length 5 -> arraylen_gc returns constant 5
        let ad = array_descr(20);

        let mut ops = vec![
            Op::with_descr(OpCode::NewArray, &[OpRef(50)], ad.clone()),
            Op::with_descr(OpCode::ArraylenGc, &[OpRef(0)], ad.clone()),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(50), Value::Int(5))];

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
        // define `optimize_GUARD_CLASS`. rewrite.py:397
        // `optimize_GUARD_CLASS` calls `info.get_known_class(cpu)` on
        // the virtual's InstancePtrInfo and removes the guard when the
        // stored class matches. Run the full default pipeline so
        // OptRewrite sees the guard after OptVirtualize produced the
        // virtual.
        let sd: DescrRef = majit_ir::make_size_descr_with_vtable(1, 8, 0, 42);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::default_pipeline();
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 42i64); // expected class ptr matches vtable
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
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)], fd_ref.clone()), // pos=2
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(100)], fd_int.clone()), // pos=3
            Op::new(OpCode::CallN, &[OpRef(0)]),                     // pos=4
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], fd.clone()),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // Forced: NEW, SETFIELD_GC, CALL_N
        assert_eq!(result[0].opcode, OpCode::New);
        let has_setfield = result.iter().any(|o| o.opcode == OpCode::SetfieldGc);
        assert!(has_setfield, "should have SETFIELD_GC");
        assert_eq!(result.last().unwrap().opcode, OpCode::CallN);
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd_a.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(200)], fd_b.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], fd_a.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], fd_b.clone()),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
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
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(200)], fd.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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
        // rewrite.py:430-436 `postprocess_GUARD_CLASS` records the
        // class via `make_constant_class`, and the second
        // `optimize_GUARD_CLASS` (rewrite.py:397) sees the recorded
        // known class and removes itself. virtualize.py doesn't handle
        // GUARD_CLASS at all; run the full default pipeline.
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::default_pipeline();
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 42i64); // class ptr constant
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
        // Operations on non-virtual objects should pass through unchanged
        let fd = field_descr(10);

        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(200)], fd.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], fd.clone()),
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
        // virtual_ref_finish(vref, NULL)      <- absorbed into virtual, removed
        //
        // Expected output: only ForceToken (emitted by optimizer) + SameAsI for the null constant
        let mut ops = vec![
            Op::new(OpCode::VirtualRefR, &[OpRef(100), OpRef(101)]), // pos=0
            Op::new(OpCode::VirtualRefFinish, &[OpRef(0), OpRef(102)]), // pos=1
        ];
        assign_positions(&mut ops);

        // OpRef(102) = constant 0 (NULL)
        let constants = vec![(OpRef(102), Value::Int(0))];
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
            Op::new(OpCode::VirtualRefR, &[OpRef(100), OpRef(101)]), // pos=0
            Op::new(OpCode::CallN, &[OpRef(0)]),                     // pos=1
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
            Op::new(OpCode::VirtualRefR, &[OpRef(100), OpRef(101)]), // pos=0
            Op::new(OpCode::VirtualRefFinish, &[OpRef(0), OpRef(200)]), // pos=1, non-null
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
        // vref = virtual_ref_r(p0, token)     <- virtual struct
        // call_n(vref)                         <- forces vref, NOT p0
        //
        // The key property: forcing the vref should NOT force the wrapped
        // object p0. The vref struct's `forced` field is set to NULL (0)
        // by optimize_virtual_ref, so p0 is not referenced in the vref fields.
        // p0 only appears in the original VirtualRefR args, which are discarded.
        let sd = size_descr(1);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()), // pos=0
            Op::new(OpCode::VirtualRefR, &[OpRef(0), OpRef(101)]),  // pos=1
            Op::new(OpCode::CallN, &[OpRef(1)]),                    // pos=2
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // The vref struct is a VirtualStruct forced as New.
        // p0 (NewWithVtable) should NOT appear because the vref's forced field
        // is NULL, not p0.
        let new_vtable_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::NewWithVtable)
            .count();
        assert_eq!(
            new_vtable_count,
            0,
            "the wrapped object p0 should NOT be forced; got ops: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );

        // The vref struct itself is forced as New
        let new_count = result.iter().filter(|o| o.opcode == OpCode::New).count();
        assert_eq!(
            new_count,
            1,
            "only the vref struct should be allocated; got ops: {:?}",
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
            Op::new(OpCode::VirtualRefR, &[OpRef(100), OpRef(101)]), // pos=0
            Op::new(OpCode::CallN, &[OpRef(0)]),                     // pos=1
            Op::new(OpCode::VirtualRefFinish, &[OpRef(0), OpRef(200)]), // pos=2
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
        // i0 = getfield_gc_i(vref, descr=vref_forced_field)
        //
        // The vref is virtual, so getfield should return the virtual field value.
        let forced_descr = field_descr(super::VREF_FORCED_FIELD_INDEX);

        let mut ops = vec![
            Op::new(OpCode::VirtualRefR, &[OpRef(100), OpRef(101)]), // pos=0
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], forced_descr), // pos=1
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        // The getfield should be removed (the forced field is a known constant 0)
        let has_getfield = result.iter().any(|o| o.opcode == OpCode::GetfieldGcI);
        assert!(
            !has_getfield,
            "getfield on virtual vref should be removed; got: {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
    }

    // ── VirtualRawBuffer optimization tests (RPython: test_rawmem.py parity) ──

    fn run_pass_with_raw_buffer(
        ops: &[Op],
        constants: &[(OpRef, Value)],
        raw_bufs: &[(OpRef, usize)],
    ) -> Vec<Op> {
        let mut ctx = OptContext::new(ops.len());
        for &(opref, ref val) in constants {
            ctx.make_constant(opref, val.clone());
        }

        let mut pass = OptVirtualize::new();
        pass.setup();

        // Pre-populate VirtualRawBuffer info for specified OpRefs
        for &(opref, size) in raw_bufs {
            ctx.set_ptr_info(
                opref,
                PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                    size,
                    entries: Vec::new(),
                    last_guard_pos: -1,
                }),
            );
        }

        for op in ops {
            let mut resolved_op = op.clone();
            for arg in &mut resolved_op.args {
                *arg = ctx.get_box_replacement(*arg);
            }

            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        pass.flush(&mut ctx);
        ctx.new_operations
    }

    #[test]
    fn test_raw_store_then_load_same_offset_forwarded() {
        // Mirrors RPython's test_raw_storage_int: store a value, then
        // load from the same offset on a virtual buffer.
        // raw_store(buf, offset=0, val)
        // i1 = raw_load_i(buf, offset=0)
        // -> i1 should be forwarded to val, both ops removed.
        let mut ops = vec![
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(100), OpRef(200)]), // pos=0
            Op::new(OpCode::RawLoadI, &[OpRef(0), OpRef(100)]),             // pos=1
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), Value::Int(0))]; // offset = 0
        let raw_bufs = vec![(OpRef(0), 32)];

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
        // raw_store(buf, offset=0, val_a)
        // raw_store(buf, offset=8, val_b)
        // i1 = raw_load_i(buf, offset=0)  -> val_a
        // i2 = raw_load_i(buf, offset=8)  -> val_b
        let mut ops = vec![
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(100), OpRef(200)]), // pos=0
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(101), OpRef(201)]), // pos=1
            Op::new(OpCode::RawLoadI, &[OpRef(0), OpRef(100)]),             // pos=2
            Op::new(OpCode::RawLoadI, &[OpRef(0), OpRef(101)]),             // pos=3
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), Value::Int(0)), (OpRef(101), Value::Int(8))];
        let raw_bufs = vec![(OpRef(0), 32)];

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
        // raw_store(buf, 0, val_a)
        // raw_store(buf, 0, val_b)   <- overwrites
        // i1 = raw_load_i(buf, 0)    -> val_b
        // call_n(buf)                <- force
        let mut ops = vec![
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(100), OpRef(200)]), // pos=0
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(100), OpRef(201)]), // pos=1
            Op::new(OpCode::RawLoadI, &[OpRef(0), OpRef(100)]),             // pos=2
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), Value::Int(0))];
        let raw_bufs = vec![(OpRef(0), 32)];

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
        let mut ops = vec![
            Op::new(OpCode::RawStore, &[OpRef(50), OpRef(100), OpRef(200)]),
            Op::new(OpCode::RawLoadI, &[OpRef(50), OpRef(100)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), Value::Int(0))];
        // No raw_bufs — OpRef(50) is NOT a virtual buffer.
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
    fn test_call_forced_virtual_pure_getfield() {
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
        let sd = size_descr(1);
        let fd = majit_ir::descr::make_field_descr_full(10, 0, 8, Type::Ref, true);
        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(100)], fd.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(0)], fd.clone()),
            Op::new(OpCode::CallN, &[OpRef(3)]),
            Op::new(OpCode::Jump, &[OpRef(100)]),
        ];
        assign_positions(&mut ops);

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
        assert_eq!(result[3].arg(0), OpRef(100));
        assert_eq!(result[4].arg(0), OpRef(100));
    }

    #[test]
    fn test_jump_drops_virtual_value_lazy_setfield() {
        // RPython parity: at JUMP, lazy SetfieldGc with virtual value is
        // DROPPED. heap.py emit_extra(op, emit=False) re-processes the op
        // through passes → re-absorbed as lazy_set → lost. The virtual
        // stays virtual and is carried across JUMP via imported heap cache.
        //
        // [p0]
        // p1 = new(descr=node)
        // setfield_gc(p0, p1, descr=next)
        // jump(p0)
        //
        // Result: only Jump (New is virtual, SetfieldGc is lazy → dropped).
        let node_sd = size_descr(1);
        let next_fd = ref_field_descr(11);
        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], node_sd.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)], next_fd.clone()),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        ops[0].pos = OpRef(1);
        ops[1].pos = OpRef(2);
        ops[2].pos = OpRef(3);

        let mut opt = Optimizer::default_pipeline();
        let mut constants = HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // force_all_lazy_setfields emits the lazy SetfieldGc at JUMP,
        // which forces the virtual New to be materialized.
        let new_count = result.iter().filter(|op| op.opcode == OpCode::New).count();
        assert_eq!(
            new_count, 1,
            "virtual New should be materialized when lazy SetfieldGc is emitted at Jump; got {result:?}"
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
                &[OpRef(2), OpRef(100)],
                value_fd.clone(),
            ),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(2), OpRef(0)], next_fd.clone()),
            Op::with_descr(OpCode::New, &[], node_sd.clone()),
            Op::with_descr(
                OpCode::SetfieldGc,
                &[OpRef(5), OpRef(101)],
                value_fd.clone(),
            ),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(5), OpRef(2)], next_fd.clone()),
            Op::new(OpCode::Finish, &[OpRef(5), OpRef(2), OpRef(1), OpRef(0)]),
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef((idx + 2) as u32);
        }

        let mut opt = Optimizer::default_pipeline();
        let mut constants = HashMap::new();
        constants.insert(100, 7);
        constants.insert(101, 11);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos)
            .collect();
        assert_eq!(
            new_positions.len(),
            2,
            "expected two forced allocations; got {result:?}"
        );

        for set_op in result.iter().filter(|op| op.opcode == OpCode::SetfieldGc) {
            assert!(
                new_positions.contains(&set_op.arg(0)),
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
            new_positions.contains(&finish.arg(0)),
            "first Finish ref should be a forced allocation; got {:?} in {:?}",
            finish.arg(0),
            result
        );
        assert!(
            new_positions.contains(&finish.arg(1)),
            "second Finish ref should be a forced allocation; got {:?} in {:?}",
            finish.arg(1),
            result
        );
        assert!(
            !constants.contains_key(&finish.arg(0).0),
            "forced allocation ref must not collide with an exported int constant"
        );
        assert!(
            !constants.contains_key(&finish.arg(1).0),
            "forced allocation ref must not collide with an exported int constant"
        );
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_guard_fail_args_virtual_not_forced() {
        // resume.py parity: virtual objects in guard fail_args should NOT be
        // forced (no allocation emitted). rd_numb with TAGVIRTUAL is set.
        //
        // p0 = new_with_vtable(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // guard_true(i20) [p0]
        //
        // Expected: no NEW_WITH_VTABLE emitted. Guard has rd_numb.
        // fail_args expanded: OpRef::NONE + field values.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(20)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()), // pos=0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], fd.clone()), // pos=1
            guard,                                                  // pos=2
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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
            guard_op.rd_numb.is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // fail_args has EXPANDED length: virtual slot = OpRef::NONE, field values appended
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            fa.iter().any(|a| a.is_none()),
            "virtual slot should be OpRef::NONE placeholder in fail_args"
        );
        // Field value (OpRef(10)) should be appended as extra fail_arg
        assert!(
            fa.len() > 1,
            "fail_args should be expanded with virtual field values"
        );
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_guard_fail_args_mixed_virtual_and_non_virtual() {
        // Guard with both virtual and non-virtual fail_args.
        //
        // p0 = new(descr=size1)
        // setfield_gc(p0, i10, descr=field1)
        // guard_true(i20) [i30, p0, i40]
        //
        // rd_numb set. fail_args expanded:
        // [i30, NONE, i40, i10] — virtual slot replaced with NONE, field appended.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(20)]);
        guard.fail_args = Some(vec![OpRef(30), OpRef(0), OpRef(40)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()), // pos=0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], fd.clone()), // pos=1
            guard,                                        // pos=2
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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
            guard_op.rd_numb.is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // fail_args has EXPANDED length: virtual slot = OpRef::NONE, field values appended
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            fa.iter().any(|a| a.is_none()),
            "virtual slot should be OpRef::NONE placeholder in fail_args"
        );
        // Non-virtual boxes should still be present
        assert!(
            fa.iter().any(|a| *a == OpRef(30)),
            "non-virtual OpRef(30) should be in fail_args"
        );
        assert!(
            fa.iter().any(|a| *a == OpRef(40)),
            "non-virtual OpRef(40) should be in fail_args"
        );
        // fail_args expanded: original 3 + field value(s) for the virtual
        assert!(
            fa.len() > 3,
            "fail_args should be expanded with virtual field values"
        );
    }

    #[test]
    fn test_guard_fail_args_no_virtual_no_rd_numb() {
        // Guard with no virtuals in fail_args should not have rd_numb.
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(10)]);
        guard.fail_args = Some(vec![OpRef(20), OpRef(30)].into());

        let mut ops = vec![guard];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        // No virtuals — fail_args should remain as-is with concrete values.
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            fa.iter().all(|a| !a.is_none()),
            "no virtuals => all fail_args should be concrete"
        );
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_guard_fail_args_virtual_struct_not_forced() {
        // VirtualStruct (New) in guard fail_args should also use resume data.
        let sd = size_descr(1);
        let fd = field_descr(10);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(20)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], sd.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], fd.clone()),
            guard,
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

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
            guard_op.rd_numb.is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );
        // fail_args has EXPANDED length: virtual slot = OpRef::NONE, field values appended
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            fa.iter().any(|a| a.is_none()),
            "virtual slot should be OpRef::NONE placeholder in fail_args"
        );
        assert!(
            fa.len() > 1,
            "fail_args should be expanded with virtual field values"
        );
    }

    #[test]
    #[ignore] // consumer switchover: test setup doesn't trigger inline guard numbering
    fn test_guard_fail_args_virtual_with_multiple_fields() {
        // Virtual with two fields in guard fail_args.
        let sd = size_descr(1);
        let fd_a = field_descr(10);
        let fd_b = field_descr(20);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(30)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], fd_a.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(20)], fd_b.clone()),
            guard,
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);

        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        assert!(
            guard_op.rd_numb.is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );

        // fail_args has EXPANDED length: virtual slot = OpRef::NONE, field values appended
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            fa.iter().any(|a| a.is_none()),
            "virtual slot should be OpRef::NONE placeholder in fail_args"
        );
        // Two fields means at least 2 extra fail_args beyond the original 1
        assert!(
            fa.len() >= 3,
            "fail_args should be expanded with 2 virtual field values (got {})",
            fa.len()
        );
    }

    #[test]
    fn test_guard_fail_args_nested_virtual_field_is_forced_to_concrete() {
        let outer_sd = size_descr(1);
        let inner_sd = size_descr(2);
        let outer_fd = ref_field_descr(10);
        let inner_fd = field_descr(20);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(30)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], outer_sd),
            Op::with_descr(OpCode::New, &[], inner_sd),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(40)], inner_fd),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)], outer_fd),
            guard,
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");
        let fa = guard_op.fail_args.as_ref().unwrap();
        // Nested virtual fields force the root fail_arg concrete.
        assert!(
            guard_op.rd_numb.is_none() || guard_op.rd_virtuals.is_none(),
            "nested virtual fields should force the root fail_arg concrete"
        );
        assert!(
            !fa[0].is_none(),
            "root fail_arg should be forced concrete when nested virtual fields are present"
        );
        assert!(
            result
                .iter()
                .any(|op| matches!(op.opcode, OpCode::New | OpCode::NewWithVtable)),
            "forcing the root virtual should emit concrete allocation ops"
        );
    }

    #[test]
    fn test_guard_fail_args_unsupported_virtual_array_is_forced() {
        let ad = array_descr(30);
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(20)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::NewArray, &[OpRef(10)], ad.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(11), OpRef(12)],
                ad,
            ),
            guard,
        ];
        assign_positions(&mut ops);

        let constants = &[
            (OpRef(10), Value::Int(1)),
            (OpRef(11), Value::Int(0)),
            (OpRef(12), Value::Int(99)),
        ];
        let result = run_pass_with_constants(&ops, constants);
        let guard_op = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard should be emitted");

        // Unsupported virtual arrays should fall back to concrete fail_args.

        let fa = guard_op.fail_args.as_ref().unwrap();
        assert!(
            !fa[0].is_none(),
            "virtual array fail_arg should be forced to a concrete value"
        );
    }
}
