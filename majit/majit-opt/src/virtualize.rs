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

use crate::info::{
    PtrInfo, VirtualArrayInfo, VirtualArrayStructInfo, VirtualInfo, VirtualRawBufferInfo,
    VirtualStructInfo, VirtualizableFieldState,
};
use crate::{OptContext, Optimization, OptimizationResult};

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
}

/// Field descriptor index for the `virtual_token` field of JitVirtualRef.
const VREF_VIRTUAL_TOKEN_FIELD_INDEX: u32 = 0x7F00;
/// Field descriptor index for the `forced` field of JitVirtualRef.
const VREF_FORCED_FIELD_INDEX: u32 = 0x7F01;
/// Size descriptor index for the JitVirtualRef struct.
const VREF_SIZE_DESCR_INDEX: u32 = 0x7F10;

/// The virtualize optimization pass.
pub struct OptVirtualize {
    /// Per-operation PtrInfo, indexed by OpRef.0.
    ptr_info: Vec<Option<PtrInfo>>,
    /// If set, frame OpRef(0) is treated as a virtualizable object
    /// whose field accesses are absorbed by the optimizer.
    vable_config: Option<VirtualizableConfig>,
    /// Maps array pointer OpRef → (array_field_index, array_descr).
    /// Populated when a GetfieldRawI reads an array pointer field from the frame.
    /// The DescrRef is captured from the first SetarrayitemRaw/GetarrayitemRaw op
    /// we absorb, so we can emit correct SetarrayitemRaw on force.
    vable_array_ptrs: HashMap<OpRef, (usize, Option<DescrRef>)>,
    /// Virtual input args for virtualizable static fields.
    /// Each entry: (config_index, virtual_input_OpRef).
    /// Created lazily on first propagate_forward call.
    vable_virtual_inputs: Vec<(usize, OpRef)>,
    /// Whether virtual inputs have been initialized.
    vable_initialized: bool,
}

impl OptVirtualize {
    pub fn new() -> Self {
        OptVirtualize {
            ptr_info: Vec::new(),
            vable_config: None,
            vable_array_ptrs: HashMap::new(),
            vable_virtual_inputs: Vec::new(),
            vable_initialized: false,
        }
    }

    /// Create with virtualizable config for frame field tracking.
    pub fn with_virtualizable(config: VirtualizableConfig) -> Self {
        OptVirtualize {
            ptr_info: Vec::new(),
            vable_config: Some(config),
            vable_array_ptrs: HashMap::new(),
            vable_virtual_inputs: Vec::new(),
            vable_initialized: false,
        }
    }

    /// Lazily initialize virtual input args for virtualizable static fields.
    /// Called on first propagate_forward. Allocates OpRef positions beyond
    /// the current num_inputs, so the first reads can be replaced with
    /// virtual input references instead of memory loads.
    fn init_virtualizable(&mut self, ctx: &mut OptContext) {
        if let Some(ref config) = self.vable_config {
            for (i, &_offset) in config.static_field_offsets.iter().enumerate() {
                let virt_ref = OpRef(ctx.num_inputs);
                ctx.num_inputs += 1;
                self.vable_virtual_inputs.push((i, virt_ref));
            }
            // vstate.fields is populated lazily on first GetfieldRawI,
            // which forwards to the virtual input ref.
        }
    }

    // ── PtrInfo accessors ──

    fn set_info(&mut self, opref: OpRef, info: PtrInfo) {
        let idx = opref.0 as usize;
        if idx >= self.ptr_info.len() {
            self.ptr_info.resize(idx + 1, None);
        }
        self.ptr_info[idx] = Some(info);
    }

    fn get_info(&self, opref: OpRef) -> Option<&PtrInfo> {
        let idx = opref.0 as usize;
        self.ptr_info.get(idx).and_then(|v| v.as_ref())
    }

    fn get_info_mut(&mut self, opref: OpRef) -> Option<&mut PtrInfo> {
        let idx = opref.0 as usize;
        self.ptr_info.get_mut(idx).and_then(|v| v.as_mut())
    }

    /// Resolve through forwarding and return PtrInfo if it's a virtual.
    fn get_virtual_info<'a>(&'a self, opref: OpRef, ctx: &OptContext) -> Option<&'a PtrInfo> {
        let resolved = ctx.get_replacement(opref);
        self.get_info(resolved).filter(|info| info.is_virtual())
    }

    fn is_virtual(&self, opref: OpRef, ctx: &OptContext) -> bool {
        self.get_virtual_info(opref, ctx).is_some()
    }

    // ── Force virtual ──

    /// Force a virtual to become concrete: emit the allocation + setfield ops.
    /// Returns the OpRef of the emitted allocation.
    fn force_virtual(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_replacement(opref);
        let info = match self
            .ptr_info
            .get(resolved.0 as usize)
            .and_then(|v| v.as_ref())
        {
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
            PtrInfo::Virtualizable(vinfo) => self.force_virtualizable(resolved, vinfo, ctx),
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
        self.set_info(opref, PtrInfo::NonNull);

        // Emit SETFIELD_RAW for each tracked field, using the original DescrRef
        for (field_idx, value_ref) in &vinfo.fields {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let descr = get_field_descr(&vinfo.field_descrs, *field_idx)
                .unwrap_or_else(|| make_field_index_descr(*field_idx));
            let mut set_op = Op::new(OpCode::SetfieldRaw, &[opref, value_ref]);
            set_op.descr = Some(descr);
            ctx.emit(set_op);
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
                    let descr_idx =
                        0x1000_0000 | (((offset as u32) & 0x000f_ffff) << 4) | (0x7 << 1);
                    get_array_op.descr = get_field_descr(&vinfo.field_descrs, descr_idx)
                        .or_else(|| Some(make_field_index_descr(descr_idx)));
                }
            }
            let array_ref = ctx.emit(get_array_op);

            for (i, elem_ref) in elements.into_iter().enumerate() {
                let elem_ref = self.force_virtual(elem_ref, ctx);
                let elem_ref = ctx.get_replacement(elem_ref);
                let idx_ref = self.emit_constant_int(ctx, i as i64);
                let mut set_op = Op::new(OpCode::SetarrayitemRaw, &[array_ref, idx_ref, elem_ref]);
                set_op.descr = array_descr.clone();
                ctx.emit(set_op);
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
        // Mark as no longer virtual FIRST (avoids infinite recursion on nested virtuals)
        self.set_info(
            opref,
            PtrInfo::KnownClass {
                class_ptr: vinfo.known_class.unwrap_or(majit_ir::GcRef::NULL),
                is_nonnull: true,
            },
        );

        // Emit the NEW_WITH_VTABLE
        let mut alloc_op = Op::new(OpCode::NewWithVtable, &[]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit(alloc_op);

        // Set forwarding only when the refs differ (avoids self-loop)
        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETFIELD_GC for each tracked field
        for (field_idx, value_ref) in vinfo.fields {
            let value_ref = self.force_virtual(value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
            set_op.descr = Some(
                get_field_descr(&vinfo.field_descrs, field_idx)
                    .unwrap_or_else(|| make_field_index_descr(field_idx)),
            );
            ctx.emit(set_op);
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
        self.set_info(opref, PtrInfo::NonNull);

        // Emit the length constant and NEW_ARRAY
        let len_ref = self.emit_constant_int(ctx, len as i64);
        let mut alloc_op = Op::new(OpCode::NewArray, &[len_ref]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit(alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETARRAYITEM_GC for each item
        for (i, item_ref) in vinfo.items.iter().enumerate() {
            let item_ref = self.force_virtual(*item_ref, ctx);
            let item_ref = ctx.get_replacement(item_ref);
            let idx_ref = self.emit_constant_int(ctx, i as i64);
            let mut set_op = Op::new(OpCode::SetarrayitemGc, &[alloc_ref, idx_ref, item_ref]);
            set_op.descr = Some(vinfo.descr.clone());
            ctx.emit(set_op);
        }

        alloc_ref
    }

    fn force_virtual_struct(
        &mut self,
        opref: OpRef,
        vinfo: VirtualStructInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        // Mark as no longer virtual FIRST (avoids infinite recursion)
        self.set_info(opref, PtrInfo::NonNull);

        // Emit NEW
        let mut alloc_op = Op::new(OpCode::New, &[]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit(alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETFIELD_GC for each tracked field
        for (field_idx, value_ref) in &vinfo.fields {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let descr = get_field_descr(&vinfo.field_descrs, *field_idx)
                .unwrap_or_else(|| make_field_index_descr(*field_idx));
            let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
            set_op.descr = Some(descr);
            ctx.emit(set_op);
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
        self.set_info(opref, PtrInfo::NonNull);

        // Emit NEW_ARRAY for the outer array
        let len_ref = self.emit_constant_int(ctx, num_elements as i64);
        let mut alloc_op = Op::new(OpCode::NewArray, &[len_ref]);
        alloc_op.descr = Some(vinfo.descr.clone());
        let alloc_ref = ctx.emit(alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit SETINTERIORFIELD_GC for each element's fields
        for (elem_idx, fields) in vinfo.element_fields.iter().enumerate() {
            let idx_ref = self.emit_constant_int(ctx, elem_idx as i64);
            for (field_idx, value_ref) in fields {
                let value_ref = self.force_virtual(*value_ref, ctx);
                let value_ref = ctx.get_replacement(value_ref);
                let mut set_op =
                    Op::new(OpCode::SetinteriorfieldGc, &[alloc_ref, idx_ref, value_ref]);
                set_op.descr = Some(make_field_index_descr(*field_idx));
                ctx.emit(set_op);
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
        self.set_info(opref, PtrInfo::NonNull);

        // Emit CALL_MALLOC_NURSERY or equivalent raw allocation
        let size_ref = self.emit_constant_int(ctx, vinfo.size as i64);
        let alloc_op = Op::new(OpCode::CallMallocNursery, &[size_ref]);
        let alloc_ref = ctx.emit(alloc_op);

        if opref != alloc_ref {
            ctx.replace_op(opref, alloc_ref);
        }

        // Emit RAW_STORE for each tracked entry
        for (offset, _length, value_ref) in &vinfo.entries {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let offset_ref = self.emit_constant_int(ctx, *offset as i64);
            let set_op = Op::new(OpCode::RawStore, &[alloc_ref, offset_ref, value_ref]);
            ctx.emit(set_op);
        }

        alloc_ref
    }

    /// Force all arguments that are virtual.
    /// Extract the int payload from a VirtualStruct (e.g., W_IntObject's intval field).
    /// Returns the OpRef of the raw int if the virtual has exactly one int field
    /// (the common case for boxed integers).
    /// Extract the int payload from a VirtualStruct (e.g., W_IntObject).
    /// Returns the OpRef of the raw int if the virtual has a field at offset 8.
    fn extract_virtual_int_field(&self, opref: OpRef) -> Option<OpRef> {
        match self.get_info(opref)? {
            PtrInfo::VirtualStruct(vinfo) => {
                // Standard layout: [type_ptr@0, intval@8]
                vinfo
                    .fields
                    .iter()
                    .find(|(idx, _)| extract_field_offset(*idx) == Some(8))
                    .map(|(_, opref)| *opref)
            }
            _ => None,
        }
    }

    fn has_any_virtual_arg(&self, op: &Op, ctx: &OptContext) -> bool {
        op.args.iter().any(|arg| {
            let resolved = ctx.get_replacement(*arg);
            self.is_virtual(resolved, ctx)
        })
    }

    fn force_all_args(&mut self, op: &Op, ctx: &mut OptContext) -> Op {
        let mut new_op = op.clone();
        for arg in &mut new_op.args {
            let resolved = ctx.get_replacement(*arg);
            if self.is_virtual(resolved, ctx) {
                let forced = self.force_virtual(resolved, ctx);
                *arg = ctx.get_replacement(forced);
            } else {
                *arg = resolved;
            }
        }
        new_op
    }

    /// Emit a constant integer, returning its OpRef.
    fn emit_constant_int(&self, ctx: &mut OptContext, value: i64) -> OpRef {
        // Pre-compute the position this op will get when emitted.
        // Use self-referencing arg so resolve_opref finds the constant in the
        // constants map rather than trying to read a nonexistent variable.
        let pos_ref = OpRef(ctx.num_inputs + ctx.new_operations.len() as u32);
        let mut op = Op::new(OpCode::SameAsI, &[pos_ref]);
        op.pos = pos_ref;
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    // ── Per-opcode handlers ──

    fn optimize_new_with_vtable(&mut self, op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
        let descr = op.descr.clone().expect("NEW_WITH_VTABLE needs descr");
        let vinfo = VirtualInfo {
            descr,
            known_class: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        };
        self.set_info(op.pos, PtrInfo::Virtual(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new(&mut self, op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
        let descr = op.descr.clone().expect("NEW needs descr");
        let vinfo = VirtualStructInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        };
        self.set_info(op.pos, PtrInfo::VirtualStruct(vinfo));
        OptimizationResult::Remove
    }

    fn optimize_new_array(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let size_ref = op.arg(0);
        if let Some(size) = ctx.get_constant_int(size_ref) {
            if size >= 0 && size <= 1024 {
                let descr = op.descr.clone().expect("NEW_ARRAY needs descr");
                let items = vec![OpRef::NONE; size as usize];
                let vinfo = VirtualArrayInfo { descr, items };
                self.set_info(op.pos, PtrInfo::VirtualArray(vinfo));
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
        let struct_ref = ctx.get_replacement(op.arg(0));
        let value_ref = ctx.get_replacement(op.arg(1));
        let field_idx = descr_index(&op.descr);

        if let Some(info) = self.get_info_mut(struct_ref) {
            if info.is_virtual() {
                match info {
                    PtrInfo::Virtual(vinfo) => {
                        set_field(&mut vinfo.fields, field_idx, value_ref);
                        if let Some(descr) = &op.descr {
                            set_field_descr(&mut vinfo.field_descrs, field_idx, descr.clone());
                        }
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
        OptimizationResult::PassOn
    }

    fn optimize_getfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let struct_ref = ctx.get_replacement(op.arg(0));
        let field_idx = descr_index(&op.descr);

        if let Some(info) = self.get_info(struct_ref) {
            if info.is_virtual() {
                let field_val = match info {
                    PtrInfo::Virtual(vinfo) => get_field(&vinfo.fields, field_idx),
                    PtrInfo::VirtualStruct(vinfo) => get_field(&vinfo.fields, field_idx),
                    PtrInfo::Virtualizable(vstate) => get_field(&vstate.fields, field_idx),
                    _ => None,
                };
                if let Some(val_ref) = field_val {
                    ctx.replace_op(op.pos, val_ref);
                    return OptimizationResult::Remove;
                }
                // Virtualizable: first read of an untracked field.
                // Let the op emit, but cache result for future reads
                // and register array pointers for array tracking.
                if let PtrInfo::Virtualizable(_) = info {
                    return self.handle_virtualizable_first_read(op, struct_ref, field_idx, ctx);
                }
            }
        }
        OptimizationResult::PassOn
    }

    /// Handle first read of a virtualizable field (no cached value yet).
    ///
    /// For static fields with virtual inputs: replace the GetfieldRawI with
    /// the virtual input OpRef (eliminating the load from the loop body).
    /// For other fields: emit the op and cache the result.
    fn handle_virtualizable_first_read(
        &mut self,
        op: &Op,
        frame_ref: OpRef,
        field_idx: u32,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let offset = extract_field_offset(field_idx);
        if let Some(ref config) = self.vable_config {
            // Static field with a virtual input → replace with input arg, no load
            if let Some(virt_idx) =
                offset.and_then(|off| config.static_field_offsets.iter().position(|&o| o == off))
            {
                if let Some(&(_, virt_ref)) = self.vable_virtual_inputs.get(virt_idx) {
                    ctx.replace_op(op.pos, virt_ref);
                    if let Some(PtrInfo::Virtualizable(vstate)) = self.get_info_mut(frame_ref) {
                        set_field(&mut vstate.fields, field_idx, virt_ref);
                        if let Some(d) = op.descr.clone() {
                            set_field_descr(&mut vstate.field_descrs, field_idx, d);
                        }
                    }
                    return OptimizationResult::Remove;
                }
            }
            // Check if this reads an array pointer field
            if let Some(arr_idx) =
                offset.and_then(|off| config.array_field_offsets.iter().position(|&o| o == off))
            {
                self.vable_array_ptrs.insert(op.pos, (arr_idx, None));
            }
            // Non-virtual field: emit the op and cache the result
            if let Some(PtrInfo::Virtualizable(vstate)) = self.get_info_mut(frame_ref) {
                set_field(&mut vstate.fields, field_idx, op.pos);
                if let Some(d) = op.descr.clone() {
                    set_field_descr(&mut vstate.field_descrs, field_idx, d);
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_setarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let value_ref = ctx.get_replacement(op.arg(2));

        // Virtualizable array check
        if let Some(&mut (arr_idx, ref mut stored_descr)) =
            self.vable_array_ptrs.get_mut(&array_ref)
        {
            // Capture the array descr from the first absorbed op
            if stored_descr.is_none() {
                *stored_descr = op.descr.clone();
            }
            if let Some(index) = ctx.get_constant_int(index_ref) {
                let frame_ref = OpRef(0);
                if let Some(PtrInfo::Virtualizable(vstate)) = self.get_info_mut(frame_ref) {
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
            if let Some(info) = self.get_info_mut(array_ref) {
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
        if self.has_any_virtual_arg(op, ctx) {
            return OptimizationResult::Replace(self.force_all_args(op, ctx));
        }
        OptimizationResult::PassOn
    }

    fn optimize_getarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);

        // Virtualizable array check
        if let Some(&mut (arr_idx, ref mut stored_descr)) =
            self.vable_array_ptrs.get_mut(&array_ref)
        {
            // Capture the array descr from the first absorbed op
            if stored_descr.is_none() {
                *stored_descr = op.descr.clone();
            }
            if let Some(index) = ctx.get_constant_int(index_ref) {
                let frame_ref = OpRef(0);
                if let Some(PtrInfo::Virtualizable(vstate)) = self.get_info(frame_ref) {
                    if let Some(val) =
                        get_array_element(&vstate.arrays, arr_idx as u32, index as usize)
                    {
                        ctx.replace_op(op.pos, val);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = self.get_info(array_ref) {
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
        let array_ref = ctx.get_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = self.get_info(array_ref) {
            let len = vinfo.items.len() as i64;
            ctx.make_constant(op.pos, Value::Int(len));
            return OptimizationResult::Remove;
        }
        OptimizationResult::PassOn
    }

    fn optimize_strlen(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let str_ref = ctx.get_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = self.get_info(str_ref) {
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
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let field_idx = descr_index(&op.descr);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(PtrInfo::VirtualArrayStruct(vinfo)) = self.get_info(array_ref) {
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
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let value_ref = ctx.get_replacement(op.arg(2));
        let field_idx = descr_index(&op.descr);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = self.get_info_mut(array_ref) {
                if let PtrInfo::VirtualArrayStruct(vinfo) = info {
                    let elem_idx = index as usize;
                    if elem_idx < vinfo.element_fields.len() {
                        set_field(&mut vinfo.element_fields[elem_idx], field_idx, value_ref);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }
        OptimizationResult::PassOn
    }

    fn optimize_raw_load(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let buf_ref = ctx.get_replacement(op.arg(0));
        let offset_ref = op.arg(1);

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(PtrInfo::VirtualRawBuffer(vinfo)) = self.get_info(buf_ref) {
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
        let buf_ref = ctx.get_replacement(op.arg(0));
        let offset_ref = op.arg(1);
        let value_ref = ctx.get_replacement(op.arg(2));

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(info) = self.get_info_mut(buf_ref) {
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

    fn optimize_guard_class(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            match info {
                // Virtual objects have a known allocation type — guard is redundant
                PtrInfo::Virtual(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_) => {
                    return OptimizationResult::Remove;
                }
                PtrInfo::KnownClass { .. } => {
                    // Class already known from prior guard or virtual forcing
                    return OptimizationResult::Remove;
                }
                _ => {}
            }
        }

        // Constant-fold: if the guarded value is a compile-time constant,
        // check the class at optimization time. If it matches, remove the guard.
        // PyPy guard.py: guards on constants are always removable.
        if op.num_args() >= 2 {
            if let Some(value) = ctx.get_constant_int(obj_ref) {
                if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                    if value == expected {
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // Record the class for future lookups.
        self.set_info(
            obj_ref,
            PtrInfo::KnownClass {
                class_ptr: majit_ir::GcRef::NULL,
                is_nonnull: true,
            },
        );

        self.force_guard_fail_args(op, ctx)
    }

    /// Force virtual references in guard fail_args and re-resolve all args.
    /// Must be called for any guard that PassOn instead of Remove.
    fn force_guard_fail_args(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if let Some(ref fail_args) = op.fail_args {
            for &fa in fail_args {
                let resolved = ctx.get_replacement(fa);
                self.force_virtual(resolved, ctx);
            }
        }
        let mut guard_op = op.clone();
        if let Some(ref mut fa) = guard_op.fail_args {
            for arg in fa.iter_mut() {
                *arg = ctx.get_replacement(*arg);
            }
        }
        for arg in &mut guard_op.args {
            *arg = ctx.get_replacement(*arg);
        }
        OptimizationResult::Replace(guard_op)
    }

    fn optimize_guard_nonnull(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            if info.is_nonnull() {
                return OptimizationResult::Remove;
            }
        }

        // Constant-fold: non-zero constant is always nonnull.
        if let Some(value) = ctx.get_constant_int(obj_ref) {
            if value != 0 {
                return OptimizationResult::Remove;
            }
        }

        self.set_info(obj_ref, PtrInfo::NonNull);
        self.force_guard_fail_args(op, ctx)
    }

    fn optimize_guard_nonnull_class(
        &mut self,
        op: &Op,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            match info {
                PtrInfo::Virtual(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_) => {
                    return OptimizationResult::Remove;
                }
                PtrInfo::KnownClass {
                    is_nonnull: true, ..
                } => {
                    return OptimizationResult::Remove;
                }
                _ => {}
            }
        }

        self.set_info(
            obj_ref,
            PtrInfo::KnownClass {
                class_ptr: majit_ir::GcRef::NULL,
                is_nonnull: true,
            },
        );
        self.force_guard_fail_args(op, ctx)
    }

    fn optimize_guard_value(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        // If the object is already a known constant matching the guard value, remove
        if let Some(val) = ctx.get_constant(obj_ref) {
            if let Some(expected) = ctx.get_constant(ctx.get_replacement(op.arg(1))) {
                if val == expected {
                    return OptimizationResult::Remove;
                }
            }
        }

        self.force_guard_fail_args(op, ctx)
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
        let token_ref = ctx.emit(token_op);

        // The emitted ForceToken may reuse an OpRef index that previously
        // had PtrInfo::Virtual attached (from an earlier NEW_WITH_VTABLE
        // that was virtualized). Clear that to prevent accidental forcing
        // of unrelated virtuals when the vref struct is forced.
        self.set_info(token_ref, PtrInfo::NonNull);

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
        };
        self.set_info(op.pos, PtrInfo::VirtualStruct(vinfo));

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
        let vref_ref = ctx.get_replacement(op.arg(0));
        let obj_ref = ctx.get_replacement(op.arg(1));

        // Check if the virtual object arg is non-null (forced case)
        let obj_is_null = ctx.get_constant_int(obj_ref).is_some_and(|v| v == 0);

        // If vref is still virtual, update the virtual struct fields directly
        if let Some(info) = self.get_info_mut(vref_ref) {
            if info.is_virtual() {
                if let PtrInfo::VirtualStruct(vinfo) = info {
                    // Set forced field to the object (or keep NULL)
                    if !obj_is_null {
                        set_field(&mut vinfo.fields, VREF_FORCED_FIELD_INDEX, obj_ref);
                    }
                    // Set virtual_token to NULL (TOKEN_NONE)
                    let null_ref = self.emit_constant_int(ctx, 0);
                    // Re-borrow: get_info_mut again after emit_constant_int
                    if let Some(PtrInfo::VirtualStruct(vinfo)) = self.get_info_mut(vref_ref) {
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
            ctx.emit(set_forced);
        }

        // Set 'virtual_token' to NULL
        let null_ref = self.emit_constant_int(ctx, 0);
        let mut set_token = Op::new(OpCode::SetfieldGc, &[vref_ref, null_ref]);
        set_token.descr = Some(make_field_index_descr(VREF_VIRTUAL_TOKEN_FIELD_INDEX));
        ctx.emit(set_token);

        OptimizationResult::Remove
    }

    /// Handle operations that may cause virtuals to escape.
    fn optimize_escaping_op(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let forced = self.force_all_args(op, ctx);
        OptimizationResult::Replace(forced)
    }

    /// Extend JUMP args with current tracked virtualizable field values.
    ///
    /// Instead of forcing the frame at JUMP (which writes to heap),
    /// carry the values through JUMP args so they flow through the loop
    /// as virtual inputs. No heap access on the hot path.
    fn augment_jump_with_virtualizable(&self, op: &mut Op, ctx: &OptContext) {
        let frame_ref = ctx.get_replacement(OpRef(0));
        let vstate = match self.get_info(frame_ref) {
            Some(PtrInfo::Virtualizable(v)) => v,
            _ => return,
        };
        let config = match &self.vable_config {
            Some(c) => c,
            None => return,
        };
        // For each virtual input, add the current tracked value to JUMP args
        for (i, &offset) in config.static_field_offsets.iter().enumerate() {
            let virt_ref = self.vable_virtual_inputs[i].1;
            let current = vstate
                .fields
                .iter()
                .find_map(|(idx, opref)| {
                    if extract_field_offset(*idx) == Some(offset) {
                        Some(ctx.get_replacement(*opref))
                    } else {
                        None
                    }
                })
                .unwrap_or(ctx.get_replacement(virt_ref));
            op.args.push(current);
        }
    }

    /// Augment guard fail_args with tracked virtualizable field values.
    ///
    /// After the original fail_args, appends:
    ///   [next_instr, stack_depth, locals_len, local_0..N, stack_len, stack_0..M]
    fn augment_guard_with_virtualizable(&mut self, op: &mut Op, ctx: &mut OptContext) {
        let frame_ref = ctx.get_replacement(OpRef(0));
        let vstate = match self.get_info(frame_ref) {
            Some(PtrInfo::Virtualizable(v)) => v.clone(),
            _ => return,
        };
        let config = match &self.vable_config {
            Some(c) => c.clone(),
            None => return,
        };
        let fail_args = op.fail_args.get_or_insert_with(Default::default);

        // Static fields (e.g., next_instr, stack_depth)
        for &offset in &config.static_field_offsets {
            // Find the tracked value by matching offset in field descr indices
            let val = vstate.fields.iter().find_map(|(idx, opref)| {
                if extract_field_offset(*idx) == Some(offset) {
                    Some(*opref)
                } else {
                    None
                }
            });
            if let Some(v) = val {
                let forced = self.force_virtual(v, ctx);
                fail_args.push(ctx.get_replacement(forced));
            } else {
                // Not tracked — emit a read from the frame
                let mut read_op = Op::new(OpCode::GetfieldRawI, &[frame_ref]);
                read_op.descr = Some(make_field_index_descr(
                    0x1000_0000 | (((offset as u32) & 0x000f_ffff) << 4) | (0x7 << 1),
                ));
                let read_ref = ctx.emit(read_op);
                fail_args.push(read_ref);
            }
        }

        // Array fields (e.g., locals_w, value_stack_w)
        for (arr_cfg_idx, _offset) in config.array_field_offsets.iter().enumerate() {
            let arr_data = vstate.arrays.iter().find(|(i, _)| *i == arr_cfg_idx as u32);
            if let Some((_, elements)) = arr_data {
                // Array length
                let len_ref = self.emit_constant_int(ctx, elements.len() as i64);
                fail_args.push(len_ref);
                // Elements
                for elem in elements {
                    if elem.is_none() {
                        // Uninitialized slot — emit a zero placeholder
                        let zero = self.emit_constant_int(ctx, 0);
                        fail_args.push(zero);
                    } else {
                        let forced = self.force_virtual(*elem, ctx);
                        fail_args.push(ctx.get_replacement(forced));
                    }
                }
            } else {
                // Not tracked — length 0
                let zero = self.emit_constant_int(ctx, 0);
                fail_args.push(zero);
            }
        }
    }
}

impl Default for OptVirtualize {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptVirtualize {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Lazily create virtual input args for virtualizable fields
        if !self.vable_initialized {
            self.init_virtualizable(ctx);
            self.vable_initialized = true;
        }
        match op.opcode {
            // Allocation — create virtual
            OpCode::NewWithVtable => self.optimize_new_with_vtable(op, ctx),
            OpCode::New => self.optimize_new(op, ctx),
            OpCode::NewArray | OpCode::NewArrayClear => self.optimize_new_array(op, ctx),

            // Field access on potentially-virtual objects
            OpCode::SetfieldGc | OpCode::SetfieldRaw => self.optimize_setfield_gc(op, ctx),
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
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

            // Guards that can be eliminated based on known info
            OpCode::GuardClass => self.optimize_guard_class(op, ctx),
            OpCode::GuardNonnull => self.optimize_guard_nonnull(op, ctx),
            OpCode::GuardNonnullClass => self.optimize_guard_nonnull_class(op, ctx),
            OpCode::GuardValue => self.optimize_guard_value(op, ctx),

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

            // virtualize.py: optimize_COND_CALL — if the call is
            // OS_JIT_FORCE_VIRTUALIZABLE and the target is virtual, remove.
            OpCode::CondCallN => {
                if let Some(ref descr) = op.descr {
                    if let Some(cd) = descr.as_call_descr() {
                        let ei = cd.effect_info();
                        if ei.oopspec_index == OopSpecIndex::JitForceVirtualizable
                            && op.num_args() >= 3
                        {
                            let target = ctx.get_replacement(op.arg(2));
                            if self.is_virtual(target, ctx) {
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
                        if ei.oopspec_index == OopSpecIndex::JitForceVirtualizable {
                            let arg_idx = if op.opcode == OpCode::CallN { 1 } else { 1 };
                            if op.num_args() > arg_idx {
                                let target = ctx.get_replacement(op.arg(arg_idx));
                                if self.is_virtual(target, ctx) {
                                    return OptimizationResult::Remove;
                                }
                            }
                        }
                    }
                }
                self.optimize_escaping_op(op, ctx)
            }

            // Calls / escaping operations — force all virtual args
            _ if op.opcode.is_call() => self.optimize_escaping_op(op, ctx),

            // JUMP — extend args with virtualizable values (no force/heap write)
            OpCode::Jump if self.vable_config.is_some() => {
                let frame_ref = ctx.get_replacement(OpRef(0));
                // Force all virtual args EXCEPT the virtualizable frame
                let mut jump_op = op.clone();
                for arg in &mut jump_op.args {
                    let resolved = ctx.get_replacement(*arg);
                    if resolved == frame_ref {
                        if matches!(self.get_info(resolved), Some(PtrInfo::Virtualizable(_))) {
                            *arg = resolved;
                            continue;
                        }
                    }
                    if self.is_virtual(resolved, ctx) {
                        // TODO: extract_virtual_int_field works correctly but
                        // causes SIGSEGV because JUMP carries raw Int while
                        // Label expects boxed Ref. Enabling this requires
                        // preamble/Label type coordination (peel loop).
                        let forced = self.force_virtual(resolved, ctx);
                        *arg = ctx.get_replacement(forced);
                    } else {
                        *arg = resolved;
                    }
                }
                // Add virtualizable field values to JUMP args
                self.augment_jump_with_virtualizable(&mut jump_op, ctx);
                OptimizationResult::Replace(jump_op)
            }

            // JUMP (no virtualizable) / FINISH — force all virtual args
            OpCode::Jump | OpCode::Finish => self.optimize_escaping_op(op, ctx),

            // Escape ops (testing)
            OpCode::EscapeI | OpCode::EscapeR | OpCode::EscapeF | OpCode::EscapeN => {
                self.optimize_escaping_op(op, ctx)
            }

            // ── Record hint opcodes ──
            // These record information about values that downstream passes can use.
            // The hints themselves are removed (no code emitted).

            // RECORD_EXACT_CLASS(ref, class_const): record that ref has class class_const.
            // Enables subsequent GUARD_CLASS elimination.
            OpCode::RecordExactClass => {
                let ref_opref = ctx.get_replacement(op.arg(0));
                if let Some(&Value::Ref(class_ref)) = ctx.get_constant(op.arg(1)) {
                    self.set_info(
                        ref_opref,
                        PtrInfo::KnownClass {
                            class_ptr: class_ref,
                            is_nonnull: true,
                        },
                    );
                }
                OptimizationResult::Remove
            }

            // RECORD_EXACT_VALUE_I(ref, int_const): record that ref has exact int value.
            OpCode::RecordExactValueI => {
                let ref_opref = ctx.get_replacement(op.arg(0));
                if let Some(val) = ctx.get_constant_int(op.arg(1)) {
                    ctx.make_constant(ref_opref, Value::Int(val));
                }
                OptimizationResult::Remove
            }

            // RECORD_EXACT_VALUE_R(ref, ref_const): record that ref equals ref_const.
            OpCode::RecordExactValueR => {
                let ref_opref = ctx.get_replacement(op.arg(0));
                ctx.replace_op(ref_opref, ctx.get_replacement(op.arg(1)));
                OptimizationResult::Remove
            }

            // RECORD_KNOWN_RESULT: record that a call with given args produces known result.
            // Consumed by pure pass (CSE) — just remove here.
            OpCode::RecordKnownResult => OptimizationResult::Remove,

            // Everything else passes through, but force any virtual args first
            _ => {
                let is_guard = op.opcode.is_guard();

                // Force virtual op args
                for i in 0..op.num_args() {
                    let arg = ctx.get_replacement(op.arg(i));
                    if is_guard
                        && self.vable_config.is_some()
                        && arg == ctx.get_replacement(OpRef(0))
                    {
                        if matches!(self.get_info(arg), Some(PtrInfo::Virtualizable(_))) {
                            continue;
                        }
                    }
                    self.force_virtual(arg, ctx);
                }

                // Guards: force virtual fail_args and re-resolve after forcing
                if is_guard {
                    if let Some(ref fail_args) = op.fail_args {
                        for &fa in fail_args {
                            let resolved = ctx.get_replacement(fa);
                            self.force_virtual(resolved, ctx);
                        }
                    }
                    // Re-resolve fail_args after forcing (force may create new forwarding)
                    let mut guard_op = op.clone();
                    if let Some(ref mut fa) = guard_op.fail_args {
                        for arg in fa.iter_mut() {
                            *arg = ctx.get_replacement(*arg);
                        }
                    }
                    // Also re-resolve op args
                    for arg in &mut guard_op.args {
                        *arg = ctx.get_replacement(*arg);
                    }

                    if self.vable_config.is_some() {
                        self.augment_guard_with_virtualizable(&mut guard_op, ctx);
                    }
                    return OptimizationResult::Replace(guard_op);
                }

                OptimizationResult::PassOn
            }
        }
    }

    fn setup(&mut self) {
        self.ptr_info.clear();
        self.vable_array_ptrs.clear();
        self.vable_virtual_inputs.clear();
        self.vable_initialized = false;
        // If virtualizable config is set, mark OpRef(0) (frame) as Virtualizable.
        if self.vable_config.is_some() {
            self.set_info(
                OpRef(0),
                PtrInfo::Virtualizable(VirtualizableFieldState {
                    fields: vec![],
                    field_descrs: vec![],
                    arrays: vec![],
                }),
            );
        }
    }

    fn name(&self) -> &'static str {
        "virtualize"
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
        // Decode field_size from bits 1-3
        ((self.0 >> 1) & 0x7) as usize
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

fn make_field_index_descr(idx: u32) -> DescrRef {
    Arc::new(FieldIndexDescr(idx))
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
    use crate::optimizer::Optimizer;
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
        opt.optimize(ops)
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
                *arg = ctx.get_replacement(*arg);
            }

            match pass.propagate_forward(&resolved_op, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::Remove => {
                    // removed, nothing to do
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved_op);
                }
            }
        }

        pass.flush();
        ctx.new_operations
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
        // p0 = new_with_vtable(descr=size1)
        // guard_class(p0, ConstClass)   <- should be removed, class is known
        let sd = size_descr(1);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
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
        assert!(
            result.is_empty(),
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

    #[test]
    fn test_virtual_array_forced_at_jump() {
        // i50 = const 2
        // p0 = new_array(i50, descr=arr1)
        // i51 = const 0
        // setarrayitem_gc(p0, i51, i_val, descr=arr1)
        // jump(p0)   <- forces the array
        let ad = array_descr(20);

        let mut ops = vec![
            Op::with_descr(OpCode::NewArray, &[OpRef(50)], ad.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(51), OpRef(100)],
                ad.clone(),
            ),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(50), Value::Int(2)), (OpRef(51), Value::Int(0))];

        let result = run_pass_with_constants(&ops, &constants);

        // Should have forced: SameAsI (len const), NewArray, SameAsI (idx const) x2, SetarrayitemGc x2, Jump
        let has_new_array = result.iter().any(|o| o.opcode == OpCode::NewArray);
        assert!(has_new_array, "forced array should emit NEW_ARRAY");
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

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
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
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

    #[test]
    fn test_escape_forces_virtual() {
        // p0 = new_with_vtable(descr=size1)
        // escape_r(p0)   <- forces the virtual
        let sd = size_descr(1);

        let mut ops = vec![
            Op::with_descr(OpCode::NewWithVtable, &[], sd.clone()),
            Op::new(OpCode::EscapeR, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let result = run_pass(&ops);
        assert!(
            result.len() >= 2,
            "escape should force virtual; got {} ops",
            result.len()
        );
        assert_eq!(result[0].opcode, OpCode::NewWithVtable);
        assert_eq!(result.last().unwrap().opcode, OpCode::EscapeR);
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
            pass.set_info(
                opref,
                PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                    size,
                    entries: Vec::new(),
                }),
            );
        }

        for op in ops {
            let mut resolved_op = op.clone();
            for arg in &mut resolved_op.args {
                *arg = ctx.get_replacement(*arg);
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
            }
        }

        pass.flush();
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
    fn test_raw_buffer_forced_at_call() {
        // Virtual raw buffer escapes at a call, forcing materialization.
        // raw_store(buf, 0, val)
        // call_n(buf)   <- force
        let mut ops = vec![
            Op::new(OpCode::RawStore, &[OpRef(0), OpRef(100), OpRef(200)]), // pos=0
            Op::new(OpCode::CallN, &[OpRef(0)]),                            // pos=1
        ];
        assign_positions(&mut ops);

        let constants = vec![(OpRef(100), Value::Int(0))];
        let raw_bufs = vec![(OpRef(0), 32)];

        let result = run_pass_with_raw_buffer(&ops, &constants, &raw_bufs);
        // Should have forced: allocation + raw_store + call
        assert!(
            result.len() >= 2,
            "forced raw buffer should emit allocation + call; got {} ops: {:?}",
            result.len(),
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        assert_eq!(
            result.last().unwrap().opcode,
            OpCode::CallN,
            "last op should be CALL_N"
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
}
