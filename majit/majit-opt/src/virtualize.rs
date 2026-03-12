/// Virtualize optimization pass: remove heap allocations for non-escaping objects.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualize.py.
///
/// Tracks "virtual" objects — allocations that never escape the trace.
/// Instead of emitting the allocation, fields are tracked in the optimizer.
/// If a virtual escapes (e.g., passed to a call or stored in a non-virtual),
/// it gets "forced" (materialized by emitting the allocation + setfield ops).
use std::sync::Arc;

use majit_ir::{Descr, DescrRef, FieldDescr, Op, OpCode, OpRef, Value};

use crate::info::{
    PtrInfo, VirtualArrayInfo, VirtualArrayStructInfo, VirtualInfo, VirtualRawBufferInfo,
    VirtualStructInfo,
};
use crate::{OptContext, OptimizationPass, PassResult};

/// The virtualize optimization pass.
pub struct OptVirtualize {
    /// Per-operation PtrInfo, indexed by OpRef.0.
    ptr_info: Vec<Option<PtrInfo>>,
}

impl OptVirtualize {
    pub fn new() -> Self {
        OptVirtualize {
            ptr_info: Vec::new(),
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
            _ => resolved,
        }
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
            set_op.descr = Some(make_field_index_descr(field_idx));
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
        for (field_idx, value_ref) in vinfo.fields {
            let value_ref = self.force_virtual(value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
            set_op.descr = Some(make_field_index_descr(field_idx));
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
        for (offset, value_ref) in &vinfo.entries {
            let value_ref = self.force_virtual(*value_ref, ctx);
            let value_ref = ctx.get_replacement(value_ref);
            let offset_ref = self.emit_constant_int(ctx, *offset as i64);
            let set_op = Op::new(OpCode::RawStore, &[alloc_ref, offset_ref, value_ref]);
            ctx.emit(set_op);
        }

        alloc_ref
    }

    /// Force all arguments that are virtual.
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
        // Emit a SameAsI(dummy) that represents the constant. The optimizer's
        // constant table stores the value so downstream passes see it as a constant.
        let mut op = Op::new(OpCode::SameAsI, &[OpRef::NONE]);
        op.pos = OpRef::NONE;
        let opref = ctx.emit(op);
        ctx.make_constant(opref, Value::Int(value));
        opref
    }

    // ── Per-opcode handlers ──

    fn optimize_new_with_vtable(&mut self, op: &Op, _ctx: &mut OptContext) -> PassResult {
        let descr = op.descr.clone().expect("NEW_WITH_VTABLE needs descr");
        let vinfo = VirtualInfo {
            descr,
            known_class: None,
            fields: Vec::new(),
        };
        self.set_info(op.pos, PtrInfo::Virtual(vinfo));
        PassResult::Remove
    }

    fn optimize_new(&mut self, op: &Op, _ctx: &mut OptContext) -> PassResult {
        let descr = op.descr.clone().expect("NEW needs descr");
        let vinfo = VirtualStructInfo {
            descr,
            fields: Vec::new(),
        };
        self.set_info(op.pos, PtrInfo::VirtualStruct(vinfo));
        PassResult::Remove
    }

    fn optimize_new_array(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let size_ref = op.arg(0);
        if let Some(size) = ctx.get_constant_int(size_ref) {
            if size >= 0 && size <= 1024 {
                let descr = op.descr.clone().expect("NEW_ARRAY needs descr");
                let items = vec![OpRef::NONE; size as usize];
                let vinfo = VirtualArrayInfo { descr, items };
                self.set_info(op.pos, PtrInfo::VirtualArray(vinfo));
                return PassResult::Remove;
            }
        }
        PassResult::PassOn
    }

    fn optimize_new_array_clear(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        self.optimize_new_array(op, ctx)
    }

    fn optimize_setfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let struct_ref = ctx.get_replacement(op.arg(0));
        let value_ref = ctx.get_replacement(op.arg(1));
        let field_idx = descr_index(&op.descr);

        if let Some(info) = self.get_info_mut(struct_ref) {
            if info.is_virtual() {
                match info {
                    PtrInfo::Virtual(vinfo) => {
                        set_field(&mut vinfo.fields, field_idx, value_ref);
                        return PassResult::Remove;
                    }
                    PtrInfo::VirtualStruct(vinfo) => {
                        set_field(&mut vinfo.fields, field_idx, value_ref);
                        return PassResult::Remove;
                    }
                    _ => {}
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_getfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let struct_ref = ctx.get_replacement(op.arg(0));
        let field_idx = descr_index(&op.descr);

        if let Some(info) = self.get_info(struct_ref) {
            if info.is_virtual() {
                let field_val = match info {
                    PtrInfo::Virtual(vinfo) => get_field(&vinfo.fields, field_idx),
                    PtrInfo::VirtualStruct(vinfo) => get_field(&vinfo.fields, field_idx),
                    _ => None,
                };
                if let Some(val_ref) = field_val {
                    ctx.replace_op(op.pos, val_ref);
                    return PassResult::Remove;
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_setarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let value_ref = ctx.get_replacement(op.arg(2));

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = self.get_info_mut(array_ref) {
                if let PtrInfo::VirtualArray(vinfo) = info {
                    let idx = index as usize;
                    if idx < vinfo.items.len() {
                        vinfo.items[idx] = value_ref;
                        return PassResult::Remove;
                    }
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_getarrayitem_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(info) = self.get_info(array_ref) {
                if let PtrInfo::VirtualArray(vinfo) = info {
                    let idx = index as usize;
                    if idx < vinfo.items.len() {
                        let item_ref = vinfo.items[idx];
                        if !item_ref.is_none() {
                            ctx.replace_op(op.pos, item_ref);
                            return PassResult::Remove;
                        }
                    }
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_arraylen_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let array_ref = ctx.get_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = self.get_info(array_ref) {
            let len = vinfo.items.len() as i64;
            ctx.make_constant(op.pos, Value::Int(len));
            return PassResult::Remove;
        }
        PassResult::PassOn
    }

    fn optimize_strlen(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let str_ref = ctx.get_replacement(op.arg(0));

        if let Some(PtrInfo::VirtualArray(vinfo)) = self.get_info(str_ref) {
            let len = vinfo.items.len() as i64;
            ctx.make_constant(op.pos, Value::Int(len));
            return PassResult::Remove;
        }
        PassResult::PassOn
    }

    fn optimize_getinteriorfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let array_ref = ctx.get_replacement(op.arg(0));
        let index_ref = op.arg(1);
        let field_idx = descr_index(&op.descr);

        if let Some(index) = ctx.get_constant_int(index_ref) {
            if let Some(PtrInfo::VirtualArrayStruct(vinfo)) = self.get_info(array_ref) {
                let elem_idx = index as usize;
                if elem_idx < vinfo.element_fields.len() {
                    if let Some(val) = get_field(&vinfo.element_fields[elem_idx], field_idx) {
                        ctx.replace_op(op.pos, val);
                        return PassResult::Remove;
                    }
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_setinteriorfield_gc(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
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
                        return PassResult::Remove;
                    }
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_raw_load(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let buf_ref = ctx.get_replacement(op.arg(0));
        let offset_ref = op.arg(1);

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(PtrInfo::VirtualRawBuffer(vinfo)) = self.get_info(buf_ref) {
                let offset = offset as usize;
                if let Some((_, val_ref)) = vinfo.entries.iter().find(|(off, _)| *off == offset) {
                    ctx.replace_op(op.pos, *val_ref);
                    return PassResult::Remove;
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_raw_store(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let buf_ref = ctx.get_replacement(op.arg(0));
        let offset_ref = op.arg(1);
        let value_ref = ctx.get_replacement(op.arg(2));

        if let Some(offset) = ctx.get_constant_int(offset_ref) {
            if let Some(info) = self.get_info_mut(buf_ref) {
                if let PtrInfo::VirtualRawBuffer(vinfo) = info {
                    let offset = offset as usize;
                    // Update or insert entry
                    if let Some(entry) = vinfo.entries.iter_mut().find(|(off, _)| *off == offset) {
                        entry.1 = value_ref;
                    } else {
                        vinfo.entries.push((offset, value_ref));
                    }
                    return PassResult::Remove;
                }
            }
        }
        PassResult::PassOn
    }

    fn optimize_guard_class(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            match info {
                // Virtual objects have a known allocation type — guard is redundant
                PtrInfo::Virtual(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_) => {
                    return PassResult::Remove;
                }
                PtrInfo::KnownClass { .. } => {
                    // Class already known from prior guard or virtual forcing
                    return PassResult::Remove;
                }
                _ => {}
            }
        }

        // Record the class for future lookups.
        // arg(1) should be the expected class, but we can't easily read it
        // as a GcRef here. Still emit the guard.
        self.set_info(
            obj_ref,
            PtrInfo::KnownClass {
                class_ptr: majit_ir::GcRef::NULL, // placeholder
                is_nonnull: true,
            },
        );
        PassResult::PassOn
    }

    fn optimize_guard_nonnull(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            if info.is_nonnull() {
                return PassResult::Remove;
            }
        }

        self.set_info(obj_ref, PtrInfo::NonNull);
        PassResult::PassOn
    }

    fn optimize_guard_nonnull_class(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        if let Some(info) = self.get_info(obj_ref) {
            match info {
                PtrInfo::Virtual(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_) => {
                    return PassResult::Remove;
                }
                PtrInfo::KnownClass {
                    is_nonnull: true, ..
                } => {
                    return PassResult::Remove;
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
        PassResult::PassOn
    }

    fn optimize_guard_value(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let obj_ref = ctx.get_replacement(op.arg(0));

        // If the object is already a known constant matching the guard value, remove
        if let Some(val) = ctx.get_constant(obj_ref) {
            if let Some(expected) = ctx.get_constant(ctx.get_replacement(op.arg(1))) {
                if val == expected {
                    return PassResult::Remove;
                }
            }
        }

        PassResult::PassOn
    }

    /// Handle operations that may cause virtuals to escape.
    fn optimize_escaping_op(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let forced = self.force_all_args(op, ctx);
        PassResult::Emit(forced)
    }
}

impl Default for OptVirtualize {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for OptVirtualize {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        match op.opcode {
            // Allocation — create virtual
            OpCode::NewWithVtable => self.optimize_new_with_vtable(op, ctx),
            OpCode::New => self.optimize_new(op, ctx),
            OpCode::NewArray | OpCode::NewArrayClear => self.optimize_new_array(op, ctx),

            // Field access on potentially-virtual objects
            OpCode::SetfieldGc => self.optimize_setfield_gc(op, ctx),
            OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                self.optimize_getfield_gc(op, ctx)
            }

            // Array access on potentially-virtual arrays
            OpCode::SetarrayitemGc => self.optimize_setarrayitem_gc(op, ctx),
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                self.optimize_getarrayitem_gc(op, ctx)
            }

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

            // Calls / escaping operations — force all virtual args
            _ if op.opcode.is_call() => self.optimize_escaping_op(op, ctx),

            // JUMP / FINISH — force all virtual jump args
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
                PassResult::Remove
            }

            // RECORD_EXACT_VALUE_I(ref, int_const): record that ref has exact int value.
            OpCode::RecordExactValueI => {
                let ref_opref = ctx.get_replacement(op.arg(0));
                if let Some(val) = ctx.get_constant_int(op.arg(1)) {
                    ctx.make_constant(ref_opref, Value::Int(val));
                }
                PassResult::Remove
            }

            // RECORD_EXACT_VALUE_R(ref, ref_const): record that ref equals ref_const.
            OpCode::RecordExactValueR => {
                let ref_opref = ctx.get_replacement(op.arg(0));
                ctx.replace_op(ref_opref, ctx.get_replacement(op.arg(1)));
                PassResult::Remove
            }

            // RECORD_KNOWN_RESULT: record that a call with given args produces known result.
            // Consumed by pure pass (CSE) — just remove here.
            OpCode::RecordKnownResult => PassResult::Remove,

            // Everything else passes through
            _ => PassResult::PassOn,
        }
    }

    fn name(&self) -> &'static str {
        "virtualize"
    }
}

// ── PtrInfo helpers ──

impl PtrInfo {
    fn is_virtual(&self) -> bool {
        matches!(
            self,
            PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
        )
    }

    fn is_nonnull(&self) -> bool {
        match self {
            PtrInfo::NonNull => true,
            PtrInfo::KnownClass { is_nonnull, .. } => *is_nonnull,
            PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_) => true,
            PtrInfo::Constant(gcref) => !gcref.is_null(),
        }
    }
}

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
}

impl FieldDescr for FieldIndexDescr {
    fn offset(&self) -> usize {
        self.0 as usize
    }

    fn field_size(&self) -> usize {
        8
    }

    fn field_type(&self) -> majit_ir::Type {
        majit_ir::Type::Int
    }
}

fn make_field_index_descr(idx: u32) -> DescrRef {
    Arc::new(FieldIndexDescr(idx))
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
                PassResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                PassResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                PassResult::Remove => {
                    // removed, nothing to do
                }
                PassResult::PassOn => {
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
}
