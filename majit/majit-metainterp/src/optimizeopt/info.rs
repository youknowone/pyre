use crate::optimizeopt::intutils::IntBound;
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::{Descr, DescrRef, GcRef, Op, OpCode, OpRef, Value};

/// resoperation.py: AbstractResOpOrInputArg._forwarded
///
/// RPython uses a single `_forwarded` field per Box that holds EITHER:
/// - None (no forwarding, no info)
/// - another Box (forwarding to that box)
/// - a PtrInfo instance (terminal info)
/// - a Const value (terminal — RPython: _forwarded = constbox)
///
/// `get_box_replacement` follows Box→Box links, stops at None/PtrInfo/Const.
/// `getptrinfo` reads PtrInfo from the terminal Box.
#[derive(Clone, Debug)]
pub enum Forwarded {
    /// No forwarding or info set.
    None,
    /// Forwarding to another OpRef (RPython: _forwarded = other_box).
    Op(OpRef),
    /// Terminal info (RPython: _forwarded = PtrInfo instance).
    Info(PtrInfo),
    /// Terminal constant (RPython: _forwarded = constbox).
    /// optimizer.py:432: box.set_forwarded(constbox).
    Const(majit_ir::Value),
    /// Terminal IntBound (RPython: _forwarded = IntBound instance).
    /// intutils.py:73: IntBound(AbstractInfo) — stored directly in
    /// _forwarded, retrieved by optimizer.py:99 getintbound().
    IntBound(crate::optimizeopt::intutils::IntBound),
}

impl Default for Forwarded {
    fn default() -> Self {
        Forwarded::None
    }
}

/// shortpreamble.py:11-49: PreambleOp
///
/// Wrapper stored in PtrInfo._fields during Phase 2 import.
/// When `_getfield` (heap.py:177-187) encounters this in a field slot,
/// it calls `force_op_from_preamble()` to lazily resolve the value
/// via the short preamble builder.
///
/// RPython stores PreambleOp directly in `_fields[]` (Python's dynamic
/// typing). Rust requires a separate storage (`preamble_fields`) with
/// the same read-before-regular-fields semantics.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// Phase 1 result box — RPython: PreambleOp.op (aka HeapOp.res)
    pub op: OpRef,
    /// Fresh Phase 2 OpRef with distinct identity from label_args.
    /// Returned by force_op_from_preamble as the resolved field value.
    /// (Rust equivalent of RPython's Phase 1 Box identity isolation.)
    pub resolved: OpRef,
    /// RPython: PreambleOp.invented_name
    pub invented_name: bool,
}

/// Information about an operation's result, attached during optimization.
///
/// info.py: AbstractInfo hierarchy — the base class for all optimization info.
#[derive(Clone, Debug)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// Known constant value (integer or pointer).
    Constant(Value),
    /// Known integer bounds.
    IntBound(IntBound),
    /// Pointer info (non-null, known class, virtual, etc.).
    Ptr(PtrInfo),
    /// Known constant float value.
    /// info.py: FloatConstInfo — tracks float constants separately
    /// because they need special boxing on 32-bit platforms.
    FloatConst(f64),
}

impl OpInfo {
    pub fn is_constant(&self) -> bool {
        matches!(
            self,
            OpInfo::Constant(_) | OpInfo::FloatConst(_) | OpInfo::Ptr(PtrInfo::Constant(_))
        )
    }

    pub fn get_constant(&self) -> Option<&Value> {
        match self {
            OpInfo::Constant(v) => Some(v),
            _ => None,
        }
    }

    /// Get the constant float value if this is a FloatConst.
    pub fn get_constant_float(&self) -> Option<f64> {
        match self {
            OpInfo::FloatConst(f) => Some(*f),
            OpInfo::Constant(Value::Float(f)) => Some(*f),
            _ => None,
        }
    }

    pub fn get_int_bound(&self) -> Option<&IntBound> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this info is known non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            OpInfo::Ptr(ptr) => ptr.is_nonnull(),
            OpInfo::Constant(Value::Int(v)) => *v != 0,
            _ => false,
        }
    }

    /// Whether this info represents a virtual (allocation-removed) object.
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(self, OpInfo::Ptr(ptr) if ptr.is_virtual())
    }

    /// Get the PtrInfo if present.
    pub fn get_ptr_info(&self) -> Option<&PtrInfo> {
        match self {
            OpInfo::Ptr(p) => Some(p),
            _ => None,
        }
    }
}

/// Information about a pointer value.
///
/// info.py: PtrInfo hierarchy:
///   NonNullPtrInfo → AbstractVirtualPtrInfo → {InstancePtrInfo, StructPtrInfo,
///   ArrayPtrInfo, ArrayStructInfo, RawBufferPtrInfo, RawStructPtrInfo, RawSlicePtrInfo}
///   ConstPtrInfo
#[derive(Clone, Debug)]
pub enum PtrInfo {
    /// Known to be non-null, nothing else.
    /// info.py: NonNullPtrInfo
    NonNull {
        /// info.py:91-92: NonNullPtrInfo.last_guard_pos = -1
        last_guard_pos: i32,
    },
    /// Known constant pointer.
    /// info.py: ConstPtrInfo (does NOT inherit NonNullPtrInfo)
    Constant(GcRef),
    /// Known class (type) of the object.
    /// info.py: NonNullPtrInfo with _known_class set
    KnownClass {
        class_ptr: GcRef,
        is_nonnull: bool,
        /// info.py:91-92
        last_guard_pos: i32,
    },
    /// Non-virtual GC object with cached field info.
    /// info.py: InstancePtrInfo (is_virtual = False)
    Instance(InstancePtrInfo),
    /// Non-virtual GC struct with cached field info.
    /// info.py: StructPtrInfo (is_virtual = False)
    Struct(StructPtrInfo),
    /// Non-virtual GC array with cached item info and lenbound.
    /// info.py: ArrayPtrInfo (is_virtual = False)
    Array(ArrayPtrInfo),
    /// Virtual object (allocation removed by the optimizer).
    /// info.py: InstancePtrInfo
    Virtual(VirtualInfo),
    /// Virtual array.
    /// info.py: ArrayPtrInfo
    VirtualArray(VirtualArrayInfo),
    /// Virtual struct (no vtable).
    /// info.py: StructPtrInfo
    VirtualStruct(VirtualStructInfo),
    /// Virtual array of structs (interior field access).
    /// info.py: ArrayStructInfo
    VirtualArrayStruct(VirtualArrayStructInfo),
    /// Virtual raw buffer.
    /// info.py: RawBufferPtrInfo
    VirtualRawBuffer(VirtualRawBufferInfo),
    /// Virtualizable object (interpreter frame).
    Virtualizable(VirtualizableFieldState),
    /// vstring.py:50: StrPtrInfo — string with known length bounds.
    /// Tracks lenbound (IntBound) and mode (string vs unicode).
    Str(StrPtrInfo),
}

/// vstring.py:50-140: StrPtrInfo
#[derive(Clone, Debug)]
pub struct StrPtrInfo {
    /// vstring.py: self.lenbound — IntBound for string length.
    pub lenbound: Option<IntBound>,
    /// vstring.py: self.mode — 0 = mode_string, 1 = mode_unicode.
    pub mode: u8,
    /// vstring.py: self.length — known exact length (-1 if unknown).
    pub length: i32,
    /// info.py:91-92: last_guard_pos
    pub last_guard_pos: i32,
}

impl PtrInfo {
    // ── Constructors (info.py: factory methods) ──

    /// Create a NonNull PtrInfo.
    pub fn nonnull() -> Self {
        PtrInfo::NonNull { last_guard_pos: -1 }
    }

    // ── info.py:100-118: last_guard_pos methods ──

    /// info.py:100-103: get_last_guard
    pub fn get_last_guard_pos(&self) -> Option<usize> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::KnownClass { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None, // ConstPtrInfo has no last_guard_pos
        };
        if pos < 0 { None } else { Some(pos as usize) }
    }

    /// Raw last_guard_pos value as i32 (-1 if none).
    pub fn last_guard_pos(&self) -> Option<i32> {
        let pos = match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::KnownClass { last_guard_pos, .. } => *last_guard_pos,
            PtrInfo::Instance(i) => i.last_guard_pos,
            PtrInfo::Struct(s) => s.last_guard_pos,
            PtrInfo::Array(a) => a.last_guard_pos,
            PtrInfo::Virtual(v) => v.last_guard_pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos,
            PtrInfo::Str(s) => s.last_guard_pos,
            PtrInfo::Constant(_) => return None,
        };
        Some(pos)
    }

    /// info.py:111-118: mark_last_guard
    pub fn set_last_guard_pos(&mut self, pos: i32) {
        match self {
            PtrInfo::NonNull { last_guard_pos, .. } => *last_guard_pos = pos,
            PtrInfo::KnownClass { last_guard_pos, .. } => *last_guard_pos = pos,
            PtrInfo::Instance(i) => i.last_guard_pos = pos,
            PtrInfo::Struct(s) => s.last_guard_pos = pos,
            PtrInfo::Array(a) => a.last_guard_pos = pos,
            PtrInfo::Virtual(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArray(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualArrayStruct(v) => v.last_guard_pos = pos,
            PtrInfo::VirtualRawBuffer(v) => v.last_guard_pos = pos,
            PtrInfo::Virtualizable(v) => v.last_guard_pos = pos,
            PtrInfo::Str(s) => s.last_guard_pos = pos,
            PtrInfo::Constant(_) => {} // ConstPtrInfo: no-op
        }
    }

    /// info.py:108-109: reset_last_guard_pos
    pub fn reset_last_guard_pos(&mut self) {
        self.set_last_guard_pos(-1);
    }

    /// Create a Constant PtrInfo.
    pub fn constant(gcref: GcRef) -> Self {
        PtrInfo::Constant(gcref)
    }

    /// Create a KnownClass PtrInfo.
    pub fn known_class(class_ptr: GcRef, is_nonnull: bool) -> Self {
        PtrInfo::KnownClass {
            class_ptr,
            is_nonnull,
            last_guard_pos: -1,
        }
    }

    /// Create a non-virtual InstancePtrInfo.
    pub fn instance(descr: Option<DescrRef>, known_class: Option<GcRef>) -> Self {
        PtrInfo::Instance(InstancePtrInfo {
            descr,
            known_class,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            preamble_fields: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual StructPtrInfo.
    pub fn struct_ptr(descr: DescrRef) -> Self {
        PtrInfo::Struct(StructPtrInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            preamble_fields: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a non-virtual ArrayPtrInfo.
    pub fn array(descr: DescrRef, lenbound: IntBound) -> Self {
        PtrInfo::Array(ArrayPtrInfo {
            descr,
            lenbound,
            items: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a Virtual PtrInfo (allocation removed).
    pub fn virtual_obj(descr: DescrRef, known_class: Option<GcRef>) -> Self {
        PtrInfo::Virtual(VirtualInfo {
            descr,
            known_class,
            ob_type_descr: None,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        })
    }

    /// Create a VirtualArray PtrInfo.
    pub fn virtual_array(descr: DescrRef, length: usize, clear: bool) -> Self {
        PtrInfo::VirtualArray(VirtualArrayInfo {
            descr,
            clear,
            items: vec![OpRef::NONE; length],
            last_guard_pos: -1,
        })
    }

    /// Create a VirtualStruct PtrInfo.
    pub fn virtual_struct(descr: DescrRef) -> Self {
        PtrInfo::VirtualStruct(VirtualStructInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
            last_guard_pos: -1,
        })
    }

    // ── Query methods ──

    /// Whether this pointer is known to be non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            PtrInfo::NonNull { .. } => true,
            PtrInfo::Constant(gcref) => !gcref.is_null(),
            PtrInfo::KnownClass { is_nonnull, .. } => *is_nonnull,
            PtrInfo::Instance(_)
            | PtrInfo::Struct(_)
            | PtrInfo::Array(_)
            | PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_)
            | PtrInfo::Virtualizable(_)
            | PtrInfo::Str(_) => true,
        }
    }

    /// Whether this pointer is a virtual (allocation removed).
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
        )
    }

    /// Whether this is a constant pointer.
    /// info.py: isinstance(info, ConstPtrInfo)
    pub fn is_constant(&self) -> bool {
        matches!(self, PtrInfo::Constant(_))
    }

    /// Get the known class, if any.
    /// info.py: get_known_class_or_none()
    pub fn get_known_class(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::KnownClass { class_ptr, .. } => Some(class_ptr),
            PtrInfo::Instance(v) => v.known_class.as_ref(),
            PtrInfo::Virtual(v) => v.known_class.as_ref(),
            _ => None,
        }
    }

    /// Get constant GcRef value if this is a constant pointer.
    pub fn get_constant_ref(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::Constant(r) => Some(r),
            _ => None,
        }
    }

    /// info.py:83: make_guards(op, short, optimizer)
    /// Append guard operations to `short` that check this PtrInfo's
    /// properties hold for `op`. Used by use_box (shortpreamble.py:382).
    pub fn make_guards(
        &self,
        op: OpRef,
        short: &mut Vec<Op>,
        const_pool: &mut Vec<(OpRef, Value)>,
    ) {
        match self {
            // info.py:83-84: PtrInfo base — no-op
            PtrInfo::NonNull { .. } => {
                // info.py:120-122: NonNullPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
            }
            PtrInfo::KnownClass { class_ptr, .. } => {
                // info.py:336-345: InstancePtrInfo.make_guards with known_class
                let class_ref = Self::alloc_guard_const(const_pool, Value::Ref(*class_ptr));
                short.push(Op::new(OpCode::GuardNonnullClass, &[op, class_ref]));
            }
            PtrInfo::Instance(info) => {
                // info.py:336-353: InstancePtrInfo.make_guards
                if let Some(cls) = &info.known_class {
                    let class_ref = Self::alloc_guard_const(const_pool, Value::Ref(*cls));
                    short.push(Op::new(OpCode::GuardNonnullClass, &[op, class_ref]));
                } else {
                    // info.py:353: fallback to AbstractStructPtrInfo (NonNull)
                    short.push(Op::new(OpCode::GuardNonnull, &[op]));
                }
            }
            PtrInfo::Struct(_) => {
                // info.py:360-366: StructPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
            }
            PtrInfo::Constant(gcref) => {
                // info.py:715-716: ConstPtrInfo.make_guards
                let c = Self::alloc_guard_const(const_pool, Value::Ref(*gcref));
                short.push(Op::new(OpCode::GuardValue, &[op, c]));
            }
            PtrInfo::Array(_) => {
                // info.py:632-633: ArrayPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
            }
            PtrInfo::Str(sinfo) => {
                // vstring.py:116-126: StrPtrInfo.make_guards
                short.push(Op::new(OpCode::GuardNonnull, &[op]));
                if let Some(ref bound) = sinfo.lenbound {
                    if bound.lower >= 1 {
                        let lenop_code = if sinfo.mode == 0 {
                            OpCode::Strlen
                        } else {
                            OpCode::Unicodelen
                        };
                        let lenop = Op::new(lenop_code, &[op]);
                        let lenop_pos = lenop.pos;
                        short.push(lenop);
                        // intutils.py: IntBound.make_guards → generate bound guards
                        for (guard_opcode, value) in bound.make_guards() {
                            let c = Self::alloc_guard_const(const_pool, Value::Int(value));
                            short.push(Op::new(guard_opcode, &[lenop_pos, c]));
                        }
                    }
                }
            }
            // Virtuals/Virtualizable: no guards needed in short preamble
            _ => {}
        }
    }

    /// Record a constant needed by a short preamble guard.
    /// The caller (collect_use_box_guards) allocates proper OpRefs
    /// via alloc_op_position and registers them in the constant map.
    fn alloc_guard_const(const_pool: &mut Vec<(OpRef, Value)>, value: Value) -> OpRef {
        // Check if we already have this value in the pool
        for &(ref_existing, ref val_existing) in const_pool.iter() {
            if *val_existing == value {
                return ref_existing;
            }
        }
        // Placeholder — caller replaces with alloc_op_position result
        let placeholder = OpRef(u32::MAX - const_pool.len() as u32);
        const_pool.push((placeholder, value));
        placeholder
    }

    /// Get the string length from a constant string pointer.
    /// info.py: getstrlen() on ConstPtrInfo
    pub fn getstrlen(&self) -> Option<usize> {
        // Only meaningful for constant string objects.
        // In RPython, this reads the string header to get the length.
        // In our implementation, GcRef doesn't carry string metadata,
        // so we return None. This can be overridden when GcRef is enriched.
        None
    }

    /// Get the string hash from a constant string pointer.
    /// info.py: getstrhash() on ConstPtrInfo
    pub fn getstrhash(&self) -> Option<i64> {
        None
    }

    /// Count the number of fields/items in this virtual object.
    /// info.py: _get_num_items() / num_fields
    pub fn num_fields(&self) -> usize {
        match self {
            PtrInfo::Instance(v) => v.fields.len(),
            PtrInfo::Struct(v) => v.fields.len(),
            PtrInfo::Array(v) => v.items.len(),
            PtrInfo::Virtual(v) => v.fields.len(),
            PtrInfo::VirtualArray(v) => v.items.len(),
            PtrInfo::VirtualStruct(v) => v.fields.len(),
            PtrInfo::VirtualArrayStruct(v) => v.element_fields.len(),
            PtrInfo::VirtualRawBuffer(v) => v.entries.len(),
            _ => 0,
        }
    }

    /// Enumerate all OpRef values stored in this virtual's fields/items.
    /// info.py: visitor_walk_recursive — walks all fields of a virtual.
    pub fn visitor_walk_recursive(&self) -> Vec<OpRef> {
        match self {
            PtrInfo::Instance(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::Struct(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::Array(v) => v.items.clone(),
            PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArray(v) => v.items.clone(),
            PtrInfo::VirtualStruct(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArrayStruct(v) => v
                .element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, r)| *r))
                .collect(),
            PtrInfo::VirtualRawBuffer(v) => v.entries.iter().map(|(_, _, r)| *r).collect(),
            PtrInfo::Virtualizable(v) => {
                let mut refs: Vec<OpRef> = v.fields.iter().map(|(_, r)| *r).collect();
                for (_, items) in &v.arrays {
                    refs.extend(items.iter().copied());
                }
                refs
            }
            _ => Vec::new(),
        }
    }

    /// info.py: force_at_the_end_of_preamble(op, optforce, rec)
    ///
    /// RPython does not blindly materialize every virtual at the end of the
    /// preamble. Struct-like virtuals recurse into pointer children and update
    /// those field/item boxes in place, while leaving the top-level virtual in
    /// the exported virtual state.
    pub fn force_at_the_end_of_preamble<F>(&mut self, mut recurse: F)
    where
        F: FnMut(OpRef) -> OpRef,
    {
        match self {
            PtrInfo::Virtual(v) => {
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(*field);
                    }
                }
            }
            PtrInfo::VirtualStruct(v) => {
                for (_, field) in &mut v.fields {
                    if !field.is_none() {
                        *field = recurse(*field);
                    }
                }
            }
            PtrInfo::VirtualArray(v) => {
                for item in &mut v.items {
                    if !item.is_none() {
                        *item = recurse(*item);
                    }
                }
            }
            PtrInfo::VirtualArrayStruct(v) => {
                for fields in &mut v.element_fields {
                    for (_, field) in fields {
                        if !field.is_none() {
                            *field = recurse(*field);
                        }
                    }
                }
            }
            PtrInfo::VirtualRawBuffer(v) => {
                for (_, _, field) in &mut v.entries {
                    if !field.is_none() {
                        *field = recurse(*field);
                    }
                }
            }
            _ => {}
        }
    }

    /// info.py:137-160 / 222-226: force_box() emits the allocation and
    /// field writes via emit_extra(), recursively forcing child virtuals.
    ///
    /// Generated ops are routed via emit_extra() (RPython
    /// emit_extra parity) so downstream passes can observe them.
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
        self.force_box_impl(opref, ctx)
    }

    fn force_box_impl(&mut self, opref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
        use majit_ir::{Op, OpCode};

        fn force_child(value_ref: OpRef, ctx: &mut crate::optimizeopt::OptContext) -> OpRef {
            let value_ref = ctx.get_box_replacement(value_ref);
            if ctx.get_ptr_info(value_ref).is_some_and(|i| i.is_virtual()) {
                let mut info = ctx.take_ptr_info(value_ref).unwrap();
                let forced = info.force_box_impl(value_ref, ctx);
                return ctx.get_box_replacement(forced);
            }
            value_ref
        }

        // RPython info.py:148,226: optforce.emit_extra(op)
        // `optforce` determines where emitted ops enter the pass chain:
        //   optforce=Optimizer (in_final_emission) → emit directly
        //   optforce=OptEarlyForce → route from earlyforce.next (= heap)
        // When called from EarlyForce pass, current_pass_idx == earlyforce_idx
        // so emit_extra automatically routes from earlyforce.next.
        // When called from _emit_operation, in_final_emission=true → direct.
        let emit_op = |ctx: &mut crate::optimizeopt::OptContext, op: Op| -> OpRef {
            if ctx.in_final_emission {
                ctx.emit(op)
            } else {
                ctx.emit_extra(ctx.current_pass_idx, op)
            }
        };

        // RPython info.py:140-145: immutable virtual filled with constants
        // → constant fold to a compile-time constant pointer.
        if self.is_immutable_and_filled_with_constants(ctx) {
            if let Some(ref alloc_fn) = ctx.constant_fold_alloc {
                let (descr, fields, field_descrs) = match self {
                    PtrInfo::Virtual(v) => (&v.descr, &v.fields, &v.field_descrs),
                    PtrInfo::VirtualStruct(v) => (&v.descr, &v.fields, &v.field_descrs),
                    _ => unreachable!(),
                };
                let obj_size = descr.as_size_descr().map(|sd| sd.size()).unwrap_or(0);
                if obj_size > 0 {
                    let ptr = alloc_fn(obj_size);
                    if !ptr.is_null() {
                        // info.py:144: _force_elements_immutable
                        // Write constant field values directly to the allocated memory.
                        for &(field_idx, val_ref) in fields.iter() {
                            let resolved = ctx.get_box_replacement(val_ref);
                            if let Some(value) = ctx.get_constant(resolved) {
                                if let Some((_, fd)) =
                                    field_descrs.iter().find(|(idx, _)| *idx == field_idx)
                                {
                                    if let Some(field_d) = fd.as_field_descr() {
                                        let offset = field_d.offset();
                                        match value {
                                            Value::Int(v) => unsafe {
                                                let dest =
                                                    (ptr.0 as *mut u8).add(offset) as *mut i64;
                                                *dest = *v;
                                            },
                                            Value::Ref(r) => unsafe {
                                                let dest =
                                                    (ptr.0 as *mut u8).add(offset) as *mut usize;
                                                *dest = r.0;
                                            },
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        // info.py:142: op.set_forwarded(constptr)
                        let const_ref = GcRef(ptr.0);
                        ctx.make_constant(opref, Value::Ref(const_ref));
                        ctx.set_ptr_info(opref, PtrInfo::Constant(const_ref));
                        return opref;
                    }
                }
            }
            // No allocator or size unknown: fall through to normal force.
        }

        match self {
            PtrInfo::VirtualStruct(vinfo) => {
                // RPython info.py:147-156: _is_virtual = False, _fields retained.
                let preserved = PtrInfo::Struct(StructPtrInfo {
                    descr: vinfo.descr.clone(),
                    fields: vinfo
                        .fields
                        .iter()
                        .map(|&(idx, val)| (idx, ctx.get_box_replacement(val)))
                        .collect(),
                    field_descrs: vinfo.field_descrs.clone(),
                    preamble_fields: Vec::new(),
                    last_guard_pos: -1,
                });
                let mut new_op = Op::new(OpCode::New, &[]);
                new_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, new_op);
                // Store preserved info on alloc_ref (canonical after force).
                ctx.set_ptr_info(alloc_ref, preserved);
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }
                for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                    let value_ref = force_child(value_ref, ctx);
                    let descr = vinfo
                        .field_descrs
                        .iter()
                        .find(|(idx, _)| *idx == field_idx)
                        .map(|(_, d)| d.clone());
                    debug_assert!(
                        descr.is_some(),
                        "force_box: field_idx={} has value but no descriptor \
                         — field_descrs out of sync with fields",
                        field_idx,
                    );
                    if let Some(descr) = descr {
                        let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
                        set_op.descr = Some(descr);
                        emit_op(ctx, set_op);
                    }
                }
                alloc_ref
            }
            PtrInfo::Virtual(vinfo) => {
                let preserved = PtrInfo::Instance(InstancePtrInfo {
                    descr: Some(vinfo.descr.clone()),
                    known_class: vinfo.known_class,
                    fields: vinfo
                        .fields
                        .iter()
                        .map(|&(idx, val)| (idx, ctx.get_box_replacement(val)))
                        .collect(),
                    field_descrs: vinfo.field_descrs.clone(),
                    preamble_fields: Vec::new(),
                    last_guard_pos: -1,
                });
                let mut new_op = Op::new(OpCode::NewWithVtable, &[]);
                new_op.descr = Some(vinfo.descr.clone());
                let alloc_ref = emit_op(ctx, new_op);
                ctx.set_ptr_info(alloc_ref, preserved);
                if opref != alloc_ref {
                    ctx.replace_op(opref, alloc_ref);
                }
                for (field_idx, value_ref) in std::mem::take(&mut vinfo.fields) {
                    let value_ref = force_child(value_ref, ctx);
                    let descr = vinfo
                        .field_descrs
                        .iter()
                        .find(|(idx, _)| *idx == field_idx)
                        .map(|(_, d)| d.clone());
                    // RPython: descriptors always exist — _fields[i] maps
                    // to descr.get_all_fielddescrs()[i]. In pyre, field_descrs
                    // is a separate list that should always be in sync with
                    // fields. A missing descriptor indicates a structural
                    // bug in field/descriptor propagation.
                    debug_assert!(
                        descr.is_some(),
                        "force_box: field_idx={} has value but no descriptor \
                         — field_descrs out of sync with fields",
                        field_idx,
                    );
                    if let Some(descr) = descr {
                        let mut set_op = Op::new(OpCode::SetfieldGc, &[alloc_ref, value_ref]);
                        set_op.descr = Some(descr);
                        emit_op(ctx, set_op);
                    }
                }
                alloc_ref
            }
            _ => opref,
        }
    }

    /// info.py: make_guards(op, short_boxes, optimizer)
    /// Generate guard opcodes (without args) to verify this pointer info.
    /// Legacy helper for tests — use make_guards() for full guard emission.
    pub fn guard_opcodes(&self) -> Vec<majit_ir::OpCode> {
        match self {
            PtrInfo::NonNull { .. } => vec![majit_ir::OpCode::GuardNonnull],
            PtrInfo::KnownClass { .. } => vec![majit_ir::OpCode::GuardNonnullClass],
            PtrInfo::Instance(info) if info.known_class.is_some() => {
                vec![majit_ir::OpCode::GuardNonnullClass]
            }
            PtrInfo::Instance(info) if info.descr.is_some() => vec![
                majit_ir::OpCode::GuardNonnull,
                majit_ir::OpCode::GuardIsObject,
                majit_ir::OpCode::GuardSubclass,
            ],
            PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                vec![
                    majit_ir::OpCode::GuardNonnull,
                    majit_ir::OpCode::GuardGcType,
                ]
            }
            PtrInfo::Constant(_) => vec![majit_ir::OpCode::GuardValue],
            _ => Vec::new(),
        }
    }

    /// info.py: is_null() — whether this pointer is known to be null.
    pub fn is_null(&self) -> bool {
        match self {
            PtrInfo::Constant(gcref) => gcref.is_null(),
            _ => false,
        }
    }

    /// info.py: is_about_object() — whether this info describes an object
    /// (has fields/vtable, as opposed to an array or raw buffer).
    pub fn is_about_object(&self) -> bool {
        matches!(
            self,
            PtrInfo::Instance(_)
                | PtrInfo::Struct(_)
                | PtrInfo::Virtual(_)
                | PtrInfo::VirtualStruct(_)
        )
    }

    /// info.py: is_precise() — whether the type info is exact (not just a bound).
    pub fn is_precise(&self) -> bool {
        matches!(
            self,
            PtrInfo::Constant(_)
                | PtrInfo::Instance(_)
                | PtrInfo::Struct(_)
                | PtrInfo::Array(_)
                | PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
        )
    }

    /// info.py: same_info(other) — whether two PtrInfos describe the same value.
    pub fn same_info(&self, other: &PtrInfo) -> bool {
        match (self, other) {
            (PtrInfo::Constant(a), PtrInfo::Constant(b)) => a == b,
            (PtrInfo::NonNull { .. }, PtrInfo::NonNull { .. }) => true,
            (PtrInfo::Instance(a), PtrInfo::Instance(b)) => {
                a.descr.as_ref().map(|d| d.index()) == b.descr.as_ref().map(|d| d.index())
                    && a.known_class == b.known_class
            }
            (PtrInfo::Struct(a), PtrInfo::Struct(b)) => a.descr.index() == b.descr.index(),
            (PtrInfo::Array(a), PtrInfo::Array(b)) => {
                a.descr.index() == b.descr.index() && a.lenbound == b.lenbound
            }
            _ => std::ptr::eq(self, other),
        }
    }

    /// info.py: get_descr() — get the size/type descriptor for virtual objects.
    pub fn get_descr(&self) -> Option<&DescrRef> {
        match self {
            PtrInfo::Instance(v) => v.descr.as_ref(),
            PtrInfo::Struct(v) => Some(&v.descr),
            PtrInfo::Array(v) => Some(&v.descr),
            PtrInfo::Virtual(v) => Some(&v.descr),
            PtrInfo::VirtualArray(v) => Some(&v.descr),
            PtrInfo::VirtualStruct(v) => Some(&v.descr),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.descr),
            _ => None,
        }
    }

    /// info.py: getlenbound() on ArrayPtrInfo.
    pub fn getlenbound(&self) -> Option<&IntBound> {
        match self {
            PtrInfo::Array(v) => Some(&v.lenbound),
            _ => None,
        }
    }

    /// info.py: setfield(field_descr, value) — set a field on a virtual object.
    /// info.py:176-200 setfield — update the field value in the virtual.
    /// RPython: _fields[fielddescr.get_index()] = op. In majit, fields
    /// is a (field_idx, OpRef) list; field_descrs is managed separately
    /// by OptVirtualize (optimize_setfield_gc).
    pub fn setfield(&mut self, field_idx: u32, value: OpRef) {
        match self {
            PtrInfo::Instance(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            PtrInfo::Struct(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            PtrInfo::Virtual(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            PtrInfo::VirtualStruct(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            _ => {}
        }
    }

    /// shortpreamble.py:73-79: HeapOp.produce_op stores PreambleOp in _fields.
    /// RPython: `opinfo.setfield(descr, struct, pop, optheap, cf)`
    /// where `pop` is a PreambleOp wrapper.
    pub fn set_preamble_field(&mut self, field_idx: u32, pop: PreambleOp) {
        match self {
            PtrInfo::Instance(v) => {
                v.preamble_fields.retain(|(k, _)| *k != field_idx);
                v.preamble_fields.push((field_idx, pop));
            }
            PtrInfo::Struct(v) => {
                v.preamble_fields.retain(|(k, _)| *k != field_idx);
                v.preamble_fields.push((field_idx, pop));
            }
            _ => {
                // RPython: AbstractStructPtrInfo always supports _fields.
                // In majit, KnownClass/NonNull/Ref lack preamble_fields.
                // Upgrade to Instance, preserving known_class.
                let known_class = if let PtrInfo::KnownClass { class_ptr, .. } = self {
                    Some(*class_ptr)
                } else {
                    None
                };
                *self = PtrInfo::Instance(InstancePtrInfo {
                    descr: None,
                    known_class,
                    fields: Vec::new(),
                    field_descrs: Vec::new(),
                    preamble_fields: vec![(field_idx, pop)],
                    last_guard_pos: -1,
                });
            }
        }
    }

    /// heap.py:177-187: CachedField._getfield detects PreambleOp in _fields.
    /// Returns and removes the PreambleOp if present for this field.
    /// RPython: `isinstance(res, PreambleOp)` check in _getfield.
    pub fn take_preamble_field(&mut self, field_idx: u32) -> Option<PreambleOp> {
        match self {
            PtrInfo::Instance(v) => {
                if let Some(pos) = v.preamble_fields.iter().position(|(k, _)| *k == field_idx) {
                    Some(v.preamble_fields.remove(pos).1)
                } else {
                    None
                }
            }
            PtrInfo::Struct(v) => {
                if let Some(pos) = v.preamble_fields.iter().position(|(k, _)| *k == field_idx) {
                    Some(v.preamble_fields.remove(pos).1)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// info.py:273-303: _is_immutable_and_filled_with_constants
    /// Check if this virtual is immutable and all fields are constants.
    /// Used by force_box to determine if the virtual can be constant-folded.
    pub fn is_immutable_and_filled_with_constants(
        &self,
        ctx: &crate::optimizeopt::OptContext,
    ) -> bool {
        let (fields, descr) = match self {
            PtrInfo::Virtual(v) => (&v.fields, &v.descr),
            PtrInfo::VirtualStruct(v) => (&v.fields, &v.descr),
            _ => return false,
        };
        if !descr.is_always_pure() {
            return false;
        }
        for &(_, val) in fields {
            let resolved = ctx.get_box_replacement(val);
            if !ctx.is_constant(resolved) {
                // Check if it's a virtual that is also immutable+constant
                if let Some(info) = ctx.get_ptr_info(resolved) {
                    if info.is_virtual() && info.is_immutable_and_filled_with_constants(ctx) {
                        continue;
                    }
                }
                return false;
            }
        }
        true
    }

    /// heap.py:194: opinfo._fields[descr.get_index()] = None
    /// Clear a cached field value. Used by CachedField.invalidate().
    /// RPython stores PreambleOp in _fields[] too, so clearing a field
    /// index removes both regular and preamble entries.
    pub fn clear_field(&mut self, field_idx: u32) {
        match self {
            PtrInfo::Instance(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.preamble_fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Struct(v) => {
                v.fields.retain(|(k, _)| *k != field_idx);
                v.preamble_fields.retain(|(k, _)| *k != field_idx);
            }
            PtrInfo::Virtual(v) => v.fields.retain(|(k, _)| *k != field_idx),
            PtrInfo::VirtualStruct(v) => v.fields.retain(|(k, _)| *k != field_idx),
            _ => {}
        }
    }

    /// info.py: all_items() — return all cached field entries.
    /// heap.py:211,214: opinfo.all_items() used by _cannot_alias_via_content.
    pub fn all_items(&self) -> &[(u32, OpRef)] {
        match self {
            PtrInfo::Instance(v) => &v.fields,
            PtrInfo::Struct(v) => &v.fields,
            PtrInfo::Virtual(v) => &v.fields,
            PtrInfo::VirtualStruct(v) => &v.fields,
            _ => &[],
        }
    }

    /// info.py: getfield(field_descr) — get a field from a virtual object.
    pub fn getfield(&self, field_idx: u32) -> Option<OpRef> {
        match self {
            PtrInfo::Instance(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            PtrInfo::Struct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            _ => None,
        }
    }

    /// info.py: setitem(index, value) — set an item in a virtual array.
    pub fn setitem(&mut self, index: usize, value: OpRef) {
        match self {
            PtrInfo::Array(v) => {
                if index >= v.items.len() {
                    v.items.resize(index + 1, OpRef::NONE);
                }
                v.items[index] = value;
            }
            PtrInfo::VirtualArray(v) => {
                if index < v.items.len() {
                    v.items[index] = value;
                }
            }
            _ => {}
        }
    }

    /// info.py: getitem(index) — get an item from a virtual array.
    pub fn getitem(&self, index: usize) -> Option<OpRef> {
        match self {
            PtrInfo::Array(v) => v.items.get(index).copied(),
            PtrInfo::VirtualArray(v) => v.items.get(index).copied(),
            _ => None,
        }
    }

    /// info.py:651-656: _compute_index(index, fielddescr)
    /// Computes flat index into VirtualArrayStruct's element_fields.
    fn compute_interior_index(
        &self,
        element_index: usize,
        field_descr_index: u32,
    ) -> Option<(usize, usize)> {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    return None;
                }
                // Find the slot for field_descr_index within this element.
                let fields = &v.element_fields[element_index];
                for (slot, &(fdidx, _)) in fields.iter().enumerate() {
                    if fdidx == field_descr_index {
                        return Some((element_index, slot));
                    }
                }
                // Field not yet present — return element index for insertion.
                Some((element_index, fields.len()))
            }
            _ => None,
        }
    }

    /// info.py:663-668: getinteriorfield_virtual(index, fielddescr)
    pub fn getinteriorfield_virtual(
        &self,
        element_index: usize,
        field_descr_index: u32,
    ) -> Option<OpRef> {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    return None;
                }
                v.element_fields[element_index]
                    .iter()
                    .find(|&&(fdidx, _)| fdidx == field_descr_index)
                    .map(|&(_, opref)| opref)
            }
            _ => None,
        }
    }

    /// info.py:658-661: setinteriorfield_virtual(index, fielddescr, fld)
    pub fn setinteriorfield_virtual(
        &mut self,
        element_index: usize,
        field_descr_index: u32,
        value: OpRef,
    ) {
        match self {
            PtrInfo::VirtualArrayStruct(v) => {
                if element_index >= v.element_fields.len() {
                    v.element_fields.resize(element_index + 1, Vec::new());
                }
                let fields = &mut v.element_fields[element_index];
                // Update existing or insert new.
                if let Some(entry) = fields
                    .iter_mut()
                    .find(|(fdidx, _)| *fdidx == field_descr_index)
                {
                    entry.1 = value;
                } else {
                    fields.push((field_descr_index, value));
                }
            }
            _ => {}
        }
    }

    /// info.py: produce_short_preamble_ops(structbox, descr, index, optimizer, shortboxes)
    ///
    /// Add cached field values to the short preamble builder.
    /// For each non-null field in the virtual, register a descriptor-carrying
    /// GETFIELD read so the bridge can re-populate the optimizer's field cache.
    pub fn produce_short_preamble_ops(&self, structbox: OpRef) -> Vec<Op> {
        let mut result = Vec::new();
        // Fields are accessed per-variant below
        if let PtrInfo::Virtual(v) = self {
            for &(field_idx, value) in &v.fields {
                if !value.is_none() {
                    if let Some((_, descr)) =
                        v.field_descrs.iter().find(|(idx, _)| *idx == field_idx)
                    {
                        result.push(Op::with_descr(
                            OpCode::GetfieldGcI,
                            &[structbox],
                            descr.clone(),
                        ));
                    }
                }
            }
        }
        if let PtrInfo::VirtualStruct(v) = self {
            for &(field_idx, value) in &v.fields {
                if !value.is_none() {
                    if let Some((_, descr)) =
                        v.field_descrs.iter().find(|(idx, _)| *idx == field_idx)
                    {
                        result.push(Op::with_descr(
                            OpCode::GetfieldGcI,
                            &[structbox],
                            descr.clone(),
                        ));
                    }
                }
            }
        }
        result
    }
}

/// A virtual object whose allocation has been removed.
///
/// Fields are tracked as OpRefs to the operations that produce their values.
#[derive(Clone, Debug)]
pub struct VirtualInfo {
    /// The size descriptor of this object.
    pub descr: DescrRef,
    /// Known class (if any).
    pub known_class: Option<GcRef>,
    /// ob_type field descriptor for force path. In RPython the vtable is
    /// set by allocate_with_vtable, not as a struct field. pyre stores
    /// ob_type at offset 0 explicitly. This descr lets force emit
    /// SetfieldGc(ob_type) without polluting `fields` (which feeds rd_virtuals).
    pub ob_type_descr: Option<DescrRef>,
    /// Field values: (field_descr_index, value_opref).
    /// Does NOT include ob_type — handled via known_class + ob_type_descr.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors, preserving offset/size/type info for forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual array.
#[derive(Clone, Debug)]
pub struct VirtualArrayInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Whether this was created by NewArrayClear (zero-initialized).
    pub clear: bool,
    /// Element values.
    pub items: Vec<OpRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual object with cached field info.
///
/// Mirrors RPython's InstancePtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct InstancePtrInfo {
    /// Best-known instance descriptor, if any.
    pub descr: Option<DescrRef>,
    /// Known class pointer, if guarded exactly.
    pub known_class: Option<GcRef>,
    /// Cached field values.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors keyed by field index.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// shortpreamble.py:11-49: PreambleOp wrappers stored during Phase 2
    /// import. RPython stores these in `_fields[]` (mixed with regular
    /// values); Rust uses a separate Vec with read-before-fields semantics.
    pub preamble_fields: Vec<(u32, PreambleOp)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC struct with cached field info.
///
/// Mirrors RPython's StructPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct StructPtrInfo {
    /// Exact struct descriptor.
    pub descr: DescrRef,
    /// Cached field values.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors keyed by field index.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// shortpreamble.py:11-49: PreambleOp wrappers (same as InstancePtrInfo).
    pub preamble_fields: Vec<(u32, PreambleOp)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A non-virtual GC array with cached item info and lenbound.
///
/// Mirrors RPython's ArrayPtrInfo in the non-virtual case.
#[derive(Clone, Debug)]
pub struct ArrayPtrInfo {
    /// Exact array descriptor.
    pub descr: DescrRef,
    /// Known bounds on the array length.
    pub lenbound: IntBound,
    /// Cached item values for constant indices.
    pub items: Vec<OpRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual struct (no vtable).
#[derive(Clone, Debug)]
pub struct VirtualStructInfo {
    /// The size descriptor.
    pub descr: DescrRef,
    /// Field values: (field_index, value, optional original field descriptor).
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors keyed by field_index, used for force.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual array of structs (interior field access pattern).
///
/// Mirrors RPython's VArrayStructInfo where each array element
/// is a fixed-size struct with named fields. Used for RPython arrays
/// with complex item types (e.g., hash table entries with key+value fields).
#[derive(Clone, Debug)]
pub struct VirtualArrayStructInfo {
    /// The array descriptor (arraydescr).
    pub descr: DescrRef,
    /// Per-element fields: outer Vec = elements, inner Vec = (field_descr_index, value_opref).
    pub element_fields: Vec<Vec<(u32, OpRef)>>,
    /// resume.py VArrayStructInfo.fielddescrs — InteriorFieldDescr per field.
    /// Used by _number_virtuals to extract item_size/field_offset/field_size.
    pub fielddescrs: Vec<DescrRef>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// A virtual raw memory buffer.
///
/// Mirrors RPython's VRawBufferInfo for virtualized raw_malloc allocations.
/// Tracks writes to byte offsets within the buffer. Entries are kept sorted
/// by offset and must never overlap (matching RPython's RawBuffer invariant).
#[derive(Clone, Debug)]
pub struct VirtualRawBufferInfo {
    /// Size of the buffer in bytes.
    pub size: usize,
    /// Values stored at byte offsets: (offset, length, value_opref).
    ///
    /// Sorted by offset. Invariant: `entries[i].0 + entries[i].1 <= entries[i+1].0`
    /// (no overlapping writes).
    pub entries: Vec<(usize, usize, OpRef)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

/// Error returned when a raw buffer operation violates invariants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawBufferError {
    /// A write overlaps with an existing write.
    OverlappingWrite {
        new_offset: usize,
        new_length: usize,
        existing_offset: usize,
        existing_length: usize,
    },
    /// A read from an offset that was never written.
    UninitializedRead { offset: usize, length: usize },
    /// A read whose length/offset doesn't match the write at that offset.
    IncompatibleRead {
        offset: usize,
        read_length: usize,
        write_length: usize,
    },
}

impl VirtualRawBufferInfo {
    /// Write a value at `(offset, length)`. Maintains sorted order by offset.
    ///
    /// If a write already exists at the same offset with the same length,
    /// updates the value. Returns `Err` on overlapping writes or
    /// incompatible length at the same offset.
    pub fn write_value(
        &mut self,
        offset: usize,
        length: usize,
        value: OpRef,
    ) -> Result<(), RawBufferError> {
        let mut insert_pos = 0;
        for (i, &(wo, wl, _)) in self.entries.iter().enumerate() {
            if wo == offset {
                if wl != length {
                    return Err(RawBufferError::OverlappingWrite {
                        new_offset: offset,
                        new_length: length,
                        existing_offset: wo,
                        existing_length: wl,
                    });
                }
                // Same offset and length: update in place.
                self.entries[i].2 = value;
                return Ok(());
            } else if wo > offset {
                break;
            }
            insert_pos = i + 1;
        }
        // Check overlap with next entry.
        if insert_pos < self.entries.len() {
            let (next_off, _, _) = self.entries[insert_pos];
            if offset + length > next_off {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: next_off,
                    existing_length: self.entries[insert_pos].1,
                });
            }
        }
        // Check overlap with previous entry.
        if insert_pos > 0 {
            let (prev_off, prev_len, _) = self.entries[insert_pos - 1];
            if prev_off + prev_len > offset {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: prev_off,
                    existing_length: prev_len,
                });
            }
        }
        self.entries.insert(insert_pos, (offset, length, value));
        Ok(())
    }

    /// Read the value at `(offset, length)`.
    ///
    /// Returns `Err(UninitializedRead)` if no write exists at that offset,
    /// or `Err(IncompatibleRead)` if the length doesn't match.
    pub fn read_value(&self, offset: usize, length: usize) -> Result<OpRef, RawBufferError> {
        for &(wo, wl, val) in &self.entries {
            if wo == offset {
                if wl != length {
                    return Err(RawBufferError::IncompatibleRead {
                        offset,
                        read_length: length,
                        write_length: wl,
                    });
                }
                return Ok(val);
            }
        }
        Err(RawBufferError::UninitializedRead { offset, length })
    }

    /// Check if a read at `(offset, size)` is fully covered by previous writes.
    ///
    /// Every byte in `[offset, offset+size)` must fall within at least one
    /// existing write region.
    pub fn is_read_fully_covered(&self, offset: usize, size: usize) -> bool {
        (0..size).all(|i| {
            let byte = offset + i;
            self.entries
                .iter()
                .any(|&(wo, wl, _)| byte >= wo && byte < wo + wl)
        })
    }

    /// Find the index of an existing write that is completely overwritten
    /// by a new write at `(offset, size)`.
    ///
    /// Returns the index of the first entry fully contained within
    /// `[offset, offset+size)`.
    pub fn find_overwritten_write(&self, offset: usize, size: usize) -> Option<usize> {
        self.entries
            .iter()
            .position(|&(wo, wl, _)| offset <= wo && offset + size >= wo + wl)
    }
}

/// Tracked field state for a virtualizable object (interpreter frame).
///
/// Mirrors RPython's virtualizable handling in the optimizer:
/// the frame already exists on the heap, but during JIT execution its
/// fields are kept in registers. The optimizer tracks the current value
/// of each field so that redundant setfield/getfield ops are eliminated.
///
/// When the virtualizable is "forced" (escapes to non-JIT code), field
/// values are written back to the heap via SETFIELD_RAW ops.
#[derive(Clone, Debug)]
pub struct VirtualizableFieldState {
    /// Tracked static field values: (field_descr_index, current_value_opref).
    /// Indices correspond to VirtualizableInfo::static_fields order.
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors: (field_descr_index, original_descr).
    /// Used to emit correct SetfieldRaw ops when forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<OpRef>)>,
    /// info.py:91-92
    pub last_guard_pos: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Descr, OpCode, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

    fn make_buf(size: usize) -> VirtualRawBufferInfo {
        VirtualRawBufferInfo {
            size,
            entries: Vec::new(),
            last_guard_pos: -1,
        }
    }

    #[test]
    fn rawbuffer_write_and_read() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();
        buf.write_value(16, 8, OpRef(30)).unwrap();

        assert_eq!(buf.read_value(0, 8).unwrap(), OpRef(10));
        assert_eq!(buf.read_value(8, 4).unwrap(), OpRef(20));
        assert_eq!(buf.read_value(16, 8).unwrap(), OpRef(30));
    }

    #[test]
    fn rawbuffer_update_same_offset() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(0, 8, OpRef(99)).unwrap();

        assert_eq!(buf.read_value(0, 8).unwrap(), OpRef(99));
        assert_eq!(buf.entries.len(), 1);
    }

    #[test]
    fn rawbuffer_overlap_next() {
        let mut buf = make_buf(32);
        buf.write_value(8, 8, OpRef(10)).unwrap();
        // Write at offset 4 with length 8 overlaps [8, 16)
        let err = buf.write_value(4, 8, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_overlap_prev() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        // Write at offset 4 overlaps with [0, 8)
        let err = buf.write_value(4, 4, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_incompatible_length_at_same_offset() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        let err = buf.write_value(0, 4, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_uninitialized_read() {
        let buf = make_buf(16);
        let err = buf.read_value(0, 8).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::UninitializedRead {
                offset: 0,
                length: 8
            }
        );
    }

    #[test]
    fn rawbuffer_incompatible_read_length() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        let err = buf.read_value(0, 4).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::IncompatibleRead {
                offset: 0,
                read_length: 4,
                write_length: 8,
            }
        );
    }

    #[test]
    fn rawbuffer_read_fully_covered() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(8, 8, OpRef(20)).unwrap();

        // [0, 16) is fully covered by [0,8) + [8,16)
        assert!(buf.is_read_fully_covered(0, 16));
        // [0, 8) covered
        assert!(buf.is_read_fully_covered(0, 8));
        // [4, 8) falls within [0, 8)
        assert!(buf.is_read_fully_covered(4, 4));
    }

    #[test]
    fn rawbuffer_read_partially_covered_fails() {
        let mut buf = make_buf(32);
        buf.write_value(0, 4, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();

        // Bytes 4..8 are not covered by any write
        assert!(!buf.is_read_fully_covered(0, 8));
        // Byte 16 was never written
        assert!(!buf.is_read_fully_covered(16, 4));
    }

    #[test]
    fn rawbuffer_overwritten_write_detected() {
        let mut buf = make_buf(32);
        buf.write_value(4, 4, OpRef(10)).unwrap();
        buf.write_value(12, 4, OpRef(20)).unwrap();

        // A write [4, 12) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(4, 8), Some(0));
        // A write [0, 16) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(0, 16), Some(0));
        // A write [12, 20) fully contains [12, 16)
        assert_eq!(buf.find_overwritten_write(12, 8), Some(1));
        // A write [0, 4) does not contain any existing entry
        assert_eq!(buf.find_overwritten_write(0, 4), None);
    }

    #[test]
    fn rawbuffer_sorted_insertion() {
        let mut buf = make_buf(32);
        buf.write_value(16, 4, OpRef(30)).unwrap();
        buf.write_value(0, 4, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();

        // Entries should be sorted by offset
        assert_eq!(buf.entries[0].0, 0);
        assert_eq!(buf.entries[1].0, 8);
        assert_eq!(buf.entries[2].0, 16);
    }

    #[test]
    fn test_ptr_info_factories() {
        let nonnull = PtrInfo::nonnull();
        assert!(nonnull.is_nonnull());
        assert!(!nonnull.is_virtual());

        let constant = PtrInfo::constant(GcRef(0x1000));
        assert!(constant.is_nonnull());
        assert!(constant.is_constant());

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        assert!(kc.is_nonnull());
        assert!(kc.get_known_class().is_some());
    }

    #[test]
    fn test_ptr_info_virtual_factories() {
        let descr: DescrRef = Arc::new(TestDescr);

        let virtual_obj = PtrInfo::virtual_obj(descr.clone(), Some(GcRef(0x3000)));
        assert!(virtual_obj.is_virtual());
        assert!(virtual_obj.is_nonnull());
        assert!(virtual_obj.get_descr().is_some());

        let virtual_arr = PtrInfo::virtual_array(descr.clone(), 5, false);
        assert!(virtual_arr.is_virtual());
        assert_eq!(virtual_arr.num_fields(), 5);

        let virtual_struct = PtrInfo::virtual_struct(descr);
        assert!(virtual_struct.is_virtual());
    }

    #[test]
    fn test_ptr_info_set_getfield() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);

        assert_eq!(info.getfield(0), None);
        info.setfield(0, OpRef(10));
        assert_eq!(info.getfield(0), Some(OpRef(10)));
        info.setfield(0, OpRef(20)); // overwrite
        assert_eq!(info.getfield(0), Some(OpRef(20)));
        info.setfield(1, OpRef(30));
        assert_eq!(info.getfield(1), Some(OpRef(30)));
    }

    #[test]
    fn test_ptr_info_set_getitem() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_array(descr, 3, false);

        assert_eq!(info.getitem(0), Some(OpRef::NONE)); // initialized to NONE
        info.setitem(0, OpRef(10));
        assert_eq!(info.getitem(0), Some(OpRef(10)));
        info.setitem(2, OpRef(30));
        assert_eq!(info.getitem(2), Some(OpRef(30)));
        assert_eq!(info.getitem(5), None); // out of bounds
    }

    #[test]
    fn test_ptr_info_guard_opcodes() {
        let nonnull = PtrInfo::nonnull();
        let guards = nonnull.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnull));

        let constant = PtrInfo::constant(GcRef(0x1000));
        let guards = constant.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardValue));

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        let guards = kc.guard_opcodes();
        assert!(guards.contains(&OpCode::GuardNonnullClass));
    }

    #[test]
    fn test_ptr_info_visitor_walk() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, OpRef(10));
        info.setfield(1, OpRef(20));
        let refs = info.visitor_walk_recursive();
        assert_eq!(refs, vec![OpRef(10), OpRef(20)]);
    }

    #[test]
    fn test_opinfo_is_nonnull() {
        assert!(!OpInfo::Unknown.is_nonnull());
        assert!(OpInfo::Constant(Value::Int(42)).is_nonnull());
        assert!(!OpInfo::Constant(Value::Int(0)).is_nonnull());
        assert!(OpInfo::Ptr(PtrInfo::nonnull()).is_nonnull());
    }

    #[test]
    fn test_opinfo_float_const() {
        let info = OpInfo::FloatConst(3.14);
        assert!(info.is_constant());
        assert_eq!(info.get_constant_float(), Some(3.14));
    }
}
