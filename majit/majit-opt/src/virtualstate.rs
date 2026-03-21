/// Virtual state export/import for loop peeling.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualstate.py.
///
/// After unrolling one iteration (the "preamble"), the optimizer captures the
/// abstract state of each value carried across the back-edge. On the next
/// iteration, this exported state is compared against the incoming values.
/// If compatible, the optimizer can directly apply known information (virtuals,
/// bounds, classes) without re-discovering it.
///
/// Key types:
/// - `VirtualState`: a snapshot of abstract info for all loop-carried values
/// - `VirtualStateInfo`: per-value abstract info (constant, virtual, class, etc.)
/// - State comparison determines if a compiled loop body can be reused
use std::collections::HashMap;

use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Value};

/// virtualstate.py: VirtualStatesCantMatch — raised when two virtual states
/// are incompatible and cannot be merged for bridge compilation.
#[derive(Clone, Debug)]
pub struct VirtualStatesCantMatch {
    pub msg: String,
}

impl VirtualStatesCantMatch {
    pub fn new(msg: &str) -> Self {
        VirtualStatesCantMatch {
            msg: msg.to_string(),
        }
    }
}

impl std::fmt::Display for VirtualStatesCantMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VirtualStatesCantMatch: {}", self.msg)
    }
}

impl std::error::Error for VirtualStatesCantMatch {}

use crate::OptContext;
use crate::info::{
    PtrInfo, VirtualArrayInfo, VirtualArrayStructInfo, VirtualInfo, VirtualRawBufferInfo,
    VirtualStructInfo,
};
use crate::intutils::IntBound;

/// virtualstate.py: GenerateGuardState — context for guard generation
/// during virtual state comparison.
///
/// Holds the optimizer reference, a collection of generated guards,
/// a renumbering map (OpRef → position), and flags.
pub struct GenerateGuardState<'a> {
    /// Reference to optimizer context.
    pub ctx: &'a OptContext,
    /// Extra guards generated during state comparison.
    pub extra_guards: Vec<majit_ir::Op>,
    /// Renumbering map: OpRef → position in the state vector.
    pub renum: HashMap<OpRef, usize>,
    /// Entries that could not be matched (incompatible state).
    pub bad: HashMap<usize, String>,
    /// Whether to force virtual boxes during comparison.
    pub force_boxes: bool,
}

impl<'a> GenerateGuardState<'a> {
    pub fn new(ctx: &'a OptContext) -> Self {
        GenerateGuardState {
            ctx,
            extra_guards: Vec::new(),
            renum: HashMap::new(),
            bad: HashMap::new(),
            force_boxes: false,
        }
    }

    /// virtualstate.py: get_runtime_field(box, descr)
    ///
    /// Read a field from a concrete object at runtime.
    /// Returns the value as an i64 (for int fields) or pointer.
    pub fn get_runtime_field(&self, obj_ptr: i64, offset: usize) -> i64 {
        if obj_ptr == 0 {
            return 0;
        }
        unsafe { *((obj_ptr as *const u8).add(offset) as *const i64) }
    }

    /// virtualstate.py: get_runtime_item(box, descr, i)
    ///
    /// Read an array item from a concrete object at runtime.
    pub fn get_runtime_item(&self, array_ptr: i64, index: usize, item_size: usize) -> i64 {
        if array_ptr == 0 {
            return 0;
        }
        unsafe { *((array_ptr as *const u8).add(index * item_size) as *const i64) }
    }

    /// Add an extra guard to be emitted before the bridge entry.
    pub fn add_guard(&mut self, guard: majit_ir::Op) {
        self.extra_guards.push(guard);
    }

    /// Mark an entry as incompatible.
    pub fn mark_bad(&mut self, index: usize, reason: &str) {
        self.bad.insert(index, reason.to_string());
    }

    /// Whether the comparison succeeded (no bad entries).
    pub fn is_ok(&self) -> bool {
        self.bad.is_empty()
    }
}

/// Abstract info for one value at the loop boundary.
///
/// Mirrors the hierarchy in RPython's `AbstractVirtualStateInfo` and its subclasses:
/// `VirtualStateInfoConst`, `VirtualStateInfoVirtual`, `VirtualStateInfoNotVirtual`, etc.
#[derive(Clone, Debug)]
pub enum VirtualStateInfo {
    /// Value is a known constant.
    Constant(Value),
    /// Value is a virtual instance with known fields.
    Virtual {
        descr: DescrRef,
        known_class: Option<GcRef>,
        /// Field values as VirtualStateInfo (recursive).
        fields: Vec<(u32, Box<VirtualStateInfo>)>,
        /// Original field descriptors for each field index.
        field_descrs: Vec<(u32, DescrRef)>,
    },
    /// Value is a virtual array with known elements.
    VirtualArray {
        descr: DescrRef,
        items: Vec<Box<VirtualStateInfo>>,
        /// virtualstate.py: lenbound — known bounds on array length.
        /// None means unbounded.
        lenbound: Option<IntBound>,
    },
    /// Value is a virtual struct.
    VirtualStruct {
        descr: DescrRef,
        fields: Vec<(u32, Box<VirtualStateInfo>)>,
        field_descrs: Vec<(u32, DescrRef)>,
    },
    /// Value is a virtual array of structs.
    VirtualArrayStruct {
        descr: DescrRef,
        element_fields: Vec<Vec<(u32, Box<VirtualStateInfo>)>>,
    },
    /// Value is a virtual raw buffer.
    VirtualRawBuffer {
        size: usize,
        entries: Vec<(usize, usize, Box<VirtualStateInfo>)>,
    },
    /// Value has a known class (non-null).
    KnownClass { class_ptr: GcRef },
    /// Value is known non-null.
    NonNull,
    /// Value has known integer bounds.
    IntBounded(IntBound),
    /// No useful info (anything is compatible).
    Unknown,
}

impl VirtualStateInfo {
    /// Check if `other` is compatible with `self`.
    ///
    /// "Compatible" means that if the loop body was optimized assuming `self`,
    /// a value described by `other` can safely enter that loop body.
    ///
    /// In RPython this is `generate_guards()` which emits any needed bridge guards.
    /// Here we just check compatibility; guard generation is separate.
    pub fn is_compatible(&self, other: &VirtualStateInfo) -> bool {
        match (self, other) {
            // Unknown accepts anything
            (VirtualStateInfo::Unknown, _) => true,

            // Constants must match
            (VirtualStateInfo::Constant(a), VirtualStateInfo::Constant(b)) => a == b,
            (VirtualStateInfo::Constant(_), _) => false,

            // Virtual instance: other must also be a matching virtual with compatible fields
            (
                VirtualStateInfo::Virtual {
                    descr: d1,
                    known_class: kc1,
                    fields: f1,
                    ..
                },
                VirtualStateInfo::Virtual {
                    descr: d2,
                    known_class: kc2,
                    fields: f2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() {
                    return false;
                }
                // Class must match (both None or same pointer)
                match (kc1, kc2) {
                    (Some(c1), Some(c2)) if c1 != c2 => return false,
                    (Some(_), None) => return false,
                    _ => {}
                }
                // All fields in self must have compatible counterparts in other
                for (idx, info) in f1 {
                    let other_info = f2.iter().find(|(i, _)| i == idx).map(|(_, v)| v.as_ref());
                    match other_info {
                        Some(oi) => {
                            if !info.is_compatible(oi) {
                                return false;
                            }
                        }
                        None => return false, // field missing in other
                    }
                }
                true
            }

            // Virtual array: must match length and each element
            (
                VirtualStateInfo::VirtualArray {
                    descr: d1,
                    items: i1,
                    ..
                },
                VirtualStateInfo::VirtualArray {
                    descr: d2,
                    items: i2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() || i1.len() != i2.len() {
                    return false;
                }
                i1.iter().zip(i2.iter()).all(|(a, b)| a.is_compatible(b))
            }

            // Virtual struct: same as virtual instance
            (
                VirtualStateInfo::VirtualStruct {
                    descr: d1,
                    fields: f1,
                    ..
                },
                VirtualStateInfo::VirtualStruct {
                    descr: d2,
                    fields: f2,
                    ..
                },
            ) => {
                if d1.index() != d2.index() {
                    return false;
                }
                for (idx, info) in f1 {
                    let other_info = f2.iter().find(|(i, _)| i == idx).map(|(_, v)| v.as_ref());
                    match other_info {
                        Some(oi) => {
                            if !info.is_compatible(oi) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                true
            }

            // Virtual array struct
            (
                VirtualStateInfo::VirtualArrayStruct {
                    descr: d1,
                    element_fields: ef1,
                },
                VirtualStateInfo::VirtualArrayStruct {
                    descr: d2,
                    element_fields: ef2,
                },
            ) => {
                if d1.index() != d2.index() || ef1.len() != ef2.len() {
                    return false;
                }
                ef1.iter().zip(ef2.iter()).all(|(fields1, fields2)| {
                    for (idx, info) in fields1 {
                        let other_info = fields2
                            .iter()
                            .find(|(i, _)| i == idx)
                            .map(|(_, v)| v.as_ref());
                        match other_info {
                            Some(oi) if info.is_compatible(oi) => {}
                            _ => return false,
                        }
                    }
                    true
                })
            }

            // Virtual raw buffer
            (
                VirtualStateInfo::VirtualRawBuffer {
                    size: s1,
                    entries: e1,
                },
                VirtualStateInfo::VirtualRawBuffer {
                    size: s2,
                    entries: e2,
                },
            ) => {
                if s1 != s2 || e1.len() != e2.len() {
                    return false;
                }
                e1.iter()
                    .zip(e2.iter())
                    .all(|((off1, len1, v1), (off2, len2, v2))| {
                        off1 == off2 && len1 == len2 && v1.is_compatible(v2)
                    })
            }

            // KnownClass: other must have the same class (or be virtual with matching class)
            (VirtualStateInfo::KnownClass { class_ptr: c1 }, other_info) => match other_info {
                VirtualStateInfo::KnownClass { class_ptr: c2 } => c1 == c2,
                VirtualStateInfo::Virtual { known_class, .. } => known_class.as_ref() == Some(c1),
                _ => false,
            },

            // NonNull: other must be nonnull (virtual is always nonnull)
            (VirtualStateInfo::NonNull, other_info) => match other_info {
                VirtualStateInfo::NonNull
                | VirtualStateInfo::KnownClass { .. }
                | VirtualStateInfo::Virtual { .. }
                | VirtualStateInfo::VirtualArray { .. }
                | VirtualStateInfo::VirtualStruct { .. }
                | VirtualStateInfo::VirtualArrayStruct { .. }
                | VirtualStateInfo::VirtualRawBuffer { .. } => true,
                VirtualStateInfo::Constant(Value::Ref(r)) => !r.is_null(),
                _ => false,
            },

            // IntBounded: other must have tighter or equal bounds
            (VirtualStateInfo::IntBounded(b1), VirtualStateInfo::IntBounded(b2)) => {
                // b2 must fit entirely within b1
                b2.lower >= b1.lower && b2.upper <= b1.upper
            }
            (VirtualStateInfo::IntBounded(b), VirtualStateInfo::Constant(Value::Int(v))) => {
                // A constant is always within bounds if the bound contains it
                b.contains(*v)
            }
            (VirtualStateInfo::IntBounded(_), _) => false,

            // Cross-type mismatches
            _ => false,
        }
    }

    /// Whether this info represents a virtual (not yet allocated) value.
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            VirtualStateInfo::Virtual { .. }
                | VirtualStateInfo::VirtualArray { .. }
                | VirtualStateInfo::VirtualStruct { .. }
                | VirtualStateInfo::VirtualArrayStruct { .. }
                | VirtualStateInfo::VirtualRawBuffer { .. }
        )
    }
}

/// A complete snapshot of abstract state at a loop boundary.
///
/// The `state` vector has one entry per loop-carried variable (matching the
/// `Jump`/`Label` args). During loop peeling, this is exported at the end
/// of the preamble and imported at the loop header.
#[derive(Clone, Debug)]
pub struct VirtualState {
    /// Abstract info for each loop-carried variable, in order matching Label/Jump args.
    pub state: Vec<VirtualStateInfo>,
    /// virtualstate.py: renum — maps virtual OpRef to numbering index.
    /// Used to ensure consistent virtual identity across loop iterations.
    pub renum: std::collections::HashMap<OpRef, usize>,
}

impl VirtualState {
    pub fn new(state: Vec<VirtualStateInfo>) -> Self {
        VirtualState {
            state,
            renum: std::collections::HashMap::new(),
        }
    }

    pub fn count_forced_boxes_for_entry_static(info: &VirtualStateInfo) -> usize {
        Self::count_forced_boxes_for_entry(info)
    }
    fn count_forced_boxes_for_entry(info: &VirtualStateInfo) -> usize {
        match info {
            VirtualStateInfo::Constant(_) => 0,
            VirtualStateInfo::Virtual { fields, .. }
            | VirtualStateInfo::VirtualStruct { fields, .. } => fields
                .iter()
                .map(|(_, child)| Self::count_forced_boxes_for_entry(child))
                .sum(),
            VirtualStateInfo::VirtualArray { items, .. } => items
                .iter()
                .map(|child| Self::count_forced_boxes_for_entry(child))
                .sum(),
            VirtualStateInfo::VirtualArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, child)| child))
                .map(|child| Self::count_forced_boxes_for_entry(child))
                .sum(),
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => entries
                .iter()
                .map(|(_, _, child)| Self::count_forced_boxes_for_entry(child))
                .sum(),
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown => 1,
        }
    }

    /// Number of non-virtual values (need concrete OpRefs at loop entry).
    /// virtualstate.py: num_boxes()
    pub fn num_boxes(&self) -> usize {
        self.state
            .iter()
            .map(Self::count_forced_boxes_for_entry)
            .sum()
    }

    /// Total number of entries (virtual + non-virtual).
    pub fn num_entries(&self) -> usize {
        self.state.len()
    }

    /// Number of virtual entries.
    pub fn num_virtuals(&self) -> usize {
        self.state.iter().filter(|s| s.is_virtual()).count()
    }

    /// Whether this state has any virtual objects.
    pub fn has_virtuals(&self) -> bool {
        self.state.iter().any(|s| s.is_virtual())
    }

    /// Generate input argument OpRefs from this state.
    ///
    /// RPython: `VirtualState.make_inputargs()` — creates the non-virtual
    /// inputarg list for the loop header Label.
    ///
    /// `concrete_refs` is the full loop-carried value list. Virtual entries
    /// recursively contribute the concrete boxes of their forced fields/items.
    pub fn make_inputargs(&self, concrete_refs: &[OpRef], ctx: &OptContext) -> Vec<OpRef> {
        let mut args = vec![OpRef::NONE; self.num_boxes()];
        let mut next_slot = 0usize;
        for (idx, info) in self.state.iter().enumerate() {
            let opref = concrete_refs.get(idx).copied().unwrap_or(OpRef::NONE);
            Self::enum_forced_boxes_for_entry(info, opref, ctx, &mut args, &mut next_slot);
        }
        args.into_iter().filter(|opref| !opref.is_none()).collect()
    }

    /// Return the original top-level input slots corresponding to each
    /// non-virtual inputarg produced by `make_inputargs()`.
    ///
    /// This mirrors RPython's top-level inputarg ownership. Leaf boxes coming
    /// out of virtual fields do not correspond to a stable original input slot,
    /// so they are reported as `OpRef::NONE`.
    pub fn make_inputarg_source_slots(&self, concrete_refs: &[OpRef]) -> Vec<OpRef> {
        let mut sources = vec![OpRef::NONE; self.num_boxes()];
        let mut next_slot = 0usize;
        for (idx, info) in self.state.iter().enumerate() {
            let opref = concrete_refs.get(idx).copied().unwrap_or(OpRef::NONE);
            Self::enum_inputarg_source_slots(info, opref, true, &mut sources, &mut next_slot);
        }
        sources
            .into_iter()
            .filter(|opref| !opref.is_none())
            .collect()
    }

    /// virtualstate.py: make_inputargs_and_virtuals(oprefs)
    /// Returns `(inputargs, virtuals)` where `inputargs` contains the
    /// non-virtual boxes and `virtuals` contains the original virtual boxes.
    pub fn make_inputargs_and_virtuals(
        &self,
        concrete_refs: &[OpRef],
        ctx: &OptContext,
    ) -> (Vec<OpRef>, Vec<OpRef>) {
        let inputargs = self.make_inputargs(concrete_refs, ctx);
        let virtuals: Vec<OpRef> = self
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| concrete_refs.get(i).copied())
            .collect();
        (inputargs, virtuals)
    }

    /// RPython: `VirtualState.make_inputargs(..., optimizer, force_boxes=...)`
    ///
    /// This is the active path used by unroll jump matching. When a non-virtual
    /// slot receives a virtual box, `force_boxes=true` forces it through the
    /// optimizer instead of immediately failing.
    pub fn make_inputargs_and_virtuals_with_optimizer(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<(Vec<OpRef>, Vec<OpRef>), ()> {
        let mut args = vec![OpRef::NONE; self.num_boxes()];
        let mut next_slot = 0usize;
        for (idx, info) in self.state.iter().enumerate() {
            let opref = concrete_refs.get(idx).copied().unwrap_or(OpRef::NONE);
            Self::enum_forced_boxes_for_entry_with_optimizer(
                info,
                opref,
                optimizer,
                ctx,
                &mut args,
                &mut next_slot,
                force_boxes,
            )?;
        }
        let inputargs: Vec<OpRef> = args.into_iter().filter(|opref| !opref.is_none()).collect();
        let virtuals: Vec<OpRef> = self
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| concrete_refs.get(i).copied())
            .collect();
        Ok((inputargs, virtuals))
    }

    fn enum_forced_boxes_for_entry(
        info: &VirtualStateInfo,
        opref: OpRef,
        ctx: &OptContext,
        boxes: &mut [OpRef],
        next_slot: &mut usize,
    ) {
        match info {
            VirtualStateInfo::Constant(_) => {}
            VirtualStateInfo::Virtual { fields, .. }
            | VirtualStateInfo::VirtualStruct { fields, .. } => {
                let resolved = ctx.get_replacement(opref);
                let ptr_info = ctx.get_ptr_info(resolved);
                for (field_idx, field_state) in fields {
                    let field_ref = ptr_info
                        .and_then(|info| info.get_field(*field_idx))
                        // Fallback: use pre_force_field_refs if virtual was forced.
                        // Try both resolved and original opref (force may forward).
                        .or_else(|| {
                            ctx.pre_force_field_refs
                                .get(&resolved)
                                .or_else(|| ctx.pre_force_field_refs.get(&opref))
                                .and_then(|flds| {
                                    flds.iter()
                                        .find(|(idx, _)| *idx == *field_idx)
                                        .map(|(_, r)| *r)
                                })
                        })
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry(
                        field_state,
                        field_ref,
                        ctx,
                        boxes,
                        next_slot,
                    );
                }
            }
            VirtualStateInfo::VirtualArray { items, .. } => {
                let resolved = ctx.get_replacement(opref);
                let ptr_info = ctx.get_ptr_info(resolved);
                for (index, item_state) in items.iter().enumerate() {
                    let item_ref = ptr_info
                        .and_then(|info| info.get_item(index))
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry(item_state, item_ref, ctx, boxes, next_slot);
                }
            }
            VirtualStateInfo::VirtualArrayStruct { element_fields, .. } => {
                let resolved = ctx.get_replacement(opref);
                let ptr_info = ctx.get_ptr_info(resolved);
                let mut flat_index = 0usize;
                for fields in element_fields {
                    for (_, field_state) in fields {
                        let item_ref = ptr_info
                            .and_then(|info| info.get_item(flat_index))
                            .unwrap_or(OpRef::NONE);
                        Self::enum_forced_boxes_for_entry(
                            field_state,
                            item_ref,
                            ctx,
                            boxes,
                            next_slot,
                        );
                        flat_index += 1;
                    }
                }
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                let resolved = ctx.get_replacement(opref);
                let ptr_info = ctx.get_ptr_info(resolved);
                for (index, (_, _, entry_state)) in entries.iter().enumerate() {
                    let entry_ref = ptr_info
                        .and_then(|info| match info {
                            PtrInfo::VirtualRawBuffer(vinfo) => {
                                vinfo.entries.get(index).map(|(_, _, value)| *value)
                            }
                            _ => None,
                        })
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry(
                        entry_state,
                        entry_ref,
                        ctx,
                        boxes,
                        next_slot,
                    );
                }
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown => {
                if let Some(slot) = boxes.get_mut(*next_slot) {
                    *slot = ctx.get_replacement(opref);
                }
                *next_slot += 1;
            }
        }
    }

    fn enum_inputarg_source_slots(
        info: &VirtualStateInfo,
        opref: OpRef,
        top_level: bool,
        sources: &mut [OpRef],
        next_slot: &mut usize,
    ) {
        match info {
            VirtualStateInfo::Constant(_) => {}
            VirtualStateInfo::Virtual { fields, .. }
            | VirtualStateInfo::VirtualStruct { fields, .. } => {
                for (_, field_state) in fields {
                    Self::enum_inputarg_source_slots(
                        field_state,
                        OpRef::NONE,
                        false,
                        sources,
                        next_slot,
                    );
                }
            }
            VirtualStateInfo::VirtualArray { items, .. } => {
                for item_state in items {
                    Self::enum_inputarg_source_slots(
                        item_state,
                        OpRef::NONE,
                        false,
                        sources,
                        next_slot,
                    );
                }
            }
            VirtualStateInfo::VirtualArrayStruct { element_fields, .. } => {
                for fields in element_fields {
                    for (_, field_state) in fields {
                        Self::enum_inputarg_source_slots(
                            field_state,
                            OpRef::NONE,
                            false,
                            sources,
                            next_slot,
                        );
                    }
                }
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                for (_, _, entry_state) in entries {
                    Self::enum_inputarg_source_slots(
                        entry_state,
                        OpRef::NONE,
                        false,
                        sources,
                        next_slot,
                    );
                }
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown => {
                if let Some(slot) = sources.get_mut(*next_slot) {
                    *slot = if top_level { opref } else { OpRef::NONE };
                }
                *next_slot += 1;
            }
        }
    }

    fn enum_forced_boxes_for_entry_with_optimizer(
        info: &VirtualStateInfo,
        opref: OpRef,
        optimizer: &mut crate::optimizer::Optimizer,
        ctx: &mut OptContext,
        boxes: &mut [OpRef],
        next_slot: &mut usize,
        force_boxes: bool,
    ) -> Result<(), ()> {
        match info {
            VirtualStateInfo::Constant(_) => Ok(()),
            VirtualStateInfo::Virtual { fields, .. }
            | VirtualStateInfo::VirtualStruct { fields, .. } => {
                let resolved = ctx.get_replacement(opref);
                for (field_idx, field_state) in fields {
                    let field_ref = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| info.get_field(*field_idx))
                        .or_else(|| {
                            ctx.pre_force_field_refs
                                .get(&resolved)
                                .or_else(|| ctx.pre_force_field_refs.get(&opref))
                                .and_then(|flds| {
                                    flds.iter()
                                        .find(|(idx, _)| *idx == *field_idx)
                                        .map(|(_, r)| *r)
                                })
                        })
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry_with_optimizer(
                        field_state,
                        field_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::VirtualArray { items, .. } => {
                let resolved = ctx.get_replacement(opref);
                for (index, item_state) in items.iter().enumerate() {
                    let item_ref = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| info.get_item(index))
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry_with_optimizer(
                        item_state,
                        item_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::VirtualArrayStruct { element_fields, .. } => {
                let resolved = ctx.get_replacement(opref);
                let mut flat_index = 0usize;
                for fields in element_fields {
                    for (_, field_state) in fields {
                        let item_ref = ctx
                            .get_ptr_info(resolved)
                            .and_then(|info| info.get_item(flat_index))
                            .unwrap_or(OpRef::NONE);
                        Self::enum_forced_boxes_for_entry_with_optimizer(
                            field_state,
                            item_ref,
                            optimizer,
                            ctx,
                            boxes,
                            next_slot,
                            force_boxes,
                        )?;
                        flat_index += 1;
                    }
                }
                Ok(())
            }
            VirtualStateInfo::VirtualRawBuffer { entries, .. } => {
                let resolved = ctx.get_replacement(opref);
                for (index, (_, _, entry_state)) in entries.iter().enumerate() {
                    let entry_ref = ctx
                        .get_ptr_info(resolved)
                        .and_then(|info| match info {
                            PtrInfo::VirtualRawBuffer(vinfo) => {
                                vinfo.entries.get(index).map(|(_, _, value)| *value)
                            }
                            _ => None,
                        })
                        .unwrap_or(OpRef::NONE);
                    Self::enum_forced_boxes_for_entry_with_optimizer(
                        entry_state,
                        entry_ref,
                        optimizer,
                        ctx,
                        boxes,
                        next_slot,
                        force_boxes,
                    )?;
                }
                Ok(())
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown => {
                let resolved = ctx.get_replacement(opref);
                let forced = match ctx.get_ptr_info(resolved) {
                    // RPython: Virtualizable refs stay virtual across iterations.
                    Some(PtrInfo::Virtualizable(_)) => resolved,
                    Some(ptr_info) if ptr_info.is_virtual() => {
                        if !force_boxes {
                            return Err(());
                        }
                        optimizer.force_box(resolved, ctx)
                    }
                    _ => resolved,
                };
                if let Some(slot) = boxes.get_mut(*next_slot) {
                    *slot = ctx.get_replacement(forced);
                }
                *next_slot += 1;
                Ok(())
            }
        }
    }

    /// Check if another VirtualState is compatible (can reuse the optimized loop body).
    ///
    /// Returns true if all entries are compatible.
    pub fn is_compatible(&self, other: &VirtualState) -> bool {
        if self.state.len() != other.state.len() {
            return false;
        }
        self.state
            .iter()
            .zip(other.state.iter())
            .all(|(a, b)| a.is_compatible(b))
    }

    /// virtualstate.py: generalization_of(other, optimizer)
    ///
    /// `self` is the target loop state's requirement and `other` is the
    /// incoming state. Returns true if `self` can safely accept `other`.
    pub fn generalization_of(&self, other: &VirtualState) -> bool {
        self.is_compatible(other)
    }

    /// Generate guards to bridge from `other` state to `self` state.
    ///
    /// Returns a list of guard operations that need to be emitted to ensure
    /// the incoming values satisfy the requirements of the optimized loop.
    ///
    /// In RPython, this is done during `VirtualState.generate_guards()`.
    pub fn generate_guards(&self, other: &VirtualState) -> Vec<GuardRequirement> {
        let mut guards = Vec::new();

        for (i, (expected, incoming)) in self.state.iter().zip(other.state.iter()).enumerate() {
            Self::generate_guards_for_entry(i, expected, incoming, &mut guards);
        }

        guards
    }

    fn generate_guards_for_entry(
        arg_idx: usize,
        expected: &VirtualStateInfo,
        incoming: &VirtualStateInfo,
        guards: &mut Vec<GuardRequirement>,
    ) {
        match (expected, incoming) {
            // If expected is a known class but incoming is unknown, need a guard_class
            (VirtualStateInfo::KnownClass { class_ptr }, VirtualStateInfo::Unknown) => {
                guards.push(GuardRequirement::GuardClass {
                    arg_index: arg_idx,
                    expected_class: *class_ptr,
                });
            }

            // If expected is nonnull but incoming is unknown, need a guard_nonnull
            (VirtualStateInfo::NonNull, VirtualStateInfo::Unknown) => {
                guards.push(GuardRequirement::GuardNonnull { arg_index: arg_idx });
            }

            // If expected has bounds but incoming doesn't
            (VirtualStateInfo::IntBounded(bounds), VirtualStateInfo::Unknown) => {
                guards.push(GuardRequirement::GuardBounds {
                    arg_index: arg_idx,
                    bounds: bounds.clone(),
                });
            }

            // If expected is constant but incoming is unknown, need a guard_value
            (VirtualStateInfo::Constant(val), VirtualStateInfo::Unknown) => {
                guards.push(GuardRequirement::GuardValue {
                    arg_index: arg_idx,
                    expected_value: val.clone(),
                });
            }

            // virtualstate.py: VirtualArray with known length vs unknown.
            // Need guards to ensure the incoming array has the expected length.
            (
                VirtualStateInfo::VirtualArray {
                    items: expected_items,
                    ..
                },
                VirtualStateInfo::Unknown,
            ) => {
                // The bridge needs to provide an array with exactly this many items.
                let expected_len = expected_items.len() as i64;
                guards.push(GuardRequirement::GuardValue {
                    arg_index: arg_idx,
                    expected_value: Value::Int(expected_len),
                });
            }

            _ => {} // Already compatible or will fail at is_compatible check
        }
    }

    /// virtualstate.py: compute_renum(oprefs)
    /// Build the renum mapping from OpRef to numbering index.
    /// This ensures consistent virtual identity across loop iterations.
    pub fn compute_renum(&mut self, oprefs: &[OpRef]) {
        self.renum.clear();
        for (i, &opref) in oprefs.iter().enumerate() {
            if !opref.is_none() {
                self.renum.insert(opref, i);
            }
        }
    }

    /// Get the numbering index for an OpRef.
    pub fn get_renum(&self, opref: OpRef) -> Option<usize> {
        self.renum.get(&opref).copied()
    }

    /// virtualstate.py: enum_forced_boxes(boxes, optimizer)
    /// Collect the concrete OpRefs for non-virtual entries.
    /// Virtual entries are skipped (they don't need concrete boxes).
    pub fn enum_forced_boxes(&self, oprefs: &[OpRef]) -> Vec<OpRef> {
        self.state
            .iter()
            .zip(oprefs.iter())
            .filter(|(info, _)| !info.is_virtual())
            .map(|(_, opref)| *opref)
            .collect()
    }

    /// virtualstate.py: debug_print(hdr, bad, metainterp_sd)
    /// Format the virtual state for debugging.
    pub fn debug_print(&self) -> String {
        let mut out = String::new();
        for (i, info) in self.state.iter().enumerate() {
            let kind = match info {
                VirtualStateInfo::Constant(_) => "Constant",
                VirtualStateInfo::Virtual { .. } => "Virtual",
                VirtualStateInfo::VirtualArray { .. } => "VArray",
                VirtualStateInfo::VirtualStruct { .. } => "VStruct",
                VirtualStateInfo::VirtualArrayStruct { .. } => "VArrayStruct",
                VirtualStateInfo::VirtualRawBuffer { .. } => "VRawBuf",
                VirtualStateInfo::KnownClass { .. } => "KnownClass",
                VirtualStateInfo::NonNull => "NonNull",
                VirtualStateInfo::IntBounded(_) => "IntBounded",
                VirtualStateInfo::Unknown => "Unknown",
            };
            out.push_str(&format!("  [{i}] {kind}\n"));
        }
        out
    }
}

impl VirtualState {
    /// virtualstate.py: generate_guards(other, boxes, runtime_boxes, optimizer)
    ///
    /// Full guard generation with GenerateGuardState context.
    /// For each state entry, generate guards that make the runtime values
    /// match the expected virtual state shape.
    pub fn generate_guards_with_state<'a>(
        &self,
        other: &VirtualState,
        boxes: &[OpRef],
        ctx: &'a OptContext,
    ) -> GenerateGuardState<'a> {
        let mut state = GenerateGuardState::new(ctx);
        let len = self.state.len().min(other.state.len()).min(boxes.len());
        for i in 0..len {
            self.generate_guard_for_entry(i, &self.state[i], &other.state[i], boxes[i], &mut state);
        }
        state
    }

    /// Per-entry guard generation (virtualstate.py: AbstractVirtualStateInfo.generate_guards)
    fn generate_guard_for_entry(
        &self,
        idx: usize,
        expected: &VirtualStateInfo,
        incoming: &VirtualStateInfo,
        _box_opref: OpRef,
        state: &mut GenerateGuardState,
    ) {
        match (expected, incoming) {
            // Constant vs non-constant → need GUARD_VALUE
            (VirtualStateInfo::Constant(val), _) => {
                if let VirtualStateInfo::Constant(other_val) = incoming {
                    if val == other_val {
                        return; // same constant, no guard needed
                    }
                }
                state.mark_bad(idx, "constant mismatch");
            }
            // KnownClass vs unknown → GUARD_CLASS
            (VirtualStateInfo::KnownClass { .. }, VirtualStateInfo::Unknown) => {
                // Guard will be generated by the caller from GuardRequirement
            }
            // NonNull vs unknown → GUARD_NONNULL
            (VirtualStateInfo::NonNull, VirtualStateInfo::Unknown) => {}
            // IntBounded vs unknown → bounds guards
            (VirtualStateInfo::IntBounded(_), VirtualStateInfo::Unknown) => {}
            // Virtual vs Virtual → check fields match
            (
                VirtualStateInfo::Virtual { descr: d1, .. },
                VirtualStateInfo::Virtual { descr: d2, .. },
            ) => {
                if d1.index() != d2.index() {
                    state.mark_bad(idx, "virtual descriptor mismatch");
                }
            }
            // Virtual vs non-virtual → cannot match
            (VirtualStateInfo::Virtual { .. }, _) => {
                state.mark_bad(idx, "expected virtual, got non-virtual");
            }
            // Compatible or both unknown → ok
            _ => {}
        }
    }

    /// virtualstate.py: force_boxes(optimizer) — force all virtual entries
    /// to be materialized. After calling this, all entries become non-virtual
    /// (Constant, KnownClass, NonNull, IntBounded, or Unknown).
    ///
    /// Returns the number of virtuals that were forced.
    pub fn force_boxes(&mut self) -> usize {
        let mut count = 0;
        for info in &mut self.state {
            if info.is_virtual() {
                // virtualstate.py: forced virtuals become NonNull
                // (they were allocated, so they're always non-null).
                match info {
                    VirtualStateInfo::Virtual { known_class, .. } => {
                        *info = if let Some(cls) = known_class.take() {
                            VirtualStateInfo::KnownClass { class_ptr: cls }
                        } else {
                            VirtualStateInfo::NonNull
                        };
                    }
                    _ => {
                        *info = VirtualStateInfo::NonNull;
                    }
                }
                count += 1;
            }
        }
        count
    }

    /// Get the lenbound of a virtual array at the given index, if any.
    pub fn get_lenbound(&self, index: usize) -> Option<&IntBound> {
        match self.state.get(index) {
            Some(VirtualStateInfo::VirtualArray { lenbound, .. }) => lenbound.as_ref(),
            _ => None,
        }
    }

    /// Merge two virtual states. For each entry, take the weaker
    /// (more general) of the two. Used when multiple paths converge.
    pub fn merge(&self, other: &VirtualState) -> VirtualState {
        let merged: Vec<VirtualStateInfo> = self
            .state
            .iter()
            .zip(other.state.iter())
            .map(|(a, b)| {
                if a.is_compatible(b) {
                    a.clone()
                } else {
                    VirtualStateInfo::Unknown
                }
            })
            .collect();
        VirtualState::new(merged)
    }
}

/// A guard that must be emitted to make an incoming state compatible.
#[derive(Clone, Debug)]
pub enum GuardRequirement {
    /// Emit GUARD_CLASS on the arg at this index.
    GuardClass {
        arg_index: usize,
        expected_class: GcRef,
    },
    /// Emit GUARD_NONNULL on the arg at this index.
    GuardNonnull { arg_index: usize },
    /// Emit GUARD_VALUE on the arg at this index.
    GuardValue {
        arg_index: usize,
        expected_value: Value,
    },
    /// Emit integer bounds guards on the arg at this index.
    GuardBounds { arg_index: usize, bounds: IntBound },
}

impl GuardRequirement {
    /// Convert this guard requirement into a concrete Op, given the
    /// concrete args vector. Returns None if the arg_index is out of bounds.
    pub fn to_op(&self, args: &[OpRef]) -> Option<Op> {
        match self {
            GuardRequirement::GuardClass {
                arg_index,
                expected_class,
            } => {
                let arg = *args.get(*arg_index)?;
                let class_const = OpRef(10_000 + *arg_index as u32); // placeholder
                let mut op = Op::new(OpCode::GuardClass, &[arg, class_const]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardNonnull { arg_index } => {
                let arg = *args.get(*arg_index)?;
                let mut op = Op::new(OpCode::GuardNonnull, &[arg]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardValue {
                arg_index,
                expected_value,
            } => {
                let arg = *args.get(*arg_index)?;
                let val_const = OpRef(10_000 + *arg_index as u32); // placeholder
                let mut op = Op::new(OpCode::GuardValue, &[arg, val_const]);
                op.fail_args = Some(Default::default());
                Some(op)
            }
            GuardRequirement::GuardBounds { arg_index, bounds } => {
                // IntBound guards are emitted as int_ge/int_le pairs
                // For now, skip — intbounds pass handles these
                None
            }
        }
    }
}

/// virtualstate.py: VirtualStateConstructor — visitor-based factory
/// for building VirtualState from optimizer state.
///
/// Walks the optimization context and PtrInfo table to create
/// a VirtualState snapshot for a set of loop-carried values.
pub struct VirtualStateConstructor<'a> {
    ctx: &'a OptContext,
    ptr_info: &'a [Option<PtrInfo>],
}

impl<'a> VirtualStateConstructor<'a> {
    pub fn new(ctx: &'a OptContext, ptr_info: &'a [Option<PtrInfo>]) -> Self {
        VirtualStateConstructor { ctx, ptr_info }
    }

    /// Build a VirtualState for the given OpRefs.
    pub fn build(&self, oprefs: &[OpRef]) -> VirtualState {
        export_state(oprefs, self.ctx, self.ptr_info)
    }

    /// Build VirtualState for label args and return it along with
    /// the non-virtual inputargs.
    pub fn build_with_inputargs(&self, oprefs: &[OpRef]) -> (VirtualState, Vec<OpRef>) {
        let state = self.build(oprefs);
        let inputargs = state.make_inputargs(oprefs, self.ctx);
        (state, inputargs)
    }
}

/// Export the abstract state of loop-carried values.
///
/// Given the current optimization context and PtrInfo table (from the virtualize pass),
/// create a VirtualState snapshot for the given OpRefs (typically the Jump args).
///
/// virtualstate.py: VirtualStateConstructor.make_virtual_state()
pub fn export_state(
    oprefs: &[OpRef],
    ctx: &OptContext,
    ptr_info: &[Option<PtrInfo>],
) -> VirtualState {
    let state = oprefs
        .iter()
        .map(|opref| {
            let resolved = ctx.get_replacement(*opref);
            export_single_value(resolved, ctx, ptr_info, &mut HashMap::new())
        })
        .collect();
    VirtualState::new(state)
}

pub(crate) fn export_value_state(
    opref: OpRef,
    ctx: &OptContext,
    ptr_info: &[Option<PtrInfo>],
) -> VirtualStateInfo {
    export_single_value(
        ctx.get_replacement(opref),
        ctx,
        ptr_info,
        &mut HashMap::new(),
    )
}

/// Export abstract info for a single value.
fn export_single_value(
    opref: OpRef,
    ctx: &OptContext,
    ptr_info: &[Option<PtrInfo>],
    visited: &mut HashMap<OpRef, ()>,
) -> VirtualStateInfo {
    // Prevent infinite recursion on circular references
    if visited.contains_key(&opref) {
        return VirtualStateInfo::Unknown;
    }
    visited.insert(opref, ());

    // Check for known constant
    if let Some(value) = ctx.get_constant(opref) {
        return VirtualStateInfo::Constant(value.clone());
    }

    // Check PtrInfo
    let idx = opref.0 as usize;
    if let Some(Some(info)) = ptr_info.get(idx) {
        match info {
            PtrInfo::Virtual(vinfo) => {
                let fields = vinfo
                    .fields
                    .iter()
                    .map(|(field_idx, field_ref)| {
                        let field_state = export_single_value(*field_ref, ctx, ptr_info, visited);
                        (*field_idx, Box::new(field_state))
                    })
                    .collect();
                return VirtualStateInfo::Virtual {
                    descr: vinfo.descr.clone(),
                    known_class: vinfo.known_class,
                    fields,
                    field_descrs: vinfo.field_descrs.clone(),
                };
            }
            PtrInfo::VirtualArray(vinfo) => {
                let items: Vec<Box<VirtualStateInfo>> = vinfo
                    .items
                    .iter()
                    .map(|item_ref| {
                        let item_state = export_single_value(*item_ref, ctx, ptr_info, visited);
                        Box::new(item_state)
                    })
                    .collect();
                let len = items.len();
                return VirtualStateInfo::VirtualArray {
                    descr: vinfo.descr.clone(),
                    items,
                    lenbound: Some(IntBound::from_constant(len as i64)),
                };
            }
            PtrInfo::VirtualStruct(vinfo) => {
                let fields = vinfo
                    .fields
                    .iter()
                    .map(|(field_idx, field_ref)| {
                        let field_state = export_single_value(*field_ref, ctx, ptr_info, visited);
                        (*field_idx, Box::new(field_state))
                    })
                    .collect();
                return VirtualStateInfo::VirtualStruct {
                    descr: vinfo.descr.clone(),
                    fields,
                    field_descrs: vinfo.field_descrs.clone(),
                };
            }
            PtrInfo::VirtualArrayStruct(vinfo) => {
                let element_fields = vinfo
                    .element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_ref)| {
                                let field_state =
                                    export_single_value(*field_ref, ctx, ptr_info, visited);
                                (*field_idx, Box::new(field_state))
                            })
                            .collect()
                    })
                    .collect();
                return VirtualStateInfo::VirtualArrayStruct {
                    descr: vinfo.descr.clone(),
                    element_fields,
                };
            }
            PtrInfo::VirtualRawBuffer(vinfo) => {
                let entries = vinfo
                    .entries
                    .iter()
                    .map(|(offset, length, value_ref)| {
                        let val_state = export_single_value(*value_ref, ctx, ptr_info, visited);
                        (*offset, *length, Box::new(val_state))
                    })
                    .collect();
                return VirtualStateInfo::VirtualRawBuffer {
                    size: vinfo.size,
                    entries,
                };
            }
            PtrInfo::KnownClass {
                class_ptr,
                is_nonnull: _,
            } => {
                return VirtualStateInfo::KnownClass {
                    class_ptr: *class_ptr,
                };
            }
            PtrInfo::NonNull => {
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Constant(gcref) => {
                return VirtualStateInfo::Constant(Value::Ref(*gcref));
            }
            PtrInfo::Virtualizable(_) => {
                // Virtualizable objects are treated as non-null in virtual state
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Instance(_) | PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                return VirtualStateInfo::NonNull;
            }
        }
    }

    VirtualStateInfo::Unknown
}

/// Import a virtual state: apply known info to the optimization context.
///
/// For each loop-carried variable, set up the context so downstream passes
/// "see" the info that was exported at the end of the preamble.
///
/// `target_oprefs` are the OpRefs of the Label args (loop header inputs).
pub fn import_state(
    vstate: &VirtualState,
    target_oprefs: &[OpRef],
    ctx: &mut OptContext,
    ptr_info: &mut Vec<Option<PtrInfo>>,
) {
    for (info, opref) in vstate.state.iter().zip(target_oprefs.iter()) {
        import_single_value(info, *opref, ctx, ptr_info);
    }
}

/// Import abstract info for a single value into the optimizer state.
fn import_single_value(
    info: &VirtualStateInfo,
    opref: OpRef,
    ctx: &mut OptContext,
    ptr_info: &mut Vec<Option<PtrInfo>>,
) {
    let idx = opref.0 as usize;
    if idx >= ptr_info.len() {
        ptr_info.resize(idx + 1, None);
    }

    match info {
        VirtualStateInfo::Constant(val) => {
            ctx.make_constant(opref, val.clone());
        }
        VirtualStateInfo::Virtual {
            descr,
            known_class,
            fields,
            field_descrs,
        } => {
            let mut vfields = Vec::new();
            for (field_idx, field_info) in fields {
                // Create a synthetic OpRef for the field value
                let field_opref =
                    ctx.emit(majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef::NONE]));
                import_single_value(field_info, field_opref, ctx, ptr_info);
                vfields.push((*field_idx, field_opref));
            }
            ptr_info[idx] = Some(PtrInfo::Virtual(VirtualInfo {
                descr: descr.clone(),
                known_class: *known_class,
                fields: vfields,
                field_descrs: field_descrs.clone(),
            }));
        }
        VirtualStateInfo::VirtualArray { descr, items, .. } => {
            let mut vitems = Vec::new();
            for item_info in items {
                let item_opref =
                    ctx.emit(majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef::NONE]));
                import_single_value(item_info, item_opref, ctx, ptr_info);
                vitems.push(item_opref);
            }
            ptr_info[idx] = Some(PtrInfo::VirtualArray(VirtualArrayInfo {
                descr: descr.clone(),
                items: vitems,
            }));
        }
        VirtualStateInfo::VirtualStruct {
            descr,
            fields,
            field_descrs,
        } => {
            let mut vfields = Vec::new();
            for (field_idx, field_info) in fields {
                let field_opref =
                    ctx.emit(majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef::NONE]));
                import_single_value(field_info, field_opref, ctx, ptr_info);
                vfields.push((*field_idx, field_opref));
            }
            ptr_info[idx] = Some(PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vfields,
                field_descrs: field_descrs.clone(),
            }));
        }
        VirtualStateInfo::VirtualArrayStruct {
            descr,
            element_fields,
        } => {
            let mut imported_elements = Vec::new();
            for fields in element_fields {
                let mut imported_fields = Vec::new();
                for (field_idx, field_info) in fields {
                    let field_opref =
                        ctx.emit(majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef::NONE]));
                    import_single_value(field_info, field_opref, ctx, ptr_info);
                    imported_fields.push((*field_idx, field_opref));
                }
                imported_elements.push(imported_fields);
            }
            ptr_info[idx] = Some(PtrInfo::VirtualArrayStruct(VirtualArrayStructInfo {
                descr: descr.clone(),
                element_fields: imported_elements,
            }));
        }
        VirtualStateInfo::VirtualRawBuffer { size, entries } => {
            let mut imported_entries = Vec::new();
            for (offset, length, entry_info) in entries {
                let entry_opref =
                    ctx.emit(majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef::NONE]));
                import_single_value(entry_info, entry_opref, ctx, ptr_info);
                imported_entries.push((*offset, *length, entry_opref));
            }
            ptr_info[idx] = Some(PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                size: *size,
                entries: imported_entries,
            }));
        }
        VirtualStateInfo::KnownClass { class_ptr } => {
            ptr_info[idx] = Some(PtrInfo::KnownClass {
                class_ptr: *class_ptr,
                is_nonnull: true,
            });
        }
        VirtualStateInfo::NonNull => {
            ptr_info[idx] = Some(PtrInfo::NonNull);
        }
        VirtualStateInfo::IntBounded(bound) => {
            // RPython virtualstate.py NotVirtualStateInfoInt: propagate
            // IntBound info so the IntBounds pass can intersect with it.
            // This tells the body optimizer that this value is a raw int,
            // preventing it from being treated as a boxed Ref.
            ctx.imported_int_bounds.insert(opref, bound.clone());
        }
        VirtualStateInfo::Unknown => {
            // Nothing to import.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Descr, FieldDescr, GcRef, Op, OpCode, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr(u32);
    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }
    impl FieldDescr for TestDescr {
        fn offset(&self) -> usize {
            self.0 as usize
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    fn test_descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    // ── Compatibility tests ──

    #[test]
    fn test_unknown_accepts_anything() {
        let unknown = VirtualStateInfo::Unknown;
        assert!(unknown.is_compatible(&VirtualStateInfo::Unknown));
        assert!(unknown.is_compatible(&VirtualStateInfo::NonNull));
        assert!(unknown.is_compatible(&VirtualStateInfo::Constant(Value::Int(42))));
    }

    #[test]
    fn test_constant_compatibility() {
        let c1 = VirtualStateInfo::Constant(Value::Int(42));
        let c2 = VirtualStateInfo::Constant(Value::Int(42));
        let c3 = VirtualStateInfo::Constant(Value::Int(99));

        assert!(c1.is_compatible(&c2));
        assert!(!c1.is_compatible(&c3));
        assert!(!c1.is_compatible(&VirtualStateInfo::Unknown));
    }

    #[test]
    fn test_nonnull_compatibility() {
        let nn = VirtualStateInfo::NonNull;
        assert!(nn.is_compatible(&VirtualStateInfo::NonNull));
        assert!(nn.is_compatible(&VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100)
        }));
        assert!(!nn.is_compatible(&VirtualStateInfo::Unknown));
    }

    #[test]
    fn test_known_class_compatibility() {
        let kc1 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100),
        };
        let kc2 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x100),
        };
        let kc3 = VirtualStateInfo::KnownClass {
            class_ptr: GcRef(0x200),
        };

        assert!(kc1.is_compatible(&kc2));
        assert!(!kc1.is_compatible(&kc3));
        assert!(!kc1.is_compatible(&VirtualStateInfo::Unknown));
    }

    #[test]
    fn test_virtual_compatibility() {
        let descr = test_descr(1);
        let v1 = VirtualStateInfo::Virtual {
            descr: descr.clone(),
            known_class: None,
            fields: vec![
                (0, Box::new(VirtualStateInfo::Constant(Value::Int(10)))),
                (1, Box::new(VirtualStateInfo::Unknown)),
            ],
            field_descrs: Vec::new(),
        };
        let v2 = VirtualStateInfo::Virtual {
            descr: descr.clone(),
            known_class: None,
            fields: vec![
                (0, Box::new(VirtualStateInfo::Constant(Value::Int(10)))),
                (1, Box::new(VirtualStateInfo::Constant(Value::Int(20)))),
            ],
            field_descrs: Vec::new(),
        };
        let v3 = VirtualStateInfo::Virtual {
            descr: descr.clone(),
            known_class: None,
            fields: vec![(0, Box::new(VirtualStateInfo::Constant(Value::Int(99))))],
            field_descrs: Vec::new(),
        };

        assert!(v1.is_compatible(&v2)); // field 0 matches, field 1 is Unknown (accepts anything)
        assert!(!v1.is_compatible(&v3)); // field 0 constant mismatch
    }

    #[test]
    fn test_virtual_array_compatibility() {
        let descr = test_descr(1);
        let a1 = VirtualStateInfo::VirtualArray {
            descr: descr.clone(),
            items: vec![
                Box::new(VirtualStateInfo::Constant(Value::Int(1))),
                Box::new(VirtualStateInfo::Unknown),
            ],
            lenbound: None,
        };
        let a2 = VirtualStateInfo::VirtualArray {
            descr: descr.clone(),
            items: vec![
                Box::new(VirtualStateInfo::Constant(Value::Int(1))),
                Box::new(VirtualStateInfo::Constant(Value::Int(2))),
            ],
            lenbound: None,
        };
        let a3 = VirtualStateInfo::VirtualArray {
            descr: descr.clone(),
            items: vec![Box::new(VirtualStateInfo::Constant(Value::Int(1)))],
            lenbound: None,
        };

        assert!(a1.is_compatible(&a2)); // same length, first matches, second is Unknown
        assert!(!a1.is_compatible(&a3)); // different length
    }

    #[test]
    fn test_int_bounded_compatibility() {
        let b1 = VirtualStateInfo::IntBounded(IntBound::bounded(0, 100));
        let b2 = VirtualStateInfo::IntBounded(IntBound::bounded(10, 50));
        let b3 = VirtualStateInfo::IntBounded(IntBound::bounded(-10, 200));
        let c = VirtualStateInfo::Constant(Value::Int(42));

        assert!(b1.is_compatible(&b2)); // b2 is within b1
        assert!(!b1.is_compatible(&b3)); // b3 exceeds b1
        assert!(b1.is_compatible(&c)); // 42 is within [0, 100]
    }

    // ── VirtualState tests ──

    #[test]
    fn test_virtual_state_compatible() {
        let s1 = VirtualState::new(vec![VirtualStateInfo::Unknown, VirtualStateInfo::NonNull]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Constant(Value::Int(42)),
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x100),
            },
        ]);

        assert!(s1.is_compatible(&s2));
    }

    #[test]
    fn test_virtual_state_generalization_direction_matches_rpython() {
        let target = VirtualState::new(vec![VirtualStateInfo::Unknown]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::NonNull]);

        assert!(target.generalization_of(&incoming));
        assert!(!incoming.generalization_of(&target));
    }

    #[test]
    fn test_virtual_state_incompatible_length() {
        let s1 = VirtualState::new(vec![VirtualStateInfo::Unknown]);
        let s2 = VirtualState::new(vec![VirtualStateInfo::Unknown, VirtualStateInfo::Unknown]);

        assert!(!s1.is_compatible(&s2));
    }

    #[test]
    fn test_virtual_state_generate_guards() {
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x100),
            },
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![VirtualStateInfo::Unknown, VirtualStateInfo::Unknown]);

        let guards = s1.generate_guards(&s2);
        assert_eq!(guards.len(), 2);
        assert!(matches!(
            &guards[0],
            GuardRequirement::GuardClass { arg_index: 0, .. }
        ));
        assert!(matches!(
            &guards[1],
            GuardRequirement::GuardNonnull { arg_index: 1 }
        ));
    }

    // ── Export/Import tests ──

    #[test]
    fn test_export_constant() {
        let mut ctx = OptContext::new(10);
        let opref = ctx.emit(Op::new(OpCode::SameAsI, &[OpRef::NONE]));
        ctx.make_constant(opref, Value::Int(42));

        let ptr_info: Vec<Option<PtrInfo>> = Vec::new();
        let state = export_state(&[opref], &ctx, &ptr_info);

        assert_eq!(state.state.len(), 1);
        assert!(matches!(
            &state.state[0],
            VirtualStateInfo::Constant(Value::Int(42))
        ));
    }

    #[test]
    fn test_export_virtual() {
        let mut ctx = OptContext::new(10);
        let opref = ctx.emit(Op::new(OpCode::NewWithVtable, &[]));
        let field_ref = ctx.emit(Op::new(OpCode::SameAsI, &[OpRef::NONE]));
        ctx.make_constant(field_ref, Value::Int(99));

        let descr = test_descr(1);
        let mut ptr_info: Vec<Option<PtrInfo>> = vec![None; 2];
        ptr_info[opref.0 as usize] = Some(PtrInfo::Virtual(VirtualInfo {
            descr: descr.clone(),
            known_class: None,
            fields: vec![(0, field_ref)],
            field_descrs: Vec::new(),
        }));

        let state = export_state(&[opref], &ctx, &ptr_info);

        assert_eq!(state.state.len(), 1);
        match &state.state[0] {
            VirtualStateInfo::Virtual { fields, .. } => {
                assert_eq!(fields.len(), 1);
                assert!(matches!(
                    fields[0].1.as_ref(),
                    VirtualStateInfo::Constant(Value::Int(99))
                ));
            }
            other => panic!("expected Virtual, got {:?}", other),
        }
    }

    #[test]
    fn test_export_import_roundtrip() {
        // Export a state, then import it and verify the optimizer has the right info
        let mut ctx = OptContext::new(10);
        let opref = ctx.emit(Op::new(OpCode::SameAsI, &[OpRef::NONE]));
        ctx.make_constant(opref, Value::Int(42));

        let ptr_info_in: Vec<Option<PtrInfo>> = Vec::new();
        let state = export_state(&[opref], &ctx, &ptr_info_in);

        // Now import into a fresh context
        let mut ctx2 = OptContext::new(10);
        let target = ctx2.emit(Op::new(OpCode::SameAsI, &[OpRef::NONE]));
        let mut ptr_info_out: Vec<Option<PtrInfo>> = Vec::new();

        import_state(&state, &[target], &mut ctx2, &mut ptr_info_out);

        // The target should now be a known constant
        assert_eq!(ctx2.get_constant_int(target), Some(42));
    }

    #[test]
    fn test_export_known_class() {
        let mut ctx = OptContext::new(10);
        let opref = ctx.emit(Op::new(OpCode::SameAsI, &[OpRef::NONE]));

        let mut ptr_info: Vec<Option<PtrInfo>> = vec![None; 1];
        ptr_info[0] = Some(PtrInfo::KnownClass {
            class_ptr: GcRef(0x100),
            is_nonnull: true,
        });

        let state = export_state(&[opref], &ctx, &ptr_info);

        assert!(matches!(
            &state.state[0],
            VirtualStateInfo::KnownClass { class_ptr } if class_ptr.as_usize() == 0x100
        ));
    }

    #[test]
    fn test_is_virtual() {
        let descr = test_descr(1);
        assert!(
            VirtualStateInfo::Virtual {
                descr: descr.clone(),
                known_class: None,
                fields: vec![],
                field_descrs: Vec::new(),
            }
            .is_virtual()
        );

        assert!(
            VirtualStateInfo::VirtualArray {
                descr: descr.clone(),
                items: vec![],
                lenbound: None,
            }
            .is_virtual()
        );

        assert!(!VirtualStateInfo::NonNull.is_virtual());
        assert!(!VirtualStateInfo::Unknown.is_virtual());
    }

    #[test]
    fn test_make_inputargs_skips_virtual_entries() {
        let descr = test_descr(7);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown,
            VirtualStateInfo::VirtualStruct {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let ctx = OptContext::new(16);
        let inputargs = state.make_inputargs(&[OpRef(10), OpRef(11), OpRef(12)], &ctx);
        assert_eq!(inputargs, vec![OpRef(10), OpRef(12)]);
    }

    #[test]
    fn test_make_inputargs_and_virtuals_returns_virtual_boxes() {
        let descr = test_descr(9);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown,
            VirtualStateInfo::VirtualStruct {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let ctx = OptContext::new(16);
        let (inputargs, virtuals) =
            state.make_inputargs_and_virtuals(&[OpRef(20), OpRef(21), OpRef(22)], &ctx);
        assert_eq!(inputargs, vec![OpRef(20), OpRef(22)]);
        assert_eq!(virtuals, vec![OpRef(21)]);
    }

    #[test]
    fn test_make_inputargs_recursively_extracts_virtual_fields() {
        let descr = test_descr(11);
        let field_value = OpRef(21);
        let virtual_ref = OpRef(20);
        let state = VirtualState::new(vec![VirtualStateInfo::VirtualStruct {
            descr: descr.clone(),
            fields: vec![
                (0, Box::new(VirtualStateInfo::Constant(Value::Int(7)))),
                (8, Box::new(VirtualStateInfo::NonNull)),
            ],
            field_descrs: Vec::new(),
        }]);
        let mut ctx = OptContext::new(32);
        ctx.set_ptr_info(
            virtual_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![(0, OpRef::NONE), (8, field_value)],
                field_descrs: Vec::new(),
            }),
        );

        let inputargs = state.make_inputargs(&[virtual_ref], &ctx);
        assert_eq!(inputargs, vec![field_value]);
    }

    #[test]
    fn test_force_boxes() {
        let descr = test_descr(0);
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Virtual {
                descr: descr.clone(),
                known_class: Some(GcRef(0x1000)),
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
            VirtualStateInfo::VirtualArray {
                descr,
                items: vec![Box::new(VirtualStateInfo::Unknown)],
                lenbound: None,
            },
            VirtualStateInfo::Unknown,
        ]);
        assert_eq!(state.num_virtuals(), 2);
        let forced = state.force_boxes();
        assert_eq!(forced, 2);
        assert_eq!(state.num_virtuals(), 0);
        // Virtual with known_class becomes KnownClass
        assert!(matches!(
            &state.state[0],
            VirtualStateInfo::KnownClass { .. }
        ));
        // VirtualArray becomes NonNull
        assert!(matches!(&state.state[2], VirtualStateInfo::NonNull));
    }

    #[test]
    fn test_make_inputargs_with_optimizer_retries_virtual_into_nonvirtual_slot() {
        let descr = test_descr(12);
        let virtual_ref = OpRef(20);
        let state = VirtualState::new(vec![VirtualStateInfo::NonNull]);
        let mut ctx = OptContext::with_num_inputs(32, 1024);
        ctx.set_ptr_info(
            virtual_ref,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: vec![],
                field_descrs: Vec::new(),
            }),
        );
        let mut optimizer = crate::optimizer::Optimizer::new();

        assert!(state
            .make_inputargs_and_virtuals_with_optimizer(
                &[virtual_ref],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .is_err());

        let (inputargs, virtuals) = state
            .make_inputargs_and_virtuals_with_optimizer(
                &[virtual_ref],
                &mut optimizer,
                &mut ctx,
                true,
            )
            .expect("force_boxes=True should retry instead of failing");
        // After forcing, the virtual struct is replaced by a concrete
        // allocation at a new position. The inputarg should be that
        // forced allocation ref (which is what ctx.get_replacement
        // resolves the original virtual_ref to).
        assert_eq!(inputargs.len(), 1);
        assert_eq!(inputargs[0], ctx.get_replacement(virtual_ref));
        assert!(virtuals.is_empty());
    }

    #[test]
    fn test_compute_renum() {
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Unknown,
            VirtualStateInfo::NonNull,
            VirtualStateInfo::Unknown,
        ]);
        state.compute_renum(&[OpRef(10), OpRef(20), OpRef(30)]);
        assert_eq!(state.get_renum(OpRef(10)), Some(0));
        assert_eq!(state.get_renum(OpRef(20)), Some(1));
        assert_eq!(state.get_renum(OpRef(30)), Some(2));
        assert_eq!(state.get_renum(OpRef(99)), None);
    }

    #[test]
    fn test_merge_states() {
        let descr = test_descr(0);
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x1000),
            },
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::KnownClass {
                class_ptr: GcRef(0x1000),
            },
            VirtualStateInfo::Unknown,
        ]);
        let merged = s1.merge(&s2);
        // First: compatible (same class) → keep s1
        assert!(matches!(
            &merged.state[0],
            VirtualStateInfo::KnownClass { .. }
        ));
        // Second: NonNull vs Unknown → not compatible → Unknown
        assert!(matches!(&merged.state[1], VirtualStateInfo::Unknown));
    }

    #[test]
    fn test_num_entries_has_virtuals() {
        let descr = test_descr(0);
        let state = VirtualState::new(vec![
            VirtualStateInfo::NonNull,
            VirtualStateInfo::Virtual {
                descr,
                known_class: None,
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::Unknown,
        ]);
        assert_eq!(state.num_entries(), 3);
        assert_eq!(state.num_boxes(), 2); // NonNull + Unknown
        assert_eq!(state.num_virtuals(), 1);
        assert!(state.has_virtuals());
    }
}
