//! Virtualizable framework: optimizing interpreter frame access in JIT code.
//!
//! In RPython's JIT, a "virtualizable" is an object (typically the interpreter
//! frame) whose fields are stored in registers/stack during JIT execution rather
//! than on the heap. This avoids expensive memory reads/writes for hot fields.
//!
//! The key mechanism:
//! - During JIT execution, virtualizable fields live in the compiled code's
//!   registers/stack, NOT in the actual heap object.
//! - A `vable_token` field on the heap object tracks whether JIT code is
//!   currently "borrowing" the fields.
//! - When non-JIT code needs to access the frame, the token is checked and
//!   the fields are flushed back to the heap (force/synchronize).
//!
//! This module provides the Rust equivalent of RPython's `virtualizable.py`.

use indexmap::IndexMap;
use std::sync::{Arc, Weak};

use majit_ir::{DescrRef, Type, descr::descr_identity};

/// `virtualizable.py TOKEN_TRACING_RESCALL`: the GCREF address of the
/// prebuilt `JITFRAME_DUMMY` object shared with virtual references.
#[inline]
pub fn token_tracing_rescall() -> u64 {
    crate::virtualref::token_tracing_rescall() as usize as u64
}

/// Token states for virtualizable objects.
///
/// TOKEN_NONE (0): not in JIT.
/// TOKEN_TRACING_RESCALL (prebuilt GCREF): tracing + residual call in progress.
/// Any other non-zero value: active JIT frame pointer (FORCE_TOKEN).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableToken {
    /// No JIT code is currently using this virtualizable.
    None,
    /// JIT tracing is active and a residual call is in progress.
    TracingRescall,
    /// JIT compiled code is executing with this virtualizable.
    /// The value is the force_token (address of the JIT frame).
    Active(u64),
}

impl VableToken {
    /// Decode a raw u64 token value into the enum.
    pub fn from_raw(raw: u64) -> Self {
        match raw {
            0 => VableToken::None,
            other if other == token_tracing_rescall() => VableToken::TracingRescall,
            other => VableToken::Active(other),
        }
    }

    /// Encode the enum as a raw u64 value.
    pub fn to_raw(self) -> u64 {
        match self {
            VableToken::None => 0,
            VableToken::TracingRescall => token_tracing_rescall(),
            VableToken::Active(ptr) => ptr,
        }
    }
}

/// Describes a single field in a virtualizable object.
#[derive(Debug, Clone)]
pub struct VableFieldInfo {
    /// Name of the field (for debugging).
    pub name: String,
    /// Type of the field value.
    pub field_type: Type,
    /// Byte offset in the heap object.
    pub offset: usize,
    /// Whether this is an immutable field (can be constant-folded).
    pub is_immutable: bool,
}

/// Describes an array field in a virtualizable object.
///
/// In RPython, virtualizable arrays are separate from static fields.
/// For example, PyPy's frame has `locals_w` as a virtualizable array.
#[derive(Debug, Clone)]
pub struct VableArrayInfo {
    /// Name of the array field (for debugging).
    pub name: String,
    /// Type of array items.
    pub item_type: Type,
    /// Byte size of a single item, taken from the field's array descriptor
    /// (`unpack_arraydescr`).  This is the authoritative stride: 64-bit
    /// payloads (e.g. `i64` list-strategy backing arrays) carry an explicit
    /// 8-byte descriptor that `item_size_for_type` would under-size to a
    /// machine word on 32-bit targets.  It is also the authoritative LOAD
    /// WIDTH: `bh_getarrayitem_gc_i` dispatches on `item_size` × sign
    /// (`majit-backend` `model.rs`), so a narrower item must not be read as a
    /// full word.
    pub item_size: usize,
    /// `arraydescr.is_item_signed()` — the other half of that dispatch.
    /// Meaningless for `Ref` and `Float` items, which have one representation.
    pub item_signed: bool,
    /// Byte offset of the array pointer in the heap object.
    pub field_offset: usize,
    /// GC type id of the array object stored in a `DirectPointer` field.
    pub array_type_id: u32,
    /// Storage model for the array field.
    pub storage: VableArrayStorage,
    /// Offset of the length field within the array object.
    /// For `DirectPointer`, this is relative to the pointee.
    /// For `EmbeddedArray`, this is relative to the embedded container.
    pub length_offset: usize,
    /// Offset of the first item within the array object.
    /// For `DirectPointer`, this is relative to the pointee.
    /// For `EmbeddedArray`, this is relative to the active data pointer.
    pub items_offset: usize,
}

/// Physical storage strategy for a virtualizable array field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableArrayStorage {
    /// The frame field stores a raw pointer directly to array items.
    DirectPointer,
    /// The frame field stores a pointer to an array container struct
    /// (e.g. `*mut PyObjectArray`). The data pointer is at `ptr_offset`
    /// within that container (2-level indirection).
    EmbeddedArray { ptr_offset: usize },
    /// The frame field embeds a Rust `Vec<T>` by value. The `Vec`'s
    /// `(ptr, cap, len)` triple has no guaranteed field order, so the data
    /// pointer and length are read through type-aware extractor functions
    /// monomorphized for the concrete frame type rather than via byte
    /// offsets. `field_offset` is unused for reads (the extractors locate
    /// the `Vec` themselves) but is retained for flat-index bookkeeping.
    RustVec {
        /// Returns the live data pointer of the embedded `Vec<i64>`.
        data_ptr_fn: fn(*mut u8) -> *mut i64,
        /// Returns the live length of the embedded `Vec<i64>`.
        len_fn: fn(*const u8) -> usize,
    },
}

/// Complete description of a virtualizable type.
///
/// Mirrors RPython's `VirtualizableInfo` class from `virtualizable.py`.
///
/// This tells the JIT how to read/write the virtualizable's fields
/// from both heap and JIT representations.
#[derive(Debug)]
pub struct VirtualizableInfo {
    /// jitdriver_sd.virtualizable name (interp_jit.py:25).
    pub name: String,
    /// Static (scalar) fields on the virtualizable.
    pub static_fields: Vec<VableFieldInfo>,
    /// Array fields on the virtualizable (e.g., `locals_w`).
    pub array_fields: Vec<VableArrayInfo>,
    /// Offset of the `vable_token` field in the heap object. Meaningless
    /// unless `has_vable_token` — see [`VirtualizableInfo::has_vable_token`].
    pub token_offset: usize,
    /// Whether the host object actually carries a `vable_token` field.
    ///
    /// `virtualizable.py:28` reads `vable_token` off every VTYPE, so
    /// upstream has no false case. Pyre's state-field machines do: the
    /// `#[jit_interp]` `state` struct is a non-GC, stack-resident struct
    /// the interpreter author declares, with no slot to spare for a token
    /// — its identity is recovered from the resume snapshot instead. Every
    /// token read/write must stay inert for those, or it lands on the
    /// struct's first live field.
    has_vable_token: bool,
    /// Total number of "boxes" (field slots + array element slots)
    /// that the JIT needs to save/restore for this virtualizable.
    ///
    /// For `n` static fields and arrays of sizes `s1, s2, ...`:
    /// `num_boxes = n + s1 + s2 + ...`
    /// (array sizes are known at trace time after a promote).
    pub num_static_extra_boxes: usize,
    /// `descr.py FieldDescr.parent_descr` for every field descriptor
    /// constructed via `static_field_descr` / `token_field_descr` /
    /// `array_pointer_field_descr`. Set by the host runtime via
    /// `set_parent_descr` so `OptContext::ensure_ptr_info_arg0` can
    /// dispatch the GETFIELD/SETFIELD branch on `parent_descr.is_object()`
    /// (`optimizer.py:478-484`). When `None` the descriptor methods fall
    /// back to bare layout — only safe for code paths that bypass
    /// `ensure_ptr_info_arg0`.
    pub parent_descr: Option<DescrRef>,
    /// virtualizable.py:58: self.array_descrs = [cpu.arraydescrof(...)]
    /// Populated at init from array_fields, one per array field.
    pub array_descrs: Vec<DescrRef>,
    /// virtualizable.py:28: self.vable_token_descr = cpu.fielddescrof(VTYPE, 'vable_token')
    /// Built by set_parent_descr(); None until then.
    pub vable_token_descr: Option<DescrRef>,
    /// virtualizable.py:71-72: self.static_field_descrs = [cpu.fielddescrof(VTYPE, name) ...]
    /// Built by set_parent_descr(); empty until then.
    _static_field_descrs: Vec<DescrRef>,
    /// virtualizable.py:73-74: self.array_field_descrs = [cpu.fielddescrof(VTYPE, name) ...]
    /// Built by set_parent_descr(); empty until then.
    _array_field_descrs: Vec<DescrRef>,
    /// Parent-struct-layout counterparts of `_static_field_descrs`, keyed by
    /// the same field index. Each carries the `index_in_parent` (and canonical
    /// identity) the parent `SizeDescr` assigns by struct declaration order,
    /// looked up by byte offset, rather than the vinfo's own sequential
    /// `[token, statics, arrays]` ordering. These two numberings diverge for a
    /// struct whose declaration order differs from `[token, statics, arrays]`
    /// (PyFrame). A vable field op against a force-materialized inline-callee
    /// VIRTUAL frame must record the struct-layout descr so the optimizer pairs
    /// the read/write with the frame's `NewWithVtable` construction (which uses
    /// the parent struct descrs) by matching `index_in_parent`. Falls back to
    /// the vinfo descr when the parent has no field at that offset.
    _static_field_struct_descrs: Vec<DescrRef>,
    /// Parent-struct-layout counterparts of `_array_field_descrs` (the
    /// array-pointer fields). Like `_static_field_struct_descrs`, these carry
    /// the parent `SizeDescr`'s `index_in_parent` so a vable op against a
    /// force-materialized VIRTUAL frame pairs with its `NewWithVtable`.
    _array_field_struct_descrs: Vec<DescrRef>,
    /// virtualizable.py:81-82: self.static_field_by_descrs = {descr: i ...}
    /// Map from descriptor identity (Arc pointer address) to field index.
    pub static_field_by_descrs: indexmap::IndexMap<usize, usize>,
    /// virtualizable.py:83-84: self.array_field_by_descrs = {descr: i ...}
    /// Map from descriptor identity (Arc pointer address) to array field index.
    pub array_field_by_descrs: indexmap::IndexMap<usize, usize>,
    /// virtualizable.py `clear_vable_ptr`: function pointer to
    /// `clear_vable_token`, callable from JIT-compiled COND_CALL.
    /// Signature: `extern "C" fn(*mut u8)`. Stored as raw address so
    /// `VirtualizableInfo` stays Send+Sync.
    pub clear_vable_ptr: Option<usize>,
    /// virtualizable.py `clear_vable_descr`: CallDescr for the
    /// COND_CALL that invokes `clear_vable_ptr`.
    pub clear_vable_descr: Option<DescrRef>,
    /// Ref-register-bank index of the virtualizable identity inputarg, when
    /// the host's lowering keeps a green ref ahead of it in the ref bank.
    ///
    /// `initialize_virtualizable` mints the standard box as
    /// `OpRef::input_arg_ref(index)`, where the orthodox `index =
    /// num_green_args + index_of_virtualizable` is a flat arg ordinal. With
    /// separate int/ref register banks the ordinal only equals the ref-bank
    /// index when every preceding ref arg is stripped (PyFrame strips its
    /// green refs, so the frame is ref-bank 0). The state-field JIT keeps the
    /// green ref it needs as an array base (`program` at ref reg 0), so its
    /// virtualizable identity is ref reg 1. When `Some`, this overrides the
    /// box's ref-bank index so it matches the lowering's `vable_input_ref_reg`.
    /// `None` (PyFrame and tests) leaves the flat formula unchanged.
    pub identity_ref_bank_index: Option<usize>,
    /// Flat position of the virtualizable identity inside the reds the host
    /// hands `initialize_virtualizable` as `live_values`.
    ///
    /// `pyjitpl.py virtualizable_box = original_boxes[index]` is a lookup
    /// at a DECLARED position — `warmspot.py:529-538` fixes
    /// `index_of_virtualizable` when the driver is registered, and nothing at
    /// snapshot time searches for the box. Hosts whose reds are not the
    /// jitdriver's `reds` list (the state-field JIT expands its state fields
    /// into the red vector) publish the identity's position here so pyre can
    /// take the same lookup instead of matching on the live pointer.
    ///
    /// `None` (PyFrame and tests) means the flat `num_green_args +
    /// index_of_virtualizable` formula already names the box.
    pub identity_live_index: Option<usize>,
}

impl Clone for VirtualizableInfo {
    fn clone(&self) -> Self {
        VirtualizableInfo {
            name: self.name.clone(),
            static_fields: self.static_fields.clone(),
            array_fields: self.array_fields.clone(),
            token_offset: self.token_offset,
            has_vable_token: self.has_vable_token,
            num_static_extra_boxes: self.num_static_extra_boxes,
            parent_descr: self.parent_descr.clone(),
            array_descrs: self.array_descrs.clone(),
            vable_token_descr: self.vable_token_descr.clone(),
            _static_field_descrs: self._static_field_descrs.clone(),
            _array_field_descrs: self._array_field_descrs.clone(),
            _static_field_struct_descrs: self._static_field_struct_descrs.clone(),
            _array_field_struct_descrs: self._array_field_struct_descrs.clone(),
            static_field_by_descrs: self.static_field_by_descrs.clone(),
            array_field_by_descrs: self.array_field_by_descrs.clone(),
            clear_vable_ptr: self.clear_vable_ptr,
            clear_vable_descr: self.clear_vable_descr.clone(),
            identity_ref_bank_index: self.identity_ref_bank_index,
            identity_live_index: self.identity_live_index,
        }
    }
}

impl majit_translate::call::VirtualizableInfoHandle for VirtualizableInfo {
    /// virtualizable.py `is_vtypeptr(TYPE) → TYPE == self.VTYPEPTR`.
    ///
    /// Pyre identifies VTYPEPTR by the SizeDescr identity stored in
    /// `parent_descr`; `vtypeptr_id` is the `descr_identity` of the
    /// SizeDescr the caller wants to match.
    fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool {
        match &self.parent_descr {
            Some(descr) => descr_identity(descr) == vtypeptr_id,
            None => false,
        }
    }
}

/// majit-ir's `FieldDescr::get_vinfo()` returns `Option<Arc<dyn VinfoMarker>>`;
/// pyre's owning vinfo is a `VirtualizableInfo`. Implementing the marker
/// here bridges the crate split (majit-ir → majit-metainterp).
impl majit_ir::descr::VinfoMarker for VirtualizableInfo {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl VirtualizableInfo {
    /// Create a new VirtualizableInfo.
    pub fn new(token_offset: usize) -> Self {
        Self::with_token(token_offset, true)
    }

    /// Create a `VirtualizableInfo` for a machine with no `vable_token`
    /// field — see [`VirtualizableInfo::has_vable_token`]. Its token
    /// protocol is inert: `tracing_before/after_residual_call`,
    /// `force_now` and the token read/write pair all no-op instead of
    /// landing on offset 0.
    pub fn without_vable_token() -> Self {
        Self::with_token(0, false)
    }

    fn with_token(token_offset: usize, has_vable_token: bool) -> Self {
        VirtualizableInfo {
            name: String::new(),
            static_fields: Vec::new(),
            array_fields: Vec::new(),
            token_offset,
            has_vable_token,
            num_static_extra_boxes: 0,
            parent_descr: None,
            array_descrs: Vec::new(),
            vable_token_descr: None,
            _static_field_descrs: Vec::new(),
            _array_field_descrs: Vec::new(),
            _static_field_struct_descrs: Vec::new(),
            _array_field_struct_descrs: Vec::new(),
            static_field_by_descrs: indexmap::IndexMap::new(),
            array_field_by_descrs: indexmap::IndexMap::new(),
            clear_vable_ptr: None,
            clear_vable_descr: None,
            identity_ref_bank_index: None,
            identity_live_index: None,
        }
    }

    /// Attach a `SizeDescr` backreference for every field descriptor
    /// produced by this `VirtualizableInfo`.
    ///
    /// `descr.py FieldDescr.parent_descr` is the SizeDescr of the host
    /// virtualizable struct (`PyFrame` for pyre). The optimizer's
    /// `ensure_ptr_info_arg0` (`optimizer.py`) reads this to
    /// pick `InstancePtrInfo` vs `StructPtrInfo` for GETFIELD/SETFIELD
    /// virtualizable field accesses. Hosts MUST call this once,
    /// immediately after `build_virtualizable_info()`, before the
    /// JIT pipeline starts emitting field-typed ops.
    pub fn set_parent_descr(&mut self, descr: DescrRef) {
        self.build_descriptors(&descr, None);
        self.parent_descr = Some(descr);
    }

    /// Consume self + wrap into `Arc<Self>` with every field descriptor
    /// carrying the `VinfoMarker` backreference that
    /// `FieldDescr::get_vinfo()` returns. Mirrors `set_parent_descr` +
    /// the RPython `descr._vinfo = self` stamping pattern
    /// (pyjitpl.py `vinfo = fielddescr.get_vinfo()`).
    ///
    /// Uses `Arc::new_cyclic` so the descriptors built during
    /// construction observe the final `Weak<Self>` — a single-pass
    /// build that avoids rewriting the descriptors after the Arc is
    /// formed.
    pub fn finalize_arc(mut self, descr: DescrRef) -> Arc<Self> {
        Arc::new_cyclic(|weak: &Weak<Self>| {
            let vinfo_weak: Weak<dyn majit_ir::descr::VinfoMarker> = weak.clone();
            self.build_descriptors(&descr, Some(vinfo_weak));
            self.parent_descr = Some(descr);
            self
        })
    }

    /// Shared descriptor-building core for `set_parent_descr` and
    /// `finalize_arc`. When `vinfo` is `Some`, every built field
    /// descriptor carries that weak backreference so
    /// `FieldDescr::get_vinfo()` resolves upstream-faithfully. When
    /// `None`, the descriptors fall through to the trait's default
    /// `None` (legacy by-value path).
    fn build_descriptors(
        &mut self,
        descr: &DescrRef,
        vinfo: Option<Weak<dyn majit_ir::descr::VinfoMarker>>,
    ) {
        // virtualizable.py:28: self.vable_token_descr = cpu.fielddescrof(VTYPE, 'vable_token')
        // descr.py:214-215 + heaptracker.py:97: index_in_parent counts
        // non-void, non-typeptr fields in struct declaration order.
        // Layout: [typeptr, vable_token(0), static_0(1), ..., array_ptr_0(n+1), ...]
        self.vable_token_descr = Some(Self::build_field_descr(
            descr,
            self.token_offset,
            item_size_for_type(Type::Ref),
            Type::Ref,
            majit_ir::ArrayFlag::Unsigned,
            0,
            vinfo.clone(),
        ));
        // virtualizable.py:71-72: self.static_field_descrs = [cpu.fielddescrof(VTYPE, name) ...]
        let num_static = self.static_fields.len();
        self._static_field_descrs = self
            .static_fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let flag = majit_ir::ArrayFlag::from_field_type(f.field_type);
                Self::build_field_descr(
                    descr,
                    f.offset,
                    item_size_for_type(f.field_type),
                    f.field_type,
                    flag,
                    1 + i,
                    vinfo.clone(),
                )
            })
            .collect();
        // virtualizable.py:73-74: self.array_field_descrs = [cpu.fielddescrof(VTYPE, name) ...]
        // The descriptor offset is the field_offset — the position of the
        // array pointer within the virtualizable struct.
        self._array_field_descrs = self
            .array_fields
            .iter()
            .enumerate()
            .map(|(j, a)| {
                let offset = a.field_offset;
                Self::build_field_descr(
                    descr,
                    offset,
                    std::mem::size_of::<usize>(),
                    Type::Ref,
                    majit_ir::ArrayFlag::Pointer,
                    1 + num_static + j,
                    vinfo.clone(),
                )
            })
            .collect();
        // Parent-struct-layout descrs, looked up by offset from the parent
        // SizeDescr's canonical `all_fielddescrs()`. Used only when a vable
        // field op targets a force-materialized inline-callee VIRTUAL frame:
        // its construction (`NewWithVtable` + `SetfieldGc`) uses the parent
        // struct descrs, and the optimizer pairs reads/writes on a virtual by
        // `index_in_parent`, so the vable op must record the struct descr (whose
        // `index_in_parent` follows struct declaration order) rather than the
        // vinfo descr (whose `index_in_parent` follows `[token, statics,
        // arrays]`). When the parent exposes no field at a vinfo field's offset
        // (e.g. the synthetic test VTYPE, or a layout without `all_fielddescrs`),
        // fall back to the vinfo descr — for those the two orderings coincide.
        let struct_descr_at = |offset: usize, fallback: &DescrRef| -> DescrRef {
            descr
                .as_size_descr()
                .and_then(|sd| {
                    sd.all_fielddescrs()
                        .iter()
                        .find(|fd| fd.offset() == offset)
                        .cloned()
                })
                .map(|fd| fd as DescrRef)
                .unwrap_or_else(|| fallback.clone())
        };
        self._static_field_struct_descrs = self
            .static_fields
            .iter()
            .enumerate()
            .map(|(i, f)| struct_descr_at(f.offset, &self._static_field_descrs[i]))
            .collect();
        self._array_field_struct_descrs = self
            .array_fields
            .iter()
            .enumerate()
            .map(|(j, a)| struct_descr_at(a.field_offset, &self._array_field_descrs[j]))
            .collect();
        // virtualizable.py:81-82: self.static_field_by_descrs = {descr: i ...}
        // RPython's dict[descr] is identity-keyed on the unique FieldDescr
        // object `cpu.fielddescrof(VTYPE, name)` returned to both vinfo
        // and the codewriter.  Pyre splits that role: vinfo carries
        // offset/size-bearing SimpleFieldDescr instances (used by
        // `compile.py patch_new_loop_to_load_virtualizable_fields`
        // to emit GETFIELD_GC at loop entry), while the codewriter emits
        // the `vable_static_field_descr(idx)` singleton in
        // `BhDescr::VableField` (`codewriter/assembler.rs`'s `encode_op`).  Both
        // refer to the same logical (vable, field), so register BOTH
        // arc identities under the same idx — `vable_getfield_int`'s
        // identity lookup then resolves whichever descr the walker
        // hands it.
        self.static_field_by_descrs = indexmap::IndexMap::new();
        for (i, d) in self._static_field_descrs.iter().enumerate() {
            self.static_field_by_descrs.insert(descr_identity(d), i);
            let canonical = majit_ir::descr::vable_static_field_descr(i as u16);
            self.static_field_by_descrs
                .insert(descr_identity(&canonical), i);
        }
        // virtualizable.py:83-84: self.array_field_by_descrs = {descr: i ...}
        self.array_field_by_descrs = indexmap::IndexMap::new();
        for (i, d) in self._array_field_descrs.iter().enumerate() {
            self.array_field_by_descrs.insert(descr_identity(d), i);
            let canonical = majit_ir::descr::vable_array_field_descr(i as u16);
            self.array_field_by_descrs
                .insert(descr_identity(&canonical), i);
        }
    }

    /// Build a FieldDescr carrying parent_descr (+ optional vinfo backref).
    /// Helper for `set_parent_descr` / `finalize_arc` descriptor caching.
    fn build_field_descr(
        parent: &DescrRef,
        offset: usize,
        field_size: usize,
        field_type: Type,
        flag: majit_ir::ArrayFlag,
        index_in_parent: usize,
        vinfo: Option<Weak<dyn majit_ir::descr::VinfoMarker>>,
    ) -> DescrRef {
        let mut descr = majit_ir::SimpleFieldDescr::new(0, offset, field_size, field_type, false)
            .with_flag(flag)
            .with_parent_descr(parent.clone(), index_in_parent);
        if let Some(w) = vinfo {
            descr = descr.with_vinfo(w);
        }
        Arc::new(descr)
    }

    /// virtualizable.py `finish()` registers the `clear_vable_ptr`
    /// function pointer and `clear_vable_descr` call descriptor.
    ///
    /// `clear_fn` clears the vable token, forcing the virtualizable if
    /// necessary. Its ABI is the one `make_clear_vable_descr` declares —
    /// `[Ref] -> Void`, taking the virtualizable as a single machine word —
    /// spelled `extern "C" fn(i64)`, because that is what the compiled
    /// `COND_CALL` `emit_force_virtualizable` builds passes and what
    /// `bh_clear_vable_token` calls it as.  Hosts must call this after
    /// `build_virtualizable_info()` to enable `emit_force_virtualizable`.
    pub fn set_clear_vable(&mut self, clear_fn: *const (), clear_descr: DescrRef) {
        self.clear_vable_ptr = Some(clear_fn as usize);
        self.clear_vable_descr = Some(clear_descr);
    }

    /// Register virtualizable array pointer fields for old virtualizable
    /// objects currently held in a blackhole ref register bank.  The type-id
    /// check keeps this targeted to the `VTYPEPTR` described by this vinfo,
    /// rather than treating every Ref register as if it had the virtualizable
    /// layout.
    pub fn push_resume_ref_roots_for_registers(&self, registers_r: &[i64]) {
        for &value in registers_r {
            self.push_resume_ref_roots_for_value(value);
        }
    }

    /// Register virtualizable array pointer fields for one old virtualizable
    /// object. Nursery virtualizable objects are already traced through the
    /// rooted Ref slot that points at the object; old virtualizables need their
    /// young array fields exposed explicitly.
    pub fn push_resume_ref_roots_for_value(&self, value: i64) {
        let Some(parent) = &self.parent_descr else {
            return;
        };
        let Some(size_descr) = parent.as_size_descr() else {
            return;
        };
        let vtype_id = size_descr.type_id();
        if value == 0 {
            return;
        }
        let ptr = value as usize;
        if !majit_gc::gc_owns_object(ptr) || majit_gc::gc_is_nursery_object(ptr) {
            return;
        }
        let type_id = unsafe { (*majit_gc::header::header_of(ptr)).type_id() };
        if type_id == vtype_id {
            crate::resume::VirtualizableInfo::push_resume_ref_roots(self, value);
        }
    }

    /// virtualizable.py `clear_vable_descr` factory. Creates
    /// the CallDescr for the `COND_CALL` that invokes `clear_vable_ptr`.
    ///
    /// ```text
    /// self.clear_vable_descr = staticdata.calldescr_for_call(
    ///     [llmemory.GCREF], lltype.Void,
    ///     EffectInfo.MOST_GENERAL,
    ///     oopspecindex=EffectInfo.OS_JIT_FORCE_VIRTUALIZABLE)
    /// ```
    pub fn make_clear_vable_descr() -> DescrRef {
        use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex};
        crate::call_descr::make_call_descr_with_effect(
            &[Type::Ref],
            Type::Void,
            // rpython/jit/metainterp/virtualizable.py:296 — the force helper
            // clears the vable_token in place; it cannot raise.
            EffectInfo::const_new(
                ExtraEffect::CannotRaise,
                OopSpecIndex::JitForceVirtualizable,
            ),
        )
    }

    /// Add a static field.
    ///
    /// virtualizable.py:61: num_static_extra_boxes = len(static_fields)
    /// ALL declared fields are included in snapshots.
    pub fn add_field(&mut self, name: impl Into<String>, field_type: Type, offset: usize) {
        self.static_fields.push(VableFieldInfo {
            name: name.into(),
            field_type,
            offset,
            is_immutable: false,
        });
        self.num_static_extra_boxes = self.static_fields.len();
    }

    /// virtualizable.py:58: self.array_descrs = [cpu.arraydescrof(...)]
    ///
    /// Add an array field with a pre-built descriptor.
    /// The descriptor must come from the GcCache (= cpu.arraydescrof)
    /// for production use. Test code may pass descriptors from
    /// `make_array_descr()`.
    pub fn add_array_field(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
        length_offset: usize,
        items_offset: usize,
        array_descr: DescrRef,
    ) {
        let name = name.into();
        let item_size = array_descr_item_size(&array_descr, item_type);
        let item_signed = array_descr_item_signed(&array_descr);
        let array_type_id = array_field_type_id(&name, &array_descr);
        self.array_fields.push(VableArrayInfo {
            name,
            item_type,
            item_size,
            item_signed,
            field_offset,
            array_type_id,
            storage: VableArrayStorage::DirectPointer,
            length_offset,
            items_offset,
        });
        self.array_descrs.push(array_descr);
    }

    /// Add an embedded array field with a pre-built descriptor.
    ///
    /// Embedded arrays live inside the virtualizable object by value
    /// (e.g. Rust Vec-like containers). The `ptr_offset` is relative
    /// to `field_offset` and locates the data pointer within the
    /// container.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn add_embedded_array_field(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
        ptr_offset: usize,
        length_offset: usize,
        items_offset: usize,
        array_descr: DescrRef,
    ) {
        let name = name.into();
        let item_size = array_descr_item_size(&array_descr, item_type);
        let item_signed = array_descr_item_signed(&array_descr);
        let array_type_id = array_field_type_id(&name, &array_descr);
        self.array_fields.push(VableArrayInfo {
            name,
            item_type,
            item_size,
            item_signed,
            field_offset,
            array_type_id,
            storage: VableArrayStorage::EmbeddedArray { ptr_offset },
            length_offset,
            items_offset,
        });
        self.array_descrs.push(array_descr);
    }

    /// Add a Rust `Vec<T>`-backed array field embedded by value in the frame.
    ///
    /// `field_offset` is the byte offset of the `Vec` within the frame (used
    /// only for flat-index bookkeeping). `data_ptr_fn`/`len_fn` read the live
    /// data pointer and length through `Vec` methods, so no assumption is
    /// made about the in-memory order of the `Vec`'s `(ptr, cap, len)` words.
    pub fn add_rust_vec_array_field(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
        data_ptr_fn: fn(*mut u8) -> *mut i64,
        len_fn: fn(*const u8) -> usize,
        array_descr: DescrRef,
    ) {
        let name = name.into();
        let item_size = array_descr_item_size(&array_descr, item_type);
        let item_signed = array_descr_item_signed(&array_descr);
        let array_type_id = array_field_type_id(&name, &array_descr);
        self.array_fields.push(VableArrayInfo {
            name,
            item_type,
            item_size,
            item_signed,
            field_offset,
            array_type_id,
            storage: VableArrayStorage::RustVec {
                data_ptr_fn,
                len_fn,
            },
            length_offset: 0,
            items_offset: 0,
        });
        self.array_descrs.push(array_descr);
    }

    /// Total number of static fields.
    pub fn num_fields(&self) -> usize {
        self.static_fields.len()
    }

    /// Total number of array fields.
    pub fn num_arrays(&self) -> usize {
        self.array_fields.len()
    }

    /// Whether this machine has a real `vable_token` field to operate on.
    /// False only for [`VirtualizableInfo::without_vable_token`] machines.
    pub fn has_vable_token(&self) -> bool {
        self.has_vable_token
    }

    /// Set token to TOKEN_TRACING_RESCALL before a residual call.
    ///
    /// The token tells the runtime that JIT tracing is active and a
    /// residual call is about to happen. If the callee touches the
    /// virtualizable, it will force the token and clear it.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn tracing_before_residual_call(&self, obj_ptr: *mut u8) {
        if !self.has_vable_token() {
            return;
        }
        unsafe {
            // The token is a word-sized (`Signed`) field: 4 bytes on wasm32.
            // The all-ones sentinel truncates to `usize::MAX` at that width.
            let token_ptr = obj_ptr.add(self.token_offset) as *mut usize;
            assert_eq!(*token_ptr, 0, "token should be NONE before residual call");
            *token_ptr = token_tracing_rescall() as usize;
        }
    }

    /// Check after residual call whether the virtualizable was forced.
    ///
    /// Returns `true` if forced (token was cleared by the callee).
    /// Returns `false` if not forced (token is still TRACING_RESCALL; clear it).
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn tracing_after_residual_call(&self, obj_ptr: *mut u8) -> bool {
        // No token to consult: the callee had no way to force this machine's
        // state, so it is never reported as escaped.
        if !self.has_vable_token() {
            return false;
        }
        unsafe {
            let token_ptr = obj_ptr.add(self.token_offset) as *mut usize;
            if *token_ptr != 0 {
                // Not forced — still TOKEN_TRACING_RESCALL
                assert_eq!(*token_ptr, token_tracing_rescall() as usize);
                *token_ptr = 0; // Clear back to TOKEN_NONE
                false
            } else {
                // Was forced — token was cleared by the force path
                true
            }
        }
    }

    /// Force the virtualizable now.
    ///
    /// If TOKEN_TRACING_RESCALL, just clear (tracing can reconstruct state).
    /// If active JIT frame pointer, call `force_fn` to flush JIT state to heap.
    /// If TOKEN_NONE, no-op.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn force_now(&self, obj_ptr: *mut u8, force_fn: impl FnOnce(u64)) {
        if !self.has_vable_token() {
            return;
        }
        unsafe {
            let token_ptr = obj_ptr.add(self.token_offset) as *mut usize;
            let token = *token_ptr;
            if token == token_tracing_rescall() as usize {
                // During tracing — just clear the marker
                *token_ptr = 0;
            } else if token != 0 {
                // Active JIT frame — force it, then verify it cleared the token
                force_fn(token as u64);
                assert_eq!(*token_ptr, 0, "force_fn should have cleared the token");
            }
        }
    }

    /// Read the current token state from the heap.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_token(&self, obj_ptr: *const u8) -> VableToken {
        if !self.has_vable_token() {
            return VableToken::None;
        }
        unsafe {
            let token_ptr = obj_ptr.add(self.token_offset) as *const usize;
            let raw = *token_ptr;
            VableToken::from_raw(raw as u64)
        }
    }

    /// Write a token state to the heap.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_token(&self, obj_ptr: *mut u8, token: VableToken) {
        if !self.has_vable_token() {
            return;
        }
        unsafe {
            let token_ptr = obj_ptr.add(self.token_offset) as *mut usize;
            *token_ptr = token.to_raw() as usize;
            if matches!(token, VableToken::Active(_)) {
                // Host-side active-token stores have the same generational
                // obligation as compiled SETFIELD_GC stores. Unmanaged test
                // objects are ignored by the barrier hook.
                majit_gc::gc_write_barrier(majit_ir::GcRef(obj_ptr as usize));
            }
        }
    }

    /// virtualizable.py:71-72: self.static_field_descrs
    /// Returns all cached static field DescrRefs.
    pub fn static_field_descrs(&self) -> &[DescrRef] {
        &self._static_field_descrs
    }

    /// virtualizable.py:73-74: self.array_field_descrs
    /// Returns all cached array pointer field DescrRefs.
    pub fn array_field_descrs(&self) -> &[DescrRef] {
        &self._array_field_descrs
    }

    /// virtualizable.py:81: `vinfo.static_field_by_descrs[fielddescr]`
    /// Descriptor-identity lookup (linear scan via IndexMap).
    pub fn static_field_by_descr(&self, descr: &DescrRef) -> Option<usize> {
        self.static_field_by_descrs
            .get(&descr_identity(descr))
            .copied()
    }

    /// virtualizable.py:83: `vinfo.array_field_by_descrs[arrayfielddescr]`
    /// Descriptor-identity lookup (linear scan via IndexMap).
    pub fn array_field_by_descr(&self, descr: &DescrRef) -> Option<usize> {
        self.array_field_by_descrs
            .get(&descr_identity(descr))
            .copied()
    }

    /// RPython parity surface: reset the virtualizable token to TOKEN_NONE.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn reset_vable_token(&self, obj_ptr: *mut u8) {
        unsafe {
            self.write_token(obj_ptr, VableToken::None);
        }
    }

    /// RPython parity surface: reset token from a GCREF/object pointer path.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn reset_token_gcref(&self, obj_ptr: *mut u8) {
        unsafe {
            self.reset_vable_token(obj_ptr);
        }
    }

    /// RPython parity surface: force only if a token is still attached.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn force_virtualizable_if_necessary(
        &self,
        obj_ptr: *mut u8,
        force_fn: impl FnOnce(u64),
    ) {
        unsafe {
            if !matches!(self.read_token(obj_ptr), VableToken::None) {
                self.force_now(obj_ptr, force_fn);
            }
        }
    }

    /// RPython parity surface: clear the virtualizable token, forcing first
    /// if JIT state is still attached.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn clear_vable_token(&self, obj_ptr: *mut u8, force_fn: impl FnOnce(u64)) {
        unsafe {
            self.force_virtualizable_if_necessary(obj_ptr, force_fn);
            assert!(
                matches!(self.read_token(obj_ptr), VableToken::None),
                "clear_vable_token must leave TOKEN_NONE"
            );
        }
    }

    /// Convert to optimizer-level config (byte offsets only).
    /// Bridges the descriptor-driven model (majit-meta) with the
    /// optimizer's offset-based tracking (majit-opt).
    ///
    /// `vable_input_offset` defaults to 0 here; callers that wire JitDriver
    /// non-vable reds (e.g. `interp_jit.py reds = ['frame', 'ec']`)
    /// should patch the field after construction — see
    /// `MetaInterp::current_virtualizable_optimizer_config`.
    pub(crate) fn to_optimizer_config(
        &self,
    ) -> crate::optimizeopt::virtualize::VirtualizableConfig {
        crate::optimizeopt::virtualize::VirtualizableConfig {
            static_field_offsets: self.static_fields.iter().map(|f| f.offset).collect(),
            static_field_types: self.static_fields.iter().map(|f| f.field_type).collect(),
            static_field_descrs: self.static_field_descrs().to_vec(),
            array_field_offsets: self.array_fields.iter().map(|a| a.field_offset).collect(),
            array_item_types: self.array_fields.iter().map(|a| a.item_type).collect(),
            array_field_descrs: self.array_field_descrs().to_vec(),
            vable_input_offset: 0,
            // Same declaration the resume path reads
            // (`MetaInterp::identity_live_position`): the loop's inputargs are
            // its reds, so the identity's position among the reds IS its flat
            // input-arg slot.
            //
            // `identity_live_index == None` is overloaded, so the layout
            // decides what it means. `identity_ref_bank_index` is the
            // structural discriminator:
            //
            // - `None` — the legacy frame-first (PyFrame) layout, whose reds are
            //   `[frame, extra_reds.., vable_scalars.., array_items..]`. The
            //   frame IS flat slot 0, so slot 0 is the answer, not a fallback.
            // - `Some(_)` — the banked-identity (macro state-field) layout,
            //   whose reds are `[int scalars.., fixed-array cells.., identity]`.
            //   Slot 0 there is an int scalar. Only a declaration names the
            //   identity, and the macro can emit one only when the state has no
            //   fixed array (`codegen_state.rs` `identity_live_index_stmt`) —
            //   with one present the position depends on that array's runtime
            //   length. Nothing supplies it later — this function is the only
            //   writer of `identity_input_index` — so such a layout stays
            //   `None` for the whole trace and `VirtualizableTracker` declines
            //   to track it. Declining costs optimization; probing slot 0
            //   costs correctness — it installs `PtrInfo::Virtualizable` on an
            //   Int scalar and the loop-close Jump then fails to match the
            //   Ref-typed preview (`VirtualStatesCantMatch`), so nothing
            //   compiles at all.
            identity_input_index: match self.identity_ref_bank_index {
                None => Some(0),
                Some(_) => self.identity_live_index,
            },
        }
    }

    /// Get the index of a static field by byte offset (fallback path).
    /// Prefer `static_field_by_descr()` for RPython parity.
    pub fn static_field_index_by_offset(&self, offset: usize) -> Option<usize> {
        self.static_fields.iter().position(|f| f.offset == offset)
    }

    /// Get the index of a static field by name.
    pub fn static_field_index_by_name(&self, name: &str) -> Option<usize> {
        self.static_fields.iter().position(|f| f.name == name)
    }

    /// Get the index of an array field by byte offset (fallback path).
    /// Prefer `array_field_by_descr()` for RPython parity.
    pub fn array_field_index_by_offset(&self, field_offset: usize) -> Option<usize> {
        self.array_fields
            .iter()
            .position(|a| a.field_offset == field_offset)
    }

    /// Get the index of an array field by name.
    pub fn array_field_index_by_name(&self, name: &str) -> Option<usize> {
        self.array_fields.iter().position(|a| a.name == name)
    }

    /// virtualizable.py:71: `self.static_field_descrs[field_index]`
    /// Returns the cached FieldDescr for a static field.
    /// Descriptors are built once in set_parent_descr(), not per-call.
    pub fn static_field_descr(&self, field_index: usize) -> DescrRef {
        self._static_field_descrs[field_index].clone()
    }

    /// Parent-struct-layout counterpart of `static_field_descr`. Returns the
    /// descr carrying the struct-declaration-order `index_in_parent`, for
    /// recording a vable field op against a force-materialized inline-callee
    /// virtual frame (see `_static_field_struct_descrs`).
    pub fn static_field_struct_descr(&self, field_index: usize) -> DescrRef {
        self._static_field_struct_descrs[field_index].clone()
    }

    /// virtualizable.py:28: self.vable_token_descr
    /// Returns the cached FieldDescr for the vable_token field.
    pub fn token_field_descr(&self) -> DescrRef {
        self.vable_token_descr
            .clone()
            .expect("token_field_descr called before set_parent_descr")
    }

    /// virtualizable.py:73: `self.array_field_descrs[array_index]`
    /// Returns the cached FieldDescr for an array pointer field.
    pub fn array_pointer_field_descr(&self, array_index: usize) -> DescrRef {
        self._array_field_descrs[array_index].clone()
    }

    /// Parent-struct-layout counterpart of `array_pointer_field_descr`. Returns
    /// the array-pointer descr carrying the struct-declaration-order
    /// `index_in_parent`, for reading the array base off a force-materialized
    /// inline-callee virtual frame (see `_array_field_struct_descrs`).
    pub fn array_pointer_struct_descr(&self, array_index: usize) -> DescrRef {
        self._array_field_struct_descrs[array_index].clone()
    }

    /// virtualizable.py:58: `self.array_descrs[array_index]`
    /// Returns the pre-built array descriptor for the given array field.
    pub fn array_item_descr(&self, array_index: usize) -> DescrRef {
        self.array_descrs[array_index].clone()
    }

    /// Minimum number of boxes needed (static fields only, no arrays).
    ///
    /// RPython equivalent: `vinfo.num_static_extra_boxes`
    pub fn minimum_size(&self) -> usize {
        self.num_static_extra_boxes
    }

    /// Get total size: number of static fields + sum of all array lengths.
    /// `array_lengths` must have one entry per array field.
    pub fn get_total_size(&self, array_lengths: &[usize]) -> usize {
        self.num_static_extra_boxes + array_lengths.iter().sum::<usize>()
    }

    /// Whether all array lengths can be derived from the heap object alone.
    pub fn can_read_all_array_lengths_from_heap(&self) -> bool {
        self.array_fields
            .iter()
            .all(VableArrayInfo::can_read_length_from_heap)
    }

    /// Get the index into the flat box array for a specific array element.
    /// `array_index` is the index of the array field, `item_index` is the
    /// element within that array.
    pub fn get_index_in_array(
        &self,
        array_index: usize,
        item_index: usize,
        array_lengths: &[usize],
    ) -> usize {
        let mut idx = self.num_static_extra_boxes;
        idx += array_lengths.iter().take(array_index).sum::<usize>();
        idx + item_index
    }

    /// Check that box array has correct size for given array lengths.
    /// The trailing `assert len(boxes) == i + 1` of `virtualizable.py
    /// check_boxes`, and ONLY that.  Upstream's `check_boxes` also compares
    /// every static field and every array item against the boxes; the port of
    /// that comparison is `trace_ctx.rs check_synchronized_virtualizable`.
    /// The name says which half this is, because a caller reaching for
    /// `check_boxes` at a `check_boxes` call site wants the other one.
    pub fn boxes_len_matches(&self, boxes: &[i64], array_lengths: &[usize]) -> bool {
        boxes.len() == self.get_total_size(array_lengths)
    }

    /// virtualizable.py:58: self.array_descrs = [cpu.arraydescrof(...)]
    ///
    /// Replace macro-generated array descriptors with GcCache-backed ones.
    /// The host runtime calls this after constructing VirtualizableInfo
    /// to provide descriptors from `gc_cache.get_array_descr(...)`,
    /// matching RPython's `cpu.arraydescrof(getattr(VTYPE, name).TO)`.
    ///
    /// `descrs` must have exactly one entry per array field, in order.
    pub fn replace_array_descrs(&mut self, descrs: Vec<DescrRef>) {
        assert_eq!(
            descrs.len(),
            self.array_fields.len(),
            "replace_array_descrs: expected {} descriptors, got {}",
            self.array_fields.len(),
            descrs.len()
        );
        // A compiled array access reads its stride and signedness off these
        // descriptors, while the blackhole reads and writes go through the
        // cached `VableArrayInfo::item_size` / `item_signed`
        // (`bh_getarrayitem_gc_i` dispatches on the pair).  A replacement that
        // disagrees would leave the two halves loading a different width from
        // a different address for the same array, so require the replacement
        // to preserve what the cache already committed to.  `item_signed`
        // selects a load only for `Int` items; `Ref` and `Float` have one
        // representation each.
        for (field, descr) in self.array_fields.iter().zip(descrs.iter()) {
            let arr = descr.as_array_descr().unwrap_or_else(|| {
                panic!(
                    "replace_array_descrs: {} was given a descriptor that is                      not an ArrayDescr",
                    field.name
                )
            });
            assert_eq!(
                arr.item_size(),
                field.item_size,
                "replace_array_descrs: {} replacement stride does not match                  the cached item_size",
                field.name,
            );
            if field.item_type == Type::Int {
                assert_eq!(
                    arr.is_item_signed(),
                    field.item_signed,
                    "replace_array_descrs: {} replacement signedness does not                      match the cached item_signed",
                    field.name,
                );
            }
        }
        self.array_descrs = descrs;
    }

    // ── RPython virtualizable.py parity: heap I/O via descriptor ──

    /// Read a static field value from the heap object.
    ///
    /// RPython equivalent: `vinfo.read_from_field(virtualizable, field_index)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_field(&self, obj_ptr: *const u8, field_index: usize) -> i64 {
        unsafe {
            let field = &self.static_fields[field_index];
            match field.field_type {
                Type::Float => {
                    let ptr = obj_ptr.add(field.offset) as *const f64;
                    f64::to_bits(*ptr) as i64
                }
                // Pointer-width field: 4 bytes on wasm32. Reading 8 bytes
                // would fold in the next field's bytes.
                Type::Ref => {
                    let ptr = obj_ptr.add(field.offset) as *const usize;
                    *ptr as i64
                }
                // Word-sized `Signed` field (`isize`/`usize`): 4 bytes on
                // wasm32. Read at word width and sign-extend so a negative
                // `last_instr` round-trips.
                _ => {
                    let ptr = obj_ptr.add(field.offset) as *const isize;
                    *ptr as i64
                }
            }
        }
    }

    /// Write a static field value to the heap object.
    ///
    /// RPython equivalent: `vinfo.write_to_field(virtualizable, field_index, value)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_field(&self, obj_ptr: *mut u8, field_index: usize, value: i64) {
        unsafe {
            let field = &self.static_fields[field_index];
            match field.field_type {
                Type::Float => {
                    let ptr = obj_ptr.add(field.offset) as *mut f64;
                    *ptr = f64::from_bits(value as u64);
                }
                // Pointer-width field: 4 bytes on wasm32. Writing 8 bytes
                // would clobber the adjacent field.
                Type::Ref => {
                    let ptr = obj_ptr.add(field.offset) as *mut usize;
                    *ptr = value as usize;
                    // The ref may be nursery-young while the virtualizable is
                    // old-gen and runs detached from the walked frame chain;
                    // upstream's write_boxes stores run under the translated
                    // write barrier (virtualizable.py:101-113), so arm the
                    // object in the remembered set here.
                    if majit_gc::gc_owns_object(obj_ptr as usize) {
                        majit_gc::gc_write_barrier(majit_ir::GcRef(obj_ptr as usize));
                    }
                }
                // Word-sized `Signed` field (`isize`/`usize`): 4 bytes on
                // wasm32. Writing 8 bytes would clobber the adjacent field.
                _ => {
                    let ptr = obj_ptr.add(field.offset) as *mut isize;
                    *ptr = value as isize;
                }
            }
        }
    }

    /// Read the length of an array field from the heap object.
    ///
    /// RPython equivalent: `vinfo.get_array_length(virtualizable, array_index)`
    ///
    /// Reads the array pointer from the virtualizable, then reads the length
    /// from the array header at `length_offset`.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn get_array_length(&self, obj_ptr: *const u8, array_index: usize) -> usize {
        unsafe {
            let ai = &self.array_fields[array_index];
            match ai.storage {
                VableArrayStorage::DirectPointer => {
                    let array_ptr = *(obj_ptr.add(ai.field_offset) as *const *const u8);
                    if array_ptr.is_null() {
                        0
                    } else {
                        *(array_ptr.add(ai.length_offset) as *const usize)
                    }
                }
                VableArrayStorage::EmbeddedArray { .. } => {
                    let container = *(obj_ptr.add(ai.field_offset) as *const *const u8);
                    *(container.add(ai.length_offset) as *const usize)
                }
                VableArrayStorage::RustVec { len_fn, .. } => len_fn(obj_ptr),
            }
        }
    }

    /// Read an array element from the heap object.
    ///
    /// RPython equivalent: `vinfo.read_from_array(virtualizable, array_index, item_index)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_array_item(
        &self,
        obj_ptr: *const u8,
        array_index: usize,
        item_index: usize,
    ) -> i64 {
        unsafe {
            vable_read_array_item(obj_ptr, &self.array_fields[array_index], item_index as i64)
        }
    }

    /// Write an array element to the heap object.
    ///
    /// RPython equivalent: `vinfo.write_to_array(virtualizable, array_index, item_index, value)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_array_item(
        &self,
        obj_ptr: *mut u8,
        array_index: usize,
        item_index: usize,
        value: i64,
    ) {
        unsafe {
            vable_write_array_item(
                obj_ptr,
                &self.array_fields[array_index],
                item_index as i64,
                value,
            )
        }
    }

    /// Load all virtualizable boxes from the heap object.
    ///
    /// RPython equivalent: `vinfo.load_list_of_boxes(virtualizable)`
    ///
    /// Returns a flat array: `[field0, field1, ..., array0[0], ..., array0[N], ...]`
    /// Array lengths are read from the actual object (not from a side-channel).
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn load_list_of_boxes(&self, obj_ptr: *const u8) -> (Vec<i64>, Vec<usize>) {
        unsafe {
            let mut boxes = Vec::new();
            let mut array_lengths = Vec::new();

            // Static fields
            for i in 0..self.static_fields.len() {
                boxes.push(self.read_field(obj_ptr, i));
            }

            // Array fields — read lengths from actual object
            for ai in 0..self.array_fields.len() {
                let len = self.get_array_length(obj_ptr, ai);
                array_lengths.push(len);
                for ei in 0..len {
                    boxes.push(self.read_array_item(obj_ptr, ai, ei));
                }
            }

            (boxes, array_lengths)
        }
    }

    /// Read only the array lengths from the heap object.
    ///
    /// RPython equivalent: `vinfo.get_array_length(vable, i)` for every array.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_array_lengths_from_heap(&self, obj_ptr: *const u8) -> Vec<usize> {
        unsafe {
            self.array_fields
                .iter()
                .enumerate()
                .map(|(index, _)| self.get_array_length(obj_ptr, index))
                .collect()
        }
    }

    /// RPython parity surface: read static boxes only.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    /// The static half of `virtualizable.py read_boxes`.  Upstream reads the
    /// statics and then every array item in one pass; [`Self::read_all_boxes`]
    /// is the port of the whole.
    pub unsafe fn read_static_boxes(&self, obj_ptr: *const u8) -> Vec<i64> {
        unsafe {
            self.static_fields
                .iter()
                .enumerate()
                .map(|(index, _)| self.read_field(obj_ptr, index))
                .collect()
        }
    }

    /// RPython parity surface: write static boxes only.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    /// The static half of `virtualizable.py write_boxes`; the port of the whole
    /// is [`Self::write_boxes_to_heap`].  Upstream closes with
    /// `assert len(boxes) == i + 1`; this one stops at the end of the static
    /// fields instead, because its callers hand it exactly that prefix.
    pub unsafe fn write_static_boxes(&self, obj_ptr: *mut u8, boxes: &[i64]) {
        unsafe {
            for (index, &value) in boxes.iter().enumerate() {
                if index >= self.static_fields.len() {
                    break;
                }
                self.write_field(obj_ptr, index, value);
            }
        }
    }

    /// Read static boxes and array boxes from the heap object.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_all_boxes(
        &self,
        obj_ptr: *const u8,
        array_lengths: &[usize],
    ) -> (Vec<i64>, Vec<Vec<i64>>) {
        unsafe {
            let static_boxes = self.read_static_boxes(obj_ptr);
            let mut array_boxes = Vec::with_capacity(self.array_fields.len());
            for (index, _) in self.array_fields.iter().enumerate() {
                let length = array_lengths.get(index).copied().unwrap_or(0);
                let mut values = Vec::with_capacity(length);
                for item_index in 0..length {
                    values.push(self.read_array_item(obj_ptr, index, item_index));
                }
                array_boxes.push(values);
            }
            (static_boxes, array_boxes)
        }
    }

    /// Write static boxes and array boxes back to the heap object.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_all_boxes(
        &self,
        obj_ptr: *mut u8,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) {
        unsafe {
            self.write_static_boxes(obj_ptr, static_boxes);
            for (array_index, values) in array_boxes.iter().enumerate() {
                for (item_index, &value) in values.iter().enumerate() {
                    self.write_array_item(obj_ptr, array_index, item_index, value);
                }
            }
        }
    }

    /// Write all boxes back to the heap object (force direction).
    ///
    /// virtualizable.py write_boxes(virtualizable, boxes).
    ///
    /// Writes static fields then every array item (using the heap array's
    /// actual length, NOT a caller-provided length). Asserts that exactly
    /// `len(boxes) == i + 1` items were consumed (the +1 is the vable_box
    /// that the caller appends but write_boxes does not write).
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_boxes_to_heap(&self, obj_ptr: *mut u8, boxes: &[i64]) {
        unsafe {
            let mut i = 0;
            // Static fields
            for fi in 0..self.static_fields.len() {
                self.write_field(obj_ptr, fi, boxes[i]);
                i += 1;
            }
            // Array elements — read actual length from heap (RPython: len(lst))
            for ai in 0..self.array_fields.len() {
                let len = self.get_array_length(obj_ptr, ai);
                for ei in 0..len {
                    self.write_array_item(obj_ptr, ai, ei, boxes[i]);
                    i += 1;
                }
            }
            // virtualizable.py:113: assert len(boxes) == i + 1
            assert_eq!(
                boxes.len(),
                i + 1,
                "write_boxes_to_heap: boxes count mismatch (expected {}, got {})",
                i + 1,
                boxes.len()
            );
        }
    }

    /// Force virtualizable: write boxes to heap and clear token.
    ///
    /// RPython equivalent: the combined force_now + write_from_resume_data flow.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn force_from_boxes(&self, obj_ptr: *mut u8, boxes: &[i64]) {
        unsafe {
            self.write_boxes_to_heap(obj_ptr, boxes);
            self.reset_vable_token(obj_ptr);
        }
    }

    /// Whether every array field can be rebuilt by the compiled entry's
    /// field-load preamble — see [`VableArrayInfo::is_entry_reloadable`].
    ///
    /// `compile.py:508-511` runs that preamble for any driver whose
    /// `virtualizable_info` is set, with no per-array escape, so the decision
    /// is all-or-nothing per virtualizable: one array the preamble cannot
    /// reload disqualifies the whole entry contract, not just that array.
    pub fn arrays_are_entry_reloadable(&self) -> bool {
        self.array_fields
            .iter()
            .all(VableArrayInfo::is_entry_reloadable)
    }
}

impl VableArrayInfo {
    /// Whether the compiled entry's field-load preamble can rebuild this
    /// array's elements from the virtualizable pointer alone.
    ///
    /// `compile.py` reaches every element with `GETFIELD_GC_R` for the
    /// array's data pointer followed by one `GETARRAYITEM_GC_*` per element —
    /// two loads expressible in trace IR with nothing but byte offsets. A
    /// storage whose data pointer sits at a fixed offset (from the field, or
    /// from a container the field points at) answers that; a `Vec` embedded by
    /// value does not, because its data pointer is one of three words in an
    /// order the language does not specify, so no field load portably finds it.
    /// `patch_new_loop_to_load_virtualizable_fields` panics rather than read a
    /// capacity as a base address, which makes this the predicate a caller must
    /// consult BEFORE putting a virtualizable on the preamble's path.
    pub fn is_entry_reloadable(&self) -> bool {
        match self.storage {
            VableArrayStorage::DirectPointer | VableArrayStorage::EmbeddedArray { .. } => true,
            VableArrayStorage::RustVec { .. } => false,
        }
    }

    pub fn can_read_length_from_heap(&self) -> bool {
        match self.storage {
            VableArrayStorage::EmbeddedArray { .. } => true,
            VableArrayStorage::RustVec { .. } => true,
            VableArrayStorage::DirectPointer => {
                !(self.length_offset == 0 && self.items_offset == 0)
            }
        }
    }

    unsafe fn data_ptr(&self, obj_ptr: *const u8) -> *const u8 {
        unsafe {
            match self.storage {
                VableArrayStorage::DirectPointer => {
                    *(obj_ptr.add(self.field_offset) as *const *const u8)
                }
                VableArrayStorage::EmbeddedArray { ptr_offset } => {
                    // 2-level: field_offset → pointer to container struct,
                    // then ptr_offset within that struct → data pointer.
                    let container = *(obj_ptr.add(self.field_offset) as *const *const u8);
                    *(container.add(ptr_offset) as *const *const u8)
                }
                VableArrayStorage::RustVec { data_ptr_fn, .. } => {
                    data_ptr_fn(obj_ptr as *mut u8) as *const u8
                }
            }
        }
    }
}

/// Reads virtualizable fields from a heap object into a flat value array.
///
/// This is the "synchronize" direction: heap → JIT representation.
///
/// `obj_ptr` is a pointer to the virtualizable heap object.
/// Returns a vector of values (static fields first, then array elements).
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object with the
/// layout described by `info`.
#[cfg(test)]
unsafe fn read_virtualizable_boxes(info: &VirtualizableInfo, obj_ptr: *const u8) -> Vec<i64> {
    unsafe {
        let mut boxes = Vec::with_capacity(info.num_static_extra_boxes);

        // Read static fields
        for field in &info.static_fields {
            let val = match field.field_type {
                Type::Int => {
                    let ptr = obj_ptr.add(field.offset) as *const i64;
                    *ptr
                }
                Type::Float => {
                    let ptr = obj_ptr.add(field.offset) as *const f64;
                    f64::to_bits(*ptr) as i64
                }
                Type::Ref => {
                    let ptr = obj_ptr.add(field.offset) as *const i64;
                    *ptr
                }
                Type::Void => 0,
            };
            boxes.push(val);
        }

        boxes
    }
}

/// Writes values back from JIT representation to a virtualizable heap object.
///
/// This is the "force" direction: JIT representation → heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object with the
/// layout described by `info`.
#[cfg(test)]
unsafe fn write_virtualizable_boxes(info: &VirtualizableInfo, obj_ptr: *mut u8, boxes: &[i64]) {
    unsafe {
        for (i, field) in info.static_fields.iter().enumerate() {
            if i >= boxes.len() {
                break;
            }
            match field.field_type {
                Type::Int | Type::Ref => {
                    let ptr = obj_ptr.add(field.offset) as *mut i64;
                    *ptr = boxes[i];
                }
                Type::Float => {
                    let ptr = obj_ptr.add(field.offset) as *mut f64;
                    *ptr = f64::from_bits(boxes[i] as u64);
                }
                Type::Void => {}
            }
        }
    }
}

/// Reset the vable_token on a virtualizable object (unconditional).
///
/// Sets the token to TOKEN_NONE without forcing. Use `force_virtualizable`
/// or `VirtualizableInfo::force_now` if you need to flush JIT state first.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
#[cfg(test)]
unsafe fn reset_vable_token(info: &VirtualizableInfo, obj_ptr: *mut u8) {
    unsafe {
        let token_ptr = obj_ptr.add(info.token_offset) as *mut usize;
        *token_ptr = 0;
    }
}

/// Check if the vable_token is non-null (JIT code may be active).
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
#[allow(dead_code)]
unsafe fn is_token_nonnull(info: &VirtualizableInfo, obj_ptr: *const u8) -> bool {
    unsafe {
        let token_ptr = obj_ptr.add(info.token_offset) as *const usize;
        *token_ptr != 0
    }
}

/// Force a virtualizable: flush JIT-held values back to the heap.
///
/// Token semantics:
/// - TOKEN_NONE (0): not in JIT, nothing to do.
/// - TOKEN_TRACING_RESCALL (prebuilt GCREF): tracing + residual call, just clear.
/// - Any other non-zero value: active JIT frame pointer. Call `force_fn`
///   with the frame pointer, which must clear the token itself.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
#[cfg(test)]
unsafe fn force_virtualizable(
    info: &VirtualizableInfo,
    obj_ptr: *mut u8,
    force_fn: impl FnOnce(u64),
) {
    unsafe {
        info.force_now(obj_ptr, force_fn);
    }
}

/// Read array lengths from a virtualizable heap object.
///
/// For each array field, reads the array pointer, then reads its length
/// from the configured `length_offset` within the array header.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid virtualizable object.
#[cfg(test)]
unsafe fn read_array_lengths(info: &VirtualizableInfo, obj_ptr: *const u8) -> Vec<usize> {
    unsafe { info.read_array_lengths_from_heap(obj_ptr) }
}

/// Read a virtualizable's array field contents from the heap.
///
/// The array pointer is read from the virtualizable at `array_info.field_offset`,
/// then `length` elements are read starting at `array_info.items_offset`.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and the array has at least `length` elements.
#[cfg(test)]
#[allow(dead_code)]
unsafe fn read_virtualizable_array(
    array_info: &VableArrayInfo,
    obj_ptr: *const u8,
    length: usize,
) -> Vec<i64> {
    unsafe {
        let mut values = Vec::with_capacity(length);
        for i in 0..length {
            let array_ptr = array_info.data_ptr(obj_ptr);
            let item_offset = array_info.items_offset + i * array_info.item_size;
            let val = match array_info.item_type {
                Type::Int | Type::Ref => *(array_ptr.add(item_offset) as *const i64),
                Type::Float => f64::to_bits(*(array_ptr.add(item_offset) as *const f64)) as i64,
                Type::Void => 0,
            };
            values.push(val);
        }
        values
    }
}

/// symbolic.py:get_array_token → itemsize = sizeof(ARRAY.OF).
///
/// Returns the byte size of a single item for the given JIT type, matching
/// RPython's `llmemory.sizeof(ARRAY.OF)` / `ctypes.sizeof(...)` for
/// standard types.
/// Resolve the authoritative byte size of an array item from the field's array
/// descriptor.  Falls back to the word-sized `item_size_for_type` only when the
/// descriptor is not an array descriptor (malformed test descrs).  Using the
/// descriptor keeps the blackhole/heap array stride in lock-step with the
/// explicit item size the JIT codegen records (e.g. a fixed `8` for `i64`
/// payloads), instead of re-deriving a machine word on 32-bit targets.
/// `virtualizable.py:44-56` — the array-field arrivals check.
///
/// ```text
/// ARRAY = ARRAYPTR.TO
/// if not isinstance(ARRAY, lltype.GcArray):
///     raise Exception("The virtualizable field '%s' is not an array (found %r). ...")
/// ```
///
/// Upstream refuses at translation time, so nothing downstream has to
/// represent "this was not an array".  Answering `0` instead would not be a
/// weaker diagnostic, it would be a WRONG ANSWER on the memory-safety path:
/// `0` is the value `walk_vable_resume_ref_roots` reads as "no type is known
/// for this array", and it uses that to SKIP the nursery type-id comparison
/// before pushing the array-pointer slot as a resume ref root.  A
/// mis-declared field would silently turn a live GC check off rather than
/// fail.  Panicking keeps the two meanings apart by making the absent case
/// unrepresentable.
///
/// The message carries upstream's remedy verbatim because it is the useful
/// half: an array that fails this test is usually one that gets resized at
/// run time, and `make_sure_not_resized()` is what pins it.
fn array_field_type_id(name: &str, array_descr: &DescrRef) -> u32 {
    let Some(descr) = array_descr.as_array_descr() else {
        panic!(
            "The virtualizable field '{name}' is not an array (found {array_descr:?}). \
             It usually means that you must try harder to ensure that the list is not \
             resized at run-time. You can do that by using make_sure_not_resized()."
        );
    };
    descr.type_id()
}

fn array_descr_item_size(array_descr: &DescrRef, item_type: Type) -> usize {
    majit_ir::descr::unpack_arraydescr(array_descr)
        .map(|(_, item_size, _)| item_size)
        .unwrap_or_else(|| item_size_for_type(item_type))
}

/// `arraydescr.is_item_signed()`, the sign half of `bh_getarrayitem_gc_i`'s
/// `(item_size, is_item_signed)` dispatch.
///
/// A field with no array descriptor falls back to signed: `item_size_for_type`
/// gives such a field the machine word, and RPython's word-sized integer array
/// is `Signed`.  The two answers only differ for a narrower item, which cannot
/// arise without a descriptor to declare the narrower size.
fn array_descr_item_signed(array_descr: &DescrRef) -> bool {
    array_descr
        .as_array_descr()
        .map(|ad| ad.is_item_signed())
        .unwrap_or(true)
}

pub fn item_size_for_type(ty: Type) -> usize {
    match ty {
        // symbolic.py:get_size → sizeof(Signed) / sizeof(Ptr).  `Signed` is
        // the machine word (4 bytes on wasm32, 8 on 64-bit), matching the
        // `isize`/`usize`/pointer virtualizable fields it describes.  A fixed
        // 8 here would over-size word fields on wasm32.  Fixed-width 64-bit
        // payloads (e.g. `W_IntObject.intval`, list-strategy backing arrays)
        // carry their own explicit `8`-byte descriptors, not this helper.
        Type::Int => std::mem::size_of::<isize>(),
        Type::Ref => std::mem::size_of::<usize>(),
        // symbolic.py:get_size → sizeof(Float) (C double)
        Type::Float => std::mem::size_of::<f64>(),
        Type::Void => 0,
    }
}

/// Write values into a virtualizable's array field on the heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and the array has sufficient space.
#[cfg(test)]
unsafe fn write_virtualizable_array(array_info: &VableArrayInfo, obj_ptr: *mut u8, values: &[i64]) {
    unsafe {
        for (i, &val) in values.iter().enumerate() {
            let array_ptr = array_info.data_ptr(obj_ptr.cast_const()) as *mut u8;
            let item_offset = array_info.items_offset + i * array_info.item_size;
            match array_info.item_type {
                Type::Int | Type::Ref => {
                    let ptr = array_ptr.add(item_offset) as *mut i64;
                    *ptr = val;
                }
                Type::Float => {
                    let ptr = array_ptr.add(item_offset) as *mut f64;
                    *ptr = f64::from_bits(val as u64);
                }
                Type::Void => {}
            }
        }
    }
}

/// Read all virtualizable state (static fields + array fields) into a flat box array.
///
/// Returns (static_boxes, array_boxes_per_field).
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and arrays have the specified lengths.
#[cfg(test)]
unsafe fn read_all_virtualizable_boxes(
    info: &VirtualizableInfo,
    obj_ptr: *const u8,
    array_lengths: &[usize],
) -> (Vec<i64>, Vec<Vec<i64>>) {
    // SAFETY: forwarded from this function's caller via the doc-comment
    // contract — `obj_ptr` valid + array lengths match storage.
    unsafe { info.read_all_boxes(obj_ptr, array_lengths) }
}

/// Write all virtualizable state back to the heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and arrays have sufficient space.
#[cfg(test)]
unsafe fn write_all_virtualizable_boxes(
    info: &VirtualizableInfo,
    obj_ptr: *mut u8,
    static_boxes: &[i64],
    array_boxes: &[Vec<i64>],
) {
    // SAFETY: per the function-level Safety contract, callers guarantee
    // `obj_ptr` is valid and the arrays have sufficient space; that
    // forwards directly to `write_all_boxes`'s same precondition.
    unsafe { info.write_all_boxes(obj_ptr, static_boxes, array_boxes) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: add an array field with a fresh descriptor.
    /// Production code should use GcCache-sourced descriptors.
    fn test_add_array_field(
        info: &mut VirtualizableInfo,
        name: &str,
        item_type: Type,
        field_offset: usize,
        length_offset: usize,
        items_offset: usize,
    ) {
        let item_size = item_size_for_type(item_type);
        let descr = majit_ir::make_array_descr(items_offset, item_size, item_type);
        info.add_array_field(
            name,
            item_type,
            field_offset,
            length_offset,
            items_offset,
            descr,
        );
    }

    /// `virtualizable.py:47-56` — a redirected field that is not an array is a
    /// translation-time refusal upstream, so it must be one here too.
    ///
    /// The regression this pins is not the diagnostic: answering `0` for the
    /// type id would hand `walk_vable_resume_ref_roots` the same value it reads
    /// as "no type is known", which switches OFF the nursery type-id compare
    /// guarding the resume ref root push.
    #[test]
    #[should_panic(expected = "is not an array")]
    fn a_non_array_descr_is_refused_at_declaration() {
        let mut info = VirtualizableInfo::new(0);
        let field_descr =
            majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::descr::ArrayFlag::Pointer);
        info.add_array_field("locals_w", Type::Ref, 16, 0, 0, field_descr);
    }

    /// Test helper: add an embedded array field with a fresh descriptor.
    fn test_add_embedded_array_field(
        info: &mut VirtualizableInfo,
        name: &str,
        item_type: Type,
        field_offset: usize,
        ptr_offset: usize,
        length_offset: usize,
        items_offset: usize,
    ) {
        let item_size = item_size_for_type(item_type);
        let descr = majit_ir::make_array_descr(items_offset, item_size, item_type);
        info.add_embedded_array_field(
            name,
            item_type,
            field_offset,
            ptr_offset,
            length_offset,
            items_offset,
            descr,
        );
    }

    #[test]
    fn finalize_arc_stamps_vinfo_backref_on_field_descrs() {
        // pyjitpl.py:1148-1149 parity — after `finalize_arc`, every
        // field descriptor returned by the vinfo (vable_token_descr +
        // static + array) must answer `get_vinfo() == Some(info)`.
        let mut info = VirtualizableInfo::new(24);
        info.add_field("pc", Type::Int, 8);
        test_add_array_field(&mut info, "locals_w", Type::Ref, 16, 0, 0);
        let descr: DescrRef = majit_ir::descr::make_size_descr(64);
        let info = info.finalize_arc(descr);

        let token = info
            .vable_token_descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .expect("vable_token_descr is a FieldDescr")
            .get_vinfo()
            .expect("token descr carries vinfo");
        assert!(std::ptr::eq(
            token.as_any().downcast_ref::<VirtualizableInfo>().unwrap() as *const _,
            Arc::as_ptr(&info),
        ));

        for d in info._static_field_descrs.iter() {
            let v = d
                .as_field_descr()
                .unwrap()
                .get_vinfo()
                .expect("static fd has vinfo");
            assert!(std::ptr::eq(
                v.as_any().downcast_ref::<VirtualizableInfo>().unwrap() as *const _,
                Arc::as_ptr(&info),
            ));
        }
        for d in info._array_field_descrs.iter() {
            let v = d
                .as_field_descr()
                .unwrap()
                .get_vinfo()
                .expect("array fd has vinfo");
            assert!(std::ptr::eq(
                v.as_any().downcast_ref::<VirtualizableInfo>().unwrap() as *const _,
                Arc::as_ptr(&info),
            ));
        }
    }

    #[test]
    fn set_parent_descr_leaves_vinfo_backref_none() {
        // Legacy by-value path: `set_parent_descr` does NOT populate
        // vinfo backref (no Arc<Self> to weak-ref). `get_vinfo()` falls
        // through to the trait default `None`.
        let mut info = VirtualizableInfo::new(24);
        info.add_field("pc", Type::Int, 8);
        let descr: DescrRef = majit_ir::descr::make_size_descr(64);
        info.set_parent_descr(descr);

        let token = info.vable_token_descr.as_ref().unwrap();
        assert!(token.as_field_descr().unwrap().get_vinfo().is_none());
    }

    #[test]
    fn test_virtualizable_info_creation() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        test_add_array_field(&mut info, "stack", Type::Int, 24, 0, 0);

        assert_eq!(info.num_fields(), 2);
        assert_eq!(info.num_arrays(), 1);
        assert_eq!(info.num_static_extra_boxes, 2);
    }

    #[test]
    fn test_read_write_boxes() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);

        // Create a fake object (24 bytes: 8 token + 8 x + 8 y)
        let mut obj = vec![0u8; 24];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            // Write values
            *(obj_ptr.add(8) as *mut i64) = 42;
            *(obj_ptr.add(16) as *mut i64) = 99;

            // Read boxes
            let boxes = read_virtualizable_boxes(&info, obj_ptr);
            assert_eq!(boxes, vec![42, 99]);

            // Write new boxes
            write_virtualizable_boxes(&info, obj_ptr, &[100, 200]);
            assert_eq!(*(obj_ptr.add(8) as *const i64), 100);
            assert_eq!(*(obj_ptr.add(16) as *const i64), 200);
        }
    }

    #[test]
    fn test_vable_token() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            assert!(!is_token_nonnull(&info, obj_ptr));

            // Simulate JIT setting the token
            *(obj_ptr as *mut u64) = 0xDEAD;
            assert!(is_token_nonnull(&info, obj_ptr));

            // Reset it
            reset_vable_token(&info, obj_ptr);
            assert!(!is_token_nonnull(&info, obj_ptr));
        }
    }

    #[test]
    fn test_force_virtualizable_not_active() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        let mut forced = false;
        unsafe {
            force_virtualizable(&info, obj_ptr, |_| {
                forced = true;
            });
        }
        assert!(!forced, "should not force when token is zero");
    }

    #[test]
    fn test_static_field_index_lookup() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        info.add_field("z", Type::Float, 24);

        assert_eq!(info.static_field_index_by_offset(8), Some(0));
        assert_eq!(info.static_field_index_by_offset(16), Some(1));
        assert_eq!(info.static_field_index_by_offset(24), Some(2));
        assert_eq!(info.static_field_index_by_offset(999), None);
    }

    #[test]
    fn test_array_field_index_lookup() {
        let mut info = VirtualizableInfo::new(0);
        test_add_array_field(&mut info, "locals", Type::Ref, 32, 0, 0);
        test_add_array_field(&mut info, "stack", Type::Int, 40, 0, 0);

        assert_eq!(info.array_field_index_by_offset(32), Some(0));
        assert_eq!(info.array_field_index_by_offset(40), Some(1));
        assert_eq!(info.array_field_index_by_offset(999), None);
    }

    #[test]
    fn test_get_total_size() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        test_add_array_field(&mut info, "locals", Type::Ref, 24, 0, 0);
        test_add_array_field(&mut info, "stack", Type::Int, 32, 0, 0);

        // 2 static + 5 + 3 = 10
        assert_eq!(info.get_total_size(&[5, 3]), 10);
        // 2 static + 0 + 0 = 2
        assert_eq!(info.get_total_size(&[0, 0]), 2);
        // empty arrays
        assert_eq!(info.get_total_size(&[]), 2);
    }

    #[test]
    fn test_get_index_in_array() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        test_add_array_field(&mut info, "locals", Type::Ref, 24, 0, 0);
        test_add_array_field(&mut info, "stack", Type::Int, 32, 0, 0);

        let lens = &[5, 3];
        // array 0, item 0 => 2 static + 0 = 2
        assert_eq!(info.get_index_in_array(0, 0, lens), 2);
        // array 0, item 4 => 2 + 4 = 6
        assert_eq!(info.get_index_in_array(0, 4, lens), 6);
        // array 1, item 0 => 2 + 5 + 0 = 7
        assert_eq!(info.get_index_in_array(1, 0, lens), 7);
        // array 1, item 2 => 2 + 5 + 2 = 9
        assert_eq!(info.get_index_in_array(1, 2, lens), 9);
    }

    #[test]
    fn test_check_boxes() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        test_add_array_field(&mut info, "locals", Type::Ref, 24, 0, 0);

        let lens = &[3usize];
        // total = 2 + 3 = 5
        assert!(info.boxes_len_matches(&[1, 2, 3, 4, 5], lens));
        assert!(!info.boxes_len_matches(&[1, 2, 3], lens));
        assert!(!info.boxes_len_matches(&[1, 2, 3, 4, 5, 6], lens));
    }

    #[test]
    fn test_force_virtualizable_active() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        // Simulate JIT setting the token to a frame pointer
        unsafe {
            *(obj_ptr as *mut u64) = 0xBEEF;
        }

        let mut force_token_received = 0u64;
        unsafe {
            force_virtualizable(&info, obj_ptr, |token| {
                force_token_received = token;
                // force_fn must clear the token (RPython semantics)
                *(obj_ptr as *mut u64) = 0;
            });
        }
        assert_eq!(force_token_received, 0xBEEF);

        // Token should be cleared after force
        unsafe {
            assert!(!is_token_nonnull(&info, obj_ptr));
        }
    }

    #[test]
    fn test_read_array_lengths_from_heap() {
        // Heap layout:
        //   obj[0..8]:  vable_token
        //   obj[8..16]: array_ptr for "locals" (pointer to array_data_0)
        //   obj[16..24]: array_ptr for "stack" (pointer to array_data_1)
        //
        // array_data layout (default: length at 0, items at 8):
        //   [0..8]: length (usize)
        //   [8..]: items

        let mut array_data_0 = vec![0u8; 8 + 3 * 8]; // length=3, 3 items
        let mut array_data_1 = vec![0u8; 8 + 5 * 8]; // length=5, 5 items

        unsafe {
            *(array_data_0.as_mut_ptr() as *mut usize) = 3;
            *(array_data_1.as_mut_ptr() as *mut usize) = 5;
        }

        // Build the virtualizable object
        let mut obj = vec![0u8; 24]; // token + 2 array pointers
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut *const u8) = array_data_0.as_ptr();
            *(obj.as_mut_ptr().add(16) as *mut *const u8) = array_data_1.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        test_add_array_field(&mut info, "locals", Type::Ref, 8, 0, 0);
        test_add_array_field(&mut info, "stack", Type::Int, 16, 0, 8);

        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![3, 5]);
    }

    #[test]
    fn test_read_array_lengths_null_pointer() {
        // If array pointer is null, length should be 0
        let mut obj = vec![0u8; 16]; // token + 1 null array pointer
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut *const u8) = std::ptr::null();
        }

        let mut info = VirtualizableInfo::new(0);
        test_add_array_field(&mut info, "locals", Type::Ref, 8, 0, 0);

        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![0]);
    }

    #[test]
    fn test_read_array_lengths_from_embedded_array_container() {
        // `EmbeddedArray` is a 2-level layout: the frame field stores a
        // pointer to a container struct that owns the items pointer.
        // Matches pyre's PyFrame.locals_cells_stack_w: *mut FixedObjectArray.
        #[repr(C)]
        struct ArrayContainer {
            ptr: *mut i64,
            len: usize,
        }

        #[repr(C)]
        struct Obj {
            token: usize,
            arr: *mut ArrayContainer,
        }

        let mut backing = vec![10i64, 20, 30];
        let mut container = ArrayContainer {
            ptr: backing.as_mut_ptr(),
            len: backing.len(),
        };
        let obj = Obj {
            token: 0,
            arr: &mut container as *mut _,
        };

        let mut info = VirtualizableInfo::new(0);
        test_add_embedded_array_field(
            &mut info,
            "arr",
            Type::Int,
            std::mem::offset_of!(Obj, arr),
            std::mem::offset_of!(ArrayContainer, ptr),
            std::mem::offset_of!(ArrayContainer, len),
            0,
        );

        let lengths = unsafe { read_array_lengths(&info, (&obj as *const Obj).cast()) };
        assert_eq!(lengths, vec![3]);
        let (boxes, array_lengths) =
            unsafe { info.load_list_of_boxes((&obj as *const Obj).cast()) };
        assert_eq!(array_lengths, vec![3]);
        assert_eq!(boxes, vec![10, 20, 30]);
    }

    #[test]
    fn test_auto_sync_reads_all_fields() {
        // Build a complete virtualizable heap object with static fields + arrays.
        //
        // Layout:
        //   obj[0..8]:   vable_token
        //   obj[8..16]:  field "x" (i64)
        //   obj[16..24]: field "y" (i64)
        //   obj[24..32]: array_ptr for "stack"
        //
        // array_data (default layout):
        //   [0..8]: length = 2
        //   [8..24]: items [10, 20]

        let mut array_data = vec![0u8; 8 + 2 * 8];
        unsafe {
            *(array_data.as_mut_ptr() as *mut usize) = 2;
            *(array_data.as_mut_ptr().add(8) as *mut i64) = 10;
            *(array_data.as_mut_ptr().add(16) as *mut i64) = 20;
        }

        let mut obj = vec![0u8; 32];
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut i64) = 42;
            *(obj.as_mut_ptr().add(16) as *mut i64) = 99;
            *(obj.as_mut_ptr().add(24) as *mut *const u8) = array_data.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        test_add_array_field(&mut info, "stack", Type::Int, 24, 0, 8);

        // Use read_array_lengths + read_all_virtualizable_boxes (the auto path)
        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![2]);

        let (static_boxes, array_boxes) =
            unsafe { read_all_virtualizable_boxes(&info, obj.as_ptr(), &lengths) };
        assert_eq!(static_boxes, vec![42, 99]);
        assert_eq!(array_boxes, vec![vec![10, 20]]);
    }

    #[test]
    fn test_array_field_with_custom_layout() {
        // Custom layout: length at offset 16, items at offset 24
        // (e.g., array header has 16 bytes of metadata before length)

        let mut array_data = vec![0u8; 24 + 3 * 8]; // header(24) + 3 items
        unsafe {
            *(array_data.as_mut_ptr().add(16) as *mut usize) = 3; // length at offset 16
            *(array_data.as_mut_ptr().add(24) as *mut i64) = 100; // item 0 at offset 24
            *(array_data.as_mut_ptr().add(32) as *mut i64) = 200;
            *(array_data.as_mut_ptr().add(40) as *mut i64) = 300;
        }

        let mut obj = vec![0u8; 16]; // token + 1 array pointer
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut *const u8) = array_data.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        test_add_array_field(&mut info, "data", Type::Int, 8, 16, 24);

        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![3]);

        let (_, array_boxes) =
            unsafe { read_all_virtualizable_boxes(&info, obj.as_ptr(), &lengths) };
        assert_eq!(array_boxes, vec![vec![100, 200, 300]]);

        // Verify write roundtrip with custom layout
        let mut obj_mut = obj.clone();
        unsafe {
            write_virtualizable_array(
                &info.array_fields[0],
                obj_mut.as_mut_ptr(),
                &[111, 222, 333],
            );
        }
        // Re-read from the actual array_data (obj_mut still points to array_data)
        let (_, array_boxes2) =
            unsafe { read_all_virtualizable_boxes(&info, obj_mut.as_ptr(), &lengths) };
        assert_eq!(array_boxes2, vec![vec![111, 222, 333]]);
    }

    #[test]
    fn test_virtualizable_with_array_read_write() {
        // RPython parity: test_virtualizable_with_array
        // VirtualizableInfo with 1 static field + 1 array field.
        // read_all → modify → write_all → verify heap updated.

        // Heap layout:
        //   obj[0..8]:   vable_token
        //   obj[8..16]:  field "pc" (i64)
        //   obj[16..24]: array_ptr for "stack"
        //
        // array layout (default):
        //   [0..8]: length = 3
        //   [8..32]: items [10, 20, 30]

        let mut array_data = vec![0u8; 8 + 3 * 8];
        unsafe {
            *(array_data.as_mut_ptr() as *mut usize) = 3;
            *(array_data.as_mut_ptr().add(8) as *mut i64) = 10;
            *(array_data.as_mut_ptr().add(16) as *mut i64) = 20;
            *(array_data.as_mut_ptr().add(24) as *mut i64) = 30;
        }

        let mut obj = vec![0u8; 24];
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut i64) = 7; // pc = 7
            *(obj.as_mut_ptr().add(16) as *mut *const u8) = array_data.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        test_add_array_field(&mut info, "stack", Type::Int, 16, 0, 8);

        // Read all boxes
        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![3]);

        let (static_boxes, array_boxes) =
            unsafe { read_all_virtualizable_boxes(&info, obj.as_ptr(), &lengths) };
        assert_eq!(static_boxes, vec![7]);
        assert_eq!(array_boxes, vec![vec![10, 20, 30]]);

        // Modify and write back
        let new_static = vec![42i64];
        let new_arrays = vec![vec![100i64, 200, 300]];
        unsafe {
            write_all_virtualizable_boxes(&info, obj.as_mut_ptr(), &new_static, &new_arrays);
        }

        // Verify heap was updated
        unsafe {
            assert_eq!(*(obj.as_ptr().add(8) as *const i64), 42);
            // Array items via raw pointer (array_data is still alive)
            assert_eq!(*(array_data.as_ptr().add(8) as *const i64), 100);
            assert_eq!(*(array_data.as_ptr().add(16) as *const i64), 200);
            assert_eq!(*(array_data.as_ptr().add(24) as *const i64), 300);
        }
    }

    #[test]
    fn test_virtualizable_with_ref_array_preserves_pointer_values() {
        // RPython parity: Ref-typed virtualizable arrays stay pointer-typed
        // through heap load and writeback. They must not degrade into raw ints.

        let mut array_data = vec![0u8; 8 + 3 * 8];
        unsafe {
            *(array_data.as_mut_ptr() as *mut usize) = 3;
            *(array_data.as_mut_ptr().add(8) as *mut usize) = 0x1000;
            *(array_data.as_mut_ptr().add(16) as *mut usize) = 0;
            *(array_data.as_mut_ptr().add(24) as *mut usize) = 0x2000;
        }

        let mut obj = vec![0u8; 16];
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut *const u8) = array_data.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        test_add_array_field(&mut info, "locals_w", Type::Ref, 8, 0, 8);

        let (boxes, lengths) = unsafe { info.load_list_of_boxes(obj.as_ptr()) };
        assert_eq!(lengths, vec![3]);
        assert_eq!(boxes, vec![0x1000, 0, 0x2000]);

        let new_array_boxes = vec![vec![0x3000_i64, 0, 0x4000_i64]];
        unsafe {
            info.write_all_boxes(obj.as_mut_ptr(), &[], &new_array_boxes);
        }

        unsafe {
            assert_eq!(*(array_data.as_ptr().add(8) as *const usize), 0x3000);
            assert_eq!(*(array_data.as_ptr().add(16) as *const usize), 0);
            assert_eq!(*(array_data.as_ptr().add(24) as *const usize), 0x4000);
        }
    }

    #[test]
    fn test_force_virtualizable_triggers_callback() {
        // Non-zero token (active JIT frame) → callback receives token value
        // → force_fn clears the token → verified.

        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        // Set a non-zero token (simulating active JIT)
        unsafe {
            *(obj_ptr as *mut u64) = 0xCAFE_BABE;
        }

        let mut received_token = 0u64;
        unsafe {
            force_virtualizable(&info, obj_ptr, |token| {
                received_token = token;
                // force_fn must clear the token
                *(obj_ptr as *mut u64) = 0;
            });
        }

        assert_eq!(received_token, 0xCAFE_BABE);
        // Token must be cleared after force
        unsafe {
            assert_eq!(*(obj.as_ptr() as *const u64), 0);
            assert!(!is_token_nonnull(&info, obj.as_ptr()));
        }
    }

    #[test]
    fn test_sync_round_trip() {
        // RPython parity: test_sync_before_after_jit
        // Write known values to heap → read boxes → modify → write back → verify.

        // Heap layout:
        //   obj[0..8]:   vable_token
        //   obj[8..16]:  field "a" (i64)
        //   obj[16..24]: field "b" (i64)
        //   obj[24..32]: array_ptr for "vals"
        //
        // array: length=2, items=[50, 60]

        let mut array_data = vec![0u8; 8 + 2 * 8];
        unsafe {
            *(array_data.as_mut_ptr() as *mut usize) = 2;
            *(array_data.as_mut_ptr().add(8) as *mut i64) = 50;
            *(array_data.as_mut_ptr().add(16) as *mut i64) = 60;
        }

        let mut obj = vec![0u8; 32];
        unsafe {
            *(obj.as_mut_ptr().add(8) as *mut i64) = 11;
            *(obj.as_mut_ptr().add(16) as *mut i64) = 22;
            *(obj.as_mut_ptr().add(24) as *mut *const u8) = array_data.as_ptr();
        }

        let mut info = VirtualizableInfo::new(0);
        info.add_field("a", Type::Int, 8);
        info.add_field("b", Type::Int, 16);
        test_add_array_field(&mut info, "vals", Type::Int, 24, 0, 8);

        // sync_before_jit: read from heap
        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        let (mut statics, mut arrays) =
            unsafe { read_all_virtualizable_boxes(&info, obj.as_ptr(), &lengths) };
        assert_eq!(statics, vec![11, 22]);
        assert_eq!(arrays, vec![vec![50, 60]]);

        // Simulate JIT execution modifying values
        statics[0] = 111;
        statics[1] = 222;
        arrays[0][0] = 500;
        arrays[0][1] = 600;

        // sync_after_jit: write back to heap
        unsafe {
            write_all_virtualizable_boxes(&info, obj.as_mut_ptr(), &statics, &arrays);
        }

        // Verify heap has new values
        unsafe {
            assert_eq!(*(obj.as_ptr().add(8) as *const i64), 111);
            assert_eq!(*(obj.as_ptr().add(16) as *const i64), 222);
            assert_eq!(*(array_data.as_ptr().add(8) as *const i64), 500);
            assert_eq!(*(array_data.as_ptr().add(16) as *const i64), 600);
        }
    }

    #[test]
    fn test_to_optimizer_config_preserves_offsets() {
        // RPython parity: test_to_optimizer_config
        // VirtualizableInfo → VirtualizableConfig, verify offsets match.

        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Float, 24);
        info.add_field("z", Type::Ref, 40);
        test_add_array_field(&mut info, "locals", Type::Ref, 48, 0, 0);
        test_add_array_field(&mut info, "stack", Type::Int, 56, 0, 0);

        let config = info.to_optimizer_config();

        assert_eq!(config.static_field_offsets, vec![8, 24, 40]);
        assert_eq!(
            config.static_field_types,
            vec![Type::Int, Type::Float, Type::Ref]
        );
        assert_eq!(config.array_field_offsets, vec![48, 56]);
        assert_eq!(config.array_item_types, vec![Type::Ref, Type::Int]);
    }

    /// `identity_live_index == None` means two different things, and the layout
    /// — not the absent declaration — decides which.
    #[test]
    fn test_to_optimizer_config_identity_slot_is_layout_discriminated() {
        // Legacy frame-first (PyFrame): the frame IS flat slot 0, so an absent
        // declaration is not a gap.
        let legacy = VirtualizableInfo::new(0);
        assert_eq!(legacy.identity_ref_bank_index, None);
        assert_eq!(legacy.identity_live_index, None);
        assert_eq!(
            legacy.to_optimizer_config().identity_input_index,
            Some(0),
            "the frame-first layout leads with the identity",
        );

        // Banked-identity (macro state-field) with a declaration: honour it.
        let mut declared = VirtualizableInfo::without_vable_token();
        declared.identity_ref_bank_index = Some(1);
        declared.identity_live_index = Some(3);
        assert_eq!(declared.to_optimizer_config().identity_input_index, Some(3));

        // Banked-identity with NO declaration — what the macro emits for a state
        // carrying a fixed `[int]` array, whose identity slot depends on that
        // array's runtime length. Slot 0 is an int scalar there, so the config
        // must carry no slot at all and let `VirtualizableTracker` decline.
        let mut undeclared = VirtualizableInfo::without_vable_token();
        undeclared.identity_ref_bank_index = Some(1);
        assert_eq!(undeclared.identity_live_index, None);
        assert_eq!(
            undeclared.to_optimizer_config().identity_input_index,
            None,
            "an undeclared banked identity must not fall back to slot 0",
        );
    }

    #[test]
    fn test_load_list_of_boxes_reads_from_object() {
        // RPython parity: vinfo.load_list_of_boxes() reads from actual object.
        #[repr(C)]
        struct Frame {
            token: u64,
            x: i64,
            y: i64,
            arr_ptr: *const u8,
        }

        let arr_data: Vec<i64> = vec![100, 200, 300];
        let mut frame = Frame {
            token: 0,
            x: 42,
            y: 99,
            arr_ptr: arr_data.as_ptr() as *const u8,
        };

        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        // Array: pointer at offset 24, no length header (items_offset=0)
        // For this test, we won't call get_array_length since there's no header.
        // Instead we test read_field/write_field directly.

        let obj = &mut frame as *mut Frame as *mut u8;
        unsafe {
            assert_eq!(info.read_field(obj, 0), 42);
            assert_eq!(info.read_field(obj, 1), 99);

            info.write_field(obj, 0, 111);
            info.write_field(obj, 1, 222);
            assert_eq!(frame.x, 111);
            assert_eq!(frame.y, 222);
        }
    }

    #[test]
    fn test_load_list_of_boxes_with_array() {
        // RPython parity: vinfo.load_list_of_boxes() with array fields.
        #[repr(C)]
        struct ArrayHeader {
            length: usize,
            items: [i64; 3],
        }

        #[repr(C)]
        struct Frame {
            token: u64,
            x: i64,
            arr_ptr: *const u8,
        }

        let arr = ArrayHeader {
            length: 3,
            items: [10, 20, 30],
        };
        let frame = Frame {
            token: 0,
            x: 42,
            arr_ptr: &arr as *const ArrayHeader as *const u8,
        };

        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        // Array at offset 16, length at offset 0, items at offset 8 (after length)
        test_add_array_field(&mut info, "arr", Type::Int, 16, 0, 8);

        let obj = &frame as *const Frame as *const u8;
        unsafe {
            assert_eq!(info.get_array_length(obj, 0), 3);
            assert_eq!(info.read_array_item(obj, 0, 0), 10);
            assert_eq!(info.read_array_item(obj, 0, 1), 20);
            assert_eq!(info.read_array_item(obj, 0, 2), 30);

            let (boxes, lengths) = info.load_list_of_boxes(obj);
            assert_eq!(lengths, vec![3]);
            assert_eq!(boxes, vec![42, 10, 20, 30]);
        }
    }

    #[test]
    fn read_write_array_item_uses_item_size_for_int() {
        // `unwrap(ARRAYITEMTYPE)` reads the real item width. A 4-byte
        // signed slot must not be loaded as i64 (that would fold in the
        // next item, or write past the last one).
        #[repr(C)]
        struct ArrayHeader {
            length: usize,
            items: [i32; 3],
        }

        #[repr(C)]
        struct Frame {
            token: u64,
            arr_ptr: *const u8,
        }

        let mut arr = ArrayHeader {
            length: 3,
            items: [10, 20, 30],
        };
        let mut frame = Frame {
            token: 0,
            arr_ptr: &mut arr as *mut ArrayHeader as *const u8,
        };

        let mut info = VirtualizableInfo::new(0);
        let items_offset = std::mem::offset_of!(ArrayHeader, items);
        let descr = majit_ir::make_array_descr_signed(items_offset, 4, Type::Int, true);
        info.add_array_field("arr", Type::Int, 8, 0, items_offset, descr);

        let obj = &mut frame as *mut Frame as *mut u8;
        unsafe {
            assert_eq!(info.read_array_item(obj, 0, 0), 10);
            assert_eq!(info.read_array_item(obj, 0, 1), 20);
            assert_eq!(info.read_array_item(obj, 0, 2), 30);
            info.write_array_item(obj, 0, 1, -7);
            assert_eq!(info.read_array_item(obj, 0, 0), 10);
            assert_eq!(info.read_array_item(obj, 0, 1), -7);
            assert_eq!(info.read_array_item(obj, 0, 2), 30);
        }
    }

    #[test]
    fn test_force_from_boxes_writes_back() {
        #[repr(C)]
        struct ArrayHeader {
            length: usize,
            items: [i64; 2],
        }

        #[repr(C)]
        struct Frame {
            token: u64,
            x: i64,
            arr_ptr: *mut u8,
        }

        let mut arr = ArrayHeader {
            length: 2,
            items: [0, 0],
        };
        let mut frame = Frame {
            token: 999,
            x: 0,
            arr_ptr: &mut arr as *mut ArrayHeader as *mut u8,
        };

        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        test_add_array_field(&mut info, "arr", Type::Int, 16, 0, 8);

        let obj = &mut frame as *mut Frame as *mut u8;
        // RPython: boxes = [x, arr[0], arr[1], vable_box]
        let boxes = vec![42, 100, 200, obj as i64];

        unsafe {
            info.force_from_boxes(obj, &boxes);

            assert_eq!(frame.x, 42);
            assert_eq!(arr.items[0], 100);
            assert_eq!(arr.items[1], 200);
            assert_eq!(frame.token, 0); // token cleared
        }
    }

    #[test]
    fn test_vable_token_roundtrip() {
        let tracing_rescall = token_tracing_rescall();
        assert_eq!(VableToken::from_raw(0), VableToken::None);
        assert_eq!(
            VableToken::from_raw(tracing_rescall),
            VableToken::TracingRescall
        );
        assert_eq!(VableToken::from_raw(0xBEEF), VableToken::Active(0xBEEF));

        assert_eq!(VableToken::None.to_raw(), 0);
        assert_eq!(VableToken::TracingRescall.to_raw(), tracing_rescall);
        assert_eq!(VableToken::Active(0xBEEF).to_raw(), 0xBEEF);
    }

    #[test]
    fn test_tracing_before_after_residual_call_not_forced() {
        // Set RESCALL → callee does NOT touch the vable → after returns false → token cleared
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            // Before: token is NONE
            assert_eq!(info.read_token(obj_ptr), VableToken::None);

            // Set RESCALL
            info.tracing_before_residual_call(obj_ptr);
            assert_eq!(info.read_token(obj_ptr), VableToken::TracingRescall);

            // After: not forced (token still RESCALL) → returns false, clears token
            let forced = info.tracing_after_residual_call(obj_ptr);
            assert!(!forced);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_tracing_after_residual_call_forced() {
        // Set RESCALL → callee forces (clears token to 0) → after returns true
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            info.tracing_before_residual_call(obj_ptr);
            assert_eq!(info.read_token(obj_ptr), VableToken::TracingRescall);

            // Simulate callee forcing: clears token to NONE
            *(obj_ptr as *mut u64) = 0;

            // After: was forced → returns true
            let forced = info.tracing_after_residual_call(obj_ptr);
            assert!(forced);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_force_now_tracing_rescall() {
        // force_now during tracing (RESCALL) — just clears, no callback
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            *(obj_ptr as *mut u64) = token_tracing_rescall();

            let mut called = false;
            info.force_now(obj_ptr, |_| {
                called = true;
            });

            assert!(!called, "force_fn should NOT be called for TRACING_RESCALL");
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_force_now_active_jit() {
        // force_now with active JIT frame pointer — calls force_fn, verifies token cleared
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            *(obj_ptr as *mut u64) = 0xDEAD_BEEF;

            let mut received = 0u64;
            info.force_now(obj_ptr, |token| {
                received = token;
                // force_fn must clear the token
                *(obj_ptr as *mut u64) = 0;
            });

            assert_eq!(received, 0xDEAD_BEEF);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_force_now_none() {
        // force_now when token is NONE — no-op
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            let mut called = false;
            info.force_now(obj_ptr, |_| {
                called = true;
            });

            assert!(!called, "force_fn should NOT be called for TOKEN_NONE");
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_clear_vable_token_forces_active_jit_token() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            *(obj_ptr as *mut u64) = 0xCAFE_BABE;

            let mut received = 0u64;
            info.clear_vable_token(obj_ptr, |token| {
                received = token;
                *(obj_ptr as *mut u64) = 0;
            });

            assert_eq!(received, 0xCAFE_BABE);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_clear_vable_token_clears_tracing_rescall_without_force() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            *(obj_ptr as *mut u64) = token_tracing_rescall();

            let mut called = false;
            info.clear_vable_token(obj_ptr, |_| {
                called = true;
            });

            assert!(!called);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_force_virtualizable_if_necessary_skips_none_token() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            let mut called = false;
            info.force_virtualizable_if_necessary(obj_ptr, |_| {
                called = true;
            });
            assert!(!called);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_reset_token_gcref_resets_active_token() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            *(obj_ptr as *mut u64) = 0xABCD;
            info.reset_token_gcref(obj_ptr);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_read_write_token() {
        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];
        let obj_ptr = obj.as_mut_ptr();

        unsafe {
            assert_eq!(info.read_token(obj_ptr), VableToken::None);

            info.write_token(obj_ptr, VableToken::TracingRescall);
            assert_eq!(info.read_token(obj_ptr), VableToken::TracingRescall);

            info.write_token(obj_ptr, VableToken::Active(0x1234));
            assert_eq!(info.read_token(obj_ptr), VableToken::Active(0x1234));

            info.write_token(obj_ptr, VableToken::None);
            assert_eq!(info.read_token(obj_ptr), VableToken::None);
        }
    }

    #[test]
    fn test_get_index_in_array_multiple_arrays() {
        // RPython parity: test_get_index_in_array with 2 static + 2 array fields.
        // Flat layout: [static0, static1, array0[0..3], array1[0..5]]
        //   index 0..1 = static fields
        //   index 2..4 = array0 (length 3)
        //   index 5..9 = array1 (length 5)

        let mut info = VirtualizableInfo::new(0);
        info.add_field("a", Type::Int, 8);
        info.add_field("b", Type::Int, 16);
        test_add_array_field(&mut info, "arr0", Type::Int, 24, 0, 0);
        test_add_array_field(&mut info, "arr1", Type::Int, 32, 0, 0);

        let lens = &[3usize, 5];

        // Total size = 2 + 3 + 5 = 10
        assert_eq!(info.get_total_size(lens), 10);

        // array0, item 2 → 2 (statics) + 2 = 4
        assert_eq!(info.get_index_in_array(0, 2, lens), 4);

        // array1, item 0 → 2 (statics) + 3 (array0 len) + 0 = 5
        assert_eq!(info.get_index_in_array(1, 0, lens), 5);

        // array1, item 4 → 2 + 3 + 4 = 9
        assert_eq!(info.get_index_in_array(1, 4, lens), 9);
    }
}

// ── resume.py:1350 VirtualizableInfo trait impl ──

impl crate::resume::VirtualizableInfo for VirtualizableInfo {
    /// virtualizable.py get_total_size
    fn get_total_size(&self, virtualizable: i64) -> usize {
        let mut size = self.static_fields.len();
        let vable_ptr = virtualizable as *const u8;
        if !vable_ptr.is_null() {
            for array in &self.array_fields {
                let arr_len = unsafe { bhimpl_arraylen_vable(vable_ptr, array) };
                size += arr_len;
            }
        }
        size
    }

    /// virtualizable.py reset_token_gcref
    fn reset_token_gcref(&self, virtualizable: i64) {
        let vable_ptr = virtualizable as *mut u8;
        if !vable_ptr.is_null() {
            unsafe { self.reset_vable_token(vable_ptr) };
        }
    }

    /// RPython keeps virtualizable array fields reachable through the
    /// GC-traced virtualizable object while resume data is decoded
    /// (`rpython/jit/metainterp/resume.py:1399`). pyre stores the same
    /// array pointers in raw frame fields, so expose the pointer slots to
    /// the resume-construction root stack before `write_from_resume_data_partial`
    /// and subsequent blackhole frame construction can allocate.
    fn push_resume_ref_roots(&self, virtualizable: i64) {
        let vable_ptr = virtualizable as *mut u8;
        if vable_ptr.is_null() {
            return;
        }
        // A nursery virtualizable moves as one GC object.  Its owner reference
        // is already rooted by the resume reader / Ref register bank, and the
        // PyFrame custom tracer follows locals_cells_stack_w after copying.
        // Registering this field's interior address would instead leave the
        // root stack pointing into poisoned from-space after the move.  Keep
        // the narrow field-root adaptation only for stationary frames.
        if majit_gc::gc_is_nursery_object(vable_ptr as usize) {
            return;
        }
        for array in &self.array_fields {
            match array.storage {
                VableArrayStorage::DirectPointer => {
                    // Pointer-width field: 4 bytes on wasm32. Reading 8 bytes
                    // would fold in the adjacent field and writing 8 would
                    // clobber it — `PyFrame.valuestackdepth` sits directly
                    // after `locals_cells_stack_w`, and zeroing it makes every
                    // later stack index address the wrong slot. Same width
                    // contract as `read_field` / `write_field`.
                    let slot = unsafe { vable_ptr.add(array.field_offset) as *mut usize };
                    let value = unsafe { *slot };
                    if value != 0 && majit_gc::gc_owns_object(value) {
                        let current = majit_gc::gc_current_object_address(value);
                        if current != value {
                            unsafe {
                                *slot = current;
                            }
                        }
                        if array.array_type_id != 0 && majit_gc::gc_is_nursery_object(current) {
                            let type_id =
                                unsafe { (*majit_gc::header::header_of(current)).type_id() };
                            if type_id != array.array_type_id {
                                continue;
                            }
                        }
                    }
                    // One pointer-wide slot. `walk_resume_ref_roots` reinterprets
                    // each entry as `&mut GcRef`, which is that same width, so a
                    // one-element slice names exactly this field.
                    unsafe {
                        majit_gc::shadow_stack::push_resume_ref_roots(
                            std::slice::from_raw_parts_mut(slot as *mut i64, 1),
                        );
                    }
                }
                VableArrayStorage::EmbeddedArray { .. } | VableArrayStorage::RustVec { .. } => {}
            }
        }
    }

    fn push_resume_ref_roots_for_value(&self, value: i64) {
        VirtualizableInfo::push_resume_ref_roots_for_value(self, value);
    }

    fn push_resume_ref_roots_for_registers(&self, registers_r: &[i64]) {
        VirtualizableInfo::push_resume_ref_roots_for_registers(self, registers_r);
    }

    /// virtualizable.py write_from_resume_data_partial
    fn write_from_resume_data_partial(
        &self,
        virtualizable: i64,
        reader: &mut crate::resume::ResumeDataDirectReader,
    ) {
        let vable_ptr = virtualizable as *mut u8;
        // virtualizable.py:131-133: ALL static fields.  Route through
        // `write_field` so the resume bits are specialized back to the field
        // type (`f64::from_bits` for `Float`); a raw `i64` write would store a
        // float's bit pattern as an integer and corrupt the field.
        //
        // The reader is a cursor over one shared stream, so the item COUNT this
        // function consumes is what keeps every later section (vrefs, then the
        // frames) aligned — and `get_total_size` right below already promises
        // that count to `consume_vable_info`'s `== vable_size - 1` assert.  It
        // counts `static_fields.len()` whether or not there is a virtualizable
        // to read the array lengths off, so the static items must be consumed
        // unconditionally too; skipping the loop on a null identity leaves them
        // in the stream to be re-read as somebody else's values.  A null gets
        // no write — upstream has no such case at all, `cast_gcref_to_vtype`
        // hands `setattr` a null and it crashes — but the read still happens.
        for (field_index, field) in self.static_fields.iter().enumerate() {
            let value = reader.next_value_of_type(field.field_type);
            if !vable_ptr.is_null() {
                unsafe {
                    self.write_field(vable_ptr, field_index, value);
                }
            }
        }
        if vable_ptr.is_null() {
            // Matches `get_total_size`, which adds no array length without a
            // virtualizable to measure: there are no array items encoded.
            return;
        }
        // virtualizable.py:134-137: array items
        for array in &self.array_fields {
            let arr_len = unsafe { bhimpl_arraylen_vable(vable_ptr as *const u8, array) };
            // `lst = getattr(virtualizable, ARRAYFIELD)` is bound outside the
            // item loop upstream, where the GC transform roots it and forwards
            // it across whatever the reader does.  A bare base pointer is not
            // such a root, and a `Ref` item's reader materializes the guard's
            // virtuals, which allocates and can move the array underneath it.
            // So the base is bound once only where the reader cannot allocate,
            // and a `Ref` array resolves per item.
            let base = (array.item_type != Type::Ref)
                .then(|| unsafe { vable_array_write_base(vable_ptr, array) });
            for j in 0..arr_len {
                let value = reader.next_value_of_type(array.item_type);
                let (data_ptr, owner_ptr) =
                    base.unwrap_or_else(|| unsafe { vable_array_write_base(vable_ptr, array) });
                unsafe {
                    vable_write_array_item_at(vable_ptr, array, data_ptr, owner_ptr, j, value);
                }
            }
        }
    }
}

/// Read the length of a virtualizable array field.
/// blackhole.py bhimpl_arraylen_vable parity.
pub(crate) unsafe fn bhimpl_arraylen_vable(vable_ptr: *const u8, array: &VableArrayInfo) -> usize {
    unsafe {
        match array.storage {
            VableArrayStorage::EmbeddedArray { .. } => {
                // Pointer to container struct: deref then read length
                let container = *(vable_ptr.add(array.field_offset) as *const *const u8);
                *(container.add(array.length_offset) as *const usize)
            }
            VableArrayStorage::DirectPointer => {
                let arr_ptr = *(vable_ptr.add(array.field_offset) as *const *const u8);
                if arr_ptr.is_null() {
                    0
                } else {
                    *(arr_ptr.add(array.length_offset) as *const usize)
                }
            }
            VableArrayStorage::RustVec { len_fn, .. } => len_fn(vable_ptr),
        }
    }
}

/// Address of item 0 of a virtualizable array field.
///
/// The three storage kinds reach the items through different indirections, and
/// both the item read and the item write below resolved that separately. They
/// now share this, so an item access and a whole-array access can never
/// disagree about where the items start.
///
/// Null when the owning pointer is null; callers must keep treating that as
/// "no items", not as address zero.
pub(crate) unsafe fn bhimpl_arraybase_vable(
    vable_ptr: *const u8,
    array: &VableArrayInfo,
) -> *const u8 {
    unsafe {
        match array.storage {
            VableArrayStorage::EmbeddedArray { ptr_offset } => {
                let container = *(vable_ptr.add(array.field_offset) as *const *const u8);
                // Offsetting a null pointer is undefined behaviour even when
                // the caller is about to reject the result: `add` requires the
                // pointer to stay inside one allocation.  Hand the null back
                // so `vable_read_array_item` / `vable_write_array_item` fail
                // on their own null assertion, which names the array.
                if container.is_null() {
                    return std::ptr::null();
                }
                *(container.add(ptr_offset) as *const *const u8)
            }
            VableArrayStorage::DirectPointer => {
                let arr_ptr = *(vable_ptr.add(array.field_offset) as *const *const u8);
                if arr_ptr.is_null() {
                    return std::ptr::null();
                }
                arr_ptr.add(array.items_offset)
            }
            VableArrayStorage::RustVec { data_ptr_fn, .. } => {
                data_ptr_fn(vable_ptr as *mut u8) as *const u8
            }
        }
    }
}

/// Read a value from a virtualizable array item.
/// blackhole.py:1374-1387 bhimpl_getarrayitem_vable_* parity.
pub(crate) unsafe fn vable_read_array_item(
    vable_ptr: *const u8,
    array: &VableArrayInfo,
    index: i64,
) -> i64 {
    unsafe {
        // Stride from the field's array descriptor: a pointer array is
        // `size_of::<usize>()` (4 bytes on wasm32) while an `i64` payload
        // array is a fixed 8, regardless of word width.
        let item_size = array.item_size;
        // `bhimpl_getarrayitem_vable_*` (`blackhole.py:1374-1387`) takes the
        // index with argcode `i` and hands it to `bh_getarrayitem_gc_*`
        // SIGNED.  The operand is an interpreter register, so a stale or
        // clobbered one arrives here as any `i64`; casting it to `usize` turns
        // a negative one into a huge offset and `data_ptr.add` then names an
        // address gigabytes away, which reads as a value of `item_type` — for
        // a `Ref` array, a pointer the caller dereferences.  Refuse it where
        // it can still be attributed to this array.
        assert!(
            index >= 0,
            "vable_read_array_item: negative index {index} into virtualizable \
             array {:?} (item_type {:?})",
            array.name,
            array.item_type,
        );
        let index = index as usize;
        let data_ptr = bhimpl_arraybase_vable(vable_ptr, array);
        // Upstream reaches the items through `bh_getfield_gc_r` and then
        // `bh_getarrayitem_gc_*`, and a null array field faults in the second.
        // Answering `0` instead hands a `Ref` read a null the caller
        // dereferences somewhere else, and an `Int` read a plausible zero that
        // never surfaces at all.
        assert!(
            !data_ptr.is_null(),
            "vable_read_array_item: virtualizable array {:?} is null \
             (item_type {:?}, index {index})",
            array.name,
            array.item_type,
        );
        // An index past the end reads whatever follows the payload and hands
        // it back as a value of `item_type`, so the fault lands in whatever
        // consumes the result rather than here.
        if crate::jit_strict_mode() {
            let len = bhimpl_arraylen_vable(vable_ptr, array);
            assert!(
                index < len,
                "vable_read_array_item: index {index} is past the end of \
                 virtualizable array {:?} (len {len}, item_type {:?})",
                array.name,
                array.item_type,
            );
        }
        let src = data_ptr.add(index * item_size);
        match array.item_type {
            // `llmodel.py read_ref_at_mem` — one machine word.
            Type::Ref => *(src as *const usize) as i64,
            // `llmodel.py read_float_at_mem` — the f64 bit pattern, which the
            // caller re-reads as one.
            Type::Float => std::ptr::read(src as *const i64),
            // `bh_getarrayitem_gc_i` dispatches on `(item_size,
            // is_item_signed)` (`majit-backend` `model.rs`); reading a
            // narrower item as a full word takes the bytes that follow it.
            Type::Int | Type::Void => match (item_size, array.item_signed) {
                (8, true) => *(src as *const i64),
                (8, false) => *(src as *const u64) as i64,
                (4, true) => *(src as *const i32) as i64,
                (4, false) => *(src as *const u32) as i64,
                (2, true) => *(src as *const i16) as i64,
                (2, false) => *(src as *const u16) as i64,
                (1, true) => *(src as *const i8) as i64,
                (1, false) => *(src as *const u8) as i64,
                // `llmodel.py:478 else: raise NotImplementedError`.
                _ => panic!(
                    "vable_read_array_item: virtualizable array {:?} has no \
                     integer load for item_size {item_size} (signed {})",
                    array.name, array.item_signed,
                ),
            },
        }
    }
}

/// The two pointers an item write needs: where the items start, and which
/// block the write barrier names.
///
/// `owner_ptr` is the block base the GC would know, i.e. before the items
/// offset — the barrier argument.  `data_ptr` is items-adjusted and is not a
/// valid object address, so the two only coincide where there is no items
/// offset to undo.
pub(crate) unsafe fn vable_array_write_base(
    vable_ptr: *mut u8,
    array: &VableArrayInfo,
) -> (*mut u8, *mut u8) {
    unsafe {
        let data_ptr = bhimpl_arraybase_vable(vable_ptr, array) as *mut u8;
        let owner_ptr = match array.storage {
            VableArrayStorage::EmbeddedArray { .. } => data_ptr,
            VableArrayStorage::DirectPointer => {
                *(vable_ptr.add(array.field_offset) as *const *mut u8)
            }
            VableArrayStorage::RustVec { .. } => std::ptr::null_mut(),
        };
        (data_ptr, owner_ptr)
    }
}

/// Write a value to a virtualizable array item.
/// blackhole.py:1390-1403 bhimpl_setarrayitem_vable_* parity.
pub(crate) unsafe fn vable_write_array_item(
    vable_ptr: *mut u8,
    array: &VableArrayInfo,
    index: i64,
    value: i64,
) {
    unsafe {
        // `bhimpl_setarrayitem_vable_*` (`blackhole.py:1390-1403`) takes the
        // index with argcode `i` and hands it to `bh_setarrayitem_gc_*`
        // SIGNED, exactly as the read path does.  Casting a clobbered
        // register to `usize` turns a negative index into a huge offset and
        // the store then lands gigabytes past the array.
        assert!(
            index >= 0,
            "vable_write_array_item: negative index {index} into virtualizable \
             array {:?} (item_type {:?})",
            array.name,
            array.item_type,
        );
        let (data_ptr, owner_ptr) = vable_array_write_base(vable_ptr, array);
        // Upstream reaches the items through `bh_getfield_gc_r` and then
        // `bh_setarrayitem_gc_*`, and a null array field faults in the second.
        // Dropping the store instead leaves the item holding whatever it held
        // before, and the mismatch surfaces at some later read with nothing
        // naming this array.
        assert!(
            !data_ptr.is_null(),
            "vable_write_array_item: virtualizable array {:?} is null \
             (item_type {:?}, index {index})",
            array.name,
            array.item_type,
        );
        if crate::jit_strict_mode() {
            let len = bhimpl_arraylen_vable(vable_ptr as *const u8, array);
            assert!(
                (index as usize) < len,
                "vable_write_array_item: index {index} is past the end of \
                 virtualizable array {:?} (len {len}, item_type {:?})",
                array.name,
                array.item_type,
            );
        }
        vable_write_array_item_at(vable_ptr, array, data_ptr, owner_ptr, index as usize, value);
    }
}

/// Write one item of a virtualizable array whose base is already resolved.
///
/// Split out of [`vable_write_array_item`] so a caller walking a whole array
/// can bind the base once, the way `virtualizable.py:134-137` binds `lst`
/// outside its item loop.  Resolving it per item costs an indirect call for
/// `RustVec` storage and two loads for the others.
pub(crate) unsafe fn vable_write_array_item_at(
    vable_ptr: *mut u8,
    array: &VableArrayInfo,
    data_ptr: *mut u8,
    owner_ptr: *mut u8,
    index: usize,
    value: i64,
) {
    unsafe {
        // Stride from the field's array descriptor: a pointer array is
        // `size_of::<usize>()` (4 bytes on wasm32) while an `i64` payload
        // array is a fixed 8, regardless of word width.
        let item_size = array.item_size;
        if !data_ptr.is_null() {
            let dest = data_ptr.add(index * item_size);
            if array.item_type == Type::Ref {
                std::ptr::write(dest as *mut usize, value as usize);
                // `llmodel.py write_ref_at_mem` — "the write barrier is
                // implied above" — is what every blackhole ref store funnels
                // through upstream.  The stored ref can be nursery-young while
                // the array and its owning frame are old-gen, so arm whichever
                // side the collector owns, exactly as the sibling
                // `VirtualizableInfo::write_array_item` already does.
                if value != 0 {
                    if majit_gc::gc_owns_object(owner_ptr as usize) {
                        majit_gc::gc_write_barrier(majit_ir::GcRef(owner_ptr as usize));
                    } else if majit_gc::gc_owns_object(vable_ptr as usize) {
                        majit_gc::gc_write_barrier(majit_ir::GcRef(vable_ptr as usize));
                    }
                }
            } else if array.item_type == Type::Float {
                // `llmodel.py write_float_at_mem` — the f64 bit pattern, which
                // the caller handed over as one.
                std::ptr::write(dest as *mut i64, value);
            } else {
                // `bh_setarrayitem_gc_i` stores at the descriptor's width
                // (`majit-backend` `model.rs`); a fixed 8-byte store into a
                // narrower item overwrites the items that follow it.
                match item_size {
                    8 => std::ptr::write(dest as *mut i64, value),
                    4 => std::ptr::write(dest as *mut i32, value as i32),
                    2 => std::ptr::write(dest as *mut i16, value as i16),
                    1 => std::ptr::write(dest as *mut i8, value as i8),
                    // `llmodel.py:478 else: raise NotImplementedError`.
                    _ => panic!(
                        "vable_write_array_item: virtualizable array {:?} has \
                         no integer store for item_size {item_size}",
                        array.name,
                    ),
                }
            }
        }
    }
}

/// virtualizable.py clear_vable_token, blackhole context.
///
/// `virtualizable.py` is `if virtualizable.vable_token: force_now(...)`
/// followed by `assert not virtualizable.vable_token`, and `force_now`
/// (`:248-260`) splits on the token:
///
///   * `TOKEN_TRACING_RESCALL` — a marker naming no register copy, so clearing
///     it is the whole of the force for that arm.
///   * anything else non-zero — the address of a live JIT frame, handed to
///     `ResumeGuardForcedDescr.force_now`, which writes the compiled
///     activation's registers back into the frame.
///
/// This used to clear both arms alike, on the argument that an Active token
/// cannot reach the blackhole because resume has already materialized every
/// field. The sibling helper on the compiled side (`frame_layout.rs`
/// `pyre_clear_vable_token`) carried the same argument, backed by a census —
/// "recorded 183 times and invoked 0 times over 462 fixtures" — and that
/// census has since stopped holding, so the arm is live there. An argument of
/// that shape is not load-bearing enough to drop a register write-back on:
/// when it fails the field reads stale and nothing reports it.
///
/// So the Active arm now goes through the host's force helper, which is the
/// very function `emit_force_virtualizable` compiles a `COND_CALL` to
/// (`trace_ctx.rs`), reached here by address instead of through compiled code.
///
/// # Safety
/// `obj_ptr` must point to a valid virtualizable object.
pub(crate) unsafe fn bh_clear_vable_token(vinfo: &VirtualizableInfo, obj_ptr: *mut u8) {
    // A machine with no real `vable_token` field (`has_vable_token`) keeps an
    // inert token protocol: writing to offset 0 would clobber the struct's
    // first live field (e.g. a `Vec`'s data pointer).
    if !vinfo.has_vable_token() {
        return;
    }
    unsafe {
        let token_ptr = obj_ptr.add(vinfo.token_offset) as *mut usize;
        let token = *token_ptr;
        if token == 0 {
            return;
        }
        if token == token_tracing_rescall() as usize {
            // virtualizable.py:250-255: the values are already correct during
            // tracing; the marker only tells the tracer this one escaped.
            *token_ptr = 0;
            return;
        }
        let Some(clear_vable_ptr) = vinfo.clear_vable_ptr else {
            // A machine that registered no force helper has no compiled
            // activation to write back either: `emit_force_virtualizable`
            // `expect`s this same field, so a trace that could have parked an
            // Active token here could not have been built.  Clear it, matching
            // the post-state `force_now` guarantees.
            *token_ptr = 0;
            return;
        };
        // `make_clear_vable_descr` declares `[Ref] -> Void` and
        // `frame_layout.rs` registers a function taking that word as `i64`;
        // spelling the pointee any other way mismatches the wasm32 signature
        // and traps on call.
        let force: unsafe extern "C" fn(i64) = std::mem::transmute(clear_vable_ptr);
        force(obj_ptr as i64);
        assert_eq!(
            *token_ptr, 0,
            "virtualizable.py:222 — force_now must leave TOKEN_NONE behind"
        );
    }
}

#[cfg(test)]
mod opt1_rustvec_abi_roundtrip {
    use super::*;

    // Mirrors examples/tl `TlState { stackpos: i64, stack: Vec<i64> }`: a
    // Rust `Vec<i64>` embedded by value, the layout `RustVec` storage targets.
    #[repr(C)]
    struct FakeState {
        stackpos: i64,
        stack: Vec<i64>,
    }

    fn fake_stack_data_ptr(p: *mut u8) -> *mut i64 {
        unsafe { (*(p as *mut FakeState)).stack.as_mut_ptr() }
    }
    fn fake_stack_len(p: *const u8) -> usize {
        unsafe { (*(p as *const FakeState)).stack.len() }
    }

    fn build_info() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_field(
            "stackpos",
            Type::Int,
            std::mem::offset_of!(FakeState, stackpos),
        );
        let descr = majit_ir::descr::make_array_descr(0, std::mem::size_of::<i64>(), Type::Int);
        info.add_rust_vec_array_field(
            "stack",
            Type::Int,
            std::mem::offset_of!(FakeState, stack),
            fake_stack_data_ptr,
            fake_stack_len,
            descr,
        );
        info
    }

    #[test]
    fn arraylen_reads_live_vec_len() {
        let mut s = FakeState {
            stackpos: 0,
            stack: vec![10, 20, 30],
        };
        let info = build_info();
        let p = (&mut s as *mut FakeState) as *const u8;
        let len = unsafe { bhimpl_arraylen_vable(p, &info.array_fields[0]) };
        assert_eq!(len, 3);
    }

    #[test]
    fn read_write_roundtrip_through_vec_data() {
        let mut s = FakeState {
            stackpos: 0,
            stack: vec![0, 0, 0, 0],
        };
        let info = build_info();
        let p = (&mut s as *mut FakeState) as *mut u8;
        unsafe {
            vable_write_array_item(p, &info.array_fields[0], 0, 111);
            vable_write_array_item(p, &info.array_fields[0], 3, 444);
        }
        assert_eq!(s.stack, vec![111, 0, 0, 444]);
        let v0 = unsafe { vable_read_array_item(p as *const u8, &info.array_fields[0], 0) };
        let v3 = unsafe { vable_read_array_item(p as *const u8, &info.array_fields[0], 3) };
        assert_eq!((v0, v3), (111, 444));
    }

    #[test]
    fn flat_index_and_total_size_match_consume_assertion() {
        let info = build_info();
        // add_field sets num_static_extra_boxes = static_fields.len().
        assert_eq!(info.num_static_extra_boxes, 1);
        let lengths = [4usize];
        // Flat layout [stackpos, stack[0..4]]; identity is appended separately
        // by init_virtualizable_boxes.
        assert_eq!(info.get_index_in_array(0, 0, &lengths), 1);
        assert_eq!(info.get_index_in_array(0, 3, &lengths), 4);
        // consume_vable_info asserts get_total_size(slice) == vable_size - 1.
        assert_eq!(info.get_total_size(&lengths), 1 + 4);
    }
}

#[cfg(test)]
mod bh_clear_vable_token_inert_token_protocol {
    use super::*;

    // Two words: `first` sits at offset 0 (a live field on a stack-resident
    // state struct — e.g. a `Vec`'s data pointer), `token` at a nonzero offset.
    #[repr(C)]
    struct TokenProbe {
        first: usize,
        token: usize,
    }

    // A state-field machine builds `VirtualizableInfo::without_vable_token()`:
    // there is no heap `vable_token` to clear. Clearing must be inert — writing
    // to offset 0 would clobber the live first field. This is the guard that lets
    // the overflow-deopt blackhole run a `[int; virt]` vable op without corrupting
    // the state struct.
    #[test]
    fn clear_is_inert_when_the_machine_has_no_vable_token() {
        let mut probe = TokenProbe {
            first: 0xDEAD_BEEF,
            token: 0x1234,
        };
        let info = VirtualizableInfo::without_vable_token();
        let p = (&mut probe as *mut TokenProbe) as *mut u8;
        unsafe { bh_clear_vable_token(&info, p) };
        assert_eq!(probe.first, 0xDEAD_BEEF, "offset-0 field must be untouched");
        assert_eq!(probe.token, 0x1234, "unrelated word must be untouched");
    }

    // A real GC virtualizable keeps `token_offset > 0` (offset 0 is its type
    // pointer). Its live token IS cleared, and the non-inert path is unchanged.
    #[test]
    fn clear_zeroes_a_real_heap_token_at_nonzero_offset() {
        let mut probe = TokenProbe {
            first: 0xDEAD_BEEF,
            token: 0x1234,
        };
        let token_off = std::mem::offset_of!(TokenProbe, token);
        assert_ne!(token_off, 0, "a real vable_token never lands at offset 0");
        let info = VirtualizableInfo::new(token_off);
        let p = (&mut probe as *mut TokenProbe) as *mut u8;
        unsafe { bh_clear_vable_token(&info, p) };
        assert_eq!(probe.token, 0, "a nonzero heap token must be cleared");
        assert_eq!(
            probe.first, 0xDEAD_BEEF,
            "the type-pointer word is untouched"
        );
    }
}
