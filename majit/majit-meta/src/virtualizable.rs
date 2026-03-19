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

use majit_ir::descr::make_array_descr;
use majit_ir::{DescrRef, Type, make_field_descr};

/// Sentinel value for TOKEN_TRACING_RESCALL.
///
/// When the token equals this value, it means JIT tracing is active and
/// a residual call is in progress. If the callee touches the virtualizable,
/// it will force the token and clear it.
///
/// Any non-zero, non-RESCALL value is an active JIT frame pointer.
pub const TOKEN_TRACING_RESCALL: u64 = u64::MAX;

/// Token states for virtualizable objects.
///
/// TOKEN_NONE (0): not in JIT.
/// TOKEN_TRACING_RESCALL (u64::MAX): tracing + residual call in progress.
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
            TOKEN_TRACING_RESCALL => VableToken::TracingRescall,
            other => VableToken::Active(other),
        }
    }

    /// Encode the enum as a raw u64 value.
    pub fn to_raw(self) -> u64 {
        match self {
            VableToken::None => 0,
            VableToken::TracingRescall => TOKEN_TRACING_RESCALL,
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
    /// Byte offset of the array pointer in the heap object.
    pub field_offset: usize,
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
    /// The frame field stores a raw pointer to an array object.
    DirectPointer,
    /// The frame field stores an embedded array container with a separate
    /// data pointer and length, e.g. `PyObjectArray`.
    EmbeddedArray { ptr_offset: usize },
}

/// Complete description of a virtualizable type.
///
/// Mirrors RPython's `VirtualizableInfo` class from `virtualizable.py`.
///
/// This tells the JIT how to read/write the virtualizable's fields
/// from both heap and JIT representations.
#[derive(Debug, Clone)]
pub struct VirtualizableInfo {
    /// Static (scalar) fields on the virtualizable.
    pub static_fields: Vec<VableFieldInfo>,
    /// Array fields on the virtualizable (e.g., `locals_w`).
    pub array_fields: Vec<VableArrayInfo>,
    /// Offset of the `vable_token` field in the heap object.
    pub token_offset: usize,
    /// Total number of "boxes" (field slots + array element slots)
    /// that the JIT needs to save/restore for this virtualizable.
    ///
    /// For `n` static fields and arrays of sizes `s1, s2, ...`:
    /// `num_boxes = n + s1 + s2 + ...`
    /// (array sizes are known at trace time after a promote).
    pub num_static_extra_boxes: usize,
}

impl VirtualizableInfo {
    /// Create a new VirtualizableInfo.
    pub fn new(token_offset: usize) -> Self {
        VirtualizableInfo {
            static_fields: Vec::new(),
            array_fields: Vec::new(),
            token_offset,
            num_static_extra_boxes: 0,
        }
    }

    /// Add a static field.
    pub fn add_field(&mut self, name: impl Into<String>, field_type: Type, offset: usize) {
        self.static_fields.push(VableFieldInfo {
            name: name.into(),
            field_type,
            offset,
            is_immutable: false,
        });
        self.num_static_extra_boxes = self.static_fields.len();
    }

    /// Add an array field with default layout (no header — items start at offset 0).
    ///
    /// Use `add_array_field_with_layout` for arrays with a length header.
    /// RPython uses translator-known descriptors for array layout, not defaults.
    pub fn add_array_field(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
    ) {
        self.add_array_field_with_layout(name, item_type, field_offset, 0, 0);
    }

    /// Add an array field with explicit layout offsets.
    pub fn add_array_field_with_layout(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
        length_offset: usize,
        items_offset: usize,
    ) {
        self.array_fields.push(VableArrayInfo {
            name: name.into(),
            item_type,
            field_offset,
            storage: VableArrayStorage::DirectPointer,
            length_offset,
            items_offset,
        });
    }

    /// Add an embedded array field with explicit pointer/length/data layout.
    ///
    /// This matches Rust containers embedded by value inside the virtualizable
    /// object, where the logical data pointer and length live in the container
    /// rather than in a separate heap array header.
    pub fn add_embedded_array_field_with_layout(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
        ptr_offset: usize,
        length_offset: usize,
        items_offset: usize,
    ) {
        self.array_fields.push(VableArrayInfo {
            name: name.into(),
            item_type,
            field_offset,
            storage: VableArrayStorage::EmbeddedArray { ptr_offset },
            length_offset,
            items_offset,
        });
    }

    /// Total number of static fields.
    pub fn num_fields(&self) -> usize {
        self.static_fields.len()
    }

    /// Total number of array fields.
    pub fn num_arrays(&self) -> usize {
        self.array_fields.len()
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
        let token_ptr = obj_ptr.add(self.token_offset) as *mut u64;
        assert_eq!(*token_ptr, 0, "token should be NONE before residual call");
        *token_ptr = TOKEN_TRACING_RESCALL;
    }

    /// Check after residual call whether the virtualizable was forced.
    ///
    /// Returns `true` if forced (token was cleared by the callee).
    /// Returns `false` if not forced (token is still TRACING_RESCALL; clear it).
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn tracing_after_residual_call(&self, obj_ptr: *mut u8) -> bool {
        let token_ptr = obj_ptr.add(self.token_offset) as *mut u64;
        if *token_ptr != 0 {
            // Not forced — still TOKEN_TRACING_RESCALL
            assert_eq!(*token_ptr, TOKEN_TRACING_RESCALL);
            *token_ptr = 0; // Clear back to TOKEN_NONE
            false
        } else {
            // Was forced — token was cleared by the force path
            true
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
        let token_ptr = obj_ptr.add(self.token_offset) as *mut u64;
        let token = *token_ptr;
        if token == TOKEN_TRACING_RESCALL {
            // During tracing — just clear the marker
            *token_ptr = 0;
        } else if token != 0 {
            // Active JIT frame — force it, then verify it cleared the token
            force_fn(token);
            assert_eq!(*token_ptr, 0, "force_fn should have cleared the token");
        }
    }

    /// Read the current token state from the heap.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_token(&self, obj_ptr: *const u8) -> VableToken {
        let token_ptr = obj_ptr.add(self.token_offset) as *const u64;
        VableToken::from_raw(*token_ptr)
    }

    /// Write a token state to the heap.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_token(&self, obj_ptr: *mut u8, token: VableToken) {
        let token_ptr = obj_ptr.add(self.token_offset) as *mut u64;
        *token_ptr = token.to_raw();
    }

    /// RPython parity surface: descriptor list for static fields.
    pub fn static_field_descrs(&self) -> &[VableFieldInfo] {
        &self.static_fields
    }

    /// RPython parity surface: descriptor list for array fields.
    pub fn array_field_descrs(&self) -> &[VableArrayInfo] {
        &self.array_fields
    }

    /// RPython equivalent: `vinfo.static_field_by_descrs[descr]`.
    pub fn static_field_by_descr_offset(&self, offset: usize) -> Option<(usize, &VableFieldInfo)> {
        self.static_fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.offset == offset)
    }

    /// RPython equivalent: `vinfo.array_field_by_descrs[descr]`.
    pub fn array_field_by_descr_offset(&self, offset: usize) -> Option<(usize, &VableArrayInfo)> {
        self.array_fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.field_offset == offset)
    }

    /// RPython parity surface: reset the virtualizable token to TOKEN_NONE.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn reset_vable_token(&self, obj_ptr: *mut u8) {
        self.write_token(obj_ptr, VableToken::None);
    }

    /// RPython parity surface: reset token from a GCREF/object pointer path.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn reset_token_gcref(&self, obj_ptr: *mut u8) {
        self.reset_vable_token(obj_ptr);
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
        if !matches!(self.read_token(obj_ptr), VableToken::None) {
            self.force_now(obj_ptr, force_fn);
        }
    }

    /// RPython parity surface: clear the virtualizable token, forcing first
    /// if JIT state is still attached.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn clear_vable_token(&self, obj_ptr: *mut u8, force_fn: impl FnOnce(u64)) {
        self.force_virtualizable_if_necessary(obj_ptr, force_fn);
        assert!(
            matches!(self.read_token(obj_ptr), VableToken::None),
            "clear_vable_token must leave TOKEN_NONE"
        );
    }

    /// Convert to optimizer-level config (byte offsets only).
    /// Bridges the descriptor-driven model (majit-meta) with the
    /// optimizer's offset-based tracking (majit-opt).
    pub fn to_optimizer_config(&self) -> majit_opt::virtualize::VirtualizableConfig {
        majit_opt::virtualize::VirtualizableConfig {
            static_field_offsets: self.static_fields.iter().map(|f| f.offset).collect(),
            static_field_types: self.static_fields.iter().map(|f| f.field_type).collect(),
            array_field_offsets: self.array_fields.iter().map(|a| a.field_offset).collect(),
            array_item_types: self.array_fields.iter().map(|a| a.item_type).collect(),
        }
    }

    /// Get the index of a static field by its descriptor offset.
    ///
    /// RPython equivalent: `vinfo.static_field_by_descrs[descr]`
    pub fn static_field_index(&self, offset: usize) -> Option<usize> {
        self.static_field_by_descr_offset(offset)
            .map(|(idx, _)| idx)
    }

    /// Get the index of a static field by name.
    pub fn static_field_index_by_name(&self, name: &str) -> Option<usize> {
        self.static_fields.iter().position(|f| f.name == name)
    }

    /// Get the index of an array field by its field offset.
    ///
    /// RPython equivalent: `vinfo.array_field_by_descrs[descr]`
    pub fn array_field_index(&self, field_offset: usize) -> Option<usize> {
        self.array_field_by_descr_offset(field_offset)
            .map(|(idx, _)| idx)
    }

    /// Get the index of an array field by name.
    pub fn array_field_index_by_name(&self, name: &str) -> Option<usize> {
        self.array_fields.iter().position(|a| a.name == name)
    }

    /// Descriptor for a static virtualizable field.
    pub fn static_field_descr(&self, field_index: usize) -> DescrRef {
        let field = &self.static_fields[field_index];
        make_field_descr(
            field.offset,
            item_size_for_type(field.field_type),
            field.field_type,
            field.field_type != Type::Ref,
        )
    }

    /// Descriptor for the token field on the virtualizable object.
    pub fn token_field_descr(&self) -> DescrRef {
        make_field_descr(self.token_offset, 8, Type::Int, false)
    }

    /// Descriptor for the field that yields the backing array pointer.
    ///
    /// For embedded-array containers, this is the container's internal data
    /// pointer field, not the container field itself.
    pub fn array_pointer_field_descr(&self, array_index: usize) -> DescrRef {
        let array = &self.array_fields[array_index];
        let offset = match array.storage {
            VableArrayStorage::DirectPointer => array.field_offset,
            VableArrayStorage::EmbeddedArray { ptr_offset } => array.field_offset + ptr_offset,
        };
        make_field_descr(offset, 8, Type::Ref, false)
    }

    /// Descriptor for array element accesses on a virtualizable array field.
    pub fn array_item_descr(&self, array_index: usize) -> DescrRef {
        let array = &self.array_fields[array_index];
        make_array_descr(
            array.items_offset,
            item_size_for_type(array.item_type),
            array.item_type,
        )
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
        for i in 0..array_index {
            idx += array_lengths[i];
        }
        idx + item_index
    }

    /// Check that box array has correct size for given array lengths.
    pub fn check_boxes(&self, boxes: &[i64], array_lengths: &[usize]) -> bool {
        boxes.len() == self.get_total_size(array_lengths)
    }

    // ── RPython virtualizable.py parity: heap I/O via descriptor ──

    /// Read a static field value from the heap object.
    ///
    /// RPython equivalent: `vinfo.read_from_field(virtualizable, field_index)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_field(&self, obj_ptr: *const u8, field_index: usize) -> i64 {
        let field = &self.static_fields[field_index];
        match field.field_type {
            Type::Float => {
                let ptr = obj_ptr.add(field.offset) as *const f64;
                f64::to_bits(*ptr) as i64
            }
            _ => {
                let ptr = obj_ptr.add(field.offset) as *const i64;
                *ptr
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
        let field = &self.static_fields[field_index];
        match field.field_type {
            Type::Float => {
                let ptr = obj_ptr.add(field.offset) as *mut f64;
                *ptr = f64::from_bits(value as u64);
            }
            _ => {
                let ptr = obj_ptr.add(field.offset) as *mut i64;
                *ptr = value;
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
                *(obj_ptr.add(ai.field_offset + ai.length_offset) as *const usize)
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
        let ai = &self.array_fields[array_index];
        let array_ptr = ai.data_ptr(obj_ptr);
        let item_offset = ai.items_offset + item_index * item_size_for_type(ai.item_type);
        match ai.item_type {
            Type::Float => {
                let ptr = array_ptr.add(item_offset) as *const f64;
                f64::to_bits(*ptr) as i64
            }
            _ => {
                let ptr = array_ptr.add(item_offset) as *const i64;
                *ptr
            }
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
        let ai = &self.array_fields[array_index];
        let array_ptr = ai.data_ptr(obj_ptr.cast_const()) as *mut u8;
        let item_offset = ai.items_offset + item_index * item_size_for_type(ai.item_type);
        match ai.item_type {
            Type::Float => {
                let ptr = array_ptr.add(item_offset) as *mut f64;
                *ptr = f64::from_bits(value as u64);
            }
            _ => {
                let ptr = array_ptr.add(item_offset) as *mut i64;
                *ptr = value;
            }
        }
    }

    /// Load all virtualizable boxes from the heap object.
    ///
    /// RPython equivalent: `vinfo.load_list_of_boxes(virtualizable)`
    ///
    /// Returns a flat array: [field0, field1, ..., array0[0], ..., array0[N], ...]
    /// Array lengths are read from the actual object (not from a side-channel).
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn load_list_of_boxes(&self, obj_ptr: *const u8) -> (Vec<i64>, Vec<usize>) {
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

    /// Read only the array lengths from the heap object.
    ///
    /// RPython equivalent: `vinfo.get_array_length(vable, i)` for every array.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_array_lengths_from_heap(&self, obj_ptr: *const u8) -> Vec<usize> {
        self.array_fields
            .iter()
            .enumerate()
            .map(|(index, _)| self.get_array_length(obj_ptr, index))
            .collect()
    }

    /// RPython parity surface: read static boxes only.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn read_boxes(&self, obj_ptr: *const u8) -> Vec<i64> {
        self.static_fields
            .iter()
            .enumerate()
            .map(|(index, _)| self.read_field(obj_ptr, index))
            .collect()
    }

    /// RPython parity surface: write static boxes only.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_boxes(&self, obj_ptr: *mut u8, boxes: &[i64]) {
        for (index, &value) in boxes.iter().enumerate() {
            if index >= self.static_fields.len() {
                break;
            }
            self.write_field(obj_ptr, index, value);
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
        let static_boxes = self.read_boxes(obj_ptr);
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
        self.write_boxes(obj_ptr, static_boxes);
        for (array_index, values) in array_boxes.iter().enumerate() {
            for (item_index, &value) in values.iter().enumerate() {
                self.write_array_item(obj_ptr, array_index, item_index, value);
            }
        }
    }

    /// Write all boxes back to the heap object (force direction).
    ///
    /// RPython equivalent: `vinfo.write_from_resume_data_partial(virtualizable, ...)`
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_boxes_to_heap(
        &self,
        obj_ptr: *mut u8,
        boxes: &[i64],
        array_lengths: &[usize],
    ) {
        // Static fields
        for i in 0..self.static_fields.len() {
            if i < boxes.len() {
                self.write_field(obj_ptr, i, boxes[i]);
            }
        }

        // Array elements
        let mut idx = self.static_fields.len();
        for (ai, &len) in array_lengths.iter().enumerate() {
            for ei in 0..len {
                if idx < boxes.len() {
                    self.write_array_item(obj_ptr, ai, ei, boxes[idx]);
                }
                idx += 1;
            }
        }
    }

    /// RPython equivalent: `vinfo.write_from_resume_data_partial(...)`.
    ///
    /// The boxed data is already split into static and per-array slices, so the
    /// implementation only writes the provided data back and leaves token
    /// handling to the caller.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn write_from_resume_data_partial(
        &self,
        obj_ptr: *mut u8,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) {
        self.write_all_boxes(obj_ptr, static_boxes, array_boxes);
    }

    /// Force virtualizable: write boxes to heap and clear token.
    ///
    /// RPython equivalent: the combined force_now + write_from_resume_data flow.
    ///
    /// # Safety
    /// `obj_ptr` must point to a valid virtualizable object.
    pub unsafe fn force_from_boxes(
        &self,
        obj_ptr: *mut u8,
        boxes: &[i64],
        array_lengths: &[usize],
    ) {
        self.write_boxes_to_heap(obj_ptr, boxes, array_lengths);
        self.reset_vable_token(obj_ptr);
    }
}

impl VableArrayInfo {
    pub fn can_read_length_from_heap(&self) -> bool {
        match self.storage {
            VableArrayStorage::EmbeddedArray { .. } => true,
            VableArrayStorage::DirectPointer => {
                !(self.length_offset == 0 && self.items_offset == 0)
            }
        }
    }

    unsafe fn data_ptr(&self, obj_ptr: *const u8) -> *const u8 {
        match self.storage {
            VableArrayStorage::DirectPointer => {
                *(obj_ptr.add(self.field_offset) as *const *const u8)
            }
            VableArrayStorage::EmbeddedArray { ptr_offset } => {
                *(obj_ptr.add(self.field_offset + ptr_offset) as *const *const u8)
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

/// Writes values back from JIT representation to a virtualizable heap object.
///
/// This is the "force" direction: JIT representation → heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object with the
/// layout described by `info`.
#[cfg(test)]
unsafe fn write_virtualizable_boxes(info: &VirtualizableInfo, obj_ptr: *mut u8, boxes: &[i64]) {
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

/// Reset the vable_token on a virtualizable object (unconditional).
///
/// Sets the token to TOKEN_NONE without forcing. Use `force_virtualizable`
/// or `VirtualizableInfo::force_now` if you need to flush JIT state first.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
#[cfg(test)]
unsafe fn reset_vable_token(info: &VirtualizableInfo, obj_ptr: *mut u8) {
    let token_ptr = obj_ptr.add(info.token_offset) as *mut u64;
    *token_ptr = 0;
}

/// Check if the vable_token is non-null (JIT code may be active).
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
pub unsafe fn is_token_nonnull(info: &VirtualizableInfo, obj_ptr: *const u8) -> bool {
    let token_ptr = obj_ptr.add(info.token_offset) as *const u64;
    *token_ptr != 0
}

/// Force a virtualizable: flush JIT-held values back to the heap.
///
/// Token semantics:
/// - TOKEN_NONE (0): not in JIT, nothing to do.
/// - TOKEN_TRACING_RESCALL (u64::MAX): tracing + residual call, just clear.
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
    info.force_now(obj_ptr, force_fn);
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
    info.read_array_lengths_from_heap(obj_ptr)
}

/// Read a virtualizable's array field contents from the heap.
///
/// The array pointer is read from the virtualizable at `array_info.field_offset`,
/// then `length` elements are read starting at `array_info.items_offset`.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and the array has at least `length` elements.
#[cfg(test)]
unsafe fn read_virtualizable_array(
    array_info: &VableArrayInfo,
    obj_ptr: *const u8,
    length: usize,
) -> Vec<i64> {
    let mut values = Vec::with_capacity(length);
    for i in 0..length {
        let array_ptr = array_info.data_ptr(obj_ptr);
        let item_offset = array_info.items_offset + i * item_size_for_type(array_info.item_type);
        let val = match array_info.item_type {
            Type::Int | Type::Ref => *(array_ptr.add(item_offset) as *const i64),
            Type::Float => f64::to_bits(*(array_ptr.add(item_offset) as *const f64)) as i64,
            Type::Void => 0,
        };
        values.push(val);
    }
    values
}

/// Returns the byte size of a single item for the given type.
fn item_size_for_type(ty: Type) -> usize {
    match ty {
        Type::Int | Type::Ref | Type::Float => 8,
        Type::Void => 0,
    }
}

/// Write values into a virtualizable's array field on the heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and the array has sufficient space.
#[cfg(test)]
unsafe fn write_virtualizable_array(array_info: &VableArrayInfo, obj_ptr: *mut u8, values: &[i64]) {
    for (i, &val) in values.iter().enumerate() {
        let array_ptr = array_info.data_ptr(obj_ptr.cast_const()) as *mut u8;
        let item_offset = array_info.items_offset + i * item_size_for_type(array_info.item_type);
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
    info.read_all_boxes(obj_ptr, array_lengths)
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
    info.write_all_boxes(obj_ptr, static_boxes, array_boxes);
}

/// Generate accessor functions for virtualizable field access from JIT code.
///
/// Returns (getter_ptr, setter_ptr) for each static field.
/// These are function pointers that can be called from JIT-compiled code
/// to read/write virtualizable fields without going through the heap.
///
/// In RPython, this is done by `virtualizable.py`'s `_generate_ACCESS()`.
pub fn generate_field_accessors(_info: &VirtualizableInfo) -> Vec<(FieldGetter, FieldSetter)> {
    // In the final implementation, these would be generated functions
    // that know how to extract fields from the JIT's frame/registers.
    // For now, we provide the offset-based accessors.
    _info
        .static_fields
        .iter()
        .map(|field| {
            let offset = field.offset;
            let getter: FieldGetter = Box::new(move |obj_ptr: *const u8| unsafe {
                let ptr = obj_ptr.add(offset) as *const i64;
                *ptr
            });
            let setter: FieldSetter = Box::new(move |obj_ptr: *mut u8, value: i64| unsafe {
                let ptr = obj_ptr.add(offset) as *mut i64;
                *ptr = value;
            });
            (getter, setter)
        })
        .collect()
}

/// Type aliases for field accessor function pointers.
pub type FieldGetter = Box<dyn Fn(*const u8) -> i64 + Send + Sync>;
pub type FieldSetter = Box<dyn Fn(*mut u8, i64) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtualizable_info_creation() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        info.add_array_field("stack", Type::Int, 24);

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

        assert_eq!(info.static_field_index(8), Some(0));
        assert_eq!(info.static_field_index(16), Some(1));
        assert_eq!(info.static_field_index(24), Some(2));
        assert_eq!(info.static_field_index(999), None);
    }

    #[test]
    fn test_array_field_index_lookup() {
        let mut info = VirtualizableInfo::new(0);
        info.add_array_field("locals", Type::Ref, 32);
        info.add_array_field("stack", Type::Int, 40);

        assert_eq!(info.array_field_index(32), Some(0));
        assert_eq!(info.array_field_index(40), Some(1));
        assert_eq!(info.array_field_index(999), None);
    }

    #[test]
    fn test_get_total_size() {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("x", Type::Int, 8);
        info.add_field("y", Type::Int, 16);
        info.add_array_field("locals", Type::Ref, 24);
        info.add_array_field("stack", Type::Int, 32);

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
        info.add_array_field("locals", Type::Ref, 24);
        info.add_array_field("stack", Type::Int, 32);

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
        info.add_array_field("locals", Type::Ref, 24);

        let lens = &[3usize];
        // total = 2 + 3 = 5
        assert!(info.check_boxes(&[1, 2, 3, 4, 5], lens));
        assert!(!info.check_boxes(&[1, 2, 3], lens));
        assert!(!info.check_boxes(&[1, 2, 3, 4, 5, 6], lens));
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
        info.add_array_field("locals", Type::Ref, 8);
        info.add_array_field_with_layout("stack", Type::Int, 16, 0, 8);

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
        info.add_array_field("locals", Type::Ref, 8);

        let lengths = unsafe { read_array_lengths(&info, obj.as_ptr()) };
        assert_eq!(lengths, vec![0]);
    }

    #[test]
    fn test_read_array_lengths_from_embedded_array_container() {
        #[repr(C)]
        struct InlineArray {
            ptr: *mut i64,
            len: usize,
        }

        #[repr(C)]
        struct Obj {
            token: usize,
            arr: InlineArray,
        }

        let mut backing = vec![10i64, 20, 30];
        let obj = Obj {
            token: 0,
            arr: InlineArray {
                ptr: backing.as_mut_ptr(),
                len: backing.len(),
            },
        };

        let mut info = VirtualizableInfo::new(0);
        info.add_embedded_array_field_with_layout(
            "arr",
            Type::Int,
            std::mem::offset_of!(Obj, arr),
            std::mem::offset_of!(InlineArray, ptr),
            std::mem::offset_of!(InlineArray, len),
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
        info.add_array_field_with_layout("stack", Type::Int, 24, 0, 8);

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
        info.add_array_field_with_layout("data", Type::Int, 8, 16, 24);

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
        info.add_array_field_with_layout("stack", Type::Int, 16, 0, 8);

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
        info.add_array_field_with_layout("locals_w", Type::Ref, 8, 0, 8);

        let (boxes, lengths) = unsafe { info.load_list_of_boxes(obj.as_ptr()) };
        assert_eq!(lengths, vec![3]);
        assert_eq!(boxes, vec![0x1000, 0, 0x2000]);

        let new_array_boxes = vec![vec![0x3000_i64, 0, 0x4000_i64]];
        unsafe {
            info.write_from_resume_data_partial(obj.as_mut_ptr(), &[], &new_array_boxes);
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
        info.add_array_field_with_layout("vals", Type::Int, 24, 0, 8);

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
        info.add_array_field("locals", Type::Ref, 48);
        info.add_array_field("stack", Type::Int, 56);

        let config = info.to_optimizer_config();

        assert_eq!(config.static_field_offsets, vec![8, 24, 40]);
        assert_eq!(
            config.static_field_types,
            vec![Type::Int, Type::Float, Type::Ref]
        );
        assert_eq!(config.array_field_offsets, vec![48, 56]);
        assert_eq!(config.array_item_types, vec![Type::Ref, Type::Int]);
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
        info.add_array_field_with_layout("arr", Type::Int, 16, 0, 8);

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
        info.add_array_field_with_layout("arr", Type::Int, 16, 0, 8);

        let obj = &mut frame as *mut Frame as *mut u8;
        let boxes = vec![42, 100, 200]; // x=42, arr=[100, 200]
        let lengths = vec![2];

        unsafe {
            info.force_from_boxes(obj, &boxes, &lengths);

            assert_eq!(frame.x, 42);
            assert_eq!(arr.items[0], 100);
            assert_eq!(arr.items[1], 200);
            assert_eq!(frame.token, 0); // token cleared
        }
    }

    #[test]
    fn test_vable_token_roundtrip() {
        assert_eq!(VableToken::from_raw(0), VableToken::None);
        assert_eq!(
            VableToken::from_raw(TOKEN_TRACING_RESCALL),
            VableToken::TracingRescall
        );
        assert_eq!(VableToken::from_raw(0xBEEF), VableToken::Active(0xBEEF));

        assert_eq!(VableToken::None.to_raw(), 0);
        assert_eq!(VableToken::TracingRescall.to_raw(), TOKEN_TRACING_RESCALL);
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
            *(obj_ptr as *mut u64) = TOKEN_TRACING_RESCALL;

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
            *(obj_ptr as *mut u64) = TOKEN_TRACING_RESCALL;

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
        info.add_array_field("arr0", Type::Int, 24);
        info.add_array_field("arr1", Type::Int, 32);

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
