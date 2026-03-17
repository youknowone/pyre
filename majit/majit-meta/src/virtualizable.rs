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

use majit_ir::Type;

/// Token states for virtualizable objects.
///
/// Mirrors RPython's TOKEN_NONE and TOKEN_TRACING_RESCALL.
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
    /// Offset of the length field within the array object.
    /// Default: 0 (length stored at start of array).
    pub length_offset: usize,
    /// Offset of the first item within the array object.
    /// Default: 8 (items start after the length field).
    pub items_offset: usize,
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
    pub num_static_fields: usize,
}

impl VirtualizableInfo {
    /// Create a new VirtualizableInfo.
    pub fn new(token_offset: usize) -> Self {
        VirtualizableInfo {
            static_fields: Vec::new(),
            array_fields: Vec::new(),
            token_offset,
            num_static_fields: 0,
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
        self.num_static_fields = self.static_fields.len();
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

    /// Convert to optimizer-level config (byte offsets only).
    /// Bridges the descriptor-driven model (majit-meta) with the
    /// optimizer's offset-based tracking (majit-opt).
    /// RPython equivalent: jtransform → optimizer handoff.
    pub fn to_optimizer_config(&self) -> majit_opt::virtualize::VirtualizableConfig {
        majit_opt::virtualize::VirtualizableConfig {
            static_field_offsets: self.static_fields.iter().map(|f| f.offset).collect(),
            array_field_offsets: self.array_fields.iter().map(|a| a.field_offset).collect(),
        }
    }

    /// Get the index of a static field by its descriptor offset.
    /// RPython equivalent: static_field_by_descrs
    pub fn static_field_index(&self, offset: usize) -> Option<usize> {
        self.static_fields.iter().position(|f| f.offset == offset)
    }

    /// Get the index of an array field by its field offset.
    /// RPython equivalent: array_field_by_descrs
    pub fn array_field_index(&self, field_offset: usize) -> Option<usize> {
        self.array_fields
            .iter()
            .position(|a| a.field_offset == field_offset)
    }

    /// Get total size: number of static fields + sum of all array lengths.
    /// `array_lengths` must have one entry per array field.
    pub fn get_total_size(&self, array_lengths: &[usize]) -> usize {
        self.num_static_fields + array_lengths.iter().sum::<usize>()
    }

    /// Get the index into the flat box array for a specific array element.
    /// `array_index` is the index of the array field, `item_index` is the
    /// element within that array.
    /// RPython equivalent: get_index_in_array
    pub fn get_index_in_array(
        &self,
        array_index: usize,
        item_index: usize,
        array_lengths: &[usize],
    ) -> usize {
        let mut idx = self.num_static_fields;
        for i in 0..array_index {
            idx += array_lengths[i];
        }
        idx + item_index
    }

    /// Returns symbolic descriptions of SETFIELD_GC ops to emit before a
    /// residual call during tracing, flushing JIT state to heap.
    /// RPython equivalent: tracing_before_residual_call
    pub fn tracing_before_residual_call_ops(&self, vable_ref: &str) -> Vec<String> {
        self.static_fields
            .iter()
            .map(|f| format!("SetfieldGc({vable_ref}, field_{})", f.name))
            .collect()
    }

    /// Returns symbolic descriptions of GETFIELD_GC ops to emit after a
    /// residual call during tracing, re-reading fields the callee may have
    /// modified.
    /// RPython equivalent: tracing_after_residual_call
    pub fn tracing_after_residual_call_ops(&self, vable_ref: &str) -> Vec<String> {
        self.static_fields
            .iter()
            .map(|f| format!("GetfieldGc({vable_ref}, field_{})", f.name))
            .collect()
    }

    /// Check that box array has correct size for given array lengths.
    /// RPython equivalent: check_boxes
    pub fn check_boxes(&self, boxes: &[i64], array_lengths: &[usize]) -> bool {
        boxes.len() == self.get_total_size(array_lengths)
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
pub unsafe fn read_virtualizable_boxes(info: &VirtualizableInfo, obj_ptr: *const u8) -> Vec<i64> {
    let mut boxes = Vec::with_capacity(info.num_static_fields);

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
pub unsafe fn write_virtualizable_boxes(info: &VirtualizableInfo, obj_ptr: *mut u8, boxes: &[i64]) {
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

/// Clear the vable_token on a virtualizable object.
///
/// This must be called before non-JIT code accesses the virtualizable.
/// If the token indicates JIT code is active, this triggers a "force"
/// to flush JIT-held values back to the heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
pub unsafe fn clear_vable_token(info: &VirtualizableInfo, obj_ptr: *mut u8) {
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
/// This is the full force sequence:
/// 1. Check if JIT code is active (token non-null and not tracing)
/// 2. If so, force the JIT frame to write back its values
/// 3. Clear the token
///
/// `force_fn` is called if the token indicates active JIT code,
/// receiving the force_token value.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid object.
pub unsafe fn force_virtualizable(
    info: &VirtualizableInfo,
    obj_ptr: *mut u8,
    force_fn: impl FnOnce(u64),
) {
    let token_ptr = obj_ptr.add(info.token_offset) as *mut u64;
    let token_val = *token_ptr;

    if token_val == 0 {
        return; // TOKEN_NONE: not in JIT
    }

    // Token is non-zero: JIT code is active, force it
    force_fn(token_val);

    // Clear the token after forcing
    *token_ptr = 0;
}

/// Read array lengths from a virtualizable heap object.
///
/// For each array field, reads the array pointer, then reads its length
/// from the configured `length_offset` within the array header.
///
/// # Safety
/// The caller must ensure `obj_ptr` points to a valid virtualizable object.
pub unsafe fn read_array_lengths(info: &VirtualizableInfo, obj_ptr: *const u8) -> Vec<usize> {
    info.array_fields
        .iter()
        .map(|array_info| {
            let array_ptr_ptr = obj_ptr.add(array_info.field_offset) as *const *const u8;
            let array_ptr = *array_ptr_ptr;
            if array_ptr.is_null() {
                0
            } else {
                let len_ptr = array_ptr.add(array_info.length_offset) as *const usize;
                *len_ptr
            }
        })
        .collect()
}

/// Read a virtualizable's array field contents from the heap.
///
/// The array pointer is read from the virtualizable at `array_info.field_offset`,
/// then `length` elements are read starting at `array_info.items_offset`.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and the array has at least `length` elements.
pub unsafe fn read_virtualizable_array(
    array_info: &VableArrayInfo,
    obj_ptr: *const u8,
    length: usize,
) -> Vec<i64> {
    let array_ptr_ptr = obj_ptr.add(array_info.field_offset) as *const *const u8;
    let array_ptr = *array_ptr_ptr;
    let item_size = item_size_for_type(array_info.item_type);

    let mut values = Vec::with_capacity(length);
    for i in 0..length {
        let val = match array_info.item_type {
            Type::Int | Type::Ref => {
                let ptr = array_ptr.add(array_info.items_offset + i * item_size) as *const i64;
                *ptr
            }
            Type::Float => {
                let ptr = array_ptr.add(array_info.items_offset + i * item_size) as *const f64;
                f64::to_bits(*ptr) as i64
            }
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
pub unsafe fn write_virtualizable_array(
    array_info: &VableArrayInfo,
    obj_ptr: *mut u8,
    values: &[i64],
) {
    let array_ptr_ptr = obj_ptr.add(array_info.field_offset) as *const *mut u8;
    let array_ptr = *array_ptr_ptr;
    let item_size = item_size_for_type(array_info.item_type);

    for (i, &val) in values.iter().enumerate() {
        match array_info.item_type {
            Type::Int | Type::Ref => {
                let ptr = array_ptr.add(array_info.items_offset + i * item_size) as *mut i64;
                *ptr = val;
            }
            Type::Float => {
                let ptr = array_ptr.add(array_info.items_offset + i * item_size) as *mut f64;
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
pub unsafe fn read_all_virtualizable_boxes(
    info: &VirtualizableInfo,
    obj_ptr: *const u8,
    array_lengths: &[usize],
) -> (Vec<i64>, Vec<Vec<i64>>) {
    let static_boxes = read_virtualizable_boxes(info, obj_ptr);

    let mut array_boxes = Vec::with_capacity(info.array_fields.len());
    for (i, afield) in info.array_fields.iter().enumerate() {
        let length = array_lengths.get(i).copied().unwrap_or(0);
        let values = read_virtualizable_array(afield, obj_ptr, length);
        array_boxes.push(values);
    }

    (static_boxes, array_boxes)
}

/// Write all virtualizable state back to the heap.
///
/// # Safety
/// The caller must ensure `obj_ptr` is valid and arrays have sufficient space.
pub unsafe fn write_all_virtualizable_boxes(
    info: &VirtualizableInfo,
    obj_ptr: *mut u8,
    static_boxes: &[i64],
    array_boxes: &[Vec<i64>],
) {
    write_virtualizable_boxes(info, obj_ptr, static_boxes);

    for (i, afield) in info.array_fields.iter().enumerate() {
        if let Some(values) = array_boxes.get(i) {
            write_virtualizable_array(afield, obj_ptr, values);
        }
    }
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
        assert_eq!(info.num_static_fields, 2);
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

            // Clear it
            clear_vable_token(&info, obj_ptr);
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

        // Simulate JIT setting the token
        unsafe {
            *(obj_ptr as *mut u64) = 0xBEEF;
        }

        let mut force_token_received = 0u64;
        unsafe {
            force_virtualizable(&info, obj_ptr, |token| {
                force_token_received = token;
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
    fn test_force_virtualizable_triggers_callback() {
        // RPython parity: test_force_virtualizable_by_hint
        // Non-zero token → callback receives token value → token cleared.

        let info = VirtualizableInfo::new(0);
        let mut obj = vec![0u8; 8];

        // Set a non-zero token (simulating active JIT)
        unsafe {
            *(obj.as_mut_ptr() as *mut u64) = 0xCAFE_BABE;
        }

        let mut received_token = 0u64;
        unsafe {
            force_virtualizable(&info, obj.as_mut_ptr(), |token| {
                received_token = token;
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
        assert_eq!(config.array_field_offsets, vec![48, 56]);
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
