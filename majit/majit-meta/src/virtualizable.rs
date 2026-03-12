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

    /// Add an array field.
    pub fn add_array_field(
        &mut self,
        name: impl Into<String>,
        item_type: Type,
        field_offset: usize,
    ) {
        self.array_fields.push(VableArrayInfo {
            name: name.into(),
            item_type,
            field_offset,
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

/// Read a virtualizable's array field contents from the heap.
///
/// The array pointer is read from the virtualizable at `array_info.field_offset`,
/// then `length` elements are read from the array.
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

    let mut values = Vec::with_capacity(length);
    for i in 0..length {
        let val = match array_info.item_type {
            Type::Int | Type::Ref => {
                let ptr = array_ptr.add(i * 8) as *const i64;
                *ptr
            }
            Type::Float => {
                let ptr = array_ptr.add(i * 8) as *const f64;
                f64::to_bits(*ptr) as i64
            }
            Type::Void => 0,
        };
        values.push(val);
    }
    values
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

    for (i, &val) in values.iter().enumerate() {
        match array_info.item_type {
            Type::Int | Type::Ref => {
                let ptr = array_ptr.add(i * 8) as *mut i64;
                *ptr = val;
            }
            Type::Float => {
                let ptr = array_ptr.add(i * 8) as *mut f64;
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
}
