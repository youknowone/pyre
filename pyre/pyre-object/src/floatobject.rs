//! W_FloatObject — Python `float` type backed by f64.

use crate::pyobject::*;

/// Python float object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | floatval: f64]`
/// The JIT reads `floatval` via `GetfieldGcF` at `FLOAT_FLOATVAL_OFFSET`.
#[repr(C)]
pub struct W_FloatObject {
    pub ob_header: PyObject,
    pub floatval: f64,
}

/// Field offset of `floatval` within `W_FloatObject`, for JIT field access.
pub const FLOAT_FLOATVAL_OFFSET: usize = std::mem::offset_of!(W_FloatObject, floatval);

/// GC type id assigned to `W_FloatObject` at JitDriver init time.
/// Held as a constant here (rather than runtime-queried) so the
/// allocation hook can reach it without a back-channel.
pub const W_FLOAT_GC_TYPE_ID: u32 = 2;

/// Fixed payload size for `W_FloatObject`, mirroring `info.fixedsize`
/// in `framework.py:811`.
pub const W_FLOAT_OBJECT_SIZE: usize = std::mem::size_of::<W_FloatObject>();

impl crate::lltype::GcType for W_FloatObject {
    const TYPE_ID: u32 = W_FLOAT_GC_TYPE_ID;
    const SIZE: usize = W_FLOAT_OBJECT_SIZE;
}

/// Allocate a new W_FloatObject on the heap.
///
/// Routes through [`crate::lltype::malloc_typed`] (Task #145), the
/// typed unified allocation lowering that mirrors RPython's
/// `lltype.malloc(W_FloatObject)`
/// (`rpython/rtyper/lltypesystem/lltype.py:2192`). PyPy's
/// `pypy/objspace/std/floatobject.py:299 newfloat` produces the
/// same shape: a single allocation call that the GC transform stage
/// eventually rewrites into managed alloc + push/pop_roots
/// (`rpython/memory/gctransform/framework.py:803-853`). The typed
/// variant carries `W_FLOAT_GC_TYPE_ID` and `W_FLOAT_OBJECT_SIZE`
/// via the [`crate::lltype::GcType`] impl so the future managed
/// allocator can read them without a runtime registry lookup,
/// matching `gct_fv_gc_malloc`'s `c_type_id` / `c_size` constants
/// (`framework.py:807-811`).
///
/// Phase 1: `lltype::malloc_typed` is `Box::into_raw`. Future GC
/// integration replaces only that body; this constructor stays
/// unchanged.
pub fn w_float_new(value: f64) -> PyObjectRef {
    crate::lltype::malloc_typed(W_FloatObject {
        ob_header: PyObject {
            ob_type: &FLOAT_TYPE as *const PyType,
            w_class: get_instantiate(&FLOAT_TYPE),
        },
        floatval: value,
    }) as PyObjectRef
}

/// Box a float constant into a heap Python float object.
pub fn box_float_constant(value: f64) -> PyObjectRef {
    w_float_new(value)
}

/// Extract the f64 value from a known W_FloatObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_FloatObject`.
#[inline]
pub unsafe fn w_float_get_value(obj: PyObjectRef) -> f64 {
    unsafe { (*(obj as *const W_FloatObject)).floatval }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_float_new(value_bits: i64) -> i64 {
    let value = f64::from_bits(value_bits as u64);
    w_float_new(value) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    // GC-flavored allocations (`malloc_typed`) are leaked in these
    // tests; `Box::from_raw` is unsound once Phase 2 routes
    // `malloc_typed` through the managed allocator.

    #[test]
    fn test_float_create_and_read() {
        let obj = w_float_new(3.14);
        unsafe {
            assert!(is_float(obj));
            assert!(!is_int(obj));
            assert_eq!(w_float_get_value(obj), 3.14);
        }
    }

    #[test]
    fn test_float_negative() {
        let obj = w_float_new(-2.5);
        unsafe {
            assert_eq!(w_float_get_value(obj), -2.5);
        }
    }

    #[test]
    fn test_box_float_constant_reads_back() {
        let obj = box_float_constant(6.25);
        unsafe {
            assert_eq!(w_float_get_value(obj), 6.25);
        }
    }

    #[test]
    fn test_float_field_offset() {
        assert_eq!(FLOAT_FLOATVAL_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
    }
}
