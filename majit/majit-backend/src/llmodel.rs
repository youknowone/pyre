//! AbstractLLCPU jitframe accessors —
//! `rpython/jit/backend/llsupport/llmodel.py` parity.
//!
//! Upstream these live as methods on `AbstractLLCPU`, invoked as
//! `cpu.get_int_value(deadframe, index)`. In majit there is no
//! `AbstractLLCPU`-equivalent trait with `self`-carried state that a
//! backend would override — every backend shares the same
//! JITFRAME-backed deadframe layout — so the accessors are free
//! functions keyed on a raw `*const JitFrame`.
//!
//! The `AbstractCPU` base class (rpython/jit/backend/model.py:95-133)
//! declares the abstract contract for these accessors; all entries
//! below match those signatures.

use crate::jitframe::{FIRST_ITEM_OFFSET, JitFrame};

/// llmodel.py:412-420 — get_latest_descr.
///
/// Returns the `jf_descr` field, which holds the descr pointer of
/// the last GUARD or FINISH operation executed.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn get_latest_descr(ptr: *const JitFrame) -> usize {
    unsafe { (*ptr).jf_descr }
}

/// Store the `jf_descr` field directly.
///
/// Upstream writes `jf_descr` through generated assembly or through
/// `compile.py` finish-descr injection; this free-function form exists
/// for host-side test / arena runners that bypass the compiled-code
/// write path.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn set_latest_descr(ptr: *mut JitFrame, descr: usize) {
    unsafe {
        (*ptr).jf_descr = descr;
    }
}

/// llmodel.py:440-444 — `get_int_value_direct(deadframe, pos)`.
///
/// Read the `Signed` slot at pre-decoded position `slot` from
/// `jf_frame`.  `slot` is a post-`rd_locs[i]` slot index (i.e.
/// `pos / WORD` in upstream terms), NOT a raw byte offset and NOT
/// the logical fail-arg index.  Upstream's `get_int_value_direct`
/// takes a byte `pos`; majit's `JitFrame::slot_ptr_const` already
/// scales slot * WORD internally, so the pre-WORD-scaled slot is
/// what this accessor expects.
///
/// The logical `get_int_value(deadframe, index)` entry point —
/// which first calls `_decode_pos(deadframe, index)` to translate
/// `index` through `rd_locs[]` — belongs at the dynasm/cranelift
/// layer that owns the `FailDescr` type hierarchy (majit-backend
/// is descr-agnostic, so it cannot fetch `get_latest_descr` into
/// a concrete subclass here).
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_int_value_direct(ptr: *const JitFrame, slot: usize) -> isize {
    unsafe { *JitFrame::slot_ptr_const(ptr, slot) }
}

/// Symmetric setter for `get_int_value_direct`.
///
/// llsupport/llmodel.py does not expose this: compiled code writes
/// `jf_frame[i]` directly. It is retained here for host-side test /
/// arena runners only.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn set_int_value(ptr: *mut JitFrame, slot: usize, value: isize) {
    unsafe {
        *JitFrame::slot_ptr(ptr, slot) = value;
    }
}

/// llmodel.py:449-453 — `get_ref_value_direct(deadframe, pos)`.
///
/// Read the slot at pre-decoded position `slot` as a reference
/// (pointer-sized).  See `get_int_value_direct` for the slot/index
/// distinction.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_ref_value_direct(ptr: *const JitFrame, slot: usize) -> usize {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize;
        *base.add(slot)
    }
}

/// llmodel.py:458-462 — `get_float_value_direct(deadframe, pos)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `slot + 1`
/// trailing array slots.
pub unsafe fn get_float_value_direct(ptr: *const JitFrame, slot: usize) -> u64 {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const u64;
        *base.add(slot)
    }
}

/// llmodel.py:248-251 — get_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn get_savedata_ref(ptr: *const JitFrame) -> usize {
    unsafe { (*ptr).jf_savedata }
}

/// llmodel.py:244-246 — set_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn set_savedata_ref(ptr: *mut JitFrame, value: usize) {
    unsafe {
        (*ptr).jf_savedata = value;
    }
}
