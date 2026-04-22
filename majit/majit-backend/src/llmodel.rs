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
//! Two flavours exist upstream:
//!
//! - `get_*_value_direct(deadframe, pos)` (llmodel.py:441-462) reads
//!   the physical jitframe slot `pos` without any translation.
//! - `get_*_value(deadframe, index)` (llmodel.py:437-457) first calls
//!   `__decode_pos(deadframe, index)` which looks up `descr.rd_locs[index]`
//!   to turn a logical failarg index into a physical slot, then
//!   delegates to `get_*_value_direct`.
//!
//! The free functions in this module mirror the same split.

use crate::jitframe::{FIRST_ITEM_OFFSET, JitFrame};
use majit_ir::FailDescr;

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

/// llmodel.py:441-444 — `get_int_value_direct(deadframe, pos)`.
///
/// Read the physical jf_frame slot at `pos` (a slot index, not a
/// byte offset). Does not apply `__decode_pos`; callers that already
/// have a physical slot (e.g. iterating fail_arg_locs on the host)
/// go through this variant. Upstream takes a byte `pos`; majit's
/// `JitFrame::slot_ptr_const` already scales slot * WORD internally,
/// so `pos` here is pre-WORD-scaled.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `pos + 1`
/// trailing array slots.
pub unsafe fn get_int_value_direct(ptr: *const JitFrame, pos: usize) -> isize {
    unsafe { *JitFrame::slot_ptr_const(ptr, pos) }
}

/// Symmetric setter for `get_int_value_direct`.
///
/// llsupport/llmodel.py does not expose this: compiled code writes
/// `jf_frame[i]` directly. Retained here for host-side test / arena
/// runners only.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `pos + 1`
/// trailing array slots.
pub unsafe fn set_int_value_direct(ptr: *mut JitFrame, pos: usize, value: isize) {
    unsafe {
        *JitFrame::slot_ptr(ptr, pos) = value;
    }
}

/// llmodel.py:450-453 — `get_ref_value_direct(deadframe, pos)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `pos + 1`
/// trailing array slots.
pub unsafe fn get_ref_value_direct(ptr: *const JitFrame, pos: usize) -> usize {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize;
        *base.add(pos)
    }
}

/// llmodel.py:459-462 — `get_float_value_direct(deadframe, pos)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame with at least `pos + 1`
/// trailing array slots.
pub unsafe fn get_float_value_direct(ptr: *const JitFrame, pos: usize) -> u64 {
    unsafe {
        let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const u64;
        *base.add(pos)
    }
}

/// llmodel.py:422-424 — `_decode_pos(deadframe, index)`.
///
/// Upstream:
/// ```python
/// def _decode_pos(self, deadframe, index):
///     descr = self.get_latest_descr(deadframe)
///     return rffi.cast(lltype.Signed, descr.rd_locs[index]) * WORD
/// ```
///
/// Reads `descr.rd_locs[index]` (the compiled slot-position table
/// materialized by `assembler.py:240-278 store_info_on_descr`).
/// Upstream returns a byte offset (`rd_locs[index] * WORD`); pyre
/// returns the slot index directly since the companion `get_*_value_
/// direct` free functions below multiply by `WORD` internally through
/// `JitFrame::slot_ptr_const`.
///
/// Virtualized entries never reach this function (upstream assumption
/// encoded in `AbstractFailDescr._attrs_.rd_locs` being a plain
/// USHORT array).  Panics if `index` is out of range — mirroring the
/// bare Python indexing at upstream line 424.
pub fn _decode_pos(descr: &dyn FailDescr, index: usize) -> usize {
    descr.rd_locs()[index] as usize
}

/// llmodel.py:437-439 — `get_int_value(deadframe, index)`.
///
/// Upstream:
/// ```python
/// def get_int_value(self, deadframe, index):
///     pos = self._decode_pos(deadframe, index)
///     return self.get_int_value_direct(deadframe, pos)
/// ```
///
/// # Safety
/// `ptr` must point to a valid JitFrame matching `descr`.
pub unsafe fn get_int_value(ptr: *const JitFrame, descr: &dyn FailDescr, index: usize) -> isize {
    let pos = _decode_pos(descr, index);
    unsafe { get_int_value_direct(ptr, pos) }
}

/// llmodel.py:446-448 — `get_ref_value(deadframe, index)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame matching `descr`.
pub unsafe fn get_ref_value(ptr: *const JitFrame, descr: &dyn FailDescr, index: usize) -> usize {
    let pos = _decode_pos(descr, index);
    unsafe { get_ref_value_direct(ptr, pos) }
}

/// llmodel.py:455-457 — `get_float_value(deadframe, index)`.
///
/// # Safety
/// `ptr` must point to a valid JitFrame matching `descr`.
pub unsafe fn get_float_value(ptr: *const JitFrame, descr: &dyn FailDescr, index: usize) -> u64 {
    let pos = _decode_pos(descr, index);
    unsafe { get_float_value_direct(ptr, pos) }
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
