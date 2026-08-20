//! AbstractLLCPU accessors —
//! `rpython/jit/backend/llsupport/llmodel.py` parity.
//!
//! Two families live here. The jitframe accessors read and write a
//! deadframe's slots; the `*_at_mem` accessors read and write a field
//! of a heap struct at a byte offset.
//!
//! Upstream both live as methods on `AbstractLLCPU`, invoked as
//! `cpu.get_int_value(deadframe, index)` /
//! `cpu.write_int_at_mem(struct, ofs, size, value)`. In majit there is
//! no `AbstractLLCPU`-equivalent trait with `self`-carried state that a
//! backend would override — every backend shares the same
//! JITFRAME-backed deadframe layout and the same raw-memory field
//! layout — so the accessors are free functions keyed on a raw
//! `*const JitFrame` or on a base address.
//!
//! The `AbstractCPU` base class (rpython/jit/backend/model.py:95-133)
//! declares the abstract contract for these accessors; all entries
//! below match those signatures.

use majit_ir::FailDescr;

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

/// llmodel.py:422-424 — `_decode_pos(deadframe, index)`.
///
/// Translate one `rd_locs[index]` entry into the jitframe slot
/// `get_int_value_direct(jf, slot)` consumes.  Returns `None` for
/// 0xFFFF (unmapped — the resume system handles those through the
/// `rd_numb` TAGCONST/TAGVIRTUAL encoding) or for out-of-range indices.
///
/// Upstream `_decode_pos` is a method on `AbstractLLCPU` and fetches the
/// descr itself through `get_latest_descr(deadframe)`; here the descr is
/// passed in, because the deadframe types that hold one are the callers.
#[inline]
pub fn decode_rd_loc_slot(descr: &dyn FailDescr, index: usize) -> Option<usize> {
    let pos = *descr.rd_locs().get(index)?;
    if pos == 0xFFFF {
        None
    } else {
        Some(pos as usize)
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
/// `index` through `rd_locs[]` — is a method on the deadframe
/// types instead ([`crate::deadframe`], [`crate::libc_deadframe`]),
/// because it needs the descr that `_decode_pos` reaches through
/// `get_latest_descr(deadframe)` and a free function keyed on a
/// bare frame pointer has no way to get one.
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

/// llmodel.py:481-488 — `write_int_at_mem(gcref, ofs, size, newvalue)`.
///
/// Stores the low `size` bytes of `newvalue` at `base + ofs`.
///
/// The width is the field descriptor's, not the value's: an integer
/// field narrower than a word is a real field, and a store that ignores
/// `size` writes over whatever follows it in the struct.
///
/// Upstream walks `unroll_basic_sizes` (symbolic.py:73-77 — word, char,
/// short, int) and falls through to
/// `raise NotImplementedError("size = %d" % size)` when nothing
/// matches. A size not in that set means the descriptor disagrees with
/// the struct it describes, so this panics rather than widening the
/// store; silently falling back to a word would corrupt the neighbour.
///
/// Float fields are deliberately absent from that table
/// (symbolic.py:78 "does not contain Float ^^^ which must be
/// special-cased") and go through [`write_float_at_mem`].
///
/// # Safety
/// `base + ofs` must be a writable field of at least `size` bytes.
pub unsafe fn write_int_at_mem(base: usize, ofs: usize, size: usize, newvalue: i64) {
    let addr = base.wrapping_add(ofs);
    // Truncation is width-identical for the signed and unsigned member of
    // each `unroll_basic_sizes` pair, so the store needs the size but not
    // the sign — which is why upstream discards it (`_` at llmodel.py:482)
    // while the matching read keeps it.
    unsafe {
        match size {
            1 => (addr as *mut u8).write_unaligned(newvalue as u8),
            2 => (addr as *mut u16).write_unaligned(newvalue as u16),
            4 => (addr as *mut u32).write_unaligned(newvalue as u32),
            8 => (addr as *mut i64).write_unaligned(newvalue),
            _ => panic!(
                "write_int_at_mem: unsupported size {size} \
                 (llmodel.py:488 NotImplementedError)"
            ),
        }
    }
}

/// llmodel.py:495-497 — `write_ref_at_mem(gcref, ofs, newvalue)`.
///
/// Pointer-width store. Upstream takes no `size` here: pointer fields
/// have one width, which is why `bh_setfield_gc_r` (llmodel.py:723-725)
/// unpacks only the offset while `bh_setfield_gc_i` unpacks the size
/// too.
///
/// Upstream's trailing comment reads "the write barrier is implied
/// above" — implied by the `llop.raw_store` that the framework GC
/// transformer rewrites. Nothing rewrites this store, so a caller whose
/// container may be old-generation while `newvalue` is young owes the
/// barrier itself.
///
/// # Safety
/// `base + ofs` must be a writable pointer-width field.
pub unsafe fn write_ref_at_mem(base: usize, ofs: usize, newvalue: usize) {
    unsafe { (base.wrapping_add(ofs) as *mut usize).write_unaligned(newvalue) }
}

/// llmodel.py:504-506 — `write_float_at_mem(gcref, ofs, newvalue)`.
///
/// `FLOATSTORAGE`-width store. Like the ref store this takes no `size`:
/// floats are excluded from `unroll_basic_sizes` (symbolic.py:78) and
/// `bh_setfield_gc_f` (llmodel.py:730-734) unpacks only the offset.
///
/// # Safety
/// `base + ofs` must be a writable float-width field.
pub unsafe fn write_float_at_mem(base: usize, ofs: usize, newvalue: f64) {
    unsafe { (base.wrapping_add(ofs) as *mut f64).write_unaligned(newvalue) }
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
