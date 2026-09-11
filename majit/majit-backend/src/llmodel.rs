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
//! The `AbstractCPU` base class (rpython/jit/backend/model.py)
//! declares the abstract contract for these accessors; all entries
//! below match those signatures.

use majit_gc::shadow_stack::OwnerRootGuard;
use majit_ir::{FailDescr, GcRef};

use crate::jitframe::{FIRST_ITEM_OFFSET, JitFrame};

/// llmodel.py — get_latest_descr.
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

/// llmodel.py — `_decode_pos(deadframe, index)`.
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
    let locs = descr.rd_locs();
    // Synthetic descrs never receive `write_failure_recovery_description`,
    // so the table stays empty and the fail-arg index *is* the slot
    // (`runner.rs` identity fallback). A stamped table uses `0xFFFF`
    // for a numbering hole (`optimizeopt` `logical_rd_locs`); resume
    // reconstructs those through TAGCONST/TAGVIRTUAL, not the jitframe.
    if locs.is_empty() {
        return Some(index);
    }
    match locs.get(index).copied() {
        None | Some(0xFFFF) => None,
        Some(pos) => Some(pos as usize),
    }
}

/// llmodel.py — `get_int_value_direct(deadframe, pos)`.
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

/// llmodel.py `get_int_value(deadframe, index)`.
///
/// `_decode_pos` then `get_int_value_direct`. Values stay in
/// `jf_frame[]`; nothing is copied into a host list.
#[inline]
pub unsafe fn get_int_value(ptr: *const JitFrame, descr: &dyn FailDescr, index: usize) -> i64 {
    let ptr = unsafe { JitFrame::resolve(ptr as *mut JitFrame) };
    // `_decode_pos` is only invoked for a live box. A 0xFFFF hole has
    // no jitframe word — cranelift writes fail args densely and skips
    // `None` holes, so treating the hole as `jf_frame[index]` reads
    // uninitialized memory and hands it to residual calls as a pointer.
    match decode_rd_loc_slot(descr, index) {
        Some(slot) => unsafe { get_int_value_direct(ptr, slot) as i64 },
        None => 0,
    }
}

/// Fail-arg source for `resume.py` TAGBOX decode.
///
/// RPython's decoder calls `cpu.get_int_value` / `get_ref_value` on the
/// deadframe. Tests that already hold a dense fail-arg list keep the
/// slice arm; compiled guard failure uses the jitframe.
///
/// The jitframe arm holds an [`OwnerRootGuard`] so a collection during
/// `blackhole_from_resumedata` updates the address; `get` re-reads the
/// root and walks `jf_forward` (`jitframe_resolve`).
pub enum FailArgSource<'a> {
    Slice(&'a [i64]),
    JitFrame {
        root: OwnerRootGuard,
        descr: &'a dyn FailDescr,
        n: usize,
    },
}

impl<'a> FailArgSource<'a> {
    pub fn from_jitframe(ptr: *const JitFrame, descr: &'a dyn FailDescr, n: usize) -> Self {
        let ptr = unsafe { JitFrame::resolve(ptr as *mut JitFrame) };
        Self::JitFrame {
            root: OwnerRootGuard::new(GcRef(ptr as usize)),
            descr,
            n,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Slice(s) => s.len(),
            Self::JitFrame { n, .. } => *n,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> i64 {
        match self {
            Self::Slice(s) => s.get(index).copied().unwrap_or(0),
            Self::JitFrame { root, descr, n } => {
                debug_assert!(index < *n);
                let ptr = root.get().0 as *const JitFrame;
                unsafe { get_int_value(ptr, *descr, index) }
            }
        }
    }

    #[inline]
    pub fn first(&self) -> Option<i64> {
        (self.len() > 0).then(|| self.get(0))
    }
}

impl Clone for FailArgSource<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::Slice(s) => Self::Slice(s),
            Self::JitFrame { root, descr, n } => Self::JitFrame {
                root: OwnerRootGuard::new(root.get()),
                descr: *descr,
                n: *n,
            },
        }
    }
}

impl<'a> From<&'a [i64]> for FailArgSource<'a> {
    fn from(s: &'a [i64]) -> Self {
        Self::Slice(s)
    }
}

impl<'a> From<&'a Vec<i64>> for FailArgSource<'a> {
    fn from(s: &'a Vec<i64>) -> Self {
        Self::Slice(s)
    }
}

impl<'a, const N: usize> From<&'a [i64; N]> for FailArgSource<'a> {
    fn from(s: &'a [i64; N]) -> Self {
        Self::Slice(s)
    }
}

impl std::fmt::Debug for FailArgSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vals: Vec<i64> = (0..self.len()).map(|i| self.get(i)).collect();
        match self {
            Self::Slice(_) => f.debug_tuple("Slice").field(&vals).finish(),
            Self::JitFrame { root, n, .. } => f
                .debug_struct("JitFrame")
                .field("ptr", &(root.get().0 as *const JitFrame))
                .field("n", n)
                .field("vals", &vals)
                .finish(),
        }
    }
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

/// llmodel.py — `get_ref_value_direct(deadframe, pos)`.
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

/// llmodel.py — `get_float_value_direct(deadframe, pos)`.
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

/// llmodel.py — `write_int_at_mem(gcref, ofs, size, newvalue)`.
///
/// Stores the low `size` bytes of `newvalue` at `base + ofs`.
///
/// The width is the field descriptor's, not the value's: an integer
/// field narrower than a word is a real field, and a store that ignores
/// `size` writes over whatever follows it in the struct.
///
/// Upstream walks `unroll_basic_sizes` (symbolic.py — word, char,
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

/// llmodel.py — `write_ref_at_mem(gcref, ofs, newvalue)`.
///
/// Pointer-width store. Upstream takes no `size` here: pointer fields
/// have one width, which is why `bh_setfield_gc_r` (llmodel.py)
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

/// llmodel.py — `write_float_at_mem(gcref, ofs, newvalue)`.
///
/// `FLOATSTORAGE`-width store. Like the ref store this takes no `size`:
/// floats are excluded from `unroll_basic_sizes` (symbolic.py) and
/// `bh_setfield_gc_f` (llmodel.py) unpacks only the offset.
///
/// # Safety
/// `base + ofs` must be a writable float-width field.
pub unsafe fn write_float_at_mem(base: usize, ofs: usize, newvalue: f64) {
    unsafe { (base.wrapping_add(ofs) as *mut f64).write_unaligned(newvalue) }
}

/// llmodel.py — get_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn get_savedata_ref(ptr: *const JitFrame) -> usize {
    unsafe { (*ptr).jf_savedata }
}

/// llmodel.py — set_savedata_ref.
///
/// # Safety
/// `ptr` must point to a valid JitFrame payload.
pub unsafe fn set_savedata_ref(ptr: *mut JitFrame, value: usize) {
    unsafe {
        (*ptr).jf_savedata = value;
    }
}

#[cfg(test)]
mod tests {
    use super::{FailArgSource, decode_rd_loc_slot, get_int_value, set_int_value};
    use crate::jitframe::{JitFrame, alloc_off_gc_jitframe, free_off_gc_jitframe};
    use crate::resume_guard_descr::make_resume_guard_descr_typed;
    use majit_ir::Type;

    #[test]
    fn empty_slice_get_does_not_panic() {
        // A host copy can be empty when resume numbering still asks
        // for TAGBOX 0 (`cpu.get_int_value(deadframe, 0)`). The
        // jitframe arm is the deadframe; a missing host slot reads 0.
        let src = FailArgSource::Slice(&[]);
        assert_eq!(src.get(0), 0);
        assert_eq!(src.len(), 0);
    }

    #[test]
    fn get_int_value_reads_zero_for_ffff_hole() {
        // optimizeopt `logical_rd_locs` stamps 0xFFFF on a None fail-arg.
        // The slot is not written (`emit_guard_exit` skips None). Reading
        // it as `jf_frame[index]` is how cranelift fed residual memmove a
        // poison pointer on exception_reused_object_tb_not_doubled.
        let descr = make_resume_guard_descr_typed(vec![Type::Int, Type::Ref, Type::Int]);
        let fd = descr.as_fail_descr().expect("typed resume guard");
        fd.set_rd_locs(vec![0, 0xFFFF, 2].into());
        assert_eq!(decode_rd_loc_slot(fd, 0), Some(0));
        assert_eq!(decode_rd_loc_slot(fd, 1), None);
        assert_eq!(decode_rd_loc_slot(fd, 2), Some(2));

        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(4));
        assert!(!frame.is_null());
        unsafe {
            for i in 0..4 {
                set_int_value(frame, i, 0x4155_8127);
            }
            set_int_value(frame, 0, 11);
            set_int_value(frame, 2, 22);
            assert_eq!(get_int_value(frame, fd, 0), 11);
            assert_eq!(
                get_int_value(frame, fd, 1),
                0,
                "0xFFFF hole must not surface the unwritten slot"
            );
            assert_eq!(get_int_value(frame, fd, 2), 22);
            free_off_gc_jitframe(frame);
        }
    }

    #[test]
    fn get_int_value_uses_identity_when_rd_locs_is_empty() {
        let descr = make_resume_guard_descr_typed(vec![Type::Int, Type::Int]);
        let fd = descr.as_fail_descr().expect("typed resume guard");
        assert!(fd.rd_locs().is_empty());
        assert_eq!(decode_rd_loc_slot(fd, 0), Some(0));
        assert_eq!(decode_rd_loc_slot(fd, 1), Some(1));

        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(2));
        assert!(!frame.is_null());
        unsafe {
            set_int_value(frame, 0, 7);
            set_int_value(frame, 1, 9);
            assert_eq!(get_int_value(frame, fd, 0), 7);
            assert_eq!(get_int_value(frame, fd, 1), 9);
            free_off_gc_jitframe(frame);
        }
    }
}
