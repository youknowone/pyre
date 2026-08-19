//! The deadframe a jitframe-backed backend returns from a compiled run.
//!
//! `llmodel.py:323 return ll_frame` — the deadframe IS the JitFrame. Values
//! stay in `jf_frame[]` and are never copied out: `get_int_value(deadframe,
//! index)` (`llmodel.py:437-451`) casts the opaque deadframe back to a
//! JITFRAMEPTR and reads `jf_frame[index]` in place, and `get_latest_descr`
//! (`llmodel.py:411-419`) does the same for `jf_descr`.

use majit_gc::shadow_stack::OwnerRootGuard;
use majit_ir::{DescrRef, GcRef};

use crate::ExitRecoveryLayout;
use crate::jitframe::{FIRST_ITEM_OFFSET, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JitFrame};

/// Where a held deadframe keeps its jitframe pointer.
///
/// The frame is an ordinary GC object (`jitframe.py:33-60` makes JITFRAME a
/// GcStruct and `llmodel.py:298 malloc_jitframe` allocates it like any other),
/// so a collection during the window the deadframe is held promotes it to a
/// different address while every accessor is still reading through the old
/// one. The pointer therefore has to live somewhere the collector rewrites.
///
/// A root slot is that place. `shadowstack.py:100-106` (`push_stack` /
/// `pop_stack`) names a root by its POSITION on a per-thread stack rather than
/// by the address of the variable holding it, and `walk_stack_root`
/// (`shadowstack.py:44-70`) reads each live slot and stores the forwarded
/// address back into the slot it walked. Addressing by position is what makes
/// the holder's own address irrelevant, so this frame can be returned by value
/// instead of being pinned behind a per-exit heap allocation.
enum JitFrameRoot {
    /// The frame's position among this thread's root slots. Every read goes
    /// back through the position, so a collection that moved the frame between
    /// two reads is already accounted for by the second one.
    ///
    /// Slots are taken from the fixed-slot vector rather than the strictly
    /// LIFO stack next to it: a deadframe's lifetime is the caller's, and two
    /// live deadframes are released in whatever order their owners drop, which
    /// a stack discipline cannot express.
    ///
    /// The vector is per-thread, so the position names a slot on the thread
    /// that acquired it and the guard must be dropped there: releasing it
    /// elsewhere would free a foreign thread's slot, or find none active. The
    /// guard is `!Send` for that reason, which makes the whole deadframe
    /// thread-confined.
    Slot(OwnerRootGuard),
    /// The frame is not a GC object — it was allocated out of the Rust heap
    /// because no type registry was available to allocate it under — so it
    /// cannot move and taking a slot would hand the collector an address
    /// outside both generations.
    Unrooted(GcRef),
}

impl JitFrameRoot {
    #[inline]
    fn get(&self) -> GcRef {
        match self {
            JitFrameRoot::Slot(guard) => guard.get(),
            JitFrameRoot::Unrooted(gcref) => *gcref,
        }
    }
}

/// The deadframe of a backend whose frames are jitframes.
pub struct JitFrameDeadFrame {
    /// The JitFrame this deadframe reads through.
    jf_root: JitFrameRoot,
    /// The fail descriptor for this exit.  Stored as `DescrRef`
    /// (`Arc<dyn Descr>`) so the deadframe carries the same Arc identity
    /// the metainterp stamps onto `op.descr` — matching `frame.jf_descr =
    /// descr` (llmodel.py:270) line-by-line.
    pub fail_descr: DescrRef,
    /// Original attached `jf_descr` identity for finish exits emitted by
    /// the metainterp (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
    /// Side-channel: caller-prefix layout assembled from the
    /// `CALL_ASSEMBLER_CALLER_STACK` top at deadframe interception
    /// (`wrap_call_assembler_deadframe_with_caller_prefix`).  When `Some`,
    /// the exit's recovery layout is prefixed by this value.  Replaces the old
    /// overlay descr synthesis — the deadframe's `fail_descr` keeps the
    /// callee's own Arc identity rather than being swapped for a synthetic one.
    pub call_assembler_caller_layout: Option<ExitRecoveryLayout>,
    /// Keeps the frame memory alive for non-GC allocations.
    _heap_owner: Option<Vec<i64>>,
    /// Whether dropping this deadframe should release the frame's `jf_gcmap`.
    ///
    /// True for the deadframe a compiled run returns — the exit established
    /// that map and the frame is off the JF shadow stack by then.  False for a
    /// deadframe built over a frame that is still executing, whose map belongs
    /// to the call site that pushed it.
    owns_gcmap: bool,
}

impl JitFrameDeadFrame {
    /// Take a root slot for `jf_gcref` and hold it for the deadframe's life.
    ///
    /// `heap_owner` decides whether the frame is rooted at all, because it is
    /// the same condition: a frame handed a `Vec` owner was allocated out of
    /// the Rust heap precisely because there was no type registry to allocate
    /// it from the nursery under, and a frame allocated from the nursery is
    /// always handed `None`.
    pub fn new(
        jf_gcref: GcRef,
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
        heap_owner: Option<Vec<i64>>,
    ) -> Self {
        let jf_root = if heap_owner.is_some() {
            JitFrameRoot::Unrooted(jf_gcref)
        } else {
            JitFrameRoot::Slot(OwnerRootGuard::new(jf_gcref))
        };
        JitFrameDeadFrame {
            jf_root,
            fail_descr,
            latest_descr,
            call_assembler_caller_layout: None,
            _heap_owner: heap_owner,
            owns_gcmap: true,
        }
    }

    /// Present a frame that is still executing as a deadframe, without taking
    /// its `jf_gcmap` over.
    ///
    /// `llmodel.py:280-284 force` casts the resolved frame to a GCREF and
    /// returns it: the forced frame IS the deadframe, and it belongs to the
    /// compiled run that is still on the JF shadow stack.  That run pushed the
    /// map before the residual call it is inside and clears it with
    /// `pop_gcmap` when the call returns, so releasing it here would leave the
    /// frame untraced for the rest of the call while its spilled `Ref` slots
    /// are still the only reference to their objects.
    pub fn borrowing(
        jf_gcref: GcRef,
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        JitFrameDeadFrame {
            jf_root: JitFrameRoot::Slot(OwnerRootGuard::new(jf_gcref)),
            fail_descr,
            latest_descr,
            call_assembler_caller_layout: None,
            _heap_owner: None,
            owns_gcmap: false,
        }
    }

    /// The frame's CURRENT address, re-read through the root slot.
    ///
    /// Not cached anywhere: the slot is the only place the address is correct
    /// after a collection, so every accessor below goes through here.
    #[inline]
    pub fn jf_gcref(&self) -> GcRef {
        self.jf_root.get()
    }

    #[inline]
    pub fn get_int(&self, index: usize) -> i64 {
        // A safe function that dereferences an index the caller chose, so the
        // only thing between an index error and an out-of-bounds read is this
        // check. The frame carries its own slot count in the length word ahead
        // of `jf_frame`, which is the bound. `get_float` and `get_ref` read
        // through here, so one check covers all three.
        debug_assert!(
            (index as isize)
                < unsafe { JitFrame::frame_length(self.jf_gcref().0 as *const JitFrame) },
            "jf_frame index {index} is past the frame's own length",
        );
        unsafe { *((self.jf_gcref().0 + FIRST_ITEM_OFFSET + index * 8) as *const i64) }
    }

    #[inline]
    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    #[inline]
    pub fn get_savedata_ref(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref().0 + JF_SAVEDATA_OFS as usize) as *const usize) })
    }

    #[inline]
    pub fn try_get_savedata_ref(&self) -> Option<GcRef> {
        let r = self.get_savedata_ref();
        if r.is_null() { None } else { Some(r) }
    }

    #[inline]
    pub fn set_savedata_ref(&mut self, data: GcRef) {
        unsafe { *((self.jf_gcref().0 + JF_SAVEDATA_OFS as usize) as *mut usize) = data.0 };
    }

    #[inline]
    pub fn grab_exc_value(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref().0 + JF_GUARD_EXC_OFS as usize) as *const usize) })
    }
}

impl Drop for JitFrameDeadFrame {
    /// Release the frame's `jf_gcmap` before the root slot goes, for the
    /// deadframe that owns it.
    ///
    /// Releasing the root does not take the frame out of the collector's
    /// remembered set, so the frame stays reachable there after this owner is
    /// gone — and `jitframe_trace` (`jitframe.py:115-135`) would keep walking
    /// the exiting guard's map over `jf_frame` items nothing owns any more.
    /// A null `jf_gcmap` traces no items (`jitframe.py:115-116`) while the
    /// fixed header GCREFs stay traceable, which is what the frame needs once
    /// its values have been read out (`llmodel.py:437-451`).
    ///
    /// Only the rooted arm: an `Unrooted` frame is not a GC object, so nothing
    /// traces it at all — and only an owning deadframe, because a `borrowing`
    /// one stands for a frame whose compiled run is still inside the residual
    /// call that pushed the map.
    fn drop(&mut self) {
        if let JitFrameRoot::Slot(guard) = &self.jf_root {
            if self.owns_gcmap {
                unsafe { (*(guard.get().0 as *mut JitFrame)).jf_gcmap = std::ptr::null() };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitframe::{alloc_off_gc_jitframe, free_off_gc_jitframe};

    fn a_descr() -> DescrRef {
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(0, 8, 0))
    }

    /// The deadframe a compiled run returns owns the map its exit established;
    /// the one `force()` builds over a frame that is still executing does not.
    #[test]
    fn only_an_owning_deadframe_releases_the_frames_gcmap() {
        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(8));
        assert!(!frame.is_null());
        let sentinel = 0x1234usize as *const u8;

        unsafe { (*frame).jf_gcmap = sentinel };
        drop(JitFrameDeadFrame::borrowing(
            GcRef(frame as usize),
            a_descr(),
            None,
        ));
        assert_eq!(
            unsafe { (*frame).jf_gcmap },
            sentinel,
            "a borrowed frame's map belongs to the call site that pushed it"
        );

        unsafe { (*frame).jf_gcmap = sentinel };
        drop(JitFrameDeadFrame::new(
            GcRef(frame as usize),
            a_descr(),
            None,
            None,
        ));
        assert!(
            unsafe { (*frame).jf_gcmap }.is_null(),
            "an owning deadframe releases the map once its values are read out"
        );

        unsafe { free_off_gc_jitframe(frame) };
    }
}
