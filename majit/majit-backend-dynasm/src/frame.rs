//! jitframe.py / assembler.py frame parity:
//! the deadframe a compiled run hands back, and the values read out of it.
use std::sync::{OnceLock, RwLock};

use majit_backend::jitframe::JitFrame;
use majit_ir::DescrRef;
use majit_ir::GcRef;

/// Concrete data stored in DeadFrame by the dynasm backend.
///
/// **THE DEADFRAME IS THE JITFRAME.** `llmodel.py:240-250` obtains it by
/// `lltype.cast_opaque_ptr(jitframe.JITFRAMEPTR, deadframe)` and reads
/// `jf_guard_exc` / `jf_savedata` straight off it; `llmodel.py:298` allocates
/// exactly one `malloc_jitframe` per `execute_token`, and `:323` returns that
/// same object as the deadframe. So there is nothing to copy out of the frame
/// and no second structure to describe it: this type is the cast, and the
/// accessors below are the reads.
///
/// `fail_descr` carries the metainterp class-distinct Arc identity
/// (ResumeGuardDescr family for guards, DoneWithThisFrame*/
/// ExitFrameWithExceptionDescrRef for FINISH exits).  FailDescr-trait
/// operations on the descr go through `DescrRef::as_fail_descr`.
pub struct FrameData {
    /// The frame compiled code returned — the tip of `head`'s `jf_forward`
    /// chain whenever `_check_frame_depth` reallocated on the way.
    tip: *mut JitFrame,
    /// Head of the chain this deadframe OWNS and frees on drop, or `None`
    /// when the frames belong to somebody else. `Backend::force` mints a
    /// deadframe over the frame of a compiled run that is still executing
    /// (`llmodel.py:280-284 force` likewise returns that frame rather than a
    /// copy), and freeing it there would pull the ground out from under the
    /// caller.
    owned_head: Option<*mut JitFrame>,
    /// Number of slots the frame carries. Bounds the index space the
    /// accessors answer for; an index past it reads 0.
    num_slots: usize,
    /// Backend-local fail descriptor used for slot decoding / bridge data.
    pub(crate) fail_descr: DescrRef,
    /// Original `jf_descr` object identity when the exit used an attached
    /// metainterp descr (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub(crate) latest_descr: Option<DescrRef>,
}

// SAFETY: `tip` / `owned_head` are libc allocations owned solely by this value
// for its whole lifetime (`owned_head`), or borrowed from a compiled run on
// the same thread (`None`). They are never aliased for mutation. `DeadFrame`
// requires `Send` on its payload, and `JitCellToken` already carries the same
// single-JIT-thread promise for the descrs held beside them here.
unsafe impl Send for FrameData {}

impl std::fmt::Debug for FrameData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameData")
            .field("num_values", &self.num_slots)
            .field("owns_frame", &self.owned_head.is_some())
            .field("fail_descr", &self.fail_descr.repr())
            .field(
                "latest_descr",
                &self.latest_descr.as_ref().map(|descr| descr.repr()),
            )
            .finish()
    }
}

impl FrameData {
    /// Take ownership of the jitframe chain `head` and present its tip as the
    /// deadframe.
    ///
    /// # Safety
    /// `head` must be a live chain of `register_libc_jitframe`-tracked frames
    /// that no one else frees, `tip` must be a node of that chain, and the
    /// compiled epilogue must already have popped every node off the JF shadow
    /// stack.
    pub unsafe fn owning(
        head: *mut JitFrame,
        tip: *mut JitFrame,
        num_slots: usize,
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        register_live_deadframe(tip as usize);
        FrameData {
            tip,
            owned_head: Some(head),
            num_slots,
            fail_descr,
            latest_descr,
        }
    }

    /// Present `frame` as a deadframe without taking ownership of it.
    ///
    /// # Safety
    /// `frame` must outlive the returned value.
    pub unsafe fn borrowing(
        frame: *mut JitFrame,
        num_slots: usize,
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        FrameData {
            tip: frame,
            owned_head: None,
            num_slots,
            fail_descr,
            latest_descr,
        }
    }

    /// `llmodel.py:422-424 _decode_pos` — the jitframe slot logical failarg
    /// `index` lives in, or `None` when the descr maps it nowhere.
    ///
    /// Indices past `rd_locs` fall through to identity slot indexing, which is
    /// what the synthetic and out-of-range descrs the runner mints rely on.
    fn slot_of(&self, index: usize) -> Option<usize> {
        if index >= self.num_slots {
            return None;
        }
        let descr = self.fail_descr.as_fail_descr()?;
        if index < descr.rd_locs().len() {
            crate::guard::decode_rd_loc_slot(descr, index)
        } else {
            Some(index)
        }
    }

    pub fn get_int(&self, index: usize) -> i64 {
        match self.slot_of(index) {
            Some(slot) => unsafe { crate::llmodel::get_int_value_direct(self.tip, slot) as i64 },
            None => 0,
        }
    }

    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    /// `cpu.grab_exc_value(deadframe)` (llmodel.py:240-242) — read
    /// `jf_guard_exc` off the frame. The exc=True failure-recovery stub staged
    /// `pos_exc_value` there for must_save_exception guards; other guards
    /// leave it NULL.
    pub fn exc_value(&self) -> GcRef {
        GcRef(unsafe { (*self.tip).jf_guard_exc })
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        let Some(head) = self.owned_head else {
            return;
        };
        unregister_live_deadframe(self.tip as usize);
        // jitframe.py:139-145 — when `_check_frame_depth`'s realloc slowpath
        // fires, the outgrown frame's `jf_forward` links to its replacement
        // (`dynasm_realloc_frame`), and every node in that chain was
        // `register_libc_jitframe`'d. Walk it, reading `jf_forward` before
        // freeing each node, so every frame is unregistered and freed exactly
        // once. In the common no-realloc case the chain is one node.
        let mut cur = head;
        while !cur.is_null() {
            let next = unsafe { (*cur).jf_forward };
            majit_gc::shadow_stack::unregister_libc_jitframe(cur as usize);
            // Frees the block base, which sits one header word behind `cur`.
            unsafe { majit_backend::jitframe::free_off_gc_jitframe(cur) };
            cur = next;
        }
    }
}

// ── live deadframes as GC roots ────────────────────────────────────────────

/// Payload addresses of the jitframes currently held as deadframes.
///
/// Between a compiled run returning and its `FrameData` dropping, the frame is
/// off the JF shadow stack — `gen_footer_shadowstack` pops it in the epilogue —
/// but its interior `Ref` slots are still live, because that is the window in
/// which the frontend reads them. Nothing else in the collector's root phase
/// can see them, and majit has no conservative stack scan to fall back on, so
/// the set is published to `majit-gc` through
/// [`majit_gc::ActiveGcDeadFrameHooks`] and walked there as a root source.
///
/// Process-global rather than thread-local, matching
/// `shadow_stack::register_libc_jitframe`'s own registry: a collection can run
/// on a thread other than the one holding the deadframe, and a thread-local set
/// would be invisible to it.
static LIVE_DEADFRAMES: OnceLock<RwLock<indexmap::IndexSet<usize>>> = OnceLock::new();

fn live_deadframes() -> &'static RwLock<indexmap::IndexSet<usize>> {
    LIVE_DEADFRAMES.get_or_init(|| RwLock::new(indexmap::IndexSet::new()))
}

fn register_live_deadframe(addr: usize) {
    live_deadframes().write().unwrap().insert(addr);
}

fn unregister_live_deadframe(addr: usize) {
    live_deadframes().write().unwrap().swap_remove(&addr);
}

/// [`majit_gc::LiveDeadFrameWalkerFn`] for this backend.
pub fn walk_live_deadframes(visit: &mut dyn FnMut(usize)) {
    for &addr in live_deadframes().read().unwrap().iter() {
        visit(addr);
    }
}
