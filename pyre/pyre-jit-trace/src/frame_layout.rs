use majit_metainterp::virtualizable::VirtualizableInfo;
use pyre_interpreter::pyframe::PyFrame;

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `last_instr` in `PyFrame`.
pub const PYFRAME_LAST_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, last_instr);

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrame, locals_cells_stack_w);

/// Byte offset of `pycode` in `PyFrame`.
pub const PYFRAME_PYCODE_OFFSET: usize = std::mem::offset_of!(PyFrame, pycode);

/// Byte offset of `flags` in `PyFrame` — the `u8` holding `FLAG_ESCAPED`
/// (`pyframe.py escaped`).  A traced `tb_frame` read has to set that bit
/// the way the getter's `mark_as_escaped()` does.
pub const PYFRAME_FLAGS_OFFSET: usize = std::mem::offset_of!(PyFrame, flags);

/// Byte offset of the CPython-observable failed-attribute cleanup state.
pub const PYFRAME_FAILED_ATTR_CLEANUP_OFFSET: usize =
    std::mem::offset_of!(PyFrame, failed_attr_cleanup);

/// Byte offset of `debugdata` in `PyFrame`.
pub const PYFRAME_DEBUGDATA_OFFSET: usize = std::mem::offset_of!(PyFrame, debugdata);

/// Byte offset of `lastblock` in `PyFrame`.
pub const PYFRAME_LASTBLOCK_OFFSET: usize = std::mem::offset_of!(PyFrame, lastblock);

/// Byte offset of `f_generator_nowref` in `PyFrame`.
pub const PYFRAME_F_GENERATOR_NOWREF_OFFSET: usize =
    std::mem::offset_of!(PyFrame, f_generator_nowref);

/// Byte offset of `w_yielding_from` in `PyFrame`.
pub const PYFRAME_W_YIELDING_FROM_OFFSET: usize = std::mem::offset_of!(PyFrame, w_yielding_from);

/// Byte offset of `f_backref` in `PyFrame`.
pub const PYFRAME_F_BACKREF_OFFSET: usize = std::mem::offset_of!(PyFrame, f_backref);

/// Byte offset of `w_builtin` in `PyFrame`.
///
/// `frame.w_builtin` carries the picked builtin Module (`pick_builtin_w`
/// result) so `frame.get_builtin()` returns the same identity PyPy would.
/// The slot is a GCREF and must be visible to the descr GC walker so a
/// collection between guard exit and re-entry doesn't leave a dangling
/// pointer.
pub const PYFRAME_W_BUILTIN_OFFSET: usize = std::mem::offset_of!(PyFrame, w_builtin);

// Backward-compat aliases used by JIT descriptor helpers.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

// Compile-time consistency check: frame_layout and pyframe offsets must match.
const _: () = {
    assert!(PYFRAME_VABLE_TOKEN_OFFSET == pyre_interpreter::pyframe::PYFRAME_VABLE_TOKEN_OFFSET);
    assert!(PYFRAME_LAST_INSTR_OFFSET == pyre_interpreter::pyframe::PYFRAME_LAST_INSTR_OFFSET);
    assert!(PYFRAME_PYCODE_OFFSET == pyre_interpreter::pyframe::PYFRAME_PYCODE_OFFSET);
    assert!(
        PYFRAME_VALUESTACKDEPTH_OFFSET == pyre_interpreter::pyframe::PYFRAME_VALUESTACKDEPTH_OFFSET
    );
    assert!(
        PYFRAME_LOCALS_CELLS_STACK_OFFSET
            == pyre_interpreter::pyframe::PYFRAME_LOCALS_CELLS_STACK_OFFSET
    );
    assert!(PYFRAME_DEBUGDATA_OFFSET == pyre_interpreter::pyframe::PYFRAME_DEBUGDATA_OFFSET);
    assert!(PYFRAME_LASTBLOCK_OFFSET == pyre_interpreter::pyframe::PYFRAME_LASTBLOCK_OFFSET);
    assert!(
        PYFRAME_F_GENERATOR_NOWREF_OFFSET
            == pyre_interpreter::pyframe::PYFRAME_F_GENERATOR_NOWREF_OFFSET
    );
    assert!(
        PYFRAME_W_YIELDING_FROM_OFFSET == pyre_interpreter::pyframe::PYFRAME_W_YIELDING_FROM_OFFSET
    );
    assert!(PYFRAME_F_BACKREF_OFFSET == pyre_interpreter::pyframe::PYFRAME_F_BACKREF_OFFSET);
    assert!(PYFRAME_W_BUILTIN_OFFSET == pyre_interpreter::pyframe::PYFRAME_W_BUILTIN_OFFSET);
};

/// virtualizable.py `clear_vable_ptr` — C-ABI helper behind the `COND_CALL`
/// `emit_force_virtualizable` records, built out of `clear_vable_token`:
/// force the virtualizable when the token is set, then leave TOKEN_NONE.
///
/// ```python
/// def clear_vable_token(virtualizable):
///     virtualizable = cast_gcref_to_vtype(virtualizable)
///     if virtualizable.vable_token:
///         force_now(virtualizable)
///         assert not virtualizable.vable_token
/// ```
///
/// Upstream never writes the slot itself: a set token is cleared by the force
/// it triggers, and a clear one is left alone.  `force_now` has two arms and
/// only one of them is a bare write.  On TOKEN_TRACING_RESCALL it clears the
/// marker, which is what the post-residual probe reads as "the callee
/// escaped".  On a machine-frame token it runs
/// `ResumeGuardForcedDescr.force_now`, writing the compiled activation's
/// registers back into the frame — a step this helper used to skip, on the
/// evidence that the call was recorded 183 times and invoked 0 times over 462
/// fixtures.  That measurement no longer holds: the call fires on the corpus
/// today, so the arm is reachable and the frame it hands back has to carry the
/// compiled values.  `executioncontext.rs force_frame` dispatches to whichever
/// arm the token names — the same split `force_virtualizable_if_necessary`
/// runs.
///
/// The trailing write is pyre's and has no upstream counterpart: `force_frame`
/// goes through a `OnceLock` hook a JIT-less embedding never fills, and
/// `force_pyframe` declines a token the metainterp does not recognise as
/// armed, so a token left behind would fault later in `JitFrame::resolve`
/// rather than here.  Upstream asserts at that point; the debug assertion
/// below is that assert, and since a panic reached from compiled code is the
/// worse failure in a release build, the write stays as its backstop.
unsafe extern "C" fn pyre_clear_vable_token(obj_ptr: i64) {
    unsafe {
        let ptr = obj_ptr as *mut u8;
        if ptr.is_null() {
            return;
        }
        // `vable_token` is `usize` (pointer-width: 4 on wasm32). Reading or
        // writing 8 bytes would reach the following field.
        let token_ptr = ptr.add(PYFRAME_VABLE_TOKEN_OFFSET) as *mut usize;
        if *token_ptr == 0 {
            return;
        }
        pyre_interpreter::executioncontext::force_frame(ptr as *mut pyre_interpreter::PyFrame);
        debug_assert_eq!(
            *token_ptr, 0,
            "clear_vable_token: force_now must leave TOKEN_NONE behind"
        );
        if *token_ptr != 0 {
            if majit_metainterp::majit_log_enabled() {
                let token = *token_ptr;
                eprintln!("[jit][clear-vable] force left token=0x{token:x} frame={ptr:p}");
            }
            *token_ptr = 0;
        }
    }
}

/// Build the virtualizable layout description for `PyFrame`.
///
/// Delegates to `virtualizable_gen::build_virtualizable_info()` which is
/// auto-generated by the `virtualizable!` macro from the canonical field
/// declaration, and immediately attaches the host `PyFrame` SizeDescr
/// via `set_parent_descr` so every field descriptor produced by the
/// `VirtualizableInfo` carries `descr.py FieldDescr.parent_descr` —
/// required by `OptContext::ensure_ptr_info_arg0` (`optimizer.py`)
/// to dispatch GETFIELD/SETFIELD to `InstancePtrInfo` / `StructPtrInfo`.
pub fn build_pyframe_virtualizable_info() -> std::sync::Arc<VirtualizableInfo> {
    let mut info = crate::virtualizable_gen::build_virtualizable_info();
    // rpython/jit/metainterp/virtualizable.py `clear_vable_ptr`
    // + `clear_vable_descr`. The descr must carry
    // EffectInfo.MOST_GENERAL + OopSpecIndex.JitForceVirtualizable
    // and mark the call CANNOT_RAISE — `VirtualizableInfo::make_clear
    // _vable_descr` is the single-source-of-truth factory that
    // constructs exactly that descriptor. Using
    // `make_call_descr(.., EffectInfo::default())` here would drop
    // the CANNOT_RAISE / OS_JIT_FORCE_VIRTUALIZABLE flags and cause
    // the optimizer to treat the call as a raising general call.
    //
    // Populate clear_vable BEFORE `finalize_arc`: finalize consumes self
    // and rebuilds descriptors inside `Arc::new_cyclic`, so any fields
    // set here survive into the returned Arc. After the Arc is formed
    // the vinfo is immutable through the shared handle.
    info.clear_vable_ptr = Some(pyre_clear_vable_token as *const () as usize);
    info.clear_vable_descr = Some(VirtualizableInfo::make_clear_vable_descr());
    info.finalize_arc(crate::state::pyframe_size_descr())
}

#[cfg(test)]
mod tests {
    use super::build_pyframe_virtualizable_info;
    use super::{PYFRAME_VABLE_TOKEN_OFFSET, pyre_clear_vable_token};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Token value the stand-in force hook was handed, or `usize::MAX` while it
    /// has not run.  `usize::MAX` rather than 0 so "not called" and "called on a
    /// cleared slot" stay distinguishable.
    static FORCED_WITH_TOKEN: AtomicUsize = AtomicUsize::new(usize::MAX);

    /// Stand-in for `eval.rs force_pyframe`.  Upstream's `force_now` leaves
    /// TOKEN_NONE behind (`virtualizable.py force_now`), so this does too —
    /// `pyre_clear_vable_token`'s own debug assertion checks exactly that.
    unsafe extern "C" fn recording_force_hook(frame: *mut pyre_interpreter::PyFrame) {
        unsafe {
            let token_ptr = (frame as *mut u8).add(PYFRAME_VABLE_TOKEN_OFFSET) as *mut usize;
            FORCED_WITH_TOKEN.store(*token_ptr, Ordering::SeqCst);
            *token_ptr = 0;
        }
    }

    /// virtualizable.py `clear_vable_token`:
    ///
    /// ```python
    /// if virtualizable.vable_token:
    ///     force_now(virtualizable)
    ///     assert not virtualizable.vable_token
    /// ```
    ///
    /// A live token is FORCED, not overwritten — overwriting it drops the
    /// compiled activation's write-back.  A clear one calls nothing at all.
    ///
    /// Both halves share one test because the force hook is a process-wide
    /// `OnceLock` (`executioncontext.rs register_force_frame_hook`): a second
    /// test registering its own would be silently ignored.
    #[test]
    fn clear_vable_token_forces_a_live_token_and_leaves_a_clear_one_alone() {
        pyre_interpreter::executioncontext::register_force_frame_hook(recording_force_hook);

        // Only the token slot is read or written, by the helper and by the
        // hook alike, so a buffer that reaches past it stands in for a frame.
        let mut frame = vec![0u8; PYFRAME_VABLE_TOKEN_OFFSET + 64];
        let base = frame.as_mut_ptr();
        let token_ptr = unsafe { base.add(PYFRAME_VABLE_TOKEN_OFFSET) as *mut usize };

        // A cleared slot: upstream's `if virtualizable.vable_token` is false and
        // nothing runs.
        FORCED_WITH_TOKEN.store(usize::MAX, Ordering::SeqCst);
        unsafe { pyre_clear_vable_token(base as i64) };
        assert_eq!(
            FORCED_WITH_TOKEN.load(Ordering::SeqCst),
            usize::MAX,
            "TOKEN_NONE must not force"
        );

        // A live token: forced first, and the force is what clears it.
        unsafe { *token_ptr = 0xF00D_1000 };
        unsafe { pyre_clear_vable_token(base as i64) };
        assert_eq!(
            FORCED_WITH_TOKEN.load(Ordering::SeqCst),
            0xF00D_1000,
            "the force must see the token still live"
        );
        assert_eq!(unsafe { *token_ptr }, 0, "the slot ends at TOKEN_NONE");

        // A null frame is a no-op, not a fault.
        FORCED_WITH_TOKEN.store(usize::MAX, Ordering::SeqCst);
        unsafe { pyre_clear_vable_token(0) };
        assert_eq!(FORCED_WITH_TOKEN.load(Ordering::SeqCst), usize::MAX);
    }

    #[test]
    fn pyframe_token_descr_parent_survives_vinfo_drop() {
        let info = build_pyframe_virtualizable_info();
        let token_descr = info.token_field_descr();
        drop(info);

        let parent = token_descr
            .as_field_descr()
            .and_then(|fd| fd.get_parent_descr());
        assert!(
            parent.is_some(),
            "PyFrame token field descr must keep a live parent SizeDescr \
             like GcCache.get_size_descr()"
        );
    }
}
