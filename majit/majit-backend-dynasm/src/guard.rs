/// assembler.py ResumeGuardDescr parity: fail descriptor; the patchable
/// jump offset (`history.py:132 _attrs_` `adr_jump_offset`) lives on the
/// metainterp `ResumeGuardDescr` (`majit-metainterp/src/compile.rs`) and
/// is accessed here via `meta_resume_fd()` forwarding.
use majit_ir::FailDescr;

// Descr-by-address recovery is now a pure `Arc::from_raw` against the
// `FailDescrCell` wrapper baked at codegen time —
// `majit_ir::recover_fail_descr_cell(addr)` in `majit-ir/src/descr.rs`.
// `history.py:113 AbstractDescr.show(cpu, descr_gcref) =
// cast_gcref_to_instance(...)` parity.  The strong refcount is held by
// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py:294`).

// RECOVERY_LAYOUT_TABLE removed (Slice NN): `recovery_layout` is not in
// PyPy `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream resume
// code (`resume.py:450-488`) decodes recovery on demand from
// `rd_numb / rd_consts / rd_virtuals / rd_pendingfields`.  Pyre's
// metainterp `StoredExitLayout.recovery_layout` (populated by
// `compile.rs::patch_guard_recovery_layouts_for_trace` from
// the resume snapshot) is the canonical store; the backend no longer
// caches.  At deopt, `FailDescrLayout.recovery_layout` is `None` and
// `pyjitpl/mod.rs:6322` falls back to `trace_layout_ref.recovery_layout`
// which reads from the metainterp cache.

// FAIL_ARG_LOCS_TABLE removed (Slice MM): duplicates `rd_locs`
// (`history.py:132 _attrs_`).  All readers (`lib.rs:handle_fail_*` and
// `runner.rs::execute_token`) now decode `descr.rd_locs()` directly
// via `decode_rd_loc_slot` below, matching PyPy's
// `llmodel.py:422-424 _decode_pos`.

/// PyPy `llmodel.py:422-424 _decode_pos` parity.  Translate one
/// `rd_locs[index]` entry into the jitframe slot pyre's
/// `get_int_value_direct(jf, slot)` consumes.  Returns `None` for
/// 0xFFFF (unmapped — resume system handles via `rd_numb`
/// TAGCONST/TAGVIRTUAL encoding) or for out-of-range indices.
#[inline]
pub fn decode_rd_loc_slot(descr: &dyn FailDescr, index: usize) -> Option<usize> {
    let pos = *descr.rd_locs().get(index)?;
    if pos == 0xFFFF {
        None
    } else {
        Some(pos as usize)
    }
}

/// Re-export the shared per-cpu descr attachment types so existing
/// `crate::guard::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrHandle}`
/// imports keep resolving while the canonical definitions live in
/// `majit-backend` (shared with cranelift).
///
/// `rpython/jit/backend/model.py AbstractCPU` — descr attachments are a
/// cross-backend base-class concern; `compile.py:665-674
/// make_and_attach_done_descrs` binds them on each `cpu` instance
/// regardless of backend.
pub use majit_backend::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrHandle};

// `DynasmFailDescr` removed (Slice 7-Tα7).  PyPy's dynasm-equivalent
// `assembler.py` carries no per-emission descr wrapper — every guard
// op's `op.descr` is the same `ResumeGuardDescr` Arc that the metainterp
// stamps, and every FINISH emission writes the cpu-attached singleton
// (`compile.py:665-674 make_and_attach_done_descrs`) directly into
// `jf_descr`.  After Phases A/B/C of the Unified-Descr port the backend
// no longer owned any field worth wrapping, and the singleton-direct
// push (Slices 7-Tα3..7-Tα6) eliminated the last constructor of the
// wrapper.  All readers reach the descr through `DescrRef` and dispatch
// trait methods on `&dyn FailDescr`.

/// `compile.py:849` build a FailDescrLayout snapshot for any FailDescr
/// descr identity (post Phase C-1 cascade — `find_descr_by_ptr` returns
/// `DescrRef`, so the layout extractor takes the trait object plus the
/// data-pointer address used to key backend-static side-tables
/// (`SOURCE_OP_INDEX_TABLE`, `RECOVERY_LAYOUT_TABLE`)).
///
/// `fail_index` / `trace_id` are explicit caller arguments rather than
/// reads off `fd.fail_index_per_trace()` / `fd.trace_id()` so the
/// `fail_descrs` Vec position can be the identity source.  Singleton
/// FINISH descrs (`compile.py:623-662 _DoneWithThisFrameDescr` /
/// `ExitFrameWithExceptionDescrRef`) share an Arc across emissions and
/// answer the trait-default `0` for both methods; the layout pipeline
/// must read position from the Vec, not from the descr.
pub fn layout_for_fail_descr(
    fd: &dyn FailDescr,
    fail_index: u32,
    trace_id: u64,
) -> majit_backend::FailDescrLayout {
    // resume.py:450-488 propagate rd_* so `compiled_exit_layout_from_backend`
    // can reach them after the frontend trace cache evicts the owning
    // `CompiledTrace` entry (pyjitpl/mod.rs:817-845).
    let fail_arg_types = fd.fail_arg_types();
    majit_backend::FailDescrLayout {
        fail_index,
        fail_arg_types: fail_arg_types.to_vec(),
        is_finish: fd.is_finish(),
        is_exception_exit: fd.is_exit_frame_with_exception(),
        trace_id,
        source_op_index: fd.source_op_index(),
        // Forward through `FailDescr::is_gc_ref_slot` / `force_token_slots`
        // so concrete descrs that override these (e.g. cranelift descrs
        // that suppress force-token producer slots from GC classification)
        // contribute correct metadata to the layout.
        gc_ref_slots: (0..fail_arg_types.len())
            .filter(|&i| fd.is_gc_ref_slot(i))
            .collect(),
        force_token_slots: fd.force_token_slots(),
        frame_stack: None,
        // Slice NN: backend no longer caches recovery_layout.  The
        // metainterp `pyjitpl/mod.rs:6322` falls back to
        // `trace_layout_ref.recovery_layout` (read from
        // `StoredExitLayout.recovery_layout`).
        recovery_layout: None,
        trace_info: None,
        rd_numb: fd.rd_numb().map(|s| s.to_vec()),
        rd_consts: fd.rd_consts().map(|s| s.to_vec()),
        rd_virtuals: fd.rd_virtuals().map(|s| s.to_vec()),
        rd_pendingfields: fd.rd_pendingfields().map(|s| s.to_vec()),
    }
}

// `Debug`, `impl Descr`, `impl FailDescr` for the removed
// `DynasmFailDescr` all went away with the struct.  The metainterp
// `AbstractFailDescr` Arc carries the canonical `FailDescr` impl now
// reached through `op.descr` and `DescrRef::as_fail_descr`.
