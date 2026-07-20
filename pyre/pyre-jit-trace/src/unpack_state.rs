//! Dormant jd1 (`unpackiterable_driver`) symbolic + `JitState` scaffolding.
//!
//! `baseobjspace.py:29`
//! `unpackiterable_driver = JitDriver(greens=['greenkey'], reds='auto', ...)`
//! drives the unknown-length unpack loop `_unpackiterable_unknown_length`
//! (`baseobjspace.py:1003-1024`, merge at `:1012`). This module supplies the
//! *dormant* second-driver types for the generic LLBC meta-tracer, via
//! [`UnpackJitState`]'s [`JitState`] implementation, without touching jd0's
//! `state.rs`:
//!
//! * [`UnpackSym`] — the `reds='auto'` symbolic state: `w_iterator` (the
//!   iterator drained by `self.next(w_iterator)`) and `items` (the list grown
//!   by `items.append(w_item)`).
//! * [`UnpackJitState`] — `Meta = PyreMeta`, `Sym = UnpackSym`, `Env = PyreEnv`.
//!
//! The descriptor builder and its second `register_jitdriver_sd` registration
//! in `build_jit_driver_pair` now exist. The remaining activation follow-ups
//! are blackhole entry and the merge-point invocation at
//! `_unpackiterable_unknown_length`.

use majit_ir::{OpRef, Type};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::{PY_NULL, PyObjectRef};

/// jd1 symbolic state carried across the `unpackiterable_driver` back-edge.
///
/// The two `reds='auto'` values are [`Self::w_iterator`] and [`Self::items`];
/// `greenkey` is the merge-point green (const across the loop, never a jump
/// arg).
#[allow(dead_code)]
pub struct UnpackSym {
    /// baseobjspace.py:1012 jit_merge_point(greenkey=greenkey) green.
    pub greenkey: PyObjectRef,
    /// reds='auto' #1 — the iterator drained by `self.next(w_iterator)`.
    pub w_iterator: OpRef,
    /// reds='auto' #2 — the `items` list grown by `items.append(w_item)`.
    pub items: OpRef,
}

impl JitCodeSym for UnpackSym {
    fn total_slots(&self) -> usize {
        0
    }

    fn loop_header_pc(&self) -> usize {
        0x5b
    }

    fn fail_args(&self) -> Option<Vec<OpRef>> {
        Some(vec![self.w_iterator, self.items])
    }

    fn fail_args_types(&self) -> Option<Vec<Type>> {
        Some(vec![Type::Ref, Type::Ref])
    }
}

/// jd1 `JitState`: reuses [`PyreMeta`] (the "shape of the traced code") and the
/// empty [`PyreEnv`], driving [`UnpackSym`]. Novable, so the meta carries no
/// virtualizable payload and no extra reds.
pub struct UnpackJitState {
    /// The merge-point green pinned for this driver activation.
    pub greenkey: PyObjectRef,
}

impl UnpackJitState {
    /// jd1 (`unpackiterable_driver`) portal descriptor.
    /// `baseobjspace.py:29` `greens=['greenkey'], reds='auto'` — the two
    /// `reds='auto'` values are `w_iterator` and `items` (see [`UnpackSym`]),
    /// in the argument order `create_sym` seeds and `collect_jump_args`
    /// returns. Novable: no virtualizable name, so
    /// `elect_active_jitdriver_sd`'s vinfo-scan keeps electing jd0. The driver
    /// yields the grown `items` list → `Type::Ref` (the `new` default).
    pub fn unpackiterable_driver_descriptor() -> JitDriverStaticData {
        JitDriverStaticData::new(
            vec![("greenkey", Type::Ref)],
            vec![("w_iterator", Type::Ref), ("items", Type::Ref)],
        )
    }
}

impl JitState for UnpackJitState {
    type Meta = PyreMeta;
    type Sym = UnpackSym;
    type Env = PyreEnv;

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta {
        let _ = (header_pc, env);
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 0,
            array_capacity: 0,
            trace_extra_reds: 0,
            has_virtualizable: false,
            slot_types: Vec::new(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        // The live reds (`w_iterator`, `items`) are the tracer's InputArgs, not
        // interpreter-frame values; the dormant driver holds no concrete frame
        // to project them from.
        let _ = meta;
        Vec::new()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = (meta, header_pc);
        UnpackSym {
            greenkey: PY_NULL,
            // reds='auto' seeded as the merge-point InputArgs in argument order.
            w_iterator: OpRef::input_arg_typed(0, Type::Ref),
            items: OpRef::input_arg_typed(1, Type::Ref),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // Single novable shape per greenkey.
        let _ = meta;
        true
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        // No interpreter frame to restore into; the reds live in the compiled
        // loop's registers.
        let _ = (meta, values);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.w_iterator, sym.items]
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_metainterp::{JitCode, RuntimeBhDescr};
    use majit_translate::jitcode::BhDescr;

    /// jd1's extracted main JitCode carries an empty per-jitcode `exec.descrs`;
    /// its `d`/`j` argcodes resolve through the shared global build-time descr
    /// pool (`install_global_build_descr_pool` -> `JitCode::descr_at`),
    /// mirroring RPython's single shared `Assembler.descrs`. This checks that
    /// resolution wiring — not the full trace walk, which executes each
    /// residual (`self.next`, `items.append`) for real and so needs the live
    /// runtime with patched fn-addresses (a later activation slice), not a bare
    /// unit test.
    #[test]
    fn jd1_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::get_jitcode_by_index(0)
            .expect("jd1's extracted main JitCode must occupy index 0");
        // The extracted body is the walkable unpack loop: exactly one merge
        // point at the documented loop-header pc.
        let merge_points: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .filter(|op| op.opname == "jit_merge_point")
            .collect();
        assert_eq!(merge_points.len(), 1);
        assert_eq!(merge_points[0].pc, 0x5b);

        crate::jitcode_runtime::install_global_build_descr_pool();
        let jitcode = JitCode::from_canonical((*canonical).clone());
        // A build-time jitcode has no per-jitcode descr pool; every resolution
        // below therefore exercises the global fallback in `descr_at`.
        assert!(jitcode.exec.descrs.is_empty());

        // jd1's first inline-call (`inline_call_r_r/dR>r`) names its callee in
        // the leading 2-byte `d` operand; `descr_at` must map it to a JitCode
        // through the global pool.
        let inline = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "inline_call_r_r")
            .expect("jd1 body has an inline_call_r_r");
        let callee_idx = canonical.code[inline.pc + 1] as usize
            | ((canonical.code[inline.pc + 2] as usize) << 8);
        assert!(
            jitcode
                .descr_at(callee_idx)
                .and_then(RuntimeBhDescr::as_jitcode)
                .is_some(),
            "inline_call callee must resolve to a JitCode through the global pool"
        );

        // Every slot in the shared pool resolves, and it carries both the
        // JitCode (inline-call) and Call (residual-call) descr families the jd1
        // loop body reads back via `descr_at` — the residual-call descr read
        // that previously panicked on the empty per-jitcode pool.
        let n = crate::jitcode_runtime::all_descrs().len();
        assert!(n > 0, "build-time descr pool must be non-empty");
        let mut jitcodes = 0usize;
        let mut calls = 0usize;
        for i in 0..n {
            let d = jitcode
                .descr_at(i)
                .expect("every global build-time descr slot must resolve");
            if d.as_jitcode().is_some() {
                jitcodes += 1;
            }
            if matches!(d.as_bh_descr(), Some(BhDescr::Call { .. })) {
                calls += 1;
            }
        }
        assert!(jitcodes > 0, "pool must resolve inline-call callees");
        assert!(calls > 0, "pool must resolve residual-call descrs");
    }
}
