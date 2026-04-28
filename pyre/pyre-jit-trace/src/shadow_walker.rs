//! Walker fix A — shadow execution scaffolding.
//!
//! Master plan tingly-splashing-balloon.md Phase D-2: production tracing
//! runs the trait-driven `MIFrame::execute_opcode_step` (today's only
//! path) AND additionally re-runs the equivalent jitcode through the
//! symbolic walker (`jitcode_dispatch::dispatch_via_miframe`), then
//! asserts the two paths recorded the same trace ops. Mismatches panic
//! with a structured diff so divergences surface immediately.
//!
//! This module is the **scaffolding** layer — env gate + diff helper —
//! with no production hook wired yet. Slice 2 will wire
//! `MIFrame::execute_opcode_step` to consult [`shadow_walker_enabled`]
//! and, when on, drive a single opname through the diff helper. Slices
//! 3+ expand the gated opname set until every Python opcode is shadow-
//! validated; slices E/F finally switch the production path to the
//! walker and remove the trait dispatch.
//!
//! RPython parity: there is no direct counterpart — RPython has only
//! the jitcode interpreter and never carried two parallel trace
//! recorders. Pyre's shadow_walker exists exclusively as a migration
//! safety net while the trait dispatch is being retired; once the
//! migration completes, this module can be removed (Phase E).

use std::sync::OnceLock;

use majit_ir::Op;

/// Cache the env-var read so per-call sites don't pay the syscall on
/// every Python opcode. RPython has no equivalent — env-var feature
/// flags are a pyre-only adaptation, used during the trait→walker
/// migration only. See module-level docs for the migration plan.
fn shadow_walker_enabled_once() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MAJIT_SHADOW_WALKER")
            .ok()
            .as_deref()
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Returns `true` iff the process started with `MAJIT_SHADOW_WALKER=1`.
/// Production hook sites (Slice 2) gate their shadow walker call on
/// this. Default off — production unaffected.
pub fn shadow_walker_enabled() -> bool {
    shadow_walker_enabled_once()
}

/// Compare the trace ops recorded by the trait-dispatch path against
/// the walker-shadow path for the same Python opcode. Returns:
/// * `None` — both paths produced the same op sequence (per
///   `OpCode + args` equality; descr identity via `Arc::ptr_eq`).
/// * `Some(diff)` — structured human-readable diff suitable for
///   panic message. Caller is expected to panic; this helper does not
///   abort process so unit tests can exercise both branches.
///
/// The comparison is intentionally narrow today: opcode + args + descr
/// pointer identity. `pos` differs by construction (the two paths
/// record into separate `TraceCtx` instances) and is excluded.
/// `fail_args`, `fail_arg_types`, `rd_resume_position`, `vecinfo` are
/// all set by later optimizer / unroll passes — they are absent at
/// recording time so excluding them adds no false-negatives. As later
/// slices land richer state in walker (e.g. capture_resumedata for
/// num_live), this helper can be tightened.
pub fn diff_recorded_ops(production: &[Op], shadow: &[Op]) -> Option<String> {
    if production.len() != shadow.len() {
        return Some(format!(
            "shadow_walker: op count mismatch — production={}, shadow={}\n\
             production opcodes: {:?}\nshadow opcodes:     {:?}",
            production.len(),
            shadow.len(),
            production.iter().map(|o| o.opcode).collect::<Vec<_>>(),
            shadow.iter().map(|o| o.opcode).collect::<Vec<_>>(),
        ));
    }
    for (idx, (p, s)) in production.iter().zip(shadow.iter()).enumerate() {
        if p.opcode != s.opcode {
            return Some(format!(
                "shadow_walker: op[{}] opcode mismatch — production={:?}, shadow={:?}",
                idx, p.opcode, s.opcode,
            ));
        }
        if p.args.as_slice() != s.args.as_slice() {
            return Some(format!(
                "shadow_walker: op[{}] ({:?}) args mismatch — \
                 production={:?}, shadow={:?}",
                idx,
                p.opcode,
                p.args.as_slice(),
                s.args.as_slice(),
            ));
        }
        match (&p.descr, &s.descr) {
            (None, None) => {}
            (Some(pd), Some(sd)) => {
                if !std::sync::Arc::ptr_eq(pd, sd) {
                    return Some(format!(
                        "shadow_walker: op[{}] ({:?}) descr identity mismatch — \
                         production={:p}, shadow={:p}",
                        idx,
                        p.opcode,
                        std::sync::Arc::as_ptr(pd),
                        std::sync::Arc::as_ptr(sd),
                    ));
                }
            }
            (production_some, shadow_some) => {
                return Some(format!(
                    "shadow_walker: op[{}] ({:?}) descr presence mismatch — \
                     production_has_descr={}, shadow_has_descr={}",
                    idx,
                    p.opcode,
                    production_some.is_some(),
                    shadow_some.is_some(),
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{DescrRef, OpCode, OpRef};

    fn op(opcode: OpCode, args: &[OpRef], descr: Option<DescrRef>) -> Op {
        Op {
            opcode,
            args: args.iter().copied().collect(),
            descr,
            pos: OpRef(0),
            fail_args: None,
            fail_arg_types: None,
            rd_resume_position: -1,
            vecinfo: None,
        }
    }

    #[test]
    fn shadow_walker_enabled_defaults_off() {
        // Without `MAJIT_SHADOW_WALKER=1` set in the test process env,
        // the gate must read `false`. The OnceLock cache means the
        // value is fixed for the whole test process — toggling the env
        // var inside one test would not flip the gate. This test only
        // verifies the default branch.
        if std::env::var("MAJIT_SHADOW_WALKER").ok().as_deref() == Some("1") {
            // Skip: the test process happens to have the gate on; the
            // default-off branch is unobservable.
            return;
        }
        assert!(!shadow_walker_enabled());
    }

    #[test]
    fn diff_recorded_ops_returns_none_for_identical_traces() {
        let descr = make_fail_descr_ref();
        let trace_a = vec![
            op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], None),
            op(OpCode::Finish, &[OpRef(2)], Some(descr.clone())),
        ];
        let trace_b = trace_a.clone();
        assert_eq!(diff_recorded_ops(&trace_a, &trace_b), None);
    }

    #[test]
    fn diff_recorded_ops_reports_count_mismatch() {
        let trace_a = vec![op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], None)];
        let trace_b: Vec<Op> = vec![];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op count mismatch"), "diff was: {diff}");
        assert!(diff.contains("production=1"));
        assert!(diff.contains("shadow=0"));
    }

    #[test]
    fn diff_recorded_ops_reports_opcode_mismatch() {
        let trace_a = vec![op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], None)];
        let trace_b = vec![op(OpCode::IntSub, &[OpRef(0), OpRef(1)], None)];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op[0] opcode mismatch"), "diff was: {diff}");
        assert!(diff.contains("IntAdd"));
        assert!(diff.contains("IntSub"));
    }

    #[test]
    fn diff_recorded_ops_reports_args_mismatch() {
        let trace_a = vec![op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], None)];
        let trace_b = vec![op(OpCode::IntAdd, &[OpRef(0), OpRef(2)], None)];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op[0]"));
        assert!(diff.contains("args mismatch"), "diff was: {diff}");
    }

    #[test]
    fn diff_recorded_ops_reports_descr_identity_mismatch() {
        let descr_a = make_fail_descr_ref();
        let descr_b = make_fail_descr_ref();
        let trace_a = vec![op(OpCode::Finish, &[OpRef(0)], Some(descr_a))];
        let trace_b = vec![op(OpCode::Finish, &[OpRef(0)], Some(descr_b))];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("descr identity mismatch"), "diff was: {diff}");
    }

    #[test]
    fn diff_recorded_ops_reports_descr_presence_mismatch() {
        let descr = make_fail_descr_ref();
        let trace_a = vec![op(OpCode::Finish, &[OpRef(0)], Some(descr))];
        let trace_b = vec![op(OpCode::Finish, &[OpRef(0)], None)];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("descr presence mismatch"), "diff was: {diff}");
        assert!(diff.contains("production_has_descr=true"));
        assert!(diff.contains("shadow_has_descr=false"));
    }

    fn make_fail_descr_ref() -> DescrRef {
        std::sync::Arc::new(majit_ir::SimpleFailDescr::new(0, 0, vec![]))
    }
}
