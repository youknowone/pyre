//! jd3 (`sortkey_jmp`) symbolic + `JitState` scaffolding.
//!
//! listobject.py
//! `sortkey_jmp = JitDriver(greens=['strategy_type','tp'], reds='auto',
//! name='_compute_keys_for_sorting')`
//! drives `sortkey_portal`.

use majit_ir::{GcRef, GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::PyObjectRef;

pub const SORTKEY_PORTAL_KEY: &str = "builtins::sortkey_portal";
pub const SORTKEY_JD_INDEX: usize = 3;

/// jd3 symbolic state at the `sortkey_jmp` merge point.
///
/// listobject.py names the greens `strategy_type` and `tp` and `reds='auto'`
/// the loop locals `i`, `list_w`, `w_callable`. pyre keeps keys in a parallel
/// list (`keys_w`) because TimSort compares shadow-stack slots. Extracted
/// banks: `green_r=2`, `red_i=[]`, `red_r=[2, 3, 4]` (`list_w`, `keys_w`,
/// `w_callable`). `i` is dead at the marker (`i = w_list_len(keys_w)`).
pub struct SortkeySym {
    pub list_w: OpRef,
    pub keys_w: OpRef,
    pub w_callable: OpRef,
}

impl JitCodeSym for SortkeySym {
    fn total_slots(&self) -> usize {
        0
    }
    fn loop_header_pc(&self) -> usize {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(SORTKEY_PORTAL_KEY)
            .expect("jd3 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd3 portal body must contain a jit_merge_point")
            .pc
    }
}

pub struct SortkeyJitState {
    pub strategy_type: PyObjectRef,
    pub tp: PyObjectRef,
}

pub fn sortkey_green_key(strategy_type: PyObjectRef, tp: PyObjectRef) -> GreenKey {
    GreenKey::with_types(
        vec![strategy_type as i64, tp as i64],
        vec![GreenType::Ref, GreenType::Ref],
    )
}

pub fn sortkey_green_hash(strategy_type: PyObjectRef, tp: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(
        &[strategy_type as i64, tp as i64],
        &[GreenType::Ref, GreenType::Ref],
    )
}

pub fn sortkey_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    strategy_type: PyObjectRef,
    tp: PyObjectRef,
) -> u64 {
    let hash = sortkey_green_hash(strategy_type, tp);
    warm.resolve_cell_key(hash, || sortkey_green_key(strategy_type, tp))
}

/// Extracted reds: `list_w`, `keys_w`, `w_callable` (`red_r=[2, 3, 4]`).
pub fn sortkey_live_values(
    list_w: PyObjectRef,
    keys_w: PyObjectRef,
    w_callable: PyObjectRef,
) -> Vec<Value> {
    vec![
        Value::Ref(GcRef(list_w as usize)),
        Value::Ref(GcRef(keys_w as usize)),
        Value::Ref(GcRef(w_callable as usize)),
    ]
}

impl SortkeyJitState {
    pub fn sortkey_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("strategy_type", Type::Ref), ("tp", Type::Ref)],
            vec![
                ("list_w", Type::Ref),
                ("keys_w", Type::Ref),
                ("w_callable", Type::Ref),
            ],
        );
        sd.name = "_compute_keys_for_sorting".into();
        sd.no_loop_header = true;
        sd.frame_value_count_fn = Some(crate::state::build_time_frame_value_count_at);
        sd
    }
}

impl JitState for SortkeyJitState {
    type Meta = PyreMeta;
    type Sym = SortkeySym;
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
    fn extract_live(&mut self, meta: &Self::Meta) -> Vec<i64> {
        let _ = meta;
        Vec::new()
    }
    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym {
        let _ = (meta, header_pc);
        SortkeySym {
            list_w: OpRef::input_arg_typed(0, Type::Ref),
            keys_w: OpRef::input_arg_typed(1, Type::Ref),
            w_callable: OpRef::input_arg_typed(2, Type::Ref),
        }
    }
    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        let _ = meta;
        true
    }
    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        let _ = (meta, values);
    }
    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        vec![sym.list_w, sym.keys_w, sym.w_callable]
    }
    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extracted `sortkey_portal` merge-point banks (`red_r=[2, 3, 4]`).
    #[test]
    fn jd3_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(SORTKEY_PORTAL_KEY)
            .expect("jd3's extracted main JitCode must be registered");
        let merge_points: Vec<_> = crate::jitcode_runtime::decoded_ops(&canonical.code)
            .filter(|op| op.opname == "jit_merge_point")
            .collect();
        assert_eq!(merge_points.len(), 1);
        let banks =
            majit_metainterp::decode_jit_merge_point_banks(&canonical.code, merge_points[0].pc);
        assert_eq!(banks.green_i.len(), 0);
        assert_eq!(banks.green_r.len(), 2);
        assert_eq!(banks.green_f.len(), 0);
        assert_eq!(
            banks.red_i.len(),
            0,
            "red_i={:?} red_r={:?}",
            banks.red_i,
            banks.red_r
        );
        assert_eq!(
            banks.red_r.len(),
            3,
            "list_w, keys_w, w_callable; red_i={:?} red_r={:?}",
            banks.red_i,
            banks.red_r
        );
        assert_eq!(banks.red_f.len(), 0);
    }
}
