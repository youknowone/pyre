//! jd4 (`sortkey_jmp`) symbolic + `JitState` scaffolding.
//!
//! listobject.py
//! `sortkey_jmp = JitDriver(greens=['strategy_type','tp'], reds='auto',
//! name='_compute_keys_for_sorting')`
//! drives `sortkey_portal`.

use majit_ir::{GcRef, GreenKey, GreenType, OpRef, Type, Value};
use majit_metainterp::{JitArgKind, JitCodeSym, JitDriverStaticData, JitState};

use crate::state::{PyreEnv, PyreMeta};
use pyre_object::PyObjectRef;

pub const SORTKEY_PORTAL_KEY: &str = "builtins::sortkey_portal";
pub const SORTKEY_JD_INDEX: usize = 4;

/// Marker extras after the greens:
/// `sortkey_jmp.jit_merge_point(strategy_type, tp, i, list_w, keys_w, w_callable)`.
///
/// The marker's explicit red operands are this port's declaration of which
/// live variable is which. `jtransform.py` `autodetect_jit_markers_redvars`
/// rewrites them; `warmspot.py` `make_args_specification` consumes that
/// list. Seeders permute this declaration through `red_marker_positions`.
pub const SORTKEY_MARKER_REDS: [(&'static str, Type); 4] = [
    ("i", Type::Int),
    ("list_w", Type::Ref),
    ("keys_w", Type::Ref),
    ("w_callable", Type::Ref),
];

/// jd4 symbolic state at the `sortkey_jmp` merge point.
///
/// listobject.py names the greens `strategy_type` and `tp` and `reds='auto'`
/// the loop locals `i`, `list_w`, `w_callable`. pyre keeps keys in a parallel
/// list (`keys_w`) because TimSort compares shadow-stack slots.
pub struct SortkeySym {
    pub i: OpRef,
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
            .expect("jd4 portal jitcode must be registered");
        crate::jitcode_runtime::decoded_ops(&canonical.code)
            .find(|op| op.opname == "jit_merge_point")
            .expect("jd4 portal body must contain a jit_merge_point")
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

/// `listobject.py` `greens=['strategy_type','tp']` — two Ref greens.
pub const SORTKEY_GREEN_TYPES: [GreenType; 2] = [GreenType::Ref, GreenType::Ref];

/// Hash of the jd4 greens `(strategy_type, tp)`.
#[inline]
pub fn sortkey_green_hash(strategy_type: PyObjectRef, tp: PyObjectRef) -> u64 {
    majit_metainterp::green_key_hash_typed(&[strategy_type as i64, tp as i64], &SORTKEY_GREEN_TYPES)
}

/// `warmstate.py JitCell.get_jitcell`: resolve once and carry the cell key.
///
/// `JitCell.get_jitcell(*greenargs)` compares the live greens in place, so
/// this builds no [`GreenKey`]. Function-entry uses the same
/// `resolve_cell_key_with_greenargs` door (`eval.rs`
/// `tick_function_entry_from_portal`).
#[inline]
pub fn sortkey_resolved_cell_key(
    warm: &majit_metainterp::warmstate::WarmEnterState,
    strategy_type: PyObjectRef,
    tp: PyObjectRef,
) -> u64 {
    let greenargs = [strategy_type as i64, tp as i64];
    let hash = sortkey_green_hash(strategy_type, tp);
    warm.resolve_cell_key_with_greenargs(hash, &greenargs, &SORTKEY_GREEN_TYPES)
}

fn sortkey_recorded_reds() -> Vec<(&'static str, Type)> {
    crate::jitcode_runtime::compiled_reds(SORTKEY_PORTAL_KEY, &SORTKEY_MARKER_REDS)
}

fn sortkey_marker_boxes(sym: &SortkeySym) -> [(OpRef, Type); 4] {
    [
        (sym.i, SORTKEY_MARKER_REDS[0].1),
        (sym.list_w, SORTKEY_MARKER_REDS[1].1),
        (sym.keys_w, SORTKEY_MARKER_REDS[2].1),
        (sym.w_callable, SORTKEY_MARKER_REDS[3].1),
    ]
}

fn sortkey_sym_from_recorded() -> SortkeySym {
    let mut i = None;
    let mut list_w = None;
    let mut keys_w = None;
    let mut w_callable = None;
    for (slot, (name, ty)) in sortkey_recorded_reds().into_iter().enumerate() {
        let op = OpRef::input_arg_typed(slot as u32, ty);
        match name {
            "i" => i = Some(op),
            "list_w" => list_w = Some(op),
            "keys_w" => keys_w = Some(op),
            "w_callable" => w_callable = Some(op),
            other => panic!("unknown sortkey marker red `{other}`"),
        }
    }
    SortkeySym {
        // Dead across the merge point: not an InputArg, never selected.
        i: i.unwrap_or_else(|| OpRef::input_arg_typed(0, Type::Int)),
        list_w: list_w.expect("sortkey marker red `list_w`"),
        keys_w: keys_w.expect("sortkey marker red `keys_w`"),
        w_callable: w_callable.expect("sortkey marker red `w_callable`"),
    }
}

/// Reds in rewritten merge-point order, from marker extras.
pub fn sortkey_live_values(
    i: i64,
    list_w: PyObjectRef,
    keys_w: PyObjectRef,
    w_callable: PyObjectRef,
) -> Vec<Value> {
    crate::jitcode_runtime::permute_marker_reds(
        SORTKEY_PORTAL_KEY,
        &[
            Value::Int(i),
            Value::Ref(GcRef(list_w as usize)),
            Value::Ref(GcRef(keys_w as usize)),
            Value::Ref(GcRef(w_callable as usize)),
        ],
    )
}

/// Trace `red_args` in rewritten slot order.
pub fn sortkey_trace_red_args(
    i: i64,
    list_w: PyObjectRef,
    keys_w: PyObjectRef,
    w_callable: PyObjectRef,
) -> Vec<(JitArgKind, OpRef, i64)> {
    let live = sortkey_live_values(i, list_w, keys_w, w_callable);
    sortkey_recorded_reds()
        .into_iter()
        .enumerate()
        .map(|(slot, (_, ty))| {
            (
                JitArgKind::from_type(ty).expect("sortkey red is I/R/F"),
                OpRef::input_arg_typed(slot as u32, ty),
                crate::jitcode_runtime::red_value_word(&live[slot]),
            )
        })
        .collect()
}

/// CRN I/R banks → marker extras `(i, list_w, keys_w, w_callable)`.
///
/// `jitexc.py` `ContinueRunningNormally` carries kind banks;
/// `warmspot.py` `handle_jitexception` walks the rewritten marker list.
pub fn sortkey_portal_refs(
    ints: &[i64],
    refs: &[i64],
) -> (i64, PyObjectRef, PyObjectRef, PyObjectRef) {
    let live =
        crate::jitcode_runtime::compiled_red_values_from_kind_banks(SORTKEY_PORTAL_KEY, ints, refs);
    let marker = crate::jitcode_runtime::marker_reds_from_recorded(SORTKEY_PORTAL_KEY, &live);
    let ref_at = |idx: usize| match marker.get(idx) {
        Some(Some(Value::Ref(GcRef(ptr)))) => *ptr as PyObjectRef,
        other => panic!("sortkey marker red {idx}, got {other:?}"),
    };
    let i = match marker.get(0) {
        Some(Some(Value::Int(v))) => *v,
        Some(None) | None => 0,
        other => panic!("sortkey marker red 0, got {other:?}"),
    };
    (i, ref_at(1), ref_at(2), ref_at(3))
}

impl SortkeyJitState {
    pub fn sortkey_driver_descriptor() -> JitDriverStaticData {
        let mut sd = JitDriverStaticData::new(
            vec![("strategy_type", Type::Ref), ("tp", Type::Ref)],
            sortkey_recorded_reds(),
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
        sortkey_sym_from_recorded()
    }
    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        let _ = meta;
        true
    }
    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        let _ = (meta, values);
    }
    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        crate::jitcode_runtime::permute_marker_reds(SORTKEY_PORTAL_KEY, &sortkey_marker_boxes(sym))
            .into_iter()
            .map(|(op, _)| op)
            .collect()
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
    fn jd4_build_time_descrs_resolve_through_global_pool() {
        let canonical = crate::jitcode_runtime::portal_jitcode_for_key(SORTKEY_PORTAL_KEY)
            .expect("jd4's extracted main JitCode must be registered");
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
