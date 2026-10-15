//! `IdentityDictStrategy::length` is `identity_storage(w_dict).len()`.
//!
//! `rordereddict.rtype_len` / `ll_dict_len` emit `getfield(d, "num_live_items")`
//! when the table has no custom eq/hash. The identity storage is an
//! identity-keyed `dicttable` (`mark_dict_non_null`); `.len()` must be that
//! header word, not `ll_vec_length` of the first word (the `indexes` pointer
//! under `#[repr(C)]`, or the `RandomState` keys without it).

use std::collections::HashSet;
use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerContext, lower_fun_decl, lower_function};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

fn object_llbc() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| {
        assert!(
            std::path::Path::new(OBJECT_LLBC).is_file(),
            "{OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`"
        );
        Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc")
    })
}

fn is_ll_vec_length(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().is_some_and(|leaf| {
            leaf == "ll_vec_length_r" || leaf == "ll_vec_length_i" || leaf == "ll_vec_length_f"
        }),
        _ => false,
    }
}

#[test]
fn identity_length_reads_num_live_items() {
    let llbc = object_llbc();
    let graph = lower_function(llbc, "identitydict::<Impl>::length")
        .expect("lower identitydict::<Impl>::length");
    if let Ok(path) = std::env::var("DUMP_IDENTITY_LENGTH") {
        std::fs::write(&path, graph.dump()).expect("write identity length dump");
    }
    assert_num_live_items_getfield(llbc, &graph);
}

fn callee_fun_decl_ids(graph: &FunctionGraph) -> Vec<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call { target, .. } => target.fun_decl_id(),
            _ => None,
        })
        .collect()
}

fn count_length_ops(graph: &FunctionGraph, live_reads: &mut usize, ll_vec_length: &mut usize) {
    for op in graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
    {
        match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "num_live_items" => {
                *live_reads += 1;
            }
            OpKind::Call { target, .. } if is_ll_vec_length(target) => {
                *ll_vec_length += 1;
            }
            _ => {}
        }
    }
}

fn assert_num_live_items_getfield(llbc: &Llbc, graph: &FunctionGraph) {
    let mut live_reads = 0usize;
    let mut ll_vec_length = 0usize;
    count_length_ops(graph, &mut live_reads, &mut ll_vec_length);
    let mut seen = HashSet::new();
    let mut pending = callee_fun_decl_ids(graph);
    let context = LowerContext::new(llbc);
    let mut dumps = vec![graph.dump()];
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        let path = fd.item_meta.name_path();
        if !(path.contains("rordereddict") || path.contains("identitydict")) {
            continue;
        }
        let Ok(callee) = lower_fun_decl(&context, fd) else {
            continue;
        };
        count_length_ops(&callee, &mut live_reads, &mut ll_vec_length);
        pending.extend(callee_fun_decl_ids(&callee));
        dumps.push(callee.dump());
    }
    let dump = dumps.join("\n--- callee ---\n");
    assert!(
        live_reads > 0,
        "identity length must getfield num_live_items (`rordereddict.rtype_len`) somewhere in its call chain\n{dump}"
    );
    assert_eq!(
        ll_vec_length, 0,
        "identity length must not call ll_vec_length of the first storage word\n{dump}"
    );
}
