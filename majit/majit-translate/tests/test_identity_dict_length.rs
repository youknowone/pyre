//! `IdentityDictStrategy::length` is `identity_storage(w_dict).len()`.
//!
//! `rordereddict.rtype_len` / `ll_dict_len` emit `getfield(d, "num_live_items")`
//! when the table has no custom eq/hash. The identity storage is an
//! identity-keyed `dicttable` (`mark_dict_non_null`); `.len()` must be that
//! header word, not `ll_vec_length` of the first word (the `indexes` pointer
//! under `#[repr(C)]`, or the `RandomState` keys without it).

use std::sync::OnceLock;

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

fn object_llbc() -> Option<&'static Llbc> {
    static LLBC: OnceLock<Option<Llbc>> = OnceLock::new();
    LLBC.get_or_init(|| {
        if !std::path::Path::new(OBJECT_LLBC).is_file() {
            eprintln!("skipping: {OBJECT_LLBC} is missing; run `python3 scripts/extract-llbc.py`");
            return None;
        }
        Some(Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc"))
    })
    .as_ref()
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
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "identitydict::<Impl>::length")
        .expect("lower identitydict::<Impl>::length");
    if let Ok(path) = std::env::var("DUMP_IDENTITY_LENGTH") {
        std::fs::write(&path, graph.dump()).expect("write identity length dump");
    }
    assert_num_live_items_getfield(&graph);
}

fn assert_num_live_items_getfield(graph: &FunctionGraph) {
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    let mut live_reads = 0usize;
    let mut ll_vec_length = 0usize;
    for op in &ops {
        match &op.kind {
            OpKind::FieldRead { field, .. } if field.name == "num_live_items" => {
                live_reads += 1;
            }
            OpKind::Call { target, .. } if is_ll_vec_length(target) => {
                ll_vec_length += 1;
            }
            _ => {}
        }
    }
    assert!(
        live_reads > 0,
        "identity length must getfield num_live_items (`rordereddict.rtype_len`)\n{}",
        graph.dump()
    );
    assert_eq!(
        ll_vec_length,
        0,
        "identity length must not call ll_vec_length of the first storage word\n{}",
        graph.dump()
    );
}
