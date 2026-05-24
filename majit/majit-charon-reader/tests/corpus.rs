//! End-to-end test: load the charon-spike corpus, walk every function,
//! confirm every terminator/statement decodes into the typed enums.
//!
//! Run with: `cargo test -p majit-charon-reader`.

use majit_charon_reader::{
    Llbc,
    ullbc::{CallClass, StmtKind, TermKind},
};

const CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../charon-spike/corpus.ullbc",
);

#[test]
fn loads_spike_corpus() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    assert_eq!(llbc.crate_name(), "charon_spike_corpus");
    assert!(!llbc.file.has_errors);
    let local_count = llbc
        .iter_local_fns()
        .filter(|f| {
            f.item_meta
                .name_path()
                .starts_with("charon_spike_corpus::")
        })
        .count();
    assert_eq!(local_count, 5, "5 local fns expected");
}

#[test]
fn every_corpus_function_decodes() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");

    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        let Some(u) = fd.unstructured() else { continue };

        for (bb_idx, bb) in u.body.iter().enumerate() {
            for (s_idx, st) in bb.statements.iter().enumerate() {
                let stmt = st.stmt_kind().unwrap_or_else(|e| {
                    panic!("stmt decode failed in {name} bb{bb_idx} stmt{s_idx}: {e}")
                });
                assert!(
                    !matches!(stmt, StmtKind::Unknown),
                    "Unknown StmtKind in {name} bb{bb_idx} stmt{s_idx}",
                );
            }
            let term = bb.term().unwrap_or_else(|e| {
                panic!("terminator decode failed in {name} bb{bb_idx}: {e}")
            });
            assert!(
                !matches!(term, TermKind::Unknown),
                "Unknown TermKind in {name} bb{bb_idx}",
            );
        }
    }
}

#[test]
fn straight_line_add_shape() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc
        .local_fn("straight_line_add")
        .expect("function present");
    let u = fd.unstructured().expect("Unstructured body");
    assert_eq!(u.locals.arg_count, 3);
    assert_eq!(u.body.len(), 5);

    // bb0 should end in an overflow Assert (AddChecked + Assert).
    let bb0 = &u.body[0];
    assert!(
        matches!(bb0.term().unwrap(), TermKind::Assert { .. }),
        "bb0 terminator was not Assert",
    );

    // bb4 should be the return block.
    let bb4 = &u.body[4];
    assert!(matches!(bb4.term().unwrap(), TermKind::Return));
}

#[test]
fn branch_loop_sum_has_switch_int_and_switch_if() {
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");
    let fd = llbc
        .local_fn("branch_loop_sum")
        .expect("function present");
    let u = fd.unstructured().expect("Unstructured body");

    let mut saw_switch_int = false;
    let mut saw_switch_if = false;
    for bb in &u.body {
        if let Ok(TermKind::Switch { targets, .. }) = bb.term() {
            match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(..) => saw_switch_if = true,
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(..) => {
                    saw_switch_int = true
                }
            }
        }
    }
    assert!(saw_switch_int, "expected SwitchInt in branch_loop_sum");
    assert!(saw_switch_if, "expected If switch in branch_loop_sum");
}

#[test]
fn call_classify_covers_corpus() {
    use std::collections::BTreeMap;
    let llbc = Llbc::load(CORPUS).expect("load corpus.ullbc");

    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for fd in llbc.iter_local_fns() {
        let Some(u) = fd.unstructured() else { continue };
        for bb in &u.body {
            if let Ok(TermKind::Call { call, .. }) = bb.term() {
                let label = match call.func.classify() {
                    CallClass::Direct => "direct",
                    CallClass::Trait => "trait",
                    CallClass::Dynamic => "dynamic",
                    CallClass::Ptr => "ptr",
                    CallClass::Unknown => "unknown",
                };
                *counts.entry(label).or_default() += 1;
            }
        }
    }
    // The corpus is straightforward Rust — every call should classify
    // as Direct (or Trait for the iterator `?` desugaring helpers).
    let unknown = counts.get("unknown").copied().unwrap_or(0);
    assert_eq!(unknown, 0, "corpus should not produce unknown call classifications: {counts:?}");
    assert!(
        counts.get("direct").copied().unwrap_or(0) > 0,
        "expected at least one direct call: {counts:?}",
    );
}
