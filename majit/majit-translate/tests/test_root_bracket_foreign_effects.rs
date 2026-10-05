//! A bracket in a crate that only imports the root-stack API can be erased
//! once the effects of the crates it links against are published.
//!
//! Without them every body-less callee in `pyre-interpreter` -- `core`'s
//! `ptr::is_null` as much as `pyre_object`'s `is_tagged_int` -- is an unknown
//! that may leave the stack changed, so no bracket there whose body calls
//! anything foreign is ever proved balanced.  The harvest publishes
//! `pyre-object`'s own answers; `core` and the other crates outside the
//! translation input cannot name the API at all.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};
use majit_translate::front::mir::{erased_root_bracket_guards, harvest_root_stack_touching_paths};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);
const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// Bodies that open a bracket, sampled: every `erased_root_bracket_guards`
/// call starts a fresh root-stack analysis, so the whole population costs
/// minutes.
const SAMPLE: usize = 200;

fn callee_name_paths<'a>(
    llbc: &'a Llbc,
    body: &'a majit_charon_reader::ullbc::Unstructured,
) -> impl Iterator<Item = String> + 'a {
    body.body.iter().filter_map(move |bb| {
        let Ok(TermKind::Call { call, .. }) = bb.term(llbc) else {
            return None;
        };
        let CallFunc::Regular(reg) = &call.func else {
            return None;
        };
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        llbc.fn_by_id(*id).map(|f| f.item_meta.name_path())
    })
}

fn path_is_gc_roots_leaf(path: &str, leaf: &str) -> bool {
    path.rsplit("::").next() == Some(leaf) && path.split("::").any(|s| s == "gc_roots")
}

/// True when every terminator callee outside `gc_roots` lives in a crate the
/// harvest can name. An interpreter-local helper is analysed from its own
/// body, so publishing `pyre-object` cannot change whether it touches.
fn only_foreign_non_root_callees(paths: &[String]) -> bool {
    paths.iter().all(|path| {
        path.split("::").any(|s| s == "gc_roots")
            || path.split("::").next() != Some("pyre_interpreter")
    })
}

/// Bodies holding at least one erased bracket, over the first [`SAMPLE`]
/// bodies that open one and whose other callees are all foreign. Those are
/// the bodies whose unknown `pyre-object` / `core` callees the harvest can
/// answer.
fn erased_bodies(llbc: &Llbc) -> (usize, usize) {
    let (mut opening, mut erased) = (0, 0);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let paths: Vec<String> = callee_name_paths(llbc, &body).collect();
        if !paths.iter().any(|p| path_is_gc_roots_leaf(p, "push_roots")) {
            continue;
        }
        if !only_foreign_non_root_callees(&paths) {
            continue;
        }
        if opening == SAMPLE {
            break;
        }
        opening += 1;
        if !erased_root_bracket_guards(llbc, fd, &body).is_empty() {
            erased += 1;
        }
    }
    (opening, erased)
}

#[test]
fn published_dependency_effects_let_an_importing_crate_erase_its_brackets() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let touching = harvest_root_stack_touching_paths(&object);
    assert!(
        touching
            .iter()
            .any(|p| p == "pyre_object::gc_roots::<Impl>::pin_roots"),
        "the root-stack API itself must stay a touching body"
    );
    assert!(
        !touching
            .iter()
            .any(|p| p == "pyre_object::intobject::w_int_new"),
        "an allocation leaves the root stack as it found it"
    );
    let crate_name = object.crate_name().to_string();
    drop(object);

    let interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let (opening, before) = erased_bodies(&interpreter);
    interpreter.set_root_stack_effects(vec![crate_name], touching);
    let (_, after) = erased_bodies(&interpreter);
    eprintln!("bodies opening a bracket: {opening}; erased before {before}, after {after}");
    assert!(
        after > before,
        "publishing pyre-object's effects must let more interpreter brackets go"
    );
}
