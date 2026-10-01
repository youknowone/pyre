//! A bracket in a crate that only imports the root-stack API can be erased
//! once the effects of the crates it links against are published.
//!
//! Without them every body-less callee in `pyre-interpreter` -- `core`'s
//! `ptr::is_null` as much as `pyre_object`'s `is_tagged_int` -- is an unknown
//! that may leave the stack changed, so no bracket there whose body calls
//! anything foreign is ever proved balanced.  The harvest publishes
//! `pyre-object`'s own answers; `core` and the other crates outside the
//! translation input cannot name the API at all.

use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};
use majit_charon_reader::Llbc;
use majit_translate::front::mir::{
    erased_root_bracket_guards, fn_returns_owned_scope, harvest_root_stack_touching_paths,
    harvest_scope_owning_paths, scope_owning_key,
};

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

/// Bodies holding at least one erased bracket, over the first [`SAMPLE`]
/// bodies that open one.
fn erased_bodies(llbc: &Llbc) -> (usize, usize) {
    let (mut opening, mut erased) = (0, 0);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let opens = body.body.iter().any(|bb| {
            matches!(bb.term(llbc), Ok(TermKind::Call { call, .. })
                if matches!(&call.func, CallFunc::Regular(reg)
                    if matches!(&reg.kind, CallKind::Fun(FunId::Regular { id })
                        if llbc.fn_by_id(*id).is_some_and(|f| f.item_meta.name_path() == "pyre_object::gc_roots::push_roots"))))
        });
        if !opens {
            continue;
        }
        if opening == SAMPLE {
            break;
        }
        opening += 1;
        if !erased_root_bracket_guards(llbc, &body).is_empty() {
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

#[test]
fn a_bodyless_rooted_items_new_returns_the_scope_its_defining_crate_published() {
    if !std::path::Path::new(OBJECT_LLBC).is_file()
        || !std::path::Path::new(INTERPRETER_LLBC).is_file()
    {
        eprintln!("skipping: run `python3 scripts/extract-llbc.py`");
        return;
    }
    let object = Llbc::load(OBJECT_LLBC).expect("load pyre-object");
    let owning = harvest_scope_owning_paths(&object);
    assert!(
        owning.iter().any(|p| p.ends_with("::RootedItems::new")),
        "RootedItems::new opens push_roots and returns that guard: {owning:?}"
    );
    assert!(
        owning
            .iter()
            .any(|p| p.ends_with("::DictOperationGuard::new")),
        "DictOperationGuard::new returns the bracket it opened: {owning:?}"
    );
    drop(object);

    let mut interpreter = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter");
    let external_id = interpreter
        .iter_fun_decls()
        .find(|fd| {
            scope_owning_key(&interpreter, fd)
                .is_some_and(|key| key.ends_with("::RootedItems::new"))
        })
        .expect("interpreter names RootedItems::new")
        .def_id;
    for fd in interpreter.file.translated.fun_decls.iter_mut().flatten() {
        if fd.def_id == external_id {
            fd.body = None;
        }
    }
    assert!(
        interpreter
            .fn_by_id(external_id)
            .is_some_and(|fd| fd.body.is_none()),
        "the declaration under test has no body"
    );
    assert!(
        !fn_returns_owned_scope(&interpreter, external_id),
        "without the published constructors the opaque new is not a guard"
    );
    interpreter.set_scope_owning_constructors(owning);
    assert!(
        fn_returns_owned_scope(&interpreter, external_id),
        "the published RootedItems::new is the guard its caller holds"
    );
}
