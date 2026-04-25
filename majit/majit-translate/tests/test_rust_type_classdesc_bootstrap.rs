//! Parity-proof: a synthetic `HostObject::Class` representing a
//! Rust-source type (struct or enum variant) flows through the
//! existing `Bookkeeper::getdesc` → `ClassDesc::new` →
//! `getuniqueclassdef` → `ClassDef` chain **without any new
//! primitives**. The existing infrastructure — ported under
//! `annotator/bookkeeper.rs`, `annotator/classdesc.rs`, and
//! `flowspace/model.rs` — already handles classes with
//! empty-or-explicit `bases`, no `__NOT_RPYTHON__` marker, and no
//! `_annspecialcase_` tag, which is exactly what a pyre Rust-source
//! class synthesis produces.
//!
//! This is the **deepest dependency leaf** of the annotator-
//! monomorphization plan
//! (`~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`).
//! The M3 goal
//!   `FunctionDesc.specialize(execute_opcode_step,
//!       [SomeInstance(classdef=PyFrame)])`
//! needs `SomeInstance(classdef=PyFrame)`, which needs a ClassDef for
//! `PyFrame`, which needs a ClassDesc, which needs a `HostObject`. If
//! synthesizing that `HostObject::Class` and feeding it to
//! `Bookkeeper.getuniqueclassdef` already works, then every M3 / M2.5d
//! slice 2 milestone below reduces to WIRING the existing pieces —
//! not adding new primitives.
//!
//! Upstream pipeline:
//!   `bookkeeper.py:367-373 getdesc` (class branch)
//!     → `classdesc.py:488-557 ClassDesc.__init__` (baselist /
//!        `_immutable_fields_` / mixin gates; empty-base fall-through)
//!     → `classdesc.py:699-702 ClassDesc.getuniqueclassdef`
//!     → `classdesc.py:139-174 ClassDef.__init__`.

use std::rc::Rc;

use majit_translate::annotator::annrpython::RPythonAnnotator;
use majit_translate::annotator::bookkeeper::Bookkeeper;
use majit_translate::annotator::description::DescEntry;
use majit_translate::annotator::signature::AnnotationSpec;
use majit_translate::flowspace::model::HostObject;
use majit_translate::translator::interactive::Translation;

/// Bases-empty Rust-source class synthesis — mirrors `PyFrame`, the
/// concrete impl at `pyre/pyre-interpreter/src/eval.rs:764` that the
/// plan's M3 targets as the `execute_opcode_step<E>` specialization
/// key. Upstream `cls.__bases__` for a root-like class filters out
/// `object`, so `bases=[]` matches the post-filter state that
/// `classdesc.py:538-554 ClassDesc.__init__` expects (`base = object`
/// by default).
#[test]
fn rust_struct_bases_empty_reaches_classdef() {
    let pyframe = HostObject::new_class("pypy.interpreter.pyframe.PyFrame", vec![]);
    let bookkeeper = Rc::new(Bookkeeper::new());

    // `Bookkeeper.getdesc(&cls)` at `bookkeeper.rs:902-913` routes the
    // class branch into `ClassDesc::new`. If the synthesis went
    // through, the entry is `DescEntry::Class`.
    let desc_entry = bookkeeper
        .getdesc(&pyframe)
        .expect("synthetic class must bootstrap through getdesc");
    match desc_entry {
        DescEntry::Class(_) => {}
        other => panic!("expected DescEntry::Class, got {other:?}"),
    }

    // `getuniqueclassdef` wraps `getdesc` + `ClassDesc::getuniqueclassdef`
    // — upstream `classdesc.py:699-702`. A fresh synthetic class has
    // no existing classdef, so this forges one.
    let classdef = bookkeeper
        .getuniqueclassdef(&pyframe)
        .expect("getuniqueclassdef must produce a ClassDef");

    // The ClassDef's classdesc backref points at the same ClassDesc
    // `getdesc` cached. Identity check via qualname (structural equal
    // is sufficient — ClassDef.classdesc is Rc<RefCell<ClassDesc>>
    // shared with the cache entry).
    let cd_borrow = classdef.borrow();
    let classdesc_borrow = cd_borrow.classdesc.borrow();
    assert_eq!(
        classdesc_borrow.name, "pypy.interpreter.pyframe.PyFrame",
        "ClassDef's backref classdesc.name must match the synthesis qualname"
    );

    // Re-lookup must hit the cache — `descs: HashMap<HostObject,
    // DescEntry>` keys on Arc::ptr_eq.
    let again = bookkeeper
        .getdesc(&pyframe)
        .expect("second getdesc hits the cache");
    assert!(matches!(again, DescEntry::Class(_)));
}

/// Multi-level bases — models an enum variant like
/// `Instruction::ExtendedArg` whose `__class__` is the variant, and
/// whose `__bases__` contains the enum type `Instruction`. The ClassDesc
/// bootstrap must follow the base chain.
#[test]
fn rust_enum_variant_with_explicit_base_builds_basedesc() {
    let instruction_enum = HostObject::new_class("pyre.pyopcode.Instruction", vec![]);
    let extended_arg_variant = HostObject::new_class(
        "pyre.pyopcode.Instruction.ExtendedArg",
        vec![instruction_enum.clone()],
    );
    let bookkeeper = Rc::new(Bookkeeper::new());

    // Bootstrap the variant. `ClassDesc::new` recursively calls
    // `bookkeeper.getdesc(base)` for each base (classdesc.py:543-554
    // + the recursive `me.basedesc = bookkeeper.getdesc(base)` at
    // `classdesc.py:581`), so the enum type's ClassDesc is created
    // eagerly on this lookup.
    let classdef = bookkeeper
        .getuniqueclassdef(&extended_arg_variant)
        .expect("variant class must bootstrap through the base chain");

    // The variant ClassDef's classdesc must have a `basedesc` pointing
    // at the enum type's ClassDesc.
    let cd_borrow = classdef.borrow();
    let classdesc_borrow = cd_borrow.classdesc.borrow();
    let base_rc = classdesc_borrow
        .basedesc
        .as_ref()
        .expect("variant with explicit base must carry basedesc");
    let base_borrow = base_rc.borrow();
    assert_eq!(base_borrow.name, "pyre.pyopcode.Instruction");

    // The enum type's cache entry exists (created transitively).
    let enum_entry = bookkeeper
        .getdesc(&instruction_enum)
        .expect("enum type must have been cached during variant bootstrap");
    assert!(matches!(enum_entry, DescEntry::Class(_)));
}

/// The chain is idempotent: repeated `getuniqueclassdef` for the same
/// synthetic class returns the same ClassDef `Rc`. Upstream
/// `getuniqueclassdef` at `classdesc.py:699-702` stores the classdef on
/// `self.classdef` after first call and returns it on subsequent
/// invocations.
#[test]
fn repeat_getuniqueclassdef_is_rc_identical() {
    let t = HostObject::new_class("pyre.pyopcode.TestType", vec![]);
    let bookkeeper = Rc::new(Bookkeeper::new());

    let first = bookkeeper.getuniqueclassdef(&t).expect("first");
    let second = bookkeeper.getuniqueclassdef(&t).expect("second");

    assert!(
        Rc::ptr_eq(&first, &second),
        "getuniqueclassdef must return the same ClassDef Rc on repeat calls"
    );
}

/// Up one layer: `build_types(identity_fn, [UserClass(synthetic)])`
/// threads the synthetic class through `typeannotation`
/// (`signature.py:103-104`: `SomeInstance(getuniqueclassdef(t))`),
/// binds the startblock inputarg to that `SomeInstance`, and reaches
/// the returnblock Link arg carrying the same class identity.
///
/// This is the **payoff demonstration** of the deepest-leaf proof
/// above: once the ClassDesc chain validates, threading a concrete
/// classdef through `build_types` reduces to existing machinery.
/// The plan's M3 goal — `FunctionDesc.specialize(execute_opcode_step,
/// [SomeInstance(classdef=PyFrame)])` — extends this by the MATCH/
/// method-call lowering, not by new ClassDesc infrastructure.
#[test]
fn build_types_with_synthetic_userclass_arg_threads_someinstance_through() {
    let item = syn::parse_str::<syn::ItemFn>("fn identity(x: PyFrame) -> PyFrame { x }")
        .expect("fixture parses");
    let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

    // Synthesize `PyFrame` as a bases-empty Class. Upstream path:
    // `AnnotationSpec::UserClass(cls)` → `typeannotation` →
    // `bookkeeper.getuniqueclassdef(cls)` → `SomeInstance(classdef=cls)`.
    let pyframe = HostObject::new_class("pyre.interpreter.PyFrame", vec![]);

    let ann = RPythonAnnotator::new_with_translator(Some(Rc::clone(&t.context)), None, None, false);
    let result = ann
        .build_types(
            &host,
            &[AnnotationSpec::UserClass(pyframe.clone())],
            true,
            false,
        )
        .expect("build_types with UserClass arg must succeed");

    // The returnblock Link arg is the startblock's inputarg (since
    // `identity` just returns `x`). After `build_types` binds the
    // input to `SomeInstance(classdef=PyFrame_classdef)` and the
    // propagation completes, the return annotation must be the same
    // `SomeInstance` carrying the `PyFrame` classdef.
    let sv = result.expect("identity fn must return an annotation");
    let ty = format!("{sv:?}");
    assert!(
        ty.contains("Instance") && ty.contains("PyFrame"),
        "expected SomeInstance carrying PyFrame classdef, got {ty}"
    );

    // Cross-check: the classdef inside the returned SomeValue must be
    // the SAME ClassDef that `bookkeeper.getuniqueclassdef(&pyframe)`
    // produces — `Rc::ptr_eq` identity. This validates the M3
    // assumption that "classdef is the specialization key" works on
    // synthetic Rust-source classes with no other adaptations.
    let expected_classdef = ann
        .bookkeeper
        .getuniqueclassdef(&pyframe)
        .expect("classdef lookup");
    let got_classdef = match &sv {
        majit_translate::annotator::model::SomeValue::Instance(inst) => {
            inst.classdef.clone().expect("classdef set on SomeInstance")
        }
        other => panic!("expected SomeValue::Instance, got {other:?}"),
    };
    assert!(
        Rc::ptr_eq(&expected_classdef, &got_classdef),
        "returned SomeInstance.classdef must be Rc-identical to \
         bookkeeper.getuniqueclassdef(&pyframe)"
    );
}
