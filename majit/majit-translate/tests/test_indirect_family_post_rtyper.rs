//! What cel's `CallTarget::Indirect` vtable sites become **after**
//! `translator::rtyper::rpbc::lower_indirect_calls` runs.
//!
//! `test_cel_census.rs` classifies **pre**-rtyper: it counts `CallTarget::Indirect`
//! as emitted by `front::mir::lower_fun_decl`, before any pass has resolved a
//! family. Its own legend calls `IndirectCall { graphs: None }` a **WALL**, so the
//! headline "6 walls" is only correct if those `Indirect` sites acquire a non-empty
//! family. Nothing had measured that. This does.
//!
//! ```sh
//! cargo test --release -p majit-translate --test test_indirect_family_post_rtyper \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! Same `--ignored` hole as the census: the `ignore` is
//! `cfg_attr(debug_assertions, …)`, so a release run does not ignore these and
//! `--ignored` would select nothing while printing what looks like a pass.
//!
//! ## What is real and what is reconstructed
//!
//! * **Real**: `lower_fun_decl`, `lower_indirect_calls`, `CallControl`,
//!   `register_trait_method`, and the `family.is_empty() -> None` fold — all the
//!   production functions, called directly.
//! * **Reconstructed**: the *registration order* the production pipeline performs
//!   at `lib.rs:1236-1311`, replayed here from the same `SemanticProgram` the
//!   pipeline builds. That chain has exactly one writer end to end:
//!   `all_impls_for_indirect` reads `trait_method_impls` (`call.rs:4678-4685`),
//!   written only by `register_trait_method` when `trait_root.is_some()`
//!   (`call.rs:2700-2709`), whose sole production caller is `lib.rs:1311` over
//!   `canonical_trait_impls`, itself written at exactly one site — `lib.rs:1038`,
//!   the `(self_ty_root: None, trait_root: Some(_))` arm, i.e. **trait default
//!   bodies only**. `replay_trait_method_registration` below is that chain.
//!
//! The reconstruction is reported non-vacuously: the probe prints how many
//! families it registered and how many distinct traits they cover, so a silently
//! empty replay is visible as `registered families 0` rather than being
//! indistinguishable from "the artefact has no default bodies". It is also
//! non-vacuous in the answer: on `cel.ullbc` three sites resolve to a non-empty
//! family and fourteen do not, from one replay in one run, so the instrument is
//! demonstrably able to produce either verdict.
//!
//! ## What it measured, 2026-08-08, on a fresh `cel.ullbc` (`--check cel` exits 0)
//!
//! ```text
//!   PRE-rtyper  CallTarget::Indirect   17
//!        1  AsDebug::as_debug            family=1  -> Some(..)
//!        7  AsKeyRef::as_keyref          family=0  -> None   (WALL)
//!        7  Opaque::runtime_type_name    family=0  -> None   (WALL)
//!        1  OpaqueEq::opaque_eq          family=1  -> Some(..)
//!        1  VariableResolver::resolve    family=1  -> Some(..), omits 2 concrete impls
//!   POST-rtyper: graphs=None 14 | Some(non-empty) 3 | Some(empty) 0
//! ```
//!
//! Two findings, both structural rather than cel-specific:
//!
//! 1. **A concrete `impl Trait for Type` never enters `trait_method_impls`.**
//!    `lib.rs:1002-1031` routes it to `canonical_inherent_methods` on purpose —
//!    the comment there says registering it through `register_trait_method` would
//!    also seed `method_to_impl_types` and flip `resolve_method`'s name-based
//!    lookup for every same-named method. So a trait method reached through a
//!    vtable gets an EMPTY family unless something else filed a body for it, and
//!    `rpbc.rs:417` folds empty to `None` — a wall. `AsKeyRef` and `Opaque` are
//!    absent from the replayed trait list entirely, so their empty family is a
//!    genuine absence and not a key-spelling mismatch.
//! 2. **What does get filed under `<default methods of T>` is not a trait default.**
//!    None of `AsDebug` / `OpaqueEq` / `VariableResolver` declares a default body
//!    (`cel/src/objects.rs:552-583`, `cel/src/context.rs:281-283` are all bare
//!    signatures). The bodies filed are BLANKET impls — `impl<T> AsDebug for T`,
//!    `impl<T> OpaqueEq for T`, `impl<T: VariableResolver> VariableResolver for
//!    &T` — whose Charon self type is a type parameter, so `self_ty_root` is
//!    `None` and they land in the trait-default arm. For the first two the blanket
//!    impl is the only impl and `Some([it])` is right by accident; for
//!    `VariableResolver::resolve` the family names the `&T` forwarder while
//!    `Box<T>`, `Arc<T>` and every user impl are omitted. That is a positive claim
//!    about a callee set that is measurably incomplete — the opposite failure
//!    direction from the empty-family fold, and it runs in production.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use majit_charon_reader::Llbc;
use majit_translate::codewriter::call::CallControl;
use majit_translate::front::mir::{build_semantic_program_from_llbcs, lower_fun_decl};
use majit_translate::translator::rtyper::rpbc::lower_indirect_calls;
use majit_translate::{CallTarget, FunctionGraph, OpKind};

/// `cel-jit/build/llbc/cel.ullbc`, or `CEL_CENSUS_LLBC` — the same resolution
/// `test_cel_census.rs` uses, so both probes read the same artefact by default.
fn cel_llbc_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CEL_CENSUS_LLBC") {
        return Some(PathBuf::from(p));
    }
    let path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../cel-jit/build/llbc/cel.ullbc"
    ));
    path.exists().then_some(path)
}

/// Replay `lib.rs:1038` + `lib.rs:1236-1311` — the only chain that ever writes
/// `trait_method_impls`. Returns `(families_registered, traits_covered)` so the
/// caller can report the replay's own cardinality.
fn replay_trait_method_registration(
    call_control: &mut CallControl,
    program: &majit_translate::SemanticProgram,
    // Every replayed registration, as `(trait, method) -> [module_path::name]`, so
    // the report can say WHICH body was filed under `<default methods of T>`.
    provenance: &mut BTreeMap<(String, String), Vec<String>>,
) -> (usize, BTreeSet<String>) {
    let mut families = 0usize;
    let mut traits = BTreeSet::new();
    for func in &program.functions {
        // `lib.rs:1037` — the trait default-body arm is the only one that reaches
        // `canonical_trait_impls`. A concrete `impl Trait for Type` is
        // `(Some(owner), Some(trait_leaf))` and is registered as an *inherent*
        // method instead (`lib.rs:1024-1031`), deliberately, so that it does not
        // seed `method_to_impl_types`.
        let (None, Some(trait_leaf)) = (&func.self_ty_root, &func.trait_root) else {
            continue;
        };
        // `lib.rs:1040` mints the pseudo impl type; `lib.rs:1237-1248` reads it
        // back as `impl_type` (because `self_ty_root` is `None`) and passes
        // `trait_root = Some(trait_leaf)`.
        let impl_type = format!("<default methods of {trait_leaf}>");
        call_control.register_trait_method(
            &func.name,
            Some(trait_leaf.as_str()),
            &impl_type,
            func.graph.clone(),
        );
        families += 1;
        traits.insert(trait_leaf.clone());
        provenance
            .entry((trait_leaf.clone(), func.name.clone()))
            .or_default()
            .push(format!("{}::{}", func.module_path, func.name));
    }
    (families, traits)
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: lowers the whole cel LLBC; use `cargo test --release --test test_indirect_family_post_rtyper`"
)]
fn cel_indirect_sites_after_lower_indirect_calls() {
    let Some(path) = cel_llbc_path() else {
        eprintln!(
            "skipping: cel.ullbc missing — run `python3 scripts/extract-llbc.py cel` \
             in cel-jit, or set CEL_CENSUS_LLBC"
        );
        return;
    };
    let llbc = Llbc::load(&path).expect("load cel llbc");

    let program = build_semantic_program_from_llbcs(std::slice::from_ref(&llbc))
        .expect("build semantic program from cel llbc");
    let mut call_control = CallControl::new();
    let mut provenance: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let (registered, traits_covered) =
        replay_trait_method_registration(&mut call_control, &program, &mut provenance);

    // The concrete impls of each `(trait, method)` — `lib.rs:1002-1031`'s
    // `(Some(owner), Some(trait_leaf))` arm. These are registered as *inherent*
    // methods and deliberately never reach `trait_method_impls`, so they are
    // exactly the callees a vtable family does NOT list. Counted here so the
    // report can say whether a `Some(family)` under-describes the callee set.
    let mut concrete_impls: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for func in &program.functions {
        if let (Some(owner), Some(trait_leaf)) = (&func.self_ty_root, &func.trait_root) {
            concrete_impls
                .entry((trait_leaf.clone(), func.name.clone()))
                .or_default()
                .push(owner.clone());
        }
    }

    // Lower every local body, exactly as the census does.
    let mut graphs: Vec<FunctionGraph> = Vec::new();
    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        if fd.unstructured().is_none() {
            continue;
        }
        if let Ok(graph) = lower_fun_decl(&llbc, fd) {
            graphs.push(graph);
        }
    }

    // PRE-rtyper: the census's own classification.
    let mut pre: BTreeMap<String, usize> = BTreeMap::new();
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::Call {
                    target:
                        CallTarget::Indirect {
                            trait_root,
                            method_name,
                        },
                    ..
                } = &op.kind
                {
                    *pre.entry(format!("{trait_root}::{method_name}"))
                        .or_default() += 1;
                }
            }
        }
    }

    // The family each pre-rtyper site will be handed, read from the same
    // accessor `lower_indirect_calls` reads (`rpbc.rs:416`).
    let mut family_sizes: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::Call {
                    target:
                        CallTarget::Indirect {
                            trait_root,
                            method_name,
                        },
                    ..
                } = &op.kind
                {
                    let family = call_control.all_impls_for_indirect(trait_root, method_name);
                    family_sizes
                        .entry(format!("{trait_root}::{method_name}"))
                        .or_insert_with(|| {
                            (
                                family.len(),
                                family.iter().map(|p| p.segments.join("::")).collect(),
                            )
                        });
                }
            }
        }
    }

    // POST-rtyper: run the production pass and classify what it produced.
    let mut post_none = 0usize;
    let mut post_some_nonempty = 0usize;
    let mut post_some_empty = 0usize;
    let mut surviving_indirect_targets = 0usize;
    // Pre-existing `IndirectCall` ops (the fn-pointer arm) would otherwise be
    // counted as if the vtable pass had produced them, so subtract the baseline.
    let mut baseline_indirect_ops = 0usize;
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                if matches!(op.kind, OpKind::IndirectCall { .. }) {
                    baseline_indirect_ops += 1;
                }
            }
        }
    }
    let mut lowered = graphs.clone();
    for graph in &mut lowered {
        lower_indirect_calls(graph, &call_control);
    }
    for graph in &lowered {
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::IndirectCall { graphs, .. } => match graphs {
                        None => post_none += 1,
                        Some(c) if c.is_empty() => post_some_empty += 1,
                        Some(_) => post_some_nonempty += 1,
                    },
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    } => surviving_indirect_targets += 1,
                    _ => {}
                }
            }
        }
    }

    let pre_total: usize = pre.values().sum();
    eprintln!("=== cel post-rtyper vtable census: {} ===", path.display());
    eprintln!(
        "pid {}  PYRE_FNPTR_INDIRECT={}",
        std::process::id(),
        std::env::var("PYRE_FNPTR_INDIRECT").unwrap_or_else(|_| "<unset>".into()),
    );
    eprintln!("bodies lowered                    {}", graphs.len());
    eprintln!(
        "replayed trait-default registrations {registered} over {} traits",
        traits_covered.len()
    );
    eprintln!("PRE-rtyper  CallTarget::Indirect   {pre_total}");
    for (name, n) in &pre {
        let (size, members) = family_sizes.get(name).cloned().unwrap_or((0, Vec::new()));
        let (trait_leaf, method) = name.split_once("::").unwrap_or((name.as_str(), ""));
        let owners = concrete_impls
            .get(&(trait_leaf.to_string(), method.to_string()))
            .cloned()
            .unwrap_or_default();
        let verdict = if size == 0 {
            "family EMPTY -> graphs: None  == the census's WALL class".to_string()
        } else if owners.is_empty() {
            "family non-empty -> graphs: Some(..)".to_string()
        } else {
            format!(
                "family non-empty -> graphs: Some(..)  BUT omits {} concrete impl(s)",
                owners.len()
            )
        };
        eprintln!("      {n:6}  {name}   family={size} {members:?}  {verdict}");
        eprintln!(
            "              concrete impls NOT in the family ({}): {owners:?}",
            owners.len()
        );
        let filed = provenance
            .get(&(trait_leaf.to_string(), method.to_string()))
            .cloned()
            .unwrap_or_default();
        eprintln!(
            "              bodies filed under `<default methods of {trait_leaf}>` ({}): {filed:?}",
            filed.len()
        );
    }
    eprintln!(
        "traits with a replayed `<default methods of _>` registration ({}): {:?}",
        traits_covered.len(),
        traits_covered
    );
    eprintln!("baseline IndirectCall ops (fn-pointer arm, pre-pass)  {baseline_indirect_ops}");
    eprintln!("POST-rtyper IndirectCall ops:");
    eprintln!("      graphs=None (WALL)          {post_none}");
    eprintln!("      graphs=Some(non-empty)      {post_some_nonempty}");
    eprintln!("      graphs=Some(empty)          {post_some_empty}");
    eprintln!("      CallTarget::Indirect surviving  {surviving_indirect_targets}");
    eprintln!(
        "delta attributable to the vtable pass: graphs=None {} (was {baseline_indirect_ops} total before)",
        post_none as i64 - baseline_indirect_ops as i64
    );

    // Non-vacuity, asserted rather than eyeballed. Both are properties of the
    // instrument, not of the answer: they fail if the probe measured nothing.
    assert!(
        !graphs.is_empty(),
        "instrument vacuous: no cel body lowered"
    );
    assert_eq!(
        surviving_indirect_targets, 0,
        "lower_indirect_calls left {surviving_indirect_targets} CallTarget::Indirect behind — \
         the pass did not run over every site"
    );
    assert!(
        pre_total > 0,
        "instrument vacuous: no CallTarget::Indirect site to classify"
    );
}
