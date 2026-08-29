//! Scratch diagnostic — census how many `TypeDecl`s the zero-sized-ADT
//! arm in `front::mir::tyref_to_value_type` newly turns `Void` (were
//! `Ref` before), excluding the ones the two existing fieldless-enum
//! arms already catch.
//!
//! Mirrors `tyref_is_zero_sized` / `type_decl_is_fieldless_enum` from
//! `majit-translate/src/front/mir.rs` against this crate's public API
//! directly, rather than against a `TyRef` — for a `TyRef` pointing at
//! `def_id`, `tyref_is_zero_sized` reduces exactly to
//! `type_by_id(def_id).layout_for_target("").size == Some(0)`, so
//! checking each `TypeDecl`'s own layout is equivalent and simpler.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release -p majit-charon-reader --example census_zst -- build/llbc/pyre-jit.ullbc
//! ```
use majit_charon_reader::{
    Llbc,
    ullbc::{Rvalue, StmtKind, TyRef, TypeDecl, TypeDeclKind},
};
use serde_json::Value;
use std::collections::BTreeSet;

fn inline_adt_def_id(body: &Value) -> Option<u64> {
    body.as_object()?
        .get("Adt")?
        .as_object()?
        .get("id")?
        .as_object()?
        .get("Adt")?
        .as_u64()
}

fn tyref_is_zero_sized(ty: &TyRef, llbc: &Llbc) -> bool {
    let def_id = match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
        TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
    };
    def_id
        .and_then(|def_id| llbc.type_by_id(def_id))
        .and_then(|td| td.layout_for_target(""))
        .is_some_and(|layout| layout.size == Some(0))
}

fn type_decl_is_fieldless_enum(td: &TypeDecl, llbc: &Llbc) -> bool {
    match &td.kind {
        TypeDeclKind::Enum(variants) => {
            !variants.is_empty()
                && variants
                    .iter()
                    .all(|v| v.fields.iter().all(|f| tyref_is_zero_sized(&f.ty, llbc)))
        }
        _ => false,
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: census_zst <file.ullbc>");
        std::process::exit(2);
    });
    let llbc = Llbc::load(&path).unwrap_or_else(|e| {
        eprintln!("error: cannot load {path}: {e}");
        std::process::exit(1);
    });

    // Control: a type decl we already know is a fieldless enum
    // (non-zero size, multi-variant) must NOT appear in the newly-Void
    // list, and the total type_decls count must be non-zero — proves
    // the scan is reading real type decls before trusting a 0 result.
    let mut total = 0usize;
    let mut zero_sized = 0usize;
    let mut fieldless_enum_excluded = 0usize;
    let mut newly_void: Vec<String> = Vec::new();

    for td in llbc.file.translated.type_decls.iter().flatten() {
        total += 1;
        let is_zero_sized = td.layout_for_target("").is_some_and(|l| l.size == Some(0));
        if !is_zero_sized {
            continue;
        }
        zero_sized += 1;
        if type_decl_is_fieldless_enum(td, &llbc) {
            fieldless_enum_excluded += 1;
            continue;
        }
        newly_void.push(td.item_meta.name_path());
    }

    println!("file:                       {path}");
    println!("total type_decls:           {total}");
    println!("zero-sized (any kind):      {zero_sized}");
    println!("  of which fieldless enum:  {fieldless_enum_excluded} (excluded, unaffected)");
    println!("newly Void (the count):     {}", newly_void.len());
    println!(
        "control — PyreEnv present:  {}",
        newly_void.iter().any(|n| n.contains("PyreEnv"))
    );
    println!("--- distinct types ---");
    for n in &newly_void {
        println!("{n}");
    }

    // Cross-reference against reality: of the newly-Void types, how
    // many are actually *constructed* anywhere (an `Assign(place,
    // Rvalue::Aggregate(..))` whose destination type resolves to one
    // of them)? That's the exact code path the `build_rvalue` fold
    // touches, so it's the set that can panic downstream — not merely
    // "declared as a type".
    let newly_void_def_ids: std::collections::BTreeSet<u64> = llbc
        .file
        .translated
        .type_decls
        .iter()
        .flatten()
        .filter(|td| {
            td.layout_for_target("").is_some_and(|l| l.size == Some(0))
                && !type_decl_is_fieldless_enum(td, &llbc)
        })
        .map(|td| td.def_id)
        .collect();

    let mut reached_types: BTreeSet<String> = BTreeSet::new();
    let mut construction_sites = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for bb in &body.body {
            for st in &bb.statements {
                let Ok(StmtKind::Assign(place, rvalue)) = st.stmt_kind() else {
                    continue;
                };
                if !matches!(rvalue, Rvalue::Aggregate(..)) {
                    continue;
                }
                let def_id = match &place.ty {
                    TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
                    TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
                };
                let Some(def_id) = def_id else { continue };
                if newly_void_def_ids.contains(&def_id) {
                    construction_sites += 1;
                    if let Some(td) = llbc.type_by_id(def_id) {
                        reached_types.insert(td.item_meta.name_path());
                    }
                }
            }
        }
    }
    println!("--- cross-reference: actually constructed ---");
    println!("construction sites (Assign+Aggregate): {construction_sites}");
    println!(
        "distinct newly-Void types constructed: {}",
        reached_types.len()
    );
    for n in &reached_types {
        println!("{n}");
    }
}
