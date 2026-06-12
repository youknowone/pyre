//! Stable-Rust parser for Charon `.llbc` / `.ullbc` JSON artefacts.
//!
//! This crate is the input layer of the MIR-driven flowspace driver.
//! It exposes:
//!
//!   - [`schema`] — `serde::Deserialize` structs covering the subset of
//!     Charon's IR we actually consume. Schema fields we do not yet
//!     consume are kept as opaque [`serde_json::Value`] so that newer
//!     Charon versions stay round-trippable; the typed schema is widened
//!     incrementally as each piece is needed.
//!   - [`Llbc`] — a thin wrapper around [`schema::LlbcFile`] with
//!     lookup helpers (`local_fn`, `iter_local_fns`, etc.).
//!   - [`SchemaError`] — fail-loud error type. The crate never silently
//!     drops bodies; an unrecognised variant returns a hard error.
//!
//! The crate compiles on **stable Rust**. The pinned-nightly toolchain
//! required to produce `.llbc` lives inside Charon itself
//! (`scripts/install-charon.sh`); nothing in this crate touches it.

#![forbid(unsafe_code)]

pub mod schema;
pub mod ullbc;

pub use schema::LlbcFile;
pub use ullbc::{
    BasicBlock, FieldDecl, FunDecl, GlobalDecl, Locals, Statement, StmtKind, TermKind, TraitDecl,
    TypeDecl, TypeDeclKind, Unstructured, VariantDecl,
};

use std::path::Path;

/// Loaded `.llbc` / `.ullbc` artefact + lookup helpers.
#[derive(Debug)]
pub struct Llbc {
    pub file: LlbcFile,
    /// `dedup_id → ADT def_id` index built from inline
    /// `HashConsedValue: [id, body]` occurrences whose body decodes as
    /// `{"Adt": {"id": {"Adt": <def_id>}}}`.  Sorted by `dedup_id` for
    /// binary search.  Populated once at parse time.
    ///
    /// Consumed by `front::mir::Lowering` to resolve a Charon `Impl`
    /// segment's `skip_binder: {"Deduplicated": <id>}` reference to
    /// the receiver type's small `def_id` so `CallTarget::Method` can
    /// carry the leaf type name.  Without this, an inherent-impl
    /// method called through `CallTarget::FunctionPath` leaves the
    /// callee body's `self` arg typed as `SomeInstance(classdef=None)`
    /// and any `.field` projection on it panics in the annotator
    /// (`annotator/unaryop.rs:3587`).
    dedup_adt: Vec<(u64, u64)>,
    /// `dedup_id → body Value` index built from every inline
    /// `HashConsedValue: [id, body]` occurrence in the raw LLBC JSON.
    /// Sorted by `dedup_id` for binary search.  Populated once at
    /// parse time.
    ///
    /// Consumed by `front::mir::Lowering::tyref_to_value_type` so a
    /// `TyRef::Deduplicated{id}` reference can be projected to its
    /// underlying `ValueType` (primitive `Literal` bodies → `Int` /
    /// `Bool` / `Float`, `Adt` / `Ref` / `RawPtr` → `Ref`).  Without
    /// this index, FunDecl return types serialized as `Deduplicated`
    /// (≈8190 of 8940 typed return signatures in `pyre-interpreter.ullbc`)
    /// fall back to `Ref` and downstream callers cannot distinguish
    /// `i64`-returning helpers from pointer-returning ones, defeating
    /// `fn_return_types`-based type checks.
    dedup_body: Vec<(u64, serde_json::Value)>,
}

impl Llbc {
    /// Load and parse a `.llbc` / `.ullbc` JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SchemaError::Io)?;
        Self::from_slice(&bytes)
    }

    /// Parse a `.llbc` / `.ullbc` artefact from an in-memory byte slice.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, SchemaError> {
        // Parse to `Value` first so we can scan every nested
        // `HashConsedValue` entry; then re-deserialize the same JSON
        // into the typed `LlbcFile`.  Peak memory is ~3× the bytes
        // (input slice + Value + LlbcFile) but settles back to
        // LlbcFile + small `dedup_adt` / `dedup_body` once the Value
        // is dropped.
        let raw: serde_json::Value = serde_json::from_slice(bytes).map_err(SchemaError::Parse)?;
        let mut dedup_adt: Vec<(u64, u64)> = Vec::new();
        let mut dedup_body: Vec<(u64, serde_json::Value)> = Vec::new();
        collect_dedup_bodies(&raw, &mut dedup_adt, &mut dedup_body);
        dedup_adt.sort_by_key(|&(id, _)| id);
        dedup_adt.dedup_by_key(|p| p.0);
        dedup_body.sort_by_key(|p| p.0);
        dedup_body.dedup_by_key(|p| p.0);
        let file: LlbcFile = serde_json::from_value(raw).map_err(SchemaError::Parse)?;
        Ok(Self {
            file,
            dedup_adt,
            dedup_body,
        })
    }

    /// Resolve a Charon `Deduplicated: <id>` type reference to the
    /// underlying ADT `def_id` (suitable for [`Self::type_by_id`]).
    /// Returns `None` for non-ADT types (primitives, references,
    /// tuples) and for ids whose inline form never appeared in the
    /// LLBC.  See the [`Self::dedup_adt`] field doc for context.
    pub fn dedup_to_adt_def_id(&self, id: u64) -> Option<u64> {
        self.dedup_adt
            .binary_search_by_key(&id, |&(d, _)| d)
            .ok()
            .map(|i| self.dedup_adt[i].1)
    }

    /// Resolve a Charon `Deduplicated: <id>` reference to its
    /// underlying inline body (a `serde_json::Value` of the same
    /// shape Charon emits inline for a `HashConsedValue: [id, body]`).
    /// Returns `None` for ids whose inline form never appeared in
    /// this LLBC.  See the [`Self::dedup_body`] field doc for
    /// context.
    pub fn dedup_body(&self, id: u64) -> Option<&serde_json::Value> {
        self.dedup_body
            .binary_search_by_key(&id, |p| p.0)
            .ok()
            .map(|i| &self.dedup_body[i].1)
    }

    /// Look up a local-crate function whose name ends with `::<name>`.
    pub fn local_fn(&self, name: &str) -> Option<&FunDecl> {
        let suffix = format!("::{name}");
        for f in self.iter_local_fns() {
            let path = f.item_meta.name_path();
            if path == name || path.ends_with(&suffix) {
                return Some(f);
            }
        }
        None
    }

    /// Look up a `FunDecl` by its Charon `def_id`. The `fun_decls`
    /// array is indexed by `def_id` (verified against extracted
    /// corpora), so this is an O(1) bounds-checked lookup.
    pub fn fn_by_id(&self, def_id: u64) -> Option<&FunDecl> {
        self.file
            .translated
            .fun_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `GlobalDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn global_by_id(&self, def_id: u64) -> Option<&GlobalDecl> {
        self.file
            .translated
            .global_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `TypeDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn type_by_id(&self, def_id: u64) -> Option<&TypeDecl> {
        self.file
            .translated
            .type_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `TraitDecl` by its Charon `def_id`.
    pub fn trait_by_id(&self, def_id: u64) -> Option<&TraitDecl> {
        self.file
            .translated
            .trait_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Iterate over every present `TypeDecl`.
    pub fn iter_type_decls(&self) -> impl Iterator<Item = &TypeDecl> {
        self.file
            .translated
            .type_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `TraitDecl`.
    pub fn iter_trait_decls(&self) -> impl Iterator<Item = &TraitDecl> {
        self.file
            .translated
            .trait_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// The raw `trait_impls` table (schema-opaque; entries may be
    /// `null`).  Consumed by the front-end's trait-associated-type
    /// resolution: an `impl Trait for T` entry binds each associated
    /// type (`kind: {"TraitType": [trait_id, idx]}`) to a concrete
    /// type in its `types[].skip_binder.value`.
    pub fn trait_impls_raw(&self) -> &[serde_json::Value] {
        self.file
            .translated
            .rest
            .get("trait_impls")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Iterate over every present `FunDecl` (skipping opaque `null` entries).
    pub fn iter_local_fns(&self) -> impl Iterator<Item = &FunDecl> {
        self.file
            .translated
            .fun_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `GlobalDecl` (skipping opaque `null`
    /// entries).  Used by the hint harvester to read the macro-emitted
    /// `_elidable_function_<NAME>` / `_jit_*_<NAME>` marker consts.
    pub fn iter_global_decls(&self) -> impl Iterator<Item = &GlobalDecl> {
        self.file
            .translated
            .global_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Crate name (the `crate_name` field from `.llbc.translated`).
    pub fn crate_name(&self) -> &str {
        &self.file.translated.crate_name
    }
}

/// Walk a raw `Value` tree, recording every inline
/// `HashConsedValue: [id, body]` occurrence.  Records the body into
/// `bodies` (the generic dedup-id → body index) and, when the body
/// decodes as `{"Adt": {"id": {"Adt": <def_id>}}}`, also into `adt`
/// (the dedup-id → ADT def_id index for fast Adt resolution).  Used
/// during [`Llbc::from_slice`].
fn collect_dedup_bodies(
    v: &serde_json::Value,
    adt: &mut Vec<(u64, u64)>,
    bodies: &mut Vec<(u64, serde_json::Value)>,
) {
    match v {
        serde_json::Value::Object(m) => {
            if let Some(arr) = m
                .get("HashConsedValue")
                .and_then(serde_json::Value::as_array)
                && arr.len() == 2
                && let Some(id) = arr[0].as_u64()
            {
                if let Some(def_id) = adt_def_id_from_ty_body(&arr[1]) {
                    adt.push((id, def_id));
                }
                bodies.push((id, arr[1].clone()));
            }
            for vv in m.values() {
                collect_dedup_bodies(vv, adt, bodies);
            }
        }
        serde_json::Value::Array(arr) => {
            for vv in arr {
                collect_dedup_bodies(vv, adt, bodies);
            }
        }
        _ => {}
    }
}

/// Project a type-expression body to its underlying ADT `def_id`,
/// when the body has shape `{"Adt": {"id": {"Adt": <def_id>}}}`.
/// Returns `None` for non-ADT bodies (`Literal`, `Ref`, `Tuple`, …).
fn adt_def_id_from_ty_body(body: &serde_json::Value) -> Option<u64> {
    body.as_object()?
        .get("Adt")?
        .as_object()?
        .get("id")?
        .as_object()?
        .get("Adt")?
        .as_u64()
}

/// Errors produced when loading / parsing a `.llbc` artefact.
#[derive(Debug)]
pub enum SchemaError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    Decode(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::Io(e) => write!(f, "io: {e}"),
            SchemaError::Parse(e) => write!(f, "parse: {e}"),
            SchemaError::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchemaError::Io(e) => Some(e),
            SchemaError::Parse(e) => Some(e),
            SchemaError::Decode(_) => None,
        }
    }
}
