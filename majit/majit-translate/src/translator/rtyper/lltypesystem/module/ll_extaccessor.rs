//! Signature-only externals for callees whose body the translator cannot
//! walk, but whose address and scalar ABI are known.
//!
//! `extfunc.py` `register_external` / `ExtFuncEntry` is the owner: a residual
//! call of the real function returns the runtime value.  Callers enumerate
//! the functions `build_semantic_program` declined with `unsupported MIR:
//! atomic load ordering`, and this module keeps only a word-only reader:
//! a function that loads one word from a process-global `AtomicU32` and
//! returns it, with no allocation, collection, blocking, or thread
//! creation.
//!
//! Shape alone does not establish that.  `ExtFuncEntry` carries
//! `function / safe_not_sandboxed / signature_args / signature_result /
//! name / lltypeimpl / lltypefakeimpl` and no effect metadata;
//! `rffi.py` `llexternal` takes `random_effects_on_gcobjs`, `releasegil`,
//! `threadsafe`, which this port has not reached.  A zero-arg scalar
//! result declined for an ordered load still matches `safepoint`,
//! `__collect_step_impl`, `spawn_thread`, `acquire_lock` — each of those
//! contains an Acquire load and then allocates, collects, blocks, or
//! creates a thread.  Registering one as a signature-only external
//! tells the rest of the pipeline nothing about any of that.
//!
//! The harvested [`DeclinedFunDecl`] has path, scalar lltypes, and the
//! decline reason; it has no body.  The harvest lives in `front::mir`
//! and does not attach one.  The gate therefore requires:
//!
//! - no translatable body
//! - the ordered-load decline (not some other omitted body)
//! - zero arguments (a process-global cell; a pointer argument is a
//!   visitor / method, not this reader)
//! - an unsigned scalar result (`u32` is in `residual_scalar!`; `Void`
//!   is a scalar lltype but `safepoint` returns `()`)
//! - the declaration's path is one of the four functions whose Rust
//!   source was read and is exactly `AtomicU32::load(Acquire)` of a
//!   process-global cell
//!
//! A pointer argument (`walk_*` GC visitors, `w_type_get_version_tag`'s
//! receiver) or a compound result is not an external.  A body that
//! failed for any other reason is also refused.
//!
//! The result is never a translation-time constant: the GC type id is
//! published at runtime, so the stub annotation is a non-const scalar
//! (`SomeInteger` with no `const_box`).

use std::cell::RefCell;

use super::super::lltype::LowLevelType;
use crate::translator::rtyper::extfunc::ExternalAnnotation;

thread_local! {
    /// Per-pipeline harvest of [`DeclinedFunDecl`] rows, replaced each
    /// `build_semantic_program_via_active_frontend` invocation the same
    /// way `register_foldable_const_lits` replaces its literal set.
    /// `populate_call_registry_from_call_graphs` reads this slot because
    /// the decls are not a `CallControl` / `SemanticProgram` field.
    static HARVESTED_ATOMIC_LOAD_DECLS: RefCell<Vec<DeclinedFunDecl>> =
        const { RefCell::new(Vec::new()) };
}

/// Replace this thread's harvested Acquire-load declarations.
pub fn register_harvested_atomic_load_decls(decls: Vec<DeclinedFunDecl>) {
    HARVESTED_ATOMIC_LOAD_DECLS.with(|slot| *slot.borrow_mut() = decls);
}

/// Clone of the harvested Acquire-load declarations for this pipeline.
pub fn harvested_atomic_load_decls() -> Vec<DeclinedFunDecl> {
    HARVESTED_ATOMIC_LOAD_DECLS.with(|slot| slot.borrow().clone())
}

/// Needle of the `LowerError::Unsupported` Display string
/// `build_semantic_program` records in its local `skipped` vec.
pub const ATOMIC_LOAD_ORDERING_DECLINE: &str = "atomic load ordering";

/// One function `build_semantic_program` declined, presented as a
/// signature-only declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclinedFunDecl {
    pub segments: Vec<String>,
    pub arg_lltypes: Vec<LowLevelType>,
    pub result_lltype: LowLevelType,
    /// `true` when a SemanticFunction graph exists.  An external has none.
    pub has_translatable_body: bool,
    /// `LowerError` Display (`unsupported MIR: atomic load ordering …`)
    /// or another `skipped` / `declaration-has-no-unstructured-body` reason.
    pub decline_reason: String,
}

/// One accepted `register_external` row, driven off [`DeclinedFunDecl`]
/// shape rather than off a name list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicLoadLlexternal {
    pub segments: Vec<String>,
    pub arg_lltypes: Vec<LowLevelType>,
    pub result: LowLevelType,
}

impl AtomicLoadLlexternal {
    pub fn arity(&self) -> usize {
        self.arg_lltypes.len()
    }
}

/// True when `reason` is the MIR loop's ordered-load refusal, not some
/// other omitted body.
pub fn is_atomic_load_ordering_decline(reason: &str) -> bool {
    reason.contains(ATOMIC_LOAD_ORDERING_DECLINE)
}

/// Scalar ABI the residual call can return in one word.
///
/// Matches the primitive tokens `residual_return_shell` can model
/// (`i64` / `u64` / `bool` / `f64` / unit).  128-bit integers have no
/// `getkind` at the codewriter boundary and are refused.  Pointers,
/// aggregates, and `Address`/`Char` are not this external shape.
pub fn is_external_scalar_lltype(lltype: &LowLevelType) -> bool {
    matches!(
        lltype,
        LowLevelType::Void
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::Bool
            | LowLevelType::Float
            | LowLevelType::SingleFloat
    )
}

/// Hand-audited word-only Acquire readers.  Each body's Rust source is
/// exactly `AtomicU32::load(Acquire)` of a process-global cell and a
/// return of that word — no allocation, collection, blocking, or thread
/// creation.  `ExtFuncEntry` has no `random_effects_on_gcobjs` /
/// `releasegil` / `threadsafe` (`rffi.py` `llexternal`), so a
/// signature-only residual is only sound for a function with that
/// property.  The harvest does not attach a body, so the declaration
/// cannot prove it; these four were verified by reading the functions:
///
/// - `lowlevel_string::lowlevel_str_gc_type_id`
/// - `lowlevel_string::lowlevel_unicode_gc_type_id`
/// - `rbuilder::stringbuilder_gc_type_id`
/// - `rbuilder::stringpiece_gc_type_id`
const VERIFIED_WORD_ONLY_ATOMIC_READERS: &[&[&str]] = &[
    &["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"],
    &[
        "pyre_object",
        "lowlevel_string",
        "lowlevel_unicode_gc_type_id",
    ],
    &["pyre_object", "rbuilder", "stringbuilder_gc_type_id"],
    &["pyre_object", "rbuilder", "stringpiece_gc_type_id"],
];

fn is_verified_word_only_atomic_reader(segments: &[String]) -> bool {
    VERIFIED_WORD_ONLY_ATOMIC_READERS.iter().any(|path| {
        path.len() == segments.len()
            && path
                .iter()
                .zip(segments.iter())
                .all(|(want, got)| *want == got)
    })
}

/// Whether `decl` is a word-only Acquire reader the residual ABI can
/// call: ordered-load decline, no translatable body, zero arguments,
/// unsigned result, and a path whose body was verified to load one
/// word and return it.
pub fn is_external_shaped_atomic_accessor(decl: &DeclinedFunDecl) -> bool {
    if decl.has_translatable_body {
        return false;
    }
    if !is_atomic_load_ordering_decline(&decl.decline_reason) {
        return false;
    }
    if !decl.arg_lltypes.is_empty() {
        return false;
    }
    if decl.result_lltype != LowLevelType::Unsigned {
        return false;
    }
    is_verified_word_only_atomic_reader(&decl.segments)
}

/// Keep only the external-shaped rows of `decls`.
pub fn collect_atomic_load_llexternals<'a, I>(decls: I) -> Vec<AtomicLoadLlexternal>
where
    I: IntoIterator<Item = &'a DeclinedFunDecl>,
{
    decls
        .into_iter()
        .filter(|decl| is_external_shaped_atomic_accessor(decl))
        .map(|decl| AtomicLoadLlexternal {
            segments: decl.segments.clone(),
            arg_lltypes: decl.arg_lltypes.clone(),
            result: decl.result_lltype.clone(),
        })
        .collect()
}

/// Project a residual scalar lltype onto `register_external`'s annotation.
pub fn lltype_to_external_annotation(lltype: &LowLevelType) -> Option<ExternalAnnotation> {
    match lltype {
        LowLevelType::Void => Some(ExternalAnnotation::None),
        LowLevelType::Signed | LowLevelType::SignedLongLong => Some(ExternalAnnotation::Int),
        LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => {
            Some(ExternalAnnotation::Unsigned)
        }
        LowLevelType::Bool => Some(ExternalAnnotation::Bool),
        LowLevelType::Float | LowLevelType::SingleFloat => Some(ExternalAnnotation::Float),
        _ => None,
    }
}

/// Dotted host-callable name `extfuncregistry` keys on, matching the
/// `ll_math.math_floor` spelling of the C llexternal table.
pub fn host_qualname(segments: &[String]) -> String {
    segments.join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acquire_reason() -> String {
        "unsupported MIR: atomic load ordering Acquire requires \
         address-preserving ordered lowering"
            .to_string()
    }

    fn zero_arg_unsigned_at(segments: &[&str], reason: String, has_body: bool) -> DeclinedFunDecl {
        DeclinedFunDecl {
            segments: segments.iter().map(|s| (*s).to_string()).collect(),
            arg_lltypes: vec![],
            result_lltype: LowLevelType::Unsigned,
            has_translatable_body: has_body,
            decline_reason: reason,
        }
    }

    fn zero_arg_unsigned(reason: String, has_body: bool) -> DeclinedFunDecl {
        zero_arg_unsigned_at(
            &["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"],
            reason,
            has_body,
        )
    }

    #[test]
    fn is_external_shaped_atomic_accessor_accepts_zero_arg_unsigned_acquire_reader() {
        let decl = zero_arg_unsigned(acquire_reason(), false);
        assert!(is_atomic_load_ordering_decline(&decl.decline_reason));
        assert!(is_external_shaped_atomic_accessor(&decl));
        let rows = collect_atomic_load_llexternals(std::slice::from_ref(&decl));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].arity(), 0);
        assert_eq!(rows[0].result, LowLevelType::Unsigned);
        assert_eq!(
            rows[0].segments,
            ["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"]
        );
        assert_eq!(
            lltype_to_external_annotation(&rows[0].result),
            Some(ExternalAnnotation::Unsigned)
        );
    }

    #[test]
    fn is_external_shaped_atomic_accessor_accepts_the_four_verified_word_readers() {
        for segments in [
            ["pyre_object", "lowlevel_string", "lowlevel_str_gc_type_id"],
            [
                "pyre_object",
                "lowlevel_string",
                "lowlevel_unicode_gc_type_id",
            ],
            ["pyre_object", "rbuilder", "stringbuilder_gc_type_id"],
            ["pyre_object", "rbuilder", "stringpiece_gc_type_id"],
        ] {
            let decl = zero_arg_unsigned_at(&segments, acquire_reason(), false);
            assert!(
                is_external_shaped_atomic_accessor(&decl),
                "{segments:?} is a verified word-only Acquire reader"
            );
        }
        let decls: Vec<_> = VERIFIED_WORD_ONLY_ATOMIC_READERS
            .iter()
            .map(|path| zero_arg_unsigned_at(path, acquire_reason(), false))
            .collect();
        assert_eq!(collect_atomic_load_llexternals(&decls).len(), 4);
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_scalar_args() {
        let decl = DeclinedFunDecl {
            segments: vec![
                "pyre_object".into(),
                "lowlevel_string".into(),
                "lowlevel_str_gc_type_id".into(),
            ],
            arg_lltypes: vec![LowLevelType::Bool, LowLevelType::Signed],
            result_lltype: LowLevelType::Unsigned,
            has_translatable_body: false,
            decline_reason: acquire_reason(),
        };
        assert!(!is_external_shaped_atomic_accessor(&decl));
        assert!(collect_atomic_load_llexternals(std::slice::from_ref(&decl)).is_empty());
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_void_result_even_with_acquire() {
        // `safepoint` is zero-arg, declined for an ordered load, and
        // returns `()`.  Shape without the word-reader property would
        // admit it; the gate must not.
        let decl = DeclinedFunDecl {
            segments: vec!["pyre_object".into(), "gc_interp".into(), "safepoint".into()],
            arg_lltypes: vec![],
            result_lltype: LowLevelType::Void,
            has_translatable_body: false,
            decline_reason: acquire_reason(),
        };
        assert!(!is_external_shaped_atomic_accessor(&decl));
        assert!(collect_atomic_load_llexternals(std::slice::from_ref(&decl)).is_empty());
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_unverified_zero_arg_unsigned() {
        let decl = zero_arg_unsigned_at(
            &["pyre_object", "gc_interp", "some_flag"],
            acquire_reason(),
            false,
        );
        assert!(!is_external_shaped_atomic_accessor(&decl));
        assert!(collect_atomic_load_llexternals(std::slice::from_ref(&decl)).is_empty());
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_pointer_args_even_with_acquire() {
        // A name that matches the census leaf still fails: the filter is
        // the declaration shape, not the leaf.
        let decl = DeclinedFunDecl {
            segments: vec![
                "pyre_object".into(),
                "lowlevel_string".into(),
                "lowlevel_str_gc_type_id".into(),
            ],
            arg_lltypes: vec![LowLevelType::Func(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                },
            ))],
            result_lltype: LowLevelType::Unsigned,
            has_translatable_body: false,
            decline_reason: acquire_reason(),
        };
        assert!(!is_external_shaped_atomic_accessor(&decl));
        assert!(collect_atomic_load_llexternals(std::slice::from_ref(&decl)).is_empty());
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_compound_result() {
        let decl = DeclinedFunDecl {
            segments: vec!["mod".into(), "walk_gc".into()],
            arg_lltypes: vec![],
            result_lltype: LowLevelType::Func(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                },
            )),
            has_translatable_body: false,
            decline_reason: acquire_reason(),
        };
        assert!(!is_external_shaped_atomic_accessor(&decl));
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_other_decline_reasons() {
        let setter = zero_arg_unsigned("declaration-has-no-unstructured-body".to_string(), false);
        assert!(!is_external_shaped_atomic_accessor(&setter));
        let other = zero_arg_unsigned("unsupported MIR: uninitialised local 3".to_string(), false);
        assert!(!is_external_shaped_atomic_accessor(&other));
    }

    #[test]
    fn is_external_shaped_atomic_accessor_rejects_translatable_body() {
        let decl = zero_arg_unsigned(acquire_reason(), true);
        assert!(!is_external_shaped_atomic_accessor(&decl));
    }

    #[test]
    fn collect_atomic_load_llexternals_keeps_only_external_shape() {
        let keep = zero_arg_unsigned(acquire_reason(), false);
        let drop_reason = zero_arg_unsigned("schema decode: x".to_string(), false);
        let drop_body = zero_arg_unsigned(acquire_reason(), true);
        let rows = collect_atomic_load_llexternals([&keep, &drop_reason, &drop_body]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].segments, keep.segments);
    }

    #[test]
    fn host_qualname_matches_ll_math_dotted_spelling() {
        assert_eq!(
            host_qualname(&["ll_math".into(), "math_floor".into()]),
            "ll_math.math_floor"
        );
        assert_eq!(
            host_qualname(&[
                "pyre_object".into(),
                "lowlevel_string".into(),
                "lowlevel_str_gc_type_id".into()
            ]),
            "pyre_object.lowlevel_string.lowlevel_str_gc_type_id"
        );
    }

    #[test]
    fn harvested_atomic_load_decls_replace_per_pipeline() {
        let keep = zero_arg_unsigned(acquire_reason(), false);
        register_harvested_atomic_load_decls(vec![keep.clone()]);
        assert_eq!(harvested_atomic_load_decls(), vec![keep.clone()]);
        register_harvested_atomic_load_decls(Vec::new());
        assert!(harvested_atomic_load_decls().is_empty());
        register_harvested_atomic_load_decls(vec![keep.clone()]);
        assert_eq!(harvested_atomic_load_decls().len(), 1);
        register_harvested_atomic_load_decls(Vec::new());
    }
}
