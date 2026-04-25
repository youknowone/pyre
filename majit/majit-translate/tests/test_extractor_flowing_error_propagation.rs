//! Parity test: `parse::extract_trait_impls`,
//! `parse::extract_inherent_impl_methods`, and
//! `parse::collect_function_graphs` propagate `FlowingError` as
//! `Err(...)` when any function body contains an unsupported
//! construct.  They must NOT silently drop the function / method via
//! `filter_map + Err(_) => None` or `let Ok(...) = ... else { continue; }`.
//!
//! RPython parity references:
//!
//! * `rpython/flowspace/objspace.py:49 build_flow` re-raises
//!   `FlowingError` on any unsupported opcode.
//! * `rpython/flowspace/flowcontext.py:417 recorder.crash_on_FlowingError`
//!   forwards the error through the translator.
//! * `rpython/translator/translator.py:55 buildflowgraph` raises
//!   `FlowingError` up to the translator — no graph is ever produced
//!   for a function whose body has an unsupported construct.
//!
//! The pyre port mirrors that: a function body with e.g. an
//! `async { ... }` / `x.await` / `yield` expression (neither of which
//! has a flow-space analogue yet) must surface as an `Err` at the
//! extractor boundary so the caller (`lib.rs` analyzer) can abort via
//! `.expect("... must lower without FlowingError")` rather than
//! receiving a partially-collected `Vec` with silently-missing rows.
//!
//! A function body containing `async { 1 }` hits the `stop_unsupported`
//! arm in `ast.rs::lower_expr`'s fallback branch
//! (`syn::Expr::Async(_)` is not in the `is_data_creation` set) which
//! returns `Err(FlowingError::Unsupported { ... Async })`.

use majit_translate::{
    ParsedInterpreter, extract_inherent_impl_methods, extract_trait_impls, parse_source,
};

const TRAIT_IMPL_WITH_ASYNC: &str = r#"
    trait Handler {
        fn go(&mut self);
    }
    struct Runtime;
    impl Handler for Runtime {
        fn go(&mut self) {
            let _ = async { 1 };
        }
    }
"#;

const INHERENT_METHOD_WITH_ASYNC: &str = r#"
    struct Runtime;
    impl Runtime {
        fn go(&mut self) {
            let _ = async { 1 };
        }
    }
"#;

const TRAIT_DEFAULT_WITH_ASYNC: &str = r#"
    trait Handler {
        fn go(&mut self) {
            let _ = async { 1 };
        }
    }
"#;

fn parse(src: &str) -> ParsedInterpreter {
    parse_source(src)
}

fn empty_sf() -> majit_translate::front::StructFieldRegistry {
    majit_translate::front::StructFieldRegistry::default()
}
fn empty_frt() -> std::collections::HashMap<String, String> {
    std::collections::HashMap::new()
}
fn empty_names() -> std::collections::HashSet<String> {
    std::collections::HashSet::new()
}

#[test]
fn extract_trait_impls_propagates_flowing_error_from_impl_method_body() {
    let parsed = parse(TRAIT_IMPL_WITH_ASYNC);
    let result = extract_trait_impls(&parsed, &empty_sf(), &empty_frt(), &empty_names());
    // RPython: `build_flow()` would re-raise FlowingError at
    // `flowspace/objspace.py:49` — pyre must match with `Err`.
    assert!(
        result.is_err(),
        "extract_trait_impls must propagate FlowingError rather than \
         silently dropping the impl method; got Ok with len={:?}",
        result.as_ref().ok().map(|v| v.len())
    );
}

#[test]
fn extract_trait_impls_propagates_flowing_error_from_trait_default_body() {
    let parsed = parse(TRAIT_DEFAULT_WITH_ASYNC);
    let result = extract_trait_impls(&parsed, &empty_sf(), &empty_frt(), &empty_names());
    // Same re-raise semantic applies to trait defaults — upstream's
    // `buildflowgraph` pass does not distinguish concrete impl from
    // default body.
    assert!(
        result.is_err(),
        "extract_trait_impls must propagate FlowingError from a trait \
         default whose body contains an unsupported construct; got \
         Ok with len={:?}",
        result.as_ref().ok().map(|v| v.len())
    );
}

#[test]
fn extract_inherent_impl_methods_propagates_flowing_error() {
    let parsed = parse(INHERENT_METHOD_WITH_ASYNC);
    let result = extract_inherent_impl_methods(&parsed, &empty_sf(), &empty_frt(), &empty_names());
    assert!(
        result.is_err(),
        "extract_inherent_impl_methods must propagate FlowingError \
         from an inherent method's body; got Ok with len={:?}",
        result.as_ref().ok().map(|v| v.len())
    );
}

#[test]
fn happy_path_still_returns_ok() {
    // Regression fence: the extractor's Result path must still return
    // Ok on well-formed input.  If this starts failing, the change
    // that broke it over-tightened the extractor's error budget.
    let parsed = parse(
        r#"
        struct Runtime;
        impl Runtime {
            fn identity(x: i64) -> i64 { x }
        }
        trait Handler {
            fn identity(x: i64) -> i64;
        }
        impl Handler for Runtime {
            fn identity(x: i64) -> i64 { x }
        }
        "#,
    );
    assert!(extract_trait_impls(&parsed, &empty_sf(), &empty_frt(), &empty_names()).is_ok());
    assert!(
        extract_inherent_impl_methods(&parsed, &empty_sf(), &empty_frt(), &empty_names()).is_ok()
    );
}
