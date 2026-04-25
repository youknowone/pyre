//! Bundle an adapter-produced `FunctionGraph` into the
//! `(HostObject, PyGraph)` pair the annotator pipeline expects.
//!
//! Upstream analogue — `rpython/translator/interactive.py:25-26`:
//!
//! ```python
//! graph = self.context.buildflowgraph(entry_point)
//! self.context._prebuilt_graphs[entry_point] = graph
//! ```
//!
//! Line 25 runs upstream `build_flow` on Python bytecode and wraps the
//! resulting `FunctionGraph` inside a `PyGraph`. Line 26 seeds the
//! translator's prebuilt-graph cache so subsequent
//! `buildflowgraph(same entry_point)` calls short-circuit without
//! re-building.
//!
//! The Rust-source counterpart has no bytecode, so
//! `build_flow_from_rust` replaces line 25's work; this helper packages
//! the same `(host, pygraph)` pair that line 26 inserts into the cache.
//! Seeding the cache stays the caller's responsibility so this module
//! does not need to depend on `TranslationContext`.
//!
//! The synthetic [`HostCode`] populated here is the minimum needed for
//! upstream `cpython_code_signature` (`flowspace/bytecode.py`) to read
//! back the right argnames — `co_argcount`, `co_varnames`, `co_flags`.
//! `co_code` is empty because the function has no bytecode. Callers
//! that later introspect the code object (e.g. `is_generator`) will
//! see `CO_GENERATOR` unset, which is the correct Rust-fn answer.
//!
//! Upstream RPython's `_assert_rpythonic` (`objspace.py:33-35`) requires
//! `CO_NEWLOCALS` on any RPython function's code object, so we set it
//! here even though the adapter itself bypasses `build_flow` /
//! `_assert_rpythonic`; downstream consumers that re-run
//! `_assert_rpythonic` on the pair (e.g. a later `PyGraph::new` rebuild)
//! must see a structurally valid code object.
//!
//! `co_nlocals` / `co_varnames` cover formal arguments **and** every
//! `let`-bound / `for`-pattern identifier that [`build_flow_from_rust`]
//! may have introduced as an extra local. Upstream `pygraph.py:14-16`
//! sizes the initial `locals = [None] * co_nlocals` array by the full
//! local count; synthesizing only the formal-arg prefix here would let
//! a downstream `PyGraph::new` rebuild produce an under-sized locals
//! array that disagrees with the adapter's by-name `HashMap`.
//!
//! `co_firstlineno` reads `syn::ItemFn`'s `fn_token` span (requires
//! the `proc-macro2/span-locations` feature — see this crate's
//! `Cargo.toml`). `co_filename` is supplied by the caller via the
//! `source_filename: Option<&str>` parameter — `syn::Span` has no
//! stable accessor for the source file path, so the caller (who
//! performed the `parse_file` / `parse_str` call in the first place)
//! is the authoritative source. When the caller has no file context
//! (e.g., `parse_str` on a fixture), passing `None` falls back to
//! the `<rust-source>` sentinel upstream would never emit but the
//! error-rendering code (`tool/error.rs:304`) handles gracefully.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{ExprForLoop, FnArg, ItemFn, Local, Pat, PatIdent};

use super::build_flow::{AdapterError, build_flow_from_rust};
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::objspace::CO_NEWLOCALS;
use crate::flowspace::pygraph::PyGraph;

/// Walk `item_fn`, run the Rust-AST adapter, and return the
/// `(HostObject, PyGraph)` pair that the upstream translator cache
/// expects. The caller is responsible for seeding
/// `TranslationContext._prebuilt_graphs` with the returned pair, exactly
/// as `interactive.py:26` does:
///
/// ```ignore
/// let (host, pygraph) = build_host_function_from_rust(
///     &item_fn,
///     Some("pyre/src/pyopcode.rs"),
///     Some(src),
/// )?;
/// translator
///     ._prebuilt_graphs
///     .borrow_mut()
///     .insert(host.clone(), pygraph);
/// ```
///
/// - `source_filename` populates `HostCode.co_filename` — upstream reads
///   `func.__code__.co_filename` at `model.py:54` for graph-rendering
///   error messages (`tool/error.rs:304`). `syn::Span` has no stable
///   file-path accessor, so the caller (who originally invoked
///   `syn::parse_file` / `parse_str`) is the authoritative source.
///   Passing `None` falls back to the `<rust-source>` sentinel.
/// - `source_text` populates `GraphFunc.source` (upstream
///   `inspect.getsource(func)` at `flowspace/bytecode.py:50`) **and**
///   `FunctionGraph._source` (upstream `model.py:35-47` `source`
///   setter). When `None`, `graph.source()` falls back to the GraphFunc
///   setting, then to the `"source not found"` error surfaced by
///   `tool/error.rs:300`.
pub fn build_host_function_from_rust(
    item_fn: &ItemFn,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    let argnames = extract_argnames(item_fn)?;
    let mut graph = build_flow_from_rust(item_fn)?;

    let name = item_fn.sig.ident.to_string();
    // upstream `pygraph.py:14-16`: `locals = [None] * code.co_nlocals;
    //   for i in range(code.formalargcount): locals[i] = Variable(...)`.
    // Synthesize the same shape by extending `co_varnames` with every
    // extra local the body walker introduced (let-pattern / for-pattern
    // identifiers), so `co_nlocals = formalargcount + extras`.
    let extras = collect_local_names(item_fn, &argnames);
    let mut co_varnames = argnames.clone();
    co_varnames.extend(extras.iter().cloned());
    let nlocals = co_varnames.len() as u32;

    // upstream `objspace.py:33-35` `_assert_rpythonic`: any RPython
    // function's code object must carry `CO_NEWLOCALS`. The adapter
    // bypasses `_assert_rpythonic` (no `build_flow` call) but the
    // synthetic HostCode must still satisfy the invariant so later
    // consumers can re-verify.
    let co_flags = CO_NEWLOCALS;

    // upstream `bytecode.py:46-60` stores `co_firstlineno` from the
    // source code object. `syn::Span::start().line` is 1-based within
    // the span's source input — `parse_file` seeds this as the file
    // line, `parse_str` as the offset within the string (usually 1
    // for a single-fn fixture). The `proc-macro2/span-locations`
    // feature (pulled in via this crate's `Cargo.toml`) is what
    // exposes `start()` outside of a proc-macro runtime.
    let co_firstlineno = item_fn.sig.fn_token.span().start().line as u32;

    // PRE-EXISTING-ADAPTATION: upstream `model.py:54 FunctionGraph.filename`
    // surfaces `func.__code__.co_filename` (a real filesystem path).
    // `syn::Span::source_file()` is nightly-only in `proc_macro2`, so
    // stable Rust cannot recover the path the ItemFn parsed from.
    // Caller threading through `source_filename` is the parity-
    // preserving channel; when the caller has no filename (typical
    // `syn::parse_str` fixtures, or ingestion paths that haven't been
    // taught to thread the path yet), fall back to the `<rust-source>`
    // sentinel. `tool/error.rs:304` renders this sentinel gracefully
    // on the graph-error path.
    //
    // *Convergence path*: when `proc_macro2`'s `span-locations`
    // feature exposes source-file accessors on stable Rust (or we
    // wrap `parse_file` in a helper that preserves the path itself),
    // drop the sentinel and derive from `Span` directly.
    let co_filename = source_filename
        .map(str::to_owned)
        .unwrap_or_else(|| "<rust-source>".to_string());
    let host_code = HostCode::new(
        argnames.len() as u32,
        nlocals,
        0,
        co_flags,
        rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
        Vec::new(),
        Vec::new(),
        co_varnames,
        co_filename,
        name.clone(),
        co_firstlineno,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new().into_boxed_slice(),
    );
    let empty_globals = Constant::new(ConstValue::Dict(Default::default()));
    let mut gf = GraphFunc::from_host_code(host_code.clone(), empty_globals, Vec::new());
    // upstream `bytecode.py:46-60` populates `GraphFunc.source` from
    // `inspect.getsource(func)`. When the caller threads in the
    // source text, mirror that — downstream readers (`model.rs:3210
    // FunctionGraph::source`, `tool/error.rs:300-320`) walk
    // `func.source` as a fallback when `graph._source` is unset, so
    // one assignment covers both paths.
    if let Some(src) = source_text {
        gf.source = Some(src.to_owned());
    }
    let host = HostObject::new_user_function(gf.clone());

    // upstream `PyGraph.__init__` (pygraph.py:20) assigns
    // `FunctionGraph.func = func` via `super().__init__`. Mirror that so
    // downstream helpers (`FlowContext::new`, `FunctionDesc.getuniquegraph`)
    // see the same GraphFunc the HostObject exposes.
    graph.func = Some(gf.clone());
    // upstream `model.py:35-47` exposes `FunctionGraph.source` as a
    // property-with-setter backed by `_source`. The Translation
    // constructor at `interactive.py:25` delegates to
    // `buildflowgraph`, whose non-prebuilt branch leaves
    // `graph._source` untouched — but `inspect.getsource(func)` has
    // already populated `GraphFunc.source`, and the `FunctionGraph.source`
    // property returns it via the `func.source` fallback at
    // `model.py:42`. We mirror the same pair assignment explicitly
    // so `graph.source()` at `model.rs:3207-3216` hits `_source`
    // first (fast path for graph-render error messages).
    if let Some(src) = source_text {
        graph._source = Some(src.to_owned());
    }

    let pygraph = Rc::new(PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        signature: RefCell::new(host_code.signature.clone()),
        // upstream `PyGraph.__init__`: `self.defaults =
        // func.__defaults__ or ()`. Rust-source adapter does not yet
        // surface default values; use the empty tuple shape.
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
        func: gf,
    });
    Ok((host, pygraph))
}

/// Walk the function body and return the ordered unique set of
/// `let`-bound / `for`-pattern identifiers that the adapter's builder
/// introduces as extra locals beyond the formal arguments.
///
/// Mirrors what the Python compiler would emit into `co_varnames`
/// after the formal-arg prefix: one entry per distinct local name
/// assigned anywhere inside the function (`compile.c:compiler_nameop`
/// on the CPython side; `pygraph.py:14-16` reads the resulting
/// `co_nlocals` back when seeding the initial `FrameState`).
///
/// The adapter's `BlockBuilder::locals` also carries synthetic slots
/// named `#for_iter_{depth}` (`build_flow.rs:1266`) — those are *not*
/// upstream `co_varnames` entries (Python would have kept the
/// iterator on the value stack) so they are filtered out by rejecting
/// names starting with `#`.
///
/// Formals are excluded via `argnames_in_order` so the caller can
/// simply append `extras` after `argnames` without deduping again.
fn collect_local_names(item_fn: &ItemFn, argnames_in_order: &[String]) -> Vec<String> {
    struct LocalCollector<'a> {
        argnames: &'a [String],
        seen: std::collections::HashSet<String>,
        order: Vec<String>,
    }

    impl<'a> LocalCollector<'a> {
        fn record(&mut self, pat: &Pat) {
            let ident = match pat {
                Pat::Ident(PatIdent {
                    ident,
                    by_ref: None,
                    subpat: None,
                    ..
                }) => ident.to_string(),
                Pat::Type(pat_type) => {
                    if let Pat::Ident(PatIdent {
                        ident,
                        by_ref: None,
                        subpat: None,
                        ..
                    }) = &*pat_type.pat
                    {
                        ident.to_string()
                    } else {
                        return;
                    }
                }
                _ => return,
            };
            if ident.starts_with('#') || self.argnames.iter().any(|a| a == &ident) {
                return;
            }
            if self.seen.insert(ident.clone()) {
                self.order.push(ident);
            }
        }
    }

    impl<'ast, 'a> Visit<'ast> for LocalCollector<'a> {
        fn visit_local(&mut self, node: &'ast Local) {
            self.record(&node.pat);
            visit::visit_local(self, node);
        }

        fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
            self.record(&node.pat);
            visit::visit_expr_for_loop(self, node);
        }
    }

    let mut collector = LocalCollector {
        argnames: argnames_in_order,
        seen: std::collections::HashSet::new(),
        order: Vec::new(),
    };
    collector.visit_block(&item_fn.block);
    collector.order
}

/// Extract the formal-parameter identifiers from a `syn::ItemFn`,
/// mirroring `collect_params` in `build_flow.rs`. Duplicated rather
/// than shared because the two callers consume different outputs — the
/// adapter needs `Hlvalue`s for the startblock, while this helper needs
/// the plain `String` names for `HostCode::co_varnames`.
fn extract_argnames(item_fn: &ItemFn) -> Result<Vec<String>, AdapterError> {
    let mut out = Vec::new();
    for input in &item_fn.sig.inputs {
        let ident = match input {
            FnArg::Receiver(_) => "self".to_string(),
            FnArg::Typed(pat_type) => match &*pat_type.pat {
                Pat::Ident(PatIdent {
                    ident,
                    by_ref: None,
                    subpat: None,
                    ..
                }) => ident.to_string(),
                _ => {
                    return Err(AdapterError::InvalidSignature {
                        reason: "parameter pattern must be a plain identifier".into(),
                    });
                }
            },
        };
        out.push(ident);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    #[test]
    fn zero_arg_function_produces_matching_signature() {
        let item = parse("fn zero() -> i64 { 1 }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        assert_eq!(host.qualname(), "zero");
        assert!(host.is_user_function());

        let sig = pygraph.signature.borrow();
        assert!(sig.argnames.is_empty());
        assert!(sig.varargname.is_none());
        assert!(sig.kwargname.is_none());

        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 0);
        assert_eq!(code.co_nlocals, 0);
        assert!(code.co_varnames.is_empty());
        // upstream `objspace.py:33-35` — any RPython function's code
        // object must carry `CO_NEWLOCALS`.
        assert_ne!(code.co_flags & CO_NEWLOCALS, 0);
    }

    #[test]
    fn let_bindings_extend_co_varnames_and_co_nlocals() {
        // upstream `pygraph.py:14-16` — `co_nlocals` must size the
        // full locals array (formals + extras); `co_varnames` names
        // each slot in order.
        let item = parse("fn f(a: i64, b: i64) -> i64 { let x = a + b; let y = x + 1; y }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 2);
        assert_eq!(code.co_nlocals, 4);
        assert_eq!(
            code.co_varnames,
            vec![
                "a".to_string(),
                "b".to_string(),
                "x".to_string(),
                "y".to_string(),
            ],
        );
    }

    #[test]
    fn duplicate_let_names_appear_once() {
        // Shadowing `let x` twice still records one slot; upstream
        // Python compilers likewise collapse repeated assignments to
        // the same name into one `co_varnames` entry.
        let item = parse("fn f(a: i64) -> i64 { let x = a; let x = x + 1; x }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_nlocals, 2);
        assert_eq!(code.co_varnames, vec!["a".to_string(), "x".to_string()],);
    }

    #[test]
    fn for_pattern_identifier_is_recorded_as_local() {
        // upstream Python `for item in iter:` introduces `item` as a
        // fast local. Mirror that so the `co_varnames` collector
        // picks the loop variable up even when the adapter itself
        // can't yet lower assignments (`Expr::Assign` is
        // M2.5b-subset-rejected at `build_flow.rs:2145`), so we call
        // the helper directly instead of routing through
        // `build_host_function_from_rust`.
        //
        // The `#for_iter_N` synthetic slot from `build_flow.rs:1266`
        // stays out of `co_varnames` because `#` is not a valid
        // Python identifier character — the collector filters on
        // that prefix.
        let item = parse("fn f(xs: i64) -> i64 { for x in xs { let y = x; } xs }");
        let argnames = extract_argnames(&item).expect("formal args");
        let extras = collect_local_names(&item, &argnames);
        assert!(extras.contains(&"x".to_string()));
        assert!(extras.contains(&"y".to_string()));
        assert!(
            !extras.iter().any(|n| n.starts_with('#')),
            "synthetic iter slot leaked: {:?}",
            extras,
        );
    }

    #[test]
    fn two_arg_function_preserves_order_and_identity() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        let sig = pygraph.signature.borrow();
        assert_eq!(sig.argnames, vec!["a".to_string(), "b".to_string()]);

        // FunctionGraph.func points at the same GraphFunc the
        // HostObject wraps — parity with upstream PyGraph.__init__.
        let graph_func_id = pygraph
            .graph
            .borrow()
            .func
            .as_ref()
            .expect("graph.func set")
            .id;
        let host_func_id = host.user_function().expect("user function").id;
        assert_eq!(graph_func_id, host_func_id);
    }

    #[test]
    fn startblock_inputargs_match_argnames() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (_host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let inputargs = pygraph.graph.borrow().startblock.borrow().inputargs.clone();
        assert_eq!(inputargs.len(), 2);
        // Adapter builds startblock with named Variables — the names
        // come from the Rust parameter identifiers via `collect_params`.
        // `Variable::rename` (model.rs:2050) always trails the prefix
        // with `_` for valid-Python-identifier parity.
        for (expected, arg) in ["a_", "b_"].iter().zip(inputargs.iter()) {
            match arg {
                crate::flowspace::model::Hlvalue::Variable(v) => {
                    assert_eq!(v.name_prefix(), *expected);
                }
                other => panic!("expected Variable, got {other:?}"),
            }
        }
    }

    #[test]
    fn co_firstlineno_reflects_fn_span() {
        // `span-locations` (Cargo.toml) gives `Span::start().line`
        // a non-zero 1-based reading. A leading newline pushes the
        // `fn` token to line 2; assert that the synthetic HostCode
        // picks that up rather than keeping the prior `0` placeholder.
        let item = parse("\n    fn shifted() -> i64 { 1 }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_firstlineno, 2);
    }

    #[test]
    fn rejects_tuple_pattern_parameter() {
        // Matches `collect_params` in `build_flow.rs` — only plain
        // identifier patterns are accepted.
        let item = parse("fn f((a, b): (i64, i64)) -> i64 { a + b }");
        let err = build_host_function_from_rust(&item, None, None).unwrap_err();
        match err {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("plain identifier"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature, got {other:?}"),
        }
    }

    #[test]
    fn seeds_into_translator_prebuilt_graphs_roundtrip() {
        use crate::translator::translator::TranslationContext;

        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        let ctx = TranslationContext::new();
        ctx._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph.clone());

        // `buildflowgraph` must return the prebuilt graph unchanged
        // and leave no residual entry in the cache (upstream
        // `translator.py:50-51` pops).
        let retrieved = ctx.buildflowgraph(host.clone(), false).expect("prebuilt");
        assert!(Rc::ptr_eq(&retrieved, &pygraph));
        assert!(!ctx._prebuilt_graphs.borrow().contains_key(&host));
    }
}
