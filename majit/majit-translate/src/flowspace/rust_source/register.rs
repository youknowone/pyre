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

use std::collections::HashMap as StdHashMap;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    BinOp, Expr, ExprBinary, ExprForLoop, ExprLit, ExprPath, ExprUnary, File, FnArg, Item,
    ItemEnum, ItemFn, ItemStruct, Lit, Local, Pat, PatIdent, UnOp,
};

use super::build_flow::{AdapterError, build_flow_from_rust_in_module_with_globals};
use super::host_env::{
    ModuleId, clear_walker_error, lookup_walker_pygraph, module_globals_lookup,
    module_globals_snapshot, pyre_stdlib_lookup, register_module_global, register_walker_error,
    register_walker_pygraph,
};
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::objspace::CO_NEWLOCALS;
use crate::flowspace::operation::{
    ArithOps, cmp_fold, coerce_arith, coerce_int_pair, float_py_mod, int_py_floor_div, int_py_mod,
    is_foldable_numeric, python_eq_const,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::lltypesystem::lltype::{
    GcKind, LowLevelType, OpaqueType, Ptr, StructType,
};

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
    // Single-`ItemFn` entry mints a fresh ModuleId — no walker
    // pre-pass means the registry slice is empty for this id, so
    // every `LOAD_GLOBAL` lookup falls through to
    // `pyre_stdlib_lookup` / mint exactly as the pre-Issue-1.3
    // process-global path did. Callers that want sibling-item
    // resolution should route through
    // [`build_host_function_from_rust_file`] instead.
    build_host_function_from_rust_in_module(
        item_fn,
        ModuleId::fresh(),
        source_filename,
        source_text,
        None,
        None,
    )
}

/// Internal helper used by [`build_host_function_from_rust`] and
/// [`build_host_function_from_rust_file`] — both lower the body
/// under an explicit `module_id` so the body's `LOAD_GLOBAL`
/// lookups resolve against the matching registry partition.
///
/// When `func_globals` is `Some(ns)` (Slice O21), the body's
/// `LOAD_GLOBAL` lookups consult `ns.class_get(name)` instead of
/// the module-globals partition — this is the inner-mod case where
/// the fn's `__globals__` IS the inline mod's namespace dict, per
/// Python `function.__globals__ = inner_mod.__dict__`. `None`
/// preserves the outer-module fn shape (partition lookup keyed on
/// `module_id`).
fn build_host_function_from_rust_in_module(
    item_fn: &ItemFn,
    module_id: ModuleId,
    source_filename: Option<&str>,
    source_text: Option<&str>,
    func_globals: Option<&HostObject>,
    class_: Option<HostObject>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    let HostMetadataParts {
        host,
        host_code,
        gf,
    } = build_host_metadata_parts(item_fn, module_id, source_filename, source_text, class_)?;
    let pygraph = lower_body_into_pygraph(
        item_fn,
        module_id,
        func_globals,
        gf,
        &host_code,
        source_text,
    )?;
    Ok((host, pygraph))
}

/// Strict-parity Issue 1 split (2026-05-08): the body-lowering /
/// PyGraph wrapping half of [`build_host_function_from_rust_in_module`].
/// Pairs with [`build_host_metadata_parts`] so the walker can
/// pre-register the `HostObject` (via `build_host_metadata_parts` +
/// `register_module_global`) and only afterwards lower the body —
/// mirroring upstream Python `def f` populating `module.__dict__`
/// BEFORE flow analysis (`flowcontext.py:847 find_global` reads from
/// the live module dict). This is the channel that makes direct
/// (`fn f() { f() }`) and mutual recursion resolve in pyre's adapter.
///
/// `gf` and `host_code` MUST be the pair returned from
/// `build_host_metadata_parts(item_fn, ...)` for the same `item_fn`,
/// so the resulting `PyGraph.func` shares identity with the host's
/// embedded `GraphFunc` (`graph.func` invariant per `pygraph.py:20`).
fn lower_body_into_pygraph(
    item_fn: &ItemFn,
    module_id: ModuleId,
    func_globals: Option<&HostObject>,
    gf: GraphFunc,
    host_code: &HostCode,
    source_text: Option<&str>,
) -> Result<Rc<PyGraph>, AdapterError> {
    let mut graph =
        build_flow_from_rust_in_module_with_globals(item_fn, module_id, func_globals.cloned())?;

    // upstream `PyGraph.__init__` (pygraph.py:20) assigns
    // `FunctionGraph.func = func` via `super().__init__`. Mirror that so
    // downstream helpers (`FlowContext::new`, `FunctionDesc.getuniquegraph`)
    // see the same GraphFunc the HostObject exposes.
    graph.func = Some(gf.clone());
    // upstream `PyGraph.__init__` calls `super().__init__(self._sanitize_funcname(func), ...)`
    // (pygraph.py:18-22) — when `func.class_` is set, the graph name
    // becomes `Class.method` rather than the bare fn ident. Strict-parity
    // (2026-05-10): `build_flow_from_rust_in_module_with_globals` doesn't
    // know about `gf.class_` (it predates Audit 1.5), so re-stamp the
    // graph name here through `PyGraph::sanitize_funcname` so impl-method
    // graphs render as `Class.method` matching upstream `pygraph.py:24
    // _sanitize_funcname`. Top-level fns (`gf.class_ == None`) round-trip
    // to the same bare ident the builder produced.
    graph.name = crate::flowspace::pygraph::PyGraph::sanitize_funcname(&gf);
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

    Ok(Rc::new(PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        signature: RefCell::new(host_code.signature.clone()),
        // upstream `PyGraph.__init__`: `self.defaults =
        // func.__defaults__ or ()`. Rust-source adapter does not yet
        // surface default values; use the empty tuple shape.
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
        func: gf,
    }))
}

/// File-aware sibling of [`build_host_function_from_rust`]: walk
/// every top-level item in `file` through [`register_rust_module`]
/// FIRST (so sibling enums/structs/fns are seeded into the
/// module-globals registry), then locate the `entry_point_name`
/// `Item::Fn` and run the body lowerer on it.
///
/// This is the upstream-orthodox shape for the
/// `interactive.py:14 def __init__(self, entry_point, ...)` →
/// `:25 buildflowgraph(entry_point)` chain: by the time
/// `build_flow(entry_point)` runs upstream, `entry_point.func_globals`
/// already contains every other top-level definition in the same
/// source module (Python's module-import bound them at `def` /
/// `class` time — `flowcontext.py:847 w_globals.value[varname]`).
/// The Rust analogue is the walker pre-pass over `file.items`.
///
/// `entry_point_name` is the bare ident of the target fn — matches
/// the upstream `Translation(entry_point=funcobj)` carrier where
/// `funcobj.__name__` identifies which fn is the build target.
///
/// Returns `AdapterError::Unsupported` if the entry-point name is
/// not a top-level `Item::Fn` in `file`. Other items (enums,
/// structs, etc.) are walked unconditionally — only the entry
/// point's body lowering is gated.
pub fn build_host_function_from_rust_file(
    file: &File,
    entry_point_name: &str,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    // Walker pre-pass — register every top-level item under a
    // `ModuleId` keyed on `source_filename`. The same id is then
    // threaded into body lowering so the entry-point's
    // `LOAD_GLOBAL` resolutions see exactly the bindings the
    // walker just wrote (Issue 1.3 per-module scoping). When the
    // caller threads in a path, the id is path-keyed (Issue 2,
    // 2026-05-05): two walks of the same source file converge on
    // the same id, mirroring upstream
    // `entry_point_a.__globals__ is entry_point_b.__globals__`
    // for two functions defined in the same Python module.
    // Audit 1.2 (2026-05-08): walker now threads source_filename /
    // source_text through to every walked fn's metadata, so the
    // entry-point's `HostObject` registered in `module.__dict__`
    // already carries the caller's source pair. We can drop the
    // post-walk re-build and reuse the walker's host directly,
    // mirroring upstream Python identity invariant
    // `module.__dict__[entry] is caller.entry_point`.
    let module_id = register_rust_module_at_with_source(file, source_filename, source_text)?;

    // Locate the entry-point fn. Upstream `interactive.py:14` takes
    // the function object directly; here the caller names it because
    // a `&syn::File` carries multiple items.
    let item_fn = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(item_fn) if item_fn.sig.ident == entry_point_name => Some(item_fn),
            _ => None,
        })
        .ok_or_else(|| AdapterError::Unsupported {
            reason: format!(
                "entry-point fn `{entry_point_name}` not found among top-level items in the \
                 supplied `syn::File` — `interactive.py:14 entry_point` requires a real function \
                 object as the build target"
            ),
        })?;

    // Audit 1.2 happy path: pass-2 of `register_rust_module_at_with_source`
    // ran body lowering for every top-level `Item::Fn`, so when the
    // body lowered successfully the resulting `HostObject` already
    // sits in the module-globals registry under `entry_point_name`
    // and its `PyGraph` is pinned in `HOST_RUST_PYGRAPHS`. Reuse
    // both — that is the parity-orthodox identity invariant
    // `module.__dict__[entry] is caller.entry_point`.
    if let Some(ConstValue::HostObject(host)) = module_globals_lookup(module_id, entry_point_name)
        && host.is_user_function()
        && let Some(pygraph) = lookup_walker_pygraph(&host)
    {
        return Ok((host, pygraph));
    }

    // Walker miss: the body was un-lowerable during pass-2. Re-run
    // the build directly so the caller receives the actual adapter
    // error (e.g. `InvalidSignature`) rather than a generic "missing
    // from registry" wrapper. This is the only path that mints a
    // second `HostObject`; the post-walk re-build runs only on the
    // failure trajectory and the registry has nothing usable to
    // collide with.
    build_host_function_from_rust_in_module(
        item_fn,
        module_id,
        source_filename,
        source_text,
        None,
        None,
    )
}

/// Build a `HostObject::UserFunction` for `item_fn` carrying the
/// synthetic `HostCode` (signature, co_varnames, co_firstlineno) but
/// **without** running [`build_flow_from_rust`] on the body. The
/// embedded `GraphFunc.prebuilt_flow_graph` stays `None`.
///
/// This is the Rust-source analogue of Python's module-import-time
/// function creation: at `import` time, the Python interpreter binds
/// the name in `module.__dict__` to a function object whose
/// `__code__` is set but whose flowspace `FunctionGraph` has not been
/// built yet.
///
/// **Status**: as of Issue 1.2 (TODO), this helper
/// is no longer the walker's body-deferral path —
/// [`register_rust_module`] does not register `Item::Fn` because
/// the rebuild path between `FunctionDesc.buildgraph`
/// (`description.py:140`) and the Rust-AST adapter is missing,
/// so a deferred-body `HostObject` would supply empty bytecode at
/// lowering time. The helper remains exported as a public utility
/// for callers that explicitly want metadata-only construction
/// (e.g. a future M2.5g side-table walker that pairs the metadata
/// HostObject with a stored `&syn::ItemFn` for later replay).
pub fn build_host_function_metadata_from_rust(
    item_fn: &ItemFn,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<HostObject, AdapterError> {
    // No walker pre-pass on the metadata-only path — module dict is
    // empty, matching upstream `func.__globals__ == {}` for a
    // function defined with no module bindings yet visible.
    Ok(build_host_metadata_parts(
        item_fn,
        ModuleId::fresh(),
        source_filename,
        source_text,
        None,
    )?
    .host)
}

/// Walk a parsed Rust source `file` and register every top-level
/// **class-shaped** item (`Item::Enum` / `Item::Struct`) and
/// **literal const** (`Item::Const`) into the process-global
/// module-globals registry (`HOST_RUST_MODULE_GLOBALS`).
///
/// Mirrors Python module import: when the Python interpreter executes
/// a `class` statement or a top-level constant assignment at module
/// scope, it binds the name in `module.__dict__` to the freshly-built
/// class object / value. This walker is the Rust-source counterpart
/// for the *bindable-without-body* subset.
///
/// Subsequent `Builder::resolve_path_constant` lookups route through
/// `module_globals_lookup` and return the registered value directly,
/// matching upstream `flowcontext.py:847 w_globals.value[varname]`.
///
/// ### `Item::Fn` registration (Slice O16)
///
/// Upstream Python `def` populates `module.__dict__[name]` with a
/// function object whose body is callable through the standard
/// `LOAD_GLOBAL` → `simple_call` chain. The walker mirrors this:
/// each top-level `Item::Fn` is eagerly lowered via
/// `build_flow_from_rust_in_module`, and on success the resulting
/// `HostObject::UserFunction` is registered under the fn's name.
/// Sibling-fn calls in another fn's body (`fn caller() { helper() }`)
/// then resolve through the registry as `Constant(HostObject)` and
/// emit a clean `simple_call(<host>, args)` SpaceOperation.
///
/// **Try-build-then-register-on-success**: when
/// `build_flow_from_rust_in_module` rejects the body (e.g. `as T`
/// cast — task #94, or other un-roadmapped constructs), the walker
/// silently skips registration. The downstream resolver falls
/// through to `Builder::resolve_path_constant`'s mint-or-fail path,
/// matching the pre-O9 behavior for un-lowerable bodies.
///
/// **Forward references between sibling `Item::Fn`s resolve through
/// the iterative pass-2 loop** (Slice O17). Pass 1 registers
/// non-fn items in source order; pass 2 sweeps the deferred
/// `Item::Fn` set repeatedly until no more registrations succeed.
/// A caller-before-helper pattern (`fn caller() { helper() }`
/// declared above `fn helper()`) takes two iterations: the first
/// fails the caller (helper missing) and succeeds the helper; the
/// second succeeds the caller against the now-registered helper.
/// True mutual recursion between two un-otherwise-resolvable fns
/// stays unregistered (no progress → loop terminates).
///
/// **Production-side double-build caveat**:
/// [`build_host_function_from_rust_file`] calls
/// [`register_rust_module_at`] FIRST (which now eagerly builds the
/// entry-point fn during the walker pass) and then calls
/// [`build_host_function_from_rust_in_module`] on the same fn to
/// return a fresh `(HostObject, Rc<PyGraph>)`. The two builds
/// produce distinct HostObjects whose qualnames are equal but
/// whose underlying GraphFunc identities differ. Callers that compare
/// the returned HostObject against a registry lookup must use
/// qualname identity, not pointer identity.
///
/// ### Re-registration semantics
///
/// `register_module_global` is **last-writer-wins** under a fixed
/// `(module_id, name)`: every call unconditionally overwrites the
/// prior entry. This mirrors upstream `module.__dict__[name] =
/// value` — every top-level binding statement is an unconditional
/// assignment, whether on first import, `exec(source, dict)`, or
/// `importlib.reload`. Within a single walk Rust syntax does not
/// allow duplicate top-level item names, so the observable effect
/// is across walks of the same path-keyed module: a second walk's
/// bindings supersede the first. Callers who want `sys.modules`
/// cache-hit semantics ("don't re-execute when already loaded")
/// must gate the call themselves on a prior `module_globals_lookup`
/// — the registry does not implement that gate. Production
/// callsites (`from_rust_file_entry_point_with_source`) invoke
/// the walker once per `Translation`, so the re-walk window is
/// open only in tests and cross-`Translation` workflows.
///
/// ### Scope (Slice O10 walker — Item::Enum / Item::Struct / Item::Const)
///
/// - **`Item::Enum`** → `class StepResult: ...` with each variant
///   populated as a class-dict entry (`class_set(variant_name,
///   ConstValue::HostObject(variant_class))`). The variant class is
///   a subclass of the parent enum class — Rust's `match` semantics
///   line up with Python `isinstance(x, StepResult.Continue)`.
///   Stored as `ConstValue::HostObject(<class>)`.
/// - **`Item::Struct`** → `class Foo: ...` with empty class dict.
///   Struct fields live on instances, not the class object.
///   Stored as `ConstValue::HostObject(<class>)`.
/// - **`Item::Const`** → `MODULE_NAME = <expr>` at module top
///   level. Bound to `module.__dict__[MODULE_NAME]` as the
///   evaluated value. Stored as `ConstValue::Int/Bool/UniStr/ByteStr/Float`
///   directly (no HostObject wrapper) — mirrors upstream
///   `find_global` returning `const(value)` regardless of value
///   type. RHS evaluation is delegated to [`eval_const_expr`],
///   which covers literal forms (`Lit::Int` / `Lit::Bool` /
///   `Lit::Str` / `Lit::ByteStr` / `Lit::Float` / `Lit::Char` /
///   `Lit::Byte`), unary `Neg` / `Not`, single-segment Path
///   lookups against prior `bindings` and the registry partition
///   (forward-ref + re-walk paths), and binary ops (`+` / `-` /
///   `*` / `/` / `%` / `<<` / `>>` / `&` / `|` / `^` / `==` /
///   `!=` / `<` / `<=` / `>` / `>=` / `&&` / `||`) over
///   evaluated operands. Shapes [`eval_const_expr`] does not yet
///   cover (function calls, struct literals, multi-segment paths)
///   decline-fold to `Ok(None)` and the walker silently skips —
///   those bindings then fall through to
///   `Builder::resolve_path_constant`'s mint-or-fail path at
///   call sites.
///
/// Other `Item::*` kinds (external-rooted `Item::Use`, generic
/// `Item::Impl`, external `Item::Mod`, …) are silently skipped as
/// upstream-walker follow-ups (each populates `module.__dict__` at
/// Python import time too). Self-impl `Item::Impl` (Slice O18),
/// trait `Item::Impl` (Slice O22), inline `Item::Mod` (Slices
/// O19/O20), `Item::Fn` (Slices O16/O17), and local-rooted
/// `Item::Use` (Slice O23 — single-segment alias / inline-mod
/// cascade / group expansion / glob) ARE walked.
/// **Immutable** `Item::Static` is admitted alongside `Item::Const`
/// (Slice O12) — upstream Python's `module.__dict__` sees both
/// shapes identically. `static mut FOO` is **NOT** registered
/// (Slice O15 / codex audit 2026-05-06): runtime mutation makes
/// the initial-value snapshot unsound for constfold reads, so the
/// adapter skips until a live-store path lands. Each future slice
/// extends the dispatch match without changing the call sites.
///
/// ### Per-module scoping (Issue 1.3, 2026-05-05)
///
/// Returns a fresh [`ModuleId`] every time — anonymous walks
/// never merge. Mirrors upstream Python's `exec(source,
/// fresh_dict)` semantic: each call runs the source against an
/// independent `__dict__`, even if the source bytes are
/// byte-identical to a prior `exec`.
///
/// This BC entry routes through [`register_rust_module_at`] with
/// `None` path. Callers that need shared registry slices across
/// multiple walks (e.g. two entry points from the same source
/// file sharing `func.__globals__`) MUST thread a stable path
/// through [`register_rust_module_at`] — that's the only orthodox
/// way to opt into upstream `sys.modules[name]` import-cache
/// behavior.
pub fn register_rust_module(file: &File) -> Result<ModuleId, AdapterError> {
    register_rust_module_at(file, None)
}

/// Lifted from [`register_rust_module_at`] (Slice O18) so the inline-
/// `Item::Mod` recursive helper [`register_items_into_namespace`]
/// (Slice O20) can reuse the same shape. Carries a single self-impl
/// method's `(target_class_path, method_name, body)` triple between
/// pass 1 (collection) and pass 2 (lower-and-`class_set`).
///
/// `class_path` widened from `String` (single segment) to
/// `Vec<String>` (1+ segments) so multi-segment self-types (`impl
/// Trait for foo::Bar` where `foo` is a registered inline-mod
/// namespace) cascade through [`try_resolve_use_path`] in pass 2.
/// Length-1 paths preserve prior behavior; length-2+ paths
/// add the inline-mod cascade pending external-crate / `crate::` /
/// `super::` resolution.
struct DeferredImpl {
    class_path: Vec<String>,
    method_name: String,
    item_fn: syn::ItemFn,
}

/// Slice O23: a single binding produced by flattening one `Item::Use`
/// tree. Each leaf-level `UseTree::Name` / `UseTree::Rename` /
/// `UseTree::Glob` becomes one or more `DeferredUse` entries
/// (group expansion handled at flatten time). Resolution is deferred
/// to the same pass-2 fixed-point loop that handles `Item::Fn`s and
/// self-impl `Item::Impl`s so a `use foo::Bar` can pick up a sibling
/// `mod foo { struct Bar; }` declared later in source order.
///
/// Mirrors Python `from x import y [as alias]`: each leaf adds one
/// `module.__dict__[name] = x.y` binding (RPython parity through
/// upstream `flowcontext.py:847 w_globals.value[varname]` reading
/// the populated dict at `LOAD_GLOBAL` time).
struct DeferredUse {
    /// Local name to bind in the registry (or class dict for inner
    /// mods). For `use foo::Bar` this is `"Bar"`; for `use foo::Bar
    /// as Baz` this is `"Baz"`; for `use foo::*` this is empty (glob
    /// expanded at resolution time, see [`UseKind::Glob`]).
    binding_name: String,
    /// Full path segments to resolve. For `use foo::Bar` this is
    /// `["foo", "Bar"]`; for `use foo::*` this is `["foo"]` (the path
    /// itself, the glob's contents come from cascading `class_dict_items`).
    path_segments: Vec<String>,
    /// `false` for `Name` / `Rename`, `true` for `Glob`. Glob
    /// resolution iterates `path_segments`'s `class_dict_items` and
    /// binds each entry under its original key (no rename — Python
    /// `from x import *` parity).
    glob: bool,
}

/// Slice O23: leading-segment classification for use-tree paths.
///
/// `Local` paths cascade through the local registry / namespace dict
/// (single-session feasible). `External` paths root at `crate::`,
/// `super::`, `self::`, or a well-known external crate root —
/// resolution requires multi-file walking / external crate registry,
/// so the use is silently skipped (matches Python's `ImportError`
/// failing soft when the importing module re-runs and the name isn't
/// available yet).
enum UseRoot {
    /// Path resolves through the local module's registry partition or
    /// inline-mod namespace dict. The first segment is treated as a
    /// top-level ident lookup; subsequent segments cascade via
    /// `class_get`.
    Local,
    /// Path roots at `crate::` / `super::` / `self::` / leading-colon
    /// / `std` / `core` / `alloc` / external workspace crates. Skipped
    /// pending multi-file walker support.
    External,
}

/// Recognise leading-segment idents that mark a use path as
/// requiring multi-file resolution. Conservative match — accepts
/// `crate` / `super` / `self` plus the well-known stdlib roots
/// (`std`, `core`, `alloc`) and the project's external workspace
/// crates (`pyre_object`, `pyre_jit`, `pyre_jit_trace`, `majit_*`,
/// etc.). New external crates that crop up should be added here as
/// they fail tests; no harm in over-skipping (failure mode is the
/// pre-O23 behavior of falling through to `mint_unknown` / cascade
/// resolution at call sites).
fn classify_use_root(first_segment: &str) -> UseRoot {
    let well_known_external = matches!(
        first_segment,
        "crate"
            | "super"
            | "self"
            | "std"
            | "core"
            | "alloc"
            | "pyre_object"
            | "pyre_jit"
            | "pyre_jit_trace"
            | "pyre_interpreter"
            | "pyre_module"
            | "pyre_wasm"
            | "pyrex"
    ) || first_segment.starts_with("majit_");
    well_known_external
        .then_some(UseRoot::External)
        .unwrap_or(UseRoot::Local)
}

/// Slice O23: flatten a `UseTree` into one or more [`DeferredUse`]
/// entries. Recursive: `Path` extends the prefix, `Group` fans out,
/// `Name` / `Rename` / `Glob` produce leaf entries.
///
/// Mirrors syn's UseTree variants:
///
/// - `UseTree::Name(name)` — `prefix::name` → `(name, prefix::name)`.
/// - `UseTree::Rename(rename)` — `prefix::rename.ident as
///   rename.rename` → `(rename.rename, prefix::rename.ident)`.
/// - `UseTree::Path(path)` — extend prefix with `path.ident`,
///   recurse into `path.tree`.
/// - `UseTree::Group(group)` — recurse into each item with the same
///   prefix.
/// - `UseTree::Glob(_)` — emit a glob entry with `binding_name = ""`
///   and `path_segments = prefix`.
fn flatten_use_tree(tree: &syn::UseTree, prefix: Vec<String>, out: &mut Vec<DeferredUse>) {
    match tree {
        syn::UseTree::Name(name) => {
            let n = name.ident.to_string();
            let mut path = prefix;
            path.push(n.clone());
            out.push(DeferredUse {
                binding_name: n,
                path_segments: path,
                glob: false,
            });
        }
        syn::UseTree::Rename(rename) => {
            let original = rename.ident.to_string();
            let alias = rename.rename.to_string();
            let mut path = prefix;
            path.push(original);
            out.push(DeferredUse {
                binding_name: alias,
                path_segments: path,
                glob: false,
            });
        }
        syn::UseTree::Path(path) => {
            let mut new_prefix = prefix;
            new_prefix.push(path.ident.to_string());
            flatten_use_tree(&path.tree, new_prefix, out);
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(item, prefix.clone(), out);
            }
        }
        syn::UseTree::Glob(_) => {
            out.push(DeferredUse {
                binding_name: String::new(),
                path_segments: prefix,
                glob: true,
            });
        }
    }
}

/// Slice O23: cascade-resolve a single path through the local
/// registry / namespace dict. Returns `Some(value)` on hit,
/// `None` on miss (forward-ref retry candidate).
///
/// Mirrors upstream `flowcontext.py:861 LOAD_ATTR` →
/// `op.getattr(w_obj, w_name).eval(self)`'s constfold step:
/// each segment lookup is a `class_get` against the prior
/// `HostObject::Class`. The initial segment lookup splits on
/// whether the use is at module top level (`namespace=None`,
/// hits the registry partition) or inside an inline mod
/// (`namespace=Some`, hits the namespace's class dict).
fn try_resolve_use_path(
    segments: &[String],
    module_id: ModuleId,
    namespace: Option<&HostObject>,
) -> Option<ConstValue> {
    if segments.is_empty() {
        return None;
    }
    let initial = match namespace {
        Some(ns) => ns.class_get(&segments[0]),
        None => module_globals_lookup(module_id, &segments[0]),
    }?;
    let mut current = initial;
    for seg in &segments[1..] {
        let next = match &current {
            ConstValue::HostObject(h) => h.class_get(seg)?,
            _ => return None,
        };
        current = next;
    }
    Some(current)
}

/// Slice O24: extract a target-class path from an `Item::Impl` self-
/// type for the deferred-impl queue. Accepts `Type::Path` whose path
/// has no `qself`, no leading colon, only lifetime-only per-segment
/// generic args (Slice O25), and is rooted at a non-external first
/// segment (per [`classify_use_root`]). Returns `None` for any other
/// shape — the walker silently skips those impls (matches Slice
/// O18's pre-O24 behavior of treating multi-segment / type-generic /
/// external paths as out of scope).
///
/// Mirrors the input shape that [`try_resolve_use_path`] consumes,
/// so pass 2 reuses the same cascade logic (Python parity:
/// `getattr(getattr(getattr(module, "foo"), "bar"), "Baz")` for
/// `foo::bar::Baz`).
/// Detect `impl ... for <external-rooted path>` so the impl arm can
/// silent-skip per the conceded external-scope adaptation, rather
/// than fall through to `extract_impl_target_path`'s general failure
/// path (which fail-louds the truly-unmodeled self-type shapes —
/// tuple, fn-pointer, slice, etc.).  Returns `true` when the
/// self-type is a `Type::Path` rooted at `crate::` / `super::` /
/// `self::` / `std::` / `core::` / `alloc::`, or a leading-`::`
/// path.  Non-`Type::Path` self-types are NOT external — the walker
/// has no model for them at all, which is the fail-loud case
/// `extract_impl_target_path` reports.
fn is_external_rooted_impl_self_type(self_ty: &syn::Type) -> bool {
    let tp = match self_ty {
        syn::Type::Path(tp) => tp,
        _ => return false,
    };
    if tp.qself.is_some() {
        return false;
    }
    if tp.path.leading_colon.is_some() {
        return true;
    }
    let Some(first) = tp.path.segments.first() else {
        return false;
    };
    matches!(
        classify_use_root(&first.ident.to_string()),
        UseRoot::External
    )
}

fn extract_impl_target_path(self_ty: &syn::Type) -> Option<Vec<String>> {
    let tp = match self_ty {
        syn::Type::Path(tp) => tp,
        _ => return None,
    };
    if tp.qself.is_some() {
        return None;
    }
    if tp.path.leading_colon.is_some() {
        return None;
    }
    let mut segments = Vec::with_capacity(tp.path.segments.len());
    for seg in &tp.path.segments {
        if !is_lifetime_only_path_arguments(&seg.arguments) {
            return None;
        }
        segments.push(seg.ident.to_string());
    }
    if segments.is_empty() {
        return None;
    }
    if matches!(classify_use_root(&segments[0]), UseRoot::External) {
        return None;
    }
    Some(segments)
}

/// Slice O25: a path-segment's `PathArguments` is "lifetime-only" if
/// it carries no arguments (`PathArguments::None`) or only `Lifetime`
/// args inside the angle brackets. Lifetimes have no Python parity
/// (RPython lacks the borrow-checker concept), so the impl-target
/// class identity for `Foo<'a>` is the same as `Foo` — the lifetime
/// is purely a Rust-language adaptation that the walker drops at the
/// adapter boundary. Type / const generic arguments still reject
/// because they DO change the classdef identity (each
/// instantiation = distinct classdef per
/// `description.py:228-249 cachedgraph`); reification is a separate
/// future slice.
fn is_lifetime_only_path_arguments(args: &syn::PathArguments) -> bool {
    match args {
        syn::PathArguments::None => true,
        syn::PathArguments::AngleBracketed(angled) => angled
            .args
            .iter()
            .all(|a| matches!(a, syn::GenericArgument::Lifetime(_))),
        syn::PathArguments::Parenthesized(_) => false,
    }
}

/// Slice O25: an `Item::Impl` introduces no Python-side
/// specialization axis iff every entry of `generics.params` is a
/// `LifetimeParam`. Type / const params reject because they need
/// reification at impl-target time (each `<T = …>` instantiation =
/// distinct classdef per `description.py:228-249 cachedgraph`).
///
/// The `where_clause` is intentionally NOT inspected: when `params`
/// already excludes type/const generics, every `WherePredicate`
/// constrains either an existing lifetime or a concrete type, and
/// upstream `classdesc.py:590-634 add_source_attribute` flat-stores
/// `classdict[name] = Constant(value)` without any
/// reification-equivalent check. Cases like
/// `impl Foo where Foo: SomeTrait { fn bar(&self) }`,
/// `impl Foo where Self: Send { fn bar(&self) }`, or HRTB-bearing
/// `impl Foo where for<'a> &'a Foo: Trait { fn bar(&self) }` keep
/// the self-type concrete, so the methods attach to `Foo`'s class
/// dict identically to a bare `impl Foo { fn bar(&self) }`. The
/// trait bounds are Rust-language constraints with no Python parity.
///
/// Mirrors the parity rationale of [`is_lifetime_only_path_arguments`]:
/// `impl<'a> Trait for Foo<'a> { fn bar(&self) }` produces methods
/// on the same `Foo` classdict as the non-generic `impl Trait for Foo
/// { fn bar(&self) }` would — the `'a` carries no semantic the Python
/// flow analysis observes.
fn impl_generics_only_introduce_lifetimes(generics: &syn::Generics) -> bool {
    generics
        .params
        .iter()
        .all(|p| matches!(p, syn::GenericParam::Lifetime(_)))
}

/// Slice O20: recursive helper that walks `items` into the inline-mod
/// `namespace` `HostObject::Class`, mirroring [`register_rust_module_at`]'s
/// pass-1/pass-2 shape but redirecting the storage target from
/// `register_module_global(module_id, …)` to `namespace.class_set(…)`.
///
/// Mirrors Python `mod foo: …` populating `mod.__dict__[name]` for
/// every binding statement inside the mod body. Inner `Item::Mod`
/// recursion lets `mod a { mod b { … } }` populate `a.b.<name>` via
/// nested `class_set` calls; `Item::Fn` and self-impl `Item::Impl`
/// inside the inner mod resolve their bodies and bind on the inner
/// namespace's dict (NOT the outer module's globals — Rust scoping
/// does not auto-import outer items into inner mods, so an `impl X`
/// inside `mod foo` looks up `X` from `foo.__dict__` only).
///
/// Pass-2 fixed-point loop has the same termination bound as the
/// outer walker: each iteration either makes progress (registers at
/// least one fn / impl method) or exits via the no-progress check.
///
/// Const RHS evaluation runs in this mod's own scoped registry
/// partition (`inner_const_scope`, freshly minted at entry): bare-
/// name lookups that miss the per-mod `inner_bindings` do NOT fall
/// through to the outer module's `module_globals_lookup` partition.
/// Both Rust and Python require explicit `use super::X` to see
/// outer items; the audit 1.4 fix retires the prior outer-fallback
/// behavior so inner-mod const RHSes have the same scoping invariant
/// as inner-mod fn bodies (`register_rust_module_inline_item_mod_inner_fn_does_not_see_outer_module_globals`).
/// `super::` use trees are still classified `External` by
/// `classify_use_root` (silently skipped pending multi-file walker),
/// so this fix removes a leak rather than restricting any working
/// use case.
fn register_items_into_namespace<'a>(
    namespace: &HostObject,
    items: &'a [Item],
    module_id: ModuleId,
    source_filename: Option<&str>,
    source_text: Option<&str>,
    module_prefix: &str,
) -> Result<(), AdapterError> {
    // Pre-collect this inline mod's own `type T = U;` aliases. The
    // guard replaces (not extends) the outer scope's alias map so
    // outer-mod aliases are intentionally NOT visible inside —
    // matching Rust's lexical-scope rule that an inner `mod` block
    // shadows the outer module's `use` / `type` bindings unless they
    // are re-imported. Outer-mod aliases reach the inner scope only
    // through explicit `use super::T` (which the walker resolves
    // through the deferred-uses pipeline, not the catalog).
    let _walker_aliases_guard = WalkerTypeAliasGuard::enter(collect_type_aliases(items));
    // Each scope owns its own `Item::Struct` → `Ptr(GcStruct(...))`
    // map for by-value embedding lookup; inner-mod struct walks must
    // not leak refs into the outer module's nested-embed lookup, so
    // the guard enters with an empty map and reverts on drop.
    let _walker_struct_ptrs_guard = WalkerStructPtrsGuard::enter();
    preseed_struct_ptrs(items);
    // Inner-mod's own scoped const-eval partition (audit 1.4, 2026-05-08).
    let inner_const_scope = ModuleId::fresh();
    // Inner-mod's own module-globals registry partition (audit 1.3,
    // 2026-05-08). Every `namespace.class_set(name, value)` mirror-
    // registers `(name, value)` into this scope so inner-mod fn /
    // impl-method metadata paths (`build_host_function_from_rust_in_module`
    // → `build_host_metadata_parts`) snapshot the inner namespace's
    // dict instead of the outer module's partition. Mirrors upstream
    // Python `function.__globals__ = inner_mod.__dict__` for fns
    // defined inside `mod foo: …`. Body-side `LOAD_GLOBAL` already
    // routes through `func_globals: Some(namespace)` (Slice O21);
    // this fix only closes the metadata divergence Issue 2.1's
    // live re-snapshot reads through.
    //
    // The outer `module_id` parameter remains threaded through for
    // bookkeeping that ties to the surrounding file (e.g. nested
    // `register_items_into_namespace` recursion which mints its own
    // fresh inner_module_scope at entry).
    let inner_module_scope = ModuleId::fresh();
    let mut inner_bindings: StdHashMap<String, ConstValue> = StdHashMap::new();
    // Mirror helper: every namespace.class_set must also stamp the
    // inner-mod registry partition so downstream metadata snapshots
    // see the same content the body's class_get channel does.
    let mirror_set = |name: &str, value: ConstValue| {
        register_module_global(inner_module_scope, name, value.clone());
        namespace.class_set(name, value);
    };
    let mut deferred_fns: Vec<&'a syn::ItemFn> = Vec::new();
    let mut deferred_impls: Vec<DeferredImpl> = Vec::new();
    let mut deferred_uses: Vec<DeferredUse> = Vec::new();
    for inner in items {
        match inner {
            Item::Const(item_const) => {
                let name = item_const.ident.to_string();
                // Strict-parity (2026-05-10): propagate the full
                // `eval_const_expr` outcome — including
                // `AdapterError::Flowing` (NameError on unresolved
                // single-segment Path) — exactly as the top-level
                // `Item::Const` arm does at `register_rust_module_at`.
                // Upstream `flowcontext.py:845 find_global` raises
                // `FlowingError` after both globals and builtins miss;
                // import-time const RHS surfaces that error directly.
                // `inner_const_scope` (audit 1.4) is the per-mod
                // ModuleId, so an outer ident only resolves through
                // explicit `use super::X`. A bare miss is the
                // parity-correct hard error.
                if let Some(value) =
                    eval_const_expr(&item_const.expr, &inner_bindings, inner_const_scope)?
                {
                    mirror_set(&name, value.clone());
                    inner_bindings.insert(name, value);
                }
            }
            Item::Static(item_static)
                if matches!(item_static.mutability, syn::StaticMutability::None) =>
            {
                let name = item_static.ident.to_string();
                if let Some(value) =
                    eval_const_expr(&item_static.expr, &inner_bindings, inner_const_scope)?
                {
                    mirror_set(&name, value.clone());
                    inner_bindings.insert(name, value);
                }
            }
            Item::Enum(item_enum) => {
                let name = item_enum.ident.to_string();
                let host = build_host_class_from_enum(item_enum);
                mirror_set(&name, ConstValue::HostObject(host));
            }
            Item::Struct(item_struct) => {
                let name = item_struct.ident.to_string();
                let host = build_host_class_from_struct(item_struct, module_prefix);
                mirror_set(&name, ConstValue::HostObject(host));
            }
            Item::Fn(item_fn) => {
                deferred_fns.push(item_fn);
            }
            Item::Mod(nested_item_mod) => {
                let Some((_, nested_inner_items)) = &nested_item_mod.content else {
                    continue;
                };
                let nested_name = nested_item_mod.ident.to_string();
                let nested_namespace = HostObject::new_class(&nested_name, vec![]);
                // Push the nested mod ident onto the module-prefix path
                // so `build_host_class_from_struct` keys inner structs
                // under the qualified path, mirroring PyPy nested
                // module scoping (`classdesc.py:672` per-class identity).
                let nested_prefix = if module_prefix.is_empty() {
                    nested_name.clone()
                } else {
                    format!("{module_prefix}::{nested_name}")
                };
                register_items_into_namespace(
                    &nested_namespace,
                    nested_inner_items,
                    module_id,
                    source_filename,
                    source_text,
                    &nested_prefix,
                )?;
                mirror_set(&nested_name, ConstValue::HostObject(nested_namespace));
            }
            // Slice O22 + O24 + O25: admit `impl Trait for Foo`
            // alongside self-impl `impl Foo`, plus multi-segment
            // self-types (`impl Trait for foo::Bar` where `foo` is a
            // registered inline-mod, Slice O24) and lifetime-only
            // generics (`impl<'a> Trait for Foo<'a>`, Slice O25).
            // All shapes populate `<target>.classdict[method_name]`
            // per upstream `classdesc.py:590-634 add_source_attribute`'s
            // flat `self.classdict[name] = Constant(value)` assignment.
            // The trait's identity is not consulted; closed-world
            // dispatch through `bookkeeper.py:431-442 getmethoddesc`
            // keys on `(originclassdef, name, …)`. Lifetime parameters
            // carry no Python-observable semantic (RPython lacks the
            // borrow concept) so `Foo<'a>` resolves to the same
            // classdef as `Foo`.
            //
            // **TODO (trait-name namespace
            // collapse).** Rust's trait method dispatch can
            // disambiguate same-named methods from different traits
            // (`<Foo as TraitA>::name` vs `<Foo as TraitB>::name`)
            // through the trait identity carried in the call site.
            // Upstream Python's flat class dict (`classdesc.py:590-634
            // add_source_attribute`) cannot — it only knows
            // `name → callable`. The walker therefore enforces
            // convergence option (b): a *walker-time ban* on cross-
            // trait method-name collisions, keyed on the resolved
            // target `HostObject.identity_id()` so alias paths
            // (`use a::Foo as X; use a::Foo as Y`) cannot sneak past
            // the textual class-path key — see
            // `seen_method_collisions` in both walkers (round-4
            // 2026-05-11). Convergence option (a) — a trait-aware
            // dispatch table keyed on `(trait_id, method_name)` — is
            // still deferred (no upstream basis) and
            // would let two trait impls with the same method name
            // coexist instead of erroring out. Today's closed-world
            // `pyre-interpreter` codebase avoids such collisions so
            // the ban surfaces as no observable regression.
            //
            // External-rooted self-types (`impl Trait for
            // crate::foo::Bar`, including generic impl blocks on
            // external roots) remain skipped — cross-file resolution is
            // its own future slice. Local type / const generic impls
            // (`impl<T> Foo<T>`, `impl<E: Trait> X for E`) fail loud
            // below because type-arg reification is required before a
            // method can attach to one concrete class dict.
            // Concrete-self impls with
            // `where` predicates (`impl Foo where Foo: SomeTrait`,
            // `impl Foo where Self: Send`) DO pass: the params list
            // introduces no specialization axis, so the methods
            // attach to `Foo`'s class dict per `classdesc.py:590-634
            // add_source_attribute`'s `classdict[name] =
            // Constant(value)` flat assignment.
            Item::Impl(item_impl) if is_external_rooted_impl_self_type(&item_impl.self_ty) => {
                // TODO (external scope): see the outer walker's
                // identical short-circuit.
                let _ = item_impl;
                continue;
            }
            Item::Impl(item_impl)
                if impl_generics_only_introduce_lifetimes(&item_impl.generics) =>
            {
                let class_path = extract_impl_target_path(&item_impl.self_ty).ok_or_else(|| {
                    AdapterError::Unsupported {
                        reason: "inline `mod`: unsupported `impl` self-type \
                                 (only path-shaped class identifiers with \
                                 lifetime-only generics are modeled)."
                            .to_string(),
                    }
                })?;
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Fn(method) = impl_item {
                        let method_name = method.sig.ident.to_string();
                        let item_fn = syn::ItemFn {
                            attrs: method.attrs.clone(),
                            vis: method.vis.clone(),
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        deferred_impls.push(DeferredImpl {
                            class_path: class_path.clone(),
                            method_name,
                            item_fn,
                        });
                    }
                }
            }
            Item::Impl(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "inline `mod`: `impl` block introduces \
                             type / const generic parameters; walker \
                             has no reification path."
                        .to_string(),
                });
            }
            // Slice O23: collect `Item::Use` bindings into the
            // deferred queue. The pass-2 fixed-point loop retries
            // them after every iteration's class / fn registrations
            // so a `use foo::Bar` resolves once `mod foo { struct
            // Bar; }` is registered (regardless of source order).
            // Skip leading-`::` paths and paths rooted at well-known
            // external prefixes (`crate::`, `super::`, `self::`,
            // `std::`, `core::`, `alloc::`) — multi-file resolution
            // is its own future port (TODO,
            // outer-walker convergence path applies).
            Item::Use(item_use) => {
                if item_use.leading_colon.is_some() {
                    continue;
                }
                let mut staged: Vec<DeferredUse> = Vec::new();
                flatten_use_tree(&item_use.tree, Vec::new(), &mut staged);
                for du in staged {
                    if du.path_segments.is_empty() {
                        continue;
                    }
                    if matches!(classify_use_root(&du.path_segments[0]), UseRoot::External) {
                        continue;
                    }
                    deferred_uses.push(du);
                }
            }
            _ => {
                // TODO: same set of silently-skipped
                // item kinds as the outer walker — `static mut`,
                // `Item::Trait` / `TraitAlias`, `Item::Type`,
                // `Item::Macro`, `Item::ForeignMod`, `Item::ExternCrate`,
                // `Item::Union`, `Item::Verbatim`, plus the external
                // `Item::Use` / no-body `Item::Mod` short-circuits
                // handled in their own arms above. See the outer
                // walker's catch-all docstring for per-kind reasoning.
            }
        }
    }
    // Strict-parity (2026-05-11): pre-register every inner-mod fn's
    // metadata-only host into the namespace dict BEFORE any body is
    // lowered, mirroring the outer-walker pass-1 introduced by Issue 1.
    // Forward references between inner-mod fns (`fn caller() {
    // helper() }` ahead of `fn helper() { ... }`) need every sibling
    // placeholder visible through `namespace.class_get(name)` at body
    // lowering time so the body's `LOAD_GLOBAL helper` cascades through
    // the inline-mod's class dict.
    //
    // Slice O21: pass `namespace` as `func_globals` so the inner-mod fn
    // body's `LOAD_GLOBAL` lookups target the inline-mod's class dict
    // (sibling fn / class / const) instead of the outer module's
    // partition. Mirrors Python `function.__globals__ =
    // inner_mod.__dict__`. `class_ = None`: inner-mod fns are
    // module-scoped, not class-owned (audit 1.5 only stamps
    // `GraphFunc.class_` for the impl-method path).
    //
    // Audit 1.3 (2026-05-08): pass `inner_module_scope` (not outer
    // `module_id`) so `GraphFunc.module_globals_id` re-snapshots the
    // inner namespace's dict rather than the outer module's partition.
    //
    // Audit 1.2 (2026-05-08): thread caller's source_filename /
    // source_text into inner-mod fn metadata.
    //
    // `pending_inner_fns` carries the (name, host, gf, host_code,
    // item_fn) tuple needed to retry `lower_body_into_pygraph` while
    // keeping the host identity stable across iterations.
    // `metadata_failed_fns` holds fns whose metadata extraction itself
    // failed (e.g. arg-name extraction error); they fall back into
    // `deferred_fns` as a residue set with no further retry. Fns whose
    // bodies never lower keep their placeholder host in namespace dict
    // — mirrors Python's import-time `def f` semantic where failure
    // surfaces at later `buildflowgraph` time rather than at the
    // walker pass.
    let mut pending_inner_fns: Vec<(String, HostObject, GraphFunc, HostCode, &'a syn::ItemFn)> =
        Vec::with_capacity(deferred_fns.len());
    let mut metadata_failed_fns: Vec<&'a syn::ItemFn> = Vec::new();
    for item_fn in deferred_fns.drain(..) {
        let name = item_fn.sig.ident.to_string();
        match build_host_metadata_parts(
            item_fn,
            inner_module_scope,
            source_filename,
            source_text,
            None,
        ) {
            Ok(HostMetadataParts {
                host,
                host_code,
                gf,
            }) => {
                mirror_set(&name, ConstValue::HostObject(host.clone()));
                pending_inner_fns.push((name, host, gf, host_code, item_fn));
            }
            Err(_) => metadata_failed_fns.push(item_fn),
        }
    }
    deferred_fns = metadata_failed_fns;

    // Strict-parity (2026-05-11): impl-method handling splits into two
    // phases — Phase A resolves the target class + builds metadata +
    // `class_set`s the placeholder ONCE; Phase B retries body lowering
    // with the host identity stable. `pending_inner_methods` carries
    // the (host, gf, host_code, item_fn) tuple for Phase B. Methods
    // whose target class never resolves stay in `deferred_impls` and
    // never advance to Phase A.
    let mut pending_inner_methods: Vec<(HostObject, GraphFunc, HostCode, syn::ItemFn)> = Vec::new();

    // Strict-parity round-4 (2026-05-11): cross-trait method-name
    // collision check, keyed on resolved HostObject identity. See
    // outer-walker `seen_method_collisions` for the alias-path
    // rationale; the inner-mod walker has the same exposure since
    // `try_resolve_use_path` consults the namespace dict and a `use
    // outer::Foo as X; use outer::Foo as Y;` cascade resolves both
    // aliases to the same `Arc<HostObjectInner>`.
    let mut seen_method_collisions: StdHashMap<(usize, String), Vec<String>> = StdHashMap::new();

    loop {
        let mut made_progress = false;

        // Phase A (fns): retry pending bodies; host already in
        // namespace dict from the pre-pass.
        let mut still_pending_fns = Vec::with_capacity(pending_inner_fns.len());
        for (name, host, gf, host_code, item_fn) in pending_inner_fns.drain(..) {
            match lower_body_into_pygraph(
                item_fn,
                inner_module_scope,
                Some(namespace),
                gf.clone(),
                &host_code,
                source_text,
            ) {
                Ok(pygraph) => {
                    clear_walker_error(&host);
                    register_walker_pygraph(host, pygraph);
                    made_progress = true;
                }
                Err(err) => {
                    register_walker_error(host.clone(), err.to_string());
                    still_pending_fns.push((name, host, gf, host_code, item_fn));
                }
            }
        }
        pending_inner_fns = still_pending_fns;

        // Phase A (impls): for each unresolved impl, try resolving the
        // target class. On success, build metadata + `class_set` the
        // placeholder + push to `pending_inner_methods`. Inner-mod
        // scoping (Slice O20 + O24) cascades the impl's target class
        // path through the inline mod's own dict. An `impl X` inside
        // `mod foo` that references an outer-module class without a
        // matching inner declaration stays deferred — the outer-module
        // scope is intentionally not consulted here.
        //
        // Slice O21: pass `namespace` as `func_globals` so the method
        // body's `LOAD_GLOBAL` lookups target the inline-mod dict.
        //
        // Audit 1.5 (2026-05-08): pass the resolved target `class`
        // through to `GraphFunc.class_` so the method's
        // `HostObject::UserFunction` carries class-owned identity
        // (mirrors upstream Python `func.class_` at `class Foo: def
        // bar(self): ...`).
        //
        // Audit 1.3 (2026-05-08): pass `inner_module_scope` so
        // `gf.module_globals_id` snapshots the inner namespace dict.
        //
        // Audit 1.2 (2026-05-08): thread caller's source_filename /
        // source_text into the impl-method metadata.
        let mut still_deferred_impls = Vec::with_capacity(deferred_impls.len());
        for di in deferred_impls.drain(..) {
            let class = match try_resolve_use_path(&di.class_path, module_id, Some(namespace)) {
                Some(ConstValue::HostObject(h)) if h.is_class() => h,
                _ => {
                    still_deferred_impls.push(di);
                    continue;
                }
            };
            // Strict-parity round-4 (2026-05-11): identity-keyed
            // collision check. See the outer-walker variant.
            let collision_key = (class.identity_id(), di.method_name.clone());
            if let Some(prior_path) = seen_method_collisions.get(&collision_key) {
                return Err(AdapterError::Unsupported {
                    reason: format!(
                        "Cross-trait method-name collision on `{}::{}` \
                         inside `mod {}`: a prior `impl … for {}` block \
                         already wrote the method name into the flat \
                         class dict (resolved host identity matches \
                         across alias paths). Upstream \
                         `classdesc.py:590-634 add_source_attribute` \
                         has no trait-namespace channel; rename one \
                         method or move the impls outside the inline mod.",
                        di.class_path.join("::"),
                        di.method_name,
                        namespace.qualname(),
                        prior_path.join("::"),
                    ),
                });
            }
            let parts = match build_host_metadata_parts(
                &di.item_fn,
                inner_module_scope,
                source_filename,
                source_text,
                Some(class.clone()),
            ) {
                Ok(p) => p,
                Err(_) => {
                    still_deferred_impls.push(di);
                    continue;
                }
            };
            let HostMetadataParts {
                host,
                host_code,
                gf,
            } = parts;
            seen_method_collisions.insert(collision_key, di.class_path.clone());
            class.class_set(&di.method_name, ConstValue::HostObject(host.clone()));
            pending_inner_methods.push((host, gf, host_code, di.item_fn));
            made_progress = true;
        }
        deferred_impls = still_deferred_impls;

        // Phase B (impls): retry pending method bodies; host already
        // in class dict via the Phase A `class_set`.
        let mut still_pending_methods = Vec::with_capacity(pending_inner_methods.len());
        for (host, gf, host_code, item_fn) in pending_inner_methods.drain(..) {
            match lower_body_into_pygraph(
                &item_fn,
                inner_module_scope,
                Some(namespace),
                gf.clone(),
                &host_code,
                source_text,
            ) {
                Ok(pygraph) => {
                    clear_walker_error(&host);
                    register_walker_pygraph(host, pygraph);
                    made_progress = true;
                }
                Err(err) => {
                    register_walker_error(host.clone(), err.to_string());
                    still_pending_methods.push((host, gf, host_code, item_fn));
                }
            }
        }
        pending_inner_methods = still_pending_methods;

        // Slice O23: retry deferred uses against the namespace dict.
        // Each successful resolution binds `binding_name` →
        // `resolved_value` via `class_set`; failure stays deferred
        // (forward-ref retry next iteration). Mirrors Python
        // `from x import y`: `module.__dict__["y"] = x.y` once `x.y`
        // is resolvable.
        let mut still_deferred_uses = Vec::with_capacity(deferred_uses.len());
        for du in deferred_uses.drain(..) {
            if du.glob {
                let Some(ConstValue::HostObject(target)) =
                    try_resolve_use_path(&du.path_segments, module_id, Some(namespace))
                else {
                    still_deferred_uses.push(du);
                    continue;
                };
                if !target.is_class() {
                    still_deferred_uses.push(du);
                    continue;
                }
                // Strict-parity (2026-05-10): same import-star
                // rule as the outer-module branch below. PyPy
                // `import_all_from` skips leading-underscore names
                // when the source object has no `__all__` analogue.
                for (name, value) in target.class_dict_items() {
                    if name.starts_with('_') {
                        continue;
                    }
                    mirror_set(&name, value);
                }
                made_progress = true;
                continue;
            }
            match try_resolve_use_path(&du.path_segments, module_id, Some(namespace)) {
                Some(value) => {
                    mirror_set(&du.binding_name, value);
                    made_progress = true;
                }
                None => still_deferred_uses.push(du),
            }
        }
        deferred_uses = still_deferred_uses;

        if !made_progress
            || (pending_inner_fns.is_empty()
                && deferred_impls.is_empty()
                && pending_inner_methods.is_empty()
                && deferred_uses.is_empty())
        {
            break;
        }
    }
    // `deferred_fns` (metadata-failed fns) and `pending_inner_fns` /
    // `pending_inner_methods` residues whose bodies never lowered keep
    // their placeholder hosts in the namespace / class dict — mirrors
    // Python's import-time `def f` semantic where failure surfaces at
    // later `buildflowgraph` time rather than at the walker pass.
    let _ = deferred_fns;
    let _ = pending_inner_fns;
    let _ = pending_inner_methods;

    // Strict-parity: same fail-loud contract as the outer walker —
    // `deferred_uses` and `deferred_impls` residues mean walker-known
    // local references that never reached a registered binding /
    // class. External-rooted prefixes are filtered at intake, so
    // anything left here is a local-rooted miss inside this inline
    // mod's namespace.
    if let Some(du) = deferred_uses.into_iter().next() {
        let path = du.path_segments.join("::");
        let suffix = if du.glob {
            "::*".to_string()
        } else if du
            .path_segments
            .last()
            .is_some_and(|seg| seg != &du.binding_name)
        {
            format!(" as {}", du.binding_name)
        } else {
            String::new()
        };
        return Err(AdapterError::Unsupported {
            reason: format!(
                "unresolved local-rooted `use {path}{suffix}` inside \
                 inline `mod`: walker found no registered binding for \
                 the path's head segment after the fixed-point loop."
            ),
        });
    }
    if let Some(di) = deferred_impls.into_iter().next() {
        return Err(AdapterError::Unsupported {
            reason: format!(
                "unresolved `impl ... for {}` target inside inline `mod` \
                 after fixed-point loop: walker found no registered \
                 `HostObject::Class` for the self-type path. Method \
                 `{}::{}` cannot attach to a class that does not exist.",
                di.class_path.join("::"),
                di.class_path.join("::"),
                di.method_name,
            ),
        });
    }

    Ok(())
}

/// Path-aware sibling of [`register_rust_module`]. When
/// `source_filename` is `Some(path)`, the registry id is keyed on
/// `path` so that two walks of files at the same path converge on
/// the same [`ModuleId`] (scoped sharing of the `(module_id, name)`
/// partition; **not** upstream `sys.modules` import-cache, which
/// short-circuits the second `import` entirely — the walker
/// re-executes every time and overwrites entries last-writer-wins).
/// When `None`, the call mints a fresh [`ModuleId`] every time —
/// anonymous walks NEVER merge.
///
/// Two walk shapes:
///
/// - **Path-keyed** (`Some(path)`): every walk re-executes against
///   the shared `(module_id, name)` partition under
///   `exec(source, shared_dict)` semantics — entries from the
///   second walk overwrite those from the first per the
///   [`register_module_global`] last-writer-wins rule. Mid-walk
///   failure (an `Item::Const` RHS that surfaces `Err`) leaves
///   prior bindings of THIS walk in the partition. Callers that
///   want `sys.modules`-style "skip the second import" semantics
///   must gate the call themselves on a prior
///   [`module_globals_lookup`] for a known sentinel name. Two
///   `Translation` instances built against entry points from the
///   same file see identical registry partitions only when both
///   re-walks succeed end-to-end.
/// - **Anonymous** (`None`) ↔ `exec(source, fresh_dict)`: each
///   `exec` runs the code against a fresh namespace, even if the
///   source string is byte-identical to a prior `exec`. Two
///   anonymous walks of structurally-identical content therefore
///   produce **independent** module identities — upstream Python
///   never merges two `exec` calls into one `__dict__` based on
///   content.
///
/// **Why not content-hash anonymous?** A prior revision (Issue 2.2,
/// 2026-05-05) keyed the `None` branch on a token-stream content
/// hash so two walks of the same source string converged on one
/// `ModuleId`. Codex parity audit (2026-05-05) flagged that as a
/// regression: `exec(source, dict_a)` and `exec(source, dict_b)`
/// in upstream Python produce **distinct** `__dict__`s — content
/// equality does not imply module identity. Path-keyed sharing
/// is the only way to opt into a shared registry slice; callers
/// who need that contract MUST thread a stable path (filesystem
/// path, fixture label, anything) through `Some(...)`.
pub fn register_rust_module_at(
    file: &File,
    source_filename: Option<&str>,
) -> Result<ModuleId, AdapterError> {
    register_rust_module_at_with_source(file, source_filename, None)
}

/// Audit 1.2 (2026-05-08) source-aware sibling of [`register_rust_module_at`].
///
/// Threads `source_filename` and `source_text` through to every
/// `build_host_function_from_rust_in_module` call inside the walker
/// so the resulting `GraphFunc.{filename, source}` of every walked fn
/// carries the caller's source pair. The entry-point caller
/// ([`build_host_function_from_rust_file`]) uses this to drop the
/// post-walk re-build of the entry fn — it now looks up the
/// walker-built `HostObject` + `PyGraph` from
/// [`HOST_RUST_MODULE_GLOBALS`] and the walker pygraph registry,
/// matching upstream Python `module.__dict__[entry] is
/// caller.entry_point` identity.
pub fn register_rust_module_at_with_source(
    file: &File,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<ModuleId, AdapterError> {
    let module_id = match source_filename {
        Some(path) => ModuleId::for_path(path),
        None => ModuleId::fresh(),
    };
    // Pre-collect `type T = U;` aliases so field-type resolution in
    // `syn_primitive_to_lltype` can chase `PyObjectRef → *mut PyObject`
    // through the catalog. The guard keeps the alias map active for
    // the entire walker scope (pass 1 + pass 2 fixed-point loop +
    // every recursive `register_items_into_namespace` call), and the
    // guard's Drop revert keeps thread-local state clean across
    // sibling walks.
    let _walker_aliases_guard = WalkerTypeAliasGuard::enter(collect_type_aliases(&file.items));
    // Per-file `Item::Struct` → `Ptr(GcStruct(...))` map for
    // by-value embedding lookup; symmetric to the alias guard.
    let _walker_struct_ptrs_guard = WalkerStructPtrsGuard::enter();
    preseed_struct_ptrs(&file.items);
    // Source-order accumulator of `Item::Const` bindings produced
    // during this walk. Mirrors Python module-import semantics:
    // top-level statements run in order and each binding is visible
    // to subsequent ones via `module.__dict__`. The walker passes
    // this dict to `eval_const_expr` so compound consts (`const Y =
    // X + 1`) resolve their forward dependencies through prior
    // entries.
    let mut const_bindings: StdHashMap<String, ConstValue> = StdHashMap::new();
    // Two-pass walker (Slice O17): pass 1 registers `Item::Enum`,
    // `Item::Struct`, `Item::Const`, `Item::Static` in source order.
    // `Item::Fn` entries are collected into `deferred_fns` and
    // resolved by an iterative pass-2 fixed-point loop below — this
    // lets a fn earlier in source resolve a sibling fn declared
    // later, mirroring upstream Python where module-import populates
    // `module.__dict__` for every `def` before any function body
    // actually runs (`flowcontext.py:847 w_globals.value[varname]`
    // sees all sibling defs at lookup time).
    let mut deferred_fns: Vec<&syn::ItemFn> = Vec::new();
    // Slice O18 + Slice O22: `Item::Impl` for self-impl blocks
    // (`impl Foo { … }`) AND trait impl blocks (`impl Trait for Foo
    // { … }`) both add methods / associated fns into the target
    // class's dict. Pass 1 collects each method into
    // `deferred_impls`; pass 2 (alongside the `Item::Fn` sweep)
    // tries to lower the body and call `class_set(method_name, ...)`
    // on the resolved class. The class lookup is deferred to pass 2
    // so an `impl Foo { … }` block declared above its `struct Foo {}`
    // still resolves.
    //
    // Trait impls are treated identically to self-impls per upstream
    // `classdesc.py:590-634 add_source_attribute`'s flat
    // `self.classdict[name] = Constant(value)` shape — the trait
    // identity is not consulted because closed-world dispatch through
    // `bookkeeper.py:431-442 getmethoddesc` keys on
    // `(originclassdef, name, …)`, not on the trait that defined
    // the method.
    let mut deferred_impls: Vec<DeferredImpl> = Vec::new();
    // Slice O23: queue `Item::Use` bindings for the same pass-2
    // fixed-point loop. Each iteration retries each unresolved use
    // against the registry partition; mirrors Python's `from x import
    // y` adding `module.__dict__["y"] = x.y` once `x.y` is in scope.
    let mut deferred_uses: Vec<DeferredUse> = Vec::new();
    for item in &file.items {
        match item {
            Item::Enum(item_enum) => {
                let name = item_enum.ident.to_string();
                // Last-writer-wins per upstream
                // `module.__dict__[name] = value` (every top-level
                // binding statement is an unconditional assignment;
                // `exec(source, dict)` / `importlib.reload`
                // semantics). Rust syntax does not allow duplicate
                // top-level item names within a single source file,
                // so the observable effect is across walks of the
                // same path-keyed `module_id`: the second walk's
                // bindings supersede the first.
                let host = build_host_class_from_enum(item_enum);
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Struct(item_struct) => {
                let name = item_struct.ident.to_string();
                let host = build_host_class_from_struct(item_struct, "");
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Const(item_const) => {
                let name = item_const.ident.to_string();
                // upstream Python import-time evaluation: the RHS
                // runs against the partially-built `module.__dict__`,
                // so compound expressions like `const Y: i64 = X + 1`
                // resolve `X` through the prior binding. The walker
                // threads `const_bindings` (the local source-order
                // accumulator) into the evaluator so forward
                // dependencies between sibling consts work; the
                // evaluator ALSO consults `module_globals_lookup` as
                // fallback when the path-keyed `module_id` was already
                // populated by a prior walk (Issue 2.3, 2026-05-05) —
                // mirrors upstream `module.__dict__` being the live
                // reference visible across re-imports.
                // Failure modes:
                //
                // - **Type mismatch / zero divisor** (`true + 1`,
                //   `1 / 0`) → `Err`. Walker aborts the file, matching
                //   upstream Python's import-time exception.
                // - **`i64` overflow** (`MAX + 1`, `MIN / -1`,
                //   `1 << 64`) → `Unsupported`. Upstream Python
                //   would bind a bignum at module top level; pyre
                //   adapter has no bignum carrier, and this is not an
                //   upstream `FlowingError`. Surface the unsupported
                //   carrier loudly instead of pretending Python raised.
                //   (Function-body constfold takes a different path —
                //   it declines per `operation.py:140-142` and lets
                //   runtime emit. Module top level has no analogous
                //   "let runtime do it" stage.)
                // - **Unsupported shapes** (function calls, struct
                //   literals, multi-segment paths) → `Ok(None)`.
                //   Walker-coverage gaps (Issue 2.3 follow-up).
                // - **Unresolved single-segment Path** → `Err(NameError)`,
                //   matching `flowcontext.py:853 find_global` which
                //   raises `FlowingError("global name '...' is not
                //   defined")` after both globals and builtins miss.
                //   Walker aborts the file. Walker-coverage gaps for
                //   `Item::Fn` / `Item::Use` / `Item::Mod` / `Item::Impl`
                //   surface as parity-correct hard errors here rather
                //   than silent dropped bindings.
                if let Some(value) = eval_const_expr(&item_const.expr, &const_bindings, module_id)?
                {
                    register_module_global(module_id, &name, value.clone());
                    const_bindings.insert(name, value);
                }
            }
            // `static FOO: T = <expr>;` (immutable static, Slice
            // O12 + O15). Module-globals registry binding parallels
            // `Item::Const`: the import-time RHS folds through the
            // same `eval_const_expr` and the same source-order
            // `const_bindings` accumulator. Mirrors `flowcontext.py:847
            // w_globals.value[varname]` which sees `def`-bound and
            // assignment-bound names identically.
            //
            // **Mutable static (`static mut FOO: T = <expr>;`) is
            // NOT registered** (codex parity audit
            // 2026-05-06). TODO: Rust's `mut`
            // marks the global as runtime-mutated; the registry
            // entry would be a stale snapshot of the *initial*
            // value the moment any code writes to it, and folding
            // reads against the initial value would silently
            // produce wrong constfold results. Upstream Python does
            // not have this problem (its flow analysis tracks each
            // `STORE_GLOBAL` at function-body time); pyre's adapter
            // has no analogous live-store path at module-globals
            // prepass. Until that lands, skipping mutable statics
            // entirely is the parity-safe choice — downstream
            // lookups fall through to `Builder::resolve_path_constant`'s
            // mint-or-fail path, which is the right shape for
            // "opaque runtime-mutated symbol".
            Item::Static(item_static)
                if matches!(item_static.mutability, syn::StaticMutability::None) =>
            {
                let name = item_static.ident.to_string();
                if let Some(value) = eval_const_expr(&item_static.expr, &const_bindings, module_id)?
                {
                    register_module_global(module_id, &name, value.clone());
                    const_bindings.insert(name, value);
                }
            }
            // `fn name(...) -> ... { ... }` — collect for pass 2.
            // Mirrors Python `def name(...): ...` populating
            // `module.__dict__[name]` with a function object; the
            // iterative pass-2 loop below resolves both straight-line
            // helper-before-caller and forward-ref helper-after-caller
            // patterns.
            Item::Fn(item_fn) => {
                deferred_fns.push(item_fn);
            }
            // `mod foo { ... }` (Slice O19 / O20) — inline submodule.
            // Inner items register on a fresh `HostObject::Class`
            // namespace; the namespace is bound at the outer
            // module's top level under the mod's ident. Mirrors
            // Python `import foo` populating
            // `module.__dict__["foo"]` with a module object whose
            // attribute access (`foo.A`) traverses the inner
            // namespace's dict — pyre's existing path-cascade
            // resolver (`build_flow.rs::resolve_path_constant`)
            // already handles `foo::A` by recursively `getattr`-ing
            // along the segment chain.
            //
            // External `mod foo;` (no body) is silently skipped —
            // resolving the file system is out of scope for this
            // slice.
            //
            // Scope (Slice O20 widening): inner `Item::Const` /
            // `Item::Static` / `Item::Enum` / `Item::Struct` /
            // `Item::Fn` (with try-build-then-`class_set`-on-success)
            // / self-impl `Item::Impl` (single bare ident self-type,
            // no generics, no trait) / nested `Item::Mod`
            // (recursive). Inner non-self impl variants
            // (`impl Trait for X`, generic `impl<T> X<T>`) are
            // skipped, mirroring the outer walker.
            //
            // Inner const evaluation uses a per-mod source-order
            // bindings dict so consts within the mod resolve each
            // other's forward refs; the outer module's
            // `const_bindings` is not threaded in because Rust
            // scoping does not auto-import outer items into inner
            // mods (`use super::X` path resolution is its own
            // future slice).
            Item::Mod(item_mod) => {
                let Some((_, inner_items)) = &item_mod.content else {
                    continue;
                };
                let mod_name = item_mod.ident.to_string();
                let namespace = HostObject::new_class(&mod_name, vec![]);
                register_items_into_namespace(
                    &namespace,
                    inner_items,
                    module_id,
                    source_filename,
                    source_text,
                    &mod_name,
                )?;
                register_module_global(module_id, &mod_name, ConstValue::HostObject(namespace));
            }
            // `impl Foo { fn bar(...) { ... } ... }` (Slice O18) +
            // `impl Trait for Foo { fn bar(...) { ... } ... }`
            // (Slice O22) — both shapes contribute methods to the
            // target class's dict. Mirrors upstream
            // `classdesc.py:590-634 add_source_attribute`'s
            // `self.classdict[name] = Constant(value)` flat
            // assignment: RPython populates the class dict regardless
            // of whether the method comes from a base class through
            // inheritance, because `lookup_filter`
            // (`classdesc.py:336-374`) does the subclass-aware
            // filtering at lookup time. Closed-world dispatch
            // (`bookkeeper.py:431-442 getmethoddesc` keys on
            // `(originclassdef, name, …)`, not on the trait that
            // defined the method) means trait identity is not
            // consulted here.
            //
            // Slice O24 widens self-type to multi-segment paths
            // (`impl Trait for foo::Bar` cascades through the
            // inline-mod namespace dict). Slice O25 widens generics
            // to lifetime-only-params (`impl<'a> Trait for Foo<'a>`,
            // `impl Foo where Foo: SomeTrait`, etc. — lifetimes have
            // no Python parity, the impl-target class is identical
            // to the non-generic shape, and `where` predicates are
            // Rust-language constraints with no
            // `classdesc.py:590-634 add_source_attribute` analogue
            // because they do not change classdef identity).
            //
            // External-rooted self-types (`impl Trait for
            // crate::foo::Bar`, including generic impl blocks on
            // external roots) remain skipped — cross-file resolution is
            // its own future slice. Local type / const generic impls
            // (`impl<T> Foo<T>`, `impl<E: Trait> X for E`) fail loud
            // below because type-arg reification is required before a
            // method can attach to one concrete class dict.
            Item::Impl(item_impl) if is_external_rooted_impl_self_type(&item_impl.self_ty) => {
                // TODO (external scope):
                // `impl Trait for crate::foo::Bar`,
                // `impl Trait for ::leading::Anchor`,
                // `impl Trait for std::collections::HashMap`, etc.
                // need cross-file / external-crate registry
                // resolution. Silent skip keeps the walker
                // composable with source files that reference
                // external types; the conceded scope is per
                // Section 4 of the strict-parity audit.
                let _ = item_impl;
                continue;
            }
            Item::Impl(item_impl)
                if impl_generics_only_introduce_lifetimes(&item_impl.generics) =>
            {
                let class_path = extract_impl_target_path(&item_impl.self_ty).ok_or_else(|| {
                    AdapterError::Unsupported {
                        reason: "unsupported `impl` self-type: walker can \
                                 only extract a path-shaped class identifier \
                                 (single or multi-segment ident path with \
                                 lifetime-only generics). Tuple, \
                                 function-pointer, reference, slice, array, \
                                 trait-object self-types have no Python-side \
                                 class equivalent in the flowspace model."
                            .to_string(),
                    }
                })?;
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Fn(method) = impl_item {
                        let method_name = method.sig.ident.to_string();
                        // Convert ImplItemFn to ItemFn so it can flow
                        // through `build_host_function_from_rust_in_module`.
                        // Both shapes carry the same fields
                        // (`attrs` / `vis` / `sig` / `block`); the
                        // conversion is structural.
                        let item_fn = syn::ItemFn {
                            attrs: method.attrs.clone(),
                            vis: method.vis.clone(),
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        deferred_impls.push(DeferredImpl {
                            class_path: class_path.clone(),
                            method_name,
                            item_fn,
                        });
                    }
                }
            }
            // Generic `impl<T> Foo<T>` / `impl<E: Trait> X for E` /
            // `impl<const N: usize> Foo<N>` — type / const generic
            // parameters introduced in `generics.params` require
            // reification against a concrete instantiation (the
            // annotator's `SomeInstance.classdef` choice per call
            // site, mirroring upstream `MethodDesc.selfclassdef`).
            // Walker has no analyzer-side specialization yet, so
            // the methods cannot be attached to a single class
            // dict. Fail-loud rather than silently drop the methods
            // — local Rust source declaring such a block is not an
            // external-scope adaptation, it is a walker-modeling
            // gap. (Note: `impl Foo where Foo: SomeTrait { … }` is
            // NOT this case — `params` is empty, the `where` clause
            // only constrains concrete types, and the guard above
            // already accepted it per `classdesc.py:590-634`.)
            Item::Impl(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "`impl` block introduces type / const \
                             generic parameters: walker has no \
                             reification path; method graphs cannot \
                             be attached to a single class dict \
                             without an annotator-side specialization \
                             choice."
                        .to_string(),
                });
            }
            // Slice O23: `Item::Use` queues bindings for the pass-2
            // fixed-point loop. Mirrors Python `from x import y`:
            // `module.__dict__["y"] = x.y` once `x.y` is in scope.
            // Skip leading-`::`-rooted paths (`::std::result::Result`)
            // and well-known external prefixes (`crate::`, `super::`,
            // `self::`, `std::`, `core::`, `alloc::`) — multi-file /
            // external-crate resolution is its own future slice.
            // Glob imports (`use foo::*`) ARE handled: the path
            // resolves to a `HostObject::Class` namespace and every
            // entry of `class_dict_items` is mirrored into the
            // current registry partition (mirrors Python `from x
            // import *` copying every public name).
            Item::Use(item_use) => {
                if item_use.leading_colon.is_some() {
                    continue;
                }
                let mut staged: Vec<DeferredUse> = Vec::new();
                flatten_use_tree(&item_use.tree, Vec::new(), &mut staged);
                for du in staged {
                    if du.path_segments.is_empty() {
                        continue;
                    }
                    if matches!(classify_use_root(&du.path_segments[0]), UseRoot::External) {
                        continue;
                    }
                    deferred_uses.push(du);
                }
            }
            _ => {
                // TODO: silently-skipped item
                // kinds. Each carries a documented convergence path
                // and explicit reasoning for why fail-loud would
                // overreach beyond the strict-parity contract.
                //
                // - **`Item::Use` rooted at external prefixes**
                //   (`crate::`, `super::`, `self::`, `std::`,
                //   `core::`, `alloc::`, leading-`::`) and
                //   **External `Item::Mod`** (file-system
                //   `mod foo;` with no body) need cross-file /
                //   external-crate registry resolution that the
                //   closed-world walker does not implement. Short-
                //   circuited inside `Item::Use` / `Item::Mod`
                //   above rather than reaching this arm.
                //
                // - **`Item::Static` with `static mut`** — runtime-
                //   mutated module storage whose initial-value
                //   snapshot would silently produce stale constfold
                //   reads. Lookups fall through to
                //   `Builder::resolve_path_constant`'s mint-or-fail,
                //   the right shape for an opaque runtime-mutated
                //   symbol. Promoting to fail-loud would block any
                //   source file containing a live-store global from
                //   walking at all.
                //
                // - **`Item::Trait` / `Item::TraitAlias`** — the
                //   trait identity is not consulted in closed-world
                //   dispatch (`bookkeeper.py:431-442 getmethoddesc`
                //   keys on `(originclassdef, name, …)`). A trait
                //   declaration's only walker-observable effect would
                //   be registering an abstract HostObject; today's
                //   `impl Trait for X` flattens the method onto X's
                //   class dict regardless, so the trait carrier is
                //   unused.
                //
                // - **`Item::Type`** — Rust type aliases collapse at
                //   compile time. The flowspace model has no
                //   compile-time fold stage, so the alias has no
                //   runtime-observable binding.
                //
                // - **`Item::Macro`** — macros must expand before
                //   the walker parses the file. A surviving
                //   `Item::Macro` is by definition unexpanded; the
                //   walker has no expander.
                //
                // - **`Item::ForeignMod` / `Item::ExternCrate` /
                //   `Item::Union` / `Item::Verbatim`** — FFI, cross-
                //   crate registry, overlay storage, and syn parse
                //   verbatim respectively. Each needs its own slice
                //   promoting to fail-loud would
                //   block any source containing them from walking.
            }
        }
    }

    // Strict-parity Issue 1 pre-pass (2026-05-08): register every
    // deferred fn's metadata-only host into `module.__dict__` BEFORE
    // any body is lowered. Mirrors upstream Python's import-time
    // `def f` populating module dict ahead of flow analysis
    // (`flowcontext.py:847 find_global` reads live module dict). With
    // every sibling host already in place, body lowering supports
    // direct recursion (`fn f() { f() }`) AND mutual recursion
    // (`A` calls `B`, `B` calls `A`). Outer-module fns: `__globals__`
    // IS the module dict, so the body lowering passes `None` for
    // func_globals and lookups hit the partition keyed on
    // `module_id`. `class_ = None`: top-level fns are module-scoped
    // (audit 1.5).
    //
    // Audit 1.2: caller's `source_filename` / `source_text` are
    // threaded so the entry-point path can re-use this walker-built
    // host instead of re-building. When the caller passes `None`
    // (e.g. `register_rust_module(file)` fixtures), the legacy
    // empty-source behavior is preserved.
    //
    // `pending_fns` carries the (name, host, gf, host_code, item_fn)
    // tuple needed by `lower_body_into_pygraph` for each placeholder.
    // `metadata_failed` holds fns whose metadata extraction itself
    // failed (e.g. arg-name extraction error); they go back into
    // `deferred_fns` (which from this point is treated as a residue
    // set — the iterative loop below does not retry them since the
    // pre-register pass is one-shot). Fns whose bodies never lower
    // keep their placeholder host in module dict — the post-loop pass
    // does not roll them back, mirroring Python's import-time `def f`
    // semantic where failure surfaces at later `buildflowgraph` time
    // rather than at the walker pass.
    let mut pending_fns: Vec<(String, HostObject, GraphFunc, HostCode, &syn::ItemFn)> =
        Vec::with_capacity(deferred_fns.len());
    let mut metadata_failed: Vec<&syn::ItemFn> = Vec::new();
    for item_fn in deferred_fns.drain(..) {
        let name = item_fn.sig.ident.to_string();
        match build_host_metadata_parts(item_fn, module_id, source_filename, source_text, None) {
            Ok(HostMetadataParts {
                host,
                host_code,
                gf,
            }) => {
                register_module_global(module_id, &name, ConstValue::HostObject(host.clone()));
                pending_fns.push((name, host, gf, host_code, item_fn));
            }
            Err(_) => metadata_failed.push(item_fn),
        }
    }
    deferred_fns = metadata_failed;

    // Strict-parity (2026-05-11): impl-method handling splits into two
    // phases mirroring the inner walker. Phase A resolves the target
    // class + builds metadata + `class_set`s the placeholder ONCE;
    // Phase B retries body lowering with the host identity stable
    // across iterations. `pending_methods` carries the (host, gf,
    // host_code, item_fn) tuple for Phase B. Methods whose target
    // class never resolves stay in `deferred_impls` and never advance
    // to Phase A. Forward references between sibling methods on the
    // same class (`fn caller() { self.helper() }` ahead of `fn
    // helper(&self) { ... }`) need every sibling placeholder visible
    // through `class.class_get(name)` at body lowering time.
    // 5th tuple slot holds the impl-target class HostObject so the
    // method body's `&self` typed-input lift
    // (`build_flow::annotate_typed_ptr_inputs`) can resolve `Self`
    // against it via `func_globals`. The impl class doubles as the
    // method's class-side scope, matching the pyre convention
    // established by the inner-mod path (line 1154).
    let mut pending_methods: Vec<(HostObject, GraphFunc, HostCode, syn::ItemFn, HostObject)> =
        Vec::new();

    // Strict-parity round-4 (2026-05-11): cross-trait method-name
    // collision check, keyed on the *resolved* target HostObject
    // identity rather than the textual `class_path` from
    // `extract_impl_target_path`. Two `use a::Foo as X; use a::Foo as
    // Y; impl TraitA for X { fn name … } impl TraitB for Y { fn
    // name … }` resolve `["X"]` and `["Y"]` through
    // `try_resolve_use_path` to the *same* `Arc<HostObjectInner>`
    // (`identity_id() == Arc::as_ptr(&inner) as usize`); a textual
    // key would miss the alias and let the second `class_set`
    // silently overwrite the first's entry, losing dispatch identity.
    // The check therefore lives inside the Phase A resolution loop
    // (after `try_resolve_use_path` succeeds, before metadata build).
    // Map value carries the diagnostic class-path of the first
    // sighting so the error message names both colliding paths.
    let mut seen_method_collisions: StdHashMap<(usize, String), Vec<String>> = StdHashMap::new();

    // Pass 2 (Slice O17 + Issue 1 retrofit): iteratively lower
    // bodies, resolve impls and uses until no more progress is made.
    // Each iteration retries every still-pending fn body — failures
    // leave the placeholder host in module dict for the next
    // iteration so a downstream impl/use sweep can resolve a
    // transitive dependency. Loop terminates when a full sweep
    // yields zero registrations.
    //
    // Bound: each iteration consumes at least one pending entry on
    // success or exits via the no-progress check, so the loop runs
    // at most `pending_fns.len() + deferred_impls.len() +
    // deferred_uses.len() + 1` iterations.
    loop {
        let mut made_progress = false;
        let mut still_pending = Vec::with_capacity(pending_fns.len());
        for (name, host, gf, host_code, item_fn) in pending_fns.drain(..) {
            match lower_body_into_pygraph(
                item_fn,
                module_id,
                None,
                gf.clone(),
                &host_code,
                source_text,
            ) {
                Ok(pygraph) => {
                    clear_walker_error(&host);
                    register_walker_pygraph(host, pygraph);
                    made_progress = true;
                }
                Err(err) => {
                    register_walker_error(host.clone(), err.to_string());
                    still_pending.push((name, host, gf, host_code, item_fn));
                }
            }
        }
        pending_fns = still_pending;

        // Phase A (impls): for each unresolved impl, try resolving
        // the target class. On success, build metadata + `class_set`
        // the placeholder + push to `pending_methods`. Slice O18 +
        // O24: each method needs (a) the target class path resolvable
        // through the registry partition (single segment) or via
        // inline-mod cascade (multi-segment, Slice O24) to a
        // `HostObject::Class` and (b) its body to lower cleanly.
        // Successful entries `class.class_set(method_name, <host>)`
        // to add the method to the class dict — mirrors Python `class
        // Foo: def bar(self): ...` populating `Foo.__dict__["bar"]`.
        // Methods whose target class never resolves (`impl Bar { … }`
        // where `Bar` is absent or not a class) stay deferred but
        // never make progress, so the loop termination condition
        // still bounds them.
        //
        // Audit 1.5 (2026-05-08): thread the resolved target `class`
        // so `GraphFunc.class_` carries class-owned identity for the
        // impl method (mirrors upstream `func.class_` at `class Foo:
        // def bar(self): ...`).
        //
        // Audit 1.2 (2026-05-08): same caller source_filename /
        // source_text threading as the deferred_fn loop above.
        let mut still_deferred_impls = Vec::with_capacity(deferred_impls.len());
        for di in deferred_impls.drain(..) {
            let class = match try_resolve_use_path(&di.class_path, module_id, None) {
                Some(ConstValue::HostObject(h)) if h.is_class() => h,
                _ => {
                    still_deferred_impls.push(di);
                    continue;
                }
            };
            // Strict-parity round-4 (2026-05-11): collision check on
            // the resolved HostObject identity. See
            // `seen_method_collisions` declaration above for why the
            // textual `class_path` key is insufficient.
            let collision_key = (class.identity_id(), di.method_name.clone());
            if let Some(prior_path) = seen_method_collisions.get(&collision_key) {
                return Err(AdapterError::Unsupported {
                    reason: format!(
                        "Cross-trait method-name collision on `{}::{}`: \
                         a prior `impl … for {}` block already wrote the \
                         method name into the flat class dict (resolved \
                         host identity matches across alias paths). \
                         Upstream `classdesc.py:590-634 \
                         add_source_attribute` has no trait-namespace \
                         channel; rename one method or move the impls \
                         into separate files.",
                        di.class_path.join("::"),
                        di.method_name,
                        prior_path.join("::"),
                    ),
                });
            }
            let parts = match build_host_metadata_parts(
                &di.item_fn,
                module_id,
                source_filename,
                source_text,
                Some(class.clone()),
            ) {
                Ok(p) => p,
                Err(_) => {
                    still_deferred_impls.push(di);
                    continue;
                }
            };
            let HostMetadataParts {
                host,
                host_code,
                gf,
            } = parts;
            seen_method_collisions.insert(collision_key, di.class_path.clone());
            class.class_set(&di.method_name, ConstValue::HostObject(host.clone()));
            pending_methods.push((host, gf, host_code, di.item_fn, class.clone()));
            made_progress = true;
        }
        deferred_impls = still_deferred_impls;

        // Phase B (impls): retry pending method bodies; host already
        // in class dict via the Phase A `class_set`.
        let mut still_pending_methods = Vec::with_capacity(pending_methods.len());
        for (host, gf, host_code, item_fn, impl_class) in pending_methods.drain(..) {
            match lower_body_into_pygraph(
                &item_fn,
                module_id,
                Some(&impl_class),
                gf.clone(),
                &host_code,
                source_text,
            ) {
                Ok(pygraph) => {
                    clear_walker_error(&host);
                    register_walker_pygraph(host, pygraph);
                    made_progress = true;
                }
                Err(err) => {
                    register_walker_error(host.clone(), err.to_string());
                    still_pending_methods.push((host, gf, host_code, item_fn, impl_class));
                }
            }
        }
        pending_methods = still_pending_methods;

        // Slice O23: retry deferred uses against the registry
        // partition. Each successful resolution `register_module_global`s
        // `binding_name` → `resolved_value`; failure stays deferred.
        // Mirrors Python `from x import y`: `module.__dict__["y"] =
        // x.y` once `x.y` is resolvable.
        let mut still_deferred_uses = Vec::with_capacity(deferred_uses.len());
        for du in deferred_uses.drain(..) {
            if du.glob {
                let Some(ConstValue::HostObject(target)) =
                    try_resolve_use_path(&du.path_segments, module_id, None)
                else {
                    still_deferred_uses.push(du);
                    continue;
                };
                if !target.is_class() {
                    still_deferred_uses.push(du);
                    continue;
                }
                // Strict-parity (2026-05-10): mirror Python `from foo
                // import *` which skips leading-underscore names when
                // the source module has no `__all__`
                // (`pyopcode.py:2221 import_star_skip_underscore`). Rust
                // glob import has different visibility semantics, but
                // the comment above explicitly claims `from x import *`
                // parity, so honour the underscore filter to match
                // upstream's import-star semantic in absence of an
                // `__all__` analogue.
                for (name, value) in target.class_dict_items() {
                    if name.starts_with('_') {
                        continue;
                    }
                    register_module_global(module_id, &name, value);
                }
                made_progress = true;
                continue;
            }
            match try_resolve_use_path(&du.path_segments, module_id, None) {
                Some(value) => {
                    register_module_global(module_id, &du.binding_name, value);
                    made_progress = true;
                }
                None => still_deferred_uses.push(du),
            }
        }
        deferred_uses = still_deferred_uses;

        if !made_progress
            || (pending_fns.is_empty()
                && deferred_fns.is_empty()
                && deferred_impls.is_empty()
                && pending_methods.is_empty()
                && deferred_uses.is_empty())
        {
            break;
        }
    }

    // Strict-parity (2026-05-09): a fn whose body never lowered keeps
    // its pre-registered placeholder host in `module.__dict__`. Mirrors
    // upstream Python's `def f` populating the module dict at import
    // time regardless of any later flow-analysis state — failure
    // surfaces lazily when a caller actually invokes
    // `buildflowgraph(callee)` on that host and finds no PyGraph
    // attached, not at walker time. Rolling back the placeholder would
    // diverge: a sibling fn that already captured the placeholder via
    // `LOAD_GLOBAL` during its own body lowering would point at a host
    // that is no longer present in the live module dict.
    //
    // `pending_methods` residue carries the same semantic for impl
    // methods whose body never lowered: the `class_set` placeholder
    // stays in the class dict and `buildflowgraph` reports the empty
    // `co_code` lazily.
    let _ = pending_fns;
    let _ = pending_methods;

    // Strict-parity: a local-rooted `use foo::Bar` that never
    // resolved through the fixed-point loop is a genuine import
    // miss. External-rooted prefixes (`crate::`, `super::`, `self::`,
    // `std::`, `core::`, `alloc::`, leading-`::`) are filtered at
    // intake (UseRoot::External above), so anything left in
    // `deferred_uses` is a local-rooted reference whose target
    // failed to register. Upstream `pyopcode.py` IMPORT_FROM raises
    // `ImportError` at import time when `getattr(foo, "Bar")` fails;
    // Rust's compiler likewise rejects unresolved `use`. Surface the
    // miss as `AdapterError::Unsupported` so registration does not
    // silently complete with a missing namespace entry.
    if let Some(du) = deferred_uses.into_iter().next() {
        let path = du.path_segments.join("::");
        let suffix = if du.glob {
            "::*".to_string()
        } else if du
            .path_segments
            .last()
            .is_some_and(|seg| seg != &du.binding_name)
        {
            format!(" as {}", du.binding_name)
        } else {
            String::new()
        };
        return Err(AdapterError::Unsupported {
            reason: format!(
                "unresolved local-rooted `use {path}{suffix}`: walker \
                 found no registered binding for the path's head segment \
                 after the fixed-point loop. Either the target is \
                 declared below an external-rooted boundary (filter at \
                 intake), or no `pub` item with that name reached the \
                 registry partition."
            ),
        });
    }

    // Strict-parity: an `impl … for X { ... }` block whose target `X`
    // never resolved to a registered `HostObject::Class` would
    // otherwise drop every method graph silently. Closed-world Rust
    // source declares its impl targets locally, so a non-resolving
    // target means the metadata-only carrier is missing — a
    // registration miss, not a documented external-scope adaptation.
    if let Some(di) = deferred_impls.into_iter().next() {
        return Err(AdapterError::Unsupported {
            reason: format!(
                "unresolved `impl ... for {}` target after fixed-point \
                 loop: walker found no registered `HostObject::Class` \
                 for the self-type path. Methods `{}::{}` (and any \
                 siblings in the same impl block) cannot be added to a \
                 class dict that does not exist.",
                di.class_path.join("::"),
                di.class_path.join("::"),
                di.method_name,
            ),
        });
    }

    Ok(module_id)
}

/// Build the `HostObject::Class` corresponding to `item_enum` and
/// populate its class dict with every variant as a child class.
///
/// Mirrors the closest Python analogue:
///
/// ```python
/// class StepResult: pass
/// class StepResult_Continue(StepResult): pass
/// class StepResult_Return(StepResult): pass
/// StepResult.Continue = StepResult_Continue
/// StepResult.Return = StepResult_Return
/// ```
///
/// Each variant's child class carries the parent in its `bases`
/// vector so `is_subclass_of(parent)` returns `true` — matches
/// upstream `classdef.py:336 ClassDef.lookup_filter` walking the
/// `__bases__` chain when computing `isinstance(x, StepResult)`
/// against an instance of `StepResult_Continue`.
///
/// The variant carrier qualname is `"<EnumName>.<VariantName>"`
/// (dot-separator) matching upstream Python's `cls.__qualname__`
/// shape for nested classes.
fn build_host_class_from_enum(item_enum: &ItemEnum) -> HostObject {
    let parent_name = item_enum.ident.to_string();
    let parent = HostObject::new_class(&parent_name, vec![]);
    for variant in &item_enum.variants {
        let v_name = variant.ident.to_string();
        let v_qualname = format!("{}.{}", parent_name, v_name);
        let v_class = HostObject::new_class(v_qualname, vec![parent.clone()]);
        parent.class_set(&v_name, ConstValue::HostObject(v_class));
    }
    parent
}

/// Evaluate a `const` RHS expression to a [`ConstValue`] using
/// `bindings` as the lookup environment for prior `const` names in
/// the same module walk.
///
/// Tri-state return:
///
/// - `Ok(Some(v))` — RHS evaluated successfully; bind `name = v`.
/// - `Ok(None)` — decline-fold. Used for **shape unsupported** at
///   this evaluator: function call, struct literal, multi-segment
///   path. Walker silently skips (walker-coverage gap, Issue 2.3
///   follow-up) and the const stays unbound; downstream lookups
///   fall through to mint-or-fail.
/// - `Err(e)` — supported shape but the operation cannot produce
///   a representable binding at module top level:
///   - **Type mismatch** in unary / binary op (`!"x"`,
///     `1 < "x"` ordering).
///   - **Zero-divisor** (`1 / 0`, `1 % False`, `1.0 / 0.0`).
///   - **Negative shift count** (`1 << -1`, `1 >> -1`) per
///     `operator.lshift/rshift` raising `ValueError` upstream.
///   - **`i64` overflow** on `Lit::Int` / unary `Neg` / binop
///     (`MAX + 1`, `MIN / -1`, `1 << 64`). Upstream Python module
///     execution would bind a bignum here and continue; pyre
///     adapter has no bignum carrier so surfaces the divergence as
///     `AdapterError::Unsupported`, not an upstream-shaped
///     `FlowingError`. **Note**: this is structurally distinct from
///     `PureOperation.constfold`'s
///     `can_overflow and type(result) is long` decline arm at
///     `operation.py:140-142`. THAT rule lives at function-body
///     constfold time — it lets the runtime evaluate the op as
///     bignum after declining. There is no analogous "let the
///     runtime do it" at module-import-time prepass.
///   - **Unresolved single-segment Path** — name not in `bindings`
///     and not in the registry partition. Mirrors
///     `flowcontext.py:853 find_global` raising
///     `FlowingError("global name '...' is not defined")` after
///     both globals and builtins miss. Walker-coverage gaps for
///     `Item::Fn` / `Item::Use` / `Item::Mod` / `Item::Impl`
///     surface here — those are fixed by extending walker
///     coverage, not by silently dropping the binding.
///   Walker propagates the error to abort the walk, matching
///   upstream Python's module-execution-aborts-on-exception
///   semantics (`Y = X + 1; X = 1` fails on `Y` with NameError;
///   the `X = 1` statement never runs).
///
/// Supported shapes:
///
/// - `Lit::Int(n)` → `ConstValue::Int(n)` (parsed as `i64`).
/// - `Lit::Bool(b)` → `ConstValue::Bool(b)`.
/// - `Lit::Str(s)` → `ConstValue::uni_str(s)`. Matches the in-body
///   `Lit::Str` lowering at `build_flow.rs::lower_literal` and
///   Python 3 unicode-string semantics — every `"..."` literal is
///   unicode regardless of where it appears.
/// - `Lit::ByteStr(s)` → `ConstValue::byte_str(s)` (Rust `b"..."`
///   bytes literal stays bytes).
/// - `Lit::Float(f)` → `ConstValue::Float(f.to_bits())`. Same shape
///   as `build_flow.rs::lower_literal::Lit::Float`. Out-of-`f64`-range
///   raises `AdapterError::Unsupported` at module top level (no carrier).
/// - `Lit::Char(c)` → `ConstValue::uni_str(c.to_string())`. Single-
///   char unicode string — upstream RPython has no `char` type,
///   single-char strings fill the role.
/// - `Lit::Byte(b)` → `ConstValue::ByteStr(vec![b])`. Mirrors
///   `build_flow.rs::lower_literal::Lit::Byte`.
/// - `-<Lit::Int>` / `-<Lit::Float>` (unary negation over a
///   numeric literal) → `ConstValue::Int(-n)` /
///   `ConstValue::Float(-f).to_bits()`. `syn` parses `const X:
///   i64 = -1` as `Expr::Unary { op: Neg, expr: Lit(1) }` (and
///   likewise for floats), not a signed literal, so unwrap one
///   level. Mirrors `operation.rs::pyfunc`'s
///   `(OpKind::Neg, [ConstValue::Int])` and
///   `(OpKind::Neg, [ConstValue::Float])` arms so a top-level
///   const and an in-body unary `-` agree on the value.
/// - **`Expr::Path` (single segment)** → resolution chain mirrors
///   upstream `flowcontext.py:845-853 find_global`:
///   1. `bindings.get(name)` — per-walk source-order accumulator,
///      stand-in for the in-progress module dict.
///   2. `module_globals_lookup(module_id, name)` — registry
///      partition; covers re-walks where the second walk's fresh
///      `bindings` is empty but the partition was populated by the
///      first walk.
///   3. `pyre_stdlib_lookup(name)` — closed-world builtins
///      (`Ok` / `Some` / `Err` / `Result` / `Option`). Mirrors
///      upstream's `getattr(__builtin__, varname)` second arm.
///   4. NameError. All three channels miss matches upstream's
///      `AttributeError → FlowingError` final step.
///   Multi-segment paths fall through to `None` (a path like
///   `mod::CONST_X` would require cross-file lookup and is out
///   of scope for this slice).
/// - **`Expr::Binary { Add | Sub | Mul | Div | Rem | Shl | Shr |
///   BitAnd | BitOr | BitXor, lhs, rhs }`** over `Int` operands →
///   `Int(a OP b)`. `Div` / `Rem` use Python floor-div / floor-mod
///   semantics (negative-divisor sign-flip) line-by-line per
///   `rpython/rtyper/rint.py:398 ll_int_py_div` /
///   `:496 ll_int_py_mod`, matching `flowspace::operation`'s
///   `int_py_floor_div` / `int_py_mod` so a `const Y = 3 / -2` at
///   module top level and `3 / -2` inside a function body produce
///   the same value. Zero divisor surfaces as
///   `Err(ZeroDivisionError)` per upstream `operator.floordiv(_, 0)`
///   raising at import time. `i64` overflow on `+` / `-` / `*` /
///   `i64::MIN / -1` / shifts that don't fit in `u32` / `>= 64`
///   surface as `AdapterError::Unsupported` (pyre adapter has no bignum
///   carrier; the function-body `PureOperation.constfold` decline
///   path at `operation.py:140-142` does not apply at import-time
///   prepass — see fn-level doc).
/// - **`Expr::Binary { Add | Sub | Mul | Div, lhs, rhs }`** over
///   `Float` operands → `Float(a OP b)`. Mirrors `operation.rs::pyfunc`
///   Float arms (`operation.rs:1367-1383`). Rust `BinOp::Div` over
///   `f64` is true division (matches Python `/`). `Float / 0.0`
///   raises `Err(ZeroDivisionError)` per Python semantics.
/// - **`Expr::Binary { Eq | Ne, lhs, rhs }`** over any operand
///   pair → `Bool(...)`. Same-type pairs use the matching arm's
///   structural equality; mixed-type pairs (e.g. `Int` vs `UniStr`)
///   return `Bool(false)` for `Eq` / `Bool(true)` for `Ne` — Python
///   3 does NOT raise on `==` / `!=` between distinct primitive
///   types. Numeric coercion for `True == 1` etc. is folded by
///   `HLOperation::constfold` once the const reaches the SSA layer.
/// - **`Expr::Binary { Lt | Le | Gt | Ge, lhs, rhs }`** over
///   same-type operands → `Bool(...)`. Mixed-type ordering raises
///   `Err(TypeError)` matching Python 3's
///   `'<' not supported between instances of …`.
/// - **`Expr::Binary { And | Or, lhs, rhs }`** over `Bool`
///   operands → `Bool(...)`. Rust `&&`/`||` are typed `bool ->
///   bool -> bool` so unlike Python's value-returning
///   short-circuit, the result is always `Bool`.
/// - **`Expr::Unary { Not, expr }`** over `Bool` → `Bool(!b)` (Rust
///   `!bool` is logical negation), or over `Int` → `Int(!n)` (Rust
///   `!int` is bitwise complement, mirroring Python `~`).
/// - **`Expr::Paren(expr)` / `Expr::Group(expr)`** — transparent
///   delegation to the inner expression. Upstream Python has no
///   parenthesisation node (parens evaporate at parse time and only
///   affect operator precedence) so a `const X: bool = (1 == 2)`
///   resolves identically to `const X: bool = 1 == 2`. Mirrors the
///   body lowerer's transparency at `build_flow.rs:1317`.
fn eval_const_expr(
    expr: &Expr,
    bindings: &StdHashMap<String, ConstValue>,
    module_id: ModuleId,
) -> Result<Option<ConstValue>, AdapterError> {
    let raise = |reason: String| AdapterError::Flowing { reason };
    let unsupported = |reason: String| AdapterError::Unsupported { reason };
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            // Literal that does not fit in `i64`. Upstream Python
            // module-top-level `X = 9223372036854775808` would bind
            // `X` to a Python long (bignum) without raising — Python
            // ints are arbitrary precision. Pyre's adapter has no
            // bignum carrier, so the binding cannot be created. The
            // orthodox response is `AdapterError::Unsupported`:
            // declining to `Ok(None)` would silently produce a
            // `module.__dict__` missing a name that upstream Python
            // would have bound, creating worse divergence than
            // aborting the walk. This is not a `FlowingError` because
            // upstream Python does not raise.
            //
            // Note: this is structurally distinct from
            // `PureOperation.constfold`'s `can_overflow and type(result)
            // is long` decline arm at `operation.py:140-142`. THAT
            // rule lives at function-body constfold time — it lets the
            // runtime evaluate the op as bignum. There is no analogous
            // "let the runtime do it" at module-import-time prepass:
            // upstream Python would have already produced the bignum
            // by the time this prepass ran.
            Lit::Int(n) => match n.base10_parse::<i64>() {
                Ok(v) => Ok(Some(ConstValue::Int(v))),
                Err(_) => Err(unsupported(format!(
                    "integer literal {} exceeds i64 carrier; bignum constants are not supported",
                    n
                ))),
            },
            Lit::Bool(b) => Ok(Some(ConstValue::Bool(b.value))),
            // `"..."` literal — unicode. Same shape as
            // `build_flow.rs::lower_literal::Lit::Str` so the
            // identical `"abc"` source carries the identical
            // ConstValue regardless of position.
            Lit::Str(s) => Ok(Some(ConstValue::uni_str(s.value()))),
            // `b"..."` literal — bytes. Mirrors
            // `build_flow.rs::lower_literal::Lit::ByteStr`.
            Lit::ByteStr(s) => Ok(Some(ConstValue::ByteStr(s.value()))),
            // Float literal — `ConstValue::Float` stores
            // `f64::to_bits()` so the enum keeps `Eq + Hash`
            // (`model.rs:1696-1701`). Matches the in-body shape
            // at `build_flow.rs::lower_literal::Lit::Float`. Out-
            // of-`f64`-range surfaces `AdapterError::Unsupported` for
            // the same reason `Lit::Int` overflow does: upstream
            // Python would still bind a value (even if `inf`), but
            // the representation we encode through `f64::to_bits()`
            // cannot carry the source-text the user wrote.
            Lit::Float(f) => match f.base10_parse::<f64>() {
                Ok(v) => Ok(Some(ConstValue::Float(v.to_bits()))),
                Err(e) => Err(unsupported(format!(
                    "float literal out of f64 range; float carrier cannot represent source literal: {e}"
                ))),
            },
            // Rust `'a'` char literal — single Unicode scalar.
            // Upstream RPython has no `char` type; single-char
            // strings fill the role (`model.py:658`,
            // `operation.py` string ops accept `len == 1` like
            // any other unicode). Emit as `ConstValue::UniStr`
            // matching `build_flow.rs::lower_literal::Lit::Char`.
            Lit::Char(ch) => Ok(Some(ConstValue::uni_str(ch.value().to_string()))),
            // Rust `b'a'` byte literal — single byte. Mirrors
            // `build_flow.rs::lower_literal::Lit::Byte` which
            // emits `ConstValue::ByteStr(vec![b])`.
            Lit::Byte(b) => Ok(Some(ConstValue::ByteStr(vec![b.value()]))),
            _ => Ok(None),
        },
        // `const X: i64 = -1` — `syn` parses as `Unary { op: Neg,
        // expr: Lit(1) }` rather than a signed literal. Unwrap one
        // level so the common signed-int form is recognised.
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let Some(v) = eval_const_expr(expr, bindings, module_id)? else {
                return Ok(None);
            };
            match v {
                // `-(i64::MIN)` overflows the `i64` carrier. Upstream
                // Python would bind `2**63` (bignum) without raising;
                // pyre adapter has no bignum carrier, so surface as
                // `AdapterError::Unsupported`. See `Lit::Int` doc
                // above for why decline-fold (Ok(None)) is wrong at
                // module top level (it lives at function-body constfold
                // time per `operation.py:140-142`, not import-time
                // prepass).
                ConstValue::Int(n) => match n.checked_neg() {
                    Some(neg) => Ok(Some(ConstValue::Int(neg))),
                    None => Err(unsupported(format!(
                        "-i64::MIN ({}) exceeds i64 carrier; bignum constants are not supported",
                        n
                    ))),
                },
                // `-3.14` / `-2.5` etc. — `syn` parses the leading
                // minus as `Expr::Unary { Neg, Lit::Float(...) }`,
                // not a signed float literal. Mirrors
                // `operation.rs::pyfunc`'s
                // `(OpKind::Neg, [ConstValue::Float(bits)]) =>
                //   Some(ConstValue::float(-f64::from_bits(*bits)))`
                // so a module-top `const X: f64 = -3.14` and an
                // in-body `-3.14` agree on the bit pattern.
                ConstValue::Float(bits) => Ok(Some(ConstValue::float(-f64::from_bits(bits)))),
                other => Err(raise(format!(
                    "TypeError: unary `-` operand must be Int or Float, got {other:?}"
                ))),
            }
        }
        // `const X: bool = !true` over Bool → logical negation.
        // `const X: i64 = !0` over Int → bitwise complement
        // (Rust's `!` on integers is the same as Python's `~`).
        Expr::Unary(ExprUnary {
            op: UnOp::Not(_),
            expr,
            ..
        }) => {
            let Some(v) = eval_const_expr(expr, bindings, module_id)? else {
                return Ok(None);
            };
            match v {
                ConstValue::Bool(b) => Ok(Some(ConstValue::Bool(!b))),
                ConstValue::Int(n) => Ok(Some(ConstValue::Int(!n))),
                other => Err(raise(format!(
                    "TypeError: unary `!` operand must be Bool or Int, got {other:?}"
                ))),
            }
        }
        // `const Y: i64 = X` — single-segment path. Resolution
        // order matches upstream Python's import-time name
        // resolution against `module.__dict__`: per-walk source-
        // order `bindings` first, then the registry partition
        // (Issue 2.3 — covers re-walk: a path-keyed `module_id`
        // already populated by a prior walk has the entry in
        // the registry but not in this walk's fresh `bindings`).
        // Multi-segment paths fall through to `Ok(None)` (cross-
        // file lookup is a separate slice).
        //
        // Unresolved single-segment names raise
        // `FlowingError("global name '...' is not defined")`,
        // matching upstream `flowcontext.py:853 find_global` which
        // raises after both globals and builtins miss. Aborts the
        // walker — same as Python module execution where
        // `Y = X + 1; X = 1` fails with NameError on Y and the
        // remaining statements never execute. Walker gaps for
        // `Item::Use` / `Item::Mod` / `Item::Impl` (and `Item::Fn`)
        // surface here as parity-correct hard errors rather than
        // silent dropped bindings — pyre adapter cannot pretend a
        // name resolves when upstream Python's same execution would
        // have aborted.
        Expr::Path(ExprPath {
            qself: None, path, ..
        }) if path.segments.len() == 1 => {
            let seg = &path.segments[0];
            if !seg.arguments.is_empty() {
                return Ok(None);
            }
            let name = seg.ident.to_string();
            // Resolution order mirrors `flowcontext.py:845-853 find_global`:
            //
            //     try:
            //         value = w_globals.value[varname]   # 1. globals
            //     except KeyError:
            //         try:
            //             value = getattr(__builtin__, varname)  # 2. builtins
            //         except AttributeError:
            //             raise FlowingError("global name '%s' is not defined")
            //
            // pyre's adapter has two channels in place of one Python
            // dict: per-walk source-order `bindings` (the in-progress
            // module dict) and `module_globals_lookup` (re-walks /
            // pre-walk registry partition). Both stand in for the
            // `w_globals.value[varname]` lookup. The closed-world
            // `pyre_stdlib_lookup` is the `__builtin__` analogue
            // (`Ok` / `Some` / `Err` / `Result` / `Option` HostObject
            // singletons; documented at `host_env.rs:170-191` as the
            // adapter's auxiliary builtins). NameError fires only when
            // all three channels miss, matching upstream's final
            // `AttributeError → FlowingError` step.
            match bindings
                .get(&name)
                .cloned()
                .or_else(|| module_globals_lookup(module_id, &name))
                .or_else(|| pyre_stdlib_lookup(&name).map(ConstValue::HostObject))
            {
                Some(v) => Ok(Some(v)),
                None => Err(raise(format!("global name '{name}' is not defined"))),
            }
        }
        // `const Y = X + 1` etc — operator dispatch over evaluated
        // operands. Type-mismatch and zero-divisor surface as
        // `Err`, matching upstream Python's import-time
        // exception. `i64` overflow at module top level is a local
        // carrier limitation: upstream Python would bind a bignum, so
        // the adapter returns `Unsupported` rather than silently
        // dropping the binding or pretending Python raised. Truly
        // unsupported operator/operand combinations return `Ok(None)`
        // (silent skip).
        //
        // **Short-circuit `&&` / `||`** (codex parity audit,
        // 2026-05-05): both Rust source and upstream Python's
        // `and`/`or` are short-circuit at the source level —
        // `false && BAD` evaluates to `false` without touching
        // `BAD`, and `true || BAD` evaluates to `true` similarly.
        // The naive "evaluate both operands then dispatch" path
        // would force-evaluate the RHS, diverging in cases where
        // the RHS would itself raise (e.g. unbound name, division
        // by zero). LHS is evaluated first; if it's a `Bool` whose
        // value determines the result, the RHS is skipped.
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let Some(lhs) = eval_const_expr(left, bindings, module_id)? else {
                return Ok(None);
            };
            if let ConstValue::Bool(b) = lhs {
                match op {
                    BinOp::And(_) if !b => return Ok(Some(ConstValue::Bool(false))),
                    BinOp::Or(_) if b => return Ok(Some(ConstValue::Bool(true))),
                    _ => {}
                }
            }
            let Some(rhs) = eval_const_expr(right, bindings, module_id)? else {
                return Ok(None);
            };
            eval_binop(op, &lhs, &rhs)
        }
        // `Expr::Paren` / `Expr::Group` are transparent. Upstream
        // Python has no parenthesisation node — parens evaporate at
        // parse time and only affect operator precedence — so a
        // `const X: bool = (1 == 2)` resolves identically to
        // `const X: bool = 1 == 2`. Mirrors the body lowerer's
        // transparency at `build_flow.rs:1317
        // Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr)`.
        // `Expr::Group` is the proc-macro-emitted invisible grouping
        // wrapper; same semantic.
        Expr::Paren(syn::ExprParen { expr, .. }) | Expr::Group(syn::ExprGroup { expr, .. }) => {
            eval_const_expr(expr, bindings, module_id)
        }
        _ => Ok(None),
    }
}

/// Evaluate a Rust `BinOp` over two evaluated `ConstValue` operands.
///
/// Tri-state return mirrors [`eval_const_expr`]:
///
/// - `Ok(Some(v))` — success.
/// - `Ok(None)` — decline-fold for **shape unsupported** by this
///   evaluator (operator/operand pair never reachable from typed
///   Rust source, e.g. `BinOp::Add` over `(UniStr, Int)`). Walker
///   silently skips so the const stays unbound and downstream
///   lookups fall through to mint-or-fail. **Note**: `i64` overflow
///   on `+` / `-` / `*` / `/` / `%` / `<<` does NOT decline-fold at
///   module top level — it returns `AdapterError::Unsupported`
///   instead. The
///   `PureOperation.constfold` decline arm at `operation.py:140-142`
///   lives at *function-body* constfold time, where the runtime
///   evaluates the op as bignum after the fold is declined; there
///   is no analogous "let the runtime do it" at module-import-time
///   prepass — upstream Python would have already produced the
///   bignum and bound it. Pyre adapter surfaces the divergence
///   (no bignum carrier) loudly.
///   `>>` count `>= 64` is the one exception: `operation.py:484`
///   registers `rshift` without `ovf=True`, and Python's arithmetic
///   right shift saturates rather than producing a long, so the
///   constfold *can* still produce a representable result.
/// - `Err(e)` — supported shape but operation cannot produce a
///   representable binding:
///   - **Type mismatch** in unary / binary op
///     (`!"x"`, `1 < "x"` ordering).
///   - **Zero-divisor** (`1 / 0`, `1 // False`, `1 % 0`,
///     `1.0 / 0.0`).
///   - **Negative shift count** (`1 << -1`, `1 >> -1`).
///     Upstream `operator.lshift` / `operator.rshift` raise
///     `ValueError: negative shift count` directly; the
///     constfold's `try/except Exception` (`operation.py:120-127`)
///     captures and re-raises as `FlowingError`.
///   - **`i64` overflow** on `Add` / `Sub` / `Mul` / `Div` (`MIN /
///     -1`) / `Shl` (`count >= 64`). Upstream Python would bind
///     a bignum; pyre adapter has no bignum carrier so surfaces
///     the divergence as `AdapterError::Unsupported`.
///   Walker propagates the error to abort the walk, matching
///   upstream Python's module-execution-aborts-on-exception
///   semantics.
fn eval_binop(
    op: &BinOp,
    lhs: &ConstValue,
    rhs: &ConstValue,
) -> Result<Option<ConstValue>, AdapterError> {
    let raise = |reason: String| AdapterError::Flowing { reason };
    let unsupported = |reason: String| AdapterError::Unsupported { reason };
    let int_zerodiv = |op_name: &str| {
        Err(raise(format!(
            "ZeroDivisionError: integer {op_name} by zero"
        )))
    };
    let int_overflow = |op_name: &str, a: i64, b: i64| {
        Err(unsupported(format!(
            "i64 {op_name} ({a} {op_name} {b}) exceeds i64 carrier; bignum constants are not supported"
        )))
    };
    match (lhs, rhs) {
        (ConstValue::Int(a), ConstValue::Int(b)) => match op {
            // `+` / `-` / `*` overflow → `Unsupported`. At
            // module top level upstream Python would bind a bignum;
            // pyre adapter has no bignum carrier so the divergence
            // surfaces loudly. (Function-body constfold is a
            // *different* path — it declines per
            // `operation.py:140-142`.)
            BinOp::Add(_) => match a.checked_add(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("+", *a, *b),
            },
            BinOp::Sub(_) => match a.checked_sub(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("-", *a, *b),
            },
            BinOp::Mul(_) => match a.checked_mul(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("*", *a, *b),
            },
            // `Div` and `Rem` use Python floor-div / floor-mod
            // semantics (negative-divisor sign-flip) — same helpers
            // as `flowspace::operation::int_py_floor_div` /
            // `int_py_mod` so a `const Y: i64 = 3 / -2` and the
            // function-body `3 / -2` produce the same value. Zero
            // divisor → `Err(ZeroDivisionError)` per upstream
            // `operator.floordiv(_, 0)` raising. `i64::MIN / -1`
            // overflow → `Unsupported` at module top level
            // (upstream binds bignum `2**63`).
            BinOp::Div(_) => {
                if *b == 0 {
                    return int_zerodiv("division");
                }
                match int_py_floor_div(*a, *b) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("/", *a, *b),
                }
            }
            BinOp::Rem(_) => {
                if *b == 0 {
                    return int_zerodiv("modulo");
                }
                // `x % -1 == 0` for any `x` upstream — fold
                // unconditionally. `int_py_mod` would otherwise
                // route through `i64::MIN.checked_rem(-1)` which
                // returns `None` (the *quotient* `2^63` overflows,
                // even though the remainder `0` itself fits),
                // declining a fold that has a well-defined value.
                if *b == -1 {
                    return Ok(Some(ConstValue::Int(0)));
                }
                // `int_py_mod` only returns `None` on
                // `i64::MIN.checked_rem(-1)` (handled above) —
                // every other `(x, y)` with `y != 0` produces a
                // well-defined result. Unwrap defensively, but if
                // a future helper change introduces another
                // overflow path, surface it as Err.
                match int_py_mod(*a, *b) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("%", *a, *b),
                }
            }
            // `<<` — Negative `count` raises `ValueError: negative
            // shift count` upstream (`operator.lshift(_, -1)`).
            // `count >= 64` overflows the i64 carrier; upstream
            // Python would bind a bignum, pyre adapter cannot, so
            // surface as `Unsupported`.
            BinOp::Shl(_) => {
                if *b < 0 {
                    return Err(raise(format!("ValueError: negative shift count {b}")));
                }
                match u32::try_from(*b).ok().and_then(|n| a.checked_shl(n)) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("<<", *a, *b),
                }
            }
            // `>>` — Python's arithmetic right shift saturates
            // on counts `>= 64`: any non-negative `a` shifts to
            // `0`, any negative `a` sign-extends to `-1`. Rust's
            // `i64 >> 64` panics in debug, so we explicit-saturate.
            // `rshift` is NOT registered with `ovf=True` upstream
            // (`operation.py:484 add_operator('rshift', 2,
            // dispatch=2, pure=True)`), so the long-decline arm
            // doesn't apply — we MUST fold the saturation result.
            // Negative count surfaces as `ValueError` like `Shl`.
            BinOp::Shr(_) => {
                if *b < 0 {
                    return Err(raise(format!("ValueError: negative shift count {b}")));
                }
                let saturated = if *a < 0 { -1 } else { 0 };
                let folded = u32::try_from(*b)
                    .ok()
                    .and_then(|n| a.checked_shr(n))
                    .unwrap_or(saturated);
                Ok(Some(ConstValue::Int(folded)))
            }
            BinOp::BitAnd(_) => Ok(Some(ConstValue::Int(a & b))),
            BinOp::BitOr(_) => Ok(Some(ConstValue::Int(a | b))),
            BinOp::BitXor(_) => Ok(Some(ConstValue::Int(a ^ b))),
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        // Float-Float arithmetic mirrors `operation.rs::pyfunc`'s
        // `(OpKind::{Add,Sub,Mul,TrueDiv,Mod}, [ConstValue::Float,
        // ConstValue::Float])` arms (`operation.rs:1336-1393`). Rust
        // `BinOp::Div` over `f64` IS Python's `/` (true division),
        // not floor-division — Rust has no integer-floor `/` for
        // floats. `BinOp::Rem` is Rust's `%`, which we route through
        // `float_py_mod` so the const-evaluator and the function-body
        // constfold produce bit-identical results (signed-zero
        // copysign + sign-of-denominator correction per upstream
        // `pypy/objspace/std/floatobject.py:543-563 descr_mod`).
        // Comparison ops use `partial_cmp` semantics on the underlying
        // `f64` (NaN compares as false everywhere, matching Python
        // `float('nan') < 1.0 == False`).
        (ConstValue::Float(a_bits), ConstValue::Float(b_bits)) => {
            let a = f64::from_bits(*a_bits);
            let b = f64::from_bits(*b_bits);
            match op {
                BinOp::Add(_) => Ok(Some(ConstValue::float(a + b))),
                BinOp::Sub(_) => Ok(Some(ConstValue::float(a - b))),
                BinOp::Mul(_) => Ok(Some(ConstValue::float(a * b))),
                BinOp::Div(_) => {
                    if b == 0.0 {
                        return Err(raise(
                            "ZeroDivisionError: float division by zero".to_string(),
                        ));
                    }
                    Ok(Some(ConstValue::float(a / b)))
                }
                BinOp::Rem(_) => {
                    if b == 0.0 {
                        return Err(raise("ZeroDivisionError: float modulo".to_string()));
                    }
                    Ok(Some(ConstValue::float(float_py_mod(a, b))))
                }
                // `==` / `!=` use IEEE 754 semantics on `f64`
                // (`NaN != NaN`), matching Python 3's
                // `float('nan') == float('nan') is False`.
                BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
                BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
                BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
                BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
                BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
                BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
                _ => Ok(None),
            }
        }
        (ConstValue::Bool(a), ConstValue::Bool(b)) => match op {
            // Rust `&&` / `||` are short-circuit at the source
            // level — the lowerer in `eval_const_expr` handles the
            // short-circuit semantics before we reach here. By the
            // time `eval_binop` runs both operands have been fully
            // evaluated; the operator semantic is then `bool && bool`
            // / `bool || bool` (boolean conjunction / disjunction
            // returning bool, not Python's value-returning short-
            // circuit).
            BinOp::And(_) => Ok(Some(ConstValue::Bool(*a && *b))),
            BinOp::Or(_) => Ok(Some(ConstValue::Bool(*a || *b))),
            BinOp::BitAnd(_) => Ok(Some(ConstValue::Bool(*a & *b))),
            BinOp::BitOr(_) => Ok(Some(ConstValue::Bool(*a | *b))),
            BinOp::BitXor(_) => Ok(Some(ConstValue::Bool(*a ^ *b))),
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            _ => Ok(None),
        },
        (ConstValue::UniStr(a), ConstValue::UniStr(b)) => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        (ConstValue::ByteStr(a), ConstValue::ByteStr(b)) => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        // Mixed-numeric arm: cross-type pairs over `Int` / `Bool` /
        // `Float` follow Python's numeric tower (`bool ⊂ int ⊂ float`)
        // via the same `coerce_arith` / `coerce_int_pair` helpers
        // `flowspace::operation::pyfunc` uses, so the const-evaluator
        // and the function-body constfold agree on values for
        // `Int + Float`, `Bool + Int`, `Int % Float`, `True / 0.0`,
        // etc. Mirrors upstream `PureOperation.constfold`
        // (`operation.py:120-127`) which calls `operator.<op>(*args)`
        // directly — Python's runtime coerces at the operator level.
        //
        // Reached only for cross-type pairs (same-type pairs match
        // typed arms above). Bool/Bool arithmetic is intentionally
        // NOT routed here; the typed `(Bool, Bool)` arm declines to
        // `Ok(None)` for arithmetic ops, matching the prior behavior
        // where the walker silent-skips that combination.
        (_, _) if is_foldable_numeric(lhs) && is_foldable_numeric(rhs) => match op {
            BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match (op, pair) {
                    (BinOp::Add(_), ArithOps::Int(x, y)) => match x.checked_add(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("+", x, y),
                    },
                    (BinOp::Add(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x + y))),
                    (BinOp::Sub(_), ArithOps::Int(x, y)) => match x.checked_sub(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("-", x, y),
                    },
                    (BinOp::Sub(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x - y))),
                    (BinOp::Mul(_), ArithOps::Int(x, y)) => match x.checked_mul(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("*", x, y),
                    },
                    (BinOp::Mul(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x * y))),
                    _ => unreachable!(),
                }
            }
            BinOp::Div(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match pair {
                    ArithOps::Int(x, y) => {
                        if y == 0 {
                            return int_zerodiv("division");
                        }
                        match int_py_floor_div(x, y) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("/", x, y),
                        }
                    }
                    ArithOps::Float(x, y) => {
                        if y == 0.0 {
                            return Err(raise(
                                "ZeroDivisionError: float division by zero".to_string(),
                            ));
                        }
                        Ok(Some(ConstValue::float(x / y)))
                    }
                }
            }
            BinOp::Rem(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match pair {
                    ArithOps::Int(x, y) => {
                        if y == 0 {
                            return int_zerodiv("modulo");
                        }
                        if y == -1 {
                            return Ok(Some(ConstValue::Int(0)));
                        }
                        match int_py_mod(x, y) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("%", x, y),
                        }
                    }
                    ArithOps::Float(x, y) => {
                        if y == 0.0 {
                            return Err(raise("ZeroDivisionError: float modulo".to_string()));
                        }
                        Ok(Some(ConstValue::float(float_py_mod(x, y))))
                    }
                }
            }
            // Shifts and bitwise: int-only coercion; Float operand
            // surfaces `TypeError` per Python `1 << 1.0`. `coerce_int_pair`
            // returns `None` when either side is a Float.
            BinOp::Shl(_)
            | BinOp::Shr(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::BitXor(_) => {
                let Some((x, y)) = coerce_int_pair(lhs, rhs) else {
                    return Err(raise(format!(
                        "TypeError: unsupported operand types for binop: {lhs:?} OP {rhs:?}"
                    )));
                };
                match op {
                    BinOp::Shl(_) => {
                        if y < 0 {
                            return Err(raise(format!("ValueError: negative shift count {y}")));
                        }
                        match u32::try_from(y).ok().and_then(|n| x.checked_shl(n)) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("<<", x, y),
                        }
                    }
                    BinOp::Shr(_) => {
                        if y < 0 {
                            return Err(raise(format!("ValueError: negative shift count {y}")));
                        }
                        Ok(Some(ConstValue::Int(if y >= 64 {
                            if x < 0 { -1 } else { 0 }
                        } else {
                            x >> (y as u32)
                        })))
                    }
                    BinOp::BitAnd(_) => Ok(Some(ConstValue::Int(x & y))),
                    BinOp::BitOr(_) => Ok(Some(ConstValue::Int(x | y))),
                    BinOp::BitXor(_) => Ok(Some(ConstValue::Int(x ^ y))),
                    _ => unreachable!(),
                }
            }
            // Eq / Ne / Lt / Le / Gt / Ge over numeric pairs go through
            // the precision-correct helpers (handle ints beyond the
            // f64 53-bit mantissa boundary per upstream
            // `floatobject.py:135-148`).
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(python_eq_const(lhs, rhs)))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(!python_eq_const(lhs, rhs)))),
            BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => match cmp_fold(lhs, rhs) {
                Some(o) => {
                    let v = match op {
                        BinOp::Lt(_) => o.is_lt(),
                        BinOp::Le(_) => o.is_le(),
                        BinOp::Gt(_) => o.is_gt(),
                        BinOp::Ge(_) => o.is_ge(),
                        _ => unreachable!(),
                    };
                    Ok(Some(ConstValue::Bool(v)))
                }
                None => Ok(None),
            },
            _ => Ok(None),
        },
        // Catch-all for non-numeric cross-type pairs.
        //
        // **Eq / Ne / Lt / Le / Gt / Ge** delegate to the same helpers
        // that `pyfunc` uses for function-body constfold so the
        // const-evaluator and the in-flow constfold agree on Python
        // semantics. Concretely:
        //
        //   `Int(1) == "1"`         → False  (`python_eq_const` falls
        //                                     through to derived PartialEq)
        //
        // Upstream `PureOperation.constfold` calls
        // `operator.eq(*args)` / `operator.lt(*args)` directly, so
        // Python's runtime drives the result.
        //
        // **Cross-type ordering** — `cmp_fold` returns `None` for
        // genuinely-incomparable pairs (`Int <-> UniStr` etc.).
        // Python 3 raises `TypeError("'<' not supported between
        // instances of …")` for those; surface as walker error.
        //
        // **All other ops** (`+`, `-`, `*`, `/`, etc.) over distinct
        // types: `TypeError` per upstream `operator.add` etc.
        _ => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(python_eq_const(lhs, rhs)))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(!python_eq_const(lhs, rhs)))),
            BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => match cmp_fold(lhs, rhs) {
                Some(o) => {
                    let v = match op {
                        BinOp::Lt(_) => o.is_lt(),
                        BinOp::Le(_) => o.is_le(),
                        BinOp::Gt(_) => o.is_gt(),
                        BinOp::Ge(_) => o.is_ge(),
                        _ => unreachable!(),
                    };
                    Ok(Some(ConstValue::Bool(v)))
                }
                None => Err(raise(format!(
                    "TypeError: ordering comparison not supported between {lhs:?} and {rhs:?}"
                ))),
            },
            _ => Err(raise(format!(
                "TypeError: unsupported operand types for binop: {lhs:?} OP {rhs:?}"
            ))),
        },
    }
}

/// Build the `HostObject::Class` corresponding to `item_struct`.
/// The class dict is left empty — Rust struct fields are accessed on
/// *instances*, not the class object, so `Foo.x` is a meaningful
/// expression only when `Foo` is a value (e.g. an enum variant
/// constructor with named fields like `Foo::Variant { x }`).
///
/// `pyre`'s match-arm cascade (`build_flow.rs::lower_match_variant_cascade`)
/// uses the class identity for `isinstance(scrutinee, Foo)` at the
/// fork; named-field bindings then emit `getattr(scrutinee, "x")` on
/// the *instance* (not the class object) — the empty class dict
/// matches that semantic exactly.
///
/// Side effect: each `syn::Fields::Named` entry is also written into
/// [`crate::annotator::classdesc::FORCE_ATTRIBUTES_INTO_CLASSES`] via
/// [`crate::annotator::classdesc::register_struct_fields`] so
/// `ClassDesc::_init_classdef` can pre-populate `ClassDef.attrs` with
/// the field's typed `Attribute` shell.  Same dict as the hand-coded
/// EnvironmentError entry (classdesc.py:957-961); the walker is just
/// an additional writer to the same store.
fn build_host_class_from_struct(item_struct: &ItemStruct, module_prefix: &str) -> HostObject {
    let name = item_struct.ident.to_string();
    let qualified_key = if module_prefix.is_empty() {
        name.clone()
    } else {
        format!("{module_prefix}::{name}")
    };
    // Project the declared named fields into the
    // `FORCE_ATTRIBUTES_INTO_CLASSES` dict.  Same store consumed by
    // `ClassDesc::_init_classdef` for the hand-coded EnvironmentError
    // override.
    //
    // Keyed under the caller-supplied module prefix so two structs
    // with the same bare ident declared in different inline mods
    // occupy distinct registry slots, mirroring PyPy `ClassDesc`
    // per-class identity (`classdesc.py:672`).
    if let syn::Fields::Named(named) = &item_struct.fields {
        let mut stubs: Vec<(String, crate::model::ValueType)> = Vec::new();
        for field in &named.named {
            let Some(ident) = field.ident.as_ref() else {
                continue;
            };
            let field_name = ident.to_string();
            let ty = crate::front::ast::classify_fn_arg_ty(&field.ty);
            stubs.push((field_name, ty));
        }
        if !stubs.is_empty() {
            crate::annotator::classdesc::register_struct_fields(&qualified_key, &stubs);
        }
    }
    let host = HostObject::new_class(&name, vec![]);
    if let Some(ptr) = try_build_gc_struct_ptr(&name, &item_struct.fields) {
        // Same-scope struct registry — later structs in the same scope
        // can embed this one by-value via `syn_primitive_to_lltype` →
        // `walker_struct_ptr_lookup`. Mirrors upstream
        // `lltype.GcStruct("Outer", ("first", INNER_GCSTRUCT))` shape.
        walker_struct_ptr_register(&name, ptr.clone());
        // `set_lltype_ptr` rejects double-set; structurally equal
        // re-walks are harmless because the previously-stored `Ptr`
        // is structurally equal. Ignore the duplicate-set error to
        // match `lltype.Ptr.__new__`'s "return the cached instance"
        // behaviour (`rpython/rtyper/lltypesystem/lltype.py:721-739`).
        let _ = host.set_lltype_ptr(ptr);
    }
    host
}

thread_local! {
    /// Per-walker-scope `type T = U` map consulted by
    /// [`syn_primitive_to_lltype`] when an unrecognized single-segment
    /// identifier needs alias resolution. Set by the caller via
    /// [`with_walker_type_aliases`] before invoking
    /// [`build_host_class_from_struct`] / [`try_build_gc_struct_ptr`],
    /// and restored to the prior value when the guard drops.  Mirrors
    /// upstream Rust's compile-time `Item::Type` resolution at the
    /// catalog layer: `type PyObjectRef = *mut PyObject;` lets a
    /// struct field declared as `PyObjectRef` resolve through the
    /// catalog the same way `*mut PyObject` already does.
    static WALKER_TYPE_ALIASES: RefCell<indexmap::IndexMap<String, syn::Type>> =
        RefCell::new(indexmap::IndexMap::new());
}

/// RAII guard that extends the walker's per-thread type-alias map
/// with the entering scope's aliases and restores the prior contents
/// on drop.  Nested scopes compose by merging into the parent's
/// alias map: inline-mod entries either shadow a same-name parent
/// alias (`IndexMap::insert` overwrites) or add new entries, and
/// drop restores the parent scope verbatim — outer-scope aliases
/// remain visible inside the inline mod, matching RPython's whole-
/// program alias visibility (`flowcontext.py:847` resolves names
/// through `func_globals` which spans the imported namespace).
///
/// Cross-file aliases are seeded by the top-level pipeline entry
/// (`analyze_pipeline_from_parsed`) which constructs the union of
/// every parsed file's top-level `type T = U;` declarations and
/// installs them via [`Self::enter`] before any per-file walker
/// runs.
struct WalkerTypeAliasGuard {
    prev: indexmap::IndexMap<String, syn::Type>,
}

impl WalkerTypeAliasGuard {
    fn enter(aliases: indexmap::IndexMap<String, syn::Type>) -> Self {
        let prev = WALKER_TYPE_ALIASES.with(|cell| {
            let prev = cell.borrow().clone();
            let mut current = cell.borrow_mut();
            for (name, ty) in aliases {
                current.insert(name, ty);
            }
            prev
        });
        WalkerTypeAliasGuard { prev }
    }
}

impl Drop for WalkerTypeAliasGuard {
    fn drop(&mut self) {
        WALKER_TYPE_ALIASES.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.prev);
        });
    }
}

/// Consult the walker scope's alias map for a single-segment ident.
/// Returns `None` when no alias matches.  Outer-scope aliases stay
/// addressable inside inline-mod entries because
/// [`WalkerTypeAliasGuard::enter`] merges (rather than replaces) the
/// parent scope.
fn walker_type_alias_lookup(name: &str) -> Option<syn::Type> {
    WALKER_TYPE_ALIASES.with(|cell| cell.borrow().get(name).cloned())
}

/// Cross-file alias floor — RAII handle that seeds the walker's
/// alias map with every parsed file's top-level `type T = U;`
/// declarations before per-file walks run.  Mirrors PyPy
/// `Bookkeeper`'s whole-program import-resolution map: each module's
/// type aliases stay visible to every other module's walker pass,
/// regardless of source-file iteration order.
///
/// Construct one at the top of `analyze_pipeline_from_parsed` (or
/// any caller that walks multiple parsed files in sequence) and bind
/// it as `let _floor = ...;` so it lives across the per-file calls;
/// drop restores the prior thread-local content.
pub struct WalkerAliasFloorGuard {
    _inner: WalkerTypeAliasGuard,
}

impl WalkerAliasFloorGuard {
    /// Build the floor from every parsed file's top-level
    /// `Item::Type` declarations and install it on the per-thread
    /// walker alias map.
    pub fn install<'a>(files: impl IntoIterator<Item = &'a syn::File>) -> Self {
        let mut floor: indexmap::IndexMap<String, syn::Type> = indexmap::IndexMap::new();
        for file in files {
            for (name, ty) in collect_type_aliases(&file.items) {
                floor.entry(name).or_insert(ty);
            }
        }
        Self {
            _inner: WalkerTypeAliasGuard::enter(floor),
        }
    }
}

thread_local! {
    /// Per-walker-scope `Item::Struct` → `Ptr(GcStruct(...))` map
    /// populated by [`build_host_class_from_struct`] after each
    /// in-scope struct is minted. Consulted by
    /// [`syn_primitive_to_lltype`] when an unrecognized single-segment
    /// identifier names a struct that has already been walked in the
    /// same scope, enabling by-value embedding of one cataloged struct
    /// inside another — mirroring upstream
    /// `lltype.GcStruct("Outer", ("first", INNER_GCSTRUCT))` (the
    /// canonical "subclass marker" / `PyObject_HEAD` inheritance shape;
    /// `lltype.py:296-303 Struct._first_struct`).
    ///
    /// Cross-file struct ptrs are seeded by
    /// [`WalkerStructPtrsFloorGuard::install`] at the top of
    /// `analyze_pipeline_from_parsed`, so a struct declared in one
    /// file resolves through this map when a sibling file embeds it
    /// by bare ident.  Per-file / per-mod walker guards merge into
    /// the same map (clone-on-enter, restore-on-drop) so the floor
    /// stays visible across the whole walker pass.
    static WALKER_STRUCT_PTRS: RefCell<StdHashMap<String, Ptr>> =
        RefCell::new(StdHashMap::new());
}

/// RAII guard that snapshots the walker's thread-local struct-Ptr map
/// on construction and restores the snapshot on drop. Mirrors
/// [`WalkerTypeAliasGuard`]; nested guards compose by merging into
/// the parent scope's map (clone-on-enter) and reverting on drop.
struct WalkerStructPtrsGuard {
    prev: StdHashMap<String, Ptr>,
}

impl WalkerStructPtrsGuard {
    fn enter() -> Self {
        let prev = WALKER_STRUCT_PTRS.with(|cell| cell.borrow().clone());
        WalkerStructPtrsGuard { prev }
    }
}

impl Drop for WalkerStructPtrsGuard {
    fn drop(&mut self) {
        WALKER_STRUCT_PTRS.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.prev);
        });
    }
}

/// Install-only RAII guard that pre-mints `Ptr(GcStruct(...))` entries
/// for every top-level `Item::Struct` across the supplied parsed files,
/// then keeps them visible on the per-thread walker map for the
/// duration of the analyze pipeline.
///
/// Mirrors upstream `Bookkeeper`'s whole-program type cache (per-
/// translation single dictionary in `pypy/annotation/bookkeeper.py:67
/// self.descs = {}`) — a struct field declared as `pub ob_header:
/// PyObject` in `bytesobject.rs` resolves through the `PyObject`
/// minted from `pyobject.rs` regardless of walker iteration order.
///
/// Construct one at the top of `analyze_pipeline_from_parsed` (or
/// any caller that walks multiple parsed files in sequence) and bind
/// it as `let _floor = ...;` so it lives across the per-file calls.
pub struct WalkerStructPtrsFloorGuard {
    _inner: WalkerStructPtrsGuard,
}

impl WalkerStructPtrsFloorGuard {
    /// Build the floor by running the same fixed-point minting
    /// [`preseed_struct_ptrs`] uses, but over the union of every
    /// parsed file's top-level `Item::Struct`s. Cross-file dependencies
    /// (struct B in file 2 embeds struct A from file 1) resolve through
    /// the same iteration: A mints on iteration 1, B mints on
    /// iteration 2 once A is registered.
    pub fn install<'a>(files: impl IntoIterator<Item = &'a syn::File>) -> Self {
        let inner = WalkerStructPtrsGuard::enter();
        let structs: Vec<&'a syn::ItemStruct> = files
            .into_iter()
            .flat_map(|file| file.items.iter())
            .filter_map(|item| match item {
                syn::Item::Struct(s) => Some(s),
                _ => None,
            })
            .collect();
        let mut registered = std::collections::HashSet::new();
        loop {
            let mut progressed = false;
            for item_struct in &structs {
                let name = item_struct.ident.to_string();
                if registered.contains(&name) {
                    continue;
                }
                if let Some(ptr) = try_build_gc_struct_ptr(&name, &item_struct.fields) {
                    walker_struct_ptr_register(&name, ptr);
                    registered.insert(name);
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        Self { _inner: inner }
    }
}

/// Register a freshly-minted `Ptr(GcStruct(...))` into the walker's
/// struct-Ptr map under its bare name so that subsequent struct walks
/// can embed it by-value via [`syn_primitive_to_lltype`]. Silently
/// overwrites a previous entry under the same name — mirrors upstream
/// `lltype.Ptr.__new__`'s structural cache (`lltype.py:721-739`)
/// where structurally equal `Ptr`s share identity.
fn walker_struct_ptr_register(name: &str, ptr: Ptr) {
    WALKER_STRUCT_PTRS.with(|cell| {
        cell.borrow_mut().insert(name.to_string(), ptr);
    });
}

/// Consult the walker scope's struct-Ptr map for a single-segment
/// ident.  Returns the inner `StructType` (`Ptr.TO` unwrapped) when
/// the name matches a previously-minted `Ptr(GcStruct(...))`.
///
/// Cross-file embedding (e.g. `W_BytesObject { pub ob_header:
/// PyObject }` in `bytesobject.rs` referencing the `PyObject` minted
/// by `pyobject.rs`) resolves through the floor pre-mint installed
/// by [`WalkerStructPtrsFloorGuard::install`] at the top of the
/// analyze pipeline.
fn walker_struct_ptr_lookup(name: &str) -> Option<StructType> {
    let ptr = WALKER_STRUCT_PTRS.with(|cell| cell.borrow().get(name).cloned())?;
    match ptr.TO {
        crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => Some(s),
        _ => None,
    }
}

/// Pre-collect `Item::Type T = U;` definitions from a flat list of
/// items into an [`indexmap::IndexMap`]. Skips items with generic
/// parameters (`type T<U> = Vec<U>;`) — those would require
/// substitution semantics the catalog does not implement yet.
///
/// Insertion-order preservation is load-bearing: downstream
/// [`WalkerTypeAliasGuard::enter`] mirrors each alias into the
/// process-wide registry via `first-writer-wins` semantics.  A
/// non-deterministic hash-table iteration order would make the first
/// writer platform-dependent (random hasher seeds on Linux vs macOS
/// vs Windows), turning what looks like a stable registry into
/// platform-divergent analyzer state.  Source-order keeps the floor
/// reproducible across hosts and across cargo-test runs.
fn collect_type_aliases<'a>(
    items: impl IntoIterator<Item = &'a syn::Item>,
) -> indexmap::IndexMap<String, syn::Type> {
    let mut map = indexmap::IndexMap::new();
    for item in items {
        if let syn::Item::Type(item_type) = item
            && item_type.generics.params.is_empty()
        {
            map.insert(item_type.ident.to_string(), (*item_type.ty).clone());
        }
    }
    map
}

/// Populate the current walker scope's struct-Ptr catalog before the
/// source-order item pass runs. This is a fixed-point pass over only
/// the current scope's `Item::Struct`s: primitive-only structs register
/// first, then structs whose first field embeds one of those registered
/// GcStructs can register on a later iteration. That removes Rust
/// forward-reference sensitivity without introducing a side table at
/// use sites — the normal [`walker_struct_ptr_lookup`] path still reads
/// the same per-scope catalog.
fn preseed_struct_ptrs<'a>(items: impl IntoIterator<Item = &'a syn::Item>) {
    let structs: Vec<&'a syn::ItemStruct> = items
        .into_iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item_struct) => Some(item_struct),
            _ => None,
        })
        .collect();
    let mut registered = std::collections::HashSet::new();
    loop {
        let mut progressed = false;
        for item_struct in &structs {
            let name = item_struct.ident.to_string();
            if registered.contains(&name) {
                continue;
            }
            if let Some(ptr) = try_build_gc_struct_ptr(&name, &item_struct.fields) {
                walker_struct_ptr_register(&name, ptr);
                registered.insert(name);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
}

/// Attempt to synthesize a `Ptr(GcStruct(name, fields))` for an
/// `Item::Struct` whose every field has a direct lltype shape. Returns
/// `None` if any field's `syn::Type` falls outside the strict catalog
/// covered by `syn_primitive_to_lltype` (Box/Rc/Arc/Vec/Atomic/dyn
/// trait/generic — those require explicit RPython-shape modeling before
/// they can be admitted).
///
/// Mirrors upstream `rpython/rtyper/lltypesystem/lltype.py:258-380
/// class Struct.__init__` field-tuple constructor + `:721-739
/// Ptr.__new__` validation. The resulting `Ptr` is what `_ptrEntry`
/// (`lltype.py:1513-1518`) returns from `compute_annotation` when the
/// `typeOf(self.instance)` of a typed-ref instance is evaluated.
fn try_build_gc_struct_ptr(name: &str, fields: &syn::Fields) -> Option<Ptr> {
    let field_iter: Box<dyn Iterator<Item = &syn::Field>> = match fields {
        syn::Fields::Named(named) => Box::new(named.named.iter()),
        syn::Fields::Unnamed(unnamed) => Box::new(unnamed.unnamed.iter()),
        // `struct Foo;` — unit struct. Upstream `lltype.Struct(name)`
        // with no fields is legal (`lltype.py:258` accepts `*fields`);
        // produce an empty-field GcStruct so unit structs reach the
        // catalog symmetrically with field-bearing ones.
        syn::Fields::Unit => {
            let struct_t = StructType::gc(name, vec![]);
            return Ptr::from_container_type(LowLevelType::Struct(Box::new(struct_t))).ok();
        }
    };
    let mut resolved: Vec<(String, LowLevelType)> = Vec::new();
    for (idx, field) in field_iter.enumerate() {
        let ll = syn_primitive_to_lltype(&field.ty)?;
        // RPython `Struct._build` (`lltype.py:289-291`) rejects
        // embedding a `GcStruct` as anything other than the first
        // field of another `GcStruct` (the inheritance / subclass
        // marker shape, e.g. `PyObject_HEAD`).  Pre-validate here so
        // a non-first nested `GcStruct` rejects the parent gracefully
        // instead of triggering the downstream `_note_inlined_into`
        // panic.  Mirrors `StructType._note_inlined_into`'s
        // `!first || !same_gc` arm.
        if idx > 0
            && let LowLevelType::Struct(inner) = &ll
            && inner._gckind == GcKind::Gc
        {
            return None;
        }
        let field_name = field
            .ident
            .as_ref()
            .map(|i| i.to_string())
            .unwrap_or_else(|| idx.to_string());
        resolved.push((field_name, ll));
    }
    let struct_t = StructType::gc(name, resolved);
    Ptr::from_container_type(LowLevelType::Struct(Box::new(struct_t))).ok()
}

/// Map a `syn::Type` to a `LowLevelType` for the strict field-shape
/// subset recognized by the struct catalog.
/// Returns `None` for any shape that still needs catalog extension
/// (`Vec<T>`, `Box<T>`, `Rc<T>`, `Arc<T>`, `Atomic*`, `dyn Trait`,
/// generic type parameter, tuple, array, slice, fn type, …). Raw
/// pointer fields and `&RegisteredStruct` are admitted because they
/// have direct `Ptr(OpaqueType)` / `Ptr(GcStruct)` lltype counterparts.
///
/// The mapping preserves RPython `lltype.py` primitive identity
/// (`Char`, `UniChar`, `SingleFloat`, `Float`, …) as it appears in
/// `rpython/rtyper/lltypesystem/lltype.py:88-198`. This is the
/// *lltype-level* identity — it lives one layer below the
/// register-class collapse that `getkind` performs
/// (`rpython/jit/codewriter/support.py:getkind`), where `Char` /
/// `UniChar` / `SingleFloat` all fold to `'int'`. The
/// register-class lift in `front/ast.rs::classify_fn_arg_ty` works
/// at the `ValueType` (= register class) layer; that layer agrees
/// with `getkind`, not with the lltype-primitive identity captured
/// here. Mapping `char → Signed` or `f32 → Float` here would erase
/// the lltype identity even though it would coincidentally pick the
/// right register class.
///
/// Extract a single-segment leaf identifier from a raw pointer's
/// target type for use as an `OpaqueType` tag. Returns `None` if
/// the target is anything other than a single-segment path or a
/// `()` unit-type — the caller falls back to the conventional
/// `"?"` tag in that case.
fn raw_pointer_target_tag(target: &syn::Type) -> Option<String> {
    match target {
        syn::Type::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
            let seg = p.path.segments.first()?;
            if !matches!(seg.arguments, syn::PathArguments::None) {
                return None;
            }
            Some(seg.ident.to_string())
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("Void".to_string()),
        _ => None,
    }
}

fn syn_primitive_to_lltype(ty: &syn::Type) -> Option<LowLevelType> {
    // `*const T` / `*mut T` — raw pointer to (typically opaque-to-the
    // -catalog) target. Lift to `Ptr(OpaqueType(T, gckind=Raw))` so
    // structs with raw-pointer fields register in the catalog
    // without requiring the referent type to be known. Mirrors
    // RPython `lltype.Ptr(OpaqueType("T"))` usage for externally
    // declared opaque pointer targets (`lltype.py:564-572 class
    // OpaqueType`). The opaque tag preserves the referent's leaf
    // identifier so two distinct raw-pointer fields point to
    // identity-distinct opaques (`*mut PyExecutionContext` is not
    // equal to `*mut FrameDebugData`).
    if let syn::Type::Ptr(ptr) = ty {
        let tag = raw_pointer_target_tag(&ptr.elem).unwrap_or_else(|| "?".to_string());
        let opaque = LowLevelType::Opaque(Box::new(OpaqueType::new(&tag)));
        let raw_ptr = Ptr::from_container_type(opaque).ok()?;
        return Some(LowLevelType::Ptr(Box::new(raw_ptr)));
    }
    // `&T` borrowed reference. RPython convention models gc-tracked
    // references as `lltype.Ptr(GcStruct(T))` — the same shape used
    // by `_ptrEntry.compute_annotation` (`lltype.py:1513-1518`) when
    // a Python-level instance is annotated. The catalog admits `&T`
    // only when `T` is a single-segment identifier resolving to a
    // walker-registered struct (visible through `WALKER_STRUCT_PTRS`
    // either from the current walker scope or the cross-file floor
    // installed at pipeline entry).  `&str`, `&dyn
    // Trait`, `&[T]` etc. reject — `rstr.STR` and trait-object
    // dispatch are not yet ported.  The borrow lifetime annotation
    // is dropped: lltype has no lifetime concept; the gc-pointer
    // semantics carry over regardless of the source-level lifetime.
    if let syn::Type::Reference(r) = ty {
        let syn::Type::Path(p) = &*r.elem else {
            return None;
        };
        if p.qself.is_some() || p.path.segments.len() != 1 {
            return None;
        }
        let inner_seg = p.path.segments.first()?;
        if !matches!(inner_seg.arguments, syn::PathArguments::None) {
            return None;
        }
        let inner_name = inner_seg.ident.to_string();
        let inner_struct = walker_struct_ptr_lookup(&inner_name)?;
        if inner_struct._gckind != GcKind::Gc {
            return None;
        }
        let ptr = Ptr::from_container_type(LowLevelType::Struct(Box::new(inner_struct))).ok()?;
        return Some(LowLevelType::Ptr(Box::new(ptr)));
    }
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    // Multi-segment paths reject here. Cross-module idents and Rust
    // stdlib wrappers need explicit source-level modeling; collapsing
    // them by leaf name would invent lltype shapes that RPython did not
    // declare.
    let seg = path.path.segments.last()?;
    if path.path.segments.len() > 1 {
        return None;
    }
    let name_outer = seg.ident.to_string();
    // `Option<T>` — nullable pointer mirroring `lltype.nullptr(...)`
    // semantics. Accept only when `T` itself resolves directly to
    // `LowLevelType::Ptr(_)` without Rust wrapper/Vec collapse. This
    // admits pointer-like Rust spellings such as `Option<*mut T>` and
    // `Option<&RegisteredStruct>`, while rejecting `Option<Vec<T>>` and
    // `Option<Box<T>>` whose storage shape is not a single lltype ptr.
    if name_outer == "Option" {
        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
            for arg in &args.args {
                if let syn::GenericArgument::Type(inner) = arg {
                    let inner_ll = syn_primitive_to_lltype(inner)?;
                    if matches!(inner_ll, LowLevelType::Ptr(_)) {
                        return Some(inner_ll);
                    }
                    return None;
                }
            }
        }
        return None;
    }
    if !matches!(seg.arguments, syn::PathArguments::None) {
        return None;
    }
    let name = seg.ident.to_string();
    // Walker-scope `type T = U` alias resolution.  Only consult the
    // alias map when the leaf identifier does not collide with a
    // primitive keyword below (`i32` etc. shouldn't shadow), and
    // recurse on the aliased type so chains like
    // `type Outer = Inner; type Inner = *mut Foo;` collapse.
    if !matches!(
        name.as_str(),
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "char"
            | "f32"
            | "f64"
    ) {
        if let Some(alias_target) = walker_type_alias_lookup(&name) {
            return syn_primitive_to_lltype(&alias_target);
        }
        // Same-scope `Item::Struct` previously minted as
        // `Ptr(GcStruct(name, ...))` — return the unwrapped
        // `StructType` so the parent embedding lifts as
        // `lltype.GcStruct("Outer", ("first", INNER_GCSTRUCT))`
        // (the canonical inheritance / `PyObject_HEAD` shape;
        // `lltype.py:296-303 Struct._first_struct`).
        // `try_build_gc_struct_ptr` enforces the upstream
        // `_note_inlined_into` rule (gc field must be at index 0).
        if let Some(inner_struct) = walker_struct_ptr_lookup(&name) {
            return Some(LowLevelType::Struct(Box::new(inner_struct)));
        }
    }
    let ll = match name.as_str() {
        // `lltype.Signed` family — native int size on this platform.
        "i8" | "i16" | "i32" | "i64" | "isize" => LowLevelType::Signed,
        // `lltype.Unsigned` family — `getkind(Unsigned) == 'int'`
        // collapses to the int register class at the producer side
        // (`flatten.py:getkind`); the lltype primitive stays
        // `Unsigned` so the annotator selects
        // `SomeInteger(unsigned=True)`.
        "u8" | "u16" | "u32" | "u64" | "usize" => LowLevelType::Unsigned,
        "bool" => LowLevelType::Bool,
        // `lltype.UniChar` — Rust `char` is a Unicode scalar value,
        // matching RPython's `UniChar` primitive
        // (`lltype.py:UniChar = Primitive("UniChar", "\x00")`).
        // `getkind(UniChar) == 'int'` folds the register class to
        // int downstream, but lltype identity must stay `UniChar`
        // so the rtyper picks `UniCharRepr`
        // (`rpython/rtyper/lltypesystem/rstr.py`).
        "char" => LowLevelType::UniChar,
        // `lltype.SingleFloat` — Rust `f32`. `getkind(SingleFloat)
        // == 'int'` (the value is boxed through an int register
        // word, see `rpython/jit/codewriter/support.py:getkind`);
        // lltype identity stays distinct from `Float` so
        // `cast_singlefloat_to_float` / `cast_float_to_singlefloat`
        // dispatch correctly.
        "f32" => LowLevelType::SingleFloat,
        // `lltype.Float` — Rust `f64`, the canonical float register
        // class (`getkind(Float) == 'float'`).
        "f64" => LowLevelType::Float,
        _ => return None,
    };
    Some(ll)
}

// ---------------------------------------------------------------------------
// Rust struct field projection — writes into
// `crate::annotator::classdesc::FORCE_ATTRIBUTES_INTO_CLASSES`.
// ---------------------------------------------------------------------------

/// Pre-pass that mirrors [`build_host_class_from_struct`]'s field
/// projection but runs on a parsed `syn::File` without driving the
/// full walker.  The production pipeline (`analyze_pipeline_from_parsed`)
/// does not call [`register_rust_module_at_with_source`], which means
/// [`build_host_class_from_struct`] never fires there and the
/// `FORCE_ATTRIBUTES_INTO_CLASSES` dict stays unpopulated for parsed-
/// only structs.  Without these entries `ClassDesc::_init_classdef`
/// cannot pre-fill `ClassDef.attrs`, so the receiver-narrowing gate
/// at `flowspace_adapter.rs::derive_subject_inputcells` skips and
/// every `impl`-method `self` carries `SomeInstance(classdef=None)`.
///
/// `prefix` matches the walker-module-prefix shape: empty for
/// top-level structs (registered under the bare ident), non-empty
/// for structs declared inside an inline `mod` block (registered
/// under `{prefix}::{ident}`).
pub fn pre_register_struct_fields_from_file(file: &syn::File, prefix: &str) {
    pre_register_struct_fields_from_items(&file.items, prefix);
}

fn pre_register_struct_fields_from_items(items: &[syn::Item], prefix: &str) {
    for item in items {
        match item {
            Item::Struct(item_struct) => {
                if let syn::Fields::Named(named) = &item_struct.fields {
                    let mut stubs: Vec<(String, crate::model::ValueType)> = Vec::new();
                    for field in &named.named {
                        let Some(ident) = field.ident.as_ref() else {
                            continue;
                        };
                        let field_name = ident.to_string();
                        let ty = crate::front::ast::classify_fn_arg_ty(&field.ty);
                        stubs.push((field_name, ty));
                    }
                    if !stubs.is_empty() {
                        let bare_name = item_struct.ident.to_string();
                        let qualified_key = if prefix.is_empty() {
                            bare_name
                        } else {
                            format!("{prefix}::{bare_name}")
                        };
                        crate::annotator::classdesc::register_struct_fields(&qualified_key, &stubs);
                    }
                }
            }
            Item::Mod(item_mod) => {
                let Some((_, inner)) = &item_mod.content else {
                    continue;
                };
                let inner_name = item_mod.ident.to_string();
                let inner_prefix = if prefix.is_empty() {
                    inner_name
                } else {
                    format!("{prefix}::{inner_name}")
                };
                pre_register_struct_fields_from_items(inner, &inner_prefix);
            }
            _ => continue,
        }
    }
}

/// Test-only accessor for the per-`ModuleId` slice of the
/// module-globals registry. Used by `interactive.rs::tests` to
/// verify that the file-aware entry's walker pre-pass registered
/// sibling items before the entry-point body lowered. Re-exports
/// `module_globals_lookup` under a `pub(crate)` name so cross-
/// module tests can read the registry without exposing the
/// `pub(super)` API surface.
#[allow(dead_code)]
pub(crate) fn module_globals_for_test(module_id: ModuleId, name: &str) -> Option<ConstValue> {
    module_globals_lookup(module_id, name)
}

/// Inner builder shared by [`build_host_function_from_rust`] (full
/// body-lowering path) and
/// [`build_host_function_metadata_from_rust`] (import-time-only
/// path). Returns the `HostObject` plus the underlying `HostCode` /
/// `GraphFunc` so the full path can wire `graph.func` after running
/// `build_flow_from_rust`.
struct HostMetadataParts {
    host: HostObject,
    host_code: HostCode,
    gf: GraphFunc,
}

fn build_host_metadata_parts(
    item_fn: &ItemFn,
    module_id: ModuleId,
    source_filename: Option<&str>,
    source_text: Option<&str>,
    class_: Option<HostObject>,
) -> Result<HostMetadataParts, AdapterError> {
    let argnames = extract_argnames(item_fn)?;
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

    // TODO: upstream `model.py:54 FunctionGraph.filename`
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
    // upstream `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
    // wraps the function's owning module dict — every entry the
    // module bound at import time is visible. Mirror that by
    // snapshotting `module_id`'s slice of the registry. When the
    // walker registered no entries (anonymous fixture / metadata-
    // only entry), the snapshot is an empty dict, matching upstream
    // `func.__globals__ == {}` for a function defined with no
    // enclosing module bindings.
    let func_globals = Constant::new(ConstValue::Dict(module_globals_snapshot(module_id)));
    let mut gf = GraphFunc::from_host_code(host_code.clone(), func_globals, Vec::new());
    // Issue 2.1 (2026-05-05): record `module_id` on `GraphFunc` so
    // `func.__globals__` introspection paths
    // (`HostObject::class_get("__globals__")` /
    // `getattr(func, "__globals__")`) re-snapshot the registry at
    // access time — mirrors upstream `flowcontext.py:284
    // self.w_globals = Constant(func.__globals__)` where
    // `func.__globals__` is a *live* reference to the module
    // dict. The static snapshot stored in `gf.globals` above
    // remains as the canonical `module = __name__`-extraction
    // source per `from_host_code` and as the static fallback
    // for non-rust-source callers.
    gf.module_globals_id = Some(module_id);
    // upstream `bytecode.py:46-60` populates `GraphFunc.source` from
    // `inspect.getsource(func)`. When the caller threads in the
    // source text, mirror that — downstream readers (`model.rs:3210
    // FunctionGraph::source`, `tool/error.rs:300-320`) walk
    // `func.source` as a fallback when `graph._source` is unset, so
    // one assignment covers both paths.
    if let Some(src) = source_text {
        gf.source = Some(src.to_owned());
    }
    // Audit 1.5 (2026-05-08): when the caller is the impl-method
    // deferred path, thread the resolved target class through to
    // `GraphFunc.class_` so downstream consumers see method-owned
    // identity. Mirrors upstream Python `Foo.bar.__self_class__`
    // / `func.class_` set at `class Foo: def bar(self): ...`
    // creation time. `None` for non-impl paths (regular fns,
    // entry-point fns, inner-mod siblings) keeps the legacy
    // shape — class-less fns have no owner and upstream's
    // `func.class_` defaults to absent.
    gf.class_ = class_;
    let host = HostObject::new_user_function(gf.clone());
    Ok(HostMetadataParts {
        host,
        host_code,
        gf,
    })
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
    extract_argnames_from_sig(&item_fn.sig)
}

fn extract_argnames_from_sig(sig: &syn::Signature) -> Result<Vec<String>, AdapterError> {
    let mut out = Vec::new();
    for input in &sig.inputs {
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

/// Z2.5 Path C slice 1 — walk a parsed `syn::File` to collect
/// `(path-segments, Signature)` pairs for every top-level `unsafe fn`
/// and unsafe impl-method.  These callees cannot lower their bodies via
/// the adapter (`build_flow.rs:215` rejects `sig.unsafety.is_some()`
/// because the bodies access raw pointers that the flowspace adapter
/// does not model), but downstream `OpKind::Call::FunctionPath` sites
/// still need their signature registered in `PyreCallRegistry` —
/// otherwise the dual gate Skips with "not registered in
/// PyreCallRegistry" at `flowspace_adapter.rs:1059`.
///
/// `prefix` is the module-path stem the caller derived from the source
/// filename (`pyobject` for `pyre/pyre-object/src/pyobject.rs`); each
/// free-fn key becomes `[prefix, ident]` (or `[ident]` when prefix is
/// empty, matching the bare-name registration path in
/// `front/ast.rs::collect_fn_returns`).  Impl-methods key as
/// `[prefix-segments..., ImplTy-segments..., method]` — the file's
/// module prefix is prepended only when the impl target was written as
/// a bare ident (`impl PyFrame { ... }` in `pyframe.rs` → `["pyframe",
/// "PyFrame", method]`).  When the impl target carries its own
/// qualifier (`impl pyframe::PyFrame { ... }` from a different file)
/// the syntactic spelling is used verbatim.  This matches the
/// module-qualified `CallPath::for_impl_method(impl_type_root, name)`
/// shape that `lib.rs::register_function_graph` registers (`impl_type`
/// is derived from `method_info.self_ty_root`, the qualified path).
///
/// Pure-extract — no registration, no side effects.  The slice 3
/// consumer (`cutover.rs::populate_call_registry_from_call_graphs`)
/// drives the registration once the slice 2 stub PyGraph builder is in
/// place; until then, calling this helper is a no-op observable only
/// through the return Vec.  Returns an empty Vec when `file` contains
/// no unsafe fns / methods.
pub fn extract_unsafe_fn_signatures(
    file: &File,
    prefix: &str,
) -> Vec<(Vec<String>, crate::flowspace::argument::Signature)> {
    use crate::flowspace::argument::Signature;
    let mut out = Vec::new();
    for item in &file.items {
        match item {
            Item::Fn(func) if func.sig.unsafety.is_some() => {
                let Ok(argnames) = extract_argnames(func) else {
                    continue;
                };
                let mut segments = Vec::with_capacity(2);
                if !prefix.is_empty() {
                    segments.push(prefix.to_string());
                }
                segments.push(func.sig.ident.to_string());
                out.push((segments, Signature::new(argnames, None, None)));
            }
            Item::Impl(impl_block) => {
                let Some(target_path) = extract_impl_target_path(&impl_block.self_ty) else {
                    continue;
                };
                // Module-qualify the impl-method key so it matches the
                // runtime `register_function_graph` shape
                // (`["pyframe", "PyFrame", method]`) rather than the
                // 2-segment leaf-only shape (`["PyFrame", method]`).
                // When the impl target was written bare (`impl PyFrame
                // {...}`), prepend the file's module prefix; when the
                // impl target was written with its own qualifier (`impl
                // pyframe::PyFrame {...}`), use that path verbatim —
                // the syntactic spelling already disambiguates.
                let impl_ty_segments: Vec<String> = if target_path.len() > 1 {
                    target_path.clone()
                } else if prefix.is_empty() {
                    target_path.clone()
                } else {
                    let mut segs: Vec<String> = prefix.split("::").map(String::from).collect();
                    segs.extend(target_path.iter().cloned());
                    segs
                };
                for sub in &impl_block.items {
                    let syn::ImplItem::Fn(method) = sub else {
                        continue;
                    };
                    if method.sig.unsafety.is_none() {
                        continue;
                    }
                    let Ok(argnames) = extract_argnames_from_sig(&method.sig) else {
                        continue;
                    };
                    let mut path = impl_ty_segments.clone();
                    path.push(method.sig.ident.to_string());
                    out.push((path, Signature::new(argnames, None, None)));
                }
            }
            _ => {}
        }
    }
    out
}

/// Walk a parsed `syn::File` for `pub static X: T = ...` items and
/// project each to `[prefix?, ident]` segments + `ValueType` from `T`.
///
/// Cat 2.1 Slice 1 (Z2.5 SHOUTY_CASE cluster): the lazy install
/// fallback at `front/ast.rs::Expr::Path` emits a body-`OpKind::Input`
/// for any single-segment name not bound in `ctx.local_value_ids` —
/// crate-local statics (e.g. `INT_TYPE`, `BYTES_TYPE`, `TRUE_SINGLETON`)
/// fall into that hole and surface as "adapter cross-block body Input"
/// Skip events when the same static is read across blocks (predecessor
/// link.args carries no matching entry because the name is not a
/// local).  The orthodox fix is to recognise the name as a known
/// crate-level static at lowering time and emit a `Constant`-style
/// load instead of a body-`Input`, but the immediate prerequisite is a
/// catalogue of every `pub static` in the pyre source tree paired with
/// its declared type so the lowering site can decide.
///
/// This helper produces that catalogue.  Pure-extract — no
/// registration, no side effects; the slice 2 consumer will plumb the
/// result through `lib.rs::analyze_pipeline_from_parsed` into the
/// front-end ctx alongside `unsafe_fn_stubs`.  `prefix` mirrors the
/// other Slice-1 helpers: passed at the file's module path so segments
/// resolve to `[module, name]` (`module_path::name` form).  An empty
/// prefix yields `[name]` single-segment entries (test fixtures /
/// crate-root statics).
///
/// Type projection: routes the static's `syn::Type` through
/// `front/ast.rs::classify_fn_arg_ty` which already handles
/// `i8..i64` / `u8..u64` / `bool` / `f32`/`f64` / `Box<T>`/`Rc<T>`/`Arc<T>`
/// unwrapping + the `Self::Truth` self-ty special case.  Compound
/// types (`PyType`, `LazyLock<...>`, custom structs) surface as
/// `ValueType::Ref` since the address `&STATIC` is the lowering shape
/// upstream `LOAD_GLOBAL` produces for module-level constants
/// (`flowcontext.py:841` pushes the literal value; for compound
/// values pyre emits the address).  `static` (non-`pub`) items are
/// included too — they're still in-scope inside the defining crate
/// and same-crate cross-block reads trip the same Skip family.
pub fn extract_static_decls(
    file: &File,
    prefix: &str,
) -> Vec<(
    Vec<String>,
    crate::model::ValueType,
    Option<crate::flowspace::model::ConstValue>,
)> {
    let mut out = Vec::new();
    let mut push_entry = |ident: &syn::Ident, ty: &syn::Type, init: Option<&syn::Expr>| {
        let mut segments = Vec::with_capacity(2);
        if !prefix.is_empty() {
            segments.push(prefix.to_string());
        }
        segments.push(ident.to_string());
        let value_type = crate::front::ast::classify_fn_arg_ty(ty);
        let const_value = init.and_then(eval_literal_init_to_const_value);
        out.push((segments, value_type, const_value));
    };
    for item in &file.items {
        // `Item::Const` covers crate-level `const X: T = ...` declarations
        // (e.g. `pub const PY_NULL: PyObjectRef = std::ptr::null_mut();`,
        // `pub const WITHPREBUILTINT: bool = false;`). Same Skip family
        // as `static` — single-segment reads cross block boundaries via
        // body-`OpKind::Input`, surface as "adapter cross-block body
        // Input" without a defining link.arg.
        match item {
            Item::Static(s) => push_entry(&s.ident, &s.ty, Some(&s.expr)),
            Item::Const(c) => push_entry(&c.ident, &c.ty, Some(&c.expr)),
            // `thread_local! { static X: T = ...; ... }` expands to a
            // set of TLS-keyed statics; the inner names are
            // single-segment crate-local references that
            // `front/ast.rs::Expr::Path` reads exactly like a normal
            // `static`.  `syn::Item::Macro` carries the body as opaque
            // tokens — parse them as a sequence of `ItemStatic`s and
            // re-use the same emit path.  Other macros are ignored.
            Item::Macro(m) if m.mac.path.is_ident("thread_local") => {
                if let Ok(body) = syn::parse2::<ThreadLocalBody>(m.mac.tokens.clone()) {
                    for s in &body.statics {
                        push_entry(&s.ident, &s.ty, Some(&s.expr));
                    }
                }
            }
            _ => continue,
        }
    }
    out
}

/// Z2.5 Cat 2.1 Slice C — project a `static`/`const` initializer
/// expression to a `ConstValue` when the RHS is a trivial literal.
///
/// PyPy `LOAD_GLOBAL` (`flowspace/flowcontext.py:856`) resolves the
/// name to the actual host object at flowspace time and pushes a
/// `Constant(value)`.  The pyre extraction runs at compile time and
/// has no host-runtime evaluator; the safe subset is literal RHS:
/// `bool` / integer / float / string literals plus the `const { LIT }`
/// block wrapper used inside `thread_local!`.  Returns `None` for any
/// non-literal initializer (function calls, struct constructors,
/// references, offset_of!, etc.) — the caller falls back to the
/// `UniStr(joined_path)` sentinel for those, preserving the
/// pre-Slice-C lowering shape.
///
/// Unary `-LIT` is accepted for signed integer / float literals to
/// match Rust's parse tree (`syn::Expr::Unary { op: Neg, expr: Lit }`).
fn eval_literal_init_to_const_value(
    expr: &syn::Expr,
) -> Option<crate::flowspace::model::ConstValue> {
    use crate::flowspace::model::ConstValue;
    match expr {
        syn::Expr::Lit(syn::ExprLit { lit, .. }) => match lit {
            syn::Lit::Bool(b) => Some(ConstValue::Bool(b.value)),
            syn::Lit::Int(i) => i.base10_parse::<i64>().ok().map(ConstValue::Int),
            syn::Lit::Float(f) => f.base10_parse::<f64>().ok().map(ConstValue::float),
            syn::Lit::Str(s) => Some(ConstValue::UniStr(s.value())),
            syn::Lit::ByteStr(b) => Some(ConstValue::ByteStr(b.value())),
            _ => None,
        },
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr: inner,
            ..
        }) => match inner.as_ref() {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(i),
                ..
            }) => i.base10_parse::<i64>().ok().map(|v| ConstValue::Int(-v)),
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Float(f),
                ..
            }) => f.base10_parse::<f64>().ok().map(|v| ConstValue::float(-v)),
            _ => None,
        },
        // `std::ptr::null()` / `std::ptr::null_mut()` — a NULL raw
        // pointer constant (e.g. `pub const PY_NULL: PyObjectRef =
        // std::ptr::null_mut();`). RPython models the same value as
        // `llmemory.NULL = fakeaddress(None)`; project the call to the
        // `_address::Null` carrier so the `LoadStatic` lowers to a
        // proper null-ref `Constant` instead of the `UniStr(path)`
        // sentinel. The argument-less call with a callee path whose
        // final segment is `null` / `null_mut` is the only shape that
        // resolves; anything with arguments stays unrecognised.
        syn::Expr::Call(syn::ExprCall { func, args, .. }) if args.is_empty() => {
            if let syn::Expr::Path(syn::ExprPath { path, .. }) = func.as_ref() {
                match path.segments.last().map(|s| s.ident.to_string()).as_deref() {
                    Some("null") | Some("null_mut") => Some(ConstValue::LLAddress(
                        crate::translator::rtyper::lltypesystem::lltype::_address::Null,
                    )),
                    _ => None,
                }
            } else {
                None
            }
        }
        // `thread_local! { static X: T = const { LIT }; }` — the
        // initializer wraps a literal in a `syn::Expr::Const` block.
        // Plain `syn::Expr::Block` is also accepted for static
        // initializers that compute the value inside a non-const
        // block (rare; `unsafe { LIT }` and `{ LIT }` shapes).
        syn::Expr::Const(syn::ExprConst { block, .. })
        | syn::Expr::Block(syn::ExprBlock { block, .. })
            if block.stmts.len() == 1 =>
        {
            if let syn::Stmt::Expr(e, None) = &block.stmts[0] {
                eval_literal_init_to_const_value(e)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `thread_local! { ... }` body parser — accepts a sequence of
/// `static NAME: TYPE = EXPR;` entries.  Matches the grammar
/// documented at the std `thread_local!` macro
/// (`rustc_src/library/std/src/thread/local.rs`).
struct ThreadLocalBody {
    statics: Vec<syn::ItemStatic>,
}

impl syn::parse::Parse for ThreadLocalBody {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut statics = Vec::new();
        while !input.is_empty() {
            statics.push(input.parse()?);
        }
        Ok(Self { statics })
    }
}

/// Z2.5 Path C slice 3a — project a `syn::ReturnType` to a
/// `LowLevelType` representable by the slice 2 stub-pygraph builder.
/// Coverage is intentionally narrow at this slice (Bool / Void only) —
/// it lands the wiring shape early so the dominant `pyre_object::is_*`
/// predicate family (every `unsafe fn is_X(obj) -> bool`) can hit the
/// stub-pygraph path immediately.  Other return types surface `None`
/// and the slice 3c caller skips registration, preserving the original
/// "not registered" Skip behavior for those fns until a follow-on
/// widens the projection.
///
/// `Default` return (`fn foo()`) projects to `LowLevelType::Void`.
/// Explicit `() -> ()` does too (single-segment path "()" rejected at
/// the syn level — `syn::Type::Tuple` with no elements is the parse).
/// `bool` matches the literal `bool` identifier at the type-path leaf.
///
/// Returns `None` when the type is unsupported (any non-`bool` type
/// path, references, generics, tuples with elements, function
/// pointers, etc.) — caller treats `None` as "skip this fn".
pub fn simple_return_type_to_lltype(
    ret: &syn::ReturnType,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    match ret {
        syn::ReturnType::Default => Some(LowLevelType::Void),
        syn::ReturnType::Type(_, ty) => match ty.as_ref() {
            syn::Type::Tuple(tup) if tup.elems.is_empty() => Some(LowLevelType::Void),
            syn::Type::Path(tp)
                if tp.qself.is_none()
                    && tp.path.leading_colon.is_none()
                    && tp.path.segments.len() == 1 =>
            {
                let leaf = &tp.path.segments[0];
                if !matches!(leaf.arguments, syn::PathArguments::None) {
                    return None;
                }
                match leaf.ident.to_string().as_str() {
                    "bool" => Some(LowLevelType::Bool),
                    _ => None,
                }
            }
            _ => None,
        },
    }
}

/// Z2.5 Path C slice 3b — combine slice 1's argname-extraction with
/// slice 3a's return-type projection.  Each entry of the returned Vec
/// is a fully-qualified spec tuple ready for slice 3c's
/// `register_unsafe_fn_stubs` (cutover.rs): `(path-segments, Signature,
/// return_lltype)`.
///
/// Filters out unsafe fns whose return type slice 3a cannot project —
/// the caller treats those as "skip" and the original Skip path covers
/// them.  Mirrors the per-fn discard contract documented on
/// [`simple_return_type_to_lltype`].
pub fn extract_unsafe_fn_stubs(
    file: &File,
    prefix: &str,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
)> {
    let pairs = extract_unsafe_fn_signatures(file, prefix);
    let mut out = Vec::with_capacity(pairs.len());
    for (segments, signature) in pairs {
        let Some(ret_ty) = lookup_return_type_for_signature(file, prefix, &segments) else {
            continue;
        };
        let Some(lltype) = simple_return_type_to_lltype(&ret_ty) else {
            continue;
        };
        out.push((segments, signature, lltype));
    }
    out
}

/// Find the `syn::ReturnType` for a registered `(segments, signature)`
/// pair by re-walking the file.  Each `extract_unsafe_fn_signatures`
/// entry corresponds to exactly one `Item::Fn` (free-fn shape
/// `[prefix?, ident]`) or one `Item::Impl::ImplItem::Fn` (impl-method
/// shape `[prefix-segments?..., ImplTy-segments..., method]`).
fn lookup_return_type_for_signature(
    file: &File,
    prefix: &str,
    segments: &[String],
) -> Option<syn::ReturnType> {
    if segments.is_empty() {
        return None;
    }
    let method_or_fn_ident = segments.last()?.as_str();
    let is_free_fn_segments = if prefix.is_empty() {
        segments.len() == 1
    } else {
        segments.len() == 2 && segments[0] == prefix
    };
    for item in &file.items {
        match item {
            Item::Fn(func) if is_free_fn_segments => {
                if func.sig.unsafety.is_some() && func.sig.ident == method_or_fn_ident {
                    return Some(func.sig.output.clone());
                }
            }
            Item::Impl(impl_block) if !is_free_fn_segments => {
                let Some(target_path) = extract_impl_target_path(&impl_block.self_ty) else {
                    continue;
                };
                // Reproduce the prefix-prepend logic of
                // `extract_unsafe_fn_signatures` so the lookup key
                // matches when the impl was written bare under a
                // non-empty prefix.
                let expected_impl_ty: Vec<String> = if target_path.len() > 1 {
                    target_path.clone()
                } else if prefix.is_empty() {
                    target_path.clone()
                } else {
                    let mut segs: Vec<String> = prefix.split("::").map(String::from).collect();
                    segs.extend(target_path.iter().cloned());
                    segs
                };
                let segs_lead = &segments[..segments.len() - 1];
                if segs_lead != expected_impl_ty.as_slice() {
                    continue;
                }
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        if method.sig.unsafety.is_some() && method.sig.ident == method_or_fn_ident {
                            return Some(method.sig.output.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    fn parse_file(src: &str) -> File {
        syn::parse_str::<File>(src).expect("test fixture must parse as syn::File")
    }

    #[test]
    fn extract_unsafe_fn_signatures_empty_file_returns_empty() {
        let file = parse_file("");
        assert!(extract_unsafe_fn_signatures(&file, "anyprefix").is_empty());
    }

    #[test]
    fn extract_unsafe_fn_signatures_safe_fn_skipped() {
        let file = parse_file("pub fn safe_helper(x: i64) -> i64 { x + 1 }");
        assert!(extract_unsafe_fn_signatures(&file, "modprefix").is_empty());
    }

    #[test]
    fn extract_unsafe_fn_signatures_free_unsafe_fn_prefixed() {
        let file = parse_file("pub unsafe fn is_none(obj: *const u8) -> bool { obj.is_null() }");
        let pairs = extract_unsafe_fn_signatures(&file, "pyobject");
        assert_eq!(pairs.len(), 1);
        let (segments, sig) = &pairs[0];
        assert_eq!(
            segments,
            &vec!["pyobject".to_string(), "is_none".to_string()]
        );
        assert_eq!(sig.argnames, vec!["obj".to_string()]);
    }

    #[test]
    fn extract_unsafe_fn_signatures_empty_prefix_emits_bare_name() {
        let file = parse_file("pub unsafe fn bare_unsafe(x: i64) {}");
        let pairs = extract_unsafe_fn_signatures(&file, "");
        assert_eq!(pairs.len(), 1);
        let (segments, _) = &pairs[0];
        assert_eq!(segments, &vec!["bare_unsafe".to_string()]);
    }

    #[test]
    fn extract_unsafe_fn_signatures_impl_unsafe_method_keys_by_impl_type() {
        let file = parse_file(
            "impl Foo { pub unsafe fn method_a(&self, x: i64) -> bool { true } pub fn safe(&self) {} }",
        );
        let pairs = extract_unsafe_fn_signatures(&file, "pyobject");
        assert_eq!(pairs.len(), 1, "safe method must be filtered out");
        let (segments, sig) = &pairs[0];
        // Module-qualified: bare `impl Foo {...}` in prefix `pyobject`
        // → `["pyobject", "Foo", "method_a"]`.
        assert_eq!(
            segments,
            &vec![
                "pyobject".to_string(),
                "Foo".to_string(),
                "method_a".to_string()
            ],
        );
        assert_eq!(sig.argnames, vec!["self".to_string(), "x".to_string()]);
    }

    #[test]
    fn extract_unsafe_fn_signatures_impl_unsafe_method_keeps_qualifier_when_written() {
        let file = parse_file("impl other_mod::Foo { pub unsafe fn method_q(&self) {} }");
        let pairs = extract_unsafe_fn_signatures(&file, "pyobject");
        assert_eq!(pairs.len(), 1);
        let (segments, _) = &pairs[0];
        // Qualified `impl other_mod::Foo {...}` keeps its syntactic
        // path verbatim regardless of the file's prefix.
        assert_eq!(
            segments,
            &vec![
                "other_mod".to_string(),
                "Foo".to_string(),
                "method_q".to_string()
            ],
        );
    }

    #[test]
    fn extract_unsafe_fn_signatures_trait_impl_unsafe_method_keys_by_self_type() {
        let file = parse_file("impl SomeTrait for Bar { unsafe fn op(&self) -> i64 { 0 } }");
        // Empty prefix: bare `impl ... for Bar` stays at `["Bar", "op"]`
        // because there is no module-stem to prepend.
        let pairs = extract_unsafe_fn_signatures(&file, "");
        assert_eq!(pairs.len(), 1);
        let (segments, _) = &pairs[0];
        assert_eq!(segments, &vec!["Bar".to_string(), "op".to_string()]);
    }

    #[test]
    fn extract_unsafe_fn_signatures_mixed_file_returns_all_unsafe_entries() {
        let file = parse_file(
            "pub fn safe_a() {} pub unsafe fn unsafe_a(p: i64) {} impl Cls { pub unsafe fn m(&self) -> bool { true } pub fn safe_m(&self) {} }",
        );
        let pairs = extract_unsafe_fn_signatures(&file, "modx");
        assert_eq!(pairs.len(), 2);
        let segments_set: std::collections::HashSet<Vec<String>> =
            pairs.iter().map(|(s, _)| s.clone()).collect();
        assert!(segments_set.contains(&vec!["modx".to_string(), "unsafe_a".to_string()]));
        // Impl method picks up the file prefix (`modx`) on top of the
        // bare impl target ident.
        assert!(segments_set.contains(&vec![
            "modx".to_string(),
            "Cls".to_string(),
            "m".to_string()
        ]));
    }

    // ─── Z2.5 Path C slice 3a/3b ───

    fn parse_return_ty(src: &str) -> syn::ReturnType {
        let item: ItemFn = syn::parse_str(src).expect("test fixture must parse");
        item.sig.output
    }

    #[test]
    fn simple_return_type_default_yields_void() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let ret = parse_return_ty("fn f() {}");
        assert!(matches!(
            simple_return_type_to_lltype(&ret),
            Some(LowLevelType::Void)
        ));
    }

    #[test]
    fn simple_return_type_unit_tuple_yields_void() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let ret = parse_return_ty("fn f() -> () {}");
        assert!(matches!(
            simple_return_type_to_lltype(&ret),
            Some(LowLevelType::Void)
        ));
    }

    #[test]
    fn simple_return_type_bool_yields_bool() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let ret = parse_return_ty("fn f() -> bool { true }");
        assert!(matches!(
            simple_return_type_to_lltype(&ret),
            Some(LowLevelType::Bool)
        ));
    }

    #[test]
    fn simple_return_type_unsupported_yields_none() {
        // Coverage intentionally narrow at this slice — `i64`, `*const u8`,
        // generic types, references all surface None so the caller skips.
        for src in [
            "fn f() -> i64 { 0 }",
            "fn f() -> u32 { 0 }",
            "fn f() -> *const u8 { std::ptr::null() }",
            "fn f() -> Vec<u8> { Vec::new() }",
            "fn f() -> &'static str { \"\" }",
        ] {
            assert!(
                simple_return_type_to_lltype(&parse_return_ty(src)).is_none(),
                "unsupported return must surface None: {src}"
            );
        }
    }

    #[test]
    fn extract_unsafe_fn_stubs_keeps_only_supported_return_types() {
        let file = parse_file(
            "pub unsafe fn pred(p: *const u8) -> bool { true } \
             pub unsafe fn malloc(n: usize) -> *mut u8 { std::ptr::null_mut() } \
             pub unsafe fn unit_helper(p: i64) {}",
        );
        let stubs = extract_unsafe_fn_stubs(&file, "mem");
        let segments_set: std::collections::HashSet<Vec<String>> =
            stubs.iter().map(|(s, _, _)| s.clone()).collect();
        assert!(
            segments_set.contains(&vec!["mem".to_string(), "pred".to_string()]),
            "bool-returning unsafe fn must produce a stub"
        );
        assert!(
            segments_set.contains(&vec!["mem".to_string(), "unit_helper".to_string()]),
            "()-returning unsafe fn must produce a stub"
        );
        assert!(
            !segments_set.contains(&vec!["mem".to_string(), "malloc".to_string()]),
            "*mut u8 return is unsupported at slice 3a, must skip"
        );
    }

    #[test]
    fn extract_unsafe_fn_stubs_resolves_impl_method_return_types() {
        let file =
            parse_file("impl Foo { pub unsafe fn method_b(&self, x: i64) -> bool { true } }");
        let stubs = extract_unsafe_fn_stubs(&file, "pyobject");
        assert_eq!(stubs.len(), 1);
        let (segments, _, lltype) = &stubs[0];
        // Module-qualified key matches `register_function_graph`'s
        // `["pyobject", "Foo", "method_b"]` shape.
        assert_eq!(
            segments,
            &vec![
                "pyobject".to_string(),
                "Foo".to_string(),
                "method_b".to_string()
            ],
        );
        assert!(matches!(
            lltype,
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Bool
        ));
    }

    /// Verify that `build_host_class_from_struct` writes one entry per
    /// named field into `FORCE_ATTRIBUTES_INTO_CLASSES`, shelled to
    /// the matching `SomeValue` variant.  Skips other test cases'
    /// state by using a uniquely-named struct.
    #[test]
    fn struct_field_attrs_snapshot_carries_named_field_value_types() {
        let item: ItemStruct =
            syn::parse_str("struct PyreFieldStubProbe { count: i64, name: String, flag: bool }")
                .expect("test fixture must parse");
        let _ = build_host_class_from_struct(&item, "");
        let snap = crate::annotator::classdesc::forced_attributes_for("PyreFieldStubProbe")
            .expect("registration must publish under the struct's local name");
        assert_eq!(snap.len(), 3, "every named field must round-trip");
        assert!(matches!(
            snap.get("count"),
            Some(crate::annotator::model::SomeValue::Integer(_))
        ));
        assert!(matches!(
            snap.get("name"),
            Some(crate::annotator::model::SomeValue::Instance(_))
        ));
        assert!(matches!(
            snap.get("flag"),
            Some(crate::annotator::model::SomeValue::Bool(_))
        ));
    }

    /// Verify that re-registering the same struct name overwrites
    /// the prior field set (last-writer-wins on the outer key).
    #[test]
    fn struct_field_attrs_snapshot_re_registration_overwrites() {
        let first: ItemStruct = syn::parse_str("struct PyreFieldStubOverwriteProbe { a: i64 }")
            .expect("fixture parses");
        let second: ItemStruct =
            syn::parse_str("struct PyreFieldStubOverwriteProbe { b: bool, c: f64 }")
                .expect("fixture parses");
        let _ = build_host_class_from_struct(&first, "");
        let _ = build_host_class_from_struct(&second, "");
        let snap =
            crate::annotator::classdesc::forced_attributes_for("PyreFieldStubOverwriteProbe")
                .unwrap();
        assert_eq!(snap.len(), 2, "second registration replaces first");
        assert!(snap.contains_key("b"));
        assert!(snap.contains_key("c"));
        assert!(!snap.contains_key("a"), "first walk's `a` must be evicted");
    }

    /// Qualified-key contract: a struct walked under a non-empty
    /// `module_prefix` keys its field-attr stub under
    /// `"{prefix}::{ident}"`.  Bare lookup misses (no fallback);
    /// only the exact qualified key resolves.  Mirrors PyPy
    /// `ClassDesc` per-class identity (`classdesc.py:672`).
    #[test]
    fn struct_field_attrs_snapshot_qualified_key_no_bare_fallback() {
        let item: ItemStruct =
            syn::parse_str("struct PyreFieldStubQualifiedProbe { count: i64, name: String }")
                .expect("fixture parses");
        let _ = build_host_class_from_struct(&item, "mymod");
        // Bare lookup must miss — fallback was removed.
        assert!(
            crate::annotator::classdesc::forced_attributes_for("PyreFieldStubQualifiedProbe")
                .is_none(),
            "bare lookup must not resolve a qualified registration"
        );
        // Exact qualified lookup hits.
        let snap = crate::annotator::classdesc::forced_attributes_for(
            "mymod::PyreFieldStubQualifiedProbe",
        )
        .expect("qualified lookup matches the registered key");
        assert_eq!(snap.len(), 2);
        assert!(matches!(
            snap.get("count"),
            Some(crate::annotator::model::SomeValue::Integer(_))
        ));
        assert!(matches!(
            snap.get("name"),
            Some(crate::annotator::model::SomeValue::Instance(_))
        ));
    }

    /// Nested-mod prefix composition: a struct inside `mod outer { mod
    /// inner { struct Foo { ... } } }` keys under
    /// `"outer::inner::Foo"`.  Verifies that
    /// `register_items_into_namespace`'s nested-mod arm composes the
    /// `module_prefix` argument from the outer and inner mod idents.
    #[test]
    fn struct_field_attrs_snapshot_nested_mod_qualified_key() {
        let item: ItemStruct = syn::parse_str("struct PyreFieldStubNestedProbe { value: bool }")
            .expect("fixture parses");
        let _ = build_host_class_from_struct(&item, "outer::inner");
        assert!(
            crate::annotator::classdesc::forced_attributes_for("PyreFieldStubNestedProbe")
                .is_none(),
            "bare lookup must miss"
        );
        assert!(
            crate::annotator::classdesc::forced_attributes_for("outer::PyreFieldStubNestedProbe")
                .is_none(),
            "single-segment lookup must miss when registered with two segments"
        );
        let snap = crate::annotator::classdesc::forced_attributes_for(
            "outer::inner::PyreFieldStubNestedProbe",
        )
        .expect("nested-mod prefix must compose into the registry key");
        assert!(matches!(
            snap.get("value"),
            Some(crate::annotator::model::SomeValue::Bool(_))
        ));
    }

    /// Tuple structs and unit structs have no named fields, so the
    /// dict must not be populated for them.
    #[test]
    fn struct_field_attrs_snapshot_skips_unnamed_field_structs() {
        let tuple: ItemStruct =
            syn::parse_str("struct PyreFieldStubTupleProbe(i64, bool);").expect("fixture parses");
        let unit: ItemStruct =
            syn::parse_str("struct PyreFieldStubUnitProbe;").expect("fixture parses");
        let _ = build_host_class_from_struct(&tuple, "");
        let _ = build_host_class_from_struct(&unit, "");
        assert!(
            crate::annotator::classdesc::forced_attributes_for("PyreFieldStubTupleProbe").is_none(),
            "tuple structs do not publish field stubs"
        );
        assert!(
            crate::annotator::classdesc::forced_attributes_for("PyreFieldStubUnitProbe").is_none(),
            "unit structs do not publish field stubs"
        );
    }

    /// Test helper: lookup `name` in `module_id`'s slice of the
    /// module-globals registry and unwrap the expected
    /// `ConstValue::HostObject` shape. Per-module scoping (Issue
    /// 1.3) makes the lookup id-aware; tests pass the id returned
    /// by `register_rust_module`.
    fn lookup_host(module_id: ModuleId, name: &str) -> Option<HostObject> {
        match module_globals_lookup(module_id, name)? {
            ConstValue::HostObject(h) => Some(h),
            other => panic!("expected HostObject for {name}, got {other:?}"),
        }
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

    // ---- Slice O7 — module-globals walker (RPython parity for
    //      `flowcontext.py:847 w_globals.value[varname]` /
    //      `interactive.py:25-26 buildflowgraph` import-time shape).

    #[test]
    fn metadata_only_helper_does_not_lower_body() {
        // upstream Python module import: `def f(...): <body>` creates
        // a function object with `__code__` set; the flowspace graph
        // is NOT built at import time. The metadata-only helper must
        // mirror that — no `build_flow_from_rust` call, no graph in
        // hand, no PyGraph wrapped.
        //
        // A body using a construct the body lowerer rejects (a
        // leading-`::` globally-anchored path — see
        // `build_flow.rs::resolve_path_constant` Unsupported branch)
        // demonstrates this directly: the body lowerer would surface
        // `AdapterError::Unsupported` if invoked, but the metadata
        // path bypasses the body and succeeds.
        let item = parse("fn helper(x: i64) -> i64 { ::std::result::Result::Ok(x); x }");
        // First confirm the body lowerer rejects this fixture so the
        // bypass is actually load-bearing — if leading-`::` paths
        // ever land in `build_flow_from_rust`'s subset, this
        // assertion will fail loudly and the test author can refresh
        // the body to a still-rejected construct.
        assert!(
            super::super::build_flow::build_flow_from_rust(&item).is_err(),
            "fixture body must be rejected by build_flow_from_rust so the \
             metadata-only path is the load-bearing reason this test passes",
        );
        let host = build_host_function_metadata_from_rust(&item, None, None)
            .expect("metadata path skips body lowering");
        assert_eq!(host.qualname(), "helper");
        assert!(host.is_user_function());
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 1);
        assert_eq!(code.co_varnames, vec!["x".to_string()]);
        // upstream `objspace.py:33-35` invariant — code object must
        // carry CO_NEWLOCALS even on the metadata-only path so any
        // later `_assert_rpythonic` re-verify succeeds.
        assert_ne!(code.co_flags & CO_NEWLOCALS, 0);
    }

    #[test]
    fn register_rust_module_registers_item_fn_on_successful_body_lowering() {
        // Slice O16: top-level `Item::Fn` whose body lowers cleanly
        // through `build_flow_from_rust_in_module` is registered as
        // `HostObject::UserFunction` carrying a `prebuilt_flow_graph`.
        // Mirrors upstream Python `def name(): ...` populating
        // `module.__dict__[name]` with a callable function object
        // (`flowcontext.py:847 w_globals.value[varname]`).
        //
        // The eager-build approach uses the `prebuilt_flow_graph`
        // mechanism so downstream resolution gets a real graph, not
        // the empty bytecode that motivated the prior TODO skip.
        // Bodies
        // the lowerer rejects (covered by
        // `register_rust_module_skip_extends_to_unsupported_bodies`)
        // remain unregistered, falling through to the resolver's
        // mint-or-fail path.

        let src = "fn parity_probe_walker_alpha() -> i64 { 1 }
                   fn parity_probe_walker_beta(a: i64) -> i64 { a }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");

        for name in ["parity_probe_walker_alpha", "parity_probe_walker_beta"] {
            match module_globals_lookup(module_id, name) {
                Some(ConstValue::HostObject(host)) => {
                    assert!(
                        host.is_user_function(),
                        "{name}: registered HostObject must be a UserFunction, got non-fn",
                    );
                }
                other => panic!("{name}: expected HostObject(UserFunction), got {other:?}"),
            }
        }
    }

    #[test]
    fn register_rust_module_sibling_fn_resolves_from_caller_body() {
        // Slice O16 acceptance: with eager-build sibling-fn
        // registration, a caller's `LOAD_GLOBAL <sibling_name>`
        // resolves to the registered HostObject and the call site
        // emits a clean `simple_call(<host>, args)` SpaceOperation.
        // Mirrors upstream Python where `def caller(): return
        // helper(x)` resolves `helper` through `module.__dict__`
        // (`flowcontext.py:847 w_globals.value[varname]`).
        //
        // Approach: walk the file (registers helper + caller), then
        // re-build the caller fresh so its prebuilt PyGraph is
        // observable from the test. Walking the freshly-built
        // operations confirms the `LOAD_GLOBAL helper` cascade
        // resolved through the module-globals registry instead of
        // falling into the `mint_unknown` placeholder path.
        use super::super::build_flow::build_flow_from_rust_in_module;
        use crate::flowspace::model::Hlvalue;

        let src = "
            fn parity_probe_helper() -> i64 { 7 }
            fn parity_probe_caller() -> i64 { parity_probe_helper() }
        ";
        let file = syn::parse_file(src).expect("sibling-call fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // Both fns must register because both bodies lower cleanly.
        for name in ["parity_probe_helper", "parity_probe_caller"] {
            match module_globals_lookup(module_id, name) {
                Some(ConstValue::HostObject(host)) => assert!(
                    host.is_user_function(),
                    "{name}: registered HostObject must be UserFunction"
                ),
                other => panic!("{name}: expected HostObject(UserFunction), got {other:?}"),
            }
        }

        // Re-build the caller fresh against the now-populated
        // registry. Iterate `simple_call` operations: at least one
        // must reference the helper's HostObject (qualname-identity
        // match). Pre-O16, the LOAD_GLOBAL would mint a placeholder
        // and the cascade would emit a raw `getattr` chain instead.
        let caller_fn = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(f) if f.sig.ident == "parity_probe_caller" => Some(f),
                _ => None,
            })
            .expect("caller fn present in fixture");
        let graph = build_flow_from_rust_in_module(caller_fn, module_id)
            .expect("caller body lowers cleanly with helper in registry");
        let mut found = false;
        for block_ref in graph.iterblocks() {
            let block = block_ref.borrow();
            for op in &block.operations {
                if op.opname == "simple_call"
                    && let Some(Hlvalue::Constant(c)) = op.args.first()
                    && let ConstValue::HostObject(h) = &c.value
                    && h.qualname() == "parity_probe_helper"
                {
                    found = true;
                }
            }
        }
        assert!(
            found,
            "caller body must contain `simple_call(<helper-host>, ...)` via walker-registered \
             sibling fn (LOAD_GLOBAL → HOST_RUST_MODULE_GLOBALS hit)"
        );
    }

    #[test]
    fn register_rust_module_forward_ref_between_sibling_fns_resolves() {
        // Slice O17: two-pass walker iteratively registers `Item::Fn`s
        // until stable. A caller-before-helper pattern resolves: the
        // first pass-2 iteration fails the caller (helper missing),
        // succeeds the helper; the second iteration succeeds the
        // caller. Mirrors upstream Python where module-import binds
        // `def helper` → `def caller` together so a runtime
        // `caller()` finds `helper` regardless of source order
        // (`flowcontext.py:847 w_globals.value[varname]`).
        use super::super::build_flow::build_flow_from_rust_in_module;
        use crate::flowspace::model::Hlvalue;

        let src = "
            fn parity_probe_fwd_caller() -> i64 { parity_probe_fwd_helper() }
            fn parity_probe_fwd_helper() -> i64 { 42 }
        ";
        let file = syn::parse_file(src).expect("forward-ref fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // Both fns must register because the iterative pass-2 loop
        // unrolls the dependency.
        for name in ["parity_probe_fwd_caller", "parity_probe_fwd_helper"] {
            match module_globals_lookup(module_id, name) {
                Some(ConstValue::HostObject(host)) => assert!(
                    host.is_user_function(),
                    "{name}: registered HostObject must be UserFunction (forward-ref resolved)"
                ),
                other => panic!(
                    "{name}: forward-ref must register (Slice O17 two-pass walker), got {other:?}"
                ),
            }
        }

        // Re-build the caller fresh to confirm its body's
        // `LOAD_GLOBAL parity_probe_fwd_helper` resolved through the
        // registry (not mint).
        let caller_fn = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(f) if f.sig.ident == "parity_probe_fwd_caller" => Some(f),
                _ => None,
            })
            .expect("caller fn present");
        let graph = build_flow_from_rust_in_module(caller_fn, module_id)
            .expect("caller body lowers cleanly with helper registered");
        let mut found = false;
        for block_ref in graph.iterblocks() {
            let block = block_ref.borrow();
            for op in &block.operations {
                if op.opname == "simple_call"
                    && let Some(Hlvalue::Constant(c)) = op.args.first()
                    && let ConstValue::HostObject(h) = &c.value
                    && h.qualname() == "parity_probe_fwd_helper"
                {
                    found = true;
                }
            }
        }
        assert!(
            found,
            "forward-ref `simple_call(<helper-host>, ...)` must resolve via walker registry"
        );
    }

    #[test]
    fn register_rust_module_unlowerable_fn_keeps_placeholder_host() {
        // Slice O17 invariant + strict-parity (2026-05-09): the
        // walker pre-registers every deferred fn's metadata host into
        // `module.__dict__` before any body is lowered, mirroring
        // Python's import-time `def f` populating the module dict
        // ahead of flow analysis (`flowcontext.py:847 find_global`
        // reads live module dict). A fn whose body the adapter
        // rejects (`as T` cast, etc.) keeps its placeholder host —
        // failure surfaces lazily at `buildflowgraph(host)` call time
        // when no PyGraph is attached, not at walker time.
        let src = "
            fn parity_probe_fwd_caller_with_bad() -> i64 { parity_probe_bad_helper(0) }
            fn parity_probe_bad_helper(x: u32) -> i64 { x as i64 }
        ";
        let file = syn::parse_file(src).expect("fixture parses");
        let module_id = register_rust_module(&file).expect("walker terminates");
        // Helper's body fails to lower (`as` cast rejected); its
        // placeholder host stays in `module.__dict__` per PyPy's
        // import-time `def` semantic — `module.__dict__["helper"]`
        // is populated at import time regardless of any later flow
        // failure. The host carries no attached PyGraph, so an
        // actual call would surface the failure downstream.
        assert!(
            matches!(
                module_globals_lookup(module_id, "parity_probe_bad_helper"),
                Some(ConstValue::HostObject(ref h)) if h.is_user_function(),
            ),
            "helper placeholder host must remain in module dict",
        );
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&match module_globals_lookup(
                module_id,
                "parity_probe_bad_helper"
            ) {
                Some(ConstValue::HostObject(h)) => h,
                _ => unreachable!(),
            })
            .is_none(),
            "helper host carries no PyGraph because its body never lowered",
        );
        // Caller's body lowered against the placeholder helper host —
        // matches PyPy's import-time `def` populating module dict
        // before any flow analysis runs.
        assert!(
            matches!(
                module_globals_lookup(module_id, "parity_probe_fwd_caller_with_bad"),
                Some(ConstValue::HostObject(ref h)) if h.is_user_function()
            ),
            "caller fn must register because pre-register-once gave its body \
             visibility into the placeholder helper host"
        );
    }

    #[test]
    fn register_rust_module_item_impl_methods_carry_class_owned_identity_audit_1_5() {
        // Audit 1.5 (2026-05-08): impl methods registered into a
        // class dict must carry `GraphFunc.class_ = Some(target
        // class)` so downstream consumers see class-owned identity.
        // Mirrors upstream Python `func.class_` populated when a
        // `def bar(self): ...` lives inside `class Foo:`. Both the
        // outer-walker path (`register_rust_module_at`) and the
        // inner-mod path (`register_items_into_namespace`) must
        // satisfy this invariant.
        let src = "
            struct ParityProbeAudit15Outer;
            impl ParityProbeAudit15Outer {
                fn outer_method() -> i64 { 1 }
            }
            mod parity_probe_audit_1_5_inner_mod {
                struct InnerStruct;
                impl InnerStruct {
                    fn inner_method() -> i64 { 2 }
                }
            }
        ";
        let file = syn::parse_file(src).expect("audit-1.5 fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // Outer-walker path: class registered, method has class_owned identity.
        let outer_class = match module_globals_lookup(module_id, "ParityProbeAudit15Outer") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected outer class HostObject, got {other:?}"),
        };
        let outer_method = match outer_class.class_get("outer_method") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected outer_method HostObject, got {other:?}"),
        };
        let outer_gf = outer_method
            .user_function()
            .expect("outer method must be UserFunction");
        match &outer_gf.class_ {
            Some(owner) => assert!(
                *owner == outer_class,
                "outer_method.class_ must point at ParityProbeAudit15Outer"
            ),
            None => panic!(
                "audit 1.5: outer_method GraphFunc.class_ must be Some(ParityProbeAudit15Outer); got None"
            ),
        }

        // Inner-mod path: same invariant.
        let inner_ns = match module_globals_lookup(module_id, "parity_probe_audit_1_5_inner_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected inner mod namespace HostObject, got {other:?}"),
        };
        let inner_class = match inner_ns.class_get("InnerStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected InnerStruct HostObject, got {other:?}"),
        };
        let inner_method = match inner_class.class_get("inner_method") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected inner_method HostObject, got {other:?}"),
        };
        let inner_gf = inner_method
            .user_function()
            .expect("inner method must be UserFunction");
        match &inner_gf.class_ {
            Some(owner) => assert!(
                *owner == inner_class,
                "inner_method.class_ must point at InnerStruct"
            ),
            None => panic!(
                "audit 1.5: inner_method GraphFunc.class_ must be Some(InnerStruct); got None"
            ),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_fn_metadata_uses_inner_scope_audit_1_3() {
        // Audit 1.3 (2026-05-08): inner-mod fn's `GraphFunc.module_globals_id`
        // must point at a registry partition mirroring the inner
        // namespace's class dict — NOT the outer module's partition.
        // Mirrors upstream Python `function.__globals__ = inner_mod.__dict__`
        // for fns defined inside `mod foo: ...`. The walker now mints
        // `inner_module_scope` at `register_items_into_namespace`
        // entry and mirrors every `namespace.class_set` into that
        // registry partition; fn / impl-method body builders see this
        // scope as their `module_id` so the metadata snapshot matches
        // the body-side `class_get` channel.
        let src = "
            const OUTER_CONST_FOR_AUDIT_1_3: i64 = 11;
            mod parity_probe_audit_1_3_inner_mod {
                const INNER_CONST_FOR_AUDIT_1_3: i64 = 22;
                fn references_inner_const() -> i64 { INNER_CONST_FOR_AUDIT_1_3 }
            }
        ";
        let file = syn::parse_file(src).expect("audit-1.3 fixture parses");
        let outer_module_id = register_rust_module(&file).expect("walker succeeds");

        // Sanity: outer const IS registered under the outer module id.
        assert_eq!(
            module_globals_lookup(outer_module_id, "OUTER_CONST_FOR_AUDIT_1_3"),
            Some(ConstValue::Int(11))
        );

        let ns = match module_globals_lookup(outer_module_id, "parity_probe_audit_1_3_inner_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected inner mod namespace, got {other:?}"),
        };
        let inner_fn = match ns.class_get("references_inner_const") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected inner fn HostObject, got {other:?}"),
        };
        let gf = inner_fn.user_function().expect("inner fn is UserFunction");

        // Audit 1.3 invariant: the metadata id is NOT the outer mod's.
        assert_ne!(
            gf.module_globals_id,
            Some(outer_module_id),
            "audit 1.3: inner-mod fn metadata must NOT point at outer module_id"
        );
        let inner_scope = gf
            .module_globals_id
            .expect("audit 1.3: inner-mod fn must carry a non-None module_globals_id");

        // The inner scope's snapshot includes the inner const + the
        // fn's own host (re-registered when the walker added it to
        // namespace.class_set). It must NOT include the outer const.
        let inner_snapshot = super::super::host_env::module_globals_snapshot_pub(inner_scope);
        let inner_const_key = ConstValue::byte_str(b"INNER_CONST_FOR_AUDIT_1_3");
        let outer_const_key = ConstValue::byte_str(b"OUTER_CONST_FOR_AUDIT_1_3");
        assert_eq!(
            inner_snapshot.get(&inner_const_key),
            Some(&ConstValue::Int(22)),
            "audit 1.3: inner-mod metadata snapshot must include inner const"
        );
        assert!(
            inner_snapshot.get(&outer_const_key).is_none(),
            "audit 1.3: inner-mod metadata snapshot must NOT include outer const"
        );
    }

    #[test]
    fn register_rust_module_impl_method_host_qualname_is_class_qualified_audit_1_5() {
        // Audit 1.5 extension (2026-05-08): impl methods produce a
        // `HostObject` whose `qualname` is `Class.method` (mirroring
        // upstream Python `func.__qualname__` populated when a
        // `def bar(self): ...` lives inside `class Foo:`). Module-
        // scoped fns keep their bare ident as qualname.
        let src = "
            struct ParityProbeAudit15QualnameOuter;
            impl ParityProbeAudit15QualnameOuter {
                fn outer_method() -> i64 { 1 }
            }
            fn parity_probe_audit_1_5_qualname_free() -> i64 { 0 }
        ";
        let file = syn::parse_file(src).expect("qualname fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // Class-owned method: qualname is `Outer.outer_method`.
        let outer_class = match module_globals_lookup(module_id, "ParityProbeAudit15QualnameOuter")
        {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected class HostObject, got {other:?}"),
        };
        let outer_method = match outer_class.class_get("outer_method") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected method HostObject, got {other:?}"),
        };
        assert_eq!(
            outer_method.qualname(),
            "ParityProbeAudit15QualnameOuter.outer_method",
            "audit 1.5 ext: impl method qualname must be `Class.method`"
        );
        // Short name (`__name__`) stays at the bare method ident —
        // upstream Python `Foo.bar.__name__ == 'bar'`.
        assert_eq!(
            outer_method.simple_name(),
            "outer_method",
            "audit 1.5 ext: impl method `__name__` short-name must stay at the bare ident"
        );

        // Free fn: qualname is just the ident (no class prefix).
        let free_fn = match module_globals_lookup(module_id, "parity_probe_audit_1_5_qualname_free")
        {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected free fn HostObject, got {other:?}"),
        };
        assert_eq!(
            free_fn.qualname(),
            "parity_probe_audit_1_5_qualname_free",
            "audit 1.5 ext: free fn qualname must remain bare ident"
        );
    }

    #[test]
    fn register_rust_module_direct_recursion_resolves_strict_parity_issue_1() {
        // Strict-parity Issue 1 (2026-05-08): walker pre-registers
        // metadata-only HostObject BEFORE body lowering so direct
        // recursion `fn f() { f() }` resolves through the freshly-
        // populated module dict — mirroring upstream Python's
        // import-time `def f` populating `module.__dict__` BEFORE
        // any flow analysis (`flowcontext.py:847 find_global` reads
        // from the live module dict).
        let src = "
            fn parity_probe_strict_issue_1_recursive(n: i64) -> i64 {
                parity_probe_strict_issue_1_recursive(n)
            }
        ";
        let file = syn::parse_file(src).expect("recursive fixture parses");
        let module_id = register_rust_module(&file).expect("walker terminates");
        let host = match module_globals_lookup(module_id, "parity_probe_strict_issue_1_recursive") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected recursive fn HostObject, got {other:?}"),
        };
        assert!(
            host.is_user_function(),
            "strict-parity Issue 1: direct-recursive fn must register under \
             pre-register-then-lower walker shape"
        );
    }

    #[test]
    fn register_rust_module_mutual_recursion_resolves_strict_parity_issue_1() {
        // Strict-parity Issue 1: pre-registration also unblocks
        // mutual recursion (`A` calls `B`, `B` calls `A`). The
        // outer-walker pass-2 registers both metadata hosts BEFORE
        // either body is lowered, so each body's `LOAD_GLOBAL` finds
        // its sibling in the module dict.
        let src = "
            fn parity_probe_mutrec_a(n: i64) -> i64 { parity_probe_mutrec_b(n) }
            fn parity_probe_mutrec_b(n: i64) -> i64 { parity_probe_mutrec_a(n) }
        ";
        let file = syn::parse_file(src).expect("mutual-rec fixture parses");
        let module_id = register_rust_module(&file).expect("walker terminates");
        for name in ["parity_probe_mutrec_a", "parity_probe_mutrec_b"] {
            match module_globals_lookup(module_id, name) {
                Some(ConstValue::HostObject(h)) if h.is_user_function() => {}
                other => panic!("expected mutual-rec {name} HostObject, got {other:?}"),
            }
        }
    }

    #[test]
    fn register_rust_module_top_level_fn_carries_no_class_owner_audit_1_5() {
        // Audit 1.5 (2026-05-08) negative invariant: top-level
        // (module-scoped) fns are not class-owned, so
        // `GraphFunc.class_` stays `None`. Mirrors upstream Python
        // free-fn `func.class_` defaulting to absent.
        let src = "fn parity_probe_audit_1_5_free_fn() -> i64 { 0 }";
        let file = syn::parse_file(src).expect("free-fn fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let host = match module_globals_lookup(module_id, "parity_probe_audit_1_5_free_fn") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected free-fn HostObject, got {other:?}"),
        };
        let gf = host.user_function().expect("free fn is UserFunction");
        assert!(
            gf.class_.is_none(),
            "audit 1.5: top-level fn GraphFunc.class_ must be None"
        );
    }

    #[test]
    fn register_rust_module_item_impl_self_block_populates_class_dict() {
        // Slice O18: `impl Foo { ... }` self-impl blocks contribute
        // associated fns / methods to the target class's dict via
        // `class_set`. Mirrors Python `class Foo: def bar(self): ...`
        // populating `Foo.__dict__["bar"]`. The walker resolves the
        // class name in pass 2 so the impl block can appear above
        // its `struct Foo` declaration in source.
        let src = "
            struct ParityProbeImplStruct;
            impl ParityProbeImplStruct {
                fn associated_helper() -> i64 { 7 }
                fn second_helper(x: i64) -> i64 { x }
            }
        ";
        let file = syn::parse_file(src).expect("self-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // The class itself is registered at module top level.
        let class = match module_globals_lookup(module_id, "ParityProbeImplStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected class HostObject, got {other:?}"),
        };
        assert!(class.is_class());

        // Each impl method becomes a class-dict entry.
        for method_name in ["associated_helper", "second_helper"] {
            match class.class_get(method_name) {
                Some(ConstValue::HostObject(host)) => assert!(
                    host.is_user_function(),
                    "{method_name}: expected UserFunction in class dict"
                ),
                other => panic!(
                    "{method_name}: expected HostObject(UserFunction) in class dict, got {other:?}"
                ),
            }
        }
    }

    #[test]
    fn register_rust_module_item_impl_works_when_struct_appears_after() {
        // Slice O18 forward-resolution: `impl Foo { … }` declared
        // ABOVE `struct Foo` still resolves because the class lookup
        // is deferred to pass 2 (after pass 1 registers all
        // structs/enums).
        let src = "
            impl ParityProbeReverseStruct {
                fn helper() -> i64 { 1 }
            }
            struct ParityProbeReverseStruct;
        ";
        let file = syn::parse_file(src).expect("reverse-order impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeReverseStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match class.class_get("helper") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected helper in class dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_rejects_generic_blocks() {
        // Type / const generic impls (`impl<T> Foo<T>`,
        // `impl<E: Trait> X for E`, `impl<const N: usize> Foo<N>`)
        // need annotator-side specialization (a
        // `SomeInstance.classdef` choice per call site, mirroring
        // upstream `MethodDesc.selfclassdef`) before their methods
        // can attach to a single class dict. Local Rust source
        // declaring such a block is a walker-modeling gap, not an
        // external-scope adaptation; surface the gap as
        // `AdapterError::Unsupported` rather than silently dropping
        // the method graph. (Concrete-self impls with `where`
        // predicates do NOT belong here — see
        // `register_rust_module_item_impl_accepts_where_clause_on_concrete_self`.)
        let src = "
            struct ParityProbeGenericRejectStruct;
            impl<T> ParityProbeGenericRejectStruct {
                fn generic_helper() -> i64 { 3 }
            }
        ";
        let file = syn::parse_file(src).expect("generic-impl fixture parses");
        match register_rust_module(&file) {
            Err(AdapterError::Unsupported { reason }) => {
                assert!(
                    reason.contains("type / const generic"),
                    "expected generic-impl rejection, got: {reason}"
                );
            }
            other => panic!("expected Unsupported(generic impl), got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_accepts_where_clause_on_concrete_self() {
        // An `impl` block whose `generics.params` introduces no
        // type / const parameter must be accepted even when a
        // `where` clause is present, because the predicates only
        // constrain concrete types and lifetimes — no Python-side
        // specialization axis is introduced. Upstream
        // `classdesc.py:590-634 add_source_attribute` flat-stores
        // `classdict[name] = Constant(value)` without inspecting
        // bounds, so the methods attach to `Foo`'s class dict
        // identically to a bare `impl Foo { fn helper(&self) }`.
        //
        // Two shapes covered: `impl Self-only where Self: Trait`
        // (self-impl) and `impl Trait for Foo where Foo: OtherTrait`
        // (trait-impl). Both have empty `generics.params`, only the
        // where-clause carries the trait bound.
        let src = "
            trait ParityProbeWhereMarker {}
            struct ParityProbeWhereStruct;
            impl ParityProbeWhereMarker for ParityProbeWhereStruct {}
            impl ParityProbeWhereStruct where Self: ParityProbeWhereMarker {
                fn self_impl_helper() -> i64 { 7 }
            }
            trait ParityProbeWhereTrait { fn trait_helper(&self) -> i64; }
            impl ParityProbeWhereTrait for ParityProbeWhereStruct
            where ParityProbeWhereStruct: ParityProbeWhereMarker {
                fn trait_helper(&self) -> i64 { 9 }
            }
        ";
        let file = syn::parse_file(src).expect("where-clause fixture parses");
        let module_id =
            register_rust_module(&file).expect("walker accepts concrete-self where-clauses");
        let class = match module_globals_lookup(module_id, "ParityProbeWhereStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected ParityProbeWhereStruct in module dict, got {other:?}"),
        };
        match class.class_get("self_impl_helper") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected self_impl_helper in class dict, got {other:?}"),
        }
        match class.class_get("trait_helper") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected trait_helper in class dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_trait_for_class_populates_class_dict() {
        // Slice O22: `impl Trait for Foo { fn bar(...) { ... } ... }`
        // contributes methods to `Foo`'s class dict identically to a
        // self-impl block. Mirrors upstream
        // `classdesc.py:590-634 add_source_attribute`'s flat
        // `self.classdict[name] = Constant(value)` assignment — the
        // class dict is populated regardless of whether the method
        // comes from a base class through inheritance, because
        // `lookup_filter` (`classdesc.py:336-374`) does the
        // subclass-aware filtering at lookup time. Closed-world
        // dispatch through `bookkeeper.py:431-442 getmethoddesc`
        // keys on `(originclassdef, name, …)`, not on the trait that
        // defined the method.
        let src = "
            struct ParityProbeTraitImplStruct;
            trait ParityProbeTraitImpl { fn t_method(&self) -> i64; }
            impl ParityProbeTraitImpl for ParityProbeTraitImplStruct {
                fn t_method(&self) -> i64 { 11 }
            }
        ";
        let file = syn::parse_file(src).expect("trait-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        let class = match module_globals_lookup(module_id, "ParityProbeTraitImplStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected class HostObject, got {other:?}"),
        };
        assert!(class.is_class());
        match class.class_get("t_method") {
            Some(ConstValue::HostObject(host)) => assert!(
                host.is_user_function(),
                "t_method: expected UserFunction in class dict"
            ),
            other => {
                panic!("t_method: expected HostObject(UserFunction) in class dict, got {other:?}")
            }
        }
    }

    #[test]
    fn register_rust_module_item_impl_trait_resolves_when_struct_appears_after() {
        // Slice O22 forward-resolution mirrors Slice O18: `impl Trait
        // for Foo { … }` declared above `struct Foo` still resolves
        // because the class lookup is deferred to pass 2.
        let src = "
            trait ParityProbeFwdTrait { fn fwd_method(&self) -> i64; }
            impl ParityProbeFwdTrait for ParityProbeFwdStruct {
                fn fwd_method(&self) -> i64 { 13 }
            }
            struct ParityProbeFwdStruct;
        ";
        let file = syn::parse_file(src).expect("forward-ref trait-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeFwdStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match class.class_get("fwd_method") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected fwd_method in class dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_trait_and_self_impl_coexist() {
        // Slice O22: source-order entry into deferred_impls means a
        // later-declared impl block last-writer-wins on the class
        // dict, mirroring Python `Foo.bar = new_bar` overwriting the
        // class-body `def bar(self): …`. Both `impl Foo` and `impl
        // Trait for Foo` populate the same `Foo.classdict`; method
        // names that don't collide coexist.
        let src = "
            struct ParityProbeMixedStruct;
            trait ParityProbeMixedTrait { fn trait_method(&self) -> i64; }
            impl ParityProbeMixedStruct {
                fn self_method(&self) -> i64 { 17 }
            }
            impl ParityProbeMixedTrait for ParityProbeMixedStruct {
                fn trait_method(&self) -> i64 { 19 }
            }
        ";
        let file = syn::parse_file(src).expect("mixed impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeMixedStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        for name in ["self_method", "trait_method"] {
            match class.class_get(name) {
                Some(ConstValue::HostObject(host)) => assert!(
                    host.is_user_function(),
                    "{name}: expected UserFunction in class dict"
                ),
                other => panic!("{name}: expected UserFunction, got {other:?}"),
            }
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_namespaces_inner_items() {
        // Slice O19: `mod foo { ... }` inline submodule binds a
        // namespace `HostObject::Class` at the outer module level
        // whose class dict carries every inner const / static / enum
        // / struct. Mirrors Python `import foo` populating
        // `module.__dict__["foo"]` such that `foo.A` traverses the
        // inner namespace.
        let src = "
            mod parity_probe_inner {
                const INNER_CONST: i64 = 42;
                static INNER_STATIC: i64 = 99;
                enum InnerEnum { Alpha, Beta }
                struct InnerStruct;
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // The namespace itself is registered at module top level.
        let ns = match module_globals_lookup(module_id, "parity_probe_inner") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected namespace HostObject, got {other:?}"),
        };
        assert!(ns.is_class());

        // Inner const → class dict entry as the literal value.
        match ns.class_get("INNER_CONST") {
            Some(ConstValue::Int(42)) => {}
            other => panic!("expected Int(42), got {other:?}"),
        }
        match ns.class_get("INNER_STATIC") {
            Some(ConstValue::Int(99)) => {}
            other => panic!("expected Int(99), got {other:?}"),
        }
        // Inner enum → class with each variant on its dict.
        match ns.class_get("InnerEnum") {
            Some(ConstValue::HostObject(enum_class)) => {
                assert!(enum_class.is_class());
                assert!(enum_class.class_get("Alpha").is_some());
                assert!(enum_class.class_get("Beta").is_some());
            }
            other => panic!("expected enum class HostObject, got {other:?}"),
        }
        match ns.class_get("InnerStruct") {
            Some(ConstValue::HostObject(struct_class)) => {
                assert!(struct_class.is_class());
            }
            other => panic!("expected struct class HostObject, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_const_forward_refs() {
        // Slice O19 invariant: per-mod source-order `const_bindings`
        // resolves forward refs between sibling consts INSIDE the
        // mod (`const Y = X + 1` after `const X = 1`).
        let src = "
            mod parity_probe_inner_fwd {
                const INNER_X: i64 = 10;
                const INNER_Y: i64 = INNER_X + 1;
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_inner_fwd") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match ns.class_get("INNER_Y") {
            Some(ConstValue::Int(11)) => {}
            other => panic!("expected Int(11) (10 + 1), got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_external_item_mod_skipped() {
        // Slice O19: `mod foo;` (no body) resolves to a separate file
        // — out of scope for this slice. Walker silently skips.
        let src = "mod parity_probe_external_mod;";
        let file = syn::parse_file(src).expect("external-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        assert!(
            module_globals_lookup(module_id, "parity_probe_external_mod").is_none(),
            "external `mod foo;` (no body) must not produce a registry entry",
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_dispatches_inner_item_fn() {
        // Slice O20: `Item::Fn` inside `mod foo { ... }` lowers and
        // registers as `foo.<fn_name>` on the inner namespace. Mirrors
        // Python `mod foo: def helper(): ...` populating
        // `foo.__dict__["helper"]` with the function object.
        let src = "
            mod parity_probe_o20_fn_mod {
                fn inner_helper() -> i64 { 5 }
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod-fn fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o20_fn_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected namespace HostObject, got {other:?}"),
        };
        match ns.class_get("inner_helper") {
            Some(ConstValue::HostObject(host)) => assert!(
                host.is_user_function(),
                "expected inner fn lowered as UserFunction"
            ),
            other => panic!("expected fn HostObject in inner-mod dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_dispatches_inner_self_impl() {
        // Slice O20: self-impl `Item::Impl` inside `mod foo { struct
        // Bar; impl Bar { fn helper() {...} } }` resolves the target
        // class against the inner namespace's dict and adds the
        // method via `class_set`. Mirrors Python `class Bar: def
        // helper(self): ...` nested inside a `mod`-like scope.
        let src = "
            mod parity_probe_o20_impl_mod {
                struct InnerStruct;
                impl InnerStruct {
                    fn inner_method() -> i64 { 7 }
                }
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o20_impl_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected namespace HostObject, got {other:?}"),
        };
        let class = match ns.class_get("InnerStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected struct in inner-mod dict, got {other:?}"),
        };
        match class.class_get("inner_method") {
            Some(ConstValue::HostObject(host)) => assert!(
                host.is_user_function(),
                "expected impl method in inner class dict"
            ),
            other => panic!("expected method HostObject in inner class dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_dispatches_inner_trait_impl() {
        // Slice O22: trait `Item::Impl` inside `mod foo { … }` resolves
        // the target class against the inner namespace's dict (NOT the
        // outer module's globals — Rust scoping does not auto-import
        // outer items into inner mods) and adds the method via
        // `class_set`. Mirrors Python `class Bar(Trait): def t_method
        // (self): …` nested inside a `mod`-like scope.
        let src = "
            mod parity_probe_o22_inner_trait_mod {
                struct InnerTraitStruct;
                trait InnerTrait { fn inner_t_method(&self) -> i64; }
                impl InnerTrait for InnerTraitStruct {
                    fn inner_t_method(&self) -> i64 { 23 }
                }
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod-trait-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o22_inner_trait_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected namespace HostObject, got {other:?}"),
        };
        let class = match ns.class_get("InnerTraitStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected struct in inner-mod dict, got {other:?}"),
        };
        match class.class_get("inner_t_method") {
            Some(ConstValue::HostObject(host)) => assert!(
                host.is_user_function(),
                "expected impl method in inner class dict"
            ),
            other => panic!("expected method HostObject in inner class dict, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_multi_segment_self_type_inline_mod_cascade() {
        // Slice O24: `impl Trait for foo::Bar` resolves the target
        // class through the inline-mod cascade. `foo` is a registered
        // `HostObject::Class` namespace from Slice O19, and `Bar` is
        // its inner struct. Methods land on `foo.Bar`'s class dict
        // exactly as for self-impl, mirroring upstream
        // `classdesc.py:590-634 add_source_attribute`'s flat
        // `self.classdict[name] = Constant(value)` shape (no
        // distinction between single-segment and multi-segment paths
        // — Python's `Foo.__dict__` is the same regardless of how
        // `Foo` was reached).
        let src = "
            mod parity_probe_o24_outer_mod {
                struct InnerCascadeStruct;
            }
            trait ParityProbeCascadeTrait { fn cascade_method(&self) -> i64; }
            impl ParityProbeCascadeTrait for parity_probe_o24_outer_mod::InnerCascadeStruct {
                fn cascade_method(&self) -> i64 { 29 }
            }
        ";
        let file = syn::parse_file(src).expect("multi-segment self-type fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let outer_ns = match module_globals_lookup(module_id, "parity_probe_o24_outer_mod") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let inner_class = match outer_ns.class_get("InnerCascadeStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected inner class, got {other:?}"),
        };
        match inner_class.class_get("cascade_method") {
            Some(ConstValue::HostObject(host)) => assert!(
                host.is_user_function(),
                "expected method registered on multi-segment self-type's class dict"
            ),
            other => panic!("expected cascade_method on inline-mod target class, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_multi_segment_resolves_after_inline_mod() {
        // Slice O24: forward-reference variant — `impl X for foo::Bar`
        // declared above `mod foo { struct Bar; }` still resolves
        // because the cascade is retried in the pass-2 fixed-point
        // loop. Mirrors Slice O17 / O18 forward-resolution behavior
        // for fns / impls; the cascade just adds another lookup step.
        let src = "
            trait ParityProbeFwdCascadeTrait { fn fwd_cascade_method(&self) -> i64; }
            impl ParityProbeFwdCascadeTrait for parity_probe_o24_fwd_mod::FwdCascadeStruct {
                fn fwd_cascade_method(&self) -> i64 { 31 }
            }
            mod parity_probe_o24_fwd_mod {
                struct FwdCascadeStruct;
            }
        ";
        let file = syn::parse_file(src).expect("forward-ref multi-segment fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let outer_ns = match module_globals_lookup(module_id, "parity_probe_o24_fwd_mod") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let inner_class = match outer_ns.class_get("FwdCascadeStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match inner_class.class_get("fwd_cascade_method") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected fwd_cascade_method via forward-ref cascade, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_admits_lifetime_only_generics() {
        // Slice O25: `impl<'a> Trait for Foo<'a>` populates `Foo`'s
        // classdict identically to the non-generic shape because
        // lifetime parameters have no Python-observable semantic
        // (RPython lacks the borrow concept). Mirrors upstream
        // `classdesc.py:590-634 add_source_attribute`'s
        // `self.classdict[name] = Constant(value)` flat assignment —
        // same target class, lifetime is dropped at the adapter
        // boundary.
        let src = "
            struct ParityProbeLifetimeStruct;
            trait ParityProbeLifetimeTrait { fn t_lt_method(&self) -> i64; }
            impl<'a> ParityProbeLifetimeTrait for ParityProbeLifetimeStruct {
                fn t_lt_method(&self) -> i64 { 41 }
            }
        ";
        let file = syn::parse_file(src).expect("lifetime-impl fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeLifetimeStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match class.class_get("t_lt_method") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected t_lt_method on impl<'a> target, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_admits_lifetime_only_self_type_args() {
        // Slice O25: self_ty `Foo<'a>` carries lifetime path-args.
        // `extract_impl_target_path` accepts these because lifetimes
        // do not change classdef identity. Class name resolves to
        // `Foo` exactly as for the non-generic `impl Foo` shape.
        let src = "
            struct ParityProbeLtSelfStruct;
            impl<'a> ParityProbeLtSelfStruct {
                fn lt_self_method(&self) -> i64 { 43 }
            }
        ";
        let file = syn::parse_file(src).expect("lifetime-self-type fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeLtSelfStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match class.class_get("lt_self_method") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected lt_self_method on impl<'a> Foo<'a> target, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_admits_lifetime_only_where_clause() {
        // Slice O25: `where 'a: 'b` (lifetime-bound predicates) is
        // admitted alongside lifetime-only `generics.params`. The
        // walker no longer rejects non-lifetime where predicates by
        // themselves; only type / const parameters introduced in
        // `generics.params` create a specialization axis that needs
        // reification.
        let src = "
            struct ParityProbeLtWhereStruct;
            trait ParityProbeLtWhereTrait { fn lt_where_method(&self) -> i64; }
            impl<'a, 'b> ParityProbeLtWhereTrait for ParityProbeLtWhereStruct
            where 'a: 'b
            {
                fn lt_where_method(&self) -> i64 { 47 }
            }
        ";
        let file = syn::parse_file(src).expect("lifetime-where-clause fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let class = match module_globals_lookup(module_id, "ParityProbeLtWhereStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match class.class_get("lt_where_method") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_user_function()),
            other => panic!("expected method via lifetime-only where-clause, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_item_impl_skips_external_rooted_self_type() {
        // Slice O24 scope: paths rooted at `crate::` / `super::` /
        // `self::` / `std::` / `core::` / `alloc::` (or leading-`::`)
        // require multi-file resolution and are skipped. The walker
        // silently drops the impl; downstream resolution falls
        // through to the resolver's mint-or-fail path.
        let src = "
            struct ParitySkipExtSelfTyStruct;
            impl crate::ParitySkipExtSelfTyStruct {
                fn ext_method() -> i64 { 1 }
            }
            impl<T> crate::ParitySkipGenericExternal<T> {
                fn generic_ext_method() -> i64 { 3 }
            }
            impl ::leading::AnchorTy {
                fn leading_method() -> i64 { 2 }
            }
        ";
        let file = syn::parse_file(src).expect("external self-type skip fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        // Local struct registered.
        let local = match module_globals_lookup(module_id, "ParitySkipExtSelfTyStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        // The `impl crate::ParitySkipExtSelfTyStruct { ... }` block
        // is skipped because `crate::` is external-rooted, so its
        // `ext_method` does NOT land on the local class's dict.
        assert!(
            local.class_get("ext_method").is_none(),
            "external-rooted impl methods are skipped"
        );
    }

    #[test]
    fn register_rust_module_use_aliases_existing_module_global() {
        // Slice O23: `use Foo as Bar;` rebinds an already-registered
        // top-level name under a new alias in the module's registry
        // partition. Mirrors Python `from foo import Foo as Bar`
        // populating `module.__dict__["Bar"]` with the value bound to
        // `module.__dict__["Foo"]`.
        let src = "
            struct ParityProbeUseAliasStruct;
            use ParityProbeUseAliasStruct as ParityProbeUseAlias;
        ";
        let file = syn::parse_file(src).expect("use-alias fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let original = module_globals_lookup(module_id, "ParityProbeUseAliasStruct")
            .expect("original name registered");
        let alias = module_globals_lookup(module_id, "ParityProbeUseAlias")
            .expect("alias name registered via Slice O23 use walker");
        assert_eq!(format!("{:?}", original), format!("{:?}", alias));
    }

    #[test]
    fn register_rust_module_use_cascades_through_inline_mod() {
        // Slice O23: `use foo::Bar;` resolves `Bar` through `foo`'s
        // class dict (registered as a `HostObject::Class` namespace
        // by Slice O19's inline-mod walker). Mirrors Python `from foo
        // import Bar` populating `module.__dict__["Bar"]` with
        // `foo.Bar`.
        let src = "
            mod parity_probe_o23_inline_mod {
                struct InnerParityStruct;
                const INNER_PARITY_CONST: i64 = 42;
            }
            use parity_probe_o23_inline_mod::InnerParityStruct;
            use parity_probe_o23_inline_mod::INNER_PARITY_CONST as PARITY_ALIAS_CONST;
        ";
        let file = syn::parse_file(src).expect("use-cascade fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let inner_struct = module_globals_lookup(module_id, "InnerParityStruct")
            .expect("inner struct re-bound at outer level");
        match inner_struct {
            ConstValue::HostObject(h) => assert!(h.is_class()),
            other => panic!("expected HostObject(class), got {other:?}"),
        }
        match module_globals_lookup(module_id, "PARITY_ALIAS_CONST") {
            Some(ConstValue::Int(42)) => {}
            other => panic!("expected Int(42) under alias, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_use_group_expands_each_leaf() {
        // Slice O23: `use foo::{Bar, Baz};` flattens to two
        // bindings (`Bar`, `Baz`), each resolved independently
        // through the inline-mod cascade. Mirrors Python `from foo
        // import Bar, Baz` populating `module.__dict__["Bar"]` and
        // `module.__dict__["Baz"]`.
        let src = "
            mod parity_probe_o23_group_mod {
                struct AlphaProbe;
                struct BetaProbe;
                const GAMMA_PROBE: i64 = 7;
            }
            use parity_probe_o23_group_mod::{AlphaProbe, BetaProbe, GAMMA_PROBE};
        ";
        let file = syn::parse_file(src).expect("use-group fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        for name in ["AlphaProbe", "BetaProbe"] {
            match module_globals_lookup(module_id, name) {
                Some(ConstValue::HostObject(h)) => assert!(h.is_class(), "{name}: expected class"),
                other => panic!("{name}: expected class, got {other:?}"),
            }
        }
        match module_globals_lookup(module_id, "GAMMA_PROBE") {
            Some(ConstValue::Int(7)) => {}
            other => panic!("expected Int(7), got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_use_glob_mirrors_inline_mod_dict() {
        // Slice O23: `use foo::*;` resolves `foo` to a namespace and
        // copies every `class_dict_items` entry into the outer
        // registry partition. Mirrors Python `from foo import *`
        // populating `module.__dict__` with every public name in
        // `foo`'s dict.
        let src = "
            mod parity_probe_o23_glob_mod {
                const ALPHA_GLOB: i64 = 1;
                const BETA_GLOB: i64 = 2;
                struct GammaGlob;
            }
            use parity_probe_o23_glob_mod::*;
        ";
        let file = syn::parse_file(src).expect("use-glob fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        match module_globals_lookup(module_id, "ALPHA_GLOB") {
            Some(ConstValue::Int(1)) => {}
            other => panic!("expected Int(1), got {other:?}"),
        }
        match module_globals_lookup(module_id, "BETA_GLOB") {
            Some(ConstValue::Int(2)) => {}
            other => panic!("expected Int(2), got {other:?}"),
        }
        match module_globals_lookup(module_id, "GammaGlob") {
            Some(ConstValue::HostObject(h)) => assert!(h.is_class()),
            other => panic!("expected class, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inner_use_glob_skips_leading_underscore_names() {
        // PyPy `from foo import *` skips names beginning with `_` when
        // the source object has no `__all__`. The inner namespace branch
        // must apply the same filter as the top-level branch.
        let src = "
            mod parity_probe_o23_inner_glob_outer {
                mod source {
                    const PUBLIC_INNER_GLOB: i64 = 9;
                    const _PRIVATE_INNER_GLOB: i64 = 10;
                    struct PublicInnerGlob;
                    struct _PrivateInnerGlob;
                }
                use source::*;
            }
        ";
        let file = syn::parse_file(src).expect("inner use-glob fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let outer_ns = match module_globals_lookup(module_id, "parity_probe_o23_inner_glob_outer") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected outer namespace, got {other:?}"),
        };

        assert_eq!(
            outer_ns.class_get("PUBLIC_INNER_GLOB"),
            Some(ConstValue::Int(9))
        );
        match outer_ns.class_get("PublicInnerGlob") {
            Some(ConstValue::HostObject(h)) => assert!(h.is_class()),
            other => panic!("expected public class mirrored into outer namespace, got {other:?}"),
        }
        assert!(
            outer_ns.class_get("_PRIVATE_INNER_GLOB").is_none(),
            "inner glob import must not mirror leading-underscore consts"
        );
        assert!(
            outer_ns.class_get("_PrivateInnerGlob").is_none(),
            "inner glob import must not mirror leading-underscore classes"
        );
    }

    #[test]
    fn register_rust_module_use_skips_external_crate_paths() {
        // Slice O23: paths rooted at `crate::`, `super::`, `self::`,
        // `std::`, `core::`, `alloc::` (or leading-`::`) are
        // multi-file / external-crate resolution and skipped pending
        // a future slice. The walker silently drops them; downstream
        // resolution falls through to `mint_unknown` / cascade
        // matching the pre-O23 behavior.
        let src = "
            use crate::SomeName;
            use super::OtherName;
            use std::cell::Cell;
            use ::leading::Anchor;
            mod pyre_object { struct LocalPyreObjectThing; }
            mod majit_trace { struct LocalMajitTraceThing; }
            mod pyrex { struct LocalPyrexThing; }
            use pyre_object::LocalPyreObjectThing;
            use majit_trace::LocalMajitTraceThing;
            use pyrex::LocalPyrexThing;
            struct ParitySkipExternalProbe;
        ";
        let file = syn::parse_file(src).expect("external-skip fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        // None of the external-rooted names are registered.
        for name in [
            "SomeName",
            "OtherName",
            "Cell",
            "Anchor",
            "LocalPyreObjectThing",
            "LocalMajitTraceThing",
            "LocalPyrexThing",
        ] {
            assert!(
                module_globals_lookup(module_id, name).is_none(),
                "{name}: expected NOT registered (external-rooted use)"
            );
        }
        // The local struct is still registered (unrelated to the
        // skipped uses).
        assert!(module_globals_lookup(module_id, "ParitySkipExternalProbe").is_some());
    }

    #[test]
    fn register_rust_module_use_resolves_forward_ref_via_fixed_point() {
        // Slice O23: `use foo::Bar;` declared BEFORE `mod foo { struct
        // Bar; }` still resolves because the use is queued in
        // `deferred_uses` and retried in the same pass-2 fixed-point
        // loop that handles deferred fns / impls. Mirrors source-order
        // independence of Rust `use` statements (Python `from foo
        // import Bar` is technically order-dependent, but the walker
        // matches Rust's order-free semantic for in-file references).
        let src = "
            use parity_probe_o23_fwd_mod::FwdInner;
            mod parity_probe_o23_fwd_mod {
                struct FwdInner;
            }
        ";
        let file = syn::parse_file(src).expect("use-forward-ref fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        match module_globals_lookup(module_id, "FwdInner") {
            Some(ConstValue::HostObject(h)) => assert!(h.is_class()),
            other => panic!("expected class via forward-ref resolution, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_dispatches_inner_use() {
        // Slice O23 inner walker: `use a::b;` inside `mod outer {
        // mod inner { struct X; } use inner::X; }` resolves through
        // the outer mod's namespace dict and binds `X` on the outer
        // mod's class dict. Mirrors Python `class outer:
        // class inner: X = ...; X = inner.X`.
        let src = "
            mod parity_probe_o23_inner_use_outer {
                mod inner_provider {
                    struct ProvidedInner;
                }
                use inner_provider::ProvidedInner;
            }
        ";
        let file = syn::parse_file(src).expect("inner-use fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let outer_ns = match module_globals_lookup(module_id, "parity_probe_o23_inner_use_outer") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match outer_ns.class_get("ProvidedInner") {
            Some(ConstValue::HostObject(h)) => assert!(h.is_class()),
            other => panic!(
                "expected ProvidedInner re-bound on outer namespace via Slice O23 inner-walker use, \
                 got {other:?}"
            ),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_independent_fns_both_register() {
        // Slice O20: pass-2 fixed-point loop runs per-namespace and
        // attempts every collected `Item::Fn` until no more progress.
        // Two source-order-independent inner fns (no inter-fn calls)
        // both land successfully — confirms the per-namespace pass-2
        // sweep mirrors the outer walker's Slice O17 behavior for
        // independent siblings.
        let src = "
            mod parity_probe_o20_indep_mod {
                fn inner_first() -> i64 { 1 }
                fn inner_second() -> i64 { 2 }
            }
        ";
        let file = syn::parse_file(src).expect("inline-mod-indep fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o20_indep_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected namespace HostObject, got {other:?}"),
        };
        assert!(matches!(
            ns.class_get("inner_first"),
            Some(ConstValue::HostObject(h)) if h.is_user_function(),
        ));
        assert!(matches!(
            ns.class_get("inner_second"),
            Some(ConstValue::HostObject(h)) if h.is_user_function(),
        ));
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_sibling_call_resolves() {
        // Slice O21: inner-mod fn `inner_caller` references sibling
        // `inner_helper` through the inline-mod's namespace dict.
        // `build_host_function_from_rust_in_module` now threads the
        // namespace through `func_globals`, so the body's
        // `LOAD_GLOBAL inner_helper` resolves via
        // `namespace.class_get("inner_helper")` — mirrors Python
        // `function.__globals__ = inner_mod.__dict__`.
        //
        // Pre-Slice-O21 this test asserted the caller stays
        // UNREGISTERED (documented Slice O20 limitation). Convergence
        // landed in O21 by adding the per-fn globals channel.
        //
        // Strict-parity (2026-05-11): the fixture's source order is
        // `caller` BEFORE `helper`, exercising the inner walker's
        // pre-register pass + pass-2 body retry — without it, caller's
        // body lowers in iter 1 while `helper` is still absent from the
        // namespace dict, so the caller's body fails and the
        // placeholder host carries no PyGraph. The
        // `lookup_walker_pygraph` assertions below catch that
        // false-positive: an `is_user_function()` placeholder without a
        // PyGraph passes the prior shape check but fails the new one.
        let src = "
            mod parity_probe_o21_sibling_call {
                fn inner_caller() -> i64 { inner_helper() }
                fn inner_helper() -> i64 { 1 }
            }
        ";
        let file = syn::parse_file(src).expect("sibling-call fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o21_sibling_call") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let helper_host = match ns.class_get("inner_helper") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected inner_helper UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&helper_host).is_some(),
            "inner_helper must carry a PyGraph after walker pass-2",
        );
        let caller_host = match ns.class_get("inner_caller") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected inner_caller UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&caller_host).is_some(),
            "Slice O21 + Strict-parity (2026-05-11): inner-mod fn body \
             must lower into a PyGraph even when its sibling reference \
             points at a fn declared LATER in source order. Without the \
             inner walker's pre-register pass, the placeholder host \
             stays in the namespace dict but carries no PyGraph — \
             `is_user_function()` alone does not catch this regression.",
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_impl_method_forward_ref_resolves() {
        // Strict-parity (2026-05-11): inner-mod impl method forward
        // reference. `caller` references sibling `helper` through
        // `Self::helper(self)` AFTER `helper` is declared — but the
        // walker processes impl items in source order. The Phase A
        // pre-registers ALL methods (`class_set`s them onto the class
        // dict in iter 1's Phase A) before the Phase B body retry
        // loops on still-pending bodies, so the body's
        // `LOAD_ATTR helper` cascades through `class_get("helper")`
        // and finds the placeholder host even though `helper` is
        // declared LATER in source order.
        //
        // Mirrors upstream Python `class Foo: def caller(self):
        // self.helper(); def helper(self): pass` — class body executes
        // both `def`s into `Foo.__dict__` before any flow analysis
        // looks them up.
        let src = "
            mod parity_probe_inner_impl_fwd {
                pub struct Foo;
                impl Foo {
                    pub fn caller(&self) -> i64 { self.helper() }
                    pub fn helper(&self) -> i64 { 1 }
                }
            }
        ";
        let file = syn::parse_file(src).expect("inner-impl-fwd fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_inner_impl_fwd") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let foo = match ns.class_get("Foo") {
            Some(ConstValue::HostObject(h)) if h.is_class() => h,
            other => panic!("expected Foo class, got {other:?}"),
        };
        let helper_host = match foo.class_get("helper") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected Foo::helper UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&helper_host).is_some(),
            "Foo::helper must carry a PyGraph after walker pass-2",
        );
        let caller_host = match foo.class_get("caller") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected Foo::caller UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&caller_host).is_some(),
            "Strict-parity (2026-05-11): inner-mod impl method body \
             must lower into a PyGraph even when its sibling method \
             reference is to a method declared LATER in source order. \
             Without the inner walker's Phase A class-dict pre-register \
             pass, the placeholder method stays in the class dict but \
             carries no PyGraph.",
        );
    }

    #[test]
    fn register_rust_module_top_level_impl_method_forward_ref_resolves() {
        // Strict-parity (2026-05-11): top-level impl method forward
        // reference. Same shape as the inner-mod test above but at
        // module top level. Exercises the OUTER walker's Phase A
        // class-dict pre-register pass (the new `pending_methods`
        // queue at the outer walker).
        let src = "
            pub struct Foo;
            impl Foo {
                pub fn caller(&self) -> i64 { self.helper() }
                pub fn helper(&self) -> i64 { 1 }
            }
        ";
        let file = syn::parse_file(src).expect("outer-impl-fwd fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let foo = match module_globals_lookup(module_id, "Foo") {
            Some(ConstValue::HostObject(h)) if h.is_class() => h,
            other => panic!("expected Foo class, got {other:?}"),
        };
        let helper_host = match foo.class_get("helper") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected Foo::helper UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&helper_host).is_some(),
            "Foo::helper must carry a PyGraph after walker pass-2",
        );
        let caller_host = match foo.class_get("caller") {
            Some(ConstValue::HostObject(h)) if h.is_user_function() => h,
            other => panic!("expected Foo::caller UserFunction, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&caller_host).is_some(),
            "Strict-parity (2026-05-11): top-level impl method body \
             must lower into a PyGraph even when its sibling method \
             reference is to a method declared LATER in source order. \
             Without the outer walker's Phase A class-dict pre-register \
             pass, the placeholder method stays in the class dict but \
             carries no PyGraph.",
        );
    }

    #[test]
    fn register_rust_module_top_level_impl_cross_trait_method_collision_rejected() {
        // Strict-parity (2026-05-11, Item 5): two distinct trait
        // impls writing the same method name to the same target type
        // would collapse into a single classdict entry under the
        // walker's flat `class_set(name, ...)` semantics, losing
        // dispatch identity. The walker now detects this at pre-loop
        // time and surfaces an `AdapterError::Unsupported`.
        let src = "
            pub struct Foo;
            pub trait TraitA { fn name(&self) -> i64; }
            pub trait TraitB { fn name(&self) -> i64; }
            impl TraitA for Foo { fn name(&self) -> i64 { 1 } }
            impl TraitB for Foo { fn name(&self) -> i64 { 2 } }
        ";
        let file = syn::parse_file(src).expect("collision fixture parses");
        let err = register_rust_module(&file)
            .expect_err("outer walker must reject cross-trait method-name collision");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Cross-trait method-name collision"),
            "outer-walker collision error must carry the diagnostic; got {msg:?}",
        );
        assert!(
            msg.contains("Foo::name"),
            "outer-walker collision error must name the method; got {msg:?}",
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_impl_cross_trait_method_collision_rejected() {
        // Strict-parity (2026-05-11, Item 5): same collision check as
        // the outer-walker variant above, scoped to inline-mod impls.
        let src = "
            mod parity_probe_inner_collide {
                pub struct Foo;
                pub trait TraitA { fn name(&self) -> i64; }
                pub trait TraitB { fn name(&self) -> i64; }
                impl TraitA for Foo { fn name(&self) -> i64 { 1 } }
                impl TraitB for Foo { fn name(&self) -> i64 { 2 } }
            }
        ";
        let file = syn::parse_file(src).expect("inner collision fixture parses");
        let err = register_rust_module(&file)
            .expect_err("inner walker must reject cross-trait method-name collision");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Cross-trait method-name collision"),
            "inner-walker collision error must carry the diagnostic; got {msg:?}",
        );
        assert!(
            msg.contains("Foo::name"),
            "inner-walker collision error must name the method; got {msg:?}",
        );
    }

    #[test]
    fn register_rust_module_top_level_impl_cross_trait_alias_collision_rejected() {
        // Strict-parity round-4 (2026-05-11): cross-trait collision
        // with *aliased* receivers (`use Foo as X; use Foo as Y;`)
        // resolves both class paths to the same `Arc<HostObjectInner>`
        // — the textual class-path key (`["X"]` vs `["Y"]`) would
        // miss the conflict, but the resolved `identity_id()` keying
        // catches it. This pins the alias-path collision detection
        // that the textual key from the prior round 3 fix could not.
        let src = "
            pub struct Foo;
            pub trait TraitA { fn name(&self) -> i64; }
            pub trait TraitB { fn name(&self) -> i64; }
            use Foo as X;
            use Foo as Y;
            impl TraitA for X { fn name(&self) -> i64 { 1 } }
            impl TraitB for Y { fn name(&self) -> i64 { 2 } }
        ";
        let file = syn::parse_file(src).expect("alias collision fixture parses");
        let err = register_rust_module(&file)
            .expect_err("outer walker must reject alias-rooted cross-trait collision");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Cross-trait method-name collision"),
            "outer-walker alias collision error must carry the diagnostic; got {msg:?}",
        );
        assert!(
            msg.contains("identity matches across alias paths"),
            "outer-walker alias collision error must mention identity match; got {msg:?}",
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_impl_cross_trait_alias_collision_rejected() {
        // Strict-parity round-4 (2026-05-11): inner-walker variant of
        // the alias-path collision pin. `use Foo as X; use Foo as Y;`
        // inside an inline mod cascade through the inner namespace
        // dict — both aliases resolve to the same `Arc<HostObjectInner>`
        // for `Foo`. The resolved-identity collision check catches the
        // collision that a textual `(class_path, method_name)` key
        // would miss.
        let src = "
            mod parity_probe_inner_alias_collide {
                pub struct Foo;
                pub trait TraitA { fn name(&self) -> i64; }
                pub trait TraitB { fn name(&self) -> i64; }
                use Foo as X;
                use Foo as Y;
                impl TraitA for X { fn name(&self) -> i64 { 1 } }
                impl TraitB for Y { fn name(&self) -> i64 { 2 } }
            }
        ";
        let file = syn::parse_file(src).expect("inner alias collision fixture parses");
        let err = register_rust_module(&file)
            .expect_err("inner walker must reject alias-rooted cross-trait collision");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("Cross-trait method-name collision"),
            "inner-walker alias collision error must carry the diagnostic; got {msg:?}",
        );
        assert!(
            msg.contains("identity matches across alias paths"),
            "inner-walker alias collision error must mention identity match; got {msg:?}",
        );
    }

    #[test]
    fn register_rust_module_walker_error_registry_captures_body_lower_failure() {
        // Strict-parity round-4 (2026-05-11): the walker keeps a
        // sibling fn placeholder in the module dict when its body
        // fails to lower (PyPy `pyopcode.py:1405 STORE_NAME` import-
        // time semantic), but the original `AdapterError` would be
        // dropped without a side channel. The new
        // `HOST_RUST_WALKER_ERRORS` registry captures the rendered
        // error so a downstream `buildflowgraph(host)` boundary can
        // re-surface it. This test pins the capture at the walker
        // layer: a sibling fn whose body uses a closure expression
        // (rejected by `build_flow_from_rust` as
        // `AdapterError::Unsupported { reason: "closure (not in
        // roadmap scope)" }`) ends up in the module dict but with a
        // populated walker-error entry.
        let src = "
            fn good_entry() -> i64 { 1 }
            fn closure_body() -> i64 {
                let f = |x: i64| x + 1;
                f(0)
            }
        ";
        let file = syn::parse_file(src).expect("walker-error fixture parses");
        let module_id = register_rust_module(&file).expect("walker pass returns Ok overall");
        let bad_host = match module_globals_lookup(module_id, "closure_body") {
            Some(ConstValue::HostObject(h)) => h,
            other => {
                panic!("closure_body must stay in module dict as placeholder host, got {other:?}",)
            }
        };
        // The placeholder host carries no PyGraph (body never lowered).
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&bad_host).is_none(),
            "body-failed host must NOT have a PyGraph attached",
        );
        // The walker error registry captures the original
        // `AdapterError::Unsupported` rendering.
        let captured = crate::flowspace::rust_source::lookup_walker_error(&bad_host)
            .expect("walker must capture the body-lower AdapterError");
        assert!(
            captured.contains("closure"),
            "captured walker error must name the rejected construct; got {captured:?}",
        );
        assert!(
            captured.contains("unsupported construct"),
            "captured walker error must come from AdapterError::Unsupported Display; got {captured:?}",
        );
        // Cleanup so subsequent tests don't see a stale entry —
        // production drains via `Translation::drain_walker_pygraphs`,
        // but this test exercises the walker without constructing a
        // Translation.
        let _ = crate::flowspace::rust_source::drain_walker_errors();
        let _ = crate::flowspace::rust_source::drain_walker_pygraphs();
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_fn_references_inner_const() {
        // Slice O21 cross-shape lookup: inner-mod fn references an
        // inner-mod const through the namespace dict. Const is
        // populated as a `ConstValue::Int` directly (Slice O10
        // generalization), so `func_globals.class_get("INNER_CONST")`
        // returns `Some(ConstValue::Int(7))` and the body's `+`
        // constfolds.
        let src = "
            mod parity_probe_o21_const_ref {
                const INNER_CONST: i64 = 7;
                fn reads_inner_const() -> i64 { INNER_CONST + 1 }
            }
        ";
        let file = syn::parse_file(src).expect("inner-const-ref fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o21_const_ref") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        assert_eq!(ns.class_get("INNER_CONST"), Some(ConstValue::Int(7)));
        assert!(
            matches!(
                ns.class_get("reads_inner_const"),
                Some(ConstValue::HostObject(h)) if h.is_user_function(),
            ),
            "fn referencing an inner-mod const must resolve through \
             the namespace's class_get channel"
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_fn_does_not_see_outer_module_globals() {
        // Slice O21 scoping invariant: with `func_globals = Some(ns)`,
        // the global-lookup channel targets `ns.class_get(name)` ONLY
        // — it does NOT fall through to `module_globals_lookup` for
        // the outer module. Mirrors Python `function.__globals__`
        // being a single dict, never a chain.
        //
        // Strict-parity (2026-05-10): the inner fn's placeholder host
        // STAYS in the inner namespace even when its body fails to
        // resolve `OUTER_CONST_FOR_O21_SCOPING` — mirrors Python's
        // `def f` populating the namespace dict at class-body exec
        // time regardless of any later flow-analysis state. The
        // placeholder carries no attached PyGraph; downstream
        // `buildflowgraph(host)` surfaces the lazy failure at the
        // call site rather than retracting the dict entry here.
        // The scoping invariant is now expressed as "the placeholder
        // host has no PyGraph" rather than "the entry is absent".
        let src = "
            const OUTER_CONST_FOR_O21_SCOPING: i64 = 99;
            mod parity_probe_o21_scoping {
                fn references_outer() -> i64 { OUTER_CONST_FOR_O21_SCOPING }
            }
        ";
        let file = syn::parse_file(src).expect("scoping fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let ns = match module_globals_lookup(module_id, "parity_probe_o21_scoping") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let host = match ns.class_get("references_outer") {
            Some(ConstValue::HostObject(h)) => {
                assert!(h.is_user_function());
                h
            }
            other => {
                panic!("inner-mod fn placeholder host must remain in namespace, got {other:?}")
            }
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&host).is_none(),
            "inner-mod fn body must NOT have lowered (no PyGraph) — Rust \
             scoping requires `use super::X`; the per-fn __globals__ \
             carrier is a single dict, not a chain"
        );
        // Sanity: outer const IS registered at outer module level.
        assert_eq!(
            module_globals_lookup(module_id, "OUTER_CONST_FOR_O21_SCOPING"),
            Some(ConstValue::Int(99))
        );
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_const_does_not_see_outer_module_globals() {
        // Audit 1.4 invariant (2026-05-08) + strict-parity (2026-05-10):
        // inner-mod const RHS bare names that miss the per-mod
        // `inner_bindings` must NOT fall through to the outer module's
        // `module_globals_lookup` partition. Both Rust and Python require
        // explicit `use super::X` to reach outer items; `super::` is
        // currently classified `External` by `classify_use_root` and
        // silently skipped.
        //
        // Strict-parity (2026-05-10): unresolved single-segment Path
        // raises `AdapterError::Flowing` (NameError mirroring upstream
        // `flowcontext.py:845 find_global`), and the inner-mod walker
        // now PROPAGATES that error verbatim — same shape as the
        // top-level `Item::Const` arm. `register_rust_module` aborts
        // the file on the bare-outer-ref fixture; this matches Python's
        // import-time NameError on the same source-equivalent shape.
        let src = "
            const OUTER_CONST_FOR_AUDIT_1_4: i64 = 77;
            mod parity_probe_audit_1_4_const_scoping {
                const REFERENCES_OUTER: i64 = OUTER_CONST_FOR_AUDIT_1_4 + 1;
            }
        ";
        let file = syn::parse_file(src).expect("audit-1.4 fixture parses");
        let err = register_rust_module(&file)
            .expect_err("inner-mod outer-ref must surface FlowingError, not silent-skip");
        assert!(
            matches!(err, AdapterError::Flowing { .. }),
            "expected AdapterError::Flowing for unresolved outer ident, got {err:?}"
        );
    }

    #[test]
    fn register_rust_module_outer_module_fn_unaffected_by_o21() {
        // Slice O21 backward compatibility: outer-module fns
        // continue to lower with `func_globals = None`, falling
        // back to `module_globals_lookup(module_id, name)` for the
        // sibling-fn channel (Slice O17 behavior).
        let src = "
            fn outer_caller() -> i64 { outer_helper() }
            fn outer_helper() -> i64 { 42 }
        ";
        let file = syn::parse_file(src).expect("outer-fn fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        // Both fns register at module top level — pre-existing
        // pass-2 fixed-point loop semantic.
        assert!(matches!(
            module_globals_lookup(module_id, "outer_caller"),
            Some(ConstValue::HostObject(h)) if h.is_user_function(),
        ));
        assert!(matches!(
            module_globals_lookup(module_id, "outer_helper"),
            Some(ConstValue::HostObject(h)) if h.is_user_function(),
        ));
    }

    #[test]
    fn register_rust_module_inline_item_mod_dispatches_nested_mod() {
        // Slice O20: `mod a { mod b { ... } }` walks recursively.
        // Outer-mod namespace `a` carries nested-mod `b` as a
        // `HostObject::Class` entry; `b`'s own dict carries inner
        // items. Mirrors `a.b.X` attribute traversal.
        let src = "
            mod parity_probe_o20_outer_mod {
                mod parity_probe_o20_inner_mod {
                    const NESTED_CONST: i64 = 99;
                    fn nested_fn() -> i64 { 0 }
                    struct NestedStruct;
                }
            }
        ";
        let file = syn::parse_file(src).expect("nested-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");
        let outer = match module_globals_lookup(module_id, "parity_probe_o20_outer_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected outer namespace HostObject, got {other:?}"),
        };
        let inner = match outer.class_get("parity_probe_o20_inner_mod") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected nested namespace HostObject, got {other:?}"),
        };
        assert!(inner.is_class(), "nested mod must mint a class HostObject");
        assert_eq!(inner.class_get("NESTED_CONST"), Some(ConstValue::Int(99)));
        match inner.class_get("nested_fn") {
            Some(ConstValue::HostObject(host)) => {
                assert!(host.is_user_function(), "nested-mod fn must lower");
            }
            other => panic!("expected nested fn HostObject, got {other:?}"),
        }
        match inner.class_get("NestedStruct") {
            Some(ConstValue::HostObject(host)) => assert!(host.is_class()),
            other => panic!("expected nested struct class, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_inline_item_mod_inner_impl_does_not_leak_to_outer() {
        // Slice O20 scoping: `impl InnerStruct` inside `mod foo`
        // looks up `InnerStruct` against `foo`'s namespace, NOT the
        // outer module. If the same struct exists at outer level,
        // the inner impl methods do NOT bind on the outer struct's
        // class dict.
        let src = "
            struct ParityProbeO20OuterStruct;
            mod parity_probe_o20_scope_mod {
                struct ParityProbeO20OuterStruct;
                impl ParityProbeO20OuterStruct {
                    fn inner_only_method() -> i64 { 11 }
                }
            }
        ";
        let file = syn::parse_file(src).expect("scoping fixture parses");
        let module_id = register_rust_module(&file).expect("walker succeeds");

        // Outer struct exists at module top level...
        let outer_struct = match module_globals_lookup(module_id, "ParityProbeO20OuterStruct") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected outer struct, got {other:?}"),
        };
        // ... but it must NOT carry the inner-mod impl method.
        assert!(
            outer_struct.class_get("inner_only_method").is_none(),
            "inner-mod impl method must not leak onto outer-module struct"
        );

        // The inner-mod struct DOES carry the method.
        let inner_ns = match module_globals_lookup(module_id, "parity_probe_o20_scope_mod") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        let inner_struct = match inner_ns.class_get("ParityProbeO20OuterStruct") {
            Some(ConstValue::HostObject(h)) => h,
            _ => unreachable!(),
        };
        match inner_struct.class_get("inner_only_method") {
            Some(ConstValue::HostObject(h)) => assert!(h.is_user_function()),
            other => panic!("inner-mod struct must own the impl method, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_unsupported_body_keeps_placeholder_host() {
        // Strict-parity (2026-05-09): an Item::Fn whose body the
        // lowerer rejects (e.g. `as T` cast — task #94) keeps its
        // pre-registered placeholder host in `module.__dict__` —
        // mirrors Python's import-time `def f` populating the
        // module dict regardless of any later flow-analysis state.
        // The host carries no attached PyGraph, so a downstream
        // `buildflowgraph(host)` call would surface the failure
        // lazily rather than at walker time.

        let src = "fn parity_probe_walker_with_cast(x: u32) -> i64 { x as i64 }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let host = match module_globals_lookup(module_id, "parity_probe_walker_with_cast") {
            Some(ConstValue::HostObject(h)) => {
                assert!(h.is_user_function());
                h
            }
            other => panic!("placeholder host must remain in module dict, got {other:?}"),
        };
        assert!(
            crate::flowspace::rust_source::lookup_walker_pygraph(&host).is_none(),
            "placeholder host must NOT carry a PyGraph because body never lowered",
        );
    }

    #[test]
    fn register_rust_module_skips_non_walked_item_kinds() {
        // Walker dispatches `Item::Enum`, `Item::Struct`, `Item::Const`
        // (Slice O10), and `Item::Static` (Slice O12). Remaining kinds
        // (`Item::Use`, `Item::Mod`, `Item::Impl`, `Item::Fn`) are
        // follow-up slices — they must NOT pollute the module-globals
        // registry until their dispatch is added.
        use super::super::host_env::module_globals_lookup;

        // `use` re-export — upstream `from x import y` would bind
        // `y` in `module.__dict__`; pyre walker doesn't yet resolve
        // cross-module references, so the binding is silently
        // skipped.
        let src = "use std::collections::HashMap as PARITY_PROBE_WALKER_USE_ONLY;";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert!(module_globals_lookup(module_id, "PARITY_PROBE_WALKER_USE_ONLY").is_none());
    }

    #[test]
    fn register_rust_module_walks_item_static_like_item_const() {
        // Slice O12: immutable `static FOO: T = <expr>;` populates
        // `module.__dict__` identically to `const FOO`.
        use super::super::host_env::module_globals_lookup;

        let src = "static ParityProbe_O12_Static: i64 = 5;
                   static ParityProbe_O12_StaticStr: &str = \"hello\";
                   static ParityProbe_O12_StaticBool: bool = true;";
        let file = syn::parse_file(src).expect("static fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Static"),
            Some(ConstValue::Int(5)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_StaticStr"),
            Some(ConstValue::uni_str("hello".to_string())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_StaticBool"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_skips_mutable_static() {
        // Slice O15 / codex parity audit (2026-05-06): `static mut`
        // is NOT registered. Runtime mutation makes the initial-
        // value snapshot unsound for constfold reads. PRE-EXISTING-
        // ADAPTATION until a live-store path lands. Downstream
        // lookups for the name fall through to the mint-or-fail
        // resolver — the registry intentionally has no entry.
        use super::super::host_env::module_globals_lookup;

        let src = "static mut ParityProbe_O15_StaticMut: i64 = 7;
                   static ParityProbe_O15_StaticImmut: i64 = 9;";
        let file = syn::parse_file(src).expect("static-mut fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert!(
            module_globals_lookup(module_id, "ParityProbe_O15_StaticMut").is_none(),
            "`static mut` must NOT be registered (runtime mutation makes snapshot unsound)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O15_StaticImmut"),
            Some(ConstValue::Int(9)),
            "immutable `static` next to a `static mut` still walks normally",
        );
    }

    #[test]
    fn register_rust_module_static_resolves_compound_rhs_against_prior_bindings() {
        // Static RHS goes through the same `eval_const_expr` as
        // `Item::Const`, so forward-ref to a sibling `const` (or
        // sibling `static`) works the same way: source-order
        // `bindings` accumulator is populated before the next item
        // is evaluated. Mirrors upstream Python's import-time linear
        // execution: `X = 1; Y = X + 1` binds both, in order.
        use super::super::host_env::module_globals_lookup;

        let src = "const ParityProbe_O12_X: i64 = 10;
                   static ParityProbe_O12_Y: i64 = ParityProbe_O12_X + 5;
                   static ParityProbe_O12_Z: i64 = ParityProbe_O12_Y * 2;";
        let file = syn::parse_file(src).expect("compound static fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_X"),
            Some(ConstValue::Int(10)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Y"),
            Some(ConstValue::Int(15)),
            "`Y` resolves prior `const X` through the source-order accumulator",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Z"),
            Some(ConstValue::Int(30)),
            "`Z` resolves prior `static Y` (Slice O12 admits Item::Static into the accumulator)",
        );
    }

    #[test]
    fn register_rust_module_walks_item_enum_with_variants_as_children() {
        // upstream Python analogue: `class StepResult: pass; class
        // StepResult_Continue(StepResult): pass; StepResult.Continue
        // = StepResult_Continue`. The walker's enum dispatch produces
        // the same shape — parent class with each variant bound in
        // the class dict to a child class whose bases include the
        // parent.

        let src = "pub enum ParityProbeEnum_Slice_O8 { Alpha, Beta, Gamma }";
        let file = syn::parse_file(src).expect("enum fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");

        let parent =
            lookup_host(module_id, "ParityProbeEnum_Slice_O8").expect("enum registered after walk");
        assert!(parent.is_class());
        assert_eq!(parent.qualname(), "ParityProbeEnum_Slice_O8");

        for variant in ["Alpha", "Beta", "Gamma"] {
            let entry = parent
                .class_get(variant)
                .unwrap_or_else(|| panic!("variant {variant} bound in parent class dict"));
            let child = match entry {
                ConstValue::HostObject(h) => h,
                other => panic!("variant carrier must be HostObject, got {other:?}"),
            };
            assert!(child.is_class(), "variant {variant} must be a class");
            assert!(
                child.is_subclass_of(&parent),
                "variant {variant} must be a subclass of the parent enum class \
                 (matches upstream `class V(Parent): pass` shape)",
            );
            assert_eq!(
                child.qualname(),
                format!("ParityProbeEnum_Slice_O8.{variant}")
            );
        }
    }

    #[test]
    fn register_rust_module_walks_item_struct_with_empty_class_dict() {
        // Rust struct field access `instance.x` reads from the
        // instance, not the class object — upstream `class Foo: pass`
        // likewise leaves `Foo.__dict__` empty for instance
        // attributes. The walker's struct dispatch produces the
        // identity-carrier class with an empty class dict.

        let src = "pub struct ParityProbeStruct_Slice_O8 { x: i64, y: i64 }";
        let file = syn::parse_file(src).expect("struct fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let host = lookup_host(module_id, "ParityProbeStruct_Slice_O8")
            .expect("struct registered after walk");
        assert!(host.is_class());
        assert_eq!(host.qualname(), "ParityProbeStruct_Slice_O8");
        // No instance fields populate the class dict — they live on
        // instances. `class_dict_keys()` returns the empty set.
        assert!(
            host.class_dict_keys().is_empty(),
            "struct class dict must be empty (instance fields belong on instances), \
             got keys: {:?}",
            host.class_dict_keys(),
        );
    }

    #[test]
    fn build_host_function_from_rust_file_returns_walker_built_entry_audit_1_2() {
        // Audit 1.2 (2026-05-08): the walker's pass-2 builds the
        // entry-point fn alongside its siblings; `build_host_function_from_rust_file`
        // now reuses that walker-built `HostObject` instead of
        // re-building. The returned `host` must be the same Arc-
        // identity as the one registered into `module.__dict__`,
        // mirroring upstream Python `module.__dict__[entry] is
        // caller.entry_point`.
        let src = "fn parity_probe_audit_1_2_entry() -> i64 { 42 }";
        let file = syn::parse_file(src).expect("audit-1.2 fixture parses");
        let (host, pygraph) = build_host_function_from_rust_file(
            &file,
            "parity_probe_audit_1_2_entry",
            Some("/parity_probe/audit_1_2_entry.rs"),
            Some(src),
        )
        .expect("walker entry succeeds");

        // Walker-built host lives in the registry under the entry name.
        // For path-keyed modules, the id derives deterministically from
        // the path so we can re-derive it for the lookup.
        let module_id = ModuleId::for_path("/parity_probe/audit_1_2_entry.rs");
        let walker_host = match module_globals_lookup(module_id, "parity_probe_audit_1_2_entry") {
            Some(ConstValue::HostObject(h)) => h,
            other => panic!("expected walker-registered host, got {other:?}"),
        };
        assert_eq!(
            host, walker_host,
            "audit 1.2: returned host must be the same Arc-identity as \
             `module.__dict__[entry]`; pre-fix this was a separate post-walk build"
        );

        // The pygraph must come from the walker registry too —
        // identity match against `lookup_walker_pygraph(host)`.
        let walker_pygraph =
            lookup_walker_pygraph(&walker_host).expect("walker pygraph pinned for entry");
        assert!(
            Rc::ptr_eq(&pygraph, &walker_pygraph),
            "audit 1.2: returned pygraph must be the walker-registered Rc<PyGraph>"
        );

        // GraphFunc still reads the caller's source pair (audit 1.2 is
        // about identity, not metadata loss).
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_filename, "/parity_probe/audit_1_2_entry.rs");
        assert_eq!(gf.source.as_deref(), Some(src));
    }

    #[test]
    fn graph_func_globals_reflects_module_globals_partition_after_walk() {
        // Issue 1 (2026-05-05): `GraphFunc.globals` must surface
        // the module-globals registry slice for the active
        // `ModuleId`, not an empty dict. Mirrors upstream
        // `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
        // — `func.__globals__` is the function's owning module's
        // `__dict__`, whose entries the walker has just bound.
        //
        // Path-keyed walk (Issue 2 share) so the body lowering uses
        // the same id the walker registered under. The fixture must
        // be lowerable by the current adapter subset — `fn entry()
        // -> i64 { 1 }` is enough; what we're verifying is the
        // GraphFunc-side carrier, not body resolution.
        let src = "pub struct ParityProbe_Issue1_sibling;
                   pub const ParityProbe_Issue1_const: i64 = 7;
                   fn parity_probe_issue1_entry() -> i64 { 1 }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let (host, _pygraph) = build_host_function_from_rust_file(
            &file,
            "parity_probe_issue1_entry",
            Some("/parity_probe/issue1_globals.rs"),
            None,
        )
        .expect("file-aware entry succeeds");
        let gf = host.user_function().expect("user function");
        // GraphFunc.globals is a Constant; the inner ConstValue must
        // be Dict carrying the registered struct + const.
        let dict = match &gf.globals.value {
            ConstValue::Dict(items) => items,
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };
        let struct_key = ConstValue::byte_str(b"ParityProbe_Issue1_sibling");
        let const_key = ConstValue::byte_str(b"ParityProbe_Issue1_const");
        assert!(
            dict.contains_key(&struct_key),
            "module-globals dict must contain registered struct, got keys: {:?}",
            dict.keys().collect::<Vec<_>>(),
        );
        assert_eq!(
            dict.get(&const_key),
            Some(&ConstValue::Int(7)),
            "module-globals dict must contain registered Item::Const value",
        );
    }

    #[test]
    fn graph_func_live_globals_reflects_post_construction_walker_writes() {
        // Issue 2.1 (2026-05-05): `func.__globals__` must surface
        // entries added to the registry partition AFTER `GraphFunc`
        // construction, mirroring upstream `flowcontext.py:284
        // self.w_globals = Constant(func.__globals__)` where
        // `func.__globals__` is a *live* reference to the module
        // dict. `live_globals()` re-snapshots the partition keyed
        // on `module_globals_id`.
        //
        // Construction sequence: build entry-point first (snapshot
        // taken with only items the walker has registered so far),
        // then register additional items into the same `module_id`.
        // The static `gf.globals.value` snapshot stays stale; the
        // `live_globals()` accessor returns the post-write state.
        use super::super::host_env::register_module_global;
        let path = "/parity_probe/issue21_live_globals.rs";
        let src = "fn parity_probe_issue21_entry() -> i64 { 1 }";
        let file = syn::parse_file(src).expect("entry-point fixture parses");
        let (host, _pygraph) = build_host_function_from_rust_file(
            &file,
            "parity_probe_issue21_entry",
            Some(path),
            None,
        )
        .expect("file-aware entry succeeds");
        let gf = host.user_function().expect("user function");
        let module_id = gf.module_globals_id.expect("rust-source built sets the id");

        // Static snapshot is the dict at construction time — empty
        // beyond what the walker pre-pass wrote. Sanity-check the
        // construction-time invariant first.
        let static_dict = match &gf.globals.value {
            ConstValue::Dict(items) => items.clone(),
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };

        // Now register a fresh entry into the same partition,
        // simulating either a follow-up walker pass or a deferred
        // `Item::*` registration. Upstream Python: a later
        // `module.NEW_NAME = ...` assignment would be visible to
        // any subsequent `func.__globals__["NEW_NAME"]` read.
        register_module_global(
            module_id,
            "ParityProbe_Issue21_late_addition",
            ConstValue::Int(99),
        );

        // Static snapshot is unchanged (snapshot semantic).
        let static_dict_after = match &gf.globals.value {
            ConstValue::Dict(items) => items.clone(),
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };
        assert_eq!(
            static_dict, static_dict_after,
            "static `gf.globals.value` snapshot stays frozen at construction time",
        );

        // Live snapshot via `live_globals()` reflects the new entry.
        let live = gf.live_globals();
        let live_dict = match &live {
            ConstValue::Dict(items) => items,
            other => panic!("expected live_globals() Dict, got {other:?}"),
        };
        assert_eq!(
            live_dict.get(&ConstValue::byte_str(b"ParityProbe_Issue21_late_addition")),
            Some(&ConstValue::Int(99)),
            "live_globals must surface entries added to the partition \
             post-construction (upstream `func.__globals__` is a live ref)",
        );
    }

    #[test]
    fn register_rust_module_at_with_same_path_returns_same_id() {
        // Issue 2 (2026-05-05): two walks of the same source path
        // converge on the same `ModuleId` — mirrors upstream
        // `sys.modules[path]` import-cache identity. The second
        // walk's registrations clobber the first under
        // last-writer-wins (`module.__dict__[name] = value`
        // semantics) — the cross-id lookup observes the second
        // walk's binding regardless of whether file1 / file2 are
        // identical or distinct.

        let src = "pub struct ParityProbe_Issue2_path_share;";
        let file1 = syn::parse_file(src).expect("file 1 parses");
        let file2 = syn::parse_file(src).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_share.rs"))
            .expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_share.rs"))
            .expect("walker must succeed");
        assert_eq!(
            id1, id2,
            "same path must yield the same ModuleId (sys.modules cache parity)",
        );
        let host = lookup_host(id1, "ParityProbe_Issue2_path_share")
            .expect("walk's binding visible from shared id");
        let cross = lookup_host(id2, "ParityProbe_Issue2_path_share").unwrap();
        assert_eq!(
            host, cross,
            "shared id must serve identical HostObject identity across walks \
             (last-writer-wins: the second walk's HostObject is what both ids see)",
        );
    }

    #[test]
    fn register_rust_module_at_with_distinct_paths_isolates_partitions() {
        // Issue 2 (2026-05-05): different paths mint distinct ids,
        // matching upstream's per-module `__dict__` isolation. Two
        // modules at distinct paths binding the same top-level name
        // see independent values — the cross-id lookup misses.

        let src1 = "pub struct ParityProbe_Issue2_path_distinct;";
        let src2 = "pub struct ParityProbe_Issue2_path_distinct { x: i64 }";
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_distinct_a.rs"))
            .expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_distinct_b.rs"))
            .expect("walker must succeed");
        assert_ne!(id1, id2, "distinct paths must mint distinct ids");
        let host1 = lookup_host(id1, "ParityProbe_Issue2_path_distinct").unwrap();
        let host2 = lookup_host(id2, "ParityProbe_Issue2_path_distinct").unwrap();
        assert_ne!(
            host1, host2,
            "distinct-path walks must produce independent class identities",
        );
    }

    #[test]
    fn register_rust_module_at_with_none_path_mints_fresh_per_call() {
        // Codex parity revert (2026-05-05): the `None` branch was
        // previously content-hashed (Issue 2.2), but that merged
        // distinct anonymous walks of identical source whenever the
        // bytes matched — diverging from upstream Python's
        // `exec(source, dict_a)` / `exec(source, dict_b)` semantic
        // (each `exec` runs against an independent `__dict__`). The
        // fix mints a fresh ModuleId per anonymous call. Callers
        // who need shared identity opt in via `Some(path)`.
        let src = "pub struct ParityProbe_Anonymous_FreshPerCall;";
        let file = syn::parse_file(src).expect("anonymous walk parses");
        let id1 = register_rust_module_at(&file, None).expect("walker must succeed");
        let id2 = register_rust_module_at(&file, None).expect("walker must succeed");
        assert_ne!(
            id1, id2,
            "anonymous walks (None path) MUST mint independent ModuleIds even \
             when the source bytes are identical — content equality does not \
             imply module identity in upstream Python",
        );
    }

    #[test]
    fn register_rust_module_at_with_none_path_distinguishes_distinct_content() {
        // Anonymous walks of distinct content produce distinct ids
        // (trivially follows from "every None call mints fresh").
        // Kept as a separate oracle so future regressions to
        // content-hash sharing fail this test too.
        let src1 = "pub struct ParityProbe_Anon_Distinct_A;";
        let src2 = "pub struct ParityProbe_Anon_Distinct_B;";
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, None).expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, None).expect("walker must succeed");
        assert_ne!(id1, id2, "distinct anonymous walks mint distinct ModuleIds",);
    }

    #[test]
    fn register_rust_module_isolates_distinct_walks_with_shared_top_level_name() {
        // Per-module scoping (Issue 1.3, 2026-05-05): two walks of
        // files containing the same top-level name now produce
        // INDEPENDENT registry partitions, not a shared first-writer
        // entry. Mirrors upstream `flowcontext.py:284 self.w_globals
        // = Constant(func.__globals__)` per-module scoping — two
        // distinct modules with identically-named top-level
        // bindings see independent values.
        //
        // (The pre-Issue-1.3 cross-walk first-writer-wins test was
        // a workaround for the missing per-module scoping; it no
        // longer applies once the registry partitions properly.)

        let src1 = "pub struct ParityProbeStruct_isolate;";
        let src2 = "pub struct ParityProbeStruct_isolate { x: i64 }"; // distinct shape, same name
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module(&file1).expect("walker must succeed");
        let id2 = register_rust_module(&file2).expect("walker must succeed");
        assert_ne!(id1, id2, "fresh walks must produce distinct ModuleIds");
        let host1 = lookup_host(id1, "ParityProbeStruct_isolate")
            .expect("file 1's binding visible from id1");
        let host2 = lookup_host(id2, "ParityProbeStruct_isolate")
            .expect("file 2's binding visible from id2");
        assert_ne!(
            host1, host2,
            "isolated walks must NOT share class identity \
             (each file mints its own HostObject under its own ModuleId)",
        );
        // Cross-id lookup remains scoped to its own partition: id1
        // does not see id2's struct shape, even though the names
        // are identical.
        let cross = lookup_host(id2, "ParityProbeStruct_isolate").unwrap();
        assert_eq!(cross, host2, "id2's lookup returns id2's binding");
        assert_ne!(cross, host1, "id2's lookup must NOT see id1's binding");
    }

    #[test]
    fn register_rust_module_last_writer_wins_under_same_module_id() {
        // Last-writer-wins under a fixed `ModuleId` mirrors upstream
        // `module.__dict__[name] = value` semantics: every top-level
        // binding statement is an unconditional assignment. Within a
        // single `register_rust_module` call Rust syntax does not
        // allow duplicate top-level item names, so the observable
        // effect is across walks of the same path-keyed module — the
        // second walk's bindings supersede the first.
        //
        // We can't feed two `pub struct Foo;` to syn::parse_file (it
        // errors on duplicate items), so we exercise
        // `register_module_global` directly under a fixed id to
        // model the cross-walk re-binding.
        let id = ModuleId::fresh();
        let first = HostObject::new_class("ParityProbeStruct_within_walk", vec![]);
        let second = HostObject::new_class("ParityProbeStruct_within_walk", vec![]);
        super::super::host_env::register_module_global(
            id,
            "ParityProbeStruct_within_walk",
            ConstValue::HostObject(first.clone()),
        );
        super::super::host_env::register_module_global(
            id,
            "ParityProbeStruct_within_walk",
            ConstValue::HostObject(second.clone()),
        );
        let observed = lookup_host(id, "ParityProbeStruct_within_walk").unwrap();
        assert_eq!(observed, second, "last registration wins within same id");
        assert_ne!(
            observed, first,
            "first registration must be clobbered by the second under same id",
        );
    }

    // ---- Slice O10 — Item::Const walker dispatch -----------------

    #[test]
    fn register_rust_module_walks_item_const_integer_literal() {
        // upstream Python `MODULE.MAX_SIZE` reads
        // `module.__dict__["MAX_SIZE"]` which holds the int the
        // top-level assignment bound. The walker mirrors this for
        // `const MAX_SIZE: i64 = 42` — registers the integer value
        // directly as `ConstValue::Int(42)`.
        let src = "pub const ParityProbe_O10_const_int: i64 = 42;";
        let file = syn::parse_file(src).expect("const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_int")
            .expect("integer const registered after walk");
        assert_eq!(value, ConstValue::Int(42));
    }

    #[test]
    fn register_rust_module_walks_item_const_negative_integer_literal() {
        // `const X: i64 = -7` parses through `Expr::Unary { op:
        // Neg, expr: Lit(7) }` — the walker must unwrap one level
        // of unary minus to recognise the signed-int form.
        let src = "pub const ParityProbe_O10_const_neg_int: i64 = -7;";
        let file = syn::parse_file(src).expect("negated const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_neg_int")
            .expect("negated integer const registered after walk");
        assert_eq!(value, ConstValue::Int(-7));
    }

    #[test]
    fn register_rust_module_walks_item_const_bool_literal() {
        let src = "pub const ParityProbe_O10_const_bool: bool = true;";
        let file = syn::parse_file(src).expect("const bool parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_bool")
            .expect("bool const registered after walk");
        assert_eq!(value, ConstValue::Bool(true));
    }

    #[test]
    fn register_rust_module_walks_item_const_str_literal() {
        // `Lit::Str` lowers to `ConstValue::UniStr` matching
        // `build_flow.rs::lower_literal::Lit::Str` and Python 3
        // unicode-string semantics. The same `"abc"` literal at
        // body position would lower identically — no shape drift
        // between expression and module-const positions.
        let src = "pub const ParityProbe_O10_const_str: &str = \"abc\";";
        let file = syn::parse_file(src).expect("const str parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_str")
            .expect("str const registered after walk");
        assert_eq!(value, ConstValue::uni_str("abc"));
    }

    #[test]
    fn register_rust_module_walks_item_const_compound_expression() {
        // Issue 4 (2026-05-05): the walker resolves compound const
        // RHS expressions through prior bindings in the same source-
        // order walk. Mirrors upstream Python module-import: by the
        // time `Y = X + 1` runs at top level, `X = 1` has already
        // bound `module.__dict__["X"]`, and the binary op evaluates
        // `module.__dict__["X"] + 1` against that.
        let src = "pub const ParityProbe_Issue4_const_X: i64 = 1;
                   pub const ParityProbe_Issue4_const_Y: i64 = ParityProbe_Issue4_const_X + 1;
                   pub const ParityProbe_Issue4_const_Z: i64 = ParityProbe_Issue4_const_Y * 3;
                   pub const ParityProbe_Issue4_const_NEG: i64 = -ParityProbe_Issue4_const_Z;";
        let file = syn::parse_file(src).expect("compound const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_X"),
            Some(ConstValue::Int(1)),
            "X registers as literal Int(1)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_Y"),
            Some(ConstValue::Int(2)),
            "Y registers as X + 1 = 2 via prior-bindings env",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_Z"),
            Some(ConstValue::Int(6)),
            "Z registers as Y * 3 = 6 via prior-bindings env",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_NEG"),
            Some(ConstValue::Int(-6)),
            "NEG registers as -Z = -6 (unary neg over evaluated path)",
        );
    }

    #[test]
    fn register_rust_module_walks_compound_const_with_shift_and_bitwise() {
        // Issue 2.4 (2026-05-05): `eval_const_expr` covers shifts
        // and bitwise ops over Int operands. Mirrors upstream
        // Python import-time evaluation: `MASK = 1 << 4 | 0x3` would
        // bind `module.__dict__["MASK"]` to the evaluated integer.
        let src = "pub const ParityProbe_Issue24_BASE: i64 = 1;
                   pub const ParityProbe_Issue24_SHIFT: i64 = ParityProbe_Issue24_BASE << 4;
                   pub const ParityProbe_Issue24_RSHIFT: i64 = 64 >> 2;
                   pub const ParityProbe_Issue24_AND: i64 = 0x1F & 0x07;
                   pub const ParityProbe_Issue24_OR: i64 = 0x10 | 0x01;
                   pub const ParityProbe_Issue24_XOR: i64 = 0x0F ^ 0x05;
                   pub const ParityProbe_Issue24_NOT: i64 = !0;";
        let file = syn::parse_file(src).expect("shift+bitwise fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_SHIFT"),
            Some(ConstValue::Int(16)),
            "1 << 4 = 16",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_RSHIFT"),
            Some(ConstValue::Int(16)),
            "64 >> 2 = 16",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_AND"),
            Some(ConstValue::Int(0x07)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_OR"),
            Some(ConstValue::Int(0x11)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_XOR"),
            Some(ConstValue::Int(0x0A)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_NOT"),
            Some(ConstValue::Int(!0_i64)),
            "Rust `!` on Int is bitwise complement (Python `~`)",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_shift_parity() {
        // `<<` / `>>` over int operands must match upstream
        // `operator.lshift` / `operator.rshift` semantics so a
        // module-top `const X = 1 >> 64` and a body `1 >> 64`
        // produce the same value.
        //
        // - **Negative shift count** raises `ValueError: negative
        //   shift count` upstream — propagate as `Err` (walker
        //   aborts the file).
        // - **`<<` count >= 64** declines per the
        //   `can_overflow and type(result) is long` arm
        //   (`operation.py:140-142`): the bignum result doesn't
        //   fit in `i64`, walker leaves the const unbound.
        // - **`>>` count >= 64** saturates: `1 >> 64 == 0`,
        //   `-1 >> 64 == -1`. RShift is not in the `can_overflow`
        //   set so the long-decline arm doesn't apply.
        let neg_shl = "pub const X: i64 = 1 << -1;";
        let file = syn::parse_file(neg_shl).expect("negative shl fixture parses");
        let err =
            register_rust_module(&file).expect_err("negative shift count must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ValueError") && reason.contains("negative shift count"),
                "expected ValueError on negative shift, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
        let neg_shr = "pub const Y: i64 = 1 >> -1;";
        let file = syn::parse_file(neg_shr).expect("negative shr fixture parses");
        register_rust_module(&file).expect_err("negative shr count must abort the walker");
        // Large positive shr — saturating fold.
        let big_shr = "pub const ParityProbe_BigShr_pos: i64 = 1 >> 64;
                       pub const ParityProbe_BigShr_neg: i64 = -1 >> 64;
                       pub const ParityProbe_BigShr_100: i64 = 100 >> 128;";
        let file = syn::parse_file(big_shr).expect("big shr fixture parses");
        let module_id = register_rust_module(&file).expect("big positive shr saturates, no abort");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_pos"),
            Some(ConstValue::Int(0)),
            "1 >> 64 == 0 (saturating non-negative)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_neg"),
            Some(ConstValue::Int(-1)),
            "-1 >> 64 == -1 (sign-extending)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_100"),
            Some(ConstValue::Int(0)),
        );
        // Large positive shl — unsupported bignum carrier at module top level.
        // Upstream Python `1 << 64` binds bignum `2**64`; pyre
        // adapter has no bignum carrier so surfaces as
        // `AdapterError::Unsupported`. The function-body constfold path
        // (`operation.py:140-142`) declines with Ok(None) instead,
        // but THAT path is at function evaluation time where the
        // runtime can produce the bignum — there is no analogous
        // path at module-import-time prepass.
        let big_shl = "pub const ParityProbe_BigShl: i64 = 1 << 64;";
        let file = syn::parse_file(big_shl).expect("big shl fixture parses");
        let err = register_rust_module(&file)
            .expect_err("1 << 64 requires unsupported bignum carrier at module top level");
        match err {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("bignum") && reason.contains("i64"),
                "expected bignum unsupported error on 1 << 64, got: {reason}",
            ),
            other => panic!("expected AdapterError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mod_minus_one() {
        // `x % -1 == 0` for every `x` upstream. The `int_py_mod`
        // closure short-circuits `y == -1` so that
        // `i64::MIN.checked_rem(-1)` (which Rust returns `None`
        // for, because the *quotient* `2^63` overflows even though
        // the remainder `0` itself fits) does not silently decline
        // a well-defined fold. Source `-9223372036854775808` lies
        // outside what `syn` can parse as a single `Neg(Lit)`
        // (the inner literal `9223372036854775808` is `i64::MAX +
        // 1`), so the operation.rs `int_py_mod`-level test pins
        // the `MIN` boundary directly. Here we pin the
        // representable-source cases.
        let src = "pub const ParityProbe_ModMinusOne_pos: i64 = 7 % -1;
                   pub const ParityProbe_ModMinusOne_neg: i64 = -7 % -1;
                   pub const ParityProbe_ModMinusOne_max: i64 = 9223372036854775807 % -1;";
        let file = syn::parse_file(src).expect("x % -1 fixture parses");
        let module_id = register_rust_module(&file).expect("x % -1 folds, no abort");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_pos"),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_neg"),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_max"),
            Some(ConstValue::Int(0)),
            "i64::MAX % -1 == 0 (no overflow path)",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_uses_python_floor_div_and_mod() {
        // The const evaluator's `Div` / `Rem` arms must agree with
        // `flowspace::operation`'s `int_py_floor_div` / `int_py_mod`
        // over negative-divisor pairs so a `const Y: i64 = 3 / -2`
        // at module top level produces the same value the function-
        // body `3 / -2` would after constfold. C-style truncation
        // (`a.checked_div(b)`) gives `-1` for `3 / -2` and `1` for
        // `3 % -2`, which would diverge from the body result.
        // Python floor-div toward `-inf` gives `-2` and `-1`
        // respectively (line-by-line port of
        // `rpython/rtyper/rint.py:398 ll_int_py_div`,
        // `:496 ll_int_py_mod`).
        let src = "pub const ParityProbe_FloorDiv_pos_div_neg: i64 = 3 / -2;
                   pub const ParityProbe_FloorMod_pos_mod_neg: i64 = 3 % -2;
                   pub const ParityProbe_FloorDiv_neg_div_pos: i64 = -3 / 2;
                   pub const ParityProbe_FloorMod_neg_mod_pos: i64 = -3 % 2;
                   pub const ParityProbe_FloorDiv_neg_div_neg: i64 = -3 / -2;
                   pub const ParityProbe_FloorMod_neg_mod_neg: i64 = -3 % -2;";
        let file = syn::parse_file(src).expect("floor-div fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_pos_div_neg"),
            Some(ConstValue::Int(-2)),
            "3 // -2 == -2 (Python floor toward -inf, not C trunc -1)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_pos_mod_neg"),
            Some(ConstValue::Int(-1)),
            "3 % -2 == -1 (sign matches divisor)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_neg_div_pos"),
            Some(ConstValue::Int(-2)),
            "-3 // 2 == -2",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_neg_mod_pos"),
            Some(ConstValue::Int(1)),
            "-3 % 2 == 1",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_neg_div_neg"),
            Some(ConstValue::Int(1)),
            "-3 // -2 == 1",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_neg_mod_neg"),
            Some(ConstValue::Int(-1)),
            "-3 % -2 == -1",
        );
    }

    #[test]
    fn register_rust_module_walks_compound_const_with_comparison_and_bool() {
        // Issue 2.4 (2026-05-05): comparison ops over Int / Bool /
        // strings produce Bool; `&&`/`||` over Bool stay Bool.
        // Mirrors Python's `MAX_NEG = MAX < 0` / `BOTH = A && B`
        // import-time evaluation.
        let src = "pub const ParityProbe_Issue24_MAX: i64 = 100;
                   pub const ParityProbe_Issue24_IS_BIG: bool = ParityProbe_Issue24_MAX > 50;
                   pub const ParityProbe_Issue24_EQ: bool = ParityProbe_Issue24_MAX == 100;
                   pub const ParityProbe_Issue24_AND_BOOL: bool = true && false;
                   pub const ParityProbe_Issue24_OR_BOOL: bool = false || true;
                   pub const ParityProbe_Issue24_NOT_BOOL: bool = !false;";
        let file = syn::parse_file(src).expect("comparison fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_IS_BIG"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_EQ"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_AND_BOOL"),
            Some(ConstValue::Bool(false)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_OR_BOOL"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_NOT_BOOL"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_short_circuits_and_or_at_compile_time() {
        // Codex parity audit (2026-05-05): both Rust source and
        // upstream Python's `and`/`or` are short-circuit.
        // `false && BAD` evaluates to `false` without ever touching
        // `BAD`; `true || BAD` evaluates to `true`. Without
        // short-circuit, the RHS reference to an unbound name would
        // surface as `NameError` and abort the walker.
        let src = "pub const ParityProbe_ShortCircuit_AND: bool = false && ParityProbe_UnboundRhs;
                   pub const ParityProbe_ShortCircuit_OR: bool = true || ParityProbe_UnboundRhs;";
        let file = syn::parse_file(src).expect("short-circuit fixture parses");
        let module_id = register_rust_module(&file).expect(
            "short-circuit must skip RHS evaluation — unbound name in RHS \
             does not abort the walker when LHS already determines result",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_AND"),
            Some(ConstValue::Bool(false)),
            "`false && X` short-circuits to false without evaluating X",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_OR"),
            Some(ConstValue::Bool(true)),
            "`true || X` short-circuits to true without evaluating X",
        );
    }

    #[test]
    fn register_rust_module_evaluates_rhs_when_lhs_does_not_short_circuit() {
        // Negative-direction oracle for short-circuit: when LHS does
        // NOT determine the result, RHS must be evaluated. `true &&
        // X` and `false || X` both depend on X.
        let src = "pub const ParityProbe_ShortCircuit_RHS_AND: bool = true && false;
                   pub const ParityProbe_ShortCircuit_RHS_OR: bool = false || true;";
        let file = syn::parse_file(src).expect("non-short-circuit fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_RHS_AND"),
            Some(ConstValue::Bool(false)),
            "`true && false` reads the RHS",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_RHS_OR"),
            Some(ConstValue::Bool(true)),
            "`false || true` reads the RHS",
        );
    }

    #[test]
    fn register_rust_module_aborts_when_short_circuit_does_not_apply() {
        // Without LHS short-circuit, the RHS is evaluated; an
        // unresolved single-segment Path raises NameError per
        // upstream `flowcontext.py:853 find_global`'s
        // `FlowingError("global name '...' is not defined")`.
        // Walker aborts the file matching upstream Python's
        // module-execution-aborts-on-exception semantic.
        let src = "pub const ParityProbe_ShortCircuit_NoShort_AND: bool = true && ParityProbe_UnboundRhs;";
        let file = syn::parse_file(src).expect("non-short-circuit-failure fixture parses");
        let err = register_rust_module(&file)
            .expect_err("unbound RHS at module top must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_UnboundRhs' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_aborts_on_overflow_const() {
        // `i64::MAX + 1` at module top level. Upstream Python would
        // bind a bignum `2**63` without raising; pyre adapter has no
        // bignum carrier so surfaces as `AdapterError::Unsupported`.
        // Walker aborts the file. (NOT the function-body
        // `PureOperation.constfold` decline path at
        // `operation.py:140-142` — that path lives at function
        // evaluation time where the runtime can produce the bignum.)
        let src = "pub const ParityProbe_Issue24_OVERFLOW: i64 = 9223372036854775807 + 1;";
        let file = syn::parse_file(src).expect("overflow fixture parses");
        let err = register_rust_module(&file)
            .expect_err("i64 overflow at module top requires unsupported bignum carrier");
        match err {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("bignum") && reason.contains("i64"),
                "expected bignum unsupported error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_rewalks_path_keyed_module_resolves_via_registry() {
        // Issue 2.3 (2026-05-05): when a path-keyed `module_id` is
        // re-walked (e.g. two `Translation::from_rust_file_entry_point_with_source`
        // calls against the same source file), the second walk's
        // local `const_bindings` is fresh-empty. `eval_const_expr`
        // must fall back to `module_globals_lookup(module_id, ...)`
        // to surface the registered prior binding — mirrors upstream
        // `module.__dict__` being the live reference visible across
        // re-imports.
        //
        // Walk 1 binds X. Walk 2 has only Y = X + 1; Y must resolve.
        let path = "issue_2_3_const_rewalk_fixture.rs";
        let src1 = "pub const ParityProbe_Issue23_X: i64 = 7;";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(id1, "ParityProbe_Issue23_X"),
            Some(ConstValue::Int(7)),
        );
        // Walk 2 — same path → same module_id. New const Y references X.
        let src2 = "pub const ParityProbe_Issue23_Y: i64 = ParityProbe_Issue23_X + 1;";
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        assert_eq!(
            module_globals_lookup(id2, "ParityProbe_Issue23_Y"),
            Some(ConstValue::Int(8)),
            "Y = X + 1 resolves X via registry fallback (X bound in prior walk, \
             absent from this walk's local const_bindings)",
        );
    }

    #[test]
    fn register_rust_module_path_keyed_rewalk_is_last_writer_wins() {
        // Codex parity audit (2026-05-05): a path-keyed re-walk that
        // re-binds the SAME name MUST clobber the prior entry, not
        // skip it. Mirrors upstream `module.__dict__[name] = value`:
        // every top-level binding statement is an unconditional
        // assignment (`exec(source, dict)` / `importlib.reload`
        // semantics). The pre-fix walker silently dropped the second
        // walk's binding under first-writer-wins, diverging from
        // Python module-execution semantics.
        let path = "issue6_lastwriter_rewalk_fixture.rs";
        let src1 = "pub const ParityProbe_LastWriter_X: i64 = 1;";
        let src2 = "pub const ParityProbe_LastWriter_X: i64 = 2;";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(id1, "ParityProbe_LastWriter_X"),
            Some(ConstValue::Int(1)),
            "walk 1 binds X = 1",
        );
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        assert_eq!(
            module_globals_lookup(id2, "ParityProbe_LastWriter_X"),
            Some(ConstValue::Int(2)),
            "walk 2 last-writer-wins: X is now 2 (was 1 from walk 1)",
        );
    }

    #[test]
    fn register_rust_module_path_keyed_rewalk_overwrites_struct_class_identity() {
        // Same last-writer-wins applies to `Item::Struct` / `Item::Enum`
        // class registrations: a path-keyed re-walk re-mints the
        // `HostObject::Class` carrier and overwrites the prior entry,
        // mirroring upstream `module.__dict__[ClassName] = <new
        // class>` after re-executing the module body.
        let path = "issue6_lastwriter_class_rewalk_fixture.rs";
        let src1 = "pub struct ParityProbe_LastWriter_Cls;";
        let src2 = "pub struct ParityProbe_LastWriter_Cls { x: i64 }";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        let host1 = lookup_host(id1, "ParityProbe_LastWriter_Cls").expect("walk 1 binds class");
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        let host2 = lookup_host(id2, "ParityProbe_LastWriter_Cls").expect("walk 2 binds class");
        assert_ne!(
            host1, host2,
            "walk 2 mints a fresh HostObject::Class carrier and clobbers walk 1's binding",
        );
    }

    #[test]
    fn register_rust_module_eq_ne_mixed_types_returns_bool_not_typeerror() {
        // Codex parity audit (2026-05-05): Python 3 `==` / `!=` over
        // distinct primitive types do NOT raise — `1 == "1"` returns
        // `False`, `1 != "1"` returns `True`. Only ordering
        // (`<`, `<=`, `>`, `>=`) raises `TypeError`. The walker's
        // `eval_binop` previously routed every mixed-type pair to
        // `TypeError` via the catch-all, mis-aborting valid module
        // top-level constants like `pub const X: bool = 1 == "1";`.
        //
        // Bare-binop and paren-wrapped fixtures must agree per
        // `Expr::Paren` transparency — `(1 == "a")` and `1 == "a"`
        // produce the same binding (matches `build_flow.rs:1317`).
        let src = "pub const ParityProbe_Issue3_eq_int_str: bool = 1 == \"a\";
                   pub const ParityProbe_Issue3_ne_int_str: bool = 1 != \"a\";";
        let file = syn::parse_file(src).expect("eq-mixed fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            super::super::host_env::module_globals_lookup(
                module_id,
                "ParityProbe_Issue3_eq_int_str"
            ),
            Some(ConstValue::Bool(false)),
            "Python 3: `1 == \"a\"` is False, NOT TypeError",
        );
        assert_eq!(
            super::super::host_env::module_globals_lookup(
                module_id,
                "ParityProbe_Issue3_ne_int_str"
            ),
            Some(ConstValue::Bool(true)),
            "Python 3: `1 != \"a\"` is True, NOT TypeError",
        );
        // Ordering on mixed types DOES raise TypeError — that arm of
        // `eval_binop` must continue to abort the walker.
        let bad_src = "pub const ParityProbe_Issue3_lt_mixed: bool = 1 < \"a\";";
        let bad_file = syn::parse_file(bad_src).expect("lt-mixed fixture parses");
        let err = register_rust_module(&bad_file)
            .expect_err("Python 3 `1 < \"a\"` raises TypeError — walker must abort");
        match err {
            AdapterError::Flowing { reason } => {
                assert!(
                    reason.contains("TypeError"),
                    "expected TypeError on mixed-type ordering, got: {reason}",
                );
            }
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_paren_is_transparent() {
        // Upstream Python parens evaporate at parse time, so
        // `(1 + 2)` and `1 + 2` produce identical bindings. `syn`
        // surfaces `Expr::Paren` and `Expr::Group` as wrapper
        // nodes; `eval_const_expr` must delegate transparently
        // to mirror `build_flow.rs:1317
        // Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr)`.
        let src = "pub const ParityProbe_Paren_BareSum: i64 = 1 + 2;
                   pub const ParityProbe_Paren_WrappedSum: i64 = (1 + 2);
                   pub const ParityProbe_Paren_NestedWrap: i64 = (((1 + 2)));
                   pub const ParityProbe_Paren_WrappedEq: bool = (1 == 1);";
        let file = syn::parse_file(src).expect("paren-transparency fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let lookup = |name: &str| super::super::host_env::module_globals_lookup(module_id, name);
        assert_eq!(
            lookup("ParityProbe_Paren_BareSum"),
            Some(ConstValue::Int(3))
        );
        assert_eq!(
            lookup("ParityProbe_Paren_WrappedSum"),
            Some(ConstValue::Int(3)),
            "`(1 + 2)` must fold identically to `1 + 2`",
        );
        assert_eq!(
            lookup("ParityProbe_Paren_NestedWrap"),
            Some(ConstValue::Int(3)),
            "nested parens are transparent at every level",
        );
        assert_eq!(
            lookup("ParityProbe_Paren_WrappedEq"),
            Some(ConstValue::Bool(true)),
            "paren around comparison stays transparent",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_path_falls_back_to_builtins() {
        // Upstream `flowcontext.py:845-853 find_global`: globals miss
        // → `getattr(__builtin__, varname)` → NameError on builtins
        // miss. pyre's adapter has `pyre_stdlib_lookup` as the
        // closed-world `__builtin__` analogue (`Ok` / `Some` / `Err` /
        // `Result` / `Option`). The const evaluator must consult it
        // before raising NameError, so a top-level
        // `pub const X: Result = Ok;` (where `Ok` is not in walker
        // bindings or registry) resolves through builtins.
        let src = "pub const ParityProbe_Builtins_Ok: i64 = 0;
                   pub const ParityProbe_Builtins_Forward: bool = Ok == Ok;";
        let file = syn::parse_file(src).expect("builtins-fallback fixture parses");
        let module_id = register_rust_module(&file).expect(
            "walker must succeed — `Ok` resolves through pyre_stdlib_lookup, not NameError",
        );
        // The eq comparison succeeds because both sides resolve to
        // the same `HostObject(Ok)` singleton (PYRE_STDLIB returns
        // identical `HostObject` clones across lookups; structural
        // equality holds).
        let v = super::super::host_env::module_globals_lookup(
            module_id,
            "ParityProbe_Builtins_Forward",
        );
        assert_eq!(
            v,
            Some(ConstValue::Bool(true)),
            "`Ok == Ok` must fold to True via builtins fallback (Ok singleton)",
        );

        // Negative control: a bare identifier that is neither in
        // bindings nor in the registry nor in PYRE_STDLIB still
        // raises NameError per upstream's final
        // AttributeError → FlowingError step.
        let bad_src = "pub const ParityProbe_Builtins_NotABuiltin: i64 =
            ParityProbe_Builtins_Unbound + 1;";
        let bad_file = syn::parse_file(bad_src).expect("bad fixture parses");
        let err = register_rust_module(&bad_file)
            .expect_err("name absent from all three channels must raise NameError");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_Builtins_Unbound' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_aborts_on_unbound_name() {
        // Forward-reference to a name not yet bound surfaces as
        // `FlowingError("global name '...' is not defined")` per
        // `flowcontext.py:853 find_global`. Mirrors upstream
        // Python's module-execution-aborts-on-exception semantic:
        // `Y = X + 1; X = 1` fails on Y with NameError and the
        // `X = 1` statement never executes. Walker therefore aborts
        // before reaching the LATER binding.
        let src =
            "pub const ParityProbe_Issue4_const_FORWARD: i64 = ParityProbe_Issue4_const_LATER + 1;
                   pub const ParityProbe_Issue4_const_LATER: i64 = 1;";
        let file = syn::parse_file(src).expect("forward-ref fixture parses");
        let err = register_rust_module(&file)
            .expect_err("forward-ref to unbound name aborts module execution");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_Issue4_const_LATER' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_float_char_byte_literals() {
        // Slice O11: `Lit::Float` / `Lit::Char` / `Lit::Byte` at the
        // module top level resolve to the same `ConstValue` shape
        // as their in-body counterparts at
        // `build_flow.rs::lower_literal`. A `const PI: f64 = 3.14`
        // and a body `3.14` must therefore agree on the bit pattern.
        let src = r#"
pub const ParityProbe_O11_Float: f64 = 3.14;
pub const ParityProbe_O11_FloatNeg: f64 = -2.5;
pub const ParityProbe_O11_FloatZero: f64 = 0.0;
pub const ParityProbe_O11_Char: char = 'a';
pub const ParityProbe_O11_CharUnicode: char = '한';
pub const ParityProbe_O11_Byte: u8 = b'a';
"#;
        let file = syn::parse_file(src).expect("float/char/byte fixture parses");
        let module_id = register_rust_module(&file).expect("float/char/byte literals walk");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Float"),
            Some(ConstValue::Float(3.14_f64.to_bits())),
            "Lit::Float folds to ConstValue::Float(to_bits())",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_FloatNeg"),
            Some(ConstValue::Float((-2.5_f64).to_bits())),
            "syn parses `-2.5` as `Expr::Unary(Neg, Lit::Float(2.5))`, \
             so the Neg arm folds via operation.rs::pyfunc's \
             (OpKind::Neg, [ConstValue::Float]) parity",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_FloatZero"),
            Some(ConstValue::Float(0.0_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Char"),
            Some(ConstValue::uni_str("a".to_string())),
            "Lit::Char folds to single-codepoint UniStr",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_CharUnicode"),
            Some(ConstValue::uni_str("한".to_string())),
            "non-ASCII char literal preserves the codepoint",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Byte"),
            Some(ConstValue::ByteStr(vec![b'a'])),
            "Lit::Byte folds to single-byte ByteStr",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_float_arithmetic() {
        // Slice O14: `eval_binop` admits `(Float, Float)` operand
        // pairs for arithmetic + comparison, mirroring
        // `operation.rs::pyfunc`'s Float arms (`operation.rs:1367-1383`).
        // Compound `const Z: f64 = X + Y` therefore folds at module
        // top level the same way the function-body `X + Y` would.
        // Rust `BinOp::Div` over `f64` is true division (matches
        // Python `/`). `Float / 0.0` raises `ZeroDivisionError`.
        let src = r#"
pub const ParityProbe_O14_FAdd: f64 = 1.0 + 2.5;
pub const ParityProbe_O14_FSub: f64 = 5.0 - 1.5;
pub const ParityProbe_O14_FMul: f64 = 2.0 * 3.5;
pub const ParityProbe_O14_FDiv: f64 = 7.0 / 2.0;
pub const ParityProbe_O14_FEq: bool = 1.5 == 1.5;
pub const ParityProbe_O14_FNe: bool = 1.5 != 2.0;
pub const ParityProbe_O14_FLt: bool = 1.5 < 2.0;
pub const ParityProbe_O14_FLe: bool = 1.5 <= 1.5;
pub const ParityProbe_O14_FGt: bool = 2.5 > 1.5;
pub const ParityProbe_O14_FGe: bool = 1.5 >= 1.5;
"#;
        let file = syn::parse_file(src).expect("float-arith fixture parses");
        let module_id = register_rust_module(&file).expect("Float arithmetic walks");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FAdd"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FSub"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FMul"),
            Some(ConstValue::Float(7.0_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FDiv"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
            "Rust `/` over f64 is true division (matches Python `/`)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FEq"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FNe"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FLt"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FLe"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FGt"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FGe"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_float_div_by_zero_raises() {
        // Float `/ 0.0` raises `ZeroDivisionError` per upstream
        // `operator.truediv(_, 0.0)`. Walker aborts the file.
        let src = "pub const ParityProbe_O14_FDivZero: f64 = 1.0 / 0.0;";
        let file = syn::parse_file(src).expect("float-zerodiv fixture parses");
        let err =
            register_rust_module(&file).expect_err("float division by zero must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError"),
                "expected ZeroDivisionError, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_float_mod_binds_and_zero_raises() {
        // Rust `%` is `BinOp::Rem`. The const-evaluator routes
        // `Float % Float` through `flowspace::operation::float_py_mod`
        // so the bound value matches the function-body constfold,
        // including the signed-zero copysign and sign-of-denominator
        // correction from `pypy/objspace/std/floatobject.py:543-563
        // descr_mod`. Mirror of
        // `register_rust_module_const_evaluator_float_arithmetic` for
        // the `%` operator.
        let src = "\
            pub const ParityProbe_FModOK: f64 = 7.0 % 2.0;
            pub const ParityProbe_FModSign: f64 = -7.0 % 2.0;
            pub const ParityProbe_FModNegDiv: f64 = 7.0 % -2.0;
        ";
        let file = syn::parse_file(src).expect("float-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        match module_globals_for_test(module_id, "ParityProbe_FModOK") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), 1.0),
            other => panic!("expected Float(1.0), got {other:?}"),
        }
        // `(-7.0) % 2.0`: divisor is positive, remainder follows
        // divisor sign → +1.0.
        match module_globals_for_test(module_id, "ParityProbe_FModSign") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), 1.0),
            other => panic!("expected Float(1.0), got {other:?}"),
        }
        // `7.0 % (-2.0)`: divisor is negative, remainder follows
        // divisor sign → -1.0.
        match module_globals_for_test(module_id, "ParityProbe_FModNegDiv") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), -1.0),
            other => panic!("expected Float(-1.0), got {other:?}"),
        }

        // Zero-divisor surfaces as FlowingError matching upstream
        // `operator.mod(_, 0.0)` ZeroDivisionError. Aborts walker.
        let src = "pub const ParityProbe_FModZero: f64 = 1.0 % 0.0;";
        let file = syn::parse_file(src).expect("float-mod-zero fixture parses");
        let err =
            register_rust_module(&file).expect_err("float modulo by zero must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError") && reason.contains("float modulo"),
                "expected ZeroDivisionError float modulo, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mixed_numeric_arithmetic_uses_python_coercion() {
        // Mirrors `flowspace::operation::pyfunc`'s mixed-numeric
        // arithmetic so the const-evaluator and the function-body
        // constfold agree on Python `bool ⊂ int ⊂ float`. Cross-type
        // pairs (Int+Float, Bool+Int, Int%Float, etc.) MUST bind a
        // value, not abort with TypeError.
        //
        // Upstream `PureOperation.constfold` (operation.py:120-127)
        // calls `operator.<op>(*args)` directly; Python's runtime
        // coerces `1 + 1.0 == 2.0`, `True * 3 == 3`, `7 % 2.5 == 2.0`,
        // etc. The mixed-numeric arm in `eval_binop` ports that
        // behavior.
        let src = "\
            pub const INT_ONE: i64 = 1;
            pub const FLOAT_HALF: f64 = 0.5;
            pub const BOOL_TRUE: bool = true;
            pub const ParityProbe_IntPlusFloat: f64 = INT_ONE + FLOAT_HALF;
            pub const ParityProbe_FloatTimesInt: f64 = FLOAT_HALF * INT_ONE;
            pub const ParityProbe_BoolPlusInt: i64 = BOOL_TRUE + INT_ONE;
            pub const ParityProbe_IntDivFloat: f64 = INT_ONE / FLOAT_HALF;
            pub const ParityProbe_IntModFloat: f64 = INT_ONE % FLOAT_HALF;
            pub const ParityProbe_BoolMulFloat: f64 = BOOL_TRUE * FLOAT_HALF;
        ";
        let file = syn::parse_file(src).expect("mixed-numeric fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        let f = |name: &str, expected: f64| match module_globals_for_test(module_id, name) {
            Some(ConstValue::Float(bits)) => assert_eq!(
                f64::from_bits(bits),
                expected,
                "{name}: expected Float({expected})",
            ),
            other => panic!("{name}: expected Float({expected}), got {other:?}"),
        };
        f("ParityProbe_IntPlusFloat", 1.5);
        f("ParityProbe_FloatTimesInt", 0.5);
        f("ParityProbe_IntDivFloat", 2.0);
        // 1 % 0.5 == 0.0 (1.0 - 0.5 * floor(1.0/0.5) == 0.0).
        f("ParityProbe_IntModFloat", 0.0);
        f("ParityProbe_BoolMulFloat", 0.5);

        match module_globals_for_test(module_id, "ParityProbe_BoolPlusInt") {
            Some(ConstValue::Int(2)) => {}
            other => panic!("expected Int(2), got {other:?}"),
        }

        // Float zero divisor through mixed-numeric Div is float-specific
        // ZeroDivisionError per upstream Python.
        let src = "\
            pub const FLOAT_ZERO: f64 = 0.0;
            pub const ParityProbe_IntDivFloatZero: f64 = 1 / FLOAT_ZERO;
        ";
        let file = syn::parse_file(src).expect("int/float-zero fixture parses");
        let err = register_rust_module(&file).expect_err("Int / Float(0.0) must raise float ZDE");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError") && reason.contains("float division by zero"),
                "expected float ZDE, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_int_float_compare_handles_mantissa_boundary() {
        // Port test for `pypy/objspace/std/floatobject.py:135-148`'s
        // bigint-aware Int↔Float comparison. f64 mantissa is 53 bits;
        // ints with `|i| > 2^53` are not exactly representable as f64.
        // Upstream's `_compare` routes through `do_compare_bigint` for
        // these values, comparing against `floor(f)` exactly. Naive
        // `*x as f64` would round `Int(2^53 + 1)` to `2^53`,
        // misclassifying it as equal to `Float(2^53)`.
        //
        // 2^53 = 9007199254740992 (exactly representable as f64)
        // 2^53 + 1 = 9007199254740993 (NOT representable; f64 rounds
        //   to 2^53)
        let src = "\
            pub const I_2_TO_53_PLUS_1: i64 = 9007199254740993;
            pub const F_2_TO_53: f64 = 9007199254740992.0;
            pub const ParityProbe_BigIntEqFloat: bool = I_2_TO_53_PLUS_1 == F_2_TO_53;
            pub const ParityProbe_BigIntGtFloat: bool = I_2_TO_53_PLUS_1 > F_2_TO_53;
            pub const ParityProbe_FloatLtBigInt: bool = F_2_TO_53 < I_2_TO_53_PLUS_1;
        ";
        let file = syn::parse_file(src).expect("mantissa-boundary fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        // Eq: i = 2^53 + 1, f = 2^53. They are NOT equal; naive cast
        // would say True (because (2^53 + 1) as f64 == 2^53).
        match module_globals_for_test(module_id, "ParityProbe_BigIntEqFloat") {
            Some(ConstValue::Bool(false)) => {}
            other => {
                panic!("Int(2^53+1) == Float(2^53) must be False (bigint compare), got {other:?}",)
            }
        }
        // Gt: i > f, since i = 2^53 + 1 > f = 2^53.
        match module_globals_for_test(module_id, "ParityProbe_BigIntGtFloat") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("Int(2^53+1) > Float(2^53) must be True, got {other:?}"),
        }
        match module_globals_for_test(module_id, "ParityProbe_FloatLtBigInt") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("Float(2^53) < Int(2^53+1) must be True, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mixed_numeric_eq_uses_python_coercion() {
        // Cross-numeric `==` / `!=` / `<` / `<=` / `>` / `>=` follow
        // Python's `bool ⊂ int ⊂ float` coercion, matching upstream
        // `PureOperation.constfold` calling `operator.eq(*args)`
        // directly. Without this, `Int(1) == Float(1.0)` would fold
        // to `False` via structural ConstValue equality, diverging
        // from Python.
        //
        // Cross-numeric pairs reach the const-evaluator via the
        // bindings/registry/builtins resolver chain (e.g. an `Int`
        // const compared to a `Float` const). Rust's typed source
        // can't write `1 == 1.0` directly, but it can write
        // `INT_ONE == FLOAT_ONE` after binding both names — that's
        // the path this test exercises.
        let src = "\
            pub const INT_ONE: i64 = 1;
            pub const FLOAT_ONE: f64 = 1.0;
            pub const BOOL_TRUE: bool = true;
            pub const ParityProbe_IntEqFloat: bool = INT_ONE == FLOAT_ONE;
            pub const ParityProbe_BoolEqInt: bool = BOOL_TRUE == INT_ONE;
            pub const ParityProbe_BoolEqFloat: bool = BOOL_TRUE == FLOAT_ONE;
            pub const ParityProbe_FloatLtInt: bool = FLOAT_ONE < INT_ONE;
        ";
        let file = syn::parse_file(src).expect("mixed-numeric fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        for (name, expected) in [
            ("ParityProbe_IntEqFloat", true),
            ("ParityProbe_BoolEqInt", true),
            ("ParityProbe_BoolEqFloat", true),
            ("ParityProbe_FloatLtInt", false), // 1.0 < 1 == False
        ] {
            match module_globals_for_test(module_id, name) {
                Some(ConstValue::Bool(v)) => assert_eq!(
                    v, expected,
                    "{name}: expected Bool({expected}), got Bool({v})",
                ),
                other => panic!("{name}: expected Bool, got {other:?}"),
            }
        }
    }

    #[test]
    fn register_host_lltype_for_primitive_struct_roundtrips() {
        // Walker pass for a struct whose every field is a recognized
        // primitive must populate the `HostObject → Ptr(GcStruct)`
        // registry. Mirrors upstream `_ptrEntry.compute_annotation`
        // (`lltype.py:1513-1518`) — the class object carries the
        // concrete `Ptr(GcStruct(...))` ready for `SomePtr` lift.
        let src = "
            pub struct ParityProbe_PrimStruct {
                pub a: i64,
                pub b: u64,
                pub c: bool,
                pub d: f64,
            }
        ";
        let file = syn::parse_file(src).expect("primitive-struct fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        let host = lookup_host(module_id, "ParityProbe_PrimStruct")
            .expect("ParityProbe_PrimStruct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host)
            .expect("primitive-only struct must populate the lltype registry");
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        assert_eq!(target._name, "ParityProbe_PrimStruct");
        let names: Vec<String> = target._names.clone();
        assert_eq!(
            names,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
        );
        assert_eq!(target._flds.get("a"), Some(&LowLevelType::Signed));
        assert_eq!(target._flds.get("b"), Some(&LowLevelType::Unsigned));
        assert_eq!(target._flds.get("c"), Some(&LowLevelType::Bool));
        assert_eq!(target._flds.get("d"), Some(&LowLevelType::Float));
    }

    #[test]
    fn register_host_lltype_lifts_raw_pointer_fields_to_ptr_opaque() {
        // Raw-pointer fields (`*const T`, `*mut T`) lift to
        // `Ptr(OpaqueType(T))` so structs like PyFrame whose layout
        // is dense raw pointers reach the catalog without requiring
        // every referent struct to be registered first. Distinct
        // referent names produce identity-distinct opaque tags.
        let src = "
            pub struct ParityProbe_RawPtrFields {
                pub a: *const ParityProbe_OtherA,
                pub b: *mut ParityProbe_OtherB,
                pub v: *mut (),
            }
        ";
        let file = syn::parse_file(src).expect("raw-ptr fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_RawPtrFields")
            .expect("raw-ptr struct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host)
            .expect("raw-ptr-field struct must populate the lltype registry");
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let extract_opaque_tag = |fld_name: &str| -> String {
            let fld = target._flds.get(fld_name).unwrap_or_else(|| {
                panic!(
                    "expected field `{fld_name}` in struct, got {:?}",
                    target._flds
                )
            });
            let inner_ptr = match fld {
                LowLevelType::Ptr(p) => p,
                other => panic!("`{fld_name}` must be Ptr(...), got {other:?}"),
            };
            match &inner_ptr.TO {
                crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Opaque(op) => {
                    op.tag.clone()
                }
                other => panic!("`{fld_name}` Ptr target must be Opaque, got {other:?}"),
            }
        };
        assert_eq!(extract_opaque_tag("a"), "ParityProbe_OtherA");
        assert_eq!(extract_opaque_tag("b"), "ParityProbe_OtherB");
        assert_eq!(extract_opaque_tag("v"), "Void");
    }

    #[test]
    fn register_host_lltype_rejects_vec_fields_without_explicit_array_model() {
        // Rust `Vec<T>` is not line-by-line equivalent to an upstream
        // `lltype.Array(T)` declaration: the Rust field stores a vector
        // header, not a direct lltype array pointer. Keep it out of the
        // catalog until the caller supplies an explicit RPython-shape
        // model for the field.
        let src = "
            pub struct ParityProbe_VecFields {
                pub a: Vec<i64>,
                pub b: Vec<u64>,
                pub c: Vec<f64>,
            }
        ";
        let file = syn::parse_file(src).expect("vec fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_VecFields")
            .expect("Vec-field struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Vec fields must not populate the lltype registry by implicit Ptr(Array) collapse",
        );
    }

    #[test]
    fn register_host_lltype_resolves_type_alias_in_field() {
        // `type T = U;` declarations in the same module register an
        // alias entry that `syn_primitive_to_lltype` chases when a
        // field's leaf identifier matches. The classic pyre shape is
        // `type PyObjectRef = *mut PyObject;` — a struct with a
        // `PyObjectRef` field should lift the field to the same
        // `Ptr(Opaque)` shape a literal `*mut PyObject` would.
        let src = "
            pub type ParityProbe_AliasRef = *mut ParityProbe_AliasTarget;
            pub struct ParityProbe_AliasUser {
                pub ptr: ParityProbe_AliasRef,
            }
        ";
        let file = syn::parse_file(src).expect("alias fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_AliasUser")
            .expect("alias-user struct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host).expect(
            "alias-user struct must populate the lltype registry \
                     after alias resolution",
        );
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let ptr_field = target._flds.get("ptr").expect("`ptr` field present");
        let inner_ptr = match ptr_field {
            LowLevelType::Ptr(p) => p,
            other => panic!("`ptr` must lift to Ptr(...), got {other:?}"),
        };
        match &inner_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Opaque(op) => {
                assert_eq!(op.tag, "ParityProbe_AliasTarget");
            }
            other => panic!("inner Ptr target must be Opaque, got {other:?}"),
        }
    }

    #[test]
    fn register_host_lltype_resolves_type_alias_declared_in_prior_file() {
        // Cross-file alias visibility — pyre source files are walked
        // sequentially and each `WalkerTypeAliasGuard::drop` restores
        // the parent scope.  The pipeline entry installs a
        // `WalkerAliasFloorGuard` carrying the union of every parsed
        // file's `type T = U;` declarations so the second file's
        // struct field declared as `field: ProbeXFileRef` resolves
        // through the seeded floor.  Mirrors RPython's whole-program
        // alias visibility — `flowcontext.py:847` resolves names
        // through `func_globals` which spans the imported namespace.
        let src_decl = "
            pub type ProbeXFileRef = *mut ProbeXFileTarget;
        ";
        let src_user = "
            pub struct ProbeXFileUser {
                pub ptr: ProbeXFileRef,
            }
        ";
        let file_decl = syn::parse_file(src_decl).expect("decl fixture parses");
        let file_user = syn::parse_file(src_user).expect("user fixture parses");
        // Install the cross-file alias floor before either walk runs,
        // matching the production pipeline's
        // `WalkerAliasFloorGuard::install` step.
        let _floor = WalkerAliasFloorGuard::install([&file_decl, &file_user]);
        let _ = register_rust_module(&file_decl).expect("decl walk succeeds");
        let module_user_id = register_rust_module(&file_user).expect("user walk succeeds");

        let host = lookup_host(module_user_id, "ProbeXFileUser")
            .expect("ProbeXFileUser struct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host).expect(
            "user struct must populate the lltype registry — the \
             cross-file alias for `ProbeXFileRef` should resolve via \
             the seeded WalkerAliasFloorGuard",
        );
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let ptr_field = target._flds.get("ptr").expect("`ptr` field present");
        let inner_ptr = match ptr_field {
            LowLevelType::Ptr(p) => p,
            other => panic!("`ptr` must lift to Ptr(...), got {other:?}"),
        };
        match &inner_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Opaque(op) => {
                assert_eq!(op.tag, "ProbeXFileTarget");
            }
            other => panic!("inner Ptr target must be Opaque, got {other:?}"),
        }
    }

    #[test]
    fn register_host_lltype_skips_unsupported_field_shape_struct() {
        // Catalog still rejects shapes that lack a recognized lltype
        // mapping (`Result<T, E>`, `&T`, `dyn Trait`, tuple /
        // fixed-array / slice / fn types, multi-segment paths,
        // by-value nested struct). `Option<T>` is admitted only when
        // the inner `T` directly resolves to a pointer-like lltype;
        // non-pointer `Option<i64>` must leave the host's lltype slot
        // empty so the `SomeInstance(classdef=None)` fallback path
        // remains available.
        let src = "
            pub struct ParityProbe_UnsupportedFieldStruct {
                pub inner: Option<i64>,
            }
        ";
        let file = syn::parse_file(src).expect("unsupported-field fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        let host = lookup_host(module_id, "ParityProbe_UnsupportedFieldStruct")
            .expect("ParityProbe_UnsupportedFieldStruct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "unsupported field shape must NOT populate the lltype \
             registry until the matching catalog port lands",
        );
    }

    #[test]
    fn register_host_lltype_lifts_option_of_pointer_field_to_inner_ptr() {
        // `Option<*mut T>` carries the same lltype shape as the inner
        // raw pointer — the `None` arm maps to `lltype.nullptr(...)`
        // and the `Some(p)` arm to the typed pointer of the same `Ptr`.
        // Only directly pointer-like inner shapes are accepted.
        let src = "
            pub struct ParityProbe_OptionPtr {
                pub raw: Option<*mut ParityProbe_OptionTarget>,
            }
        ";
        let file = syn::parse_file(src).expect("option-of-pointer fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_OptionPtr")
            .expect("Option-of-pointer struct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host).expect(
            "Option-of-pointer struct must populate the lltype registry \
             via the nullable-ptr lift",
        );
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        // `raw: Option<*mut T>` — inner is `Ptr(Opaque(T))`.
        let raw_ll = target._flds.get("raw").expect("`raw` field present");
        let raw_ptr = match raw_ll {
            LowLevelType::Ptr(p) => p,
            other => panic!("`raw` must lift to Ptr, got {other:?}"),
        };
        match &raw_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Opaque(op) => {
                assert_eq!(op.tag, "ParityProbe_OptionTarget");
            }
            other => panic!("`raw` inner must be Opaque, got {other:?}"),
        }
    }

    #[test]
    fn register_host_lltype_rejects_option_of_vec_field() {
        // `Option<Vec<T>>` is not a nullable lltype pointer. Rust stores
        // the vector shape inside the option niche optimization, and
        // the catalog must not erase that into `Ptr(Array(T))`.
        let src = "
            pub struct ParityProbe_OptionVec {
                pub vec_inner: Option<Vec<i64>>,
            }
        ";
        let file = syn::parse_file(src).expect("option-vec fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_OptionVec")
            .expect("Option<Vec>-field struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Option<Vec<T>> must not populate the lltype registry by nullable-array collapse",
        );
    }

    #[test]
    fn register_host_lltype_rejects_atomic_field_wrappers() {
        // Atomic wrappers are synchronization primitives, not upstream
        // lltype primitive declarations. Keep them out of the strict
        // field catalog unless a caller provides an explicit RPython
        // storage model.
        let src = "
            pub struct ParityProbe_Atomic {
                pub a_i64: AtomicI64,
                pub a_u8: AtomicU8,
                pub a_bool: AtomicBool,
                pub a_ptr: AtomicPtr<ParityProbe_AtomicTarget>,
                pub a_qualified: std::sync::atomic::AtomicI64,
            }
        ";
        let file = syn::parse_file(src).expect("atomic-field fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_Atomic")
            .expect("Atomic-field struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Atomic wrappers must not implicitly collapse to primitive lltypes",
        );
    }

    #[test]
    fn register_host_lltype_rejects_vec_of_by_value_registered_struct_without_panic() {
        // All Rust `Vec<T>` fields reject in the strict catalog. This
        // fixture keeps the old by-value registered-struct case covered
        // and verifies that the parent rejects without disturbing the
        // independently catalogable inner struct.
        let src = "
            pub struct ParityProbe_VecInner {
                pub flag: i64,
            }
            pub struct ParityProbe_VecByValueParent {
                pub items: Vec<ParityProbe_VecInner>,
            }
        ";
        let file = syn::parse_file(src).expect("vec-by-value fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_VecByValueParent")
            .expect("ParityProbe_VecByValueParent must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Vec-of-by-value-GcStruct must NOT populate the lltype \
             registry — gc-container elements are rejected upstream",
        );
        // Inner struct still catalogs (primitive fields only).
        let inner_host = lookup_host(module_id, "ParityProbe_VecInner")
            .expect("ParityProbe_VecInner must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&inner_host).is_some(),
            "primitive-only inner struct should cataloge independently",
        );
    }

    #[test]
    fn register_host_lltype_rejects_non_whitelisted_multi_segment_path() {
        // Multi-segment paths reject in the strict catalog. Leaf-name
        // collapse for stdlib or imported wrappers would invent shapes
        // that the corresponding RPython source did not declare.
        let src = "
            pub struct ParityProbe_MultiSegmentMiss {
                pub items: std::collections::HashMap<i64, i64>,
            }
        ";
        let file = syn::parse_file(src).expect("multi-segment fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_MultiSegmentMiss")
            .expect("ParityProbe_MultiSegmentMiss must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "multi-segment field must NOT populate the lltype registry \
             without explicit source-level modeling",
        );
    }

    #[test]
    fn register_host_lltype_rejects_fully_qualified_wrappers() {
        // Qualified Rust wrappers do not become lltype fields by leaf
        // name. They need an explicit RPython-shape model just like the
        // bare wrapper spelling.
        let src = "
            pub struct ParityProbe_QualifiedWrappers {
                pub r: std::rc::Rc<i64>,
                pub c: std::cell::RefCell<u64>,
                pub v: std::vec::Vec<i64>,
            }
        ";
        let file = syn::parse_file(src).expect("qualified-wrapper fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_QualifiedWrappers")
            .expect("qualified-wrapper struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "qualified Rust wrappers must not implicitly collapse to lltype fields",
        );
    }

    #[test]
    fn register_host_lltype_embeds_first_field_gc_struct_by_value() {
        // `pub ob_header: PyObject` as the FIRST field of a subclass
        // struct mirrors CPython's `PyObject_HEAD` inheritance marker.
        // RPython's orthodox shape is
        // `lltype.GcStruct("Outer", ("ob_header", PYOBJECT_GCSTRUCT))`
        // where the embedded GcStruct sits at field index 0
        // (`lltype.py:296-303 Struct._first_struct`). The catalog
        // collapses the by-value embedding into a nested
        // `LowLevelType::Struct(GcStruct)` field, NOT a `Ptr` —
        // `_first_struct` requires the literal struct, not an
        // indirection.
        let src = "
            pub struct ParityProbe_Base {
                pub flag: i64,
            }
            pub struct ParityProbe_Sub {
                pub ob_header: ParityProbe_Base,
                pub extra: i64,
            }
        ";
        let file = syn::parse_file(src).expect("first-field nested-struct fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let sub_host = lookup_host(module_id, "ParityProbe_Sub")
            .expect("ParityProbe_Sub must register as HostObject");
        let sub_ptr = super::super::host_env::lookup_host_lltype(&sub_host).expect(
            "subclass struct with first-field nested GcStruct must populate \
             the lltype registry",
        );
        let sub_target = match &sub_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        // `ob_header: ParityProbe_Base` — must be the first field, must
        // be a `Struct(GcStruct(ParityProbe_Base))`, NOT a `Ptr`.
        assert_eq!(
            sub_target._names.first().map(String::as_str),
            Some("ob_header"),
            "ob_header must be the first field for inheritance shape parity",
        );
        let ob_header_ll = sub_target
            ._flds
            .get("ob_header")
            .expect("`ob_header` field present");
        let inner_struct = match ob_header_ll {
            LowLevelType::Struct(s) => s,
            other => panic!("`ob_header` must be Struct(GcStruct), got {other:?}"),
        };
        assert_eq!(
            inner_struct._name, "ParityProbe_Base",
            "embedded struct name must match the referenced ident",
        );
        assert_eq!(
            inner_struct._gckind,
            GcKind::Gc,
            "embedded struct keeps its GcKind::Gc identity at the inheritance call site",
        );
        // `extra: i64` — non-first primitive field, lifts as Signed.
        assert_eq!(sub_target._flds.get("extra"), Some(&LowLevelType::Signed));
    }

    #[test]
    fn register_host_lltype_rejects_non_first_nested_gc_struct() {
        // RPython `Struct._note_inlined_into` (`lltype.py:305-312`)
        // rejects a `GcStruct` embedded at any field index other than
        // 0. The pre-validation in `try_build_gc_struct_ptr` mirrors
        // that — the parent struct should NOT register when a nested
        // `GcStruct` field appears second or later, so downstream
        // lookups fall back to the `SomeInstance(classdef=None)` path
        // until the user rewrites the field as a pointer or moves it
        // to position 0.
        let src = "
            pub struct ParityProbe_NestedBase {
                pub flag: i64,
            }
            pub struct ParityProbe_BadOrder {
                pub leading_int: i64,
                pub trailing_base: ParityProbe_NestedBase,
            }
        ";
        let file = syn::parse_file(src).expect("non-first nested-struct fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_BadOrder")
            .expect("ParityProbe_BadOrder must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "non-first nested GcStruct field must NOT populate the lltype \
             registry — RPython _note_inlined_into rejects this embedding",
        );
    }

    #[test]
    fn register_host_lltype_rejects_interior_mutability_wrappers() {
        // Interior-mutability wrappers are Rust implementation details,
        // but erasing them inside the field catalog would still invent a
        // field shape that the RPython source did not declare.
        let src = "
            pub struct ParityProbe_InteriorMut {
                pub c: Cell<i64>,
                pub rc: RefCell<u64>,
                pub oc: OnceCell<bool>,
                pub uc: UnsafeCell<f64>,
            }
        ";
        let file = syn::parse_file(src).expect("interior-mut fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_InteriorMut")
            .expect("interior-mut-field struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "interior-mutability wrappers must not implicitly collapse to primitive lltypes",
        );
    }

    #[test]
    fn register_host_lltype_lifts_reference_to_registered_struct_as_ptr() {
        // `&T` where `T` is a walker-registered struct lifts to
        // `Ptr(GcStruct(T))` — RPython's gc-reference shape
        // (`lltype.py:1513-1518 _ptrEntry.compute_annotation`).  Both
        // same-scope and process-wide registrations resolve through
        // [`walker_struct_ptr_lookup`].
        let src = "
            pub struct ParityProbe_RefTarget {
                pub flag: i64,
            }
            pub struct ParityProbe_RefUser {
                pub borrowed: &'static ParityProbe_RefTarget,
            }
        ";
        let file = syn::parse_file(src).expect("&T fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_RefUser")
            .expect("ParityProbe_RefUser must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host)
            .expect("&T-field struct must populate the lltype registry");
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let borrowed_ll = target
            ._flds
            .get("borrowed")
            .expect("`borrowed` field present");
        let borrowed_ptr = match borrowed_ll {
            LowLevelType::Ptr(p) => p,
            other => panic!("`borrowed` must lift to Ptr, got {other:?}"),
        };
        match &borrowed_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => {
                assert_eq!(s._name, "ParityProbe_RefTarget");
                assert_eq!(s._gckind, GcKind::Gc);
            }
            other => panic!("`borrowed` inner must be Struct(GcStruct), got {other:?}"),
        }
    }

    #[test]
    fn register_host_lltype_rejects_reference_to_non_struct_target() {
        // `&str`, `&dyn Trait`, `&[T]`, `&i64`, `&UnregisteredStruct`
        // all reject — the catalog only knows how to lift `&T` when
        // `T` is a walker-registered struct.
        let src = "
            pub struct ParityProbe_RefMiss {
                pub borrowed: &'static str,
            }
        ";
        let file = syn::parse_file(src).expect("&str fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_RefMiss")
            .expect("ParityProbe_RefMiss must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "`&str` (or any &T with non-struct target) must NOT populate \
             the lltype registry until rstr / dyn-Trait support lands",
        );
    }

    #[test]
    fn register_host_lltype_embeds_cross_file_nested_gc_struct_via_floor_guard() {
        // Two separate walker invocations (mirroring two pyre source
        // files, e.g. `pyobject.rs` + `bytesobject.rs`): the first
        // registers a base struct, the second names it by bare ident
        // in a `pub ob_header: BaseName` field.  Cross-file resolution
        // works because [`WalkerStructPtrsFloorGuard::install`] mints
        // every parsed file's top-level structs into the walker
        // thread-local before any per-file walk runs.  Mirrors
        // upstream `Bookkeeper`'s whole-program type cache
        // (`bookkeeper.py:67 self.descs = {}`).
        let base_src = "
            pub struct ParityProbe_CrossFileBase {
                pub flag: i64,
            }
        ";
        let base_file = syn::parse_file(base_src).expect("base fixture parses");
        let sub_src = "
            pub struct ParityProbe_CrossFileSub {
                pub ob_header: ParityProbe_CrossFileBase,
                pub tail: i64,
            }
        ";
        let sub_file = syn::parse_file(sub_src).expect("sub fixture parses");
        let _floor = WalkerStructPtrsFloorGuard::install([&base_file, &sub_file]);
        let _base_module = register_rust_module(&base_file).expect("base walk succeeds");
        let sub_module = register_rust_module(&sub_file).expect("sub walk succeeds");
        let sub_host = lookup_host(sub_module, "ParityProbe_CrossFileSub")
            .expect("ParityProbe_CrossFileSub must register as HostObject");
        let sub_ptr = super::super::host_env::lookup_host_lltype(&sub_host).expect(
            "cross-file nested-struct embedding must populate the lltype \
             registry via the floor pre-mint",
        );
        let sub_target = match &sub_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let ob_header_ll = sub_target
            ._flds
            .get("ob_header")
            .expect("`ob_header` field present");
        let inner_struct = match ob_header_ll {
            LowLevelType::Struct(s) => s,
            other => panic!("`ob_header` must be Struct, got {other:?}"),
        };
        assert_eq!(inner_struct._name, "ParityProbe_CrossFileBase");
        assert_eq!(inner_struct._gckind, GcKind::Gc);
    }

    #[test]
    fn register_host_lltype_resolves_forward_reference_nested_struct() {
        // The preseed fixed-point pass removes source-order sensitivity:
        // a first-field embedded GcStruct can resolve even when the
        // embedded struct appears later in the Rust module.
        let src = "
            pub struct ParityProbe_ForwardSub {
                pub ob_header: ParityProbe_ForwardBase,
            }
            pub struct ParityProbe_ForwardBase {
                pub flag: i64,
            }
        ";
        let file = syn::parse_file(src).expect("forward-reference fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let sub_host = lookup_host(module_id, "ParityProbe_ForwardSub")
            .expect("ParityProbe_ForwardSub must register as HostObject");
        let sub_ptr = super::super::host_env::lookup_host_lltype(&sub_host)
            .expect("forward-reference nested struct lookup must populate the lltype registry");
        let sub_target = match &sub_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let ob_header_ll = sub_target
            ._flds
            .get("ob_header")
            .expect("`ob_header` field present");
        let inner_struct = match ob_header_ll {
            LowLevelType::Struct(s) => s,
            other => panic!("`ob_header` must be Struct, got {other:?}"),
        };
        assert_eq!(inner_struct._name, "ParityProbe_ForwardBase");
        // The base IS catalogued because it has only primitive fields.
        let base_host = lookup_host(module_id, "ParityProbe_ForwardBase")
            .expect("ParityProbe_ForwardBase must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&base_host).is_some(),
            "primitive-only base struct should still cataloge",
        );
    }

    #[test]
    fn register_host_lltype_resolves_multi_level_forward_reference_chain() {
        // Three-level forward chain in reverse source order — preseed
        // fixed-point must converge across multiple iterations:
        // pass 1: C (primitive-only) registers;
        // pass 2: B's first-field C resolves, B registers;
        // pass 3: A's first-field B resolves, A registers.
        let src = "
            pub struct ParityProbe_ChainA {
                pub head: ParityProbe_ChainB,
            }
            pub struct ParityProbe_ChainB {
                pub head: ParityProbe_ChainC,
            }
            pub struct ParityProbe_ChainC {
                pub flag: i64,
            }
        ";
        let file = syn::parse_file(src).expect("multi-level forward chain parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        for name in [
            "ParityProbe_ChainA",
            "ParityProbe_ChainB",
            "ParityProbe_ChainC",
        ] {
            let host =
                lookup_host(module_id, name).unwrap_or_else(|| panic!("{name} must register"));
            assert!(
                super::super::host_env::lookup_host_lltype(&host).is_some(),
                "{name} must populate the lltype registry through preseed",
            );
        }
    }

    #[test]
    fn register_host_lltype_resolves_forward_reference_through_type_alias() {
        // Alias-mediated forward reference: `type Alias = LateStruct;`
        // followed by a struct embedding `Alias` as first field, with
        // `LateStruct` defined last. `collect_type_aliases` runs before
        // the preseed fixed-point pass, so the alias map is fully
        // populated when preseed walks structs; alias recursion in
        // `syn_primitive_to_lltype` reaches `LateStruct` once it
        // registers in a later preseed iteration.
        let src = "
            pub struct ParityProbe_AliasEmbed {
                pub head: ParityProbe_AliasName,
            }
            pub type ParityProbe_AliasName = ParityProbe_AliasTarget;
            pub struct ParityProbe_AliasTarget {
                pub flag: i64,
            }
        ";
        let file = syn::parse_file(src).expect("alias-mediated forward ref parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let embed_host = lookup_host(module_id, "ParityProbe_AliasEmbed")
            .expect("ParityProbe_AliasEmbed must register");
        let embed_ptr = super::super::host_env::lookup_host_lltype(&embed_host)
            .expect("alias-mediated forward ref must lift to lltype");
        let embed_target = match &embed_ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        let head_ll = embed_target
            ._flds
            .get("head")
            .expect("`head` field present");
        let inner_struct = match head_ll {
            LowLevelType::Struct(s) => s,
            other => panic!("`head` must be Struct, got {other:?}"),
        };
        assert_eq!(inner_struct._name, "ParityProbe_AliasTarget");
    }

    #[test]
    fn register_host_lltype_rejects_mutual_by_value_struct_cycle() {
        // Mutual by-value GcStruct embedding has no RPython equivalent
        // (the layout would be infinite size). The preseed fixed-point
        // pass makes no progress for either side, so neither struct
        // registers — matching upstream `lltype.GcStruct` which has no
        // representable cyclic by-value shape.
        let src = "
            pub struct ParityProbe_CycleA {
                pub head: ParityProbe_CycleB,
            }
            pub struct ParityProbe_CycleB {
                pub head: ParityProbe_CycleA,
            }
        ";
        let file = syn::parse_file(src).expect("mutual cycle parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        for name in ["ParityProbe_CycleA", "ParityProbe_CycleB"] {
            let host = lookup_host(module_id, name)
                .unwrap_or_else(|| panic!("{name} must register as HostObject"));
            assert!(
                super::super::host_env::lookup_host_lltype(&host).is_none(),
                "{name} must NOT populate the lltype registry — mutual by-value cycle is unrepresentable",
            );
        }
    }

    #[test]
    fn register_host_lltype_rejects_option_of_non_pointer_field() {
        // `Option<Box<i64>>` is not a nullable lltype pointer. The
        // catalog no longer unwraps `Box<T>` in fields, so this rejects
        // before any `Some(0)`/`None` primitive ambiguity can arise.
        let src = "
            pub struct ParityProbe_OptionNonPtr {
                pub boxed_int: Option<Box<i64>>,
            }
        ";
        let file = syn::parse_file(src).expect("option-non-pointer fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_OptionNonPtr")
            .expect("ParityProbe_OptionNonPtr must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Option-of-non-pointer field must NOT populate the lltype \
             registry — Some(0)/None ambiguity has no RPython equivalent",
        );
    }

    #[test]
    fn register_host_lltype_rejects_box_rc_arc_field_wrappers() {
        // Function-argument register classification may unwrap these,
        // but struct field storage parity may not. Reject Rust heap
        // wrappers unless an explicit RPython pointer/container shape is
        // modeled at the source level.
        let src = "
            pub struct ParityProbe_BoxRcArc {
                pub b: Box<i64>,
                pub r: Rc<u64>,
                pub a: Arc<f64>,
            }
        ";
        let file = syn::parse_file(src).expect("box/rc/arc fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_BoxRcArc")
            .expect("Box/Rc/Arc-field struct must register as HostObject");
        assert!(
            super::super::host_env::lookup_host_lltype(&host).is_none(),
            "Box/Rc/Arc fields must not implicitly collapse to inner lltypes",
        );
    }

    #[test]
    fn register_host_lltype_preserves_char_and_singlefloat_identity() {
        // `char` must land in `UniChar` and `f32` in `SingleFloat`,
        // matching the upstream `lltype.py` primitive identities
        // (`UniChar = Primitive("UniChar", "\x00")`,
        // `SingleFloat = Primitive("SingleFloat", r_singlefloat(0.0))`).
        // The fact that `getkind` collapses both to `'int'`
        // downstream is a register-class fold, not a lltype-identity
        // collapse — the rtyper still needs the distinct
        // `UniChar` / `SingleFloat` primitives to dispatch
        // `UniCharRepr` / `SingleFloatRepr`.
        let src = "
            pub struct ParityProbe_CharSingle {
                pub ch: char,
                pub sf: f32,
            }
        ";
        let file = syn::parse_file(src).expect("char/singlefloat fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_CharSingle")
            .expect("ParityProbe_CharSingle must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host)
            .expect("char/f32 struct must populate the lltype registry");
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        assert_eq!(target._flds.get("ch"), Some(&LowLevelType::UniChar));
        assert_eq!(target._flds.get("sf"), Some(&LowLevelType::SingleFloat));
    }

    #[test]
    fn register_host_lltype_unit_struct_yields_empty_field_ptr() {
        // Unit structs (`struct Foo;`) are valid upstream
        // (`lltype.Struct(name)` with zero `*fields`).  Lift them
        // through the catalog symmetrically.
        let src = "pub struct ParityProbe_UnitStruct;";
        let file = syn::parse_file(src).expect("unit-struct fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");
        let host = lookup_host(module_id, "ParityProbe_UnitStruct")
            .expect("unit struct must register as HostObject");
        let ptr = super::super::host_env::lookup_host_lltype(&host)
            .expect("unit struct must populate the lltype registry");
        let target = match &ptr.TO {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Struct(s) => s,
            other => panic!("expected PtrTarget::Struct, got {other:?}"),
        };
        assert_eq!(target._name, "ParityProbe_UnitStruct");
        assert!(target._names.is_empty());
    }

    #[test]
    fn extract_static_decls_empty_file_returns_empty() {
        let file = parse_file("");
        assert!(extract_static_decls(&file, "anyprefix").is_empty());
    }

    #[test]
    fn extract_static_decls_const_emitted_alongside_static() {
        use crate::model::ValueType;
        let file = parse_file("pub const FOO: i64 = 42;");
        let decls = extract_static_decls(&file, "modprefix");
        assert_eq!(decls.len(), 1, "Item::Const must be catalogued");
        assert_eq!(decls[0].0, vec!["modprefix".to_string(), "FOO".to_string()]);
        assert_eq!(decls[0].1, ValueType::Int);
    }

    #[test]
    fn extract_static_decls_pub_static_int_unsigned_bool_float() {
        use crate::model::ValueType;
        let file = parse_file(
            "pub static A: i64 = 0;
             pub static B: usize = 0;
             pub static C: bool = false;
             pub static D: f64 = 0.0;",
        );
        let decls = extract_static_decls(&file, "pyobject");
        assert_eq!(decls.len(), 4);
        assert_eq!(decls[0].0, vec!["pyobject".to_string(), "A".to_string()]);
        assert_eq!(decls[0].1, ValueType::Int);
        assert_eq!(decls[1].1, ValueType::Unsigned);
        assert_eq!(decls[2].1, ValueType::Bool);
        assert_eq!(decls[3].1, ValueType::Float);
    }

    #[test]
    fn extract_static_decls_empty_prefix_emits_bare_name() {
        let file = parse_file("pub static BARE: bool = true;");
        let decls = extract_static_decls(&file, "");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].0, vec!["BARE".to_string()]);
    }

    #[test]
    fn extract_static_decls_compound_type_classifies_as_ref() {
        use crate::model::ValueType;
        let file = parse_file(
            "pub static INT_TYPE: PyType = new_pytype(\"int\");
             pub static SMALL_INTS: LazyLock<Vec<W_IntObject>> = LazyLock::new(|| vec![]);",
        );
        let decls = extract_static_decls(&file, "pyobject");
        assert_eq!(decls.len(), 2);
        // `PyType` is a bare path leaf; `classify_fn_arg_ty` falls through
        // to the compound-type `Ref` branch.
        assert_eq!(decls[0].1, ValueType::Ref(Some("PyType".to_string())));
        // `LazyLock<...>` is not in the `Box | Rc | Arc` unwrap list;
        // also falls through to Ref carrying the wrapper's leaf ident.
        assert_eq!(decls[1].1, ValueType::Ref(Some("LazyLock".to_string())));
        // Non-literal RHS: `ConstValue` resolution falls back to `None`,
        // adapter retains the `UniStr(joined)` sentinel.
        assert!(decls[0].2.is_none());
        assert!(decls[1].2.is_none());
    }

    #[test]
    fn extract_static_decls_literal_rhs_resolves_const_value() {
        use crate::flowspace::model::ConstValue;
        let file = parse_file(
            "pub const I: i64 = 42;
             pub const NEG: i32 = -7;
             pub const B: bool = false;
             pub const F: f64 = 1.5;
             pub const S: &str = \"hi\";",
        );
        let decls = extract_static_decls(&file, "");
        assert_eq!(decls.len(), 5);
        assert_eq!(decls[0].2, Some(ConstValue::Int(42)));
        assert_eq!(decls[1].2, Some(ConstValue::Int(-7)));
        assert_eq!(decls[2].2, Some(ConstValue::Bool(false)));
        assert_eq!(decls[3].2, Some(ConstValue::float(1.5)));
        assert_eq!(decls[4].2, Some(ConstValue::UniStr("hi".to_string())));
    }

    #[test]
    fn extract_static_decls_null_pointer_rhs_resolves_to_null_address() {
        use crate::flowspace::model::ConstValue;
        use crate::translator::rtyper::lltypesystem::lltype::_address;
        // `std::ptr::null_mut()` / `null()` resolve to the `_address::Null`
        // carrier so the `LoadStatic` lowers to a null-ref `Constant`
        // instead of the `UniStr(path)` sentinel.
        let file = parse_file(
            "pub const PY_NULL: PyObjectRef = std::ptr::null_mut();
             pub const CONST_NULL: *const u8 = std::ptr::null();
             pub const BARE_NULL: *mut u8 = null_mut();",
        );
        let decls = extract_static_decls(&file, "");
        assert_eq!(decls.len(), 3);
        let null = Some(ConstValue::LLAddress(_address::Null));
        assert_eq!(decls[0].2, null);
        assert_eq!(decls[1].2, null);
        assert_eq!(decls[2].2, null);
    }

    #[test]
    fn extract_static_decls_thread_local_const_literal_resolves() {
        use crate::flowspace::model::ConstValue;
        // `thread_local! { static X: T = const { LIT }; }` — the
        // `const { ... }` block wrapper unwraps to the inner literal.
        let file = parse_file(
            "thread_local! {\n\
                 static FLAG: bool = const { true };\n\
             }",
        );
        let decls = extract_static_decls(&file, "");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].2, Some(ConstValue::Bool(true)));
    }

    #[test]
    fn extract_static_decls_non_pub_static_included() {
        let file = parse_file("static PRIVATE: bool = false;");
        let decls = extract_static_decls(&file, "mod");
        assert_eq!(decls.len(), 1, "non-pub static must still be catalogued");
        assert_eq!(decls[0].0, vec!["mod".to_string(), "PRIVATE".to_string()]);
    }

    #[test]
    fn extract_static_decls_thread_local_macro_inner_statics_catalogued() {
        use crate::model::ValueType;
        let file = parse_file(
            "thread_local! {\n\
                 static CALL_DEPTH: Cell<u32> = const { Cell::new(0) };\n\
                 pub static SHADOW_STACK: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };\n\
             }",
        );
        let decls = extract_static_decls(&file, "modprefix");
        assert_eq!(
            decls.len(),
            2,
            "both thread_local statics must be catalogued"
        );
        let names: Vec<String> = decls
            .iter()
            .map(|(seg, _, _)| seg.last().cloned().unwrap())
            .collect();
        assert!(names.contains(&"CALL_DEPTH".to_string()));
        assert!(names.contains(&"SHADOW_STACK".to_string()));
        // Compound types fall through to `ValueType::Ref` per the
        // `compound_type_classifies_as_ref` invariant.
        assert!(
            decls
                .iter()
                .all(|(_, ty, _)| matches!(ty, ValueType::Ref(_)))
        );
    }

    #[test]
    fn extract_static_decls_non_thread_local_macros_ignored() {
        let file = parse_file(
            "println!(\"hello\");\n\
             lazy_static::lazy_static! {\n\
                 static ref OTHER: Vec<u8> = vec![];\n\
             }",
        );
        let decls = extract_static_decls(&file, "");
        assert!(
            decls.is_empty(),
            "only `thread_local!` is recognised; other macros stay opaque"
        );
    }

    #[test]
    fn extract_static_decls_other_items_ignored() {
        let file = parse_file(
            "pub fn foo() {}
             pub const C: i64 = 0;
             pub static S: bool = true;
             pub struct St;
             pub enum E { A }",
        );
        let decls = extract_static_decls(&file, "");
        assert_eq!(
            decls.len(),
            2,
            "Item::Static and Item::Const contribute; fn/struct/enum ignored",
        );
        let names: Vec<String> = decls.iter().map(|(seg, _, _)| seg[0].clone()).collect();
        assert!(names.contains(&"C".to_string()));
        assert!(names.contains(&"S".to_string()));
    }
}
