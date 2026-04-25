//! Port of upstream `rpython/translator/interactive.py`.
//!
//! Upstream entry point is `class Translation` at
//! `interactive.py:12-26`:
//!
//! ```python
//! class Translation(object):
//!
//!     def __init__(self, entry_point, argtypes=None, **kwds):
//!         self.driver = driver.TranslationDriver(overrides=DEFAULTS)
//!         self.config = self.driver.config
//!
//!         self.entry_point = export_symbol(entry_point)
//!         self.context = TranslationContext(config=self.config)
//!
//!         policy = kwds.pop('policy', None)
//!         self.update_options(kwds)
//!         self.ensure_setup(argtypes, policy)
//!         # for t.view() to work just after construction
//!         graph = self.context.buildflowgraph(entry_point)
//!         self.context._prebuilt_graphs[entry_point] = graph
//! ```
//!
//! Two parity adaptations apply here:
//!
//! 1. **Rust-source entry** (PRE-EXISTING-ADAPTATION, unavoidable). The
//!    upstream line 25 call `buildflowgraph(entry_point)` compiles Python
//!    bytecode. pyre-interpreter's source is Rust, so the same call has
//!    no bytecode to run on. Replaced by `build_host_function_from_rust`
//!    — the Position-2 Rust-AST adapter (see
//!    `flowspace/rust_source/mod.rs`) — which produces the same
//!    `(HostObject, PyGraph)` shape line 25 would have returned. Line
//!    26's `_prebuilt_graphs[entry_point] = graph` seed is performed
//!    verbatim after.
//!
//! 2. **Driver-backed methods** (DEFERRED). Upstream line 15 creates
//!    `TranslationDriver(overrides=DEFAULTS)`. Nearly every method on
//!    `Translation` outside `__init__` / `view` / `viewcg` forwards to
//!    `self.driver.*` (`annotate`, `rtype`, `backendopt`, `source`,
//!    `source_c`, `source_cl`, `compile`, `compile_c`, `disable`,
//!    `set_backend_extra_options`, `ensure_opt`, `ensure_type_system`,
//!    `ensure_backend`). Those require porting
//!    `rpython/translator/driver.py` (631 LOC) first — out of scope
//!    for M2.5 / M3.
//!    *Convergence path*: once `translator::driver` lands, a single
//!    `pub driver: Rc<TranslationDriver>` field plus the 13 forwarding
//!    methods will make this wrapper byte-for-byte equivalent with
//!    upstream.
//!
//! The `export_symbol(entry_point)` call at upstream line 18 is
//! ported in-place via [`crate::rlib::entrypoint::export_symbol`] —
//! it flips the `GraphFunc.exported_symbol` flag before the
//! `HostObject` is stored in `self.entry_point`, matching upstream
//! `entrypoint.py:10-12`.
//!
//! Fields ported here match upstream line-for-line except for
//! `driver`:
//!
//! | upstream field                             | local              |
//! |--------------------------------------------|--------------------|
//! | `self.driver`                              | DEFERRED           |
//! | `self.config`                              | `self.config()`    |
//! | `self.entry_point = export_symbol(…)`      | `self.entry_point` |
//! | `self.context = TranslationContext(…)`    | `self.context`     |
//! | `self.ann_argtypes` (set by ensure_setup)  | `self.ann_argtypes` |
//! | `self.ann_policy` (set by ensure_setup)    | `self.ann_policy`  |

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use quote::ToTokens;
use syn::ItemFn;

use crate::annotator::policy::AnnotatorPolicy;
use crate::annotator::signature::AnnotationSpec;
use crate::config::config::{Config, ConfigError, OptionValue};
use crate::flowspace::model::HostObject;
use crate::flowspace::rust_source::{AdapterError, build_host_function_from_rust};
use crate::rlib::entrypoint::export_symbol;
use crate::translator::driver::TranslationDriver;
use crate::translator::simplify;
use crate::translator::translator::{FlowingFlags, TranslationConfig, TranslationContext};

/// Top-level construction error for [`Translation::from_rust_item_fn`]
/// and the `_with_source` variant. Carries either an [`AdapterError`]
/// (raised by `build_host_function_from_rust`) or a [`ConfigError`]
/// (raised by `TranslationDriver::new`'s overrides plumbing). No
/// upstream counterpart — the upstream constructor relies on Python
/// raise-on-error.
#[derive(Debug)]
pub enum TranslationConstructError {
    Adapter(AdapterError),
    Config(ConfigError),
}

impl std::fmt::Display for TranslationConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslationConstructError::Adapter(e) => write!(f, "{e}"),
            TranslationConstructError::Config(e) => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for TranslationConstructError {}

impl From<AdapterError> for TranslationConstructError {
    fn from(e: AdapterError) -> Self {
        TranslationConstructError::Adapter(e)
    }
}

impl From<ConfigError> for TranslationConstructError {
    fn from(e: ConfigError) -> Self {
        TranslationConstructError::Config(e)
    }
}

/// Rust port of upstream `interactive.py:12 Translation`.
///
/// Every field except `driver` mirrors the upstream `__init__` body
/// at `interactive.py:14-26`; `driver`-dependent methods (`annotate`,
/// `rtype`, `backendopt`, `source`, `compile`, `disable`,
/// `set_backend_extra_options`, `ensure_*`) remain DEFERRED until
/// `translator::driver` ports. See the module docstring.
pub struct Translation {
    /// Upstream `self.driver = driver.TranslationDriver(overrides=DEFAULTS)`
    /// at `interactive.py:15`. Drives every `annotate` / `rtype` /
    /// `compile` step and owns the authoritative `Rc<Config>` that
    /// [`Self::update_options`] mutates and [`Self::ensure_setup`]
    /// forwards through.
    pub driver: Rc<TranslationDriver>,

    /// Upstream `self.entry_point = export_symbol(entry_point)` at
    /// `interactive.py:18`. `export_symbol` in
    /// `rpython/rlib/entrypoint.py:10-12` sets
    /// `func.exported_symbol = True` and returns the function; the
    /// Rust port routes through [`crate::rlib::entrypoint::export_symbol`]
    /// which applies the same flip on the wrapped `GraphFunc` before
    /// it is stored here, preserving the surface parity even though
    /// the flag is C-backend-only in upstream.
    pub entry_point: HostObject,

    /// Upstream `self.context = TranslationContext(config=self.config)`
    /// at `interactive.py:19`. Held as `Rc` so `RPythonAnnotator` can
    /// share the same context instance via `new_with_translator`.
    pub context: Rc<TranslationContext>,

    /// Upstream `self.ann_argtypes` at `interactive.py:37`. Populated
    /// by [`Self::ensure_setup`] which forwards the value through to
    /// `self.driver.setup(...)` and also writes here for direct
    /// readers in this module.
    pub ann_argtypes: RefCell<Option<Vec<AnnotationSpec>>>,

    /// Upstream `self.ann_policy` at `interactive.py:38`. Stored as
    /// `Option<AnnotatorPolicy>` matching the driver's `policy` slot
    /// (`driver.rs:setup`'s `policy: Option<AnnotatorPolicy>`); the
    /// previous string placeholder is gone now that the driver owns
    /// the canonical type.
    pub ann_policy: RefCell<Option<AnnotatorPolicy>>,
}

impl Translation {
    /// Accessor matching upstream `self.config` at `interactive.py:16`.
    /// Upstream binds `self.config = self.driver.config`, so reads
    /// after construction observe every `update_options` write
    /// through the same `Rc<Config>`. Returning the driver's
    /// `Rc<Config>` directly preserves that aliasing — the previous
    /// version returned a typed `&TranslationConfig` snapshot taken
    /// at construction time, which silently diverged once
    /// `update_options` mutated the driver state.
    pub fn config(&self) -> &Rc<Config> {
        &self.driver.config
    }

    /// Port of upstream `Translation.view()` at `interactive.py:28-29`.
    /// Upstream forwards to `TranslationContext.view()` at
    /// `translator.py:122-126`, which constructs a
    /// `translator.tool.graphpage.FlowGraphPage(self).display()` —
    /// a pygame / graphviz viewer.
    ///
    /// Neither `translator.tool.graphpage` nor the pygame stack is
    /// ported locally, so this method is structurally a no-op. The
    /// method itself is present to keep the upstream Translation
    /// surface parity complete — caller code that writes
    /// `translation.view()` compiles and runs silently rather than
    /// surfacing a missing-method compile error.
    pub fn view(&self) {
        // Intentional no-op. See upstream `translator.py:122-126`
        // for the full FlowGraphPage call; `translator.tool.graphpage`
        // depends on pygame + graphviz which are not part of majit's
        // runtime footprint.
    }

    /// Port of upstream `Translation.viewcg()` at
    /// `interactive.py:31-32`. Forwards to
    /// `TranslationContext.viewcg(center_graph, huge)` at
    /// `translator.py:130-134`. Same no-op parity as [`Self::view`];
    /// the pygame/graphviz stack is out of scope.
    pub fn viewcg(&self) {
        // Intentional no-op; see `view()` docstring for the shared
        // graphpage dependency.
    }

    /// Port of upstream `Translation.ensure_setup()` at
    /// `interactive.py:34-38`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def ensure_setup(self, argtypes=None, policy=None):
    ///     self.driver.setup(self.entry_point, argtypes, policy,
    ///                       empty_translator=self.context)
    ///     self.ann_argtypes = argtypes
    ///     self.ann_policy = policy
    /// ```
    ///
    /// Forwards `argtypes` / `policy` into
    /// [`TranslationDriver::setup`] with `empty_translator=self.context`,
    /// matching upstream lines 35-36 exactly. The pure-storage lines
    /// 37-38 (`self.ann_argtypes = argtypes; self.ann_policy = policy`)
    /// run after the driver call so a `setup` failure leaves both
    /// slots untouched.
    ///
    /// Returns `Result` rather than panicking on driver-side errors
    /// (e.g. a malformed secondary-entrypoints CSV inside
    /// `setup`'s `:201-208` lookup) because `Translation`'s public
    /// surface needs to surface them to the caller — the upstream
    /// `def ensure_setup` returns nothing only because Python uses
    /// raise-on-error semantics throughout.
    pub fn ensure_setup(
        &self,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
    ) -> Result<(), crate::translator::tool::taskengine::TaskError> {
        // Upstream `:35-36`: `self.driver.setup(self.entry_point,
        // argtypes, policy, empty_translator=self.context)`.
        self.driver.setup(
            Some(self.entry_point.clone()),
            argtypes.clone(),
            policy.clone(),
            HashMap::new(),
            Some(self.context.clone()),
        )?;
        // Upstream `:37-38`: pure-storage post-conditions.
        *self.ann_argtypes.borrow_mut() = argtypes;
        *self.ann_policy.borrow_mut() = policy;
        Ok(())
    }

    /// Port of upstream `Translation.update_options()` at
    /// `interactive.py:40-44`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def update_options(self, kwds):
    ///     gc = kwds.pop('gc', None)
    ///     if gc:
    ///         self.config.translation.gc = gc
    ///     self.config.translation.set(**kwds)
    /// ```
    ///
    /// Routes `kwds` through `self.driver.config` so option overrides
    /// take effect on the driver's authoritative `Rc<Config>`. The
    /// `gc` short-circuit at upstream `:41-43` handles the explicit
    /// `translation.gc` rename; everything else is fed to
    /// [`crate::config::config::Config::set`] which mirrors upstream's
    /// `Config.set(**kwds)` from `rpython/config/config.py:129-143`.
    ///
    /// Returns `Result` so the caller sees `ConfigError` rather than
    /// a silent drop. Upstream `update_options` raises directly out
    /// of `Config.set`; the Rust port surfaces the same lift point
    /// through `Result::Err`.
    pub fn update_options(
        &self,
        kwds: &HashMap<String, String>,
    ) -> Result<(), crate::config::config::ConfigError> {
        // Upstream `:41-43`: pop `gc` and special-case it as
        // `translation.gc`.
        let mut remaining: HashMap<String, OptionValue> = HashMap::with_capacity(kwds.len());
        for (k, v) in kwds.iter() {
            if k == "gc" {
                self.driver
                    .config
                    .set_value("translation.gc", OptionValue::Choice(v.clone()))?;
                continue;
            }
            remaining.insert(k.clone(), OptionValue::Str(v.clone()));
        }
        // Upstream `:44`: `self.config.translation.set(**kwds)`.
        // `Config::set` at `config.rs` walks the option tree by name,
        // matching `rpython/config/config.py:129-143`.
        if !remaining.is_empty() {
            self.driver.config.set(remaining)?;
        }
        Ok(())
    }

    /// Port of upstream `Translation.__init__` specialized for a
    /// Rust-source entry. Mirrors the `buildflowgraph(entry_point)` +
    /// `_prebuilt_graphs[entry_point] = graph` pair at
    /// `interactive.py:25-26`, replacing the bytecode-requiring
    /// `buildflowgraph` with `build_host_function_from_rust`.
    ///
    /// Returns the `Translation` wrapper plus the synthesized
    /// `HostObject` so the caller can pass it to downstream
    /// `RPythonAnnotator::build_types` (`annrpython.py:73-97`).
    ///
    /// Source filename falls back to the `<rust-source>` sentinel
    /// (syn has no stable source-path accessor, so this stays a
    /// sentinel unless the caller threads a real path through
    /// [`Self::from_rust_item_fn_with_source`]).
    ///
    /// Source text is auto-populated from `quote::ToTokens` — every
    /// `syn::ItemFn` round-trips through `to_token_stream().to_string()`
    /// to produce the textual rendering that upstream
    /// `inspect.getsource(func)` at `bytecode.py:50` would return.
    /// The rendering is whitespace-collapsed but semantically
    /// complete, which is the invariant upstream relies on at
    /// `model.py:35-47` (`FunctionGraph.source` setter) +
    /// `tool/error.rs:300` (graph-render error path).
    pub fn from_rust_item_fn(
        item: &ItemFn,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        // `quote::ToTokens` renders the original syntax back to text
        // without the caller having to keep the source string
        // around. The output is not pretty-printed but is the
        // canonical stable stringification for error rendering.
        let source_text = item.to_token_stream().to_string();
        Self::from_rust_item_fn_with_source(item, None, Some(&source_text))
    }

    /// Same as [`Self::from_rust_item_fn`] but threads upstream's
    /// `argtypes` / `policy` / `**kwds` parameters from
    /// `interactive.py:14 def __init__(self, entry_point,
    /// argtypes=None, **kwds)`. Drives the same `update_options(kwds)`
    /// + `ensure_setup(argtypes, policy)` chain as upstream
    /// `interactive.py:21-23`. Use the bare [`Self::from_rust_item_fn`]
    /// form when defaults (`argtypes=None`, `policy=None`, empty
    /// `kwds`) are sufficient.
    pub fn from_rust_item_fn_with_options(
        item: &ItemFn,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: &HashMap<String, String>,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        let source_text = item.to_token_stream().to_string();
        Self::from_rust_item_fn_with_source_and_options(
            item,
            None,
            Some(&source_text),
            argtypes,
            policy,
            kwds,
        )
    }

    /// Same as [`Self::from_rust_item_fn`] but threads caller-supplied
    /// source metadata. Upstream reads `func.__code__.co_filename` at
    /// `model.py:54` (filename) and the source text comes from
    /// `inspect.getsource(func)` at `bytecode.py:50` (GraphFunc.source)
    /// / the `FunctionGraph.source` property at `model.py:35-47`
    /// (`_source`). Both populate the `graph.source()` fast path
    /// `tool/error.rs:300` uses for graph-render error messages.
    pub fn from_rust_item_fn_with_source(
        item: &ItemFn,
        source_filename: Option<&str>,
        source_text: Option<&str>,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        Self::from_rust_item_fn_with_source_and_options(
            item,
            source_filename,
            source_text,
            None,
            None,
            &HashMap::new(),
        )
    }

    /// Full-parity counterpart of [`Self::from_rust_item_fn_with_source`]
    /// that mirrors upstream `interactive.py:14-26 __init__` end-to-end:
    /// driver → entry_point → `update_options(kwds)` → context (with
    /// `config=driver.config`) → `ensure_setup(argtypes, policy)` →
    /// `_prebuilt_graphs[entry_point] = graph`.
    ///
    /// Order note: upstream constructs `self.context` BEFORE
    /// `update_options`, but Python's `Config` is a live `Rc<Config>`
    /// shared with the context, so post-update reads see the new
    /// values automatically. The local typed `TranslationConfig`
    /// inside `TranslationContext` is a snapshot, so the snapshot
    /// must be taken AFTER `update_options` to observe the same
    /// post-construction state. Snapshotting later is a port-internal
    /// reordering — observable behaviour matches upstream.
    pub fn from_rust_item_fn_with_source_and_options(
        item: &ItemFn,
        source_filename: Option<&str>,
        source_text: Option<&str>,
        argtypes: Option<Vec<AnnotationSpec>>,
        policy: Option<AnnotatorPolicy>,
        kwds: &HashMap<String, String>,
    ) -> Result<(Self, HostObject), TranslationConstructError> {
        // Upstream `interactive.py:15`:
        // `self.driver = driver.TranslationDriver(overrides=DEFAULTS)`.
        // `DEFAULTS` at upstream `:6-10` sets `translation.verbose = True`
        // and leaves `backend` / `type_system` at `None`. Pass them
        // through `overrides=` so the driver's `Rc<Config>` carries the
        // upstream-observable post-construction state.
        let mut overrides: HashMap<String, OptionValue> = HashMap::new();
        overrides.insert("translation.verbose".to_string(), OptionValue::Bool(true));
        overrides.insert("translation.backend".to_string(), OptionValue::None);
        overrides.insert("translation.type_system".to_string(), OptionValue::None);
        let driver =
            TranslationDriver::new(None, None, Vec::new(), None, None, None, Some(overrides))?;

        // `build_host_function_from_rust` is the Rust-source analogue
        // of upstream `buildflowgraph(entry_point)` at
        // `interactive.py:25`. Output shape is the same
        // `(HostObject, PyGraph)` pair.
        let (host, pygraph) = build_host_function_from_rust(item, source_filename, source_text)?;

        // Upstream `interactive.py:21-22`: `policy = kwds.pop('policy',
        // None); self.update_options(kwds)`. The Rust port already
        // received `policy` as a typed parameter — `kwds` is everything
        // else and goes straight to `Config::set` via `update_options`.
        update_options_via_driver(&driver, kwds)?;

        // Upstream `interactive.py:19`: `self.context =
        // TranslationContext(config=self.config)`. Snapshot
        // `driver.config` into a typed `TranslationConfig` AFTER
        // `update_options` so any kwds-provided overrides land in the
        // snapshot, matching upstream's shared-Rc semantics.
        let translation_config = TranslationConfig::from_rc_config(&driver.config)?;
        let context = Rc::new(TranslationContext::with_config_and_flowing_flags(
            Some(translation_config),
            FlowingFlags::default(),
        ));

        // Upstream `interactive.py:25 buildflowgraph(entry_point)`
        // enters the non-prebuilt branch of
        // `translator.py:46-62 buildflowgraph`. That branch, after
        // `build_flow(func)` returns, runs
        // `simplify.simplify_graph(graph)` (translator.py:56),
        // optionally `detect_list_comprehension` (:57-58), and
        // finally `self.graphs.append(graph)` (:61). Our local port
        // reproduces the same sequence in `translator.rs:264-281`.
        // The adapter path skips `buildflowgraph` entirely (the
        // prebuilt short-circuit returns the graph we seed below),
        // so to keep parity with upstream's post-construction state
        // we run the same post-`build_flow` steps ourselves.
        {
            let g = pygraph.graph.borrow();
            simplify::simplify_graph(&g, None);
        }
        if context.config.translation.list_comprehension_operations {
            let g = pygraph.graph.borrow();
            simplify::detect_list_comprehension(&g);
        }
        context.graphs.borrow_mut().push(pygraph.graph.clone());

        // `self.entry_point = export_symbol(entry_point)` —
        // upstream `interactive.py:18`. Upstream mutates the Python
        // function in place and returns the same object; the Rust
        // port emits a new `HostObject` wrapper carrying a flagged
        // `GraphFunc` (see `rlib/entrypoint.rs`). The flag applies
        // before `_prebuilt_graphs[entry_point] = graph` seeds the
        // cache so the map key is the post-flag `HostObject`, exactly
        // what the caller later looks up through `buildflowgraph(entry_point)`.
        //
        // Side-effect caveat (PRE-EXISTING-ADAPTATION): upstream's
        // in-place mutation also reaches the GraphFunc referenced by
        // `pygraph.func` and `pygraph.graph.func` (Python: same
        // object). The Rust port builds those two slots as
        // independent `Clone` copies inside
        // `build_host_function_from_rust` (`register.rs:155,:165,:171`),
        // so flipping the flag on `flagged_host`'s GraphFunc does
        // not propagate there. Observable divergence is zero under
        // every currently-ported consumer — the flag is C-backend-
        // only per `rpython/translator/c/database.py`. When the C
        // backend or `register.rs`'s GraphFunc-sharing surface is
        // ported, flag those slots too so parity is complete.
        let flagged_host = export_symbol(host);
        context
            ._prebuilt_graphs
            .borrow_mut()
            .insert(flagged_host.clone(), pygraph);

        let translation = Translation {
            driver,
            entry_point: flagged_host.clone(),
            context,
            ann_argtypes: RefCell::new(None),
            ann_policy: RefCell::new(None),
        };

        // Upstream `interactive.py:23`: `self.ensure_setup(argtypes,
        // policy)`. Runs `driver.setup(self.entry_point, argtypes,
        // policy, empty_translator=self.context)` and persists
        // `ann_argtypes` / `ann_policy` slots, completing the
        // upstream-observable post-construction state.
        translation
            .ensure_setup(argtypes, policy)
            .map_err(|e| TranslationConstructError::Config(ConfigError::Generic(e.message)))?;

        Ok((translation, flagged_host))
    }
}

/// Mirrors `Translation::update_options` for use BEFORE the
/// `Translation` struct is fully constructed (the constructor needs
/// to apply kwds to `driver.config` before snapshotting it into the
/// context). Logic stays identical to the public method.
fn update_options_via_driver(
    driver: &TranslationDriver,
    kwds: &HashMap<String, String>,
) -> Result<(), ConfigError> {
    let mut remaining: HashMap<String, OptionValue> = HashMap::with_capacity(kwds.len());
    for (k, v) in kwds.iter() {
        if k == "gc" {
            driver
                .config
                .set_value("translation.gc", OptionValue::Choice(v.clone()))?;
            continue;
        }
        remaining.insert(k.clone(), OptionValue::Str(v.clone()));
    }
    if !remaining.is_empty() {
        driver.config.set(remaining)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_item_fn(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    #[test]
    fn from_rust_item_fn_seeds_prebuilt_graphs_exactly_once() {
        // Mirrors upstream `interactive.py:25-26` behaviour: the entry
        // point's graph is cached in `_prebuilt_graphs` after
        // construction. The subsequent `buildflowgraph(entry_point)`
        // call pops it per `translator.py:50-51`, so the cache is
        // empty after one consumption.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

        assert!(
            t.context._prebuilt_graphs.borrow().contains_key(&host),
            "entry point must be seeded after construction"
        );

        let pg = t
            .context
            .buildflowgraph(host.clone(), false)
            .expect("buildflowgraph hits the prebuilt cache");
        // Pop-on-hit semantic: second lookup must not find it.
        assert!(!t.context._prebuilt_graphs.borrow().contains_key(&host));
        // And the returned PyGraph is the one the adapter built.
        assert_eq!(pg.func.id, host.user_function().expect("user fn").id);
    }

    #[test]
    fn from_rust_item_fn_propagates_adapter_errors() {
        // Any `AdapterError` from `build_host_function_from_rust` must
        // bubble up — the wrapper does not swallow errors. Use an
        // unsupported signature (const generic) as a concrete trigger;
        // `build_flow.rs:138-141` rejects it with
        // `InvalidSignature { reason: "const generic parameter …" }`.
        let item = parse_item_fn("fn f<const N: usize>() -> i64 { 0 }");
        match Translation::from_rust_item_fn(&item) {
            Ok(_) => panic!("const generic must reject"),
            Err(TranslationConstructError::Adapter(AdapterError::InvalidSignature { reason })) => {
                assert!(reason.contains("const generic"), "reason: {reason}");
            }
            Err(other) => panic!("expected Adapter(InvalidSignature), got {other:?}"),
        }
    }

    #[test]
    fn from_rust_item_fn_default_verbose_true() {
        // Upstream `interactive.py:6 DEFAULTS = {..., 'translation.verbose': True}`
        // is passed to `driver.TranslationDriver(overrides=DEFAULTS)` at line 15;
        // `self.config = self.driver.config` at line 16 picks up the override.
        // The Rust port applies the single driver-equivalent default via
        // `FlowingFlags { verbose: Some(true) }`.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert!(
            t.context.config.translation.verbose,
            "interactive.py:6 DEFAULTS sets translation.verbose = True"
        );
    }

    #[test]
    fn from_rust_item_fn_appends_entry_graph_to_context_graphs() {
        // Upstream `interactive.py:25 buildflowgraph(entry_point)` goes
        // through the non-prebuilt branch of `translator.py:46-62`,
        // which at line 61 does `self.graphs.append(graph)`. The
        // adapter path skips `buildflowgraph`, so the append is
        // performed explicitly — this test pins the parity.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let graphs = t.context.graphs.borrow();
        assert_eq!(
            graphs.len(),
            1,
            "entry graph must be appended — matches upstream translator.py:61"
        );

        // The appended graph is the SAME underlying FunctionGraph the
        // PyGraph wraps (Rc pointer-equal), so `t.view()`-style
        // operations that index through `context.graphs` find it.
        let seeded_pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        assert!(
            Rc::ptr_eq(&graphs[0], &seeded_pygraph.graph),
            "context.graphs entry must be Rc-identical to the seeded PyGraph's graph"
        );
    }

    #[test]
    fn from_rust_item_fn_runs_simplify_graph() {
        // Upstream `translator.py:56 simplify_graph(graph)` runs on
        // the result of `build_flow(func)` before the graph is
        // appended to `self.graphs`. simplify_graph is idempotent
        // (eliminate_empty_blocks / join_blocks / remove_identical_vars
        // converge on a fixed point), so a second `simplify_graph`
        // call on a post-construction graph must be a structural
        // no-op. Exercise that: snapshot the block/exit count, run
        // simplify again, confirm the shape is unchanged.
        let item = parse_item_fn(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
                2
            }",
        );
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let blocks_before: Vec<_> = {
            let g = pygraph.graph.borrow();
            let mut out = Vec::new();
            let mut stack = vec![g.startblock.clone()];
            let mut seen = std::collections::HashSet::new();
            while let Some(b) = stack.pop() {
                let b_ptr = std::rc::Rc::as_ptr(&b) as usize;
                if !seen.insert(b_ptr) {
                    continue;
                }
                let blk = b.borrow();
                out.push((b_ptr, blk.operations.len(), blk.exits.len()));
                for exit in &blk.exits {
                    if let Some(tgt) = exit.borrow().target.clone() {
                        stack.push(tgt);
                    }
                }
            }
            out.sort();
            out
        };

        // Run simplify again — if `from_rust_item_fn` skipped
        // upstream's `simplify_graph(graph)` step, simplify would
        // actually change something here.
        {
            let g = pygraph.graph.borrow();
            crate::translator::simplify::simplify_graph(&g, None);
        }

        let blocks_after: Vec<_> = {
            let g = pygraph.graph.borrow();
            let mut out = Vec::new();
            let mut stack = vec![g.startblock.clone()];
            let mut seen = std::collections::HashSet::new();
            while let Some(b) = stack.pop() {
                let b_ptr = std::rc::Rc::as_ptr(&b) as usize;
                if !seen.insert(b_ptr) {
                    continue;
                }
                let blk = b.borrow();
                out.push((b_ptr, blk.operations.len(), blk.exits.len()));
                for exit in &blk.exits {
                    if let Some(tgt) = exit.borrow().target.clone() {
                        stack.push(tgt);
                    }
                }
            }
            out.sort();
            out
        };

        assert_eq!(
            blocks_before, blocks_after,
            "simplify_graph must be idempotent after from_rust_item_fn — \
                if the second run changed the graph, the first simplify was skipped"
        );
    }

    #[test]
    fn from_rust_item_fn_with_source_threads_filename_into_co_filename() {
        // Upstream reads `func.__code__.co_filename` at `model.py:54`
        // (FunctionGraph.filename) for graph-render error messages.
        // The caller-supplied filename must land in HostCode.co_filename
        // instead of the `<rust-source>` sentinel.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (_t, host) = Translation::from_rust_item_fn_with_source(
            &item,
            Some("pyre/pyre-interpreter/src/pyopcode.rs"),
            None,
        )
        .expect("translation");

        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_filename, "pyre/pyre-interpreter/src/pyopcode.rs");
    }

    #[test]
    fn from_rust_item_fn_defaults_co_filename_to_rust_source_sentinel() {
        // When no source filename is threaded in, the sentinel remains
        // `<rust-source>`. Documented behaviour for fixture callers.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (_t, host) = Translation::from_rust_item_fn(&item).expect("translation");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_filename, "<rust-source>");
    }

    #[test]
    fn from_rust_item_fn_with_source_threads_source_text_to_graph_source() {
        // Upstream `model.py:35-47` exposes `FunctionGraph.source` as a
        // property; the Translation constructor uses it through
        // `graph.source()` on the error-rendering path
        // (`tool/error.rs:300-320`). Pre-patch, the adapter left both
        // `_source` and `GraphFunc.source` unset, so `graph.source()`
        // failed with the `"source not found"` error and the render
        // fell back to "no source!".
        let src = "fn one() -> i64 { 1 }";
        let item = parse_item_fn(src);
        let (t, _host) =
            Translation::from_rust_item_fn_with_source(&item, Some("fixture.rs"), Some(src))
                .expect("translation");

        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let graph = pygraph.graph.borrow();
        // `graph.source()` fast-path reads `_source`.
        assert_eq!(graph.source().expect("source populated"), src);
        // Fallback path: even if `_source` is cleared, `GraphFunc.source`
        // holds the same text so `inspect.getsource(func)`-style callers
        // still see it (model.rs:3207-3216).
        let gf_source = graph
            .func
            .as_ref()
            .expect("func set")
            .source
            .as_deref()
            .expect("GraphFunc.source populated");
        assert_eq!(gf_source, src);
    }

    #[test]
    fn from_rust_item_fn_auto_populates_source_text_from_tokens() {
        // Default-path parity: `from_rust_item_fn` (no explicit
        // source_text) auto-populates `graph.source()` via
        // `quote::ToTokens::to_token_stream().to_string()`. Upstream
        // `inspect.getsource(func)` at `bytecode.py:50` returns the
        // verbatim source — the Rust equivalent round-trips through
        // syn's token stream, which is whitespace-collapsed but
        // semantically complete (`fn one () -> i64 { 1 }`). What
        // matters for `tool/error.rs:300` is that `graph.source()`
        // succeeds, not the exact whitespace layout.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let pygraph = t
            .context
            ._prebuilt_graphs
            .borrow()
            .values()
            .next()
            .expect("seeded entry")
            .clone();
        let graph = pygraph.graph.borrow();
        let src = graph.source().expect("auto-populated via ToTokens");
        assert!(
            src.contains("fn one") && src.contains("-> i64"),
            "token-stream render should contain fn signature: {src}"
        );
        // `GraphFunc.source` carries the same text (fallback path).
        let gf_source = graph
            .func
            .as_ref()
            .expect("func set")
            .source
            .as_deref()
            .expect("GraphFunc.source populated by default");
        assert_eq!(gf_source, src);
    }

    #[test]
    fn ensure_setup_stores_argtypes_and_policy_via_driver() {
        // Upstream `interactive.py:34-38` — `ensure_setup` calls
        // `self.driver.setup(self.entry_point, argtypes, policy,
        // empty_translator=self.context)` and then stores
        // `self.ann_argtypes = argtypes` / `self.ann_policy = policy`.
        // Round-trip both slots to confirm the storage half ran AND
        // the driver-side slot observes the same translator.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let policy = AnnotatorPolicy::default();
        t.ensure_setup(Some(vec![]), Some(policy))
            .expect("ensure_setup");
        assert!(t.ann_argtypes.borrow().is_some());
        assert!(t.ann_policy.borrow().is_some());
        // Driver's translator slot must point at this Translation's
        // context (`empty_translator=self.context` parity).
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "driver.translator must alias self.context after ensure_setup"
        );
    }

    #[test]
    fn from_rust_item_fn_with_options_runs_update_and_ensure_setup() {
        // Upstream `interactive.py:14-26 __init__` runs
        // `update_options(kwds)` + `ensure_setup(argtypes, policy)`
        // before returning. Confirm the constructor variant that
        // accepts these parameters drives the same chain end-to-end.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let mut kwds = HashMap::new();
        kwds.insert("gc".to_string(), "boehm".to_string());
        let argtypes: Vec<AnnotationSpec> = Vec::new();
        let policy = AnnotatorPolicy::default();
        let (t, _host) =
            Translation::from_rust_item_fn_with_options(&item, Some(argtypes), Some(policy), &kwds)
                .expect("translation");

        // `update_options(kwds)` landed: `translation.gc=boehm` is
        // visible on the live `Rc<Config>`.
        let gc = match t.config().get("translation.gc").expect("translation.gc") {
            crate::config::config::ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(gc, "boehm");

        // `ensure_setup` landed: `ann_argtypes` / `ann_policy` are
        // populated and the driver knows about the context.
        assert!(t.ann_argtypes.borrow().is_some());
        assert!(t.ann_policy.borrow().is_some());
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "constructor must thread self.context through driver.setup"
        );
    }

    #[test]
    fn from_rust_item_fn_default_runs_ensure_setup_with_none_args() {
        // Upstream `interactive.py:14`: `argtypes=None` default + empty
        // `**kwds`. The bare `from_rust_item_fn` constructor must still
        // run `ensure_setup(None, None)` so the driver's `translator`
        // slot is populated (downstream callers expect
        // `t.driver.translator` to alias `t.context`).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        let driver_translator = t.driver.translator.borrow().as_ref().map(Rc::clone);
        assert!(
            driver_translator
                .as_ref()
                .map(|c| Rc::ptr_eq(c, &t.context))
                .unwrap_or(false),
            "default constructor must wire driver.translator → self.context via ensure_setup"
        );
    }

    #[test]
    fn view_and_viewcg_are_no_op_safe_to_call() {
        // Upstream `view()` / `viewcg()` open pygame windows; the
        // Rust port has no pygame dependency, so both methods are
        // no-op stubs. Test invariant: calling them on a freshly-
        // constructed Translation does not panic / error (regression
        // check for accidentally changing the methods' safety
        // contract).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        t.view();
        t.viewcg();
    }

    #[test]
    fn update_options_routes_kwds_through_driver_config() {
        // Upstream `interactive.py:40-44`:
        //
        //     gc = kwds.pop('gc', None)
        //     if gc:
        //         self.config.translation.gc = gc
        //     self.config.translation.set(**kwds)
        //
        // Empty kwds = no-op. `gc` is special-cased through the
        // ChoiceOption setter; everything else flows through
        // `Config::set`.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        // Empty: no error, config unchanged.
        t.update_options(&HashMap::new()).expect("empty update");
        let verbose = match t
            .driver
            .config
            .get("translation.verbose")
            .expect("translation.verbose")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Bool(b)) => b,
            other => panic!("expected Bool, got {other:?}"),
        };
        assert!(
            verbose,
            "DEFAULTS at interactive.py:6 must keep verbose=True"
        );

        // `gc` short-circuit lands as ChoiceOption.
        let mut kwds = HashMap::new();
        kwds.insert("gc".to_string(), "boehm".to_string());
        t.update_options(&kwds).expect("gc update");
        let gc = match t
            .driver
            .config
            .get("translation.gc")
            .expect("translation.gc")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Choice(s)) => s,
            other => panic!("expected Choice, got {other:?}"),
        };
        assert_eq!(gc, "boehm");
    }

    #[test]
    fn from_rust_item_fn_flags_entry_point_as_exported() {
        // Upstream `interactive.py:18 self.entry_point =
        // export_symbol(entry_point)` sets
        // `entry_point.exported_symbol = True`. The Rust port flips the
        // flag on `GraphFunc.exported_symbol` of the HostObject stored
        // in `self.entry_point`; verify both that slot AND the
        // `_prebuilt_graphs` key carry the flagged HostObject (parity
        // with upstream's Python object-identity — the map key and
        // `self.entry_point` share the single flagged function object).
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

        // `self.entry_point` carries the flag.
        let gf = t.entry_point.user_function().expect("user function");
        assert!(
            gf.exported_symbol,
            "interactive.py:18 export_symbol(entry_point) must set exported_symbol=true"
        );

        // Returned host is the same flagged HostObject — tests that
        // round-trip through the `_prebuilt_graphs` cache must look up
        // via this host.
        assert_eq!(t.entry_point, host);
        assert!(
            t.context._prebuilt_graphs.borrow().contains_key(&host),
            "prebuilt cache key matches the returned (flagged) host"
        );
    }

    #[test]
    fn translation_carries_entry_point_and_placeholder_annotation_slots() {
        // Upstream `interactive.py:18, :37-38` — `Translation` carries
        // `entry_point` (the export_symbol-flagged function), plus
        // `ann_argtypes` / `ann_policy` set by `ensure_setup`. The
        // Rust port stores the HostObject directly in `entry_point`
        // (the `exported_symbol` flag itself is a C-backend detail —
        // see module docstring) and leaves the annotation slots
        // `None` until the driver-dependent `ensure_setup` port
        // lands.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert_eq!(t.entry_point, host);
        assert!(t.ann_argtypes.borrow().is_none());
        assert!(t.ann_policy.borrow().is_none());
    }

    #[test]
    fn translation_config_accessor_matches_driver_config() {
        // Upstream `interactive.py:16`: `self.config = self.driver.config`.
        // Our `config()` accessor returns the same `Rc<Config>` the
        // driver owns; `Rc::ptr_eq` confirms reference identity.
        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");
        assert!(Rc::ptr_eq(t.config(), &t.driver.config));
        let verbose = match t
            .config()
            .get("translation.verbose")
            .expect("translation.verbose")
        {
            crate::config::config::ConfigValue::Value(OptionValue::Bool(b)) => b,
            other => panic!("expected Bool, got {other:?}"),
        };
        assert!(
            verbose,
            "DEFAULTS at interactive.py:6 must keep verbose=True"
        );
    }

    #[test]
    fn context_is_shared_identity_between_translation_and_annotator() {
        // `context` is `Rc<TranslationContext>` so annotator
        // constructors that take `Rc<TranslationContext>` observe the
        // same instance. This matches upstream: `self.context` on
        // `Translation` and the `translator` arg threaded into
        // `RPythonAnnotator` are literal `is`-identical (Python object
        // identity).
        use crate::annotator::annrpython::RPythonAnnotator;

        let item = parse_item_fn("fn one() -> i64 { 1 }");
        let (t, _host) = Translation::from_rust_item_fn(&item).expect("translation");

        let ann =
            RPythonAnnotator::new_with_translator(Some(Rc::clone(&t.context)), None, None, false);
        assert!(
            Rc::ptr_eq(&t.context, &ann.translator),
            "Translation.context and annotator.translator must be Rc-identical"
        );
    }
}
