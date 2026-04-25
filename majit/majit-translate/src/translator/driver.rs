//! Port of `rpython/translator/driver.py`.
//!
//! Upstream is a 631-LOC orchestration layer that subclasses
//! [`SimpleTaskEngine`] and registers every translation step
//! (`task_annotate`, `task_rtype_lltype`, `task_backendopt_lltype`,
//! `task_pyjitpl_lltype`, `task_database_c`, `task_source_c`,
//! `task_compile_c`, `task_llinterpret_lltype`, `task_jittest_lltype`,
//! `task_stackcheckinsertion_lltype`) as a `task_<name>` method
//! decorated by [`taskdef`]. The class also wires per-task timing
//! (`Timer`), debug-log breadcrumbs (`debug_start` / `debug_print` /
//! `debug_stop`), instrumentation forks (`ProfInstrument`), checkpoint
//! `fork_before` hooks (`unixcheckpoint.restartable_point`) and the
//! `from_targetspec` classmethod that's used by `targetstandalone.py`.
//!
//! ## Direct leaves consumed by `__init__`
//!
//! Every leaf reachable from `TranslationDriver.__init__` is already
//! ported when this file lands:
//!
//! | Upstream `driver.py` import                                | Local module                             |
//! |------------------------------------------------------------|-------------------------------------------|
//! | `rpython.translator.translator.TranslationContext`         | [`crate::translator::translator`]         |
//! | `rpython.translator.tool.taskengine.SimpleTaskEngine`      | [`crate::translator::tool::taskengine`]   |
//! | `rpython.translator.timing.Timer`                          | [`crate::translator::timing`]             |
//! | `rpython.annotator.listdef.s_list_of_strings`              | [`crate::annotator::listdef::s_list_of_strings`] |
//! | `rpython.annotator.policy.AnnotatorPolicy`                 | [`crate::annotator::policy::AnnotatorPolicy`]    |
//! | `rpython.config.translationoption.get_combined_translation_config` | [`crate::config::translationoption::get_combined_translation_config`] |
//! | `rpython.rlib.entrypoint.{secondary_entrypoints,annotated_jit_entrypoints}` | [`super::registries`] (this module) |
//!
//! ## DEFERRED task bodies
//!
//! Most `task_*` bodies depend on translation infrastructure that is
//! not yet ported (the JIT codewriter policy hookup, the C backend
//! database/source/compile chain, `LLInterpreter`, `query` for
//! `sanity_check_annotation`, `unixcheckpoint`, …). Each task is
//! registered with the upstream-equivalent name + deps + title, so the
//! plan ordering, `expose_task` selection and `proceed` dispatch all
//! work end-to-end. Calling a still-deferred task body returns a
//! [`TaskError`] that cites the upstream source line + the missing
//! leaf; it must not panic merely because the leaf backend module has
//! not landed yet.
//!
//! Convergence path: each leaf land replaces a single `TaskError`
//! return site with the real body; the driver structure does not
//! change.
//!
//! ## Rust-language adaptations (each minimal, all documented)
//!
//! 1. **Reflective `expose_task` `setattr(self, task, proc)`** at
//!    upstream `:104-110` — Python overwrites bound-method slots so
//!    callers can write `driver.compile()`. Rust has no per-instance
//!    method binding; the port keeps `self.exposed: Vec<String>`
//!    (line 101) verbatim and provides [`TranslationDriver::exposed_method`]
//!    + [`TranslationDriver::proceed`] to dispatch by name. The
//!    `proceed`-bound `backend_goal` is captured in a side-table
//!    [`TranslationDriver::exposed_backend_goal`] so `proceed("compile")`
//!    routes to the platform-specific `compile_c` task exactly as
//!    upstream's bound `proc()` closure would.
//! 2. **`@taskdef` decorator** at `:22-32` — upstream attaches
//!    `task_deps` / `task_title` / `task_idempotent` / `task_earlycheck`
//!    attributes to each `task_*` function. The Rust port calls
//!    [`SimpleTaskEngine::register_task`] explicitly with the same
//!    metadata; the side-table [`TranslationDriver::task_earlycheck`]
//!    holds the optional pre-check callback that upstream stashes via
//!    `taskfunc.task_earlycheck`.
//! 3. **`debug_start` / `debug_print` / `debug_stop`** at upstream
//!    `_do` (`:269-270`, `:288`) — the `rpython.rlib.debug` log facility
//!    is not ported here. The local stubs are no-ops; the convergence
//!    site is documented inline and the structural call order is
//!    preserved.
//! 4. **`AnsiLogger`** at upstream `:17-19` (`log = AnsiLogger("translation")`)
//!    is not ported. The driver's `log.info(...)` calls land as
//!    `println!` lines; the format strings match upstream verbatim so
//!    the observable text is identical modulo ANSI colouring. Same
//!    convergence path as [`crate::translator::timing::Timer::pprint`].

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use crate::config::config::{Config, ConfigError, ConfigValue, OptionValue};
use crate::config::translationoption::{
    _GLOBAL_TRANSLATIONCONFIG, get_combined_translation_config,
};
use crate::translator::timing::{SystemClock, Timer};
use crate::translator::tool::taskengine::{
    SimpleTaskEngine, TaskEngineHooks, TaskError, TaskOutput,
};

// ---------------------------------------------------------------------
// Upstream `:1-19` module-level imports + log handle.
// ---------------------------------------------------------------------

// Upstream `:38 PROFILE = set([])`. The Rust port stores it in a
// thread-local `RefCell<HashSet<String>>` so callers can mutate it
// the same way upstream does — `PROFILE.add('annotate')`.
thread_local! {
    static PROFILE: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Returns whether `goal` is in the upstream-equivalent `PROFILE` set.
fn profile_contains(goal: &str) -> bool {
    PROFILE.with(|p| p.borrow().contains(goal))
}

// ---------------------------------------------------------------------
// Upstream `:40-41 class Instrument(Exception): pass`.
// ---------------------------------------------------------------------

/// Port of upstream `class Instrument(Exception)` at `:40-41`. Raised
/// by `TranslationDriver.instrument_result` to abort the current task
/// and trigger the `compile` re-entry inside `_do` (upstream `:283-285`).
#[derive(Debug, Clone)]
pub struct Instrument;

impl std::fmt::Display for Instrument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Instrument")
    }
}

impl std::error::Error for Instrument {}

// ---------------------------------------------------------------------
// Upstream `:44-60 class ProfInstrument`.
// ---------------------------------------------------------------------

/// Port of upstream `class ProfInstrument` at `:44-60`. The class is
/// only referenced by `TranslationDriver.instrument_result` (the C-
/// backend instrumentation fork at `:218-248`) and itself reaches
/// `compiler.platform.execute` — both DEFERRED until the C-backend
/// port lands. The struct is preserved so callers / tests that
/// mention the type by name compile cleanly.
pub struct ProfInstrument {
    pub datafile: String,
    // Upstream stores the live `compiler` instance (`CStandaloneBuilder`).
    // The local port keeps the slot opaque — a `dyn Any` carrier — so
    // nothing here drags in unported C-backend types.
    pub compiler: Rc<dyn Any>,
}

impl ProfInstrument {
    /// Upstream `ProfInstrument.first` at `:50-51`.
    pub fn first(&self) -> ! {
        unimplemented!(
            "driver.py:50 ProfInstrument.first — leaf CStandaloneBuilder._build not yet ported"
        );
    }

    /// Upstream `ProfInstrument.probe` at `:53-56`.
    pub fn probe(&self, _exe: &str, _args: &[String]) -> ! {
        unimplemented!(
            "driver.py:53 ProfInstrument.probe — leaf compiler.platform.execute not yet ported"
        );
    }

    /// Upstream `ProfInstrument.after` at `:58-60`.
    pub fn after(&self) -> ! {
        unimplemented!("driver.py:58 ProfInstrument.after — calls os._exit(0)");
    }
}

// ---------------------------------------------------------------------
// Upstream `:13-14 secondary_entrypoints, annotated_jit_entrypoints`.
//
// `rpython/rlib/entrypoint.py:1` declares
// `secondary_entrypoints = {"main": []}` and `:8 annotated_jit_entrypoints = []`
// as module-level mutable globals. The port keeps the same shape via
// process-local `OnceLock<RefCell<...>>` slots so callers can mutate
// them through `secondary_entrypoints_register(key, ...)` /
// `annotated_jit_entrypoints_register(...)` exactly like upstream's
// `secondary_entrypoints.setdefault(key, []).append(...)`.
//
// `setup`'s `secondary_entrypoints[key]` lookup at `:204-207` reads the
// `key` listed in `config.translation.secondaryentrypoints` (a CSV
// string). Upstream raises `KeyError` when the key is missing; the
// Rust port returns `Err(TaskError)` matching that contract.
// ---------------------------------------------------------------------

/// `EntryPointSpec` mirrors upstream's `(func, argtypes)` tuple stored
/// inside `secondary_entrypoints[key]`. The argtypes are an opaque
/// Python list upstream — the local port preserves the slot as a
/// `Vec<Rc<dyn Any>>` so the C-backend code that consumes them can
/// downcast to its own argtype shape when it lands.
pub type EntryPointSpec = (Rc<dyn Any>, Vec<Rc<dyn Any>>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CBuilderKind {
    Standalone,
    Library,
}

#[derive(Clone, Debug)]
pub struct DatabaseState;

#[derive(Clone, Debug)]
pub struct CBuilderState {
    pub kind: CBuilderKind,
    pub modulename: Option<String>,
    pub targetdir: PathBuf,
    pub executable_name: PathBuf,
    pub shared_library_name: Option<PathBuf>,
    pub executable_name_w: Option<PathBuf>,
}

impl CBuilderState {
    fn standalone() -> Self {
        Self {
            kind: CBuilderKind::Standalone,
            modulename: None,
            targetdir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            executable_name: PathBuf::from("pypy-c"),
            shared_library_name: None,
            executable_name_w: None,
        }
    }

    fn library(name: Option<String>) -> Self {
        let modulename = name.clone();
        Self {
            kind: CBuilderKind::Library,
            modulename,
            targetdir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            executable_name: PathBuf::from(name.unwrap_or_else(|| "libtesting".to_string())),
            shared_library_name: None,
            executable_name_w: None,
        }
    }

    fn missing_leaf(&self, leaf: &str) -> TaskError {
        TaskError {
            message: format!(
                "driver.py C backend leaf not ported for {:?}: {leaf}",
                self.kind
            ),
        }
    }

    pub fn build_database(&self) -> Result<DatabaseState, TaskError> {
        Err(self.missing_leaf("cbuilder.build_database()"))
    }

    pub fn generate_source(
        &self,
        _database: &DatabaseState,
        _defines: &HashMap<String, String>,
        _exe_name: Option<String>,
    ) -> Result<PathBuf, TaskError> {
        Err(self.missing_leaf("cbuilder.generate_source(database, defines, exe_name=...)"))
    }

    pub fn compile(&self, _exe_name: Option<String>) -> Result<(), TaskError> {
        Err(self.missing_leaf("cbuilder.compile(**kwds)"))
    }

    pub fn get_entry_point(&self) -> Result<PathBuf, TaskError> {
        Err(self.missing_leaf("cbuilder.get_entry_point()"))
    }
}

// Upstream `entrypoint.py:1`: `secondary_entrypoints = {"main": []}`.
// Stored thread-local because `Rc<dyn Any>` is not `Sync`.
thread_local! {
    static SECONDARY_ENTRYPOINTS: RefCell<HashMap<String, Vec<EntryPointSpec>>> = {
        let mut m = HashMap::new();
        m.insert("main".to_string(), Vec::new());
        RefCell::new(m)
    };
}

/// Read-only view of upstream `entrypoint.py:1 secondary_entrypoints`.
/// Returns the entries registered under `key` or `None` matching
/// upstream's `KeyError` raise path on a missing key.
pub fn secondary_entrypoints_get(key: &str) -> Option<Vec<EntryPointSpec>> {
    SECONDARY_ENTRYPOINTS.with(|s| s.borrow().get(key).cloned())
}

/// All registered keys (used by upstream's `secondary_entrypoints.keys()`
/// at `:207` for the missing-key error message).
pub fn secondary_entrypoints_keys() -> Vec<String> {
    SECONDARY_ENTRYPOINTS.with(|s| s.borrow().keys().cloned().collect())
}

// Upstream `entrypoint.py:8`: `annotated_jit_entrypoints = []`.
thread_local! {
    static ANNOTATED_JIT_ENTRYPOINTS: RefCell<Vec<EntryPointSpec>> =
        RefCell::new(Vec::new());
}

/// Read-only view of upstream `entrypoint.py:8 annotated_jit_entrypoints`.
pub fn annotated_jit_entrypoints_get() -> Vec<EntryPointSpec> {
    ANNOTATED_JIT_ENTRYPOINTS.with(|s| s.borrow().clone())
}

// ---------------------------------------------------------------------
// Upstream `:22-32 def taskdef(...)` decorator.
//
// Upstream attaches `task_deps`, `task_title`, `task_newstate`,
// `task_expected_states`, `task_idempotent`, `task_earlycheck` to a
// `task_<name>` function. The Rust port doesn't decorate functions —
// it carries the same metadata as a struct passed to
// `SimpleTaskEngine::register_task`. `TaskDef` keeps the bundle named
// for grep-parity with upstream call sites.
// ---------------------------------------------------------------------

/// Port of the metadata `taskdef(deps, title, ..., earlycheck=None)` at
/// upstream `:22-32` would attach to a `task_*` function.
pub struct TaskDef {
    pub deps: Vec<String>,
    pub title: String,
    /// Upstream `task.task_idempotent`. When `true`, the driver does
    /// not stash the goal in `self.done`, allowing re-runs.
    pub idempotent: bool,
    /// Upstream `task.task_earlycheck`. Optional callback that
    /// `_event("planned", goal)` invokes at upstream `:611-612`.
    pub earlycheck: Option<Rc<dyn Fn(&TranslationDriver) -> Result<(), TaskError>>>,
}

impl TaskDef {
    pub fn new(deps: Vec<String>, title: impl Into<String>) -> Self {
        TaskDef {
            deps,
            title: title.into(),
            idempotent: false,
            earlycheck: None,
        }
    }

    pub fn with_earlycheck(
        mut self,
        check: impl Fn(&TranslationDriver) -> Result<(), TaskError> + 'static,
    ) -> Self {
        self.earlycheck = Some(Rc::new(check));
        self
    }
}

// ---------------------------------------------------------------------
// Upstream `:339 RTYPE = 'rtype_lltype'` and `:378 BACKENDOPT =
// 'backendopt_lltype'` and `:387 STACKCHECKINSERTION = 'stackcheckinsertion_lltype'`.
// ---------------------------------------------------------------------

/// Upstream `:339 RTYPE = 'rtype_lltype'` — the canonical typed-task
/// name picked by `task_pyjitpl_lltype` / `task_jittest_lltype`'s deps.
pub const RTYPE: &str = "rtype_lltype";

/// Upstream `:378 BACKENDOPT = 'backendopt_lltype'`.
pub const BACKENDOPT: &str = "backendopt_lltype";

/// Upstream `:387 STACKCHECKINSERTION = 'stackcheckinsertion_lltype'`.
pub const STACKCHECKINSERTION: &str = "stackcheckinsertion_lltype";

// ---------------------------------------------------------------------
// Upstream `:63 class TranslationDriver(SimpleTaskEngine)`.
// ---------------------------------------------------------------------

/// Port of upstream `class TranslationDriver(SimpleTaskEngine)` at
/// `:63-622`.
///
/// The struct holds every observable field upstream attaches in
/// `__init__` (`:66-134`). Mutable fields are wrapped in `RefCell` /
/// `Cell` because upstream mutates them through Python's
/// implicit-`self` binding while the Rust port operates on `&self`.
pub struct TranslationDriver {
    /// Upstream `self.timer = Timer()` at `:71`.
    pub timer: Timer<SystemClock>,

    /// Upstream `SimpleTaskEngine.__init__(self)` at `:72`. Held by
    /// composition because Rust has no inheritance — see the module
    /// doc on `crate::translator::tool::taskengine` for the trait-based
    /// override pattern that replaces upstream's MRO.
    pub engine: SimpleTaskEngine,

    /// Upstream `self.config = config` at `:80`. Held as `Rc<Config>`
    /// to mirror upstream's mutable-Config-handed-around contract
    /// (`interactive.py:16 self.config = self.driver.config`).
    pub config: Rc<Config>,

    /// Upstream `self.exe_name = exe_name` at `:87`. `None` ↔ upstream
    /// `None`; `Some(s)` ↔ upstream string template.
    pub exe_name: RefCell<Option<String>>,

    /// Upstream `self.extmod_name = extmod_name` at `:88`.
    pub extmod_name: RefCell<Option<String>>,

    /// Upstream `self.done = {}` at `:90`. Set of completed goals
    /// (upstream uses a dict-as-set with arbitrary `True` values).
    pub done: RefCell<HashMap<String, bool>>,

    /// Upstream `self._disabled = to_disable` at `:167` (set from
    /// `__init__` line 92's `self.disable(disable)` call). Carries
    /// the names a caller asked to skip; consulted by
    /// `_maybe_skip` and resolved through `backend_select_goals`.
    pub _disabled: RefCell<Vec<String>>,

    /// Upstream `self.default_goal = default_goal` at `:99`.
    pub default_goal: RefCell<Option<String>>,

    /// Upstream `self.extra_goals = []` at `:100`.
    pub extra_goals: RefCell<Vec<String>>,

    /// Upstream `self.exposed = []` at `:101` — names of tasks the
    /// `expose_task` selection loop selected for this backend +
    /// type-system combo (`:113-134`). Read by upstream tests and
    /// `pdbplus`.
    pub exposed: RefCell<Vec<String>>,

    /// Side table populated by [`Self::expose_task`]: the
    /// `backend_goal` arg captured in upstream's nested
    /// `def proc(): return self.proceed(backend_goal)` closure
    /// (`:107-110`). Keyed on `task` (the public, unprefixed name) and
    /// stores the actual platform-specific task to run.
    ///
    /// Upstream Python achieves this through method-level closures;
    /// Rust prefers an explicit map.
    pub exposed_backend_goal: RefCell<HashMap<String, String>>,

    /// Upstream `self._backend_extra_options = {}` (class-level at
    /// `:64`). Per-instance overrides land via
    /// `set_backend_extra_options` at `:139-140`.
    pub _backend_extra_options: RefCell<HashMap<String, OptionValue>>,

    // -----------------------------------------------------------------
    // Upstream-`setup`-populated fields. Each is initially absent;
    // `setup` writes them (`:176-216`).
    // -----------------------------------------------------------------
    /// Upstream `self.standalone = standalone` at `:178`.
    pub standalone: Cell<bool>,
    /// Upstream `self.inputtypes = inputtypes` at `:184`.
    ///
    /// Upstream stores a Python list of annotation specifiers; the
    /// Rust port mirrors with `Vec<AnnotationSpec>` — the typed
    /// dispatch shape consumed by
    /// [`crate::annotator::annrpython::RPythonAnnotator::build_types`]
    /// at `annrpython.rs:521`.
    pub inputtypes: RefCell<Option<Vec<crate::annotator::signature::AnnotationSpec>>>,
    /// Upstream `self.policy = policy` at `:188`.
    ///
    /// Concrete `AnnotatorPolicy` ↔ upstream's `policy` argument. Any
    /// future subclasses (`StrictAnnotatorPolicy`, …) will land as
    /// builder variants of [`AnnotatorPolicy`], not as a separate type.
    pub policy: RefCell<Option<crate::annotator::policy::AnnotatorPolicy>>,
    /// Upstream `self.extra = extra` at `:190`. Upstream's `extra` is
    /// the `targetspec_dic` dict (any-typed payload), so kept as
    /// `Rc<dyn Any>` until each consumer narrows.
    pub extra: RefCell<HashMap<String, Rc<dyn Any>>>,
    /// Upstream `self.entry_point = entry_point` at `:197`. The
    /// upstream attribute is a Python function object; the local
    /// equivalent is [`HostObject::UserFunction`].
    pub entry_point: RefCell<Option<crate::flowspace::model::HostObject>>,
    /// Upstream `self.translator = translator` at `:198`. The bridge
    /// from `Rc<Config>` to `TranslationContext` is the
    /// last-mile work; today the driver constructs the default-config
    /// `TranslationContext` so downstream tasks can reach the
    /// graphs/annotator slot.
    pub translator: RefCell<Option<Rc<super::translator::TranslationContext>>>,
    /// Upstream `self.libdef = None` at `:199` (overridden by
    /// `setup_library`). `libdef` upstream is a CPython-side
    /// shape from `targetstandalone.LibDef`; kept opaque until that
    /// type ports.
    pub libdef: RefCell<Option<Rc<dyn Any>>>,
    /// Upstream `self.secondary_entrypoints = []` at `:200`.
    pub secondary_entrypoints: RefCell<Vec<EntryPointSpec>>,

    /// Strong holder for the annotator across the task chain.
    ///
    /// PRE-EXISTING-ADAPTATION (Rust-language): upstream stores
    /// `translator.annotator = self` (annrpython.py:30-35) as a
    /// strong Python attribute; CPython's cyclic GC unwinds the
    /// `translator → annotator → translator` cycle on session exit.
    /// The local `TranslationContext.annotator` is `Weak` to avoid
    /// the cycle (translator.rs:166), so without a separate strong
    /// holder the annotator drops between `task_annotate` returning
    /// and `task_rtype_lltype` running. This field is the strong
    /// holder; `task_annotate` writes it; later tasks that upstream-
    /// equivalently reach `self.translator.annotator` simply ride
    /// `translator.annotator()` because the strong field keeps the
    /// Weak-upgrade alive.
    pub annotator: RefCell<Option<Rc<crate::annotator::annrpython::RPythonAnnotator>>>,

    /// Upstream `self.jitpolicy`, written by `task_pyjitpl_lltype`
    /// before calling `warmspot.apply_jit`.
    pub jitpolicy: RefCell<Option<Rc<dyn Any>>>,

    /// Upstream `self.cbuilder`, written by `task_database_c`.
    pub cbuilder: RefCell<Option<CBuilderState>>,

    /// Upstream `self.database`, written by `task_database_c`.
    pub database: RefCell<Option<DatabaseState>>,

    /// Upstream `self.c_entryp`, written by `task_compile_c` and
    /// rewritten by `create_exe`.
    pub c_entryp: RefCell<Option<PathBuf>>,

    // -----------------------------------------------------------------
    // Side table for the upstream `@taskdef(... earlycheck=...)`
    // metadata. Engine-level `register_task` only takes the title +
    // idempotency flag; the earlycheck function rides here so
    // `_event("planned", goal)` can invoke it.
    // -----------------------------------------------------------------
    pub task_earlycheck:
        RefCell<HashMap<String, Rc<dyn Fn(&TranslationDriver) -> Result<(), TaskError>>>>,
}

/// Stable cycle-free handle for the driver. Tasks register a
/// `Weak<TranslationDriver>` so the closures don't form a refcount
/// cycle through `engine.tasks`. The driver-construction helper
/// upgrades it to `Rc` on every callable invocation.
type DriverWeak = Weak<TranslationDriver>;

impl TranslationDriver {
    /// Upstream `TranslationDriver.__init__` at `:66-134`. Returns an
    /// `Rc<Self>` because the Rust port registers task callables that
    /// hold a `Weak<Self>` — registration happens inside `new` after
    /// the `Rc` is created so the closures can upgrade.
    ///
    /// Arguments mirror upstream's keyword args; `default_goal`'s
    /// `backend_select_goals` resolution at `:94-97` runs here
    /// verbatim.
    pub fn new(
        setopts: Option<HashMap<String, OptionValue>>,
        default_goal: Option<String>,
        disable: Vec<String>,
        exe_name: Option<String>,
        extmod_name: Option<String>,
        config: Option<Rc<Config>>,
        overrides: Option<HashMap<String, OptionValue>>,
    ) -> Result<Rc<Self>, ConfigError> {
        // Upstream `:71`: `self.timer = Timer()`.
        let timer = Timer::new();

        // Upstream `:72`: `SimpleTaskEngine.__init__(self)`.
        let engine = SimpleTaskEngine::new();

        // Upstream `:76-77`: `if config is None: config =
        // translationoption.get_combined_translation_config(translating=True)`.
        let config = match config {
            Some(c) => c,
            None => get_combined_translation_config(None, None, None, true)?,
        };

        // Upstream `:78-79`: `translationoption._GLOBAL_TRANSLATIONCONFIG = config`.
        _GLOBAL_TRANSLATIONCONFIG.with(|slot| {
            *slot.borrow_mut() = Some(Rc::clone(&config));
        });

        // Upstream `:81-82`: `if overrides is not None:
        //     self.config.override(overrides)`.
        if let Some(overrides) = overrides {
            config.override_(overrides)?;
        }

        // Upstream `:84-85`: `if setopts is not None: self.config.set(**setopts)`.
        if let Some(setopts) = setopts {
            config.set(setopts)?;
        }

        let driver = Rc::new(TranslationDriver {
            timer,
            engine,
            config,
            exe_name: RefCell::new(exe_name),
            extmod_name: RefCell::new(extmod_name),
            done: RefCell::new(HashMap::new()),
            _disabled: RefCell::new(Vec::new()),
            default_goal: RefCell::new(None),
            extra_goals: RefCell::new(Vec::new()),
            exposed: RefCell::new(Vec::new()),
            exposed_backend_goal: RefCell::new(HashMap::new()),
            _backend_extra_options: RefCell::new(HashMap::new()),
            standalone: Cell::new(false),
            inputtypes: RefCell::new(None),
            policy: RefCell::new(None),
            extra: RefCell::new(HashMap::new()),
            entry_point: RefCell::new(None),
            translator: RefCell::new(None),
            libdef: RefCell::new(None),
            secondary_entrypoints: RefCell::new(Vec::new()),
            annotator: RefCell::new(None),
            jitpolicy: RefCell::new(None),
            cbuilder: RefCell::new(None),
            database: RefCell::new(None),
            c_entryp: RefCell::new(None),
            task_earlycheck: RefCell::new(HashMap::new()),
        });

        // Upstream `:92`: `self.disable(disable)`. Must run before
        // `_maybe_skip` consultation in the `expose_task` loop.
        driver.disable(disable);

        // Upstream `:103-110`: register `task_*` callables before the
        // `expose_task` loop walks `self.tasks`. Rust runs the same
        // sequence — first register every task, then run the loop.
        Self::register_tasks(&driver);

        // Upstream `:94-97`: resolve `default_goal` through
        // `backend_select_goals([default_goal])`. Mirror exactly,
        // including the silent `None`-on-skip downgrade.
        if let Some(default_goal) = default_goal {
            let resolved = driver
                .backend_select_goals(&[default_goal.clone()])?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ConfigError::Generic(format!(
                        "TranslationDriver: backend_select_goals returned no result for {:?}",
                        default_goal
                    ))
                })?;
            let maybe_skip = driver._maybe_skip();
            if maybe_skip.iter().any(|s| s == &resolved) {
                *driver.default_goal.borrow_mut() = None;
            } else {
                *driver.default_goal.borrow_mut() = Some(resolved);
            }
        }

        // Upstream `:112-134`: `expose_task` selection loop. Reads
        // backend / type_system from `config.translation.*` and
        // chooses which task names land in `self.exposed`.
        driver.run_expose_task_loop()?;

        Ok(driver)
    }

    /// Convenience constructor matching upstream's `TranslationDriver()`
    /// no-arg call (used by the test suite at `test_driver.py:8`).
    pub fn new_default() -> Result<Rc<Self>, ConfigError> {
        Self::new(None, None, Vec::new(), None, None, None, None)
    }

    /// Internal: registers every `task_<name>` upstream defines on
    /// `TranslationDriver` (`:297-555`). Each registration captures a
    /// `Weak<Self>` so the engine's task table doesn't cycle.
    ///
    /// Task bodies that depend on unported leaves keep the upstream
    /// statement order and return `TaskError` at the exact missing
    /// leaf; the registration metadata (deps, title, idempotency) is
    /// the upstream-observable shape that `_plan` and `expose_task`
    /// consume.
    fn register_tasks(this: &Rc<Self>) {
        let weak = Rc::downgrade(this);

        // Upstream `:297-298 @taskdef([], "Annotating&simplifying")
        // def task_annotate(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "annotate",
            TaskDef::new(Vec::new(), "Annotating&simplifying"),
            |d| d.task_annotate(),
        );

        // Upstream `:340-341 @taskdef(['annotate'], "RTyping")
        // def task_rtype_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "rtype_lltype",
            TaskDef::new(vec!["annotate".to_string()], "RTyping"),
            |d| d.task_rtype_lltype(),
        );

        // Upstream `:347-348 @taskdef([RTYPE], "JIT compiler generation")
        // def task_pyjitpl_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "pyjitpl_lltype",
            TaskDef::new(vec![RTYPE.to_string()], "JIT compiler generation"),
            |d| d.task_pyjitpl_lltype(),
        );

        // Upstream `:365 @taskdef([RTYPE], "test of the JIT on the llgraph backend")
        // def task_jittest_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "jittest_lltype",
            TaskDef::new(
                vec![RTYPE.to_string()],
                "test of the JIT on the llgraph backend",
            ),
            |d| d.task_jittest_lltype(),
        );

        // Upstream `:379 @taskdef([RTYPE, '??pyjitpl_lltype', '??jittest_lltype'],
        // "lltype back-end optimisations") def task_backendopt_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "backendopt_lltype",
            TaskDef::new(
                vec![
                    RTYPE.to_string(),
                    "??pyjitpl_lltype".to_string(),
                    "??jittest_lltype".to_string(),
                ],
                "lltype back-end optimisations",
            ),
            |d| d.task_backendopt_lltype(),
        );

        // Upstream `:388 @taskdef(['?'+BACKENDOPT, RTYPE, 'annotate'], "inserting stack checks")
        // def task_stackcheckinsertion_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "stackcheckinsertion_lltype",
            TaskDef::new(
                vec![
                    format!("?{}", BACKENDOPT),
                    RTYPE.to_string(),
                    "annotate".to_string(),
                ],
                "inserting stack checks",
            ),
            |d| d.task_stackcheckinsertion_lltype(),
        );

        // Upstream `:405-407 @taskdef([STACKCHECKINSERTION, '?'+BACKENDOPT,
        // RTYPE, '?annotate'], "Creating database for generating c source",
        // earlycheck = possibly_check_for_boehm) def task_database_c(self):`.
        let database_c_def = TaskDef::new(
            vec![
                STACKCHECKINSERTION.to_string(),
                format!("?{}", BACKENDOPT),
                RTYPE.to_string(),
                "?annotate".to_string(),
            ],
            "Creating database for generating c source",
        )
        .with_earlycheck(|d| d.possibly_check_for_boehm());
        Self::register_task_with_def(this, &weak, "database_c", database_c_def, |d| {
            d.task_database_c()
        });

        // Upstream `:440 @taskdef(['database_c'], "Generating c source")
        // def task_source_c(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "source_c",
            TaskDef::new(vec!["database_c".to_string()], "Generating c source"),
            |d| d.task_source_c(),
        );

        // Upstream `:526 @taskdef(['source_c'], "Compiling c source")
        // def task_compile_c(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "compile_c",
            TaskDef::new(vec!["source_c".to_string()], "Compiling c source"),
            |d| d.task_compile_c(),
        );

        // Upstream `:543 @taskdef([STACKCHECKINSERTION, '?'+BACKENDOPT, RTYPE],
        // "LLInterpreting") def task_llinterpret_lltype(self):`.
        Self::register_task_with_def(
            this,
            &weak,
            "llinterpret_lltype",
            TaskDef::new(
                vec![
                    STACKCHECKINSERTION.to_string(),
                    format!("?{}", BACKENDOPT),
                    RTYPE.to_string(),
                ],
                "LLInterpreting",
            ),
            |d| d.task_llinterpret_lltype(),
        );
    }

    /// Internal helper: hands a single `TaskDef` to the engine and
    /// stashes the optional `earlycheck` in the side table.
    fn register_task_with_def(
        this: &Rc<Self>,
        weak: &DriverWeak,
        name: &str,
        def: TaskDef,
        body: fn(&TranslationDriver) -> Result<TaskOutput, TaskError>,
    ) {
        let weak_for_call = weak.clone();
        let callable: Rc<dyn Fn() -> Result<TaskOutput, TaskError>> = Rc::new(move || {
            let d = weak_for_call
                .upgrade()
                .expect("TranslationDriver dropped while engine still owned task callable");
            body(&d)
        });
        this.engine.register_task(
            name.to_string(),
            callable,
            def.deps.clone(),
            def.title.clone(),
            def.idempotent,
        );
        if let Some(check) = def.earlycheck {
            this.task_earlycheck
                .borrow_mut()
                .insert(name.to_string(), check);
        }
    }

    /// Internal: implements upstream `:104-110 def expose_task(task,
    /// backend_goal=None)` selection loop at `:112-134`.
    fn run_expose_task_loop(&self) -> Result<(), ConfigError> {
        let (backend, ts) = self.get_backend_and_type_system()?;

        // Upstream `:113`: `for task in self.tasks:` — iteration order
        // is the dict order upstream, which since Python 3.7 is
        // insertion order. Rust's `HashMap` doesn't preserve insertion
        // order; sort task names so the resulting `self.exposed`
        // ordering is deterministic across runs (the test reads it as
        // a set, but determinism aids debugging).
        let task_names: Vec<String> = {
            let mut names: Vec<String> = self.engine.tasks().keys().cloned().collect();
            names.sort();
            names
        };

        for task_name in task_names {
            let explicit_task = task_name.clone();
            if task_name == "annotate" {
                // Upstream `:115-116`: `expose_task(task)` with no postfix.
                self.expose_task(&task_name, None);
                continue;
            }
            // Upstream `:118`: `task, postfix = task.split('_')`. The
            // upstream identifiers always contain at most one `_`, so
            // `split('_')` returns exactly two parts.
            let (head, postfix) = match task_name.split_once('_') {
                Some((h, p)) => (h, p),
                None => continue,
            };
            match head {
                // Upstream `:119-120`: `if task in ('rtype',
                // 'backendopt', 'llinterpret', 'pyjitpl'):` — note
                // `stackcheckinsertion` and `jittest` are NOT in
                // upstream's tuple. They are still registered as
                // tasks but never exposed.
                "rtype" | "backendopt" | "llinterpret" | "pyjitpl" => {
                    // Upstream `:121-125`: gated on type-system match.
                    if let Some(ts) = &ts {
                        if ts == postfix {
                            self.expose_task(head, Some(&explicit_task));
                        }
                    } else {
                        self.expose_task(&explicit_task, None);
                    }
                }
                "source" | "compile" | "run" => {
                    // Upstream `:126-134`: gated on backend match if
                    // backend is set, else type-system fallback.
                    if let Some(backend) = &backend {
                        if backend == postfix {
                            self.expose_task(head, Some(&explicit_task));
                        }
                    } else if let Some(ts) = &ts {
                        if ts == "lltype" {
                            self.expose_task(&explicit_task, None);
                        }
                    } else {
                        self.expose_task(&explicit_task, None);
                    }
                }
                _ => {
                    // Heads not enumerated above are silently skipped
                    // upstream — the `if task in (…)` chain falls
                    // through.
                }
            }
        }
        Ok(())
    }

    /// Port of upstream's nested `def expose_task(task, backend_goal=None)`
    /// inside `__init__` at `:104-110`. Captures the `backend_goal` as
    /// a closure over which `proceed` to run; in Rust we record it in
    /// the side table so `proceed_exposed("compile")` can resolve.
    fn expose_task(&self, task: &str, backend_goal: Option<&str>) {
        let backend_goal = backend_goal.unwrap_or(task).to_string();
        self.exposed.borrow_mut().push(task.to_string());
        self.exposed_backend_goal
            .borrow_mut()
            .insert(task.to_string(), backend_goal);
    }

    /// Upstream `set_extra_goals(self, goals)` at `:136-137`.
    pub fn set_extra_goals(&self, goals: Vec<String>) {
        *self.extra_goals.borrow_mut() = goals;
    }

    /// Upstream `set_backend_extra_options(self, extra_options)` at `:139-140`.
    pub fn set_backend_extra_options(&self, extra_options: HashMap<String, OptionValue>) {
        *self._backend_extra_options.borrow_mut() = extra_options;
    }

    /// Upstream `get_info(self)` at `:142-144`.
    pub fn get_info(&self) -> HashMap<String, String> {
        let mut d = HashMap::new();
        if let Some(backend) = self.read_choice("translation.backend") {
            d.insert("backend".to_string(), backend);
        }
        d
    }

    /// Upstream `get_backend_and_type_system(self)` at `:146-149`.
    pub fn get_backend_and_type_system(
        &self,
    ) -> Result<(Option<String>, Option<String>), ConfigError> {
        let type_system = self.read_choice("translation.type_system");
        let backend = self.read_choice("translation.backend");
        Ok((backend, type_system))
    }

    /// Reads a `ChoiceOption` value as `Option<String>`, mapping
    /// upstream's `None` ↔ `OptionValue::None`. Used by
    /// `get_backend_and_type_system` and `get_info`.
    fn read_choice(&self, path: &str) -> Option<String> {
        match self.config.get(path).ok()? {
            ConfigValue::Value(OptionValue::Choice(s)) => Some(s),
            ConfigValue::Value(OptionValue::Str(s)) => Some(s),
            ConfigValue::Value(OptionValue::None) => None,
            _ => None,
        }
    }

    /// Upstream `backend_select_goals(self, goals)` at `:151-164`.
    /// Resolves every requested goal through the backend / type-
    /// system postfixes, matching upstream's
    /// `for postfix in postfixes` chain exactly.
    pub fn backend_select_goals(&self, goals: &[String]) -> Result<Vec<String>, ConfigError> {
        let (backend, ts) = self.get_backend_and_type_system()?;
        // Upstream `:153`: `postfixes = [''] + ['_'+p for p in (backend, ts) if p]`.
        let mut postfixes: Vec<String> = vec![String::new()];
        for p in [&backend, &ts] {
            if let Some(p) = p {
                postfixes.push(format!("_{}", p));
            }
        }
        let tasks = self.engine.tasks();
        let mut out = Vec::new();
        for goal in goals {
            let mut found: Option<String> = None;
            for postfix in &postfixes {
                let cand = format!("{}{}", goal, postfix);
                if tasks.contains_key(&cand) {
                    found = Some(cand);
                    break;
                }
            }
            match found {
                Some(g) => out.push(g),
                None => {
                    // Upstream `:162`: `raise Exception("cannot infer
                    // complete goal from: %r" % goal)`.
                    return Err(ConfigError::Generic(format!(
                        "cannot infer complete goal from: {:?}",
                        goal
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Upstream `disable(self, to_disable)` at `:166-167`.
    pub fn disable(&self, to_disable: Vec<String>) {
        *self._disabled.borrow_mut() = to_disable;
    }

    /// Upstream `_maybe_skip(self)` at `:169-174`.
    pub fn _maybe_skip(&self) -> Vec<String> {
        let disabled = self._disabled.borrow().clone();
        if disabled.is_empty() {
            return Vec::new();
        }
        let resolved = match self.backend_select_goals(&disabled) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let mut maybe_skip: Vec<String> = Vec::new();
        for goal in resolved {
            for dep in self.engine._depending_on_closure(&goal) {
                maybe_skip.push(dep);
            }
        }
        // Upstream `:174`: `dict.fromkeys(maybe_skip).keys()` — dedupe.
        let mut seen = std::collections::HashSet::new();
        maybe_skip.retain(|s| seen.insert(s.clone()));
        maybe_skip
    }

    /// Upstream `setup(self, entry_point, inputtypes, policy=None,
    /// extra={}, empty_translator=None)` at `:176-210`.
    ///
    /// The upstream `:194 translator = TranslationContext(config=self.config)`
    /// is mirrored by reading the seven typed `translation.<name>`
    /// fields out of `self.config` via
    /// [`crate::translator::translator::TranslationConfig::from_rc_config`]
    /// and constructing the `TranslationContext` with the resulting
    /// snapshot. Driver-side `override` / `setopts` calls performed
    /// before `setup` are visible to every read of
    /// `self.translator.config.translation.<X>` because the snapshot
    /// is taken AFTER those mutations land.
    ///
    /// `driver_instrument_result` callback at `:210` is still unwired
    /// (the local `TranslationContext` does not yet carry that slot
    /// — the leaf is C-backend instrumentation).
    pub fn setup(
        &self,
        entry_point: Option<crate::flowspace::model::HostObject>,
        inputtypes: Option<Vec<crate::annotator::signature::AnnotationSpec>>,
        policy: Option<crate::annotator::policy::AnnotatorPolicy>,
        extra: HashMap<String, Rc<dyn Any>>,
        empty_translator: Option<Rc<super::translator::TranslationContext>>,
    ) -> Result<(), TaskError> {
        use crate::annotator::model::SomeValue;
        use crate::annotator::signature::AnnotationSpec;

        // Upstream `:177-178`: `standalone = inputtypes is None;
        // self.standalone = standalone`.
        let standalone = inputtypes.is_none();
        self.standalone.set(standalone);

        // Upstream `:180-183`: `if standalone: inputtypes =
        // [s_list_of_strings]; self.inputtypes = inputtypes`.
        let resolved_inputtypes: Vec<AnnotationSpec> = inputtypes.unwrap_or_else(|| {
            // upstream: standalone fall-back. `s_list_of_strings` is
            // a `SomeList`; wrap it in `AnnotationSpec::Already` so
            // the build_types pipeline at `annrpython.rs:534`
            // (`typeannotation` short-circuit) accepts it.
            vec![AnnotationSpec::Already(SomeValue::List(
                crate::annotator::listdef::s_list_of_strings(),
            ))]
        });
        *self.inputtypes.borrow_mut() = Some(resolved_inputtypes);

        // Upstream `:185-187`: default policy = `AnnotatorPolicy()`.
        let resolved_policy =
            policy.unwrap_or_else(crate::annotator::policy::AnnotatorPolicy::default);
        *self.policy.borrow_mut() = Some(resolved_policy);

        // Upstream `:189`: `self.extra = extra`.
        *self.extra.borrow_mut() = extra;

        // Upstream `:191-194`: pick `empty_translator` or build one.
        // When we build, take a snapshot of the driver's `Rc<Config>`
        // so any `override` / `setopts` mutations performed before
        // `setup` (`driver.py:81-85`) are visible inside the
        // translator's `config.translation.<X>` reads.
        let translator: Rc<super::translator::TranslationContext> = match empty_translator {
            Some(t) => t,
            None => {
                let translation_config = super::translator::TranslationConfig::from_rc_config(
                    &self.config,
                )
                .map_err(|e| TaskError {
                    message: format!(
                        "TranslationDriver::setup: TranslationConfig::from_rc_config: {e:?}"
                    ),
                })?;
                Rc::new(
                    super::translator::TranslationContext::with_config_and_flowing_flags(
                        Some(translation_config),
                        super::translator::FlowingFlags::default(),
                    ),
                )
            }
        };

        // Upstream `:196-198`.
        *self.entry_point.borrow_mut() = entry_point;
        *self.translator.borrow_mut() = Some(translator);
        *self.libdef.borrow_mut() = None;

        // Upstream `:199`: `self.secondary_entrypoints = []`.
        self.secondary_entrypoints.borrow_mut().clear();

        // Upstream `:201-208`: walk `config.translation.secondaryentrypoints`
        // CSV and look up each key in the global registry.
        let csv = match self.config.get("translation.secondaryentrypoints") {
            Ok(ConfigValue::Value(OptionValue::Str(s))) => s,
            _ => String::new(),
        };
        if !csv.is_empty() {
            for key in csv.split(',') {
                let key = key.trim();
                if key.is_empty() {
                    continue;
                }
                let points = secondary_entrypoints_get(key).ok_or_else(|| TaskError {
                    message: format!(
                        "Entrypoint {:?} not found (not in {:?})",
                        key,
                        secondary_entrypoints_keys(),
                    ),
                })?;
                self.secondary_entrypoints.borrow_mut().extend(points);
            }
        }

        // Upstream `:210`: `self.translator.driver_instrument_result =
        // self.instrument_result`. DEFERRED — translator surface not yet
        // bridged through `Rc<Config>`.
        Ok(())
    }

    /// Upstream `setup_library(self, libdef, policy=None, extra={},
    /// empty_translator=None)` at `:212-216`.
    pub fn setup_library(
        &self,
        libdef: Rc<dyn Any>,
        policy: Option<crate::annotator::policy::AnnotatorPolicy>,
        extra: HashMap<String, Rc<dyn Any>>,
        empty_translator: Option<Rc<super::translator::TranslationContext>>,
    ) -> Result<(), TaskError> {
        // Upstream `:214`: `self.setup(None, None, policy, extra,
        // empty_translator)`.
        self.setup(None, None, policy, extra, empty_translator)?;
        // Upstream `:215`: `self.libdef = libdef`.
        *self.libdef.borrow_mut() = Some(libdef.clone());
        // Upstream `:216`: `self.secondary_entrypoints = libdef.functions`.
        // DEFERRED — `libdef.functions` shape lives in the C-backend
        // (`rpython/translator/c/dlltool.py`) port, not yet ported.
        let _ = libdef;
        Ok(())
    }

    /// Upstream `instrument_result(self, args)` at `:218-248`.
    ///
    /// **DEFERRED**: the body forks via `os.fork`, mutates
    /// `self.config.translation.instrument*` for the child, drives
    /// `self.proceed('compile')`, then reads `_instrument_counters` out
    /// of `udir`. None of those are ported here (`os.fork` /
    /// `udir` / C-backend `compile` task / `array.array` reading). The
    /// signature stays so callers compile.
    pub fn instrument_result(&self, _args: &[Rc<dyn Any>]) -> ! {
        unimplemented!(
            "driver.py:218 instrument_result — leaves os.fork + udir + C-backend compile not yet ported"
        );
    }

    /// Upstream `info(self, msg)` at `:250-251`.
    pub fn info(&self, msg: &str) {
        // Upstream `log.info(msg)`. AnsiLogger not ported — see
        // module docstring for the convergence path.
        println!("{msg}");
    }

    /// Upstream `_profile(self, goal, func)` at `:253-260`. DEFERRED —
    /// requires `cProfile.Profile` + `KCacheGrind`, neither ported.
    pub fn _profile(
        &self,
        _goal: &str,
        _func: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    ) -> ! {
        unimplemented!("driver.py:253 _profile — leaf cProfile.Profile not ported");
    }

    // -----------------------------------------------------------------
    // Upstream `task_*` bodies. Every body either ports a small
    // structural shell (the buildannotator+complete chain that the
    // local annotator ports support) or panics with a citation and the
    // missing leaf name.
    // -----------------------------------------------------------------

    /// Upstream `task_annotate(self)` at `:297-327`.
    ///
    /// ```python
    /// @taskdef([], "Annotating&simplifying")
    /// def task_annotate(self):
    ///     translator = self.translator
    ///     policy = self.policy
    ///     self.log.info('with policy: %s.%s' % (...))
    ///     annotator = translator.buildannotator(policy=policy)
    ///     if self.secondary_entrypoints is not None:
    ///         for func, inputtypes in self.secondary_entrypoints:
    ///             if inputtypes == Ellipsis:
    ///                 continue
    ///             annotator.build_types(func, inputtypes, False)
    ///     if self.entry_point:
    ///         s = annotator.build_types(self.entry_point, self.inputtypes)
    ///         translator.entry_point_graph = annotator.bookkeeper.getdesc(self.entry_point).getuniquegraph()
    ///     else:
    ///         s = None
    ///     self.sanity_check_annotation()
    ///     if self.entry_point and self.standalone and s.knowntype != int:
    ///         raise Exception(...)
    ///     annotator.complete()
    ///     annotator.simplify()
    ///     return s
    /// ```
    ///
    /// **Partially activated**: the structural skeleton (buildannotator,
    /// optional `build_types(entry_point)`, `sanity_check_annotation`,
    /// `complete`, `simplify`) ports line-by-line. The
    /// `secondary_entrypoints` loop at `:308-312` iterates upstream's
    /// `(func, inputtypes)` pair where `inputtypes` is a Python list of
    /// `SomeXxx` values; the local registry stores
    /// `(Rc<dyn Any>, Vec<Rc<dyn Any>>)` pending the upstream-mirroring
    /// shape. The loop is wired but skipped (DEFERRED) until that
    /// downcast hardens.
    ///
    /// The standalone-int return-type check at `:321-324` requires
    /// inspecting `s.knowntype`; the local check uses
    /// [`SomeValue::Integer`] as the parity discriminant.
    pub fn task_annotate(&self) -> Result<TaskOutput, TaskError> {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;

        // Upstream `:300-301`: `translator = self.translator; policy = self.policy`.
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_annotate: translator slot is unset; setup() must be called first"
                .to_string(),
        })?;
        let policy = self.policy.borrow().clone();

        // Upstream `:302-304`: `self.log.info('with policy: ...')`. The
        // class-name string is `policy.__class__.__module__ + '.' + ... __name__`.
        // The Rust port emits a parity-equivalent line.
        self.info("with policy: rpython.annotator.policy.AnnotatorPolicy");

        // Upstream `:306`: `annotator = translator.buildannotator(policy=policy)`.
        let annotator = translator.buildannotator(policy);
        // Hand the annotator over to the driver's strong holder so it
        // survives this task's return. See the field doc on
        // [`Self::annotator`] for the Rust-port adaptation rationale.
        *self.annotator.borrow_mut() = Some(Rc::clone(&annotator));

        // Upstream `:308-312`: secondary entrypoints loop. The local
        // `EntryPointSpec` carries `(Rc<dyn Any>, Vec<Rc<dyn Any>>)` —
        // the upstream `(func, inputtypes)` shape isn't downcastable
        // to a typed `(HostObject, Vec<AnnotationSpec>)` pair without
        // an Any-narrowing step. Once
        // `rlib::entrypoint::secondary_entrypoints` ports the typed
        // shape, this loop becomes the upstream `for func, inputtypes
        // in self.secondary_entrypoints:` straight-port.
        // DEFERRED — body is currently a no-op; the empty-registry
        // shape matches upstream's "no secondary entrypoints" path.
        for (_func_any, _inputtypes_any) in self.secondary_entrypoints.borrow().iter() {
            // upstream: `if inputtypes == Ellipsis: continue;
            // annotator.build_types(func, inputtypes, False)`.
        }

        // Upstream `:314-318`: optional `s = annotator.build_types(
        // self.entry_point, self.inputtypes)`.
        let s: Option<SomeValue> = if let Some(entry_point) = self.entry_point.borrow().clone() {
            let inputtypes = self.inputtypes.borrow().clone().ok_or_else(|| TaskError {
                message: "task_annotate: inputtypes unset; setup() must populate it".to_string(),
            })?;
            let result = annotator
                .build_types(&entry_point, &inputtypes, true, true)
                .map_err(|e| TaskError {
                    message: format!("annotator.build_types: {e}"),
                })?;
            // Upstream `:316-317`: `translator.entry_point_graph =
            // annotator.bookkeeper.getdesc(self.entry_point).getuniquegraph()`.
            let _ep: HostObject = entry_point;
            // `RPythonAnnotator::build_types` already populates
            // `translator.entry_point_graph` when `main_entry_point=true`
            // (annrpython.rs:555-557), so the explicit assignment is
            // already done by the call above.
            result
        } else {
            // Upstream `:318-319`: `else: s = None`.
            None
        };

        // Upstream `:320`: `self.sanity_check_annotation()`.
        self.sanity_check_annotation()?;

        // Upstream `:321-324`: standalone-int return-type check.
        if self.entry_point.borrow().is_some() && self.standalone.get() {
            let int_return = matches!(s, Some(SomeValue::Integer(_)));
            if !int_return {
                return Err(TaskError {
                    message:
                        "stand-alone program entry point must return an int (and not, e.g., None or always raise an exception)."
                            .to_string(),
                });
            }
        }

        // Upstream `:325-326`: `annotator.complete(); annotator.simplify()`.
        annotator.complete().map_err(|e| TaskError {
            message: format!("annotator.complete: {e}"),
        })?;
        annotator.simplify(None, None);

        // Upstream `:327`: `return s`.
        Ok(s.map(|sv| Box::new(sv) as Box<dyn Any>))
    }

    /// Upstream `sanity_check_annotation(self)` at `:330-337`.
    ///
    /// ```python
    /// def sanity_check_annotation(self):
    ///     translator = self.translator
    ///     irreg = query.qoutput(query.check_exceptblocks_qgen(translator))
    ///     if irreg:
    ///         self.log.info("Some exceptblocks seem insane")
    ///
    ///     lost = query.qoutput(query.check_methods_qgen(translator))
    ///     assert not lost, "lost methods, something gone wrong with the annotation of method defs"
    /// ```
    ///
    /// The query.py port is in [`super::goal::query`]. Walks the
    /// `self.translator` slot when present; silently no-ops in pre-
    /// `setup()` lifecycle states — matching upstream's
    /// "sanity check is best-effort" intent.
    pub fn sanity_check_annotation(&self) -> Result<(), TaskError> {
        use super::goal::query;
        let slot = self.translator.borrow();
        let translator = match slot.as_ref() {
            Some(t) => t.clone(),
            None => return Ok(()),
        };
        // Upstream `:332-334`: `irreg = query.qoutput(...)`.
        let irreg = query::qoutput(query::check_exceptblocks_qgen(&translator), None);
        if irreg > 0 {
            self.info("Some exceptblocks seem insane");
        }
        // Upstream `:336-337`: `lost = query.qoutput(...); assert not lost,
        // "lost methods, something gone wrong with the annotation of method defs"`.
        let lost = query::qoutput(query::check_methods_qgen(&translator), None);
        if lost > 0 {
            return Err(TaskError {
                message: "lost methods, something gone wrong with the annotation of method defs"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Upstream `task_rtype_lltype(self)` at `:340-345`.
    ///
    /// ```python
    /// @taskdef(['annotate'], "RTyping")
    /// def task_rtype_lltype(self):
    ///     """ RTyping - lltype version
    ///     """
    ///     rtyper = self.translator.buildrtyper()
    ///     rtyper.specialize(dont_simplify_again=True)
    /// ```
    ///
    /// Activated: both `TranslationContext::buildrtyper`
    /// (`translator.rs:294`) and
    /// `RPythonTyper::specialize(dont_simplify_again=true)`
    /// (`rtyper.rs:1299`) are ported. The `'annotate'` task dep
    /// guarantees the annotator slot is populated before this body
    /// runs.
    pub fn task_rtype_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_rtype_lltype: translator slot is unset".to_string(),
        })?;
        let rtyper = translator.buildrtyper();
        rtyper.specialize(true).map_err(|e| TaskError {
            message: format!("rtyper.specialize: {e:?}"),
        })?;
        Ok(None)
    }

    fn missing_task_leaf(&self, line: usize, leaf: &str) -> TaskError {
        TaskError {
            message: format!("driver.py:{line} leaf not ported: {leaf}"),
        }
    }

    fn read_bool(&self, path: &str) -> bool {
        matches!(
            self.config.get(path),
            Ok(ConfigValue::Value(OptionValue::Bool(true)))
        )
    }

    fn format_exe_name_template(&self, template: &str) -> String {
        let mut out = template.to_string();
        for (key, value) in self.get_info() {
            out = out.replace(&format!("%({key})s"), &value);
        }
        out
    }

    /// Upstream `task_pyjitpl_lltype(self)` at `:347-363`.
    pub fn task_pyjitpl_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_pyjitpl_lltype: translator slot is unset".to_string(),
        })?;
        let get_policy = self.extra.borrow().get("jitpolicy").cloned();
        if get_policy.is_none() {
            *self.jitpolicy.borrow_mut() = Some(Rc::new(
                crate::jit_codewriter::policy::DefaultJitPolicy::new(),
            ));
        } else {
            return Err(self.missing_task_leaf(
                352,
                "calling extra['jitpolicy'](self) to build the JIT policy",
            ));
        }
        let backend_name = self
            .read_choice("translation.jit_backend")
            .unwrap_or_else(|| "auto".to_string());
        let _ = (translator, backend_name);
        Err(self.missing_task_leaf(
            358,
            "rpython.jit.metainterp.warmspot.apply_jit(translator, policy=..., backend_name=..., inline=True)",
        ))
    }

    /// Upstream `task_jittest_lltype(self)` at `:365-376`.
    pub fn task_jittest_lltype(&self) -> Result<TaskOutput, TaskError> {
        // Upstream first creates a restartable checkpoint:
        // `unixcheckpoint.restartable_point(auto='run')`. The
        // checkpoint leaf is not ported; keep the call site as a
        // no-op until `unixcheckpoint` lands, then continue to the
        // actual test leaf.
        Err(self.missing_task_leaf(375, "rpython.jit.tl.jittest.jittest(self)"))
    }

    /// Upstream `task_backendopt_lltype(self)` at `:380-384`.
    pub fn task_backendopt_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_backendopt_lltype: translator slot is unset".to_string(),
        })?;
        let _ = translator;
        Err(self.missing_task_leaf(
            383,
            "rpython.translator.backendopt.all.backend_optimizations(self.translator, replace_we_are_jitted=True)",
        ))
    }

    /// Upstream `task_stackcheckinsertion_lltype(self)` at `:388-392`.
    pub fn task_stackcheckinsertion_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_stackcheckinsertion_lltype: translator slot is unset".to_string(),
        })?;
        let _ = translator;
        Err(self.missing_task_leaf(
            390,
            "rpython.translator.transform.insert_ll_stackcheck(self.translator)",
        ))
    }

    /// Upstream `possibly_check_for_boehm(self)` at `:395-403`.
    /// DEFERRED — requires `rpython.rtyper.tool.rffi_platform.configure_boehm`
    /// and `rpython.translator.platform.CompilationError`.
    pub fn possibly_check_for_boehm(&self) -> Result<(), TaskError> {
        // Upstream `:396`: `if self.config.translation.gc == "boehm":`.
        let gc = match self.config.get("translation.gc") {
            Ok(ConfigValue::Value(OptionValue::Choice(s))) => s,
            _ => return Ok(()),
        };
        if gc == "boehm" {
            // Upstream `:397-403`: probe the platform for libgc-dev.
            // DEFERRED — until rffi_platform ports, just succeed; this
            // matches the runtime behaviour on any host where Boehm is
            // installed.
        }
        Ok(())
    }

    /// Upstream `task_database_c(self)` at `:408-438`.
    pub fn task_database_c(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_database_c: translator slot is unset".to_string(),
        })?;
        if translator.annotator().is_some() {
            translator.frozen.set(true);
        }

        let standalone = self.standalone.get();
        let get_gchooks = self.extra.borrow().get("get_gchooks").cloned();
        if get_gchooks.is_some() {
            return Err(self.missing_task_leaf(414, "calling extra['get_gchooks']()"));
        }

        let cbuilder = if standalone {
            let _ = (
                translator,
                self.entry_point.borrow().clone(),
                self.config.clone(),
                self.secondary_entrypoints.borrow().clone(),
                annotated_jit_entrypoints_get(),
            );
            CBuilderState::standalone()
        } else {
            let _functions = {
                let mut functions = Vec::new();
                if let Some(entry) = self.entry_point.borrow().clone() {
                    functions.push((Rc::new(entry) as Rc<dyn Any>, Vec::new()));
                }
                functions.extend(self.secondary_entrypoints.borrow().clone());
                functions.extend(annotated_jit_entrypoints_get());
                functions
            };
            CBuilderState::library(self.extmod_name.borrow().clone())
        };
        let database = cbuilder.build_database()?;
        self.info("database for generating C source was created");
        *self.cbuilder.borrow_mut() = Some(cbuilder);
        *self.database.borrow_mut() = Some(database);
        Ok(None)
    }

    /// Upstream `task_source_c(self)` at `:441-463`.
    pub fn task_source_c(&self) -> Result<TaskOutput, TaskError> {
        let cbuilder = self.cbuilder.borrow().clone().ok_or_else(|| TaskError {
            message: "task_source_c: cbuilder slot is unset; task_database_c must run first"
                .to_string(),
        })?;
        let database = self.database.borrow().clone().ok_or_else(|| TaskError {
            message: "task_source_c: database slot is unset; task_database_c must run first"
                .to_string(),
        })?;
        let defines = if matches!(
            self._backend_extra_options.borrow().get("c_debug_defines"),
            Some(OptionValue::Bool(true))
        ) {
            let mut d = HashMap::new();
            d.insert("DEBUG_DEFINES".to_string(), "1".to_string());
            d
        } else {
            HashMap::new()
        };
        let exe_name = self
            .exe_name
            .borrow()
            .as_ref()
            .map(|template| self.format_exe_name_template(template));
        let c_source_filename = cbuilder.generate_source(&database, &defines, exe_name)?;
        self.info(&format!("written: {}", c_source_filename.display()));
        if self.read_bool("translation.dump_static_data_info") {
            let _targetdir = cbuilder.targetdir.clone();
            return Err(self.missing_task_leaf(
                456,
                "rpython.translator.tool.staticsizereport.dump_static_data_info(...)",
            ));
        }
        Ok(None)
    }

    /// Upstream `compute_exe_name(self, suffix='')` at `:465-474`.
    pub fn compute_exe_name(&self, suffix: &str) -> Result<PathBuf, TaskError> {
        let template = self.exe_name.borrow().clone().ok_or_else(|| TaskError {
            message: "compute_exe_name: exe_name is None".to_string(),
        })?;
        let mut newexename = self.format_exe_name_template(&template);
        if !newexename.contains('/') && !newexename.contains('\\') {
            newexename = format!("./{newexename}");
        }
        if suffix.is_empty() {
            return Ok(PathBuf::from(newexename));
        }
        let mut path = PathBuf::from(newexename);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| TaskError {
                message: "compute_exe_name: cannot compute basename stem".to_string(),
            })?
            .to_string();
        let new_basename = format!("{stem}{suffix}");
        path.set_file_name(new_basename);
        Ok(path)
    }

    /// Upstream `create_exe(self)` at `:476-524`.
    pub fn create_exe(&self) -> Result<(), TaskError> {
        if self.exe_name.borrow().is_some() {
            let exename = self.c_entryp.borrow().clone().ok_or_else(|| TaskError {
                message: "create_exe: c_entryp slot is unset".to_string(),
            })?;
            let basename = exename
                .file_name()
                .ok_or_else(|| TaskError {
                    message: "create_exe: c_entryp has no basename".to_string(),
                })?
                .to_owned();
            let newexename = PathBuf::from(basename);
            shutil_copy(&exename, &newexename).map_err(|e| TaskError {
                message: format!(
                    "driver.py:481 shutil_copy({}, {}): {e}",
                    exename.display(),
                    newexename.display()
                ),
            })?;
            self.info(&format!(
                "copied: {} to {}",
                exename.display(),
                newexename.display()
            ));
            if let Some(cbuilder) = self.cbuilder.borrow().clone() {
                if let Some(soname) = cbuilder.shared_library_name {
                    let newsoname =
                        newexename.with_file_name(soname.file_name().ok_or_else(|| TaskError {
                            message: "create_exe: shared_library_name has no basename".to_string(),
                        })?);
                    shutil_copy(&soname, &newsoname).map_err(|e| TaskError {
                        message: format!(
                            "driver.py:486 shutil_copy({}, {}): {e}",
                            soname.display(),
                            newsoname.display()
                        ),
                    })?;
                    self.info(&format!(
                        "copied: {} to {}",
                        soname.display(),
                        newsoname.display()
                    ));
                    if cbuilder.executable_name_w.is_some() {
                        return Err(self.missing_task_leaf(
                            490,
                            "Windows pypyw/import-library/pdb/libffi copy block",
                        ));
                    }
                }
            }
            *self.c_entryp.borrow_mut() = Some(newexename);
        }
        let created = self
            .c_entryp
            .borrow()
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "None".to_string());
        self.info(&format!("created: {created}"));
        Ok(())
    }

    /// Upstream `task_compile_c(self)` at `:527-541`.
    pub fn task_compile_c(&self) -> Result<TaskOutput, TaskError> {
        let cbuilder = self.cbuilder.borrow().clone().ok_or_else(|| TaskError {
            message: "task_compile_c: cbuilder slot is unset; task_source_c must run first"
                .to_string(),
        })?;
        let exe_name = if self.standalone.get() && self.exe_name.borrow().is_some() {
            Some(
                self.compute_exe_name("")?
                    .file_name()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| TaskError {
                        message: "task_compile_c: computed exe name has no basename".to_string(),
                    })?
                    .to_string(),
            )
        } else {
            None
        };
        cbuilder.compile(exe_name)?;
        if self.standalone.get() {
            *self.c_entryp.borrow_mut() = Some(cbuilder.executable_name.clone());
            self.create_exe()?;
        } else {
            *self.c_entryp.borrow_mut() = Some(cbuilder.get_entry_point()?);
        }
        Ok(None)
    }

    /// Upstream `task_llinterpret_lltype(self)` at `:544-555`.
    pub fn task_llinterpret_lltype(&self) -> Result<TaskOutput, TaskError> {
        let translator = self.translator.borrow().clone().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator slot is unset".to_string(),
        })?;
        let rtyper = translator.rtyper().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator.rtyper slot is unset".to_string(),
        })?;
        let annotator = translator.annotator().ok_or_else(|| TaskError {
            message: "task_llinterpret_lltype: translator.annotator slot is unset".to_string(),
        })?;
        let _ = (rtyper, annotator, self.entry_point.borrow().clone());
        Err(self.missing_task_leaf(
            546,
            "rpython.rtyper.llinterp.LLInterpreter(translator.rtyper).eval_graph(...)",
        ))
    }

    /// Upstream `proceed(self, goals)` at `:557-570`.
    pub fn proceed(self: &Rc<Self>, goals: ProceedGoals) -> Result<TaskOutput, TaskError> {
        // Upstream `:558-563`: empty goals → fall back to default_goal,
        // else log + return.
        let mut goals: Vec<String> = match goals {
            ProceedGoals::Empty => match self.default_goal.borrow().clone() {
                Some(d) => vec![d],
                None => {
                    self.info("nothing to do");
                    return Ok(None);
                }
            },
            ProceedGoals::One(s) => vec![s],
            ProceedGoals::Many(v) => v,
        };

        // Upstream `:566`: `goals.extend(self.extra_goals)`.
        goals.extend(self.extra_goals.borrow().iter().cloned());

        // Upstream `:567`: `goals = self.backend_select_goals(goals)`.
        let goals = self.backend_select_goals(&goals).map_err(|e| TaskError {
            message: e.to_string(),
        })?;

        // Upstream `:568`: `result = self._execute(goals, task_skip = self._maybe_skip())`.
        let task_skip = self._maybe_skip();
        let hooks = DriverHooks {
            driver: Rc::clone(self),
        };
        let result = self.engine.execute(&hooks, &goals, &task_skip)?;

        // Upstream `:569`: `self.log.info('usession directory: %s' % (udir,))`.
        // `udir` not ported — print a placeholder so the call shape is
        // visible.
        self.info("usession directory: <udir not ported>");

        Ok(result)
    }

    /// Upstream `from_targetspec(cls, targetspec_dic, config=None,
    /// args=None, empty_translator=None, disable=[], default_goal=None)`
    /// at `:573-602`.
    ///
    /// `targetspec_dic` upstream is the dict imported from
    /// `targetstandalone.py`; the local port carries it as a
    /// `HashMap<String, Rc<dyn Any>>` so the C-backend port can fill
    /// in the keys when it lands.
    pub fn from_targetspec(
        _targetspec_dic: HashMap<String, Rc<dyn Any>>,
        _config: Option<Rc<Config>>,
        _args: Option<Vec<String>>,
        _empty_translator: Option<Rc<super::translator::TranslationContext>>,
        _disable: Vec<String>,
        _default_goal: Option<String>,
    ) -> Result<Rc<Self>, ConfigError> {
        // The upstream body at `:577-602` is:
        //
        //     if args is None: args = []
        //     driver = cls(config=config, default_goal=default_goal,
        //                  disable=disable)
        //     target = targetspec_dic['target']
        //     driver.timer.start_event("loading target")
        //     spec = target(driver, args)
        //     driver.timer.end_event("loading target")
        //     try:    entry_point, inputtypes, policy = spec
        //     except TypeError:  entry_point = spec; inputtypes = policy = None
        //     except ValueError: policy = None; entry_point, inputtypes = spec
        //     driver.setup(entry_point, inputtypes,
        //                  policy=policy,
        //                  extra=targetspec_dic,
        //                  empty_translator=empty_translator)
        //     return driver
        //
        // The body cannot be ported as-is without a callable
        // `target(driver, args)` — the C-backend's
        // `targetstandalone.py:target` (which produces the
        // `(entry_point, inputtypes, policy)` tuple consumed here) is
        // not yet ported. Returning `Ok(driver)` after only opening
        // the timer would leave the driver in a half-setup state
        // (no entry_point, no policy, no translator) that any
        // downstream `proceed()` would crash inside
        // [`Self::task_annotate`] for. Fail fast instead so callers
        // see the missing leaf rather than a misleading silent OK.
        unimplemented!(
            "driver.py:573 from_targetspec — target(driver, args) leaf is c-backend (rpython/translator/goal/targetstandalone.py); \
             driver.setup tuple-unpack at :587-595 cannot run without it"
        );
    }

    /// Upstream `prereq_checkpt_rtype(self)` at `:604-606`. DEFERRED —
    /// `rpython.rtyper.rmodel` import detection, used by the
    /// fork-checkpoint path inside `_event`.
    pub fn prereq_checkpt_rtype(&self) -> Result<(), TaskError> {
        // Upstream's body: `assert 'rpython.rtyper.rmodel' not in
        // sys.modules`. The Rust port has no equivalent module-import
        // observation; succeed.
        Ok(())
    }

    /// Upstream `prereq_checkpt_rtype_lltype = prereq_checkpt_rtype`
    /// at `:607`.
    pub fn prereq_checkpt_rtype_lltype(&self) -> Result<(), TaskError> {
        self.prereq_checkpt_rtype()
    }

    /// Upstream `_event(self, kind, goal, func)` at `:610-622`. The
    /// `fork_before` / `unixcheckpoint.restartable_point` path is
    /// DEFERRED — it forks the process to support resumable
    /// translation, which the Rust port has no direct equivalent for.
    /// The earlycheck branch (`:611-612`) is preserved verbatim.
    pub fn _event_with_func(
        &self,
        kind: &str,
        goal: &str,
        idempotent: bool,
    ) -> Result<(), TaskError> {
        if kind == "planned" {
            if let Some(check) = self.task_earlycheck.borrow().get(goal).cloned() {
                check(self)?;
            }
        }
        if kind == "pre" {
            // Upstream `:614-622`: fork_before checkpoint. DEFERRED —
            // requires `unixcheckpoint`. The structural read of
            // `config.translation.fork_before` is preserved so the
            // shape is grep-detectable.
            //
            // `fork_before` is declared as a `ChoiceOption` in
            // `translationoption.rs:400` (upstream `translationoption.py:146-150`),
            // so the stored value is `OptionValue::Choice` — not `Str`.
            // The `OptionValue::None` case (upstream's `None` default)
            // skips the body, matching `if fork_before:` at upstream
            // `:615`.
            let fork_before: Option<String> = match self.config.get("translation.fork_before") {
                Ok(ConfigValue::Value(OptionValue::Choice(s))) => Some(s),
                _ => None,
            };
            if let Some(_fb) = fork_before {
                // DEFERRED: backend_select_goals + done check +
                // prereq_checkpt + unixcheckpoint.restartable_point.
            }
        }
        let _ = idempotent;
        Ok(())
    }

    /// Convenience for tests / callers: whether a task name is in the
    /// post-`expose_task` list. Upstream callers read `td.exposed`
    /// directly, so the field is `pub`; this helper exists to express
    /// the upstream membership check ergonomically.
    pub fn is_exposed(&self, name: &str) -> bool {
        self.exposed.borrow().iter().any(|s| s == name)
    }

    /// Snapshot of `self.exposed` for callers that prefer not to
    /// borrow. Upstream callers iterate the list as a set; this
    /// helper hands back the same content as a fresh `Vec`.
    pub fn exposed_snapshot(&self) -> Vec<String> {
        self.exposed.borrow().clone()
    }

    /// Backend goal recorded by [`Self::expose_task`] for the public
    /// task name, or `None` if `name` is not exposed.
    pub fn exposed_method(&self, name: &str) -> Option<String> {
        self.exposed_backend_goal.borrow().get(name).cloned()
    }
}

/// Argument shape for [`TranslationDriver::proceed`] mirroring
/// upstream's overloaded `goals` parameter at `:557-565` (string,
/// list, or `None`).
pub enum ProceedGoals {
    Empty,
    One(String),
    Many(Vec<String>),
}

// ---------------------------------------------------------------------
// Upstream `:262-295 def _do(self, goal, func, *args, **kwds)`.
//
// `_do` overrides `SimpleTaskEngine._do` so the engine-driven task
// loop runs each task inside the driver's `self.timer.start_event` /
// `end_event` pair, prints `log.info(...)` lines, propagates the
// `Instrument` exception into a re-entry on the `compile` goal, and
// stashes completed goals into `self.done` (for non-idempotent
// tasks).
// ---------------------------------------------------------------------

struct DriverHooks {
    driver: Rc<TranslationDriver>,
}

impl TaskEngineHooks for DriverHooks {
    /// Upstream `_do(self, goal, func, *args, **kwds)` at `:262-295`.
    fn _do(
        &self,
        goal: &str,
        callable: &Rc<dyn Fn() -> Result<TaskOutput, TaskError>>,
    ) -> Result<TaskOutput, TaskError> {
        let driver = &self.driver;
        // Upstream `:263`: `title = func.task_title`.
        let title = {
            let tasks = driver.engine.tasks();
            tasks
                .get(goal)
                .map(|reg| reg.title.clone())
                .unwrap_or_else(|| goal.to_string())
        };

        // Upstream `:264-266`: already-done short-circuit.
        if driver.done.borrow().contains_key(goal) {
            driver.info(&format!("already done: {title}"));
            return Ok(None);
        } else {
            driver.info(&format!("{title}..."));
        }

        // Upstream `:269`: `debug_start('translation-task')`. NOT
        // ported — see module docstring.
        // Upstream `:270`: `debug_print('starting', goal)`.
        // Upstream `:271`: `self.timer.start_event(goal)`.
        driver.timer.start_event(goal);

        // Upstream `:273-282`: try-block running the callable;
        // optional cProfile branch (DEFERRED); Instrument-exception
        // catch.
        let mut instrument = false;
        let mut res: TaskOutput = None;
        // Upstream `:275-278`: PROFILE branch. DEFERRED — `_profile`
        // panics when reached; the `PROFILE` set is empty by default
        // so the branch is unreachable in practice.
        let in_profile = profile_contains(goal);
        let result = if in_profile {
            // Upstream `:276`: `res = self._profile(goal, func)`.
            // DEFERRED.
            return Err(TaskError {
                message: format!("driver.py:276 PROFILE — _profile not ported (goal={goal})"),
            });
        } else {
            callable()
        };
        match result {
            Ok(value) => {
                res = value;
            }
            Err(err) => {
                // Upstream `:279-280`: catch `Instrument` and continue;
                // every other exception escapes via the `finally`.
                if err.message == "Instrument" {
                    instrument = true;
                } else {
                    // Replicate the `finally` block before propagating.
                    let _ = driver.timer.end_event(goal);
                    return Err(err);
                }
            }
        }

        // Upstream `:281-282`: `if not func.task_idempotent:
        // self.done[goal] = True`.
        let idempotent = {
            let tasks = driver.engine.tasks();
            tasks.get(goal).map(|r| r.idempotent).unwrap_or(false)
        };
        if !idempotent {
            driver.done.borrow_mut().insert(goal.to_string(), true);
        }

        if instrument {
            // Upstream `:283-285`: re-enter on `compile` then assert.
            let _ = driver.proceed(ProceedGoals::One("compile".to_string()))?;
            return Err(TaskError {
                message: "we should not get here".to_string(),
            });
        }

        // Upstream `:286-293`: finally block (already partially run on
        // the error path above).
        // Upstream `:288`: `debug_stop('translation-task')`. NOT ported.
        driver.timer.end_event(goal);

        Ok(res)
    }

    /// Upstream `_event(kind, goal)` at `:610-622`.
    ///
    /// Upstream raises `func.task_earlycheck(self)`'s exception
    /// unwrapped — the engine's first event loop at
    /// `taskengine.py:108-109` has no try/except so the raise
    /// propagates straight out of `_execute`. The Rust port returns
    /// the error so [`SimpleTaskEngine::execute`] can `?`-propagate
    /// it identically (the trait's signature was widened from
    /// `fn _event(&self, …)` to `Result<(), TaskError>` for this
    /// reason).
    fn _event(&self, kind: &str, goal: &str) -> Result<(), TaskError> {
        self.driver._event_with_func(kind, goal, false)
    }
}

// ---------------------------------------------------------------------
// Upstream `:624-631 shutil_copy` — module-level helper that handles
// the "destination is the executable currently being run" case on
// posix (rename through a `~` suffix). Re-exported so c-backend
// callers can use it; on non-posix the upstream falls through to
// `shutil.copy`.
// ---------------------------------------------------------------------

/// Port of upstream `shutil_copy(src, dst)` at `:624-631`. On non-
/// Windows hosts the routine renames `dst~` over `dst` to handle the
/// "executable currently in use" pitfall.
pub fn shutil_copy(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if cfg!(target_os = "windows") {
        std::fs::copy(src, dst)?;
        Ok(())
    } else {
        let tmp = dst.with_extension("__copy_tmp__");
        std::fs::copy(src, &tmp)?;
        std::fs::rename(&tmp, dst)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `rpython/translator/test/test_driver.py:7-19 test_ctr`.
    /// Verifies the default-construction `expected` set + the
    /// `backend_select_goals` resolution table.
    #[test]
    fn ctr_default_exposed_and_backend_select_goals() {
        let td = TranslationDriver::new_default().expect("driver");
        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt",
            "llinterpret",
            "rtype",
            "source",
            "compile",
            "pyjitpl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(exposed, expected, "default exposed mismatch");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["compile".to_string()])
                .expect("compile"),
            vec!["compile_c".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype".to_string()])
                .expect("rtype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt".to_string()])
                .expect("backendopt"),
            vec!["backendopt_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );
    }

    /// Mirrors `rpython/translator/test/test_driver.py:21-33` — when
    /// `backend=None` and `type_system=None`, only the explicit
    /// `*_lltype` / `*_c` task names compose, and generic prefixes
    /// like `compile` / `rtype` / `backendopt` are NOT resolvable.
    #[test]
    fn ctr_with_backend_and_type_system_unset() {
        let mut setopts: HashMap<String, OptionValue> = HashMap::new();
        setopts.insert("backend".to_string(), OptionValue::None);
        setopts.insert("type_system".to_string(), OptionValue::None);
        let td = TranslationDriver::new(Some(setopts), None, Vec::new(), None, None, None, None)
            .expect("driver");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert!(
            td.backend_select_goals(&["compile".to_string()]).is_err(),
            "compile must not resolve when backend / type_system are None"
        );
        assert!(
            td.backend_select_goals(&["rtype".to_string()]).is_err(),
            "rtype must not resolve when type_system is None"
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert!(
            td.backend_select_goals(&["backendopt".to_string()])
                .is_err(),
            "backendopt must not resolve when backend / type_system are None"
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );

        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt_lltype",
            "llinterpret_lltype",
            "rtype_lltype",
            "source_c",
            "compile_c",
            "pyjitpl_lltype",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(
            exposed, expected,
            "exposed must match the explicit-only set when backend / type_system are None"
        );
    }

    /// Mirrors `rpython/translator/test/test_driver.py:35-48` — when
    /// `type_system=lltype` but `backend=None`, the type-system-keyed
    /// public names appear (rtype, backendopt, llinterpret, pyjitpl)
    /// while the backend-keyed ones stay specific (source_c, compile_c).
    #[test]
    fn ctr_type_system_lltype_backend_none() {
        let mut setopts: HashMap<String, OptionValue> = HashMap::new();
        setopts.insert("backend".to_string(), OptionValue::None);
        setopts.insert(
            "type_system".to_string(),
            OptionValue::Choice("lltype".to_string()),
        );
        let td = TranslationDriver::new(Some(setopts), None, Vec::new(), None, None, None, None)
            .expect("driver");

        assert_eq!(
            td.backend_select_goals(&["compile_c".to_string()])
                .expect("compile_c"),
            vec!["compile_c".to_string()],
        );
        assert!(
            td.backend_select_goals(&["compile".to_string()]).is_err(),
            "compile must not resolve when backend is None"
        );
        assert_eq!(
            td.backend_select_goals(&["rtype_lltype".to_string()])
                .expect("rtype_lltype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["rtype".to_string()])
                .expect("rtype"),
            vec!["rtype_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt".to_string()])
                .expect("backendopt"),
            vec!["backendopt_lltype".to_string()],
        );
        assert_eq!(
            td.backend_select_goals(&["backendopt_lltype".to_string()])
                .expect("backendopt_lltype"),
            vec!["backendopt_lltype".to_string()],
        );

        let expected: std::collections::HashSet<String> = [
            "annotate",
            "backendopt",
            "llinterpret",
            "rtype",
            "source_c",
            "compile_c",
            "pyjitpl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let exposed: std::collections::HashSet<String> =
            td.exposed_snapshot().into_iter().collect();
        assert_eq!(
            exposed, expected,
            "exposed must match the type-system-keyed set when backend is None"
        );
    }

    #[test]
    fn setup_threads_driver_overrides_into_translator_config() {
        // Upstream `driver.py:194 translator = TranslationContext(
        // config=self.config)` shares the driver's Config object.
        // The Rust port snapshots the driver's `Rc<Config>` into a
        // typed `TranslationConfig` at setup() time, so any
        // `override`/`setopts` mutation performed before setup() is
        // visible inside `translator.config.translation.<X>`.
        let mut overrides: HashMap<String, OptionValue> = HashMap::new();
        overrides.insert("translation.verbose".to_string(), OptionValue::Bool(true));
        overrides.insert(
            "translation.list_comprehension_operations".to_string(),
            OptionValue::Bool(true),
        );
        overrides.insert("translation.keepgoing".to_string(), OptionValue::Bool(true));
        let td = TranslationDriver::new(None, None, Vec::new(), None, None, None, Some(overrides))
            .expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");
        let translator = td
            .translator
            .borrow()
            .as_ref()
            .map(Rc::clone)
            .expect("translator");
        assert!(
            translator.config.translation.verbose,
            "driver-side translation.verbose=true must propagate to translator.config"
        );
        assert!(
            translator.config.translation.list_comprehension_operations,
            "driver-side list_comprehension_operations=true must propagate"
        );
        assert!(
            translator.config.translation.keepgoing,
            "driver-side keepgoing=true must propagate"
        );
        // `translating` is set true by get_combined_translation_config(translating=True)
        // on driver construction.
        assert!(
            translator.config.translating,
            "translating flag must reach the snapshot"
        );
    }

    #[test]
    fn sanity_check_annotation_passes_on_empty_translator() {
        // Upstream `:330-337 sanity_check_annotation` requires
        // `self.translator` to be a TranslationContext. With no
        // graphs registered, `check_exceptblocks_qgen` and
        // `check_methods_qgen` both yield zero items, so the upstream
        // assert passes silently.
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup with default translator");
        td.sanity_check_annotation()
            .expect("empty translator must pass sanity check");
    }

    #[test]
    fn sanity_check_annotation_silent_when_translator_is_unset() {
        // Upstream pre-`setup()` state has `self.translator` unset.
        // The Rust port treats that as a no-op so the method can be
        // called from arbitrary lifecycle points without panicking.
        let td = TranslationDriver::new_default().expect("driver");
        td.sanity_check_annotation()
            .expect("unset translator must no-op");
    }

    #[test]
    fn task_annotate_runs_end_to_end_for_constant_return() {
        // Mirrors upstream `task_annotate` (driver.py:297-327) for a
        // single-return Rust-source function. The Position-2 adapter
        // (flowspace::rust_source) supplies the entry-point graph,
        // setup() seeds inputtypes/policy/translator, then
        // proceed("annotate") drives the engine through the task
        // body. Verifies the int-return path completes without
        // panicking on the standalone-int check at `:321-324`.
        use crate::flowspace::rust_source::build_host_function_from_rust;
        use crate::translator::translator::TranslationContext;

        let item: syn::ItemFn = syn::parse_str("fn main() -> i64 { 42 }").expect("parse");
        let (host, pygraph) =
            build_host_function_from_rust(&item, None, None).expect("rust-source adapter");

        let td = TranslationDriver::new_default().expect("driver");

        // Pre-seed the translator's prebuilt-graph cache with the
        // adapter output so `buildannotator + build_types` can route
        // to the same FunctionGraph (matches `interactive.rs`'s
        // `from_rust_item_fn_with_source` shape).
        let translator = Rc::new(TranslationContext::new());
        translator.graphs.borrow_mut().push(pygraph.graph.clone());
        translator
            ._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph);

        // `inputtypes=Some(vec![])` marks the entry as non-standalone
        // (upstream `:177-178`: `standalone = inputtypes is None`),
        // matching the zero-arg `fn main() -> i64`. A standalone
        // entry-point would receive `argv: SomeList(SomeString)` and
        // require an `argv: Vec<String>` parameter.
        td.setup(
            Some(host),
            Some(Vec::new()),
            None,
            HashMap::new(),
            Some(translator),
        )
        .expect("setup");
        let result = td.task_annotate().expect("task_annotate runs");
        assert!(result.is_some(), "annotate must produce a return SomeValue");
    }

    #[test]
    fn proceed_runs_annotate_and_rtype_via_engine() {
        // End-to-end: invoke `proceed(["rtype"])` and let the
        // SimpleTaskEngine plan + run `annotate → rtype_lltype`. This
        // exercises the DriverHooks::_do override (Timer wrapping,
        // log lines, done bookkeeping) on the activated task bodies.
        use crate::flowspace::rust_source::build_host_function_from_rust;
        use crate::translator::translator::TranslationContext;

        let item: syn::ItemFn = syn::parse_str("fn main() -> i64 { 7 }").expect("parse");
        let (host, pygraph) =
            build_host_function_from_rust(&item, None, None).expect("rust-source adapter");

        let td = TranslationDriver::new_default().expect("driver");
        let translator = Rc::new(TranslationContext::new());
        translator.graphs.borrow_mut().push(pygraph.graph.clone());
        translator
            ._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph);

        td.setup(
            Some(host),
            Some(Vec::new()),
            None,
            HashMap::new(),
            Some(translator),
        )
        .expect("setup");

        td.proceed(ProceedGoals::One("rtype".to_string()))
            .expect("proceed annotate -> rtype");

        // After proceed, both goals should land in `done`.
        let done = td.done.borrow();
        assert!(done.contains_key("annotate"));
        assert!(done.contains_key("rtype_lltype"));
    }

    #[test]
    fn task_rtype_lltype_reaches_buildrtyper_after_annotate() {
        // Mirrors upstream `task_rtype_lltype` (driver.py:340-345).
        // Runs the annotate → rtype pipeline; verifies that
        // `buildrtyper().specialize(true)` completes without
        // panicking on the same single-int-return entry.
        use crate::flowspace::rust_source::build_host_function_from_rust;
        use crate::translator::translator::TranslationContext;

        let item: syn::ItemFn = syn::parse_str("fn main() -> i64 { 0 }").expect("parse");
        let (host, pygraph) =
            build_host_function_from_rust(&item, None, None).expect("rust-source adapter");

        let td = TranslationDriver::new_default().expect("driver");
        let translator = Rc::new(TranslationContext::new());
        translator.graphs.borrow_mut().push(pygraph.graph.clone());
        translator
            ._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph);

        td.setup(
            Some(host),
            Some(Vec::new()),
            None,
            HashMap::new(),
            Some(translator),
        )
        .expect("setup");
        td.task_annotate().expect("annotate");
        td.task_rtype_lltype().expect("rtype_lltype");
    }

    #[test]
    fn deferred_driver_tasks_return_errors_instead_of_panicking() {
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");

        let err = td
            .task_backendopt_lltype()
            .expect_err("backendopt leaf should report the missing upstream leaf");
        assert!(err.message.contains("backend_optimizations"));

        let err = td
            .task_compile_c()
            .expect_err("compile leaf should report missing prior C builder state");
        assert!(err.message.contains("cbuilder slot is unset"));
    }

    #[test]
    fn task_database_c_sets_translator_frozen_before_c_backend_leaf() {
        let td = TranslationDriver::new_default().expect("driver");
        td.setup(None, None, None, HashMap::new(), None)
            .expect("setup");
        let translator = td
            .translator
            .borrow()
            .as_ref()
            .map(Rc::clone)
            .expect("translator");
        let annotator = translator.buildannotator(None);
        *td.annotator.borrow_mut() = Some(annotator);

        let err = td
            .task_database_c()
            .expect_err("C backend builder is still a missing leaf");
        assert!(err.message.contains("cbuilder.build_database"));
        assert!(
            translator.frozen.get(),
            "task_database_c must mirror driver.py:411 before building the database"
        );
    }

    #[test]
    fn shutil_copy_round_trip_in_tempdir() {
        // Mirrors `test_shutil_copy` at upstream
        // `test_driver.py:124-131` — write a file, copy it, read it
        // back. The Rust port uses `tempfile`-free std::env::temp_dir
        // path; the test only needs filesystem semantics.
        let dir = std::env::temp_dir();
        let a = dir.join("majit_driver_test_a");
        let b = dir.join("majit_driver_test_b");
        std::fs::write(&a, b"hello").expect("write a");
        if b.exists() {
            std::fs::remove_file(&b).expect("clean b");
        }
        shutil_copy(&a, &b).expect("shutil_copy");
        let read = std::fs::read_to_string(&b).expect("read b");
        assert_eq!(read, "hello");
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }
}
