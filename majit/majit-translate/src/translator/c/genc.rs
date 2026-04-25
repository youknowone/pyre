//! Port of `rpython/translator/c/genc.py`.
//!
//! Upstream is 944 LOC of three classes plus helpers:
//! * `CCompilerDriver` (`:40-58`) — platform-bound build invocation.
//! * `CBuilder` (`:62-235`) — common base for standalone and library
//!   builders. Holds `translator`, `entrypoint`, `config`, `gcpolicy`,
//!   `gchooks`, `secondary_entrypoints`, plus the `c_source_filename`,
//!   `_compiled`, `modulename`, `split` class attrs, and the
//!   [`build_database`](Self::build_database) /
//!   [`generate_source`](Self::generate_source) /
//!   [`compile`](Self::compile) flow.
//! * `CStandaloneBuilder` (`:237-510`) — exe-producing subclass. Adds
//!   `executable_name`, `shared_library_name`, `_entrypoint_wrapper`
//!   slots and overrides `getentrypointptr`, `compile`, `gen_makefile`.
//!
//! This file ports only the structural shells: the constructor
//! (`__init__`), attribute layout, and the public method signatures
//! consumed by [`crate::translator::driver`]. Each method body returns a
//! [`TaskError`] citing the upstream line of the unported leaf;
//! convergence is per-method as the database / source-emit / compile
//! chain lands.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use crate::config::config::Config;
use crate::flowspace::model::HostObject;
use crate::translator::driver::EntryPointSpec;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// Upstream `genc.py:526-…` `LowLevelDatabase`. The full type lives in
/// `rpython/translator/c/database.py`; only the slot identity is needed
/// for the driver to thread `self.database = cbuilder.build_database()`
/// from `task_database_c` to `task_source_c`.
#[derive(Clone, Debug, Default)]
pub struct LowLevelDatabase;

/// Port of upstream `class CBuilder(object)` at `genc.py:62-235`.
///
/// Field shape mirrors upstream's `__init__` (`:68-78`) plus the
/// class-level defaults at `:62-66`. Each method below corresponds to a
/// `CBuilder` method by upstream-name; bodies return [`TaskError`] until
/// the C-backend leaf lands.
#[derive(Clone)]
pub struct CBuilder {
    // Upstream class-level slots at `:63-66`.
    /// `c_source_filename = None` (`:63`). Set by `generate_source`.
    pub c_source_filename: RefCell<Option<PathBuf>>,
    /// `_compiled = False` (`:64`). Flipped by `compile`.
    pub _compiled: bool,
    /// `modulename = None` (`:65`). Set by `generate_source` after
    /// `uniquemodulename(...)`.
    pub modulename: RefCell<Option<String>>,
    /// `split = False` (`:66`). Overridden to `True` by
    /// `CStandaloneBuilder` and `CLibraryBuilder`.
    pub split: bool,

    // Upstream `__init__` slots at `:69-78`.
    /// `self.translator` (`:70`).
    pub translator: Rc<TranslationContext>,
    /// `self.entrypoint` (`:71`).
    pub entrypoint: Option<HostObject>,
    /// `self.entrypoint_name = getattr(entrypoint, 'func_name', None)` (`:72`).
    pub entrypoint_name: Option<String>,
    /// `self.originalentrypoint = entrypoint` (`:73`).
    pub originalentrypoint: Option<HostObject>,
    /// `self.config = config` (`:74`).
    pub config: Rc<Config>,
    /// `self.gcpolicy = gcpolicy` (`:75`). Opaque to the local port —
    /// upstream's `GCPolicy` hierarchy is not yet ported.
    pub gcpolicy: Option<Rc<dyn Any>>,
    /// `self.gchooks = gchooks` (`:76`).
    pub gchooks: Option<Rc<dyn Any>>,
    /// `self.eci = self.get_eci()` (`:77`). Upstream returns an
    /// `ExternalCompilationInfo`; the local port keeps the slot opaque
    /// (`Option<Rc<dyn Any>>`) until `cbuild.ExternalCompilationInfo`
    /// lands.
    pub eci: RefCell<Option<Rc<dyn Any>>>,
    /// `self.secondary_entrypoints = secondary_entrypoints` (`:78`).
    pub secondary_entrypoints: Vec<EntryPointSpec>,

    // Upstream `targetdir` slot, set by `generate_source` at `:187`.
    pub targetdir: RefCell<Option<PathBuf>>,

    // Upstream `db` slot, set by `build_database` at `:103`.
    pub db: RefCell<Option<LowLevelDatabase>>,
}

impl CBuilder {
    /// Upstream `CBuilder.__init__(self, translator, entrypoint, config,
    /// gcpolicy=None, gchooks=None, secondary_entrypoints=())` at
    /// `:68-78`.
    pub fn new(
        translator: Rc<TranslationContext>,
        entrypoint: Option<HostObject>,
        config: Rc<Config>,
        gcpolicy: Option<Rc<dyn Any>>,
        gchooks: Option<Rc<dyn Any>>,
        secondary_entrypoints: Vec<EntryPointSpec>,
    ) -> Self {
        // Upstream `:72`: `self.entrypoint_name = getattr(self.entrypoint,
        // 'func_name', None)`. The local `GraphFunc.name` field carries
        // the same `__name__`/`func_name` value.
        let entrypoint_name = entrypoint
            .as_ref()
            .and_then(|h| h.user_function().map(|gf| gf.name.clone()));
        Self {
            c_source_filename: RefCell::new(None),
            _compiled: false,
            modulename: RefCell::new(None),
            split: false,
            translator,
            entrypoint: entrypoint.clone(),
            entrypoint_name,
            originalentrypoint: entrypoint,
            config,
            gcpolicy,
            gchooks,
            eci: RefCell::new(None),
            secondary_entrypoints,
            targetdir: RefCell::new(None),
            db: RefCell::new(None),
        }
    }

    /// Upstream `CBuilder.build_database(self)` at `:87-138`.
    ///
    /// DEFERRED — depends on `LowLevelDatabase`
    /// (`rpython/translator/c/database.py`),
    /// `getexceptiontransformer` (`translator.exceptiontransform`),
    /// `gc.name_to_gcpolicy` (`rpython/memory/gc/`), `exports.EXPORTS`
    /// (`rpython/translator/c/extfunc.py`) and `getfunctionptr`
    /// (`rpython/rtyper/lltypesystem/lltype.py`). None are ported yet.
    pub fn build_database(&self) -> Result<LowLevelDatabase, TaskError> {
        Err(TaskError {
            message: "genc.py:87 CBuilder.build_database — leaf LowLevelDatabase / getexceptiontransformer / gc.name_to_gcpolicy not yet ported".to_string(),
        })
    }

    /// Upstream `CBuilder.generate_source(self, db=None, defines={},
    /// exe_name=None)` at `:175-235`.
    pub fn generate_source(
        &self,
        _db: Option<&LowLevelDatabase>,
        _defines: &HashMap<String, String>,
        _exe_name: Option<String>,
    ) -> Result<PathBuf, TaskError> {
        Err(TaskError {
            message: "genc.py:175 CBuilder.generate_source — leaf SourceGenerator / uniquemodulename / udir not yet ported".to_string(),
        })
    }

    /// Upstream `CBuilder.compile(self, **kwds)` at `genc.py:225-…`
    /// (overridden in subclasses).
    pub fn compile(&self, _exe_name: Option<String>) -> Result<(), TaskError> {
        Err(TaskError {
            message: "genc.py:225 CBuilder.compile — leaf platform.compile not yet ported"
                .to_string(),
        })
    }
}

impl std::fmt::Debug for CBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBuilder")
            .field("entrypoint_name", &self.entrypoint_name)
            .field("split", &self.split)
            .field("modulename", &self.modulename)
            .finish_non_exhaustive()
    }
}

/// Port of upstream `class CStandaloneBuilder(CBuilder)` at
/// `genc.py:237-510`.
///
/// Adds the standalone-only slots `executable_name` (`:240`),
/// `shared_library_name` (`:241`), `_entrypoint_wrapper` (`:242`) and
/// overrides `getentrypointptr` (`:246-281`), `compile`, `gen_makefile`.
#[derive(Clone)]
pub struct CStandaloneBuilder {
    /// Inherited `CBuilder` state, with `split = true` per `:239`.
    pub base: CBuilder,
    /// `executable_name = None` (`:240`). Set by `compile` after
    /// `platform.compile` returns the executable path.
    pub executable_name: RefCell<Option<PathBuf>>,
    /// `shared_library_name = None` (`:241`). Set when the platform
    /// emits a companion shared library.
    pub shared_library_name: RefCell<Option<PathBuf>>,
    /// `_entrypoint_wrapper = None` (`:242`). Cached wrapper produced
    /// by `getentrypointptr`.
    pub _entrypoint_wrapper: RefCell<Option<Rc<dyn Any>>>,
    /// `make_entrypoint_wrapper = True  # for tests` (`:243`).
    pub make_entrypoint_wrapper: bool,
    /// Optional Windows-only sibling `pypyw.exe`. Upstream sets this in
    /// `CStandaloneBuilder.compile` after the platform build; set to
    /// `None` until that path lands.
    pub executable_name_w: RefCell<Option<PathBuf>>,
}

impl CStandaloneBuilder {
    /// Upstream `class CStandaloneBuilder(CBuilder)` inherits `__init__`
    /// from `CBuilder` and overrides only class-level defaults. The
    /// Rust port mirrors this with a constructor that accepts the same
    /// arguments and flips the `split` flag to `true` per `:239`.
    pub fn new(
        translator: Rc<TranslationContext>,
        entrypoint: Option<HostObject>,
        config: Rc<Config>,
        gcpolicy: Option<Rc<dyn Any>>,
        gchooks: Option<Rc<dyn Any>>,
        secondary_entrypoints: Vec<EntryPointSpec>,
    ) -> Self {
        let mut base = CBuilder::new(
            translator,
            entrypoint,
            config,
            gcpolicy,
            gchooks,
            secondary_entrypoints,
        );
        // Upstream `:238-239`: `standalone = True; split = True`.
        base.split = true;
        Self {
            base,
            executable_name: RefCell::new(None),
            shared_library_name: RefCell::new(None),
            _entrypoint_wrapper: RefCell::new(None),
            make_entrypoint_wrapper: true,
            executable_name_w: RefCell::new(None),
        }
    }

    /// Upstream class-level `standalone = True` (`:238`). Exposed as a
    /// const for callers that need to discriminate at the type level.
    pub const STANDALONE: bool = true;

    /// Upstream `getentrypointptr(self)` at `:246-281`.
    pub fn getentrypointptr(&self) -> Result<Rc<dyn Any>, TaskError> {
        Err(TaskError {
            message: "genc.py:246 CStandaloneBuilder.getentrypointptr — leaf MixLevelHelperAnnotator / lltype_to_annotation not yet ported".to_string(),
        })
    }

    /// Upstream `cmdexec(self, args, env, err, expect_crash, exe)` at
    /// `:283-323`. DEFERRED — depends on `translator.platform.execute`.
    pub fn cmdexec(
        &self,
        _args: &str,
        _env: Option<HashMap<String, String>>,
    ) -> Result<String, TaskError> {
        Err(TaskError {
            message:
                "genc.py:283 CStandaloneBuilder.cmdexec — leaf platform.execute not yet ported"
                    .to_string(),
        })
    }

    /// Upstream `CStandaloneBuilder.compile(self, exe_name=None)`.
    pub fn compile(&self, exe_name: Option<String>) -> Result<(), TaskError> {
        // Delegate the body to `CBuilder.compile` until the platform
        // build chain lands; the standalone-specific bookkeeping
        // (executable_name, c_entryp on the driver) happens in the
        // caller after this returns.
        self.base.compile(exe_name)
    }

    /// Upstream `CStandaloneBuilder.get_entry_point()` returns the
    /// platform-built executable path. Stub returns the cached
    /// `executable_name` slot.
    pub fn get_entry_point(&self) -> Result<PathBuf, TaskError> {
        match self.executable_name.borrow().clone() {
            Some(p) => Ok(p),
            None => Err(TaskError {
                message: "genc.py:240 CStandaloneBuilder.executable_name — slot is unset; compile() must run first".to_string(),
            }),
        }
    }
}

impl std::fmt::Debug for CStandaloneBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CStandaloneBuilder")
            .field("base", &self.base)
            .field("executable_name", &self.executable_name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::translationoption::get_combined_translation_config;
    use crate::translator::translator::TranslationContext;

    fn fixture_translator_and_config() -> (Rc<TranslationContext>, Rc<Config>) {
        let translator = Rc::new(TranslationContext::new());
        let config = get_combined_translation_config(None, None, None, true).expect("config");
        (translator, config)
    }

    #[test]
    fn cbuilder_init_mirrors_upstream_slots() {
        // Upstream `:68-78`: `__init__(translator, entrypoint, config,
        // gcpolicy=None, gchooks=None, secondary_entrypoints=())` —
        // every slot must be writable on construction.
        let (translator, config) = fixture_translator_and_config();
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        // `:63` `c_source_filename = None` — pristine slot.
        assert!(cb.c_source_filename.borrow().is_none());
        // `:64` `_compiled = False`.
        assert!(!cb._compiled);
        // `:65` `modulename = None`.
        assert!(cb.modulename.borrow().is_none());
        // `:66` `split = False`.
        assert!(!cb.split);
        // `:78` `secondary_entrypoints = ()` — empty by default.
        assert!(cb.secondary_entrypoints.is_empty());
    }

    #[test]
    fn cstandalonebuilder_flips_split_flag_per_239() {
        // Upstream `:238-239`: `standalone = True; split = True`.
        let (translator, config) = fixture_translator_and_config();
        let csb = CStandaloneBuilder::new(translator, None, config, None, None, Vec::new());
        assert!(csb.base.split, "CStandaloneBuilder must flip split=True");
        assert!(CStandaloneBuilder::STANDALONE);
    }

    #[test]
    fn build_database_returns_task_error_until_leaf_lands() {
        // Upstream `:87-138 build_database` — the local port returns
        // `TaskError` citing the upstream line until LowLevelDatabase /
        // getexceptiontransformer / gc.name_to_gcpolicy land.
        let (translator, config) = fixture_translator_and_config();
        let cb = CBuilder::new(translator, None, config, None, None, Vec::new());
        let err = cb.build_database().expect_err("must be DEFERRED");
        assert!(
            err.message.contains("genc.py:87"),
            "must cite genc.py:87, got: {}",
            err.message
        );
    }

    #[test]
    fn get_entry_point_returns_error_when_executable_unset() {
        // Upstream `getentrypointptr` reads `self.executable_name` set
        // by `compile`. Without a compile pass, the slot is None and
        // the local port surfaces a TaskError.
        let (translator, config) = fixture_translator_and_config();
        let csb = CStandaloneBuilder::new(translator, None, config, None, None, Vec::new());
        let err = csb.get_entry_point().expect_err("compile must run first");
        assert!(err.message.contains("executable_name"), "{}", err.message);
    }
}
