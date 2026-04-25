//! Port of `rpython/translator/c/dlltool.py`.
//!
//! Upstream is 41 LOC of one class, `CLibraryBuilder`, that subclasses
//! `CBuilder` and exposes a `getentrypointptr` returning a list of
//! function pointers (`getfunctionptr` per registered function), a
//! `gen_makefile` no-op stub, and a `compile` body that builds the
//! shared library via `translator.platform.compile(..., standalone=False,
//! outputfilename=name)`.
//!
//! ```python
//! class CLibraryBuilder(CBuilder):
//!     standalone = False
//!     split = True
//!
//!     def __init__(self, *args, **kwds):
//!         self.functions = kwds.pop('functions')
//!         self.name = kwds.pop('name')
//!         CBuilder.__init__(self, *args, **kwds)
//!
//!     def getentrypointptr(self):
//!         entrypoints = []
//!         bk = self.translator.annotator.bookkeeper
//!         for f, _ in self.functions:
//!             graph = bk.getdesc(f).getuniquegraph()
//!             entrypoints.append(getfunctionptr(graph))
//!         return entrypoints
//!
//!     def compile(self):
//!         extsymeci = ExternalCompilationInfo()  # empty
//!         self.eci = self.eci.merge(extsymeci)
//!         files = [self.c_source_filename] + self.extrafiles
//!         files += self.eventually_copy(self.eci.separate_module_files)
//!         self.eci.separate_module_files = ()
//!         oname = self.name
//!         self.so_name = self.translator.platform.compile(files, self.eci,
//!                                                         standalone=False,
//!                                                         outputfilename=oname)
//!
//!     def get_entry_point(self, isolated=False):
//!         return self.so_name
//! ```
//!
//! (`rpython/translator/c/dlltool.py:1-41`.)

use std::any::Any;
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use crate::config::config::Config;
use crate::flowspace::model::HostObject;
use crate::translator::c::genc::CBuilder;
use crate::translator::driver::EntryPointSpec;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// Port of `class CLibraryBuilder(CBuilder)` at
/// `dlltool.py:7-40`.
#[derive(Clone)]
pub struct CLibraryBuilder {
    /// Inherited `CBuilder` state. `:8-9` flip the class-level
    /// `standalone = False; split = True` defaults.
    pub base: CBuilder,
    /// `self.functions = kwds.pop('functions')` (`:13`).
    pub functions: Vec<EntryPointSpec>,
    /// `self.name = kwds.pop('name')` (`:14`).
    pub name: String,
    /// `self.so_name` (`:36`). Set by `compile` after
    /// `translator.platform.compile` returns the shared-library path.
    pub so_name: RefCell<Option<PathBuf>>,
}

impl CLibraryBuilder {
    /// Upstream `CLibraryBuilder.__init__(*args, **kwds)` at `:11-14`.
    pub fn new(
        translator: Rc<TranslationContext>,
        entrypoint: Option<HostObject>,
        config: Rc<Config>,
        functions: Vec<EntryPointSpec>,
        name: String,
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
        // Upstream `:9`: `split = True`.
        base.split = true;
        Self {
            base,
            functions,
            name,
            so_name: RefCell::new(None),
        }
    }

    /// Upstream class-level `standalone = False` (`:8`).
    pub const STANDALONE: bool = false;

    /// Upstream `getentrypointptr(self)` at `:16-22`.
    pub fn getentrypointptr(&self) -> Result<Vec<Rc<dyn Any>>, TaskError> {
        Err(TaskError {
            message: "dlltool.py:16 CLibraryBuilder.getentrypointptr — leaf bookkeeper.getdesc / getuniquegraph / getfunctionptr not yet ported".to_string(),
        })
    }

    /// Upstream `gen_makefile(self, targetdir, exe_name=None,
    /// headers_to_precompile=[])` at `:24-26`.
    pub fn gen_makefile(
        &self,
        _targetdir: PathBuf,
        _exe_name: Option<String>,
        _headers_to_precompile: Vec<PathBuf>,
    ) -> Result<(), TaskError> {
        // Upstream body is `pass # XXX finish` — the local port mirrors
        // that no-op: returning Ok keeps callers parity-correct.
        Ok(())
    }

    /// Upstream `compile(self)` at `:28-37`.
    pub fn compile(&self, _exe_name: Option<String>) -> Result<(), TaskError> {
        Err(TaskError {
            message: "dlltool.py:28 CLibraryBuilder.compile — leaf ExternalCompilationInfo / platform.compile not yet ported".to_string(),
        })
    }

    /// Upstream `get_entry_point(self, isolated=False)` at `:39-40`.
    pub fn get_entry_point(&self) -> Result<PathBuf, TaskError> {
        match self.so_name.borrow().clone() {
            Some(p) => Ok(p),
            None => Err(TaskError {
                message: "dlltool.py:36 CLibraryBuilder.so_name — slot is unset; compile() must run first".to_string(),
            }),
        }
    }
}

impl std::fmt::Debug for CLibraryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CLibraryBuilder")
            .field("name", &self.name)
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

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
    fn clibrarybuilder_inherits_cbuilder_with_split_true_per_line_9() {
        // Upstream `:8-9`: `standalone = False; split = True`.
        let (translator, config) = fixture_translator_and_config();
        let lb = CLibraryBuilder::new(
            translator,
            None,
            config,
            Vec::new(),
            "libtesting".to_string(),
            None,
            None,
            Vec::new(),
        );
        assert!(lb.base.split, "CLibraryBuilder must flip split=True");
        assert_eq!(lb.name, "libtesting");
        assert!(!CLibraryBuilder::STANDALONE);
    }

    #[test]
    fn gen_makefile_is_no_op_per_upstream_pass_xxx_finish() {
        // Upstream `:24-26`: body is `pass # XXX finish`. The local port
        // mirrors that with `Ok(())`.
        let (translator, config) = fixture_translator_and_config();
        let lb = CLibraryBuilder::new(
            translator,
            None,
            config,
            Vec::new(),
            "x".to_string(),
            None,
            None,
            Vec::new(),
        );
        let ok = lb.gen_makefile(PathBuf::from("/tmp"), None, Vec::new());
        assert!(ok.is_ok(), "gen_makefile must mirror upstream `pass`");
    }

    #[test]
    fn compile_returns_task_error_until_platform_compile_lands() {
        // Upstream `:28-37 compile` — depends on
        // `ExternalCompilationInfo` and `translator.platform.compile`.
        let (translator, config) = fixture_translator_and_config();
        let lb = CLibraryBuilder::new(
            translator,
            None,
            config,
            Vec::new(),
            "x".to_string(),
            None,
            None,
            Vec::new(),
        );
        let err = lb.compile(None).expect_err("must be DEFERRED");
        assert!(err.message.contains("dlltool.py:28"), "{}", err.message);
    }

    #[test]
    fn get_entry_point_reads_so_name_slot() {
        // Upstream `:39-40 get_entry_point(self, isolated=False): return
        // self.so_name`. With the slot unset (no compile() pass) the
        // port surfaces a TaskError — never panics.
        let (translator, config) = fixture_translator_and_config();
        let lb = CLibraryBuilder::new(
            translator,
            None,
            config,
            Vec::new(),
            "x".to_string(),
            None,
            None,
            Vec::new(),
        );
        assert!(lb.get_entry_point().is_err());
        *lb.so_name.borrow_mut() = Some(PathBuf::from("/tmp/libx.so"));
        assert_eq!(lb.get_entry_point().unwrap(), PathBuf::from("/tmp/libx.so"),);
    }
}
