//! `translator/c/` — port of `rpython/translator/c/`.
//!
//! Upstream is the C-backend tree. The driver tasks `task_database_c` /
//! `task_source_c` / `task_compile_c` (driver.py:408-541) construct one
//! of two builder shapes here:
//!
//! | upstream                                        | local              |
//! |-------------------------------------------------|--------------------|
//! | `rpython/translator/c/genc.py:CBuilder`          | [`genc::CBuilder`] |
//! | `rpython/translator/c/genc.py:CStandaloneBuilder`| [`genc::CStandaloneBuilder`] |
//! | `rpython/translator/c/dlltool.py:CLibraryBuilder`| [`dlltool::CLibraryBuilder`] |
//!
//! Each builder method returns [`crate::translator::tool::taskengine::TaskError`]
//! citing the still-unported leaf — the structure mirrors upstream
//! (constructor signature, attribute names, method order) so the driver
//! body can be written against the upstream call shape and only the leaf
//! body remains to land on a per-method basis.

pub mod dlltool;
pub mod genc;

use std::path::PathBuf;

use crate::translator::tool::taskengine::TaskError;

/// Sum type for the two builder shapes that `task_database_c` can pick
/// between. Upstream `driver.py:419-432` writes either
/// `CStandaloneBuilder(...)` or `CLibraryBuilder(...)` into `self.cbuilder`
/// — one Python attribute can hold either subclass via duck typing.
/// Rust requires a sum type for the same slot.
#[derive(Clone, Debug)]
pub enum CBuilderRef {
    Standalone(genc::CStandaloneBuilder),
    Library(dlltool::CLibraryBuilder),
}

impl CBuilderRef {
    /// Upstream `cbuilder.build_database()` (driver.py:435).
    pub fn build_database(&self) -> Result<genc::LowLevelDatabase, TaskError> {
        match self {
            CBuilderRef::Standalone(b) => b.base.build_database(),
            CBuilderRef::Library(b) => b.base.build_database(),
        }
    }

    /// Upstream `cbuilder.generate_source(database, defines, exe_name=...)`
    /// at `driver.py:454-455`.
    pub fn generate_source(
        &self,
        db: &genc::LowLevelDatabase,
        defines: &std::collections::HashMap<String, String>,
        exe_name: Option<String>,
    ) -> Result<PathBuf, TaskError> {
        match self {
            CBuilderRef::Standalone(b) => b.base.generate_source(Some(db), defines, exe_name),
            CBuilderRef::Library(b) => b.base.generate_source(Some(db), defines, exe_name),
        }
    }

    /// Upstream `cbuilder.compile(**kwds)` at `driver.py:535`.
    pub fn compile(&self, exe_name: Option<String>) -> Result<(), TaskError> {
        match self {
            CBuilderRef::Standalone(b) => b.compile(exe_name),
            CBuilderRef::Library(b) => b.compile(exe_name),
        }
    }

    /// Upstream `cbuilder.executable_name` (CStandaloneBuilder-only) /
    /// `cbuilder.get_entry_point()` (CLibraryBuilder).
    pub fn executable_name(&self) -> Option<PathBuf> {
        match self {
            CBuilderRef::Standalone(b) => b.executable_name.borrow().clone(),
            CBuilderRef::Library(b) => b.so_name.borrow().clone(),
        }
    }

    /// Upstream `cbuilder.get_entry_point()` (`driver.py:541`).
    pub fn get_entry_point(&self) -> Result<PathBuf, TaskError> {
        match self {
            CBuilderRef::Standalone(b) => b.get_entry_point(),
            CBuilderRef::Library(b) => b.get_entry_point(),
        }
    }

    /// Upstream `cbuilder.shared_library_name` (`driver.py:486`).
    pub fn shared_library_name(&self) -> Option<PathBuf> {
        match self {
            CBuilderRef::Standalone(b) => b.shared_library_name.borrow().clone(),
            CBuilderRef::Library(_) => None,
        }
    }

    /// Upstream `cbuilder.executable_name_w` (`driver.py:490`).
    pub fn executable_name_w(&self) -> Option<PathBuf> {
        match self {
            CBuilderRef::Standalone(b) => b.executable_name_w.borrow().clone(),
            CBuilderRef::Library(_) => None,
        }
    }

    /// Upstream `cbuilder.targetdir` written by `generate_source`
    /// (`genc.py:187`). Read by `task_source_c` `:459`.
    pub fn targetdir(&self) -> Option<PathBuf> {
        match self {
            CBuilderRef::Standalone(b) => b.base.targetdir.borrow().clone(),
            CBuilderRef::Library(b) => b.base.targetdir.borrow().clone(),
        }
    }
}
