//! Module importing — PyPy equivalent: pypy/module/imp/importing.py
//!
//! Implements the import machinery:
//! - `importhook()` — main entry point (called by IMPORT_NAME opcode)
//! - `find_module()` — locate a .py file on sys.path
//! - `load_source_module()` — compile and execute a .py file
//! - `check_sys_modules()` — consult the module cache
//! - `import_all_from()` — IMPORT_STAR handler

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pyre_bytecode::{CodeObject, Mode, compile_source_with_filename};
use pyre_object::*;
use pyre_runtime::{PyExecutionContext, PyNamespace, namespace_load, namespace_store};

use crate::eval::eval_frame_plain;
use crate::frame::PyFrame;

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: space.sys.get('modules') — a dict mapping module names
// to module objects. We use a thread-local HashMap<String, PyObjectRef>.

thread_local! {
    static SYS_MODULES: RefCell<HashMap<String, PyObjectRef>> = RefCell::new(HashMap::new());
    /// sys.path equivalent — list of directories to search for modules.
    static SYS_PATH: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
    /// Builtin modules registry — PyPy equivalent: space.builtin_modules
    ///
    /// Maps module name → initializer function that populates a PyNamespace.
    /// Each builtin module is lazily created on first import.
    static BUILTIN_MODULES: RefCell<HashMap<&'static str, fn(&mut PyNamespace)>> =
        RefCell::new(HashMap::new());
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs

/// Register a builtin module initializer.
///
/// PyPy equivalent: Module.install() → space.builtin_modules[name] = mod
pub fn register_builtin_module(name: &'static str, init: fn(&mut PyNamespace)) {
    BUILTIN_MODULES.with(|m| {
        m.borrow_mut().insert(name, init);
    });
}

/// Install all standard builtin modules.
///
/// PyPy equivalent: baseobjspace.py `make_builtins()` +
/// `install_mixedmodule()` for each module in objspace.usemodules.
pub fn install_builtin_modules() {
    register_builtin_module("math", crate::module::math::moduledef::init);
    register_builtin_module("time", crate::module::time::moduledef::init);
    register_builtin_module("sys", crate::module::sys::moduledef::init);
}

/// Try to load a builtin module by name.
///
/// PyPy equivalent: find_module() → C_BUILTIN path →
/// getbuiltinmodule() → Module.__init__ + startup()
fn load_builtin_module(name: &str) -> Option<PyObjectRef> {
    let init_fn = BUILTIN_MODULES.with(|m| m.borrow().get(name).copied())?;

    let mut namespace = Box::new(PyNamespace::new());
    namespace.fix_ptr();

    // Set __name__ (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(name);
    namespace_store(&mut namespace, "__name__", name_obj);

    // Run module-specific initializer (PyPy: interpleveldefs)
    init_fn(&mut namespace);

    let ns_ptr = Box::into_raw(namespace);
    let module = w_module_new(name, ns_ptr as *mut u8);
    Some(module)
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
pub fn init_sys_path(script_dir: &Path) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        path.clear();
        // Script directory first (PyPy: first entry in sys.path)
        path.push(script_dir.to_path_buf());
        // Current working directory as fallback
        if let Ok(cwd) = std::env::current_dir() {
            if cwd != script_dir {
                path.push(cwd);
            }
        }
    });
}

/// Add a directory to sys.path.
pub fn add_sys_path(dir: &Path) {
    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        let pb = dir.to_path_buf();
        if !path.contains(&pb) {
            path.push(pb);
        }
    });
}

// ── check_sys_modules ────────────────────────────────────────────────
// PyPy equivalent: importing.py `check_sys_modules(space, w_modulename)`

fn check_sys_modules(name: &str) -> Option<PyObjectRef> {
    SYS_MODULES.with(|m| m.borrow().get(name).copied())
}

fn set_sys_module(name: &str, module: PyObjectRef) {
    SYS_MODULES.with(|m| {
        m.borrow_mut().insert(name.to_string(), module);
    });
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    Package { dirpath: PathBuf },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

fn find_module(partname: &str) -> Option<FindInfo> {
    // Check builtin modules first (PyPy: space.builtin_modules check in find_module)
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }

    SYS_PATH.with(|p| {
        let path = p.borrow();
        for dir in path.iter() {
            // Check for package: <dir>/<partname>/__init__.py
            let pkg_dir = dir.join(partname);
            let init_file = pkg_dir.join("__init__.py");
            if init_file.is_file() {
                return Some(FindInfo::Package { dirpath: pkg_dir });
            }

            // Check for source file: <dir>/<partname>.py
            let source_file = dir.join(format!("{partname}.py"));
            if source_file.is_file() {
                return Some(FindInfo::SourceFile {
                    pathname: source_file,
                });
            }
        }
        None
    })
}

// ── parse_source_module ──────────────────────────────────────────────
// PyPy equivalent: importing.py `parse_source_module(space, pathname, source)`

fn parse_source_module(pathname: &str, source: &str) -> Result<CodeObject, String> {
    compile_source_with_filename(source, Mode::Exec, pathname)
}

// ── exec_code_module ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `exec_code_module(space, w_mod, code_w, ...)`
//
// Execute a code object in the module's namespace dict.

fn exec_code_module(
    code: CodeObject,
    namespace: *mut PyNamespace,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, pyre_runtime::PyError> {
    let code_ptr = Box::into_raw(Box::new(code));
    let mut frame = PyFrame::new_with_namespace(code_ptr, execution_context, namespace);
    eval_frame_plain(&mut frame)
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

fn load_source_module(
    modulename: &str,
    pathname: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, pyre_runtime::PyError> {
    let source = std::fs::read_to_string(pathname).map_err(|e| pyre_runtime::PyError {
        kind: pyre_runtime::PyErrorKind::ImportError,
        message: format!("cannot read '{}': {e}", pathname.display()),
    })?;

    let pathname_str = pathname.to_string_lossy();
    let code = parse_source_module(&pathname_str, &source).map_err(|e| pyre_runtime::PyError {
        kind: pyre_runtime::PyErrorKind::ImportError,
        message: format!("cannot compile '{}': {e}", pathname.display()),
    })?;

    // Create a fresh namespace for the module, seeded with builtins.
    // PyPy equivalent: Module.__init__ creates w_dict = space.newdict()
    // then exec_code_module sets __builtins__ and runs code in w_dict.
    let ctx = unsafe { &*execution_context };
    let mut namespace = Box::new(ctx.fresh_namespace());
    namespace.fix_ptr();

    // Set __name__ in the module namespace (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(modulename);
    pyre_runtime::namespace_store(&mut namespace, "__name__", name_obj);

    // Set __file__ (PyPy: _prepare_module sets __file__)
    let file_obj = pyre_object::w_str_new(&pathname_str);
    pyre_runtime::namespace_store(&mut namespace, "__file__", file_obj);

    let ns_ptr = Box::into_raw(namespace);
    exec_code_module(code, ns_ptr, execution_context)?;

    // Create the module object (PyPy: Module(space, w_name) with w_dict)
    let module = w_module_new(modulename, ns_ptr as *mut u8);
    Ok(module)
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

fn load_package(
    modulename: &str,
    dirpath: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, pyre_runtime::PyError> {
    let init_path = dirpath.join("__init__.py");
    let module = load_source_module(modulename, &init_path, execution_context)?;

    // Set __path__ on the module namespace (PyPy: space.setattr(w_mod, '__path__', ...))
    let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
    let path_str = pyre_object::w_str_new(&dirpath.to_string_lossy());
    let path_list = pyre_object::w_list_new(vec![path_str]);
    unsafe {
        pyre_runtime::namespace_store(&mut *ns_ptr, "__path__", path_list);
    }

    // Add package directory to sys.path for sub-imports
    add_sys_path(dirpath);

    Ok(module)
}

// ── load_part ────────────────────────────────────────────────────────
// PyPy equivalent: importing.py `load_part()`

fn load_part(
    modulename: &str,
    partname: &str,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, pyre_runtime::PyError> {
    // Check sys.modules cache first
    if let Some(cached) = check_sys_modules(modulename) {
        return Ok(Some(cached));
    }

    // Find the module on disk
    let find_info = find_module(partname);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        FindInfo::SourceFile { pathname } => {
            load_source_module(modulename, &pathname, execution_context)?
        }
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        FindInfo::Builtin => {
            // PyPy: getbuiltinmodule() path
            load_builtin_module(partname).ok_or_else(|| pyre_runtime::PyError {
                kind: pyre_runtime::PyErrorKind::ImportError,
                message: format!("builtin module '{modulename}' failed to initialize"),
            })?
        }
    };

    // Store in sys.modules cache (PyPy: space.sys.setmodule / sys.modules[name] = mod)
    set_sys_module(modulename, module);
    Ok(Some(module))
}

// ── _absolute_import ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `_absolute_import()`

fn absolute_import(
    modulename: &str,
    w_fromlist: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, pyre_runtime::PyError> {
    let parts: Vec<&str> = modulename.split('.').collect();
    let mut first: Option<PyObjectRef> = None;
    let mut prefix = Vec::new();

    for (level, &part) in parts.iter().enumerate() {
        prefix.push(part);
        let full_name = prefix.join(".");
        let w_mod = load_part(&full_name, part, execution_context)?;
        let Some(module) = w_mod else {
            return Err(pyre_runtime::PyError {
                kind: pyre_runtime::PyErrorKind::ImportError,
                message: format!("No module named '{modulename}'"),
            });
        };
        if level == 0 {
            first = Some(module);
        }
    }

    // PyPy: if w_fromlist is not None, return the leaf module.
    // Otherwise, return the first (top-level) module.
    if !w_fromlist.is_null() && !unsafe { is_none(w_fromlist) } {
        // `from X.Y import Z` → return the leaf module (Y)
        if let Some(cached) = check_sys_modules(modulename) {
            return Ok(cached);
        }
    }

    // `import X.Y` → return the top-level module (X)
    first.ok_or_else(|| pyre_runtime::PyError {
        kind: pyre_runtime::PyErrorKind::ImportError,
        message: format!("No module named '{modulename}'"),
    })
}

// ── importhook ───────────────────────────────────────────────────────
// PyPy equivalent: importing.py `importhook()`
//
// Main entry point called by the IMPORT_NAME opcode.
// Stack: [level, fromlist] → [module]

pub fn importhook(
    name: &str,
    _w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, pyre_runtime::PyError> {
    if name.is_empty() && level < 0 {
        return Err(pyre_runtime::PyError {
            kind: pyre_runtime::PyErrorKind::ValueError,
            message: "Empty module name".to_string(),
        });
    }

    // Level 0 = absolute import (the common case)
    // Level > 0 = relative import (not yet supported)
    if level > 0 {
        return Err(pyre_runtime::PyError {
            kind: pyre_runtime::PyErrorKind::ImportError,
            message: format!("relative imports not yet supported (level={level})"),
        });
    }

    absolute_import(name, w_fromlist, execution_context)
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

pub fn import_from(module: PyObjectRef, name: &str) -> Result<PyObjectRef, pyre_runtime::PyError> {
    // First try the module's namespace dict (PyPy: space.getattr → w_dict lookup)
    if unsafe { is_module(module) } {
        let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
        if !ns_ptr.is_null() {
            let ns = unsafe { &*ns_ptr };
            if let Ok(value) = namespace_load(ns, name) {
                return Ok(value);
            }
        }
    }

    // Fallback: try py_getattr (for non-module objects or attrs set via setattr)
    match pyre_objspace::space::py_getattr(module, name) {
        Ok(value) => Ok(value),
        Err(_) => Err(pyre_runtime::PyError {
            kind: pyre_runtime::PyErrorKind::ImportError,
            message: format!("cannot import name '{name}'"),
        }),
    }
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `import_all_from(module, into_locals)`
//
// Merge all public names from a module into the current namespace.
// If __all__ exists, use it; otherwise copy all names not starting with '_'.

pub fn import_all_from(module: PyObjectRef, into_namespace: *mut PyNamespace) {
    if !unsafe { is_module(module) } {
        return;
    }

    let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
    if ns_ptr.is_null() {
        return;
    }

    let src_ns = unsafe { &*ns_ptr };
    let dst_ns = unsafe { &mut *into_namespace };

    // Check for __all__ (PyPy: try module.__all__)
    let has_all = src_ns.get("__all__").is_some();

    if has_all {
        // TODO: iterate __all__ list and copy named entries.
        // For now, fall through to copying all non-underscore names.
    }

    // Copy all names not starting with '_' (PyPy: skip_leading_underscores)
    for name in src_ns.keys() {
        if name.starts_with('_') {
            continue;
        }
        if let Some(&value) = src_ns.get(name) {
            if !value.is_null() {
                namespace_store(dst_ns, name, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sys_modules_cache() {
        let sentinel = w_none();
        set_sys_module("test_cached", sentinel);
        let cached = check_sys_modules("test_cached");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), sentinel);
    }

    #[test]
    fn test_find_module_nonexistent() {
        // Should not find a module that doesn't exist
        let result = find_module("__nonexistent_pyre_test_module__");
        assert!(result.is_none());
    }
}
