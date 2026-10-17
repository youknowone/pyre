//! PEP 810 lazy imports (`lazyimportobject.c`, `ceval.c` `_PyEval_LazyImportName`,
//! `import.c` `_PyImport_LazyImportModuleLevelObject`).

use pyre_object::*;
use rustpython_compiler::core::{PositionEncoding, SourceFileBuilder};
use rustpython_compiler_core::bytecode::{CodeObject, CodeUnit, Instruction};

use crate::compile::codegen::error::{CodegenError, CodegenErrorType};
use crate::compile::{CompileError, Mode, ast, parser};
use crate::pyframe::PyFrame;
use crate::typedef::make_new_descr;

const LAZY_NORMAL: i32 = 0;
const LAZY_ALL: i32 = 1;

fn current_frame() -> *mut PyFrame {
    crate::eval::CURRENT_FRAME.with(|current| current.get())
}

fn str_value(obj: PyObjectRef) -> Option<&'static str> {
    if obj.is_null() || unsafe { is_none(obj) } {
        None
    } else {
        unsafe { w_str_get_value_opt(obj) }
    }
}

/// `_PyLazyImport_New`
pub fn lazy_import_new(
    lz_builtins: PyObjectRef,
    lz_from: PyObjectRef,
    lz_attr: PyObjectRef,
    lz_code: PyObjectRef,
    lz_instr_offset: i32,
) -> Result<PyObjectRef, crate::PyError> {
    if lz_from.is_null() || unsafe { !(is_str(lz_from) || is_lazy_import(lz_from)) } {
        return Err(crate::PyError::type_error(
            "expected str or lazy_import for name",
        ));
    }
    let lz_attr = if lz_attr.is_null() || unsafe { is_none(lz_attr) } {
        PY_NULL
    } else if unsafe { is_str(lz_attr) || is_tuple(lz_attr) } {
        lz_attr
    } else {
        return Err(crate::PyError::type_error(
            "lazy_import: fromlist must be None, a string, or a tuple",
        ));
    };
    Ok(w_lazy_import_new(
        lz_builtins,
        lz_from,
        lz_attr,
        lz_code,
        lz_instr_offset,
    ))
}

fn frame_code_and_offset() -> (PyObjectRef, i32) {
    let frame = current_frame();
    if frame.is_null() {
        return (PY_NULL, -1);
    }
    let pycode = unsafe { (*frame).pycode as PyObjectRef };
    (pycode, -1)
}

/// `_PyEval_LazyImportFrom`
pub fn lazy_import_from(v: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    let lz = unsafe { &*w_lazy_import_cast(v) };
    if !lz.lz_attr.is_null()
        && unsafe { is_tuple(lz.lz_attr) && w_tuple_len(lz.lz_attr) > 0 }
        && let Some(attr) = lazy_import_get_loaded_attr(lz.lz_from, name)?
    {
        return Ok(attr);
    }
    let w_name = w_str_new(name);
    let (lz_code, offset) = frame_code_and_offset();
    lazy_import_new(PY_NULL, v, w_name, lz_code, offset)
}

fn lazy_import_get_loaded_attr(
    name: PyObjectRef,
    attr_name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let Some(name_s) = str_value(name) else {
        return Ok(None);
    };
    let Some(module) = crate::importing::sys_modules_entry(name_s) else {
        return Ok(None);
    };
    if unsafe { !is_module(module) } {
        return Ok(None);
    }
    let dict = unsafe { w_module_get_w_dict(module) };
    if dict.is_null() {
        return Ok(None);
    }
    match crate::baseobjspace::finditem_str(dict, attr_name) {
        Ok(Some(attr)) if unsafe { is_lazy_import(attr) } => Ok(None),
        Ok(other) => Ok(other),
        Err(_) => Ok(None),
    }
}

fn lazy_import_path(m: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let lz = unsafe { &*w_lazy_import_cast(m) };
    if unsafe { is_lazy_import(lz.lz_from) } {
        let base = lazy_import_path(lz.lz_from)?;
        let attr = str_value(lz.lz_attr).unwrap_or("");
        let base_s = str_value(base).unwrap_or("");
        return Ok(w_str_new(&format!("{base_s}.{attr}")));
    }
    if !lz.lz_attr.is_null() && unsafe { !is_tuple(lz.lz_attr) || w_tuple_len(lz.lz_attr) > 0 } {
        return Ok(lz.lz_from);
    }
    let from = str_value(lz.lz_from).unwrap_or("");
    match from.find('.') {
        Some(dot) => Ok(w_str_new(&from[..dot])),
        None => Ok(lz.lz_from),
    }
}

fn lazy_import_name(m: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let lz = unsafe { &*w_lazy_import_cast(m) };
    if unsafe { is_lazy_import(lz.lz_from) } {
        return lazy_import_path(m);
    }
    if !lz.lz_attr.is_null() && unsafe { !is_tuple(lz.lz_attr) || w_tuple_len(lz.lz_attr) > 0 } {
        let from = str_value(lz.lz_from).unwrap_or("");
        return Ok(w_str_new(&format!("{from}...")));
    }
    Ok(lz.lz_from)
}

fn lazy_import_repr_impl(op: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let name = lazy_import_name(op)?;
    let name_s = str_value(name).unwrap_or("");
    Ok(w_str_new(&format!("<lazy_import '{name_s}'>")))
}

fn is_resolving(op: PyObjectRef) -> bool {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return false;
    }
    unsafe {
        (*ec)
            .lazy_imports_resolving
            .borrow()
            .iter()
            .any(|&p| std::ptr::eq(p, op))
    }
}

fn resolving_push(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if !ec.is_null() {
        unsafe { (*ec).lazy_imports_resolving.borrow_mut().push(op) };
    }
}

fn resolving_pop(op: PyObjectRef) {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return;
    }
    let mut stack = unsafe { (*ec).lazy_imports_resolving.borrow_mut() };
    if let Some(i) = stack.iter().rposition(|&p| std::ptr::eq(p, op)) {
        stack.remove(i);
    }
}

fn import_cycle_error(name: PyObjectRef, root_from: PyObjectRef) -> crate::PyError {
    let shown = str_value(name).unwrap_or("");
    let msg = format!("cannot import name '{shown}' (most likely due to a circular import)");
    let w_type = crate::builtins::lookup_exc_class("ImportCycleError")
        .unwrap_or_else(|| crate::typedef::gettypeobject(&pyre_object::EXC_IMPORT_ERROR_TYPE));
    let _ = root_from;
    crate::PyError::from_type_and_value(w_type, w_str_new(&msg))
}

fn root_and_first(lz: PyObjectRef) -> (PyObjectRef, PyObjectRef) {
    let mut root = lz;
    let mut first = PY_NULL;
    loop {
        let from = unsafe { (*w_lazy_import_cast(root)).lz_from };
        if unsafe { !is_lazy_import(from) } {
            break;
        }
        first = root;
        root = from;
    }
    (root, first)
}

fn replay_from(mod_: PyObjectRef, lz: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let from = unsafe { (*w_lazy_import_cast(lz)).lz_from };
    if unsafe { !is_lazy_import(from) } {
        return Ok(mod_);
    }
    let parent = replay_from(mod_, from)?;
    let attr = unsafe { (*w_lazy_import_cast(lz)).lz_attr };
    let name = str_value(attr).unwrap_or("");
    let obj = crate::importing::import_from(parent, name)?;
    resolve_result(obj)
}

fn resolve_result(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || unsafe { !is_lazy_import(obj) } {
        Ok(obj)
    } else {
        load_lazy_import(obj)
    }
}

/// `_PyImport_LoadLazyImportTstate` / `lazy_import_resolve_impl`
#[majit_macros::dont_look_inside]
pub fn load_lazy_import(lazy_import: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    load_lazy_import_impl(lazy_import).map(|(value, _imported)| value)
}

fn load_lazy_import_impl(
    lazy_import: PyObjectRef,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let (root, first) = root_and_first(lazy_import);
    if is_resolving(lazy_import) {
        let name = lazy_import_name(lazy_import)?;
        let root_from = unsafe { (*w_lazy_import_cast(root)).lz_from };
        return Err(import_cycle_error(name, root_from));
    }
    resolving_push(lazy_import);
    let result = (|| {
        let root_lz = unsafe { &*w_lazy_import_cast(root) };
        let fromlist = if !first.is_null()
            && !root_lz.lz_attr.is_null()
            && unsafe { is_tuple(root_lz.lz_attr) && w_tuple_len(root_lz.lz_attr) > 0 }
        {
            let attr = unsafe { (*w_lazy_import_cast(first)).lz_attr };
            w_tuple_new(vec![attr])
        } else if root_lz.lz_attr.is_null() {
            w_none()
        } else {
            root_lz.lz_attr
        };
        let frame = current_frame();
        let globals = if frame.is_null() {
            w_none()
        } else {
            unsafe { (*frame).get_w_globals() }
        };
        let builtins = root_lz.lz_builtins;
        let import_func = if builtins.is_null() {
            None
        } else {
            crate::baseobjspace::finditem_str(builtins, "__import__")?
        };
        let import_func = import_func.ok_or_else(|| {
            crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found")
        })?;
        let name = root_lz.lz_from;
        let level = w_int_new(0);
        let obj = crate::call::call_function_impl_result(
            import_func,
            &[name, globals, globals, fromlist, level],
        )?;
        let obj = resolve_result(obj)?;
        let obj = if !first.is_null() {
            replay_from(obj, lazy_import)?
        } else {
            obj
        };
        if let Ok(path) = lazy_import_path(lazy_import) {
            discard_lazy_module(path);
        }
        let imported = if unsafe { is_module(obj) } {
            obj
        } else {
            PY_NULL
        };
        Ok((obj, imported))
    })();
    resolving_pop(lazy_import);
    result
}

fn discard_lazy_module(name: PyObjectRef) {
    let set = crate::module::sys::state::lazy_modules();
    if !set.is_null() && unsafe { is_set_or_frozenset(set) } {
        unsafe { w_set_discard(set, name) };
    }
}

fn replace_item_if(
    namespace: PyObjectRef,
    name: PyObjectRef,
    placeholder: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if namespace.is_null() {
        return Ok(());
    }
    match crate::baseobjspace::finditem(namespace, name) {
        Ok(Some(current)) if std::ptr::eq(current, placeholder) => {
            crate::baseobjspace::setitem(namespace, name, value)?;
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(()),
        Err(e) => Err(e),
    }
}

/// `_PyLazyImport_Reify`
#[majit_macros::dont_look_inside]
pub fn reify(
    placeholder: PyObjectRef,
    name: PyObjectRef,
    namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let (value, _imported) = load_lazy_import_impl(placeholder)?;
    replace_item_if(namespace, name, placeholder, value)?;
    Ok(value)
}

/// Interpreter-visible lazy check used by LOAD_GLOBAL / LOAD_NAME.
#[inline]
pub fn maybe_reify(
    value: PyObjectRef,
    name: PyObjectRef,
    namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { is_lazy_import(value) } {
        reify(value, name, namespace)
    } else {
        Ok(value)
    }
}

#[inline]
pub fn maybe_reify_str(
    value: PyObjectRef,
    name: &str,
    namespace: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { is_lazy_import(value) } {
        reify(value, w_str_new(name), namespace)
    } else {
        Ok(value)
    }
}

/// gh-144957: give module `__getattr__` a chance before resolving.
pub fn module_getattro_lazy(
    module: PyObjectRef,
    name: &str,
    placeholder: PyObjectRef,
    dict: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if name != "__getattr__"
        && let Some(getattr) = crate::baseobjspace::finditem_str(dict, "__getattr__")?
    {
        let getattr = maybe_reify_str(getattr, "__getattr__", dict)?;
        match crate::call::call_function_impl_result(getattr, &[w_str_new(name)]) {
            Ok(result) => return Ok(result),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
            Err(e) => return Err(e),
        }
    }
    if is_resolving(placeholder) {
        return Err(crate::PyError::attribute_error(format!(
            "module has no attribute '{name}'"
        )));
    }
    reify(placeholder, w_str_new(name), dict)
}

/// `_PyImport_TryLoadLazySubmodule` (simplified).
pub fn try_load_lazy_submodule(
    module: PyObjectRef,
    attr_name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { !is_module(module) } {
        return Ok(None);
    }
    let dict = unsafe { w_module_get_w_dict(module) };
    if dict.is_null() {
        return Ok(None);
    }
    let Some(mod_name) = crate::baseobjspace::finditem_str(dict, "__name__")? else {
        return Ok(None);
    };
    let Some(mod_name_s) = str_value(mod_name) else {
        return Ok(None);
    };
    let pending = crate::module::sys::state::lazy_pending_submodules();
    if pending.is_null() {
        return Ok(None);
    }
    let Some(pending_set) = crate::baseobjspace::finditem_str(pending, mod_name_s)? else {
        return Ok(None);
    };
    let Some(_source) = crate::baseobjspace::finditem_str(pending_set, attr_name)? else {
        return Ok(None);
    };
    let full_name = format!("{mod_name_s}.{attr_name}");
    if let Some(loaded) = crate::importing::sys_modules_entry(&full_name)
        && unsafe { !is_none(loaded) }
    {
        let _ = crate::baseobjspace::setitem(dict, w_str_new(attr_name), loaded);
        let _ = crate::baseobjspace::delitem(pending_set, w_str_new(attr_name));
        return Ok(Some(loaded));
    }
    Ok(None)
}

fn is_module_level_frame() -> bool {
    let frame = current_frame();
    if frame.is_null() {
        return false;
    }
    let globals = unsafe { (*frame).get_w_globals() };
    let locals = unsafe { (*frame).get_w_locals() };
    !globals.is_null() && !locals.is_null() && std::ptr::eq(globals, locals)
}

fn check_lazy_import_compatibility(
    globals: PyObjectRef,
    name: PyObjectRef,
    level: i32,
) -> Result<bool, crate::PyError> {
    if globals.is_null() || unsafe { is_none(globals) } {
        return Ok(false);
    }
    let Some(lazy_modules) = crate::baseobjspace::finditem_str(globals, "__lazy_modules__")? else {
        return Ok(false);
    };
    let abs = import_abs_name(name, globals, level)?;
    crate::baseobjspace::contains(lazy_modules, abs)
}

fn import_abs_name(
    name: PyObjectRef,
    globals: PyObjectRef,
    level: i32,
) -> Result<PyObjectRef, crate::PyError> {
    if level <= 0 {
        return Ok(name);
    }
    if globals.is_null() || unsafe { is_none(globals) } {
        return Err(crate::PyError::key_error("'__name__' not in globals"));
    }
    let package = match crate::baseobjspace::finditem_str(globals, "__package__")? {
        Some(p) if !unsafe { is_none(p) } => p,
        _ => match crate::baseobjspace::finditem_str(globals, "__name__")? {
            Some(n) => n,
            None => return Err(crate::PyError::key_error("'__name__' not in globals")),
        },
    };
    let Some(pkg) = str_value(package) else {
        return Err(crate::PyError::type_error("package must be a string"));
    };
    let mut end = pkg.len();
    for _ in 1..level {
        match pkg[..end].rfind('.') {
            Some(dot) => end = dot,
            None => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    "attempted relative import beyond top-level package",
                ));
            }
        }
    }
    let base = &pkg[..end];
    let name_s = str_value(name).unwrap_or("");
    if name_s.is_empty() {
        Ok(w_str_new(base))
    } else {
        Ok(w_str_new(&format!("{base}.{name_s}")))
    }
}

fn lazy_modules_add(name: PyObjectRef) -> Result<(), crate::PyError> {
    if let Some(name_s) = str_value(name)
        && let Some(existing) = crate::importing::sys_modules_entry(name_s)
        && unsafe { is_module(existing) }
    {
        return Ok(());
    }
    let set = crate::module::sys::state::lazy_modules();
    if !set.is_null() {
        unsafe { w_set_add(set, name) };
    }
    Ok(())
}

fn register_lazy_on_parent(name: PyObjectRef, source: PyObjectRef) -> Result<(), crate::PyError> {
    let Some(name_s) = str_value(name) else {
        return Ok(());
    };
    let pending = crate::module::sys::state::ensure_lazy_pending_submodules();
    if pending.is_null() {
        return Ok(());
    }
    let mut end = name_s.len();
    while let Some(dot) = name_s[..end].rfind('.') {
        let parent = &name_s[..dot];
        let child = &name_s[dot + 1..end];
        let submodules = match crate::baseobjspace::finditem_str(pending, parent)? {
            Some(d) => d,
            None => {
                let d = w_dict_new();
                crate::baseobjspace::setitem(pending, w_str_new(parent), d)?;
                d
            }
        };
        crate::baseobjspace::setitem(submodules, w_str_new(child), source)?;
        end = dot;
    }
    Ok(())
}

/// `_PyImport_LazyImportModuleLevelObject`
pub fn lazy_import_module_level(
    name: PyObjectRef,
    builtins: PyObjectRef,
    globals: PyObjectRef,
    _locals: PyObjectRef,
    fromlist: PyObjectRef,
    level: i32,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { !is_str(name) } {
        return Err(crate::PyError::type_error(format!(
            "module name must be a string, got {}",
            crate::type_methods::arg_type_name(name)
        )));
    }
    if level < 0 {
        return Err(crate::PyError::value_error("level must be >= 0"));
    }
    if !is_module_level_frame() {
        return Err(crate::PyError::syntax_error(
            "'lazy import' is only allowed at module level",
        ));
    }
    let abs_name = import_abs_name(name, globals, level)?;
    let fromlist = if fromlist.is_null() || unsafe { is_none(fromlist) } {
        w_none()
    } else if unsafe { is_str(fromlist) } {
        w_tuple_new(vec![fromlist])
    } else {
        fromlist
    };
    if let Some(filter) = crate::module::sys::state::lazy_imports_filter() {
        let modname =
            crate::baseobjspace::finditem_str(globals, "__name__")?.unwrap_or_else(w_none);
        let decision =
            crate::call::call_function_impl_result(filter, &[modname, abs_name, fromlist])?;
        if !crate::baseobjspace::is_true(decision)? {
            return crate::importing::dunder_import_name_obj(
                name,
                globals,
                _locals,
                fromlist,
                level as i64,
            );
        }
    }
    if unsafe { is_tuple(fromlist) } {
        let n = unsafe { w_tuple_len(fromlist) };
        for i in 0..n {
            let item = unsafe { w_tuple_getitem(fromlist, i as i64) }.unwrap_or(PY_NULL);
            if unsafe { !is_str(item) } {
                return Err(crate::PyError::type_error(format!(
                    "Item in ``from list'' must be str, not {}",
                    crate::type_methods::arg_type_name(item)
                )));
            }
        }
    }
    let (lz_code, offset) = frame_code_and_offset();
    let res = lazy_import_new(builtins, abs_name, fromlist, lz_code, offset)?;
    lazy_modules_add(abs_name)?;
    if unsafe { is_tuple(fromlist) && w_tuple_len(fromlist) > 0 } {
        let n = unsafe { w_tuple_len(fromlist) };
        for i in 0..n {
            let item = unsafe { w_tuple_getitem(fromlist, i as i64) }.unwrap_or(PY_NULL);
            let fromname = {
                let abs = str_value(abs_name).unwrap_or("");
                let from = str_value(item).unwrap_or("");
                w_str_new(&format!("{abs}.{from}"))
            };
            lazy_modules_add(fromname)?;
            register_lazy_on_parent(fromname, w_none())?;
        }
    } else {
        register_lazy_on_parent(abs_name, res)?;
    }
    Ok(res)
}

fn level_from_obj(level: PyObjectRef) -> Result<i32, crate::PyError> {
    if level.is_null() || unsafe { is_none(level) } {
        return Ok(0);
    }
    crate::builtins::space_index_w(level).map(|v| v as i32)
}

fn builtins_mapping_from_frame(frame: &PyFrame) -> PyObjectRef {
    let w_builtin = frame.get_builtin();
    if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        let dict = unsafe { w_module_get_w_dict(w_builtin) };
        if !dict.is_null() {
            return dict;
        }
    }
    w_builtin
}

/// `_PyEval_LazyImportName`
pub fn eval_lazy_import_name(
    frame: &mut PyFrame,
    name: PyObjectRef,
    fromlist: PyObjectRef,
    level_obj: PyObjectRef,
    explicit_lazy: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let mut lazy = explicit_lazy;
    let mode = crate::module::sys::state::lazy_imports_mode();
    if mode == LAZY_ALL && !lazy {
        lazy = is_module_level_frame();
    } else if mode == LAZY_NORMAL {
        // keep the IMPORT_NAME explicit-lazy bit
    }
    if !lazy && is_module_level_frame() {
        let globals = frame.get_w_globals();
        let level = level_from_obj(level_obj)?;
        lazy = check_lazy_import_compatibility(globals, name, level)?;
    }
    if !lazy {
        return crate::importing::import_name(frame, name, fromlist, level_obj);
    }
    let builtins = builtins_mapping_from_frame(frame);
    let lazy_import_func = if builtins.is_null() {
        None
    } else {
        crate::baseobjspace::finditem_str(builtins, "__lazy_import__")?
    };
    let lazy_import_func = lazy_import_func.ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "__lazy_import__ not found")
    })?;
    if is_default_lazy_import(lazy_import_func) {
        let globals = frame.get_w_globals();
        let locals = crate::importing::import_locals(frame);
        let level = level_from_obj(level_obj)?;
        return lazy_import_module_level(name, builtins, globals, locals, fromlist, level);
    }
    let globals = frame.get_w_globals();
    let locals = crate::importing::import_locals(frame);
    crate::call::call_function_impl_result(
        lazy_import_func,
        &[name, globals, locals, fromlist, level_obj],
    )
}

static DEFAULT_LAZY_IMPORT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_default_lazy_import(w_func: PyObjectRef) {
    if DEFAULT_LAZY_IMPORT
        .compare_exchange(
            0,
            w_func as usize,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
    {
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    }
}

fn is_default_lazy_import(func: PyObjectRef) -> bool {
    let stored = DEFAULT_LAZY_IMPORT.load(std::sync::atomic::Ordering::Acquire) as PyObjectRef;
    !stored.is_null() && std::ptr::eq(stored, func)
}

/// `builtin___lazy_import___impl`
pub fn builtin_lazy_import(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let scope = crate::builtins::bind_builtin_kwargs(
        args,
        &["name", "globals", "locals", "fromlist", "level"],
        &[true, false, false, false, false],
        "__lazy_import__",
    )?;
    let name = scope[0];
    let mut globals = scope[1];
    let mut locals = scope[2];
    let fromlist = if scope[3].is_null() {
        w_tuple_new(Vec::new())
    } else {
        scope[3]
    };
    let level = if scope[4].is_null() {
        0
    } else {
        level_from_obj(scope[4])?
    };
    if globals.is_null() {
        let frame = current_frame();
        if frame.is_null() {
            return Err(crate::PyError::type_error(
                "__lazy_import__() missing globals when called without a frame",
            ));
        }
        globals = unsafe { (*frame).get_w_globals() };
    }
    if locals.is_null() {
        locals = globals;
    }
    if unsafe { !is_dict(globals) } {
        return Err(crate::PyError::type_error(format!(
            "expect dict for globals, got {}",
            crate::type_methods::arg_type_name(globals)
        )));
    }
    let mut builtins = crate::baseobjspace::finditem_str(globals, "__builtins__")?
        .ok_or_else(|| crate::PyError::value_error("unable to get builtins for lazy import"))?;
    if unsafe { is_module(builtins) } {
        let dict = unsafe { w_module_get_w_dict(builtins) };
        if dict.is_null() {
            return Err(crate::PyError::attribute_error(
                "builtins module has no dict",
            ));
        }
        builtins = dict;
    }
    lazy_import_module_level(name, builtins, globals, locals, fromlist, level)
}

fn lazy_import_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor '__repr__' requires a 'lazy_import' object",
        ));
    }
    lazy_import_repr_impl(self_)
}

fn lazy_import_descr_resolve(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor 'resolve' requires a 'lazy_import' object",
        ));
    }
    load_lazy_import(self_)
}

fn lazy_import_descr_getattro(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(PY_NULL);
    let name = args.get(1).copied().unwrap_or(PY_NULL);
    if self_.is_null() || unsafe { !is_lazy_import(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor '__getattribute__' requires a 'lazy_import' object",
        ));
    }
    let name_s = str_value(name).unwrap_or("");
    match crate::baseobjspace::object_getattribute(self_, name_s) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let lz_name = lazy_import_name(self_)?;
            let name_r = crate::builtins::builtin_repr(&[name])?;
            let lz_r = crate::builtins::builtin_repr(&[lz_name])?;
            let name_s = str_value(name_r).unwrap_or("?");
            let lz_s = str_value(lz_r).unwrap_or("?");
            Err(crate::PyError::attribute_error(format!(
                "cannot access attribute {name_s} on unresolved lazy import {lz_s}"
            )))
        }
        Err(e) => Err(e),
    }
}

pub(crate) fn init_lazy_import_type(ns: PyObjectRef) {
    let _root_scope = gc_roots::push_roots();
    let ns_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(ns);
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__new__",
            make_new_descr(|_| {
                Err(crate::PyError::type_error(
                    "cannot create 'lazy_import' instances",
                ))
            })
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::gateway::make_slot_wrapper_with_arity_and_doc(
                "__repr__",
                lazy_import_descr_repr,
                1,
                "Return repr(self).",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "resolve",
            crate::gateway::make_method_descriptor_with_arity_and_doc(
                "resolve",
                lazy_import_descr_resolve,
                1,
                "Resolve the lazy import and return the imported object.",
            )
        )
    };
    unsafe {
        crate::__pyre_put_new!(
            ns_slot,
            "__getattribute__",
            crate::gateway::make_slot_wrapper_with_arity(
                "__getattribute__",
                lazy_import_descr_getattro,
                2,
            )
        )
    };
}

pub fn apply_lazy_imports_mode_after_site() {
    if crate::importing::lazy_imports_config() == 1 {
        crate::module::sys::state::set_lazy_imports_mode(LAZY_ALL);
    }
}

// ── compile-wrapper: syntax checks + IMPORT_NAME bit0 ──────────────

#[derive(Clone, Copy)]
struct LazyCheckCtx {
    in_function: bool,
    in_class: bool,
    in_try: bool,
}

fn lazy_syntax_error(
    source: &str,
    filename: &str,
    range: ruff_text_size::TextRange,
    message: String,
) -> CompileError {
    let sf = SourceFileBuilder::new(filename, source).finish();
    let loc = sf
        .to_source_code()
        .source_location(range.start(), PositionEncoding::Utf8);
    let end = sf
        .to_source_code()
        .source_location(range.end(), PositionEncoding::Utf8);
    CompileError::Codegen(CodegenError {
        location: Some(loc),
        end_location: Some(end),
        error: CodegenErrorType::SyntaxError(message),
        source_path: filename.to_owned(),
    })
}

fn validate_lazy_stmts(
    stmts: &[ast::Stmt],
    ctx: LazyCheckCtx,
    source: &str,
    filename: &str,
) -> Result<(), CompileError> {
    for stmt in stmts {
        match stmt {
            ast::Stmt::Import(node) if node.is_lazy => {
                check_lazy_context(ctx, source, filename, node.range, "import")?;
            }
            ast::Stmt::ImportFrom(node) if node.is_lazy => {
                let module = node.module.as_ref().map(|s| s.as_str()).unwrap_or("");
                if node.level == 0 && module == "__future__" {
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        node.range,
                        "lazy from __future__ import is not allowed".to_string(),
                    ));
                }
                check_lazy_context(ctx, source, filename, node.range, "from ... import")?;
                if node.names.first().is_some_and(|n| n.name.as_str() == "*") {
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        node.range,
                        "lazy from ... import * is not allowed".to_string(),
                    ));
                }
            }
            ast::Stmt::FunctionDef(node) => {
                validate_lazy_stmts(
                    &node.body,
                    LazyCheckCtx {
                        in_function: true,
                        in_class: false,
                        in_try: false,
                    },
                    source,
                    filename,
                )?;
            }
            ast::Stmt::ClassDef(node) => {
                validate_lazy_stmts(
                    &node.body,
                    LazyCheckCtx {
                        in_function: false,
                        in_class: true,
                        in_try: false,
                    },
                    source,
                    filename,
                )?;
            }
            ast::Stmt::Try(node) => {
                let try_ctx = LazyCheckCtx {
                    in_try: true,
                    ..ctx
                };
                validate_lazy_stmts(&node.body, try_ctx, source, filename)?;
                for handler in &node.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    validate_lazy_stmts(&h.body, try_ctx, source, filename)?;
                }
                validate_lazy_stmts(&node.orelse, try_ctx, source, filename)?;
                validate_lazy_stmts(&node.finalbody, try_ctx, source, filename)?;
            }
            ast::Stmt::If(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                for clause in &node.elif_else_clauses {
                    validate_lazy_stmts(&clause.body, ctx, source, filename)?;
                }
            }
            ast::Stmt::For(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                validate_lazy_stmts(&node.orelse, ctx, source, filename)?;
            }
            ast::Stmt::While(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
                validate_lazy_stmts(&node.orelse, ctx, source, filename)?;
            }
            ast::Stmt::With(node) => {
                validate_lazy_stmts(&node.body, ctx, source, filename)?;
            }
            ast::Stmt::Match(node) => {
                for case in &node.cases {
                    validate_lazy_stmts(&case.body, ctx, source, filename)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_lazy_context(
    ctx: LazyCheckCtx,
    source: &str,
    filename: &str,
    range: ruff_text_size::TextRange,
    import_type: &str,
) -> Result<(), CompileError> {
    let message = if ctx.in_try {
        format!("lazy {import_type} not allowed inside try/except blocks")
    } else if ctx.in_function {
        format!("lazy {import_type} not allowed inside functions")
    } else if ctx.in_class {
        format!("lazy {import_type} not allowed inside classes")
    } else {
        return Ok(());
    };
    Err(lazy_syntax_error(source, filename, range, message))
}

fn collect_import_name_lazy(stmts: &[ast::Stmt], out: &mut Vec<bool>) {
    for stmt in stmts {
        match stmt {
            ast::Stmt::Import(node) => {
                for _ in &node.names {
                    out.push(node.is_lazy);
                }
            }
            ast::Stmt::ImportFrom(node) => out.push(node.is_lazy),
            ast::Stmt::FunctionDef(_) | ast::Stmt::ClassDef(_) => {}
            ast::Stmt::Try(node) => {
                collect_import_name_lazy(&node.body, out);
                for handler in &node.handlers {
                    let ast::ExceptHandler::ExceptHandler(h) = handler;
                    collect_import_name_lazy(&h.body, out);
                }
                collect_import_name_lazy(&node.orelse, out);
                collect_import_name_lazy(&node.finalbody, out);
            }
            ast::Stmt::If(node) => {
                collect_import_name_lazy(&node.body, out);
                for clause in &node.elif_else_clauses {
                    collect_import_name_lazy(&clause.body, out);
                }
            }
            ast::Stmt::For(node) => {
                collect_import_name_lazy(&node.body, out);
                collect_import_name_lazy(&node.orelse, out);
            }
            ast::Stmt::While(node) => {
                collect_import_name_lazy(&node.body, out);
                collect_import_name_lazy(&node.orelse, out);
            }
            ast::Stmt::With(node) => collect_import_name_lazy(&node.body, out),
            ast::Stmt::Match(node) => {
                for case in &node.cases {
                    collect_import_name_lazy(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

pub fn patch_import_name_bit0(code: &mut CodeObject, flags: &[bool]) {
    let mut flag_i = 0usize;
    let mut i = 0usize;
    let len = code.instructions.len();
    let ptr = code.instructions.units_ptr() as *mut CodeUnit;
    while i < len {
        let unit = code.instructions[i];
        if matches!(unit.op, Instruction::ImportName { .. }) {
            if flag_i < flags.len() && flags[flag_i] {
                let arg = u8::from(unit.arg);
                if arg & 2 == 0 {
                    unsafe {
                        (*ptr.add(i)).arg = (arg | 1).into();
                    }
                }
            }
            flag_i += 1;
        }
        i += 1 + unit.op.cache_entries();
    }
}

fn misplaced_from_lazy_import(source: &str, filename: &str) -> Result<(), CompileError> {
    // Parser/parser.c: `from ... lazy import` is rejected; ruff has no equivalent.
    let bytes = source.as_bytes();
    let mut i = 0usize;
    let mut at_stmt = true;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            at_stmt = true;
            continue;
        }
        if c == b'\'' || c == b'"' {
            i = skip_string(bytes, i);
            at_stmt = false;
            continue;
        }
        if c == b'\n' || c == b';' {
            at_stmt = true;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if at_stmt && rest_is_ident(bytes, i, b"from") {
            let start = i;
            i += 4;
            i = skip_ws_and_comments(bytes, i);
            // module / dots
            while i < bytes.len() {
                let ch = bytes[i];
                if ch == b'.' || ch.is_ascii_alphanumeric() || ch == b'_' {
                    i += 1;
                    continue;
                }
                if ch.is_ascii_whitespace() {
                    i += 1;
                    continue;
                }
                if ch == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    i += 2;
                    continue;
                }
                break;
            }
            i = skip_ws_and_comments(bytes, i);
            if rest_is_ident(bytes, i, b"lazy") {
                let after = skip_ws_and_comments(bytes, i + 4);
                if rest_is_ident(bytes, after, b"import") {
                    let range = ruff_text_size::TextRange::new(
                        ruff_text_size::TextSize::from(start as u32),
                        ruff_text_size::TextSize::from((after + 6).min(bytes.len()) as u32),
                    );
                    return Err(lazy_syntax_error(
                        source,
                        filename,
                        range,
                        "use 'lazy from ... ' instead of 'from ... lazy import'".to_string(),
                    ));
                }
            }
            at_stmt = false;
            continue;
        }
        at_stmt = false;
        i += 1;
    }
    Ok(())
}

fn rest_is_ident(bytes: &[u8], i: usize, word: &[u8]) -> bool {
    if i + word.len() > bytes.len() {
        return false;
    }
    if !bytes[i..].starts_with(word) {
        return false;
    }
    let next = i + word.len();
    next == bytes.len() || !(bytes[next].is_ascii_alphanumeric() || bytes[next] == b'_')
}

fn skip_ws_and_comments(bytes: &[u8], mut i: usize) -> usize {
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'\n' {
            i += 2;
            continue;
        }
        break;
    }
    i
}

fn skip_string(bytes: &[u8], i: usize) -> usize {
    let quote = bytes[i];
    let triple = i + 2 < bytes.len() && bytes[i + 1] == quote && bytes[i + 2] == quote;
    let mut j = if triple { i + 3 } else { i + 1 };
    while j < bytes.len() {
        if bytes[j] == b'\\' {
            j += 2;
            continue;
        }
        if bytes[j] == quote {
            if triple {
                if j + 2 < bytes.len() && bytes[j + 1] == quote && bytes[j + 2] == quote {
                    return j + 3;
                }
            } else {
                return j + 1;
            }
        }
        j += 1;
    }
    bytes.len()
}

pub fn collect_lazy_import_flags(
    source: &str,
    filename: &str,
    mode: Mode,
) -> Result<Vec<bool>, CompileError> {
    if !matches!(mode, Mode::Exec | Mode::Single | Mode::BlockExpr) {
        return Ok(Vec::new());
    }
    misplaced_from_lazy_import(source, filename)?;
    let parsed = parser::parse_unchecked(source, parser::Mode::Module.into());
    flags_from_mod(parsed.syntax(), source, filename)
}

pub fn flags_from_mod(
    module: &ast::Mod,
    source: &str,
    filename: &str,
) -> Result<Vec<bool>, CompileError> {
    let body: &[ast::Stmt] = match module {
        ast::Mod::Module(m) => &m.body,
        ast::Mod::Expression(_) => return Ok(Vec::new()),
    };
    validate_lazy_stmts(
        body,
        LazyCheckCtx {
            in_function: false,
            in_class: false,
            in_try: false,
        },
        source,
        filename,
    )?;
    let mut flags = Vec::new();
    collect_import_name_lazy(body, &mut flags);
    Ok(flags)
}

pub fn apply_lazy_import_compile(
    source: &str,
    filename: &str,
    mode: Mode,
    code: &mut CodeObject,
) -> Result<(), CompileError> {
    let flags = collect_lazy_import_flags(source, filename, mode)?;
    patch_import_name_bit0(code, &flags);
    Ok(())
}

pub fn apply_lazy_import_mod(
    module: &ast::Mod,
    source: &str,
    filename: &str,
    code: &mut CodeObject,
) -> Result<(), CompileError> {
    let flags = flags_from_mod(module, source, filename)?;
    patch_import_name_bit0(code, &flags);
    Ok(())
}
