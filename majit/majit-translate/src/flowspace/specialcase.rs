//! Flow-space special-case registry for callables whose flow-space
//! meaning differs from a plain `direct_call`.
//!
//! RPython basis: `rpython/flowspace/specialcase.py`.
//!
//! ## Deviations from upstream (parity rule #1)
//!
//! * **`SPECIAL_CASES` key type.** Upstream keys the registry on the
//!   actual Python callable object (`SPECIAL_CASES[__import__] = ...`)
//!   whose identity is Python object `is`. The Rust port keys on
//!   `HostObject` (Arc-backed) and matches upstream's `is`-semantic
//!   through `Arc::ptr_eq`. Because `HOST_ENV` materialises every
//!   registered callable as a singleton at bootstrap, lookups against
//!   `w_callable.value` hit the same Arc the registry was seeded with.
//!
//! * **`register_flow_sc` decorator.** Python's decorator pattern that
//!   mutates the global `SPECIAL_CASES` at import time has no direct
//!   Rust analogue. `SPECIAL_CASES` is populated once through
//!   `LazyLock`, performing the same insertions the decorator would
//!   have.
//!
//! * **`sc_redirected_function` closure state.** Upstream captures the
//!   target function name as a closure variable; the closure is stored
//!   in `SPECIAL_CASES`. The Rust port encodes the target as a
//!   `SpecialCaseDispatch::Redirect(HostObject)` variant so the target
//!   callable is resolved (via `HOST_ENV.import_module(...).module_get(...)`)
//!   at registration time rather than each call.

use std::collections::HashMap;
use std::sync::LazyLock;

use super::flowcontext::{FlowContext, FlowContextError, FlowingError};
use super::model::{ConstValue, Constant, HOST_ENV, Hlvalue, HostObject};
use super::operation::OpKind;

/// Registered SPECIAL_CASE callable. The variants mirror upstream's
/// two handler shapes: direct `sc_*` handlers and redirected
/// `sc_redirected_function` closures over a target callable.
#[derive(Clone)]
pub enum SpecialCaseDispatch {
    /// RPython `register_flow_sc(func)(sc_*)` — a handler registered
    /// directly via the decorator (`sc_import`, `sc_locals`,
    /// `sc_getattr`).
    Handler(SpecialCaseHandler),
    /// RPython `redirect_function(srcfunc, dstfuncname)` — the
    /// decorator-registered `sc_redirected_function` closure forwards
    /// the call to the target HostObject (resolved at registration).
    Redirect(HostObject),
}

impl std::fmt::Debug for SpecialCaseDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecialCaseDispatch::Handler(_) => f.debug_tuple("Handler").field(&"<fn>").finish(),
            SpecialCaseDispatch::Redirect(target) => {
                f.debug_tuple("Redirect").field(&target.qualname()).finish()
            }
        }
    }
}

/// RPython `sc_import` / `sc_locals` / `sc_getattr` signature, adapted
/// for Rust: receives `ctx` + a slice of flow values and returns the
/// single result `Hlvalue` or an error that unwinds through the flow
/// graph.
pub type SpecialCaseHandler = fn(&mut FlowContext, &[Hlvalue]) -> Result<Hlvalue, FlowContextError>;

fn module_attr(module: &str, attr: &str) -> HostObject {
    HOST_ENV
        .import_module(module)
        .and_then(|m| m.module_get(attr))
        .unwrap_or_else(|| panic!("HOST_ENV missing {module}.{attr}"))
}

fn builtin(name: &str) -> HostObject {
    HOST_ENV
        .lookup_builtin(name)
        .unwrap_or_else(|| panic!("HOST_ENV missing builtin {name}"))
}

/// RPython `rpython/flowspace/specialcase.py:4` — `SPECIAL_CASES = {}`
/// populated by `register_flow_sc` / `redirect_function`.
pub static SPECIAL_CASES: LazyLock<HashMap<HostObject, SpecialCaseDispatch>> =
    LazyLock::new(|| {
        let mut cases: HashMap<HostObject, SpecialCaseDispatch> = HashMap::new();
        // @register_flow_sc(__import__) / sc_import
        cases.insert(
            builtin("__import__"),
            SpecialCaseDispatch::Handler(sc_import),
        );
        // @register_flow_sc(locals) / sc_locals
        cases.insert(builtin("locals"), SpecialCaseDispatch::Handler(sc_locals));
        // @register_flow_sc(getattr) / sc_getattr
        cases.insert(builtin("getattr"), SpecialCaseDispatch::Handler(sc_getattr));
        // redirect_function(open,       'rpython.rlib.rfile.create_file')
        cases.insert(
            builtin("open"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rfile", "create_file")),
        );
        // redirect_function(os.fdopen,  'rpython.rlib.rfile.create_fdopen_rfile')
        cases.insert(
            module_attr("os", "fdopen"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rfile", "create_fdopen_rfile")),
        );
        // redirect_function(os.tmpfile, 'rpython.rlib.rfile.create_temp_rfile')
        cases.insert(
            module_attr("os", "tmpfile"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rfile", "create_temp_rfile")),
        );
        // redirect_function(os.remove,  'os.unlink')
        cases.insert(
            module_attr("os", "remove"),
            SpecialCaseDispatch::Redirect(module_attr("os", "unlink")),
        );
        // redirect_function(os.path.isdir,   'rpython.rlib.rpath.risdir')
        cases.insert(
            module_attr("os.path", "isdir"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "risdir")),
        );
        // redirect_function(os.path.isabs,   'rpython.rlib.rpath.risabs')
        cases.insert(
            module_attr("os.path", "isabs"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "risabs")),
        );
        // redirect_function(os.path.normpath,'rpython.rlib.rpath.rnormpath')
        cases.insert(
            module_attr("os.path", "normpath"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "rnormpath")),
        );
        // redirect_function(os.path.abspath, 'rpython.rlib.rpath.rabspath')
        cases.insert(
            module_attr("os.path", "abspath"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "rabspath")),
        );
        // redirect_function(os.path.join,    'rpython.rlib.rpath.rjoin')
        cases.insert(
            module_attr("os.path", "join"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "rjoin")),
        );
        // if hasattr(os.path, 'splitdrive'):
        //     redirect_function(os.path.splitdrive, 'rpython.rlib.rpath.rsplitdrive')
        cases.insert(
            module_attr("os.path", "splitdrive"),
            SpecialCaseDispatch::Redirect(module_attr("rpython.rlib.rpath", "rsplitdrive")),
        );
        cases
    });

/// Look up a SPECIAL_CASES dispatch entry for a flow-space callable
/// Constant. Returns `None` if `w_callable` is not a HostObject or
/// has no registered handler.
pub fn lookup_special_case(w_callable: &Hlvalue) -> Option<SpecialCaseDispatch> {
    let obj = match w_callable {
        Hlvalue::Constant(Constant {
            value: ConstValue::HostObject(obj),
            ..
        }) => obj,
        _ => return None,
    };
    SPECIAL_CASES.get(obj).cloned()
}

/// RPython `@register_flow_sc(__import__) def sc_import(ctx, *args_w)`
/// (`specialcase.py:27-31`).
///
/// Upstream asserts each arg is a Constant, unwraps the values, and
/// delegates to `ctx.import_name(*args)`.
fn sc_import(ctx: &mut FlowContext, args_w: &[Hlvalue]) -> Result<Hlvalue, FlowContextError> {
    let mut args = Vec::with_capacity(args_w.len());
    for arg in args_w {
        match arg {
            Hlvalue::Constant(Constant { value, .. }) => args.push(value.clone()),
            _ => {
                return Err(FlowContextError::Flowing(FlowingError::new(
                    "__import__() arguments must be constant",
                )));
            }
        }
    }
    ctx.import_name(&args)
}

/// RPython `@register_flow_sc(locals) def sc_locals(_, *args)`
/// (`specialcase.py:33-41`).
fn sc_locals(_ctx: &mut FlowContext, _args_w: &[Hlvalue]) -> Result<Hlvalue, FlowContextError> {
    Err(FlowContextError::Flowing(FlowingError::new(
        "A function calling locals() is not RPython.  \
         Note that if you're translating code outside the PyPy \
         repository, a likely cause is that py.test's --assert=rewrite \
         mode is getting in the way.  You should copy the file \
         pytest.ini from the root of the PyPy repository into your \
         own project.",
    )))
}

/// RPython `@register_flow_sc(getattr) def sc_getattr(ctx, w_obj,
/// w_index, w_default=None)` (`specialcase.py:43-49`).
///
/// With `w_default` absent → `op.getattr(w_obj, w_index).eval(ctx)`;
/// present → `ctx.appcall(getattr, w_obj, w_index, w_default)`.
fn sc_getattr(ctx: &mut FlowContext, args_w: &[Hlvalue]) -> Result<Hlvalue, FlowContextError> {
    match args_w {
        [w_obj, w_index] => ctx.eval_hlop(OpKind::GetAttr, vec![w_obj.clone(), w_index.clone()]),
        [w_obj, w_index, w_default] => {
            let callee = HOST_ENV.lookup_builtin("getattr").unwrap();
            ctx.appcall(
                callee,
                vec![w_obj.clone(), w_index.clone(), w_default.clone()],
            )
        }
        _ => Err(FlowContextError::Flowing(FlowingError::new(
            "getattr: wrong number of arguments",
        ))),
    }
}

/// `getattr(__builtin__, varname)` — returns the builtin HostObject by
/// name, wrapped in a `ConstValue::HostObject`. `flowcontext.py:851`
/// fallback after `find_global` miss.
pub fn lookup_builtin(name: &str) -> Option<ConstValue> {
    HOST_ENV.lookup_builtin(name).map(ConstValue::HostObject)
}
