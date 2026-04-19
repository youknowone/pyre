//! Flow-space special-case registry for callables whose flow-space
//! meaning differs from a plain `direct_call`.
//!
//! RPython basis: `rpython/flowspace/specialcase.py`.
//!
//! ## Deviations from upstream (parity rule #1)
//!
//! * **`SPECIAL_CASES` key type.** Upstream keys the registry on the
//!   actual Python callable object (`SPECIAL_CASES[__import__] = ...`).
//!   The Rust port identifies Python callables by their enum tag
//!   (`BuiltinFunction::Import`) because we cannot store
//!   `fn(PyObject) -> PyObject` keys. Each upstream registered callable
//!   has a matching `BuiltinFunction` variant (see `model.rs`).
//!
//! * **`register_flow_sc` decorator.** Python's decorator pattern that
//!   mutates the global `SPECIAL_CASES` at import time has no direct
//!   Rust analogue. The Rust port initialises `SPECIAL_CASES` once
//!   through `LazyLock`, using a module-level builder that performs
//!   the same insertions the decorator would have.
//!
//! * **`sc_redirected_function` closure state.** Upstream captures the
//!   target function name as a closure variable; the closure is stored
//!   in `SPECIAL_CASES`. The Rust port replaces the closure with a
//!   `SpecialCaseDispatch::Redirect(BuiltinFunction)` variant so the
//!   target callable is named at compile time.
//!
//! * **`BUILTINS` string table.** Upstream resolves builtins through
//!   Python introspection (`getattr(__builtin__, varname)`). The Rust
//!   port has no Python runtime to introspect, so `BUILTINS` carries a
//!   curated `name → ConstValue` table for the subset of `__builtin__`
//!   the flow-space builder currently needs. The table grows as more
//!   of `__builtin__` becomes reachable from ported RPython code.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::flowcontext::{FlowContext, FlowContextError, FlowingError};
use crate::model::{
    BuiltinFunction, BuiltinObject, BuiltinType, ConstValue, Constant, ExceptionClass, Hlvalue,
};

/// Registered SPECIAL_CASE callable. The variants mirror upstream's
/// two handler shapes: direct `sc_*` handlers and redirected
/// `sc_redirected_function` closures over a target name.
#[derive(Clone, Debug)]
pub enum SpecialCaseDispatch {
    /// RPython `register_flow_sc(func)(sc_*)` — a handler registered
    /// directly via the decorator (`sc_import`, `sc_locals`,
    /// `sc_getattr`).
    Handler(SpecialCaseHandler),
    /// RPython `redirect_function(srcfunc, dstfuncname)` — the
    /// decorator-registered `sc_redirected_function` closure forwards
    /// the call to `dstfuncname`. The Rust port encodes the target as
    /// an enum variant instead of a dotted-path string.
    Redirect(BuiltinFunction),
}

/// RPython `sc_import` / `sc_locals` / `sc_getattr` signature, adapted
/// for Rust: receives `ctx` + a slice of flow values and returns the
/// single result `Hlvalue` or an error that unwinds through the flow
/// graph.
pub type SpecialCaseHandler = fn(&mut FlowContext, &[Hlvalue]) -> Result<Hlvalue, FlowContextError>;

/// RPython `rpython/flowspace/specialcase.py:4` — `SPECIAL_CASES = {}`
/// populated by `register_flow_sc` / `redirect_function`.
pub static SPECIAL_CASES: LazyLock<HashMap<BuiltinFunction, SpecialCaseDispatch>> =
    LazyLock::new(|| {
        let mut cases: HashMap<BuiltinFunction, SpecialCaseDispatch> = HashMap::new();
        // @register_flow_sc(__import__) / sc_import
        cases.insert(
            BuiltinFunction::Import,
            SpecialCaseDispatch::Handler(sc_import),
        );
        // @register_flow_sc(locals) / sc_locals
        cases.insert(
            BuiltinFunction::Locals,
            SpecialCaseDispatch::Handler(sc_locals),
        );
        // @register_flow_sc(getattr) / sc_getattr
        cases.insert(
            BuiltinFunction::GetAttr,
            SpecialCaseDispatch::Handler(sc_getattr),
        );
        // redirect_function(open,       'rpython.rlib.rfile.create_file')
        cases.insert(
            BuiltinFunction::Open,
            SpecialCaseDispatch::Redirect(BuiltinFunction::CreateFile),
        );
        // redirect_function(os.fdopen,  'rpython.rlib.rfile.create_fdopen_rfile')
        cases.insert(
            BuiltinFunction::OsFdopen,
            SpecialCaseDispatch::Redirect(BuiltinFunction::CreateFdopenRfile),
        );
        // redirect_function(os.tmpfile, 'rpython.rlib.rfile.create_temp_rfile')
        cases.insert(
            BuiltinFunction::OsTmpfile,
            SpecialCaseDispatch::Redirect(BuiltinFunction::CreateTempRfile),
        );
        // redirect_function(os.remove,  'os.unlink')
        cases.insert(
            BuiltinFunction::OsRemove,
            SpecialCaseDispatch::Redirect(BuiltinFunction::OsUnlink),
        );
        // redirect_function(os.path.isdir,   'rpython.rlib.rpath.risdir')
        cases.insert(
            BuiltinFunction::OsPathIsdir,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Risdir),
        );
        // redirect_function(os.path.isabs,   'rpython.rlib.rpath.risabs')
        cases.insert(
            BuiltinFunction::OsPathIsabs,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Risabs),
        );
        // redirect_function(os.path.normpath,'rpython.rlib.rpath.rnormpath')
        cases.insert(
            BuiltinFunction::OsPathNormpath,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Rnormpath),
        );
        // redirect_function(os.path.abspath, 'rpython.rlib.rpath.rabspath')
        cases.insert(
            BuiltinFunction::OsPathAbspath,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Rabspath),
        );
        // redirect_function(os.path.join,    'rpython.rlib.rpath.rjoin')
        cases.insert(
            BuiltinFunction::OsPathJoin,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Rjoin),
        );
        // if hasattr(os.path, 'splitdrive'):
        //     redirect_function(os.path.splitdrive, 'rpython.rlib.rpath.rsplitdrive')
        cases.insert(
            BuiltinFunction::OsPathSplitdrive,
            SpecialCaseDispatch::Redirect(BuiltinFunction::Rsplitdrive),
        );
        cases
    });

/// Look up a SPECIAL_CASES dispatch entry for a flow-space callable
/// Constant. Returns `None` if `w_callable` is not a recognised
/// `BuiltinFunction` or has no registered handler.
pub fn lookup_special_case(w_callable: &Hlvalue) -> Option<SpecialCaseDispatch> {
    let func = match w_callable {
        Hlvalue::Constant(Constant {
            value: ConstValue::Builtin(BuiltinObject::Function(func)),
            ..
        }) => func,
        _ => return None,
    };
    SPECIAL_CASES.get(func).cloned()
}

/// RPython `@register_flow_sc(__import__) def sc_import(ctx, *args_w)`
/// (`specialcase.py:27-31`).
///
/// Upstream asserts each arg is a Constant, unwraps the values, and
/// delegates to `ctx.import_name(*args)`. The Rust port mirrors that
/// contract.
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
///
/// Upstream raises `Exception("...locals() is not RPython...")`. The
/// Rust port surfaces the identical message as a `FlowingError`.
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
/// Upstream's `sc_getattr(ctx, w_obj, w_index)` → `op.getattr(w_obj,
/// w_index).eval(ctx)`. With a `w_default` argument it falls through
/// to `ctx.appcall(getattr, w_obj, w_index, w_default)`. The `op.*`
/// constructor graph is tracked for F3.3 (operation.py port); until
/// it lands the Rust handler records the abstract `getattr` / 3-arg
/// `direct_call` operation in the same shape upstream's
/// `record_maybe_raise_op` uses.
fn sc_getattr(ctx: &mut FlowContext, args_w: &[Hlvalue]) -> Result<Hlvalue, FlowContextError> {
    match args_w {
        [w_obj, w_index] => {
            // upstream: op.getattr(w_obj, w_index).eval(ctx)
            ctx.record_maybe_raise_op(
                "getattr",
                vec![w_obj.clone(), w_index.clone()],
                FlowContext::common_exception_cases(),
            )
        }
        [w_obj, w_index, w_default] => {
            // upstream: ctx.appcall(getattr, w_obj, w_index, w_default)
            let w_getattr = Hlvalue::Constant(Constant::new(ConstValue::Builtin(
                BuiltinObject::Function(BuiltinFunction::GetAttr),
            )));
            ctx.record_maybe_raise_op(
                "direct_call",
                vec![w_getattr, w_obj.clone(), w_index.clone(), w_default.clone()],
                FlowContext::common_exception_cases(),
            )
        }
        _ => Err(FlowContextError::Flowing(FlowingError::new(
            "getattr: wrong number of arguments",
        ))),
    }
}

/// Convenience constructor for `ConstValue::Builtin(Function(…))` —
/// mirrors upstream's module-level `Constant(func)` usage after
/// `register_flow_sc`.
pub fn builtin_function(function: BuiltinFunction) -> ConstValue {
    ConstValue::Builtin(BuiltinObject::Function(function))
}

/// `__builtin__` name table used by `flowcontext.py:845-854`
/// `find_global` fallback. Keyed by the identifier a Python source file
/// would write (`print`, `getattr`, `open`, `os.path.join`, …). See
/// module-top deviation note for why this table is curated rather than
/// Python-introspected.
pub static BUILTINS: LazyLock<HashMap<String, ConstValue>> = LazyLock::new(|| {
    let mut builtins = HashMap::new();
    let mut insert_fn = |name: &str, func: BuiltinFunction| {
        builtins.insert(
            name.to_owned(),
            ConstValue::Builtin(BuiltinObject::Function(func)),
        );
    };
    // Functions registered in SPECIAL_CASES plus common `__builtin__`
    // callables flowspace routinely resolves through `find_global`.
    insert_fn("print", BuiltinFunction::Print);
    insert_fn("getattr", BuiltinFunction::GetAttr);
    insert_fn("setattr", BuiltinFunction::Setattr);
    insert_fn("delattr", BuiltinFunction::Delattr);
    insert_fn("__import__", BuiltinFunction::Import);
    insert_fn("locals", BuiltinFunction::Locals);
    insert_fn("all", BuiltinFunction::All);
    insert_fn("any", BuiltinFunction::Any);
    insert_fn("open", BuiltinFunction::Open);
    insert_fn("len", BuiltinFunction::Len);
    insert_fn("iter", BuiltinFunction::Iter);
    insert_fn("next", BuiltinFunction::Next);
    insert_fn("isinstance", BuiltinFunction::Isinstance);
    insert_fn("issubclass", BuiltinFunction::Issubclass);
    insert_fn("hasattr", BuiltinFunction::Hasattr);
    insert_fn("callable", BuiltinFunction::Callable);
    insert_fn("id", BuiltinFunction::Id);
    insert_fn("hash", BuiltinFunction::Hash);
    insert_fn("repr", BuiltinFunction::Repr);
    insert_fn("min", BuiltinFunction::Min);
    insert_fn("max", BuiltinFunction::Max);
    insert_fn("abs", BuiltinFunction::Abs);
    insert_fn("sum", BuiltinFunction::Sum);
    insert_fn("round", BuiltinFunction::Round);
    insert_fn("divmod", BuiltinFunction::Divmod);
    insert_fn("pow", BuiltinFunction::Pow);
    insert_fn("chr", BuiltinFunction::Chr);
    insert_fn("ord", BuiltinFunction::Ord);
    insert_fn("hex", BuiltinFunction::Hex);
    insert_fn("oct", BuiltinFunction::Oct);
    insert_fn("bin", BuiltinFunction::Bin);
    insert_fn("format", BuiltinFunction::Format);
    insert_fn("vars", BuiltinFunction::Vars);
    insert_fn("dir", BuiltinFunction::Dir);
    insert_fn("compile", BuiltinFunction::Compile);
    insert_fn("input", BuiltinFunction::Input);
    insert_fn("exec", BuiltinFunction::Exec);
    insert_fn("eval", BuiltinFunction::Eval);
    insert_fn("super", BuiltinFunction::Super);
    // SPECIAL_CASES sources that can appear as globals via `os.remove`
    // / `os.path.*` dotted-path resolution are not reachable from
    // `find_global(varname)` alone (they require LOAD_ATTR); upstream
    // resolves them through `getattr(os, 'remove')` etc. The Rust
    // port's LOAD_ATTR path constructs the same `BuiltinFunction`
    // constants on demand.

    let mut insert_type = |name: &str, ty: BuiltinType| {
        builtins.insert(
            name.to_owned(),
            ConstValue::Builtin(BuiltinObject::Type(ty)),
        );
    };
    // Container / primitive types callable as constructors.
    insert_type("tuple", BuiltinType::Tuple);
    insert_type("list", BuiltinType::List);
    insert_type("set", BuiltinType::Set);
    insert_type("frozenset", BuiltinType::Frozenset);
    insert_type("dict", BuiltinType::Dict);
    insert_type("str", BuiltinType::Str);
    insert_type("int", BuiltinType::Int);
    insert_type("float", BuiltinType::Float);
    insert_type("bool", BuiltinType::Bool);
    insert_type("bytes", BuiltinType::Bytes);
    insert_type("bytearray", BuiltinType::Bytearray);
    insert_type("complex", BuiltinType::Complex);
    insert_type("memoryview", BuiltinType::Memoryview);
    insert_type("range", BuiltinType::Range);
    insert_type("slice", BuiltinType::Slice);
    insert_type("object", BuiltinType::Object);
    insert_type("type", BuiltinType::Type);
    // Iterator / adapter types.
    insert_type("enumerate", BuiltinType::Enumerate);
    insert_type("zip", BuiltinType::Zip);
    insert_type("map", BuiltinType::Map);
    insert_type("filter", BuiltinType::Filter);
    insert_type("reversed", BuiltinType::Reversed);
    // Descriptor types.
    insert_type("property", BuiltinType::Property);
    insert_type("classmethod", BuiltinType::Classmethod);
    insert_type("staticmethod", BuiltinType::Staticmethod);

    for name in [
        "AssertionError",
        "BaseException",
        "Exception",
        "ImportError",
        "NotImplementedError",
        "RuntimeError",
        "StackOverflow",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "_StackOverflow",
    ] {
        builtins.insert(
            name.to_owned(),
            ConstValue::ExceptionClass(ExceptionClass::builtin(name)),
        );
    }
    builtins
});

/// `getattr(__builtin__, varname)` — returns the builtin by name.
/// Called by `flowcontext.py:851` after a `find_global` miss.
pub fn lookup_builtin(name: &str) -> Option<ConstValue> {
    BUILTINS.get(name).cloned()
}
