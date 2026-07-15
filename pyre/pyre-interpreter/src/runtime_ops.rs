use std::slice;
use std::sync::OnceLock;

use crate::bytecode::{BinaryOperator, ComparisonOperator, ConvertValueOparg};
use pyre_object::{
    PY_NULL, PyObjectRef, W_SeqIterObject, is_instance, is_list, is_range_iter, is_seq_iter,
    is_str, is_tuple, w_dict_new, w_dict_store_checked, w_int_get_value, w_int_new, w_list_getitem,
    w_list_len, w_list_new, w_range_iter_has_next, w_range_iter_next, w_str_from_wtf8,
    w_str_get_wtf8, w_str_len, w_tuple_getitem, w_tuple_len, w_tuple_new,
};
use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::{
    PyError, PyErrorKind, builtin_code_get, function_get_code, is_builtin_code, is_function,
    w_code_get_ptr,
};

/// `pypy/interpreter/pyopcode.py:1457 MAKE_FUNCTION` stamps the new
/// function's `w_func_globals = self.w_globals` directly from the
/// running frame's dict object.  This entry point accepts the canonical
/// PyObjectRef (`w_globals`) so the freshly-created function inherits
/// the frame's exact `__globals__` identity.
pub fn make_function_from_code_obj_with_globals_obj(
    code_obj: PyObjectRef,
    w_globals: PyObjectRef,
) -> PyObjectRef {
    let code_ptr = unsafe { w_code_get_ptr(code_obj) };
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    // `function.py:51-53 __init__`:
    //   self.name = forcename or code.co_name
    //   self.qualname = qualname or self.name
    // so `name` is the bare `co_name` and `qualname` is the dotted
    // `co_qualname`.  `pyopcode.py:1457 MAKE_FUNCTION` then stamps the
    // qualified name from `codeobj.co_qualname`, which is why a later
    // `__code__ = new_code` assignment does NOT change `__qualname__`.
    let func = crate::function::function_new_with_closure(
        code_obj as *const (),
        code.obj_name.to_string(),
        w_globals,
        pyre_object::PY_NULL,
    );
    let qualname_obj = pyre_object::w_str_new(code.qualname.as_ref());
    unsafe { crate::function::function_set_qualname(func, qualname_obj) };
    func
}

fn decode_name(name_ptr: i64, name_len: i64) -> Option<&'static str> {
    if name_ptr == 0 || name_len < 0 {
        return None;
    }
    let bytes = unsafe { slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    std::str::from_utf8(bytes).ok()
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_make_function_from_globals(globals: i64, code_obj: i64) -> i64 {
    // `globals` is the globals OBJECT (the JIT threads the vable
    // `w_globals` slot).  Capture it directly; the raw `*mut DictStorage`
    // is recovered from the object wherever a frame still needs it.
    let w_globals = globals as PyObjectRef;
    make_function_from_code_obj_with_globals_obj(code_obj as PyObjectRef, w_globals) as i64
}

/// SET_FUNCTION_ATTRIBUTE residual: stamp one attribute on `func` per the
/// compile-time `flag` discriminant, then return `func` so the caller pushes
/// the same function back onto the stack (net stack effect -1).  `flag` is the
/// `MakeFunctionFlag` bit-position discriminant (`oparg.rs`
/// `MakeFunctionFlag`): `Defaults = 0`, `KwOnlyDefaults = 1`,
/// `Annotations = 2`, `Closure = 3`, `Annotate = 4`, `TypeParams = 5`.  The
/// match must stay in sync with `eval.rs`
/// `set_function_attribute_with_flag` and the codewriter's
/// `SetFunctionAttribute` arm.  Sets a typed field on the function; runs no
/// user code and never raises → `Plain`.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_set_function_attribute(func: i64, attr: i64, flag: i64) -> i64 {
    let func = func as PyObjectRef;
    let attr = attr as PyObjectRef;
    match flag {
        0 => unsafe { crate::function_set_defaults(func, attr) },
        1 => unsafe { crate::function_set_kwdefaults(func, attr) },
        2 => {
            // `function.py:553-559 fset_func_annotations` — the eager
            // annotations dict is stamped on the typed `Function.w_ann`
            // slot via `function_write_barrier`, so `f.__annotations__ is
            // f.__annotations__` holds through the getattr arm.
            unsafe { crate::function::function_set_annotations(func, attr) };
        }
        3 => unsafe { crate::function_set_closure(func, attr) },
        4 => {
            // PEP 649: `attr` is the `__annotate__` callable the runtime
            // `__annotations__` getter evaluates lazily; stored on the
            // typed `w_annotate` slot.
            unsafe { (*(func as *mut crate::function::Function)).w_annotate = attr };
            // The direct field store bypasses `function_write_barrier`;
            // record it for the prebuilt-root minor-collection skip, exactly
            // as the interpreter does.
            pyre_object::gc_roots::mark_prebuilt_roots_dirty();
        }
        // `TypeParams = 5` carries a PEP 695 type-parameter tuple; pyre has
        // no PEP 695 surface, so the operand is accepted silently.
        _ => {}
    }
    func as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_load_name_from_namespace(
    frame_ptr: i64,
    namespace_ptr: i64,
    name_ptr: i64,
    name_len: i64,
) -> i64 {
    let w_globals = namespace_ptr as PyObjectRef;
    let Some(name) = decode_name(name_ptr, name_len) else {
        return 0;
    };
    // `pyopcode.py:959 _load_global`: `space.finditem_str(get_w_globals_storage(),
    // varname)`.  Dispatch through the dict strategy on the object
    // (`dictmultiobject.py:113-115 getitem_str`) so module dicts
    // (celldict cells) and plain dicts (exec/eval globals) are both
    // handled directly. Mirrors the
    // interpreter `load_global_value` (eval.rs).
    if !w_globals.is_null() {
        if let Some(v) =
            unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_globals, name) }
        {
            return v as i64;
        }
    }
    // Globals miss: `pyopcode.py:958-967 _load_global` falls back to
    // `self.get_builtin().getdictvalue(space, varname)`.  The frame's
    // picked builtin (`pyframe.py:115 self.builtin =
    // space.builtin.pick_builtin(w_globals)`) is the authoritative
    // builtin reference; mid-execution `__builtins__` rebind would
    // change the picked module without touching
    // `namespace["__builtins__"]`'s raw entry, so the extern must
    // route through `frame.w_builtin` rather than re-reading the dict
    // slot.
    let w_builtin = if frame_ptr != 0 {
        unsafe {
            let p = frame_ptr as *const u8;
            *(p.add(crate::pyframe::PYFRAME_W_BUILTIN_OFFSET) as *const PyObjectRef)
        }
    } else {
        std::ptr::null_mut()
    };
    if !w_globals.is_null() && unsafe { pyre_object::dictmultiobject::is_module_dict(w_globals) } {
        if let Some(v) =
            unsafe { crate::eval::load_global_via_cache_extern(w_globals, w_builtin, name) }
        {
            return v as i64;
        }
    } else if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
        // `_load_global` builtin fallback also fires on non-module-dict
        // globals (e.g. exec/eval with a plain `dict` for globals).
        let w_builtin_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if !w_builtin_dict.is_null() {
            if let Ok(Some(v)) = crate::baseobjspace::finditem_str(w_builtin_dict, name) {
                return v as i64;
            }
        }
    }
    std::ptr::null_mut::<()>() as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_store_name_to_namespace(
    namespace_ptr: i64,
    name_ptr: i64,
    name_len: i64,
    value: i64,
) -> i64 {
    let w_globals = namespace_ptr as PyObjectRef;
    let Some(name) = decode_name(name_ptr, name_len) else {
        return 0;
    };
    // `celldict.py:332 STORE_GLOBAL_cached` (jitted): `space.setitem_str(
    // get_w_globals_storage(), varname, w_newvalue)`.  Strategy dispatch on the
    // object (`dictmultiobject.py:111-112 setitem_str`) handles module
    // dicts (cells) and plain dicts (exec/eval globals) uniformly.
    if !w_globals.is_null() {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(w_globals, name, value as PyObjectRef);
        }
    }
    0
}

type JitFunctionCaller =
    extern "C" fn(frame_ptr: i64, callable: i64, args: *const i64, nargs: i64) -> i64;

static JIT_FUNCTION_CALLER: OnceLock<JitFunctionCaller> = OnceLock::new();

pub fn register_jit_function_caller(caller: JitFunctionCaller) {
    let _ = JIT_FUNCTION_CALLER.set(caller);
}

type JitExcRaiser = extern "C" fn(value: i64);

static JIT_EXC_RAISER: OnceLock<JitExcRaiser> = OnceLock::new();

pub fn register_jit_exc_raiser(raiser: JitExcRaiser) {
    let _ = JIT_EXC_RAISER.set(raiser);
}

/// llmodel.py:194-199 _store_exception: publish `exc_obj` into the
/// backend's pos_exception/pos_exc_value cells so the residual call's
/// GuardNoException sees it and side-exits into the handler. Mirrors
/// `jit_call_user_function_from_frame` (call_jit.rs:362-379). MUST NOT use
/// a side-channel TLS slot — that path is drained before the guard
/// machinery runs and would bypass try/except. The call helpers return
/// garbage on Err; resume data hands control to the except block.
#[inline]
pub(crate) fn jit_publish_exception(exc_obj: PyObjectRef) {
    if exc_obj != PY_NULL {
        if let Some(raiser) = JIT_EXC_RAISER.get() {
            raiser(exc_obj as i64);
        }
    }
}

fn call_builtin_with_args(callable: i64, args: &[i64]) -> i64 {
    let callable = callable as PyObjectRef;
    unsafe {
        let code = crate::getcode(callable);
        let func = builtin_code_get(code as PyObjectRef);
        let arg_slice = std::slice::from_raw_parts(args.as_ptr() as *const PyObjectRef, args.len());
        match func(arg_slice) {
            Ok(result) => result as i64,
            Err(e) => {
                jit_publish_exception(e.to_exc_object());
                0 // garbage — GuardNoException will fire
            }
        }
    }
}

fn jit_call_user_function_with_args(frame_ptr: i64, callable: i64, args: &[i64]) -> i64 {
    let Some(caller) = JIT_FUNCTION_CALLER.get().copied() else {
        let callable = callable as PyObjectRef;
        let code_ptr = unsafe { function_get_code(callable) };
        panic!("jit function caller bridge is not installed for code_ptr={code_ptr:p}");
    };
    caller(frame_ptr, callable, args.as_ptr(), args.len() as i64)
}

fn call_callable_with_args(frame_ptr: i64, callable: i64, args: &[i64]) -> i64 {
    let _ = frame_ptr;
    let callable_ref = callable as PyObjectRef;
    let arg_slice =
        unsafe { std::slice::from_raw_parts(args.as_ptr() as *const PyObjectRef, args.len()) };
    match crate::call::call_function_impl_result(callable_ref, arg_slice) {
        Ok(result) => result as i64,
        Err(err) => {
            jit_publish_exception(err.to_exc_object());
            0 // garbage — GuardNoException will fire
        }
    }
}

macro_rules! define_callable_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(frame_ptr: i64, callable: i64 $(, $arg: i64)*) -> i64 {
            call_callable_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_builtin_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(callable: i64 $(, $arg: i64)*) -> i64 {
            call_builtin_with_args(callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_function_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(frame_ptr: i64, callable: i64 $(, $arg: i64)*) -> i64 {
            jit_call_user_function_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_flat_ref_helper {
    ($inner:ident, $name:ident $(, $arg:ident)*) => {
        pub extern "C" fn $name($($arg: i64),*) -> i64 {
            $inner(&[$($arg),*])
        }
    };
}

define_callable_call_helper!(jit_call_callable_0);
define_callable_call_helper!(jit_call_callable_1, arg0);
define_callable_call_helper!(jit_call_callable_2, arg0, arg1);
define_callable_call_helper!(jit_call_callable_3, arg0, arg1, arg2);
define_callable_call_helper!(jit_call_callable_4, arg0, arg1, arg2, arg3);
define_callable_call_helper!(jit_call_callable_5, arg0, arg1, arg2, arg3, arg4);
define_callable_call_helper!(jit_call_callable_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_callable_call_helper!(
    jit_call_callable_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_callable_call_helper!(
    jit_call_callable_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_builtin_call_helper!(jit_call_known_builtin_0);
define_known_builtin_call_helper!(jit_call_known_builtin_1, arg0);
define_known_builtin_call_helper!(jit_call_known_builtin_2, arg0, arg1);
define_known_builtin_call_helper!(jit_call_known_builtin_3, arg0, arg1, arg2);
define_known_builtin_call_helper!(jit_call_known_builtin_4, arg0, arg1, arg2, arg3);
define_known_builtin_call_helper!(jit_call_known_builtin_5, arg0, arg1, arg2, arg3, arg4);
define_known_builtin_call_helper!(jit_call_known_builtin_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_known_builtin_call_helper!(
    jit_call_known_builtin_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_builtin_call_helper!(
    jit_call_known_builtin_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_function_call_helper!(jit_call_known_function_0);
define_known_function_call_helper!(jit_call_known_function_1, arg0);
define_known_function_call_helper!(jit_call_known_function_2, arg0, arg1);
define_known_function_call_helper!(jit_call_known_function_3, arg0, arg1, arg2);
define_known_function_call_helper!(jit_call_known_function_4, arg0, arg1, arg2, arg3);
define_known_function_call_helper!(jit_call_known_function_5, arg0, arg1, arg2, arg3, arg4);
define_known_function_call_helper!(
    jit_call_known_function_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_known_function_call_helper!(
    jit_call_known_function_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_function_call_helper!(
    jit_call_known_function_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

pub fn dispatch_callable<R, FBuiltin, FUser>(
    callable: PyObjectRef,
    on_builtin: FBuiltin,
    on_user: FUser,
) -> Result<R, PyError>
where
    FBuiltin: FnOnce(PyObjectRef) -> Result<R, PyError>,
    FUser: FnOnce(PyObjectRef) -> Result<R, PyError>,
{
    // Drain any pending JIT-prologue overflow first so a backend
    // probe that already detected an overflow surfaces here as the
    // user-visible RecursionError. The pending-exception slot is
    // populated by the backend slowpath wrapper when the JIT
    // prologue probe trips — backend raises, glue propagates
    // (rpython/rlib/rstack.py:68-73 stack_check_slowpath parity).
    crate::stack_check::drain_jit_pending_exception()?;
    // rpython/rlib/rstack.py:42 stack_check(): every interpreter call
    // boundary also checks the native stack synchronously, so deep
    // interpreter recursion (no JIT involved) raises RecursionError
    // instead of letting the OS abort on a guard-page hit.
    crate::stack_check::stack_check()?;
    unsafe {
        if is_function(callable) {
            // All callables are Function objects. Check code type to distinguish
            // builtins (BuiltinCode) from user functions (PyCode).
            let code = crate::getcode(callable);
            if is_builtin_code(code as PyObjectRef) {
                on_builtin(callable)
            } else {
                on_user(callable)
            }
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not callable",
                (*(*callable).ob_type).name
            )))
        }
    }
}

pub fn binary_op_tag(op: BinaryOperator) -> Option<i64> {
    Some(match op {
        BinaryOperator::Add => 0,
        BinaryOperator::Subtract => 1,
        BinaryOperator::Multiply => 2,
        BinaryOperator::FloorDivide => 3,
        BinaryOperator::Remainder => 4,
        BinaryOperator::TrueDivide => 5,
        BinaryOperator::Subscr => 6,
        BinaryOperator::Power => 7,
        BinaryOperator::Lshift => 8,
        BinaryOperator::Rshift => 9,
        BinaryOperator::And => 10,
        BinaryOperator::Or => 11,
        BinaryOperator::Xor => 12,
        // In-place variants get distinct tags (13-24) so the residual
        // dispatch consults the in-place special (`__iadd__` etc.) instead
        // of collapsing to the plain binary op.
        BinaryOperator::InplaceAdd => 13,
        BinaryOperator::InplaceSubtract => 14,
        BinaryOperator::InplaceMultiply => 15,
        BinaryOperator::InplaceFloorDivide => 16,
        BinaryOperator::InplaceRemainder => 17,
        BinaryOperator::InplaceTrueDivide => 18,
        BinaryOperator::InplacePower => 19,
        BinaryOperator::InplaceLshift => 20,
        BinaryOperator::InplaceRshift => 21,
        BinaryOperator::InplaceAnd => 22,
        BinaryOperator::InplaceOr => 23,
        BinaryOperator::InplaceXor => 24,
        BinaryOperator::MatrixMultiply => 25,
        BinaryOperator::InplaceMatrixMultiply => 26,
    })
}

/// Reverse of binary_op_tag: tag (0-24) → BinaryOperator.
/// The blackhole interpreter receives the compact tag from the codewriter
/// and needs to recover the original operator for binary_value dispatch.
pub fn binary_op_from_tag(tag: i64) -> Option<BinaryOperator> {
    Some(match tag {
        0 => BinaryOperator::Add,
        1 => BinaryOperator::Subtract,
        2 => BinaryOperator::Multiply,
        3 => BinaryOperator::FloorDivide,
        4 => BinaryOperator::Remainder,
        5 => BinaryOperator::TrueDivide,
        6 => BinaryOperator::Subscr,
        7 => BinaryOperator::Power,
        8 => BinaryOperator::Lshift,
        9 => BinaryOperator::Rshift,
        10 => BinaryOperator::And,
        11 => BinaryOperator::Or,
        12 => BinaryOperator::Xor,
        13 => BinaryOperator::InplaceAdd,
        14 => BinaryOperator::InplaceSubtract,
        15 => BinaryOperator::InplaceMultiply,
        16 => BinaryOperator::InplaceFloorDivide,
        17 => BinaryOperator::InplaceRemainder,
        18 => BinaryOperator::InplaceTrueDivide,
        19 => BinaryOperator::InplacePower,
        20 => BinaryOperator::InplaceLshift,
        21 => BinaryOperator::InplaceRshift,
        22 => BinaryOperator::InplaceAnd,
        23 => BinaryOperator::InplaceOr,
        24 => BinaryOperator::InplaceXor,
        25 => BinaryOperator::MatrixMultiply,
        26 => BinaryOperator::InplaceMatrixMultiply,
        _ => return None,
    })
}

/// True for the augmented-assignment (`NB_INPLACE_*`) binary-op tags (13..=24).
/// A mutable receiver (`list`/`bytearray`/`set`/`dict`/`array`) is mutated in
/// place by these and returns `self`; an immutable one (`int`/`float`/`str`/
/// `tuple`) returns a fresh object. Tags 0..=12 are the non-in-place operators.
pub fn binary_op_tag_is_inplace(tag: i64) -> bool {
    (13..=24).contains(&tag)
}

pub fn compare_op_tag(op: ComparisonOperator) -> i64 {
    match op {
        ComparisonOperator::Less => 0,
        ComparisonOperator::LessOrEqual => 1,
        ComparisonOperator::Greater => 2,
        ComparisonOperator::GreaterOrEqual => 3,
        ComparisonOperator::Equal => 4,
        ComparisonOperator::NotEqual => 5,
    }
}

/// Reverse of compare_op_tag: tag (0-5) → ComparisonOperator.
pub fn compare_op_from_tag(tag: i64) -> Option<ComparisonOperator> {
    Some(match tag {
        0 => ComparisonOperator::Less,
        1 => ComparisonOperator::LessOrEqual,
        2 => ComparisonOperator::Greater,
        3 => ComparisonOperator::GreaterOrEqual,
        4 => ComparisonOperator::Equal,
        5 => ComparisonOperator::NotEqual,
        _ => return None,
    })
}

pub fn build_list_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    w_list_new(items.to_vec())
}

pub fn build_tuple_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    w_tuple_new(items.to_vec())
}

/// BUILD_MAP evaluation, shared by the interpreter (`build_map`) and the JIT
/// residual (`bh_build_map_from_array`).  Stores each `[key, value]` pair
/// through the checked dict setitem, which hashes the key (may run user
/// `__hash__` / `__eq__`); an unhashable key raises, so — like
/// `build_set_from_refs` — this is fallible.
pub fn build_map_from_refs(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let dict = w_dict_new();
    for pair in items.chunks_exact(2) {
        let key = pair[0];
        let value = pair[1];
        unsafe {
            w_dict_store_checked(dict, key, value)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
        }
    }
    Ok(dict)
}

/// BUILD_SET evaluation, shared by the JIT residual (`bh_build_set_from_array`).
/// Builds a set from the forced element array; element hashing may run user
/// `__hash__` / `__eq__` and a non-hashable element raises `TypeError`, so —
/// unlike `build_map_from_refs` / `build_tuple_from_refs` — this is fallible.
pub fn build_set_from_refs(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::builtins::builtin_set_from_items(items)
}

/// BUILD_STRING evaluation, shared by the interpreter (`build_string`) and
/// the JIT residual (`bh_build_string_from_array`): concatenates `parts`
/// (already-stringified f-string fragments, in bottom-to-top order) into a
/// single `str`.  Each fragment is a `str` by construction (FORMAT_SIMPLE /
/// FORMAT_WITH_SPEC / CONVERT_VALUE ran first); the `bool` / `int` / `None`
/// / `<object>` arms are defensive rendering, so this never runs user code
/// and is infallible.
pub fn build_string_from_refs(parts: &[PyObjectRef]) -> PyObjectRef {
    let mut result = rustpython_wtf8::Wtf8Buf::new();
    for part in parts {
        unsafe {
            if pyre_object::is_str(*part) {
                result.push_wtf8(pyre_object::w_str_get_wtf8(*part));
            } else if pyre_object::is_bool(*part) {
                // `is_int` is true for a bool, so test `is_bool` first; a
                // bool renders "True"/"False", not its int value.
                result.push_str(if pyre_object::w_bool_get_value(*part) {
                    "True"
                } else {
                    "False"
                });
            } else if pyre_object::is_int(*part) {
                result.push_str(&pyre_object::w_int_get_value(*part).to_string());
            } else if pyre_object::is_none(*part) {
                result.push_str("None");
            } else {
                result.push_str("<object>");
            }
        }
    }
    pyre_object::w_str_from_wtf8(result)
}

/// CONVERT_VALUE conversion code, shared by the interpreter and the JIT
/// codewriter so the `convert_value` residual carries a stable integer the
/// C ABI can pass (`ConvertValueOparg` can't cross the residual boundary).
/// `0 = Str`, `1 = Repr`, `2 = Ascii`, `3 = None` (`None` behaves as `Str`).
pub fn convert_value_code(conv: ConvertValueOparg) -> i64 {
    match conv {
        ConvertValueOparg::Str => 0,
        ConvertValueOparg::Repr => 1,
        ConvertValueOparg::Ascii => 2,
        ConvertValueOparg::None => 3,
    }
}

/// CONVERT_VALUE evaluation, shared by the interpreter (`convert_value`) and
/// the JIT residual (`bh_convert_value_fn`).  `conv` is a
/// [`convert_value_code`] integer.  `Str` / `None` compute `str(value)` in
/// WTF-8 so a lone surrogate survives (the `'%s' % x` rewrite path);
/// `Repr` / `Ascii` go through `py_repr` / `py_ascii`.  A user
/// `__str__` / `__repr__` may run Python → fallible.
pub fn convert_value(value: PyObjectRef, conv: i64) -> Result<PyObjectRef, crate::PyError> {
    if conv == 0 || conv == 3 {
        let w = unsafe { crate::py_str_wtf8(value)? };
        return Ok(pyre_object::w_str_from_wtf8(w));
    }
    let s = match conv {
        1 => unsafe { crate::py_repr(value)? },
        2 => crate::builtins::py_ascii(value)?,
        _ => unsafe { crate::py_str(value)? },
    };
    Ok(pyre_object::w_str_new(&s))
}

/// FORMAT_SIMPLE / FORMAT_WITH_SPEC evaluation, shared by the interpreter
/// (`format_simple` / `format_with_spec`) and the JIT residuals
/// (`bh_format_simple_fn` / `bh_format_with_spec_fn`).  Formats `value`
/// through `format_value_dispatch` (user `__format__` may run Python →
/// fallible); a `PY_NULL` or non-`str` `spec` reads as the empty spec
/// (`str(value)`), matching `format_simple`.
pub fn format_value(value: PyObjectRef, spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let spec_str = unsafe {
        if !spec.is_null() && pyre_object::is_str(spec) {
            match pyre_object::w_str_get_wtf8(spec).as_str() {
                Ok(v) => v.to_string(),
                Err(_) => String::new(),
            }
        } else {
            String::new()
        }
    };
    let s = crate::type_methods::format_value_dispatch(value, &spec_str)?;
    Ok(pyre_object::w_str_from_wtf8(s))
}

/// STORE_SLICE evaluation (`obj[start:stop] = value`): builds a `slice`
/// object and dispatches through `setitem`, mirroring `PyObject_SetItem`
/// with a slice key. The dual of `binary_slice_values`; a `None` step.
pub fn store_slice_values(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    let slice = pyre_object::w_slice_new(start, stop, pyre_object::w_none());
    crate::baseobjspace::setitem(obj, slice, value)?;
    Ok(())
}

/// BINARY_SLICE evaluation, shared by the interpreter (`binary_slice`)
/// and the JIT residual (`bh_binary_slice_fn`): returns `obj[start:stop]`.
/// `list` / `str` / `tuple` slice on element (code-point for `str`)
/// boundaries; everything else (`bytes`, `bytearray`, instances with
/// `__getitem__`) falls back to a `slice` object dispatched through
/// `getitem`. A `None` start/stop defaults to `0` / `len`.
pub fn binary_slice_values(
    obj: PyObjectRef,
    start: PyObjectRef,
    stop: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    unsafe {
        if pyre_object::is_list(obj) {
            let len = pyre_object::w_list_len(obj) as i64;
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let e = if pyre_object::is_none(stop) {
                len
            } else {
                crate::sliceobject::eval_slice_index(stop)?
            };
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
            let mut items = Vec::new();
            for i in s..e {
                if let Some(v) = pyre_object::w_list_getitem(obj, i as i64) {
                    items.push(v);
                }
            }
            return Ok(pyre_object::w_list_new(items));
        }
        if pyre_object::is_str(obj) {
            // Slice on code-point boundaries over the WTF-8 view, so a
            // surrogate-bearing or multi-byte string slices correctly.
            let full = pyre_object::w_str_get_wtf8(obj);
            let mut offsets: Vec<usize> = full.code_point_indices().map(|(i, _)| i).collect();
            offsets.push(full.as_bytes().len());
            let len = (offsets.len() - 1) as i64;
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let e = if pyre_object::is_none(stop) {
                len
            } else {
                crate::sliceobject::eval_slice_index(stop)?
            };
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = (if e < 0 { (len + e).max(0) } else { e.min(len) } as usize).max(s);
            let part = rustpython_wtf8::Wtf8::from_bytes(&full.as_bytes()[offsets[s]..offsets[e]])
                .expect("code-point-aligned slice is WTF-8");
            return Ok(pyre_object::w_str_from_wtf8(part.to_wtf8_buf()));
        }
        if pyre_object::is_tuple(obj) {
            let len = pyre_object::w_tuple_len(obj) as i64;
            let s = if pyre_object::is_none(start) {
                0
            } else {
                crate::sliceobject::eval_slice_index(start)?
            };
            let e = if pyre_object::is_none(stop) {
                len
            } else {
                crate::sliceobject::eval_slice_index(stop)?
            };
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
            let mut items = Vec::new();
            for i in s..e {
                if let Some(v) = pyre_object::w_tuple_getitem(obj, i as i64) {
                    items.push(v);
                }
            }
            return Ok(pyre_object::w_tuple_new(items));
        }
        // Fall back to slice(start, stop) → getitem dispatch.
        // Handles bytes, bytearray, instances with __getitem__, etc.
        let slice_obj = pyre_object::sliceobject::w_slice_new(start, stop, pyre_object::w_none());
        crate::baseobjspace::getitem(obj, slice_obj)
    }
}

fn build_list_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    build_list_from_refs(&items) as i64
}

fn build_tuple_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    build_tuple_from_refs(&items) as i64
}

fn build_map_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    // Legacy fixed-arity BUILD_MAP residual reached only on the blackhole /
    // deopt path (the codewriter lowers BUILD_MAP through the array-based
    // `bh_build_map_from_array`).  An unhashable key raises; signal it through
    // `BH_LAST_EXC_VALUE` and return PY_NULL, like the other blackhole-only
    // residuals.
    match build_map_from_refs(&items) {
        Ok(dict) => dict as i64,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            PY_NULL as i64
        }
    }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_list_0() -> i64 {
    w_list_new(vec![]) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_tuple_0() -> i64 {
    w_tuple_new(vec![]) as i64
}

define_flat_ref_helper!(build_list_from_args, jit_build_list_1, arg0);
define_flat_ref_helper!(build_list_from_args, jit_build_list_2, arg0, arg1);
define_flat_ref_helper!(build_list_from_args, jit_build_list_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_1, arg0);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_2, arg0, arg1);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_map_from_args, jit_build_map_0);
define_flat_ref_helper!(build_map_from_args, jit_build_map_1, arg0, arg1);
define_flat_ref_helper!(build_map_from_args, jit_build_map_2, arg0, arg1, arg2, arg3);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_3,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_4,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

pub fn callable_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_callable_0 as *const (),
        1 => jit_call_callable_1 as *const (),
        2 => jit_call_callable_2 as *const (),
        3 => jit_call_callable_3 as *const (),
        4 => jit_call_callable_4 as *const (),
        5 => jit_call_callable_5 as *const (),
        6 => jit_call_callable_6 as *const (),
        7 => jit_call_callable_7 as *const (),
        8 => jit_call_callable_8 as *const (),
        _ => return None,
    })
}

pub fn known_builtin_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_known_builtin_0 as *const (),
        1 => jit_call_known_builtin_1 as *const (),
        2 => jit_call_known_builtin_2 as *const (),
        3 => jit_call_known_builtin_3 as *const (),
        4 => jit_call_known_builtin_4 as *const (),
        5 => jit_call_known_builtin_5 as *const (),
        6 => jit_call_known_builtin_6 as *const (),
        7 => jit_call_known_builtin_7 as *const (),
        8 => jit_call_known_builtin_8 as *const (),
        _ => return None,
    })
}

pub fn known_function_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_known_function_0 as *const (),
        1 => jit_call_known_function_1 as *const (),
        2 => jit_call_known_function_2 as *const (),
        3 => jit_call_known_function_3 as *const (),
        4 => jit_call_known_function_4 as *const (),
        5 => jit_call_known_function_5 as *const (),
        6 => jit_call_known_function_6 as *const (),
        7 => jit_call_known_function_7 as *const (),
        8 => jit_call_known_function_8 as *const (),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
pub enum FlatBuildKind {
    List,
    Tuple,
    Map,
}

pub fn list_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => jit_build_list_0 as *const (),
        1 => jit_build_list_1 as *const (),
        2 => jit_build_list_2 as *const (),
        3 => jit_build_list_3 as *const (),
        4 => jit_build_list_4 as *const (),
        5 => jit_build_list_5 as *const (),
        6 => jit_build_list_6 as *const (),
        7 => jit_build_list_7 as *const (),
        8 => jit_build_list_8 as *const (),
        _ => return None,
    })
}

pub fn tuple_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => jit_build_tuple_0 as *const (),
        1 => jit_build_tuple_1 as *const (),
        2 => jit_build_tuple_2 as *const (),
        3 => jit_build_tuple_3 as *const (),
        4 => jit_build_tuple_4 as *const (),
        5 => jit_build_tuple_5 as *const (),
        6 => jit_build_tuple_6 as *const (),
        7 => jit_build_tuple_7 as *const (),
        8 => jit_build_tuple_8 as *const (),
        _ => return None,
    })
}

pub fn map_build_helper(pair_count: usize) -> Option<*const ()> {
    Some(match pair_count {
        0 => jit_build_map_0 as *const (),
        1 => jit_build_map_1 as *const (),
        2 => jit_build_map_2 as *const (),
        3 => jit_build_map_3 as *const (),
        4 => jit_build_map_4 as *const (),
        _ => return None,
    })
}

pub fn flat_build_helper(kind: FlatBuildKind, count: usize) -> Option<*const ()> {
    match kind {
        FlatBuildKind::List => list_build_helper(count),
        FlatBuildKind::Tuple => tuple_build_helper(count),
        FlatBuildKind::Map => map_build_helper(count),
    }
}

/// Look up an entry in a proxy-free GC module/namespace dict.
pub fn module_ns_get(ns: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ns, name) }
}

/// Store into a proxy-free GC module/namespace dict (successor to
/// `dict_storage_store` for migrated namespaces). `ns` must be a
/// non-moving module dict so the by-value handle stays valid across stores.
pub fn module_ns_store(ns: PyObjectRef, name: &str, value: PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value) }
}

pub fn module_ns_store_wtf8(ns: PyObjectRef, name: &Wtf8, value: PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(ns, name, value) }
}

pub fn module_ns_delete(ns: PyObjectRef, name: &str) -> bool {
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, name) }
}

/// WTF-8 keyed delete from a proxy-free GC module/namespace dict.
pub fn module_ns_delete_wtf8(ns: PyObjectRef, name: &Wtf8) -> bool {
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_wtf8_no_proxy(ns, name) }
}

/// Run and store `f()` only when `name` is absent from a GC namespace dict.
pub fn module_ns_get_or_insert_with(ns: PyObjectRef, name: &str, f: impl FnOnce() -> PyObjectRef) {
    unsafe {
        if pyre_object::dictmultiobject::w_dict_getitem_str(ns, name).is_none() {
            let value = f();
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value);
        }
    }
}

fn type_dict_ptr(cls: PyObjectRef) -> *mut u8 {
    unsafe { pyre_object::w_type_get_dict_ptr(cls) }
}

pub fn type_dict_has_storage(cls: PyObjectRef) -> bool {
    !type_dict_ptr(cls).is_null()
}

pub fn type_dict_lookup(cls: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return None;
    }
    let value = unsafe { pyre_object::w_dict_getitem_str(dict as PyObjectRef, name)? };
    if value.is_null() { None } else { Some(value) }
}

pub fn type_dict_lookup_wtf8(cls: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return None;
    }
    let value = unsafe { pyre_object::w_dict_getitem_wtf8(dict as PyObjectRef, name)? };
    if value.is_null() { None } else { Some(value) }
}

pub fn type_dict_contains(cls: PyObjectRef, name: &str) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_getitem_str(dict as PyObjectRef, name).is_some() }
}

pub fn type_dict_store(cls: PyObjectRef, name: &str, value: PyObjectRef) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_setitem_str_no_proxy(dict as PyObjectRef, name, value) };
    true
}

pub fn type_dict_store_wtf8(cls: PyObjectRef, name: &Wtf8, value: PyObjectRef) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_setitem_wtf8_no_proxy(dict as PyObjectRef, name, value) };
    true
}

pub fn type_dict_delete(cls: PyObjectRef, name: &str) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_delitem_str_no_proxy(dict as PyObjectRef, name) }
}

pub fn type_dict_delete_wtf8(cls: PyObjectRef, name: &Wtf8) -> bool {
    let dict = type_dict_ptr(cls);
    if dict.is_null() {
        return false;
    }
    unsafe { pyre_object::w_dict_delitem_wtf8_no_proxy(dict as PyObjectRef, name) }
}

pub fn sequence_len(seq: PyObjectRef) -> Result<usize, PyError> {
    unsafe {
        if is_tuple(seq) {
            return Ok(w_tuple_len(seq));
        }
        if is_list(seq) {
            return Ok(w_list_len(seq));
        }
        if is_str(seq) {
            // Cached code point count — surrogate-safe (avoids w_str_get_value).
            return Ok(w_str_len(seq));
        }
        // Try __len__ on instances
        if is_instance(seq) {
            if let Ok(len_val) = crate::baseobjspace::len(seq) {
                return Ok(w_int_get_value(len_val) as usize);
            }
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            (*(*seq).ob_type).name
        )))
    }
}

pub fn sequence_getitem(seq: PyObjectRef, index: usize) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_tuple(seq) {
            return w_tuple_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("tuple index out of range"));
        }
        if is_list(seq) {
            return w_list_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("list index out of range"));
        }
        if is_str(seq) {
            // Walk code points through the WTF-8 view so a surrogate-bearing
            // string yields its lone surrogate instead of panicking.
            return w_str_get_wtf8(seq)
                .code_points()
                .nth(index)
                .map(|c| {
                    let mut one = Wtf8Buf::new();
                    one.push(c);
                    w_str_from_wtf8(one)
                })
                .ok_or_else(|| PyError::type_error("string index out of range"));
        }
        // Try getitem for instances
        if is_instance(seq) {
            return crate::baseobjspace::getitem(seq, w_int_new(index as i64));
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            (*(*seq).ob_type).name
        )))
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_sequence_getitem(seq: i64, index: i64) -> i64 {
    match sequence_getitem(seq as PyObjectRef, index as usize) {
        Ok(value) => value as i64,
        // Return PY_NULL on out-of-bounds — the guard after this call
        // will detect the null and side-exit to the interpreter.
        // RPython: residual calls that fail trigger guard failure, not crash.
        Err(_) => pyre_object::PY_NULL as i64,
    }
}

pub fn unpack_sequence_exact(seq: PyObjectRef, count: usize) -> Result<Vec<PyObjectRef>, PyError> {
    // Fast path only for exact built-in sequence types. Subclasses and other
    // instances may define custom `__iter__` that must be honored.
    let exact_sequence_len = unsafe {
        if pyre_object::is_exact_tuple(seq) {
            Some(w_tuple_len(seq))
        } else if pyre_object::is_exact_list(seq) {
            Some(w_list_len(seq))
        } else if pyre_object::is_exact_type(seq, &pyre_object::STR_TYPE) {
            Some(w_str_len(seq))
        } else {
            None
        }
    };
    if let Some(len) = exact_sequence_len {
        if len != count {
            // `baseobjspace.py:1041-1053 _unpackiterable_known_length_jitlook`
            // raises ValueError on length mismatch.
            let msg = if len > count {
                format!("too many values to unpack (expected {count})")
            } else {
                format!("not enough values to unpack (expected {count}, got {len})")
            };
            return Err(PyError::value_error(msg));
        }
        return (0..count).map(|idx| sequence_getitem(seq, idx)).collect();
    }
    // Fallback: iteration protocol (handles type objects with metaclass __iter__, etc.)
    // baseobjspace.py:1031 _unpackiterable_known_length_jitlook.  pyopcode.py:872
    // UNPACK_SEQUENCE wraps the whole `fixedview_unroll` (iter + known-length
    // loop) in a TypeError → "cannot unpack non-iterable %T object" remap.
    let non_iterable = || {
        PyError::type_error(format!("cannot unpack non-iterable {} object", unsafe {
            (*(*seq).ob_type).name
        }))
    };
    let iter = match crate::baseobjspace::iter(seq) {
        Ok(it) => it,
        Err(e) if e.kind == PyErrorKind::TypeError => return Err(non_iterable()),
        Err(e) => return Err(e),
    };
    let mut items = Vec::with_capacity(count);
    loop {
        match crate::baseobjspace::next(iter) {
            Ok(val) => {
                if items.len() == count {
                    return Err(PyError::value_error(format!(
                        "too many values to unpack (expected {count})"
                    )));
                }
                items.push(val);
            }
            Err(e) if e.kind == PyErrorKind::StopIteration => break,
            Err(e) if e.kind == PyErrorKind::TypeError => return Err(non_iterable()),
            Err(e) => return Err(e),
        }
    }
    if items.len() < count {
        return Err(PyError::value_error(format!(
            "not enough values to unpack (expected {count}, got {})",
            items.len()
        )));
    }
    Ok(items)
}

/// UNPACK_EX — split `value` for `a, *b, c = value` into `before` head
/// items, a starred middle list, and `after` tail items, returning the
/// `before + 1 + after` slots in TOS order ([head items], middle list,
/// [tail items]).  Shared by the interpreter's `unpack_ex` handler and the
/// JIT residual `bh_unpack_ex_fn`; the portal reads each slot back out with
/// `bh_unpack_item_fn`, exactly as `unpack_sequence_exact`.  Raises
/// ValueError when fewer than `before + after` values are available.
pub fn unpack_ex_slots(
    before: usize,
    after: usize,
    value: PyObjectRef,
) -> Result<Vec<PyObjectRef>, PyError> {
    let elements: Vec<PyObjectRef> = unsafe {
        if is_tuple(value) {
            pyre_object::w_tuple_items_copy_as_vec(value)
        } else if is_list(value) {
            pyre_object::w_list_items_copy_as_vec(value)
        } else {
            // pyopcode.py:884 UNPACK_EX wraps `fixedview` in a
            // TypeError → "cannot unpack non-iterable %T object" remap.
            // `collect_iterable` is the `fixedview` analog (iter + next
            // loop), so any TypeError it raises is remapped here.
            match crate::builtins::collect_iterable(value) {
                Ok(items) => items,
                Err(e) if e.kind == PyErrorKind::TypeError => {
                    return Err(PyError::type_error(format!(
                        "cannot unpack non-iterable {} object",
                        (*(*value).ob_type).name
                    )));
                }
                Err(e) => return Err(e),
            }
        }
    };
    let min_expected = before + after;
    if elements.len() < min_expected {
        return Err(PyError::value_error(format!(
            "not enough values to unpack (expected at least {}, got {})",
            min_expected,
            elements.len()
        )));
    }
    let middle_len = elements.len() - min_expected;
    let mut slots = Vec::with_capacity(before + 1 + after);
    for &item in elements.iter().take(before) {
        slots.push(item);
    }
    let middle: Vec<PyObjectRef> = elements[before..before + middle_len].to_vec();
    slots.push(w_list_new(middle));
    for i in 0..after {
        slots.push(elements[before + middle_len + i]);
    }
    Ok(slots)
}

pub fn ensure_range_iter(iter: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if is_range_iter(iter) || is_seq_iter(iter) {
            return Ok(());
        }
        // Convert list/tuple to seq iterator
        if is_list(iter) {
            // Replace TOS with a seq iterator wrapping the list
            // This is called on TOS after GET_ITER pops and pushes.
            // But ensure_iter is called ON the iter — it can't replace stack.
            // So we need to create iter before calling ensure.
            // Actually ensure_range_iter is called AFTER get_iter pushes.
            // The problem: GET_ITER calls ensure_iter_value on the pushed value.
            // For list, we need to push a seq_iter instead of the list itself.
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*iter).ob_type).name }
    )))
}

/// Current length of the sequence a `W_SeqIterObject` walks. `None` for a
/// payload outside list/tuple/str/array, leaving callers to fall back to
/// the length captured at iterator creation.  The covered set mirrors the
/// item-fetch arms of `baseobjspace::next` / `range_iter_next_or_null`, so
/// `range_iter_continues` and the next-value helper stay consistent (a
/// payload one can fetch from is a payload one can length).
///
/// # Safety
/// `seq` must be a valid object pointer.
unsafe fn seq_iter_current_len(seq: PyObjectRef) -> Option<i64> {
    unsafe {
        if is_list(seq) {
            Some(w_list_len(seq) as i64)
        } else if is_tuple(seq) {
            Some(w_tuple_len(seq) as i64)
        } else if is_str(seq) {
            // Code-point count, not byte count (matches the str seq-iter
            // seed in baseobjspace::iter).
            Some(w_str_len(seq) as i64)
        } else if pyre_object::interp_array::is_array(seq) {
            Some(pyre_object::interp_array::w_array_len(seq) as i64)
        } else {
            None
        }
    }
}

pub fn range_iter_continues(iter: PyObjectRef) -> Result<bool, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_has_next(iter));
        }
        if pyre_object::is_long_range_iter(iter) {
            return Ok(pyre_object::w_long_range_iter_has_next(iter));
        }
        if is_seq_iter(iter) {
            let si = &*(iter as *const W_SeqIterObject);
            // A cleared sequence (exhausted cursor, iterobject.py:90-98) has no
            // more items — a re-iteration of an exhausted iterator stops here
            // without dereferencing the freed sequence.
            if si.seq.is_null() {
                return Ok(false);
            }
            // sequenceiterator re-reads the live sequence length and PyPy's
            // W_FastListIterObject.descr_next ends on a getitem IndexError;
            // neither snapshots a length. Read the CURRENT sequence length so
            // an element appended during iteration is observed (and a removed
            // tail ends the loop) rather than the length captured at iterator
            // creation.
            let len = seq_iter_current_len(si.seq).unwrap_or(si.length);
            return Ok(si.index < len);
        }
    }
    Err(PyError::type_error("not an iterator"))
}

pub fn range_iter_next_or_null(iter: PyObjectRef) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_next(iter).unwrap_or(PY_NULL));
        }
        if pyre_object::is_long_range_iter(iter) {
            return Ok(pyre_object::w_long_range_iter_next(iter).unwrap_or(PY_NULL));
        }
        if is_seq_iter(iter) {
            let si = &mut *(iter as *mut W_SeqIterObject);
            // An exhausted cursor cleared its sequence (iterobject.py:90-98
            // `self.w_seq = None`); a re-entry stays exhausted without
            // dereferencing the freed sequence.
            if si.seq.is_null() {
                return Ok(PY_NULL);
            }
            let idx = si.index;
            // Bounds come from the sequence's CURRENT state (see
            // range_iter_continues): getitem returns None past the live
            // length, so an element appended during iteration is yielded and a
            // removed tail ends the loop. Mirrors baseobjspace::next.
            let item = if is_list(si.seq) {
                w_list_getitem(si.seq, idx)
            } else if is_tuple(si.seq) {
                w_tuple_getitem(si.seq, idx)
            } else if is_str(si.seq) {
                // Box the idx-th code point as a one-character str, reading
                // the WTF-8 view so a lone surrogate is yielded instead of
                // panicking. Only `iter(str)` through the builtin reaches here
                // with a str payload (the GET_ITER opcode materialises a char
                // list); without this arm `seq_iter_current_len` would say
                // "more" while this helper returned PY_NULL forever, hanging
                // the loop. Mirrors baseobjspace::next.
                let s = w_str_get_wtf8(si.seq);
                let mut found: Option<PyObjectRef> = None;
                let mut n = 0i64;
                for cp in s.code_points() {
                    if n == idx {
                        let mut one = Wtf8Buf::new();
                        one.push(cp);
                        found = Some(w_str_from_wtf8(one));
                        break;
                    }
                    n += 1;
                }
                found
            } else if pyre_object::interp_array::is_array(si.seq) {
                if (idx as usize) < pyre_object::interp_array::w_array_len(si.seq) {
                    Some(pyre_object::interp_array::w_array_unpack_item(
                        si.seq,
                        idx as usize,
                    ))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(v) = item {
                si.index += 1;
                return Ok(v);
            }
            // iterobject.py:96-98 — clear the sequence ref on exhaustion so a
            // held iterator's `__reduce__`/`__length_hint__` report it as an
            // exhausted cursor (matches the generic-`__getitem__` arm of
            // `baseobjspace::next`).
            si.seq = PY_NULL;
            return Ok(PY_NULL);
        }
    }
    Err(PyError::type_error("not an iterator"))
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_range_iter_next_or_null(iter: i64) -> i64 {
    match range_iter_next_or_null(iter as PyObjectRef) {
        Ok(value) => value as i64,
        Err(err) => panic!("range iter next failed in JIT: {err}"),
    }
}

/// A sequence iterator whose payload is not one of the inline fast types
/// (list/tuple/str/array, see `seq_iter_current_len`) is a generic
/// sequence-protocol iterator: `next` fetches each item through
/// `space.getitem` (iterobject.py W_SeqIterObject.descr_next), so it is
/// driven by `baseobjspace::next` rather than the inline range helper.  An
/// exhausted cursor (null seq) routes here too so the trailing StopIteration
/// is raised by `next`.
///
/// # Safety
/// `iter` must be a valid object pointer.
unsafe fn is_generic_seq_iter(iter: PyObjectRef) -> bool {
    unsafe {
        is_seq_iter(iter) && {
            let seq = (*(iter as *const W_SeqIterObject)).seq;
            seq.is_null() || seq_iter_current_len(seq).is_none()
        }
    }
}

/// FOR_ITER iterator kinds that advance through `space.next` rather than
/// the inline range helper: generic sequence-protocol iterators, generators,
/// itertools/enumerate/reversed/filter/map/zip/dictview/sre, callable
/// iterators, and user `__next__` instances. Shared so the interpreter
/// `iter_next` and the tracer agree on which iterators take the `space.next`
/// leg.
pub fn via_space_next(iter: PyObjectRef) -> bool {
    unsafe {
        is_generic_seq_iter(iter)
            || is_instance(iter)
            || pyre_object::generator::is_generator(iter)
            || pyre_object::interp_itertools::is_repeat(iter)
            || pyre_object::interp_itertools::is_count(iter)
            || pyre_object::interp_itertools::is_takewhile(iter)
            || pyre_object::interp_itertools::is_dropwhile(iter)
            || pyre_object::interp_itertools::is_filterfalse(iter)
            || pyre_object::interp_itertools::is_pairwise(iter)
            || pyre_object::interp_itertools::is_cycle(iter)
            || pyre_object::interp_itertools::is_chain(iter)
            || pyre_object::functional::is_enumerate(iter)
            || pyre_object::functional::is_reversed(iter)
            || pyre_object::functional::is_filter(iter)
            || pyre_object::functional::is_map(iter)
            || pyre_object::functional::is_zip(iter)
            || pyre_object::operation::is_callable_iterator(iter)
            || pyre_object::dictmultiobject::is_dict_view_iterator(iter)
            || pyre_object::interp_sre::is_sre_scanner(iter)
            || crate::module::r#struct::is_unpack_iter(iter)
    }
}

/// Residual FOR_ITER `space.next` for the `via_space_next` iterator kinds.
/// Exhaustion is signalled the same way as the range fast path: a null
/// return that the trailing for-iter GuardNonnull catches, side-exiting to
/// the interpreter which re-runs FOR_ITER and ends the loop (eval.rs
/// `iter_next` maps StopIteration to exhausted). A *real* exception is
/// published into BOTH the backend exception cells (so the compiled trace /
/// blackhole GuardNoException side-exits) AND `BH_LAST_EXC_VALUE` (so the
/// full-body walk's `execute_residual_call` returns Err and records the
/// can-raise path instead of mistaking the null for exhaustion). Keeping the
/// two seams in sync mirrors `call_jit::publish_residual_call_exception`, the
/// dual-publish every other MayForce residual uses.
#[majit_macros::jit_may_force]
pub extern "C" fn jit_next(iter: i64) -> i64 {
    match crate::baseobjspace::next(iter as PyObjectRef) {
        Ok(value) => value as i64,
        // StopIteration is not a frame-level exception for FOR_ITER; return
        // null so the GuardNonnull (not GuardNoException) fires.
        Err(err) if err.kind == PyErrorKind::StopIteration => 0,
        Err(err) => {
            let exc_obj = err.to_exc_object();
            if exc_obj != PY_NULL {
                majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            }
            jit_publish_exception(exc_obj);
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyExecutionContext;
    use pyre_object::{w_int_get_value, w_int_new};

    #[test]
    fn test_dispatch_callable_runs_builtin_branch() {
        let ctx = PyExecutionContext::default();
        let abs = ctx.lookup_builtin("abs").expect("abs builtin must exist");
        let result = dispatch_callable(
            abs,
            |callable| {
                // Builtins are now Function objects; extract code then func pointer.
                let code = unsafe { crate::getcode(callable) };
                let func = unsafe { crate::builtin_code_get(code as PyObjectRef) };
                func(&[w_int_new(-9)])
            },
            |_callable| panic!("builtin callable should not take user branch"),
        )
        .expect("builtin dispatch should succeed");

        unsafe {
            assert_eq!(w_int_get_value(result), 9);
        }
    }

    #[test]
    fn test_dispatch_callable_rejects_non_callable() {
        let err = dispatch_callable(w_int_new(3), |_callable| Ok(()), |_callable| Ok(()))
            .expect_err("non-callable dispatch should fail");

        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("not callable"));
    }

    #[test]
    fn test_range_iter_helpers_share_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 2, 1);
        assert!(range_iter_continues(iter).unwrap());
        let first = range_iter_next_or_null(iter).unwrap();
        let second = range_iter_next_or_null(iter).unwrap();
        let done = range_iter_next_or_null(iter).unwrap();
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_range_iter_helper_shares_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 2, 1);
        let first = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let second = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let done = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_sequence_getitem_shares_runtime_sequence_semantics() {
        let tuple = pyre_object::w_tuple_new(vec![w_int_new(3), w_int_new(5)]);
        let item = jit_sequence_getitem(tuple as i64, 1) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(item), 5);
        }
    }

    #[test]
    fn seq_iter_observes_list_growth_during_iteration() {
        // listiterator re-reads the live list size each step, so an append
        // during iteration is observed. range_iter_continues /
        // range_iter_next_or_null must read the CURRENT length, not the length
        // captured at iterator creation — otherwise `for x in xs:
        // xs.append(...)` stops at the initial length.
        let list = pyre_object::w_list_new(vec![w_int_new(0)]);
        let iter = pyre_object::w_seq_iter_new(list, 1);
        let mut seen = Vec::new();
        while range_iter_continues(iter).unwrap() {
            let v = range_iter_next_or_null(iter).unwrap();
            assert!(!v.is_null());
            let x = unsafe { w_int_get_value(v) };
            seen.push(x);
            if unsafe { pyre_object::w_list_len(list) } < 5 {
                unsafe { pyre_object::w_list_append(list, w_int_new(x + 1)) };
            }
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn seq_iter_over_str_payload_yields_codepoints_and_terminates() {
        // `iter(str)` through the builtin (baseobjspace::iter) makes a seq-iter
        // with a STR payload (the GET_ITER opcode instead materialises a char
        // list). range_iter_continues / range_iter_next_or_null must fetch and
        // advance for that payload too; otherwise `continues` stays true while
        // `next` returns PY_NULL forever, hanging `for c in iter(s)`.
        let s = pyre_object::unicodeobject::box_str_constant(Wtf8::new("abcde"));
        let iter = pyre_object::w_seq_iter_new(s, 5);
        let mut count = 0;
        while range_iter_continues(iter).unwrap() {
            let v = range_iter_next_or_null(iter).unwrap();
            assert!(
                !v.is_null(),
                "str seq-iter yielded NULL while continues==true"
            );
            count += 1;
            assert!(
                count <= 5,
                "str seq-iter did not terminate (infinite-loop regression)"
            );
        }
        assert_eq!(count, 5);
    }
}
