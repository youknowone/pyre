//! `_cffi_backend.FFI` — PyPy: `pypy/module/_cffi_backend/ffi_obj.py`.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};
use super::{
    allocator, cbuffer, cdlopen, cerrno, func, handle, lib_obj, newtype, parse_c_type,
    realize_c_type, wrapper,
};

pub const ACCEPT_STRING: i64 = 1;
pub const ACCEPT_CTYPE: i64 = 2;
pub const ACCEPT_CDATA: i64 = 4;
pub const ACCEPT_ALL: i64 = ACCEPT_STRING | ACCEPT_CTYPE | ACCEPT_CDATA;
pub const CONSIDER_FN_AS_FNPTR: i64 = 8;

/// `FreeCtxObj`: sweep frees the copied parser context and ABI-mode arrays.
pub struct FreeCtxObj {
    ctxobj: *mut parse_c_type::CtxObj,
    free_mems: Vec<*mut u8>,
}

/// `W_FFIObject`.
#[crate::pyre_class("_cffi_backend.FFI", cpython_heaptype)]
pub struct W_FFIObject {
    /// Mapdict prefix used by subclasses of `FFI`.
    pub map: usize,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    /// `W_FFIObject.types_dict`.
    pub types_dict: PyObjectRef,
    /// `W_FFIObject.ctxobj`.
    pub ctxobj: *mut parse_c_type::CtxObj,
    /// `W_FFIObject._finalizer`.
    finalizer: *mut FreeCtxObj,
    /// `W_FFIObject.cached_types`; null for a plain ABI-mode FFI.
    pub cached_types: PyObjectRef,
    pub is_static: i64,
    pub is_nonempty: i64,
    /// `(ffi, lib)` pairs walked by realization and generated libraries.
    pub included_ffis_libs: PyObjectRef,
    /// `W_FFIObject.w_init_once_cache`.
    pub w_init_once_cache: PyObjectRef,
    /// Unpublished identity marker for `W_InitOnceLock` cache placeholders.
    pub w_init_once_marker: PyObjectRef,
    /// Stable native locks named by those placeholders.
    init_once_locks: *mut InitOnceLocks,
}

impl Default for W_FFIObject {
    fn default() -> Self {
        Self {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: pyre_object::PY_NULL,
            },
            map: 0,
            storage: std::ptr::null_mut(),
            types_dict: pyre_object::PY_NULL,
            ctxobj: std::ptr::null_mut(),
            finalizer: std::ptr::null_mut(),
            cached_types: pyre_object::PY_NULL,
            is_static: 0,
            is_nonempty: 0,
            included_ffis_libs: pyre_object::PY_NULL,
            w_init_once_cache: pyre_object::PY_NULL,
            w_init_once_marker: pyre_object::PY_NULL,
            init_once_locks: std::ptr::null_mut(),
        }
    }
}

const _: () = assert!(
    std::mem::offset_of!(W_FFIObject, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_FFIObject must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_FFIObject, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_FFIObject must keep W_ObjectObject's storage offset"
);

pub(crate) fn ffi_arg(w_ffi: PyObjectRef) -> Result<&'static mut W_FFIObject, PyError> {
    W_FFIObject::from_obj(w_ffi).ok_or_else(|| PyError::type_error("expected an FFI object"))
}

pub(crate) fn ffi_error(message: impl Into<String>) -> PyError {
    let message = message.into();
    let mut error = PyError::runtime_error(message.clone());
    if let Ok(w_exc) = crate::builtins::exc_exception_new(&[
        newtype::ffi_error(),
        pyre_object::w_str_new(&message),
    ]) {
        error.exc_object = w_exc;
    }
    error
}

/// `W_FFIObject.__init__`.
fn initialize_ffi(
    w_ffitype: PyObjectRef,
    src_ctx: *const parse_c_type::TypeContextS,
) -> Result<PyObjectRef, PyError> {
    let ctxobj = parse_c_type::allocate_ctxobj(src_ctx);
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = roots.base();
    let _ = roots.pin_root(w_ffitype);
    let dict_slot = type_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let cached_slot = dict_slot + 1;
    let cached = if src_ctx.is_null() {
        pyre_object::PY_NULL
    } else {
        let count = unsafe { parse_c_type::get_num_types(src_ctx) };
        pyre_object::w_list_new((0..count).map(|_| pyre_object::w_none()).collect())
    };
    let _ = roots.pin_root(cached);
    let included_slot = cached_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let once_cache_slot = included_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let once_marker_slot = once_cache_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let obj = W_FFIObject::allocate_stable(W_FFIObject {
        types_dict: roots.get(dict_slot),
        ctxobj,
        finalizer: Box::into_raw(Box::new(FreeCtxObj {
            ctxobj,
            free_mems: Vec::new(),
        })),
        cached_types: roots.get(cached_slot),
        is_static: i64::from(!src_ctx.is_null()),
        is_nonempty: i64::from(!src_ctx.is_null()),
        included_ffis_libs: roots.get(included_slot),
        w_init_once_cache: roots.get(once_cache_slot),
        w_init_once_marker: roots.get(once_marker_slot),
        init_once_locks: Box::into_raw(Box::new(InitOnceLocks::default())),
        ..Default::default()
    });
    Ok(crate::typedef::tag_subclass_instance(
        obj,
        roots.get(type_slot),
    ))
}

/// `make_plain_ffi_object`.
pub fn make_plain_ffi_object(w_ffitype: PyObjectRef) -> Result<PyObjectRef, PyError> {
    crate::typedef::check_user_subclass(ffi_type_object(), w_ffitype)?;
    initialize_ffi(w_ffitype, std::ptr::null())
}

/// `W_FFIObject___new__`.
fn ffi_new(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_subtype = args
        .first()
        .copied()
        .ok_or_else(|| PyError::type_error("FFI.__new__() missing subtype"))?;
    make_plain_ffi_object(w_subtype)
}

/// `W_FFIObject.descr_init`.
fn ffi_init(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(
        args,
        "__init__",
        &[
            "module_name",
            "_version",
            "_types",
            "_globals",
            "_struct_unions",
            "_enums",
            "_typenames",
            "_includes",
        ],
        0,
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let ffi = ffi_arg(roots.get(base))?;
    if ffi.is_nonempty != 0 {
        return Err(PyError::value_error(
            "cannot call FFI.__init__() more than once",
        ));
    }
    ffi.is_nonempty = 1;
    let module_name = if roots.get(base + 1).is_null() {
        "?".to_string()
    } else {
        crate::baseobjspace::text_w(roots.get(base + 1))?.to_string()
    };
    let version = if roots.get(base + 2).is_null() {
        -1
    } else {
        crate::baseobjspace::int_w(roots.get(base + 2))?
    };
    let types = if roots.get(base + 3).is_null() {
        Vec::new()
    } else if unsafe { pyre_object::bytesobject::is_bytes(roots.get(base + 3)) } {
        unsafe { pyre_object::bytesobject::w_bytes_data(roots.get(base + 3)) }.to_vec()
    } else {
        return Err(PyError::type_error(format!(
            "expected bytes for _types, got '{}'",
            crate::type_methods::arg_type_name(roots.get(base + 3))
        )));
    };
    cdlopen::ffiobj_init(
        roots.get(base),
        &module_name,
        version,
        &types,
        roots.get(base + 4),
        roots.get(base + 5),
        roots.get(base + 6),
        roots.get(base + 7),
        roots.get(base + 8),
    )?;
    Ok(pyre_object::w_none())
}

/// `FreeCtxObj.__del__`.
///
/// # Safety
/// `obj` must be a GC-dead `W_FFIObject`.
pub unsafe fn w_ffi_dealloc(obj: PyObjectRef) {
    let ffi = unsafe { &mut *(obj as *mut W_FFIObject) };
    ffi.ctxobj = std::ptr::null_mut();
    if !ffi.finalizer.is_null() {
        let mut finalizer = unsafe { Box::from_raw(ffi.finalizer) };
        ffi.finalizer = std::ptr::null_mut();
        unsafe { parse_c_type::free_ctxobj(finalizer.ctxobj) };
        for ptr in finalizer.free_mems.drain(..).rev() {
            unsafe { libc::free(ptr.cast()) };
        }
    }
    if !ffi.init_once_locks.is_null() {
        drop(unsafe { Box::from_raw(ffi.init_once_locks) });
        ffi.init_once_locks = std::ptr::null_mut();
    }
}

pub(crate) fn track_free_mem(w_ffi: PyObjectRef, ptr: *mut u8) -> Result<(), PyError> {
    let ffi = ffi_arg(w_ffi)?;
    if ffi.finalizer.is_null() {
        return Err(PyError::system_error("FFI finalizer is unavailable"));
    }
    unsafe { &mut *ffi.finalizer }.free_mems.push(ptr);
    Ok(())
}

pub(crate) fn allocate_free_mem(w_ffi: PyObjectRef, nbytes: usize) -> Result<*mut u8, PyError> {
    let ptr = unsafe { libc::calloc(1, nbytes.max(1)) }.cast::<u8>();
    if ptr.is_null() {
        return Err(PyError::new(
            crate::PyErrorKind::MemoryError,
            "cannot allocate FFI context memory",
        ));
    }
    if let Err(error) = track_free_mem(w_ffi, ptr) {
        unsafe { libc::free(ptr.cast()) };
        return Err(error);
    }
    Ok(ptr)
}

/// Bind the positional-or-keyword part of an FFI method.  `names` excludes
/// the receiver and `required` counts required entries in that list.
fn bind_method(
    args: &[PyObjectRef],
    method: &str,
    names: &[&str],
    required: usize,
) -> Result<Vec<PyObjectRef>, PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(PyError::type_error(format!(
            "{method}() missing FFI receiver"
        )));
    }
    if positional.len() > names.len() + 1 {
        return Err(PyError::type_error(format!(
            "{method}() takes at most {} arguments ({} given)",
            names.len(),
            positional.len() - 1
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, names, method)?;
    let mut bound = Vec::with_capacity(names.len() + 1);
    bound.push(positional[0]);
    for (i, name) in names.iter().enumerate() {
        let positional_value = positional.get(i + 1).copied();
        let keyword_value = crate::builtins::kwarg_get(kwargs, name);
        let value = match (positional_value, keyword_value) {
            (Some(_), Some(_)) => {
                return Err(PyError::type_error(format!(
                    "{method}() got multiple values for argument '{name}'"
                )));
            }
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) if i < required => {
                return Err(PyError::type_error(format!(
                    "{method}() missing required argument '{name}'"
                )));
            }
            (None, None) => pyre_object::PY_NULL,
        };
        bound.push(value);
    }
    Ok(bound)
}

fn no_keyword_varargs<'a>(
    args: &'a [PyObjectRef],
    method: &str,
) -> Result<&'a [PyObjectRef], PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::real_kwarg_count(kwargs) != 0 {
        return Err(PyError::type_error(format!(
            "{method}() takes no keyword arguments"
        )));
    }
    Ok(positional)
}

fn dict_string_type(ffi: &W_FFIObject, string: &str) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(ffi.types_dict, string) }
}

/// `W_FFIObject.get_string_to_type`.
pub fn get_string_to_type(
    w_ffi: PyObjectRef,
    string: &str,
    consider_fn_as_fnptr: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    let Some(x) = dict_string_type(ffi_arg(w_ffi)?, string) else {
        return Ok(None);
    };
    if ctypeobj::ctype_at(x).is_some() {
        return Ok(Some(x));
    }
    let raw = realize_c_type::W_RawFuncType::from_obj(x)
        .ok_or_else(|| PyError::system_error("FFI type cache holds an unknown object"))?;
    if consider_fn_as_fnptr {
        raw.unwrap_as_fnptr_in_elidable().map(Some)
    } else {
        Ok(None)
    }
}

/// `W_FFIObject._ffi_bad_type`.
fn ffi_bad_type(w_ffi: PyObjectRef, input_text: &str) -> PyError {
    let ffi = match ffi_arg(w_ffi) {
        Ok(ffi) => ffi,
        Err(error) => return error,
    };
    let info = unsafe { &(*ffi.ctxobj).info };
    let errmsg = if info.error_message.is_null() {
        "parse error".to_string()
    } else {
        unsafe { CStr::from_ptr(info.error_message) }
            .to_string_lossy()
            .into_owned()
    };
    if input_text.len() > 500 {
        return ffi_error(errmsg);
    }
    let printable: String = input_text
        .chars()
        .map(|c| {
            if (' '..'\x7f').contains(&c) {
                c
            } else if c == '\t' || c == '\n' {
                ' '
            } else {
                '?'
            }
        })
        .collect();
    ffi_error(format!(
        "{errmsg}\n{printable}\n{}^",
        " ".repeat(info.error_location)
    ))
}

/// `W_FFIObject.parse_string_to_type`.
pub fn parse_string_to_type(
    w_ffi: PyObjectRef,
    string: &str,
    consider_fn_as_fnptr: bool,
) -> Result<PyObjectRef, PyError> {
    if let Some(x) = dict_string_type(ffi_arg(w_ffi)?, string) {
        if ctypeobj::ctype_at(x).is_some() {
            return Ok(x);
        }
        if let Some(raw) = realize_c_type::W_RawFuncType::from_obj(x) {
            return if consider_fn_as_fnptr {
                raw.unwrap_as_fnptr_in_elidable()
            } else {
                Err(raw.unexpected_fn_type(w_ffi))
            };
        }
    }
    let input = CString::new(string).map_err(|_| ffi_error("unexpected NUL character"))?;
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    realize_c_type::with_realize_lock(|| {
        let ffi = ffi_arg(roots.get(ffi_slot))?;
        let info = unsafe { &mut (*ffi.ctxobj).info };
        let index = parse_c_type::parse_type(info, &input);
        if index < 0 {
            return Err(ffi_bad_type(roots.get(ffi_slot), string));
        }
        let x_slot = ffi_slot + 1;
        let _ = roots.pin_root(realize_c_type::realize_c_type_or_func(
            roots.get(ffi_slot),
            info.output,
            index,
        )?);
        if realize_c_type::W_RawFuncType::from_obj(roots.get(x_slot)).is_some() {
            let _ = realize_c_type::W_RawFuncType::unwrap_as_fnptr(
                roots.get(x_slot),
                roots.get(ffi_slot),
            )?;
        }
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(
                ffi_arg(roots.get(ffi_slot))?.types_dict,
                string,
                roots.get(x_slot),
            )
        };
        let x = roots.get(x_slot);
        if ctypeobj::ctype_at(x).is_some() {
            Ok(x)
        } else {
            let raw = realize_c_type::W_RawFuncType::from_obj(x)
                .ok_or_else(|| PyError::system_error("realization returned an unknown object"))?;
            if consider_fn_as_fnptr {
                raw.unwrap_as_fnptr_in_elidable()
            } else {
                Err(raw.unexpected_fn_type(roots.get(ffi_slot)))
            }
        }
    })
}

/// `W_FFIObject.ffi_type`.
pub fn ffi_type(w_ffi: PyObjectRef, w_x: PyObjectRef, accept: i64) -> Result<PyObjectRef, PyError> {
    if accept & ACCEPT_STRING != 0 && unsafe { pyre_object::unicodeobject::is_str(w_x) } {
        let string = crate::baseobjspace::text_w(w_x)?.to_string();
        let consider = accept & CONSIDER_FN_AS_FNPTR != 0;
        if let Some(found) = get_string_to_type(w_ffi, &string, consider)? {
            return Ok(found);
        }
        return parse_string_to_type(w_ffi, &string, consider);
    }
    if accept & ACCEPT_CTYPE != 0 && W_CType::from_obj(w_x).is_some() {
        return Ok(w_x);
    }
    if accept & ACCEPT_CDATA != 0
        && let Some(cdata) = W_CData::from_obj(w_x)
    {
        return Ok(cdata.ctype);
    }
    let mut expected = Vec::new();
    if accept & ACCEPT_STRING != 0 {
        expected.push("string");
    }
    if accept & ACCEPT_CTYPE != 0 {
        expected.push("ctype object");
    }
    if accept & ACCEPT_CDATA != 0 {
        expected.push("cdata object");
    }
    Err(PyError::type_error(format!(
        "expected a {}, got '{}'",
        expected.join(" or "),
        crate::type_methods::arg_type_name(w_x)
    )))
}

/// `W_FFIObject.fetch_int_constant`.
pub fn fetch_int_constant(w_ffi: PyObjectRef, name: &str) -> Result<Option<PyObjectRef>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    let index = parse_c_type::search_in_globals(ctx, name);
    if index >= 0 {
        let g = unsafe { *ctx.globals.offset(index) };
        let op = parse_c_type::getop(g.type_op);
        if op == parse_c_type::OP_CONSTANT_INT || op == parse_c_type::OP_ENUM {
            return realize_c_type::realize_global_int(roots.get(ffi_slot), g, index).map(Some);
        }
        return Err(ffi_error(format!(
            "function, global variable or non-integer constant '{name}' must be fetched from its original 'lib' object"
        )));
    }
    let included_list_slot = ffi_slot + 1;
    let _ = roots.pin_root(ffi.included_ffis_libs);
    let included = crate::baseobjspace::fixedview(roots.get(included_list_slot), -1)?;
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &item in &included {
        let _ = roots.pin_root(item);
    }
    for i in 0..included.len() {
        let pair = crate::baseobjspace::fixedview(roots.get(base + i), 2)?;
        let pair_slot = pyre_object::gc_roots::shadow_stack_len();
        for &item in &pair {
            let _ = roots.pin_root(item);
        }
        if let Some(result) = fetch_int_constant(roots.get(pair_slot), name)? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

fn ffi_addressof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let args = no_keyword_varargs(args, "addressof")?;
    if args.len() < 2 {
        return Err(PyError::type_error("addressof() missing cdata argument"));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &arg in args {
        let _ = roots.pin_root(arg);
    }
    if lib_obj::W_LibObject::from_obj(roots.get(base + 1)).is_some() && args.len() == 3 {
        let name = crate::baseobjspace::text_w(roots.get(base + 2))?.to_string();
        return lib_obj::address_of_func_or_global_var(roots.get(base + 1), &name);
    }
    let mut w_ctype = ffi_type(roots.get(base), roots.get(base + 1), ACCEPT_CDATA)?;
    let mut offset = 0;
    if args.len() == 2 {
        let ct = ctypeobj::ctype_arg(w_ctype)?;
        if !ct.is_struct_or_union() && ct.kind != ctypeobj::KIND_ARRAY {
            return Err(PyError::type_error(
                "expected a cdata struct/union/array object",
            ));
        }
    } else {
        let ct = ctypeobj::ctype_arg(w_ctype)?;
        if !ct.is_struct_or_union()
            && ct.kind != ctypeobj::KIND_ARRAY
            && ct.kind != ctypeobj::KIND_POINTER
        {
            return Err(PyError::type_error(
                "expected a cdata struct/union/array/pointer object",
            ));
        }
        for i in 2..args.len() {
            let realized = func::direct_typeoffsetof(w_ctype, roots.get(base + i), i > 2)?;
            w_ctype = realized.0;
            offset += realized.1;
        }
    }
    let cdata = cdataobj::cdata_arg(roots.get(base + 1))?;
    let ptr = unsafe { cdata.ptr.offset(offset as isize) };
    let ptr_type = newtype::new_pointer_type(w_ctype)?;
    Ok(cdataobj::new_cdata(ptr, ptr_type))
}

fn ffi_alignof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "alignof", &["cdecl"], 1)?;
    let w_ctype = ffi_type(a[0], a[1], ACCEPT_ALL)?;
    Ok(pyre_object::w_int_new(
        ctypeobj::ctype_arg(w_ctype)?.alignof()?,
    ))
}

fn ffi_cast(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "cast", &["cdecl", "source"], 2)?;
    let roots = pyre_object::gc_roots::push_roots();
    let source_slot = roots.base();
    let _ = roots.pin_root(a[2]);
    let w_ctype = ffi_type(a[0], a[1], ACCEPT_STRING | ACCEPT_CTYPE)?;
    ctypeobj::cast(w_ctype, roots.get(source_slot))
}

/// `W_FFIObject.descr_callback`.
fn ffi_callback(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(
        args,
        "callback",
        &["cdecl", "python_callable", "error", "onerror"],
        1,
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let w_ctype = ffi_type(
        roots.get(base),
        roots.get(base + 1),
        ACCEPT_STRING | ACCEPT_CTYPE | CONSIDER_FN_AS_FNPTR,
    )?;
    let ctype_slot = base + a.len();
    let _ = roots.pin_root(w_ctype);
    let error_slot = ctype_slot + 1;
    let _ = roots.pin_root(if roots.get(base + 3).is_null() {
        pyre_object::w_none()
    } else {
        roots.get(base + 3)
    });
    let onerror_slot = error_slot + 1;
    let _ = roots.pin_root(if roots.get(base + 4).is_null() {
        pyre_object::w_none()
    } else {
        roots.get(base + 4)
    });
    // Read the callable back out of its slot rather than holding it in a local:
    // `pin_root` publishes through a safepoint, so a value carried across one
    // may have moved.
    let w_python_callable = roots.get(base + 2);
    if !w_python_callable.is_null() && unsafe { !pyre_object::is_none(w_python_callable) } {
        return super::ccallback::make_callback(
            roots.get(ctype_slot),
            w_python_callable,
            roots.get(error_slot),
            roots.get(onerror_slot),
        );
    }

    // `space.appexec`: keep the decorator as an app-level function with the
    // ctype and the two policy values in its globals.
    let w_module = crate::importing::get_sys_module("_cffi_backend")
        .ok_or_else(|| PyError::system_error("_cffi_backend is not loaded"))?;
    let module_slot = onerror_slot + 1;
    let _ = roots.pin_root(w_module);
    let globals_slot = module_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            roots.get(globals_slot),
            "_cffi_backend",
            roots.get(module_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            roots.get(globals_slot),
            "ctype",
            roots.get(ctype_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            roots.get(globals_slot),
            "error",
            roots.get(error_slot),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            roots.get(globals_slot),
            "onerror",
            roots.get(onerror_slot),
        );
    }
    // `appexec` compiles through `gateway.py ApplevelClass`, whose whole
    // constants graph carries `hidden_applevel=True`; compiling here rather
    // than calling the `eval` builtin keeps the decorator working when the
    // application has rebound that name.
    const DECORATOR: &str =
        "lambda python_callable: _cffi_backend.callback(ctype, python_callable, error, onerror)";
    let code = crate::compile::compile_eval(DECORATOR)
        .map_err(|_| PyError::system_error("could not compile the callback decorator"))?;
    let code_slot = globals_slot + 1;
    let _ = roots.pin_root(crate::pycode::box_code_object_with_hidden_applevel(
        code, true,
    ));
    crate::builtins::exec_or_eval(
        roots.get(code_slot),
        roots.get(globals_slot),
        pyre_object::PY_NULL,
        true,
        pyre_object::PY_NULL,
    )
}

fn ffi_from_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(
        args,
        "from_buffer",
        &["cdecl", "python_buffer", "require_writable"],
        1,
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let (w_ctype, w_buffer) = if roots.get(base + 2).is_null() {
        (newtype::new_chara_type()?, roots.get(base + 1))
    } else {
        (
            ffi_type(
                roots.get(base),
                roots.get(base + 1),
                ACCEPT_STRING | ACCEPT_CTYPE,
            )?,
            roots.get(base + 2),
        )
    };
    let writable = if roots.get(base + 3).is_null() {
        0
    } else {
        crate::baseobjspace::int_w(roots.get(base + 3))?
    };
    func::from_buffer(&[w_ctype, w_buffer, pyre_object::w_int_new(writable)])
}

fn ffi_from_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "from_handle", &["x"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(a[1]);
    handle::from_handle(&[roots.get(value_slot)])
}

fn ffi_gc(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "gc", &["cdata", "destructor", "size"], 2)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let size = if roots.get(base + 3).is_null() {
        0
    } else {
        crate::baseobjspace::int_w(roots.get(base + 3))?
    };
    cdataobj::with_gc(roots.get(base + 1), roots.get(base + 2), size)
}

fn ffi_getctype(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "getctype", &["cdecl", "replace_with"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let w_ctype = ffi_type(
        roots.get(base),
        roots.get(base + 1),
        ACCEPT_STRING | ACCEPT_CTYPE,
    )?;
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let replacement = if roots.get(base + 2).is_null() {
        String::new()
    } else {
        crate::baseobjspace::text_w(roots.get(base + 2))?
            .trim_matches(' ')
            .to_string()
    };
    if replacement.is_empty() {
        return Ok(pyre_object::w_str_new(ct.name()));
    }
    let add_paren = replacement.starts_with('*') && ct.kind == ctypeobj::KIND_ARRAY;
    let add_space = !add_paren && !replacement.starts_with('[') && !replacement.starts_with('(');
    let at = ct.name_position as usize;
    let mut result = String::new();
    result.push_str(&ct.name()[..at]);
    if add_paren {
        result.push('(');
    }
    if add_space {
        result.push(' ');
    }
    result.push_str(&replacement);
    if add_paren {
        result.push(')');
    }
    result.push_str(&ct.name()[at..]);
    Ok(pyre_object::w_str_new(&result))
}

fn ffi_memmove(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "memmove", &["dest", "src", "n"], 3)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a[1..] {
        let _ = roots.pin_root(value);
    }
    func::memmove(&[roots.get(base), roots.get(base + 1), roots.get(base + 2)])
}

fn ffi_new_value(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "new", &["cdecl", "init"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let init_slot = roots.base();
    let init = if a[2].is_null() {
        pyre_object::w_none()
    } else {
        a[2]
    };
    let _ = roots.pin_root(init);
    let w_ctype = ffi_type(a[0], a[1], ACCEPT_STRING | ACCEPT_CTYPE)?;
    ctypeobj::newp(w_ctype, roots.get(init_slot))
}

fn ffi_new_allocator(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(
        args,
        "new_allocator",
        &["alloc", "free", "should_clear_after_alloc"],
        0,
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    let w_alloc = if roots.get(base + 1).is_null() {
        pyre_object::w_none()
    } else {
        roots.get(base + 1)
    };
    let w_free = if roots.get(base + 2).is_null() {
        pyre_object::w_none()
    } else {
        roots.get(base + 2)
    };
    let clear = if roots.get(base + 3).is_null() {
        true
    } else {
        crate::baseobjspace::int_w(roots.get(base + 3))? != 0
    };
    allocator::new_allocator(roots.get(base), w_alloc, w_free, clear)
}

fn ffi_new_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "new_handle", &["x"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(a[1]);
    handle::newp_handle(&[newtype::new_voidp_type()?, roots.get(value_slot)])
}

fn ffi_offsetof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let args = no_keyword_varargs(args, "offsetof")?;
    if args.len() < 3 {
        return Err(PyError::type_error(
            "offsetof() needs a cdecl and field or array index",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &arg in args {
        let _ = roots.pin_root(arg);
    }
    let mut w_ctype = ffi_type(
        roots.get(base),
        roots.get(base + 1),
        ACCEPT_STRING | ACCEPT_CTYPE,
    )?;
    let mut offset = 0;
    for i in 2..args.len() {
        let realized = func::direct_typeoffsetof(w_ctype, roots.get(base + i), i > 2)?;
        w_ctype = realized.0;
        offset += realized.1;
    }
    Ok(pyre_object::w_int_new(offset))
}

fn ffi_release(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "release", &["cdata"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(a[1]);
    cdataobj::enter_exit(roots.get(cdata_slot), true)?;
    Ok(pyre_object::w_none())
}

fn ffi_sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "sizeof", &["cdecl"], 1)?;
    let (w_ctype, size) = if let Some(cdata) = W_CData::from_obj(a[1]) {
        (cdata.ctype, cdataobj::cdata_sizeof(a[1])?)
    } else {
        let w_ctype = ffi_type(a[0], a[1], ACCEPT_ALL)?;
        (w_ctype, ctypeobj::ctype_arg(w_ctype)?.size)
    };
    if size < 0 {
        return Err(ffi_error(format!(
            "don't know the size of ctype '{}'",
            ctypeobj::ctype_arg(w_ctype)?.name()
        )));
    }
    Ok(pyre_object::w_int_new(size))
}

fn ffi_string(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "string", &["cdata", "maxlen"], 1)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(a[1]);
    let _ = roots.pin_root(a[2]);
    let maxlen = if roots.get(base + 1).is_null() {
        -1
    } else {
        crate::baseobjspace::int_w(roots.get(base + 1))?
    };
    ctypeobj::string(roots.get(base), maxlen)
}

fn ffi_typeof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "typeof", &["cdecl"], 1)?;
    if wrapper::W_FunctionWrapper::from_obj(a[1]).is_some() {
        return wrapper::typeof_wrapper(a[1], a[0]);
    }
    ffi_type(a[0], a[1], ACCEPT_STRING | ACCEPT_CDATA)
}

fn ffi_dlopen(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "dlopen", &["name", "flags"], 1)?;
    let flags = if a[2].is_null() {
        0
    } else {
        crate::baseobjspace::int_w(a[2])?
    };
    cdlopen::dlopen(a[0], a[1], flags)
}

fn ffi_dlclose(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "dlclose", &["lib"], 1)?;
    lib_obj::cdlopen_close(a[1])?;
    Ok(pyre_object::w_none())
}

fn ffi_integer_const(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "integer_const", &["name"], 1)?;
    let name = crate::baseobjspace::text_w(a[1])?.to_string();
    fetch_int_constant(a[0], &name)?
        .ok_or_else(|| PyError::attribute_error(format!("integer constant '{name}' not found")))
}

fn ffi_list_types(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "list_types", &[], 0)?;
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(a[0]);
    let typedefs_slot = ffi_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let structs_slot = typedefs_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let unions_slot = structs_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    for i in 0..ctx.num_typenames as isize {
        let typename = unsafe { *ctx.typenames.offset(i) };
        let name = unsafe { CStr::from_ptr(typename.name) }.to_string_lossy();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new(&name));
        unsafe {
            pyre_object::listobject::w_list_append(roots.get(typedefs_slot), roots.get(name_slot))
        };
    }
    for i in 0..ctx.num_struct_unions as isize {
        let su = unsafe { *ctx.struct_unions.offset(i) };
        let name = unsafe { CStr::from_ptr(su.name) }.to_string_lossy();
        if name.starts_with('$') {
            continue;
        }
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new(&name));
        let target = if su.flags & parse_c_type::F_UNION != 0 {
            unions_slot
        } else {
            structs_slot
        };
        unsafe { pyre_object::listobject::w_list_append(roots.get(target), roots.get(name_slot)) };
    }
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(typedefs_slot),
        roots.get(structs_slot),
        roots.get(unions_slot),
    ]))
}

fn ffi_unpack(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "unpack", &["cdata", "length"], 2)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(a[1]);
    let _ = roots.pin_root(a[2]);
    let length = crate::baseobjspace::int_w(roots.get(base + 1))?;
    cdataobj::unpack(roots.get(base), length)
}

/// `W_InitOnceLock`.
#[allow(non_camel_case_types)]
pub struct W_InitOnceLock {
    lock: *mut crate::baseobjspace::Lock,
}

#[derive(Default)]
struct InitOnceLocks {
    values: Mutex<Vec<Box<W_InitOnceLock>>>,
}

impl InitOnceLocks {
    fn add(&self) -> usize {
        let mut values = self
            .values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = values.len();
        values.push(Box::new(W_InitOnceLock {
            lock: crate::baseobjspace::allocate_lock(),
        }));
        index
    }

    fn get(&self, index: usize) -> Option<&'static W_InitOnceLock> {
        let values = self
            .values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let value = values.get(index)?;
        let ptr = std::ptr::from_ref(value.as_ref());
        Some(unsafe { &*ptr })
    }
}

struct InitOnceGuard<'a> {
    lock: &'a crate::baseobjspace::Lock,
}

impl Drop for InitOnceGuard<'_> {
    fn drop(&mut self) {
        self.lock.release();
    }
}

fn placeholder_index(ffi: &W_FFIObject, value: PyObjectRef) -> Option<usize> {
    if !unsafe { pyre_object::pyobject::is_tuple(value) }
        || unsafe { pyre_object::tupleobject::w_tuple_len(value) } != 2
    {
        return None;
    }
    let marker = unsafe { pyre_object::tupleobject::w_tuple_getitem(value, 0) }?;
    if marker != ffi.w_init_once_marker {
        return None;
    }
    let index = unsafe { pyre_object::tupleobject::w_tuple_getitem(value, 1) }?;
    crate::baseobjspace::int_w(index).ok().map(|i| i as usize)
}

/// `_init_once_elidable`.
fn init_once_elidable(
    w_ffi: PyObjectRef,
    w_tag: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let tag_slot = ffi_slot + 1;
    let _ = roots.pin_root(w_tag);
    let cache_slot = tag_slot + 1;
    let _ = roots.pin_root(ffi_arg(roots.get(ffi_slot))?.w_init_once_cache);
    crate::baseobjspace::finditem(roots.get(cache_slot), roots.get(tag_slot))
}

/// `_init_once_slowpath`.
fn init_once_slowpath(
    w_ffi: PyObjectRef,
    w_func: PyObjectRef,
    w_tag: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for value in [w_ffi, w_func, w_tag] {
        let _ = roots.pin_root(value);
    }
    let ffi = ffi_arg(roots.get(base))?;
    let locks = unsafe { &*ffi.init_once_locks };
    let fresh_index = locks.add();
    let marker_slot = base + 3;
    let _ = roots.pin_root(ffi.w_init_once_marker);
    let index_slot = marker_slot + 1;
    let _ = roots.pin_root(pyre_object::w_int_new(fresh_index as i64));
    let placeholder_slot = index_slot + 1;
    let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
        roots.get(marker_slot),
        roots.get(index_slot),
    ]));
    let selected_slot = placeholder_slot + 1;
    let _ = roots.pin_root(crate::type_methods::dict_method_setdefault(&[
        ffi_arg(roots.get(base))?.w_init_once_cache,
        roots.get(base + 2),
        roots.get(placeholder_slot),
    ])?);
    let Some(index) = placeholder_index(ffi_arg(roots.get(base))?, roots.get(selected_slot)) else {
        return Ok(roots.get(selected_slot));
    };
    let once = unsafe { &*ffi_arg(roots.get(base))?.init_once_locks }
        .get(index)
        .ok_or_else(|| PyError::system_error("init_once lock index out of range"))?;
    let lock = unsafe { &*once.lock };
    lock.acquire(true);
    let _lock_guard = InitOnceGuard { lock };
    if let Some(result) = init_once_elidable(roots.get(base), roots.get(base + 2))?
        && placeholder_index(ffi_arg(roots.get(base))?, result).is_none()
    {
        return Ok(result);
    }
    let result_slot = selected_slot + 1;
    let _ = roots.pin_root(crate::call::call_function_impl_result(
        roots.get(base + 1),
        &[],
    )?);
    crate::baseobjspace::setitem(
        ffi_arg(roots.get(base))?.w_init_once_cache,
        roots.get(base + 2),
        roots.get(result_slot),
    )?;
    Ok(roots.get(result_slot))
}

fn ffi_init_once(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "init_once", &["function", "tag"], 2)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &a {
        let _ = roots.pin_root(value);
    }
    if let Some(result) = init_once_elidable(roots.get(base), roots.get(base + 2))? {
        if placeholder_index(ffi_arg(roots.get(base))?, result).is_none() {
            return Ok(result);
        }
    }
    init_once_slowpath(roots.get(base), roots.get(base + 1), roots.get(base + 2))
}

fn errno_get(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let _ = ffi_arg(args[1])?;
    cerrno::get_errno(&[])
}

fn errno_set(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let _ = ffi_arg(args[1])?;
    cerrno::set_errno(&[args[2]])
}

#[cfg(windows)]
fn ffi_getwinerror(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_method(args, "getwinerror", &["code"], 0)?;
    if a[1].is_null() {
        cerrno::getwinerror(&[])
    } else {
        cerrno::getwinerror(&[a[1]])
    }
}

static FFI_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `_cffi_backend.FFI`.
pub fn ffi_type_object() -> PyObjectRef {
    *FFI_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend.FFI",
            init_ffi_type,
            crate::typedef::w_object(),
            <W_FFIObject as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_FFIObject as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp as usize
    }) as PyObjectRef
}

fn init_ffi_type(ns: PyObjectRef) {
    let store = |name: &str, value: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value)
    };
    store("__new__", crate::typedef::make_new_descr(ffi_new));
    store(
        "__init__",
        crate::make_builtin_function("__init__", ffi_init),
    );
    for (name, function) in [
        ("addressof", ffi_addressof as crate::gateway::BuiltinCodeFn),
        ("alignof", ffi_alignof),
        ("cast", ffi_cast),
        ("callback", ffi_callback),
        ("dlclose", ffi_dlclose),
        ("dlopen", ffi_dlopen),
        ("from_buffer", ffi_from_buffer),
        ("from_handle", ffi_from_handle),
        ("gc", ffi_gc),
        ("getctype", ffi_getctype),
        ("init_once", ffi_init_once),
        ("integer_const", ffi_integer_const),
        ("list_types", ffi_list_types),
        ("memmove", ffi_memmove),
        ("new", ffi_new_value),
        ("new_allocator", ffi_new_allocator),
        ("new_handle", ffi_new_handle),
        ("offsetof", ffi_offsetof),
        ("release", ffi_release),
        ("sizeof", ffi_sizeof),
        ("string", ffi_string),
        ("typeof", ffi_typeof),
        ("unpack", ffi_unpack),
    ] {
        store(name, crate::make_builtin_function(name, function));
    }
    #[cfg(windows)]
    store(
        "getwinerror",
        crate::make_builtin_function("getwinerror", ffi_getwinerror),
    );
    let getter = crate::make_builtin_function_with_arity("errno", errno_get, 2);
    let setter = crate::make_builtin_function_with_arity("errno", errno_set, 3);
    store(
        "errno",
        crate::typedef::make_getset_property_named(getter, setter, pyre_object::PY_NULL, "errno"),
    );
    store("CData", super::cdataobj::cdata_type());
    store("CType", super::ctypeobj::ctype_type());
    let roots = pyre_object::gc_roots::push_roots();
    let voidp_slot = roots.base();
    let _ = roots.pin_root(newtype::new_voidp_type().expect("void pointer type must build"));
    let null = ctypeobj::cast(roots.get(voidp_slot), pyre_object::w_int_new(0))
        .expect("zero must cast to void pointer");
    store("NULL", null);
    store("error", newtype::ffi_error());
    store("buffer", cbuffer::buffer_type());
    super::interp_cffi_backend::register_rtld_constants(ns);
}
