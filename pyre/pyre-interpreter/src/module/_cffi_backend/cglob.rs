//! CFFI global-variable support — PyPy: `pypy/module/_cffi_backend/cglob.py`.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use super::{cdataobj, ctypeobj, ffi_obj, newtype};

/// `W_GlobSupport`.
#[crate::pyre_class("_cffi_backend.__FFIGlobSupport")]
#[derive(Default)]
pub struct W_GlobSupport {
    pub w_name: PyObjectRef,
    pub w_ctype: PyObjectRef,
    pub ptr: *mut u8,
    pub fetch_addr: *mut u8,
}

fn glob_arg(w_glob: PyObjectRef) -> Result<&'static mut W_GlobSupport, PyError> {
    W_GlobSupport::from_obj(w_glob)
        .ok_or_else(|| PyError::type_error("expected an FFI global-variable support object"))
}

pub fn new_glob(
    name: &str,
    w_ctype: PyObjectRef,
    ptr: *mut u8,
    fetch_addr: *mut u8,
) -> PyObjectRef {
    let _ = glob_type();
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let name_slot = ctype_slot + 1;
    let _ = roots.pin_root(pyre_object::w_str_new(name));
    let obj = W_GlobSupport::allocate_stable(W_GlobSupport {
        ptr,
        fetch_addr,
        ..Default::default()
    });
    let glob = W_GlobSupport::from_obj(obj).expect("allocate_stable returns this layout");
    glob.w_name = roots.get(name_slot);
    glob.w_ctype = roots.get(ctype_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj.cast::<u8>());
    obj
}

/// `W_GlobSupport.fetch_global_var_addr`.
pub fn fetch_global_var_addr(w_glob: PyObjectRef) -> Result<*mut u8, PyError> {
    let glob = glob_arg(w_glob)?;
    let result = if !glob.ptr.is_null() {
        glob.ptr
    } else {
        type FetchAddr = unsafe extern "C" fn() -> *mut u8;
        let fetch_addr: FetchAddr = unsafe { core::mem::transmute(glob.fetch_addr) };
        let _blocked = crate::module::thread::before_external_block();
        unsafe { fetch_addr() }
    };
    if result.is_null() {
        let name = crate::baseobjspace::text_w(glob.w_name)?.to_string();
        return Err(ffi_obj::ffi_error(format!(
            "global variable '{name}' is at address NULL"
        )));
    }
    Ok(result)
}

/// `W_GlobSupport.read_global_var`.
pub fn read_global_var(w_glob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let glob_slot = roots.base();
    let _ = roots.pin_root(w_glob);
    let ptr = fetch_global_var_addr(roots.get(glob_slot))?;
    let ct = ctypeobj::ctype_arg(glob_arg(roots.get(glob_slot))?.w_ctype)?;
    unsafe { ctypeobj::convert_to_object(ct, ptr as usize) }
}

/// `W_GlobSupport.write_global_var`.
pub fn write_global_var(w_glob: PyObjectRef, w_newvalue: PyObjectRef) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let glob_slot = roots.base();
    let _ = roots.pin_root(w_glob);
    let value_slot = glob_slot + 1;
    let _ = roots.pin_root(w_newvalue);
    let ptr = fetch_global_var_addr(roots.get(glob_slot))?;
    let ct = ctypeobj::ctype_arg(glob_arg(roots.get(glob_slot))?.w_ctype)?;
    unsafe { ctypeobj::convert_from_object(ct, ptr as usize, roots.get(value_slot)) }
}

/// `W_GlobSupport.address`.
pub fn address(w_glob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let glob_slot = roots.base();
    let _ = roots.pin_root(w_glob);
    let ptr = fetch_global_var_addr(roots.get(glob_slot))?;
    let w_ctypeptr = newtype::new_pointer_type(glob_arg(roots.get(glob_slot))?.w_ctype)?;
    Ok(cdataobj::new_cdata(ptr, w_ctypeptr))
}

static GLOB_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `_cffi_backend.__FFIGlobSupport`.
pub fn glob_type() -> PyObjectRef {
    *GLOB_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend.__FFIGlobSupport",
            |_| {},
            crate::typedef::w_object(),
            <W_GlobSupport as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_GlobSupport as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp as usize
    }) as PyObjectRef
}
