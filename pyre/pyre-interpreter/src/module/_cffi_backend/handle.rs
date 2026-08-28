//! CFFI object handles — PyPy: `pypy/module/_cffi_backend/handle.py`.

use crate::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj;
use super::ctypeobj;

/// `handle.py newp_handle`.
pub fn newp_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(args[0])?;
    if ct.kind != ctypeobj::KIND_POINTER || !ct.has(ctypeobj::F_VOID_PTR) {
        return Err(PyError::type_error(format!(
            "needs 'void *', got '{}'",
            ct.name()
        )));
    }
    let obj = cdataobj::new_cdata_handle(args[0], args[1]);
    let ptr = super::hide_reveal::hide_object(obj);
    cdataobj::cdata_arg(obj)?.ptr = ptr;
    Ok(obj)
}

/// `handle.py from_handle` and `_reveal`.
pub fn from_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdataobj::cdata_arg(args[0])?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if ct.kind != ctypeobj::KIND_POINTER || !ct.has(ctypeobj::F_VOIDCHAR_PTR) {
        return Err(PyError::type_error(format!(
            "expected a 'cdata' object with a 'void *' out of new_handle(), got '{}'",
            ct.name()
        )));
    }
    if cdata.ptr.is_null() {
        return Err(PyError::runtime_error(
            "cannot use from_handle() on NULL pointer",
        ));
    }
    let Some(w_handle) = super::hide_reveal::reveal_object(cdata.ptr) else {
        return Err(PyError::system_error(
            "ffi.from_handle(): dead or bogus object handle",
        ));
    };
    Ok(cdataobj::cdata_arg(w_handle)?.w_keepalive)
}
