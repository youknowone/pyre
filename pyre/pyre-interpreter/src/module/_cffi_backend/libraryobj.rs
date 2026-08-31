//! `_cffi_backend.CLibrary` — PyPy:
//! `pypy/module/_cffi_backend/libraryobj.py`.
//!
//! ABI mode's whole loader: `load_library` opens a shared object and hands
//! back an object that resolves symbols in it against a ctype the caller
//! supplies.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use super::cdataobj;
use super::ctypeobj;

/// `libraryobj.py W_Library`.
#[crate::pyre_class("_cffi_backend.CLibrary")]
#[derive(Default)]
pub struct W_Library {
    /// `W_Library.name`.
    pub w_name: PyObjectRef,
    /// `W_Library.handle`, as the index of the host loader's own record.
    /// Zero once the library has been closed.
    pub handle: i64,
    /// Whether closing this object closes the library: a `void *` handle
    /// handed in was opened by someone else, and is left alone.
    pub autoclose: i64,
}

impl W_Library {
    /// `W_Library.check_closed`.
    fn check_closed(&self) -> Result<(), PyError> {
        if self.handle == 0 {
            return Err(PyError::value_error(format!(
                "library '{}' has already been closed",
                self.name()
            )));
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        if self.w_name.is_null() {
            return "";
        }
        unsafe { pyre_object::w_str_get_value(self.w_name) }
    }
}

/// `@unwrap_spec(self=W_Library)`.
fn library_arg(w_self: PyObjectRef) -> Result<&'static mut W_Library, PyError> {
    W_Library::from_obj(w_self).ok_or_else(|| {
        PyError::type_error(format!(
            "expected a CLibrary object, got '{}'",
            crate::type_methods::arg_type_name(w_self)
        ))
    })
}

/// `libraryobj.py load_library`.
pub fn load_library(w_filename: PyObjectRef, flags: i64) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = roots.base();
    let _ = roots.pin_root(pyre_object::PY_NULL);
    let (name, handle, autoclose) = super::misc::dlopen_w(w_filename, flags)?;
    roots.set(name_slot, pyre_object::w_str_new(&name));
    let obj = W_Library::allocate_stable(W_Library {
        handle: handle as i64,
        autoclose: i64::from(autoclose),
        ..Default::default()
    });
    W_Library::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_name = roots.get(name_slot);
    // The library is born old-gen and the name it just took is young, so the
    // barrier has to run again after that write.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj.cast::<u8>());
    Ok(obj)
}

/// `W_Library._finalize_` and `close_lib`.
pub fn close_lib(lib: &mut W_Library) {
    let handle = lib.handle;
    if handle != 0 {
        lib.handle = 0;
        if lib.autoclose != 0 {
            drop_library(handle as usize);
        }
    }
}

/// The sweep-time half of `_finalize_`, for a library nothing names any more.
///
/// # Safety
/// `obj` must be a GC-dead `W_Library`.
pub unsafe fn w_library_dealloc(obj: PyObjectRef) {
    close_lib(unsafe { &mut *(obj as *mut W_Library) });
}

#[cfg(all(feature = "host_env", any(unix, windows)))]
pub(crate) fn drop_library(handle: usize) {
    rustpython_host_env::ctypes::drop_library(handle);
}

#[cfg(not(all(feature = "host_env", any(unix, windows))))]
pub(crate) fn drop_library(_handle: usize) {}

/// `rdynload.dlsym`, split the way the host loader splits it: a function
/// address and a data address are looked up differently on Windows.
pub(crate) fn dlsym(handle: usize, name: &str, is_function: bool) -> Option<usize> {
    #[cfg(all(feature = "host_env", any(unix, windows)))]
    {
        let symbol = name.as_bytes();
        let address = if is_function {
            rustpython_host_env::ctypes::lookup_function_symbol_addr(handle, symbol)
        } else {
            rustpython_host_env::ctypes::lookup_data_symbol_addr(handle, symbol)
        };
        // A resolver that itself returns NULL leaves the load reporting
        // success with address 0, which is not a symbol.
        address.ok().filter(|&a| a != 0)
    }
    #[cfg(not(all(feature = "host_env", any(unix, windows))))]
    {
        let _ = (handle, name, is_function);
        None
    }
}

/// `W_Library.load_function`.
fn load_function(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let lib = library_arg(args[0])?;
    lib.check_closed()?;
    let w_ctype = args[1];
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if !ct.is_ptr_or_array() {
        return Err(PyError::type_error(format!(
            "function or pointer or array cdata expected, got '{}'",
            ct.name()
        )));
    }
    let name = crate::baseobjspace::text_w(args[2])?;
    let Some(address) = dlsym(lib.handle as usize, name, true) else {
        return Err(PyError::attribute_error(format!(
            "function/symbol '{name}' not found in library '{}'",
            lib.name()
        )));
    };
    // An unbounded array decays to the pointer it was built from.
    let w_ctype = if ct.kind == ctypeobj::KIND_ARRAY && ct.length < 0 {
        ct.ctptr
    } else {
        w_ctype
    };
    Ok(cdataobj::new_cdata(address as *mut u8, w_ctype))
}

/// `W_Library.read_variable`.
fn read_variable(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let lib = library_arg(args[0])?;
    lib.check_closed()?;
    let ct = ctypeobj::ctype_arg(args[1])?;
    let name = crate::baseobjspace::text_w(args[2])?;
    let address = variable_address(lib, name)?;
    unsafe { ctypeobj::convert_to_object(ct, address as usize) }
}

/// `W_Library.write_variable`.
fn write_variable(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // The conversion runs arbitrary Python, so the value is read back out of
    // its slot rather than out of the native argument array.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(args[3]);
    let lib = library_arg(args[0])?;
    lib.check_closed()?;
    let ct = ctypeobj::ctype_arg(args[1])?;
    let name = crate::baseobjspace::text_w(args[2])?;
    let address = variable_address(lib, name)?;
    unsafe {
        ctypeobj::convert_from_object(ct, address.cast_mut() as usize, roots.get(value_slot))?
    };
    Ok(pyre_object::w_none())
}

/// The `dlsym` both variable accessors share, whose miss is a KeyError.
fn variable_address(lib: &W_Library, name: &str) -> Result<*const u8, PyError> {
    dlsym(lib.handle as usize, name, false)
        .map(|address| address as *const u8)
        .ok_or_else(|| {
            PyError::key_error(format!(
                "variable '{name}' not found in library '{}'",
                lib.name()
            ))
        })
}

/// `W_Library.repr`.
fn library_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let lib = library_arg(args[0])?;
    Ok(pyre_object::w_str_new(&format!(
        "<clibrary '{}'>",
        lib.name()
    )))
}

/// `W_Library.close_lib`.
fn library_close(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    close_lib(library_arg(args[0])?);
    Ok(pyre_object::w_none())
}

// ── the Python type ─────────────────────────────────────────────────────

static CLIBRARY_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `_cffi_backend.CLibrary`.
pub fn clibrary_type() -> PyObjectRef {
    *CLIBRARY_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend.CLibrary",
            init_clibrary_type,
            crate::typedef::w_object(),
            <W_Library as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Library as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp as usize
    }) as PyObjectRef
}

fn init_clibrary_type(ns: PyObjectRef) {
    let store = |name: &str, value: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value)
    };
    for (name, f, arity) in [
        (
            "__repr__",
            library_repr as crate::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("load_function", load_function, 3),
        ("read_variable", read_variable, 3),
        ("write_variable", write_variable, 4),
        ("close_lib", library_close, 1),
    ] {
        store(
            name,
            crate::make_builtin_function_with_arity(name, f, arity),
        );
    }
}
