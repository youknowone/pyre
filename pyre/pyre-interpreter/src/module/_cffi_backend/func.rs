//! The module-level functions — PyPy:
//! `pypy/module/_cffi_backend/func.py`.

use crate::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj;
use super::newtype;

/// `func.py newp`.
pub fn newp(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_init = args.get(1).copied().unwrap_or_else(pyre_object::w_none);
    ctypeobj::newp(args[0], w_init)
}

/// `func.py cast`.
pub fn cast(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    ctypeobj::cast(args[0], args[1])
}

/// `func.py typeof`.
pub fn typeof_(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(cdataobj::cdata_arg(args[0])?.ctype)
}

/// `func.py sizeof` — a cdata reports what it owns, a ctype its own size.
pub fn sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_obj = args[0];
    let (size, name) = if W_CData::from_obj(w_obj).is_some() {
        let size = cdataobj::cdata_sizeof(w_obj)?;
        let ct = ctypeobj::ctype_at(cdataobj::cdata_arg(w_obj)?.ctype)
            .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
        (size, ct.name())
    } else if let Some(ct) = ctypeobj::ctype_at(w_obj) {
        (ct.size, ct.name())
    } else {
        return Err(PyError::type_error("expected a 'cdata' or 'ctype' object"));
    };
    if size < 0 {
        return Err(PyError::value_error(format!(
            "ctype '{name}' is of unknown size"
        )));
    }
    Ok(pyre_object::w_int_new(size))
}

/// `func.py alignof`.
pub fn alignof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_int_new(
        ctypeobj::ctype_arg(args[0])?.alignof()?,
    ))
}

/// `func.py getcname`.
pub fn getcname(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(args[0])?;
    let replace_with = crate::baseobjspace::text_w(args[1])?;
    let (name, _) = ct.insert_name(&replace_with, 0);
    Ok(pyre_object::w_str_new(&name))
}

/// `func.py string`.
pub fn string(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let maxlen = match args.get(1) {
        Some(&w_maxlen) => crate::baseobjspace::int_w(w_maxlen)?,
        None => -1,
    };
    ctypeobj::string(args[0], maxlen)
}

/// `func.py unpack`.
pub fn unpack(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    cdataobj::unpack(args[0], crate::baseobjspace::int_w(args[1])?)
}

/// `func.py typeoffsetof`.
pub fn typeoffsetof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(args[0])?;
    let w_field_or_index = args[1];
    // `W_CType.direct_typeoffsetof` tries a field name first and falls back
    // to an index.
    if unsafe { pyre_object::unicodeobject::is_str(w_field_or_index) } {
        let following = match args.get(2) {
            Some(&w_following) => crate::baseobjspace::int_w(w_following)?,
            None => 0,
        };
        // `W_CTypePointer.typeoffsetof_field` reads the item's field, but
        // only for the first element of the array the pointer may be.
        let owner = if ct.is_struct_or_union() {
            ct
        } else if ct.kind == ctypeobj::KIND_POINTER && following == 0 {
            ctypeobj::ctype_arg(ct.ctitem)?
        } else {
            return Err(PyError::type_error(
                "with a field name argument, expected a struct or union ctype",
            ));
        };
        if !owner.is_struct_or_union() {
            return Err(PyError::type_error(
                "with a field name argument, expected a struct or union ctype",
            ));
        }
        let fieldname = crate::baseobjspace::text_w(w_field_or_index)?;
        let (w_ctype, offset) = super::ctypestruct::typeoffsetof_field(owner, fieldname)?;
        let roots = pyre_object::gc_roots::push_roots();
        let field_slot = roots.base();
        let _ = roots.pin_root(w_ctype);
        let w_offset = pyre_object::w_int_new(offset);
        return Ok(pyre_object::w_tuple_new(vec![
            roots.get(field_slot),
            w_offset,
        ]));
    }
    let Ok(index) = crate::baseobjspace::int_w(w_field_or_index) else {
        return Err(PyError::type_error("field name or array index expected"));
    };
    // `W_CTypePointer.typeoffsetof_index`, which an array reaches through
    // its own pointer type.
    let ct = match ct.kind {
        ctypeobj::KIND_POINTER => ct,
        ctypeobj::KIND_ARRAY => ctypeobj::ctype_arg(ct.ctptr)?,
        _ => {
            return Err(PyError::type_error(
                "with an integer argument, expected an array or pointer ctype",
            ));
        }
    };
    let ctitem = super::ctypeptr::item_of(ct)?;
    if ctitem.size < 0 {
        return Err(PyError::type_error("pointer to opaque"));
    }
    let offset = index
        .checked_mul(ctitem.size)
        .ok_or_else(|| PyError::overflow_error("array offset would overflow a ssize_t"))?;
    let roots = pyre_object::gc_roots::push_roots();
    let item_slot = roots.base();
    let _ = roots.pin_root(ctitem.as_object());
    let w_offset = pyre_object::w_int_new(offset);
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(item_slot),
        w_offset,
    ]))
}

/// `func.py rawaddressof`.
pub fn rawaddressof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(args[0])?;
    if ct.kind != ctypeobj::KIND_POINTER {
        return Err(PyError::type_error("expected a pointer ctype"));
    }
    let cdata = cdataobj::cdata_arg(args[1])?;
    let source_ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if !source_ct.is_ptr_or_array() && !source_ct.is_struct_or_union() {
        return Err(PyError::type_error(
            "expected a cdata struct/union/array/pointer object",
        ));
    }
    let offset = match args.get(2) {
        Some(&w_offset) => crate::baseobjspace::int_w(w_offset)?,
        None => 0,
    };
    let ptr = unsafe { cdata.ptr.offset(offset as isize) };
    Ok(cdataobj::new_cdata(ptr, args[0]))
}

/// `func.py memmove`.
pub fn memmove(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // A `bytes` source hands back an interior pointer, and the size argument's
    // `__int__` runs arbitrary Python, so the source is pinned across it.
    let roots = pyre_object::gc_roots::push_roots();
    let src_slot = roots.base();
    let _ = roots.pin_root(args[1]);
    let n = crate::baseobjspace::int_w(args[2])?;
    if n < 0 {
        return Err(PyError::value_error("negative size"));
    }
    let dest = writable_address(args[0])?;
    let src = readable_address(roots.get(src_slot))?;
    unsafe { std::ptr::copy(src, dest, n as usize) };
    Ok(pyre_object::w_none())
}

fn writable_address(w_obj: PyObjectRef) -> Result<*mut u8, PyError> {
    let cdata = W_CData::from_obj(w_obj)
        .ok_or_else(|| PyError::type_error("expected a cdata pointer, got a non-cdata object"))?;
    Ok(cdata.ptr)
}

fn readable_address(w_obj: PyObjectRef) -> Result<*const u8, PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes(w_obj) } {
        return Ok(unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) }.as_ptr());
    }
    Ok(writable_address(w_obj)?.cast_const())
}

/// `func.py release` — `W_CData.enter_exit(exit_now=True)`.
pub fn release(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    cdataobj::enter_exit(args[0], true)?;
    Ok(pyre_object::w_none())
}

/// `func.py _get_types` — the two types cffi's Python half checks against.
pub fn get_types(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(cdataobj::cdata_type());
    let ctype_slot = cdata_slot + 1;
    let _ = roots.pin_root(ctypeobj::ctype_type());
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(cdata_slot),
        roots.get(ctype_slot),
    ]))
}

// ── the `newtype.py` entry points ───────────────────────────────────────

/// `newtype.py new_primitive_type`.
pub fn new_primitive_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let name = crate::baseobjspace::text_w(args[0])?;
    newtype::new_primitive_type(&name)
}

/// `newtype.py new_void_type`.
pub fn new_void_type(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_void_type())
}

/// `newtype.py new_pointer_type`.
pub fn new_pointer_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    newtype::new_pointer_type(args[0])
}

/// `newtype.py new_array_type`.
pub fn new_array_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let length = newtype::array_length_arg(args[1])?;
    newtype::new_array_type(args[0], length)
}

/// `newtype.py new_struct_type`.
pub fn new_struct_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_struct_type(crate::baseobjspace::text_w(
        args[0],
    )?))
}

/// `newtype.py new_union_type`.
pub fn new_union_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_union_type(crate::baseobjspace::text_w(
        args[0],
    )?))
}

/// `newtype.py complete_struct_or_union` — the third argument is ignored, as
/// its name says, and the last four carry defaults.
pub fn complete_struct_or_union(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let int_arg = |i: usize, default: i64| -> Result<i64, PyError> {
        match args.get(i) {
            Some(&w) => crate::baseobjspace::int_w(w),
            None => Ok(default),
        }
    };
    newtype::complete_struct_or_union(
        args[0],
        args[1],
        int_arg(3, -1)?,
        int_arg(4, -1)?,
        int_arg(5, 0)?,
        int_arg(6, 0)?,
    )?;
    Ok(pyre_object::w_none())
}

/// `newtype.py new_enum_type`.
pub fn new_enum_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let name = crate::baseobjspace::text_w(args[0])?;
    newtype::new_enum_type(name, args[1], args[2], args[3])
}

/// `newtype.py new_function_type`.
pub fn new_function_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ellipsis = match args.get(2) {
        Some(&w) => crate::baseobjspace::int_w(w)? != 0,
        None => false,
    };
    let abi = match args.get(3) {
        Some(&w) => crate::baseobjspace::int_w(w)?,
        None => super::interp_cffi_backend::default_abi() as i64,
    };
    newtype::new_function_type(args[0], args[1], ellipsis, abi)
}

/// `libraryobj.py load_library`.
pub fn load_library(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let flags = match args.get(1) {
        Some(&w) => crate::baseobjspace::int_w(w)?,
        None => 0,
    };
    super::libraryobj::load_library(args[0], flags)
}
