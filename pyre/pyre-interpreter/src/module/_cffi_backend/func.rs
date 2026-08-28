//! The module-level functions — PyPy:
//! `pypy/module/_cffi_backend/func.py`.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use super::cdataobj::{self, W_CData};
use super::ctypeobj;
use super::newtype;

/// `func.py OffsetInBytes`.
#[crate::pyre_class("_cffi_backend._OffsetInBytes")]
#[derive(Default)]
pub struct OffsetInBytes {
    /// `OffsetInBytes.bytes`.
    pub w_bytes: PyObjectRef,
    /// `OffsetInBytes.offset`.
    pub offset: i64,
}

static OFFSET_IN_BYTES_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

fn offset_in_bytes_type() -> PyObjectRef {
    *OFFSET_IN_BYTES_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend._OffsetInBytes",
            |_| {},
            crate::typedef::w_object(),
            <OffsetInBytes as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<OffsetInBytes as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp as usize
    }) as PyObjectRef
}

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
    let following = match args.get(2) {
        Some(&w_following) => crate::baseobjspace::int_w(w_following)? != 0,
        None => false,
    };
    let (w_ctype, offset) = direct_typeoffsetof(args[0], args[1], following)?;
    let roots = pyre_object::gc_roots::push_roots();
    let field_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let w_offset = pyre_object::w_int_new(offset);
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(field_slot),
        w_offset,
    ]))
}

/// `W_CType.direct_typeoffsetof`, shared by the module-level function and
/// `W_FFIObject._more_{addressof,offsetof}`.
pub fn direct_typeoffsetof(
    w_ctype: PyObjectRef,
    w_field_or_index: PyObjectRef,
    following: bool,
) -> Result<(PyObjectRef, i64), PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if unsafe { pyre_object::unicodeobject::is_str(w_field_or_index) } {
        // `W_CTypePointer.typeoffsetof_field` reads the item's field, but
        // only for the first element of the array the pointer may be.
        let owner = if ct.is_struct_or_union() {
            ct
        } else if ct.kind == ctypeobj::KIND_POINTER && !following {
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
        return super::ctypestruct::typeoffsetof_field(owner, fieldname);
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
    Ok((ctitem.as_object(), offset))
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

/// `func.py from_buffer`.
pub fn from_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_ctype = args[0];
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if !ct.is_ptr_or_array() {
        return Err(PyError::type_error(format!(
            "expected a poiunter or array ctype, got '{}'",
            ct.name()
        )));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.base();
    let _ = roots.pin_root(args[1]);
    if unsafe { pyre_object::unicodeobject::is_str(roots.get(object_slot)) } {
        return Err(PyError::type_error(
            "from_buffer() cannot return the address of a unicode object",
        ));
    }
    let require_writable = match args.get(2) {
        Some(&w) => crate::baseobjspace::int_w(w)? != 0,
        None => false,
    };
    let (ptr, buffersize, owner) = if require_writable {
        let (data, owner, _) = unsafe { crate::builtins::fileio_writebuf(roots.get(object_slot)) }?;
        (data.as_mut_ptr(), data.len() as i64, owner)
    } else {
        let data = unsafe { crate::builtins::acquire_readbuf(roots.get(object_slot)) }?;
        (
            data.as_ptr().cast_mut(),
            data.len() as i64,
            roots.get(object_slot),
        )
    };
    let owner_slot = object_slot + 1;
    let _ = roots.pin_root(owner);
    let held = unsafe { crate::builtins::buffer_export_incref(roots.get(owner_slot)) };

    let arraylength = if ct.kind != ctypeobj::KIND_ARRAY {
        buffersize
    } else if ct.length >= 0 {
        if buffersize < ct.size {
            if held {
                unsafe { crate::builtins::buffer_export_decref(roots.get(owner_slot)) };
            }
            return Err(PyError::value_error(format!(
                "buffer is too small ({buffersize} bytes) for '{}' ({} bytes)",
                ct.name(),
                ct.size
            )));
        }
        ct.length
    } else {
        let itemsize = super::ctypeptr::item_of(ct)?.size;
        if itemsize == 1 {
            buffersize
        } else if itemsize > 0 {
            buffersize / itemsize
        } else {
            if held {
                unsafe { crate::builtins::buffer_export_decref(roots.get(owner_slot)) };
            }
            return Err(PyError::new(
                crate::PyErrorKind::ZeroDivisionError,
                format!(
                    "from_buffer('{}', ..): the actual length of the array cannot be computed",
                    ct.name()
                ),
            ));
        }
    };
    Ok(cdataobj::new_cdata_from_buffer(
        ptr,
        arraylength,
        w_ctype,
        roots.get(object_slot),
        roots.get(owner_slot),
        held,
    ))
}

/// `func.py gcp`.
pub fn gcp(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(args[0]);
    let destructor_slot = cdata_slot + 1;
    let _ = roots.pin_root(args[1]);
    let size = match args.get(2) {
        Some(&w_size) => crate::baseobjspace::int_w(w_size)?,
        None => 0,
    };
    cdataobj::with_gc(roots.get(cdata_slot), roots.get(destructor_slot), size)
}

/// `func.py offset_in_bytes`.
pub fn offset_in_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes(args[0]) } {
        return Err(PyError::type_error(format!(
            "must be bytes, not {}",
            crate::type_methods::arg_type_name(args[0])
        )));
    }
    let _ = offset_in_bytes_type();
    let roots = pyre_object::gc_roots::push_roots();
    let bytes_slot = roots.base();
    let _ = roots.pin_root(args[0]);
    let offset = crate::baseobjspace::int_w(args[1])?;
    let obj = OffsetInBytes::allocate_stable(OffsetInBytes {
        offset,
        ..Default::default()
    });
    OffsetInBytes::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_bytes = roots.get(bytes_slot);
    // The wrapper is born old-gen and the bytes object may be young.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj.cast::<u8>());
    Ok(obj)
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
    let mut dest_buffer = None;
    let dest = if let Some(cdata) = W_CData::from_obj(args[0]) {
        unsafe_escaping_ptr_for_ptr_or_array(cdata)?
    } else {
        dest_buffer = Some(unsafe { crate::builtins::WritableBuffer::acquire(args[0]) }?);
        unsafe {
            dest_buffer
                .as_mut()
                .expect("just filled")
                .as_mut_slice()
                .as_mut_ptr()
        }
    };
    let mut source_buffer = None;
    let src = if let Some(cdata) = W_CData::from_obj(roots.get(src_slot)) {
        unsafe_escaping_ptr_for_ptr_or_array(cdata)?.cast_const()
    } else {
        let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(roots.get(src_slot))? else {
            return Err(PyError::type_error("expected a readable buffer"));
        };
        let ptr = buffer.as_bytes().as_ptr();
        source_buffer = Some(buffer);
        ptr
    };
    unsafe { std::ptr::copy(src, dest, n as usize) };
    if let Some(buffer) = source_buffer {
        buffer.release();
    }
    drop(dest_buffer);
    Ok(pyre_object::w_none())
}

/// `func.py unsafe_escaping_ptr_for_ptr_or_array`.
fn unsafe_escaping_ptr_for_ptr_or_array(cdata: &W_CData) -> Result<*mut u8, PyError> {
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if !ct.has(ctypeobj::F_NONFUNC_POINTER_OR_ARRAY) {
        return Err(PyError::type_error(format!(
            "expected a pointer or array ctype, got '{}'",
            ct.name()
        )));
    }
    Ok(cdata.ptr)
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
