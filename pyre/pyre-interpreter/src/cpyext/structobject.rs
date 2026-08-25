//! The two counts a `_struct.Struct` block carries.
//!
//! `_struct.c` keeps `s_size` and `s_len` as fields of `PyStructObject` rather
//! than behind entry points, and the header declaring that struct is the
//! module's own -- so the way to read them is to copy the prefix and cast, as
//! `Modules/_testbuffer.c` does to size the tuple it packs one item from.  A
//! block without the two words answers that cast with whatever follows it.
//!
//! There is no `cpyext/structobject.py` upstream: PyPy leaves `_testbuffer.c`
//! unbuilt, so nothing there reaches the layout.

use super::pyobject::CPyObject;
use pyre_object::PyObjectRef;
use pyre_object::lltype::PyreClassPyTypeOf;

/// `_struct.c`'s `PyStructObject`.
///
/// Only the two counts are filled.  The three fields past them hold addresses
/// of the module's own allocations, which pyre's object does not have; they are
/// declared so that a reader casting to the whole struct stays inside the
/// block, and left at the zero `s_new` writes into `s_codes`.
#[repr(C)]
pub struct CPyStructObject {
    pub ob_base: CPyObject,
    pub s_size: isize,
    pub s_len: isize,
    pub s_codes: *mut std::ffi::c_void,
    pub s_format: *mut CPyObject,
    pub weakreflist: *mut CPyObject,
}

const _: () = {
    assert!(std::mem::offset_of!(CPyStructObject, ob_base) == 0);
    assert!(std::mem::offset_of!(CPyStructObject, s_size) == size_of::<CPyObject>());
    assert!(
        std::mem::offset_of!(CPyStructObject, s_len) == size_of::<CPyObject>() + size_of::<usize>()
    );
    assert!(size_of::<CPyStructObject>() == size_of::<CPyObject>() + 5 * size_of::<usize>());
};

/// The `_struct.Struct` class, or null before it is built.
///
/// Never builds it: this is reached while minting a mirror, and a class that
/// does not exist yet has no instances to mint one for.
fn struct_type() -> PyObjectRef {
    match crate::typedef::gettypefor(
        <crate::module::r#struct::W_Struct as PyreClassPyTypeOf>::PYTYPE,
    ) {
        Some(class) => class.as_ptr(),
        None => pyre_object::PY_NULL,
    }
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries -- the prefix
/// above for `Struct` and the classes derived from it, and 0 for every other
/// type, which asks for the plain header.
pub(super) fn basicsize(w_type: PyObjectRef) -> isize {
    let class = struct_type();
    let derived = !w_type.is_null()
        && !class.is_null()
        && unsafe { crate::baseobjspace::issubtype_w(w_type, class) };
    match derived {
        true => size_of::<CPyStructObject>() as isize,
        false => 0,
    }
}

/// Fill a freshly allocated mirror.
///
/// A `Struct` is immutable once `__init__` has run, so the two words are
/// written here and never again; one allocated by `Struct.__new__` alone
/// carries the -1 pair `s_new` leaves, which is what
/// `ENSURE_STRUCT_IS_READY` reads.
pub(super) fn attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null() || unsafe { (*tp).tp_basicsize } < size_of::<CPyStructObject>() as isize {
        return;
    }
    let w_type = match crate::typedef::r#type(w_obj) {
        Some(w_type) => w_type.as_ptr(),
        None => return,
    };
    if basicsize(w_type) == 0 {
        return;
    }
    let (size, len) = match crate::module::r#struct::W_Struct::from_obj(w_obj) {
        Some(w_struct) => w_struct.c_size_and_len(),
        // A subclass keeps its own object, so the two counts are not readable
        // off it; the pair `s_new` leaves is the answer that says so, and the
        // zeroed block would otherwise read as an empty format.
        None => (-1, -1),
    };
    unsafe {
        let block = raw as *mut CPyStructObject;
        (*block).s_size = size as isize;
        (*block).s_len = len as isize;
    }
}
