//! Raw file stream — PyPy `pypy/module/_io/interp_fileio.py` `W_FileIO`.

use pyre_object::*;

#[crate::pyre_class("_io.FileIO", cpython_heaptype, user_layout, weakrefable)]
pub struct W_FileIO {
    fd: i32,
    readable: bool,
    writable: bool,
    created: bool,
    appending: bool,
    /// `-1` until `seekable_w` probes; then `0` or `1`.
    seekable: i32,
    closefd: bool,
    w_name: PyObjectRef,
    // interp_iobase.py W_IOBase.w_dict — null until getdict.
    pub(crate) w_dict: PyObjectRef,
}

impl Default for W_FileIO {
    fn default() -> Self {
        Self {
            ob: PyObject::default(),
            fd: -1,
            readable: false,
            writable: false,
            created: false,
            appending: false,
            seekable: -1,
            closefd: true,
            w_name: PY_NULL,
            w_dict: PY_NULL,
            lifeline: PY_NULL,
        }
    }
}

#[crate::pyre_methods(
    base = super::raw_iobase_type(),
    weakrefable,
    doc = "Open a file."
)]
impl W_FileIO {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        crate::typedef::check_user_subclass(type_object(), cls)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let obj = W_FileIO::allocate_instance(
            W_FileIO::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        let obj = super::tag_io_instance(obj, pyre_object::gc_roots::shadow_stack_get(cls_slot));
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        // Public uninitialized state still lives in w_dict until the
        // remaining FileIO methods read the typed fields.
        let store = |name: &str, value: PyObjectRef| {
            crate::baseobjspace::setdictvalue_native(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                name,
                value,
            );
        };
        store("__file_fd__", pyre_object::w_int_new(-1));
        store("__file_closed__", pyre_object::w_bool_from(true));
        store("__file_closefd__", pyre_object::w_bool_from(true));
        store("__file_mode__", pyre_object::w_str_new("wb"));
        store("__file_public_mode__", pyre_object::w_str_new("wb"));
        store("__file_seekable__", pyre_object::w_none());
        store(
            "__file_blksize__",
            pyre_object::w_int_new(super::DEFAULT_BUFFER_SIZE),
        );
        for name in [
            "__file_stat_mode__",
            "__file_stat_size__",
            "__file_stat_blksize__",
        ] {
            store(name, pyre_object::w_none());
        }
        Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    }
}
