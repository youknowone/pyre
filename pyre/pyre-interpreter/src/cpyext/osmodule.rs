//! `os.fspath` for C -- `Include/osmodule.h`.

use super::pyobject::{self, CPyObject};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyOS_FSPath(path: *mut CPyObject) -> *mut CPyObject {
    let Some(path) = super::object::argument(path) else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::module::posix::interp_posix::fspath(path))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyOS_FSPath as *const ());
}
