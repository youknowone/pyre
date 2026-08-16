//! PyPy `lib_pypy/_codecs_jp.py`:
//! `from _multibytecodec import __getcodec as getcodec`.

use pyre_object::*;

fn getcodec(args: &[PyObjectRef]) -> crate::PyResult {
    super::_multibytecodec::getcodec(args)
}

crate::py_module! {
    "_codecs_jp",
    functions: {
        "getcodec" / 1 = getcodec,
    },
}
