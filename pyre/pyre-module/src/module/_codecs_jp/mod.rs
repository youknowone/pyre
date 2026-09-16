//! PyPy `lib_pypy/_codecs_jp.py`:
//! `from _multibytecodec import __getcodec as getcodec`.

use pyre_object::*;

fn getcodec(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    super::_multibytecodec::getcodec(args)
}

pyre_interpreter::py_module! {
    "_codecs_jp",
    functions: {
        "getcodec" / 1 = getcodec,
    },
}
