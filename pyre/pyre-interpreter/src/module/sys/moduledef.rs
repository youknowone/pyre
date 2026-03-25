//! sys module definition.
//!
//! PyPy equivalent: pypy/module/sys/

use crate::{PyNamespace, namespace_store};
use pyre_object::*;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "maxsize", w_int_new(i64::MAX));
    namespace_store(ns, "maxunicode", w_int_new(0x10FFFF));
    namespace_store(ns, "version", w_str_new("3.13.0 (pyre 0.0.1)"));
    namespace_store(
        ns,
        "platform",
        w_str_new(if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "win32"
        } else {
            "unknown"
        }),
    );
    namespace_store(
        ns,
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    namespace_store(
        ns,
        "version_info",
        w_tuple_new(vec![
            w_int_new(3),
            w_int_new(13),
            w_int_new(0),
            w_str_new("final"),
            w_int_new(0),
        ]),
    );
    // sys.modules — empty dict placeholder
    namespace_store(ns, "modules", w_dict_new());
    // sys.path — empty list placeholder
    namespace_store(ns, "path", w_list_new(vec![]));
    // sys.stdout/stderr — stubs
    namespace_store(ns, "stdout", w_none());
    namespace_store(ns, "stderr", w_none());
}
