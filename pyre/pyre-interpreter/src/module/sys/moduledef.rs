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
    // sys.stdout/stderr/stdin — stubs
    namespace_store(ns, "stdout", w_none());
    namespace_store(ns, "stderr", w_none());
    namespace_store(ns, "stdin", w_none());
    // sys._getframe — returns a stub frame object with f_locals/f_globals
    namespace_store(
        ns,
        "_getframe",
        crate::builtin_code_new("_getframe", |_| {
            let frame_type = crate::typedef::getobjecttype();
            let frame = pyre_object::w_instance_new(frame_type);
            let _ = crate::baseobjspace::setattr(frame, "f_locals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_globals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_code", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_back", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_lineno", w_int_new(0));
            Ok(frame)
        }),
    );
    // sys.exc_info — stub
    namespace_store(
        ns,
        "exc_info",
        crate::builtin_code_new("exc_info", |_| {
            Ok(w_tuple_new(vec![w_none(), w_none(), w_none()]))
        }),
    );
    // sys.flags — stub object with named attributes
    {
        let flags_type = crate::typedef::getobjecttype();
        let flags = pyre_object::w_instance_new(flags_type);
        let _ = crate::baseobjspace::setattr(flags, "debug", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "inspect", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "interactive", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "optimize", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "dont_write_bytecode", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "no_user_site", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "no_site", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "ignore_environment", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "verbose", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "bytes_warning", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "quiet", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "hash_randomization", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "isolated", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "dev_mode", w_bool_from(false));
        let _ = crate::baseobjspace::setattr(flags, "utf8_mode", w_int_new(1));
        let _ = crate::baseobjspace::setattr(flags, "warn_default_encoding", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "safe_path", w_bool_from(false));
        let _ = crate::baseobjspace::setattr(flags, "int_max_str_digits", w_int_new(4300));
        let _ = crate::baseobjspace::setattr(flags, "context_aware_warnings", w_bool_from(false));
        namespace_store(ns, "flags", flags);
    }
    // sys.getdefaultencoding
    namespace_store(
        ns,
        "getdefaultencoding",
        crate::builtin_code_new("getdefaultencoding", |_| Ok(w_str_new("utf-8"))),
    );
    // sys.getrecursionlimit / setrecursionlimit
    namespace_store(
        ns,
        "getrecursionlimit",
        crate::builtin_code_new("getrecursionlimit", |_| Ok(w_int_new(1000))),
    );
    namespace_store(
        ns,
        "setrecursionlimit",
        crate::builtin_code_new("setrecursionlimit", |_| Ok(w_none())),
    );
    // sys.intern
    namespace_store(
        ns,
        "intern",
        crate::builtin_code_new("intern", |args| {
            Ok(if args.is_empty() {
                w_str_new("")
            } else {
                args[0]
            })
        }),
    );
    // sys.implementation
    namespace_store(ns, "implementation", w_none());
    // sys.hash_info
    namespace_store(ns, "hash_info", w_none());
    // sys.float_info
    namespace_store(ns, "float_info", w_none());
    // sys.int_info
    namespace_store(ns, "int_info", w_none());
    // sys.executable
    namespace_store(ns, "executable", w_str_new("pyre"));
    // sys.prefix / exec_prefix
    namespace_store(ns, "prefix", w_str_new(""));
    namespace_store(ns, "exec_prefix", w_str_new(""));
    namespace_store(ns, "base_prefix", w_str_new(""));
    namespace_store(ns, "base_exec_prefix", w_str_new(""));
    // sys.argv
    namespace_store(ns, "argv", w_list_new(vec![]));
    // sys.warnoptions
    namespace_store(ns, "warnoptions", w_list_new(vec![]));
    // sys.addaudithook
    namespace_store(
        ns,
        "addaudithook",
        crate::builtin_code_new("addaudithook", |_| Ok(w_none())),
    );
}
