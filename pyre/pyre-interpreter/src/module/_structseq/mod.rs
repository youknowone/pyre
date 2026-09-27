//! `_structseq` module — PyPy: `lib_pypy/_structseq.py`.
//!
//! The whole module is the app-level source, bundled unchanged as
//! `_structseq_app.py`.  Interpreter-level callers build their types through
//! [`crate::_structseq::make_struct_seq`], which evaluates the same class
//! statement an app-level `class X(metaclass=structseqtype)` would.

crate::py_module! {
    "_structseq",
    appleveldefs: {
        "_structseq_app.py" => [
            "structseqfield",
            "structseqtype",
            "structseq_new",
            "structseq_reduce",
            "structseq_setattr",
            "structseq_repr",
            "SimpleNamespace",
        ],
    },
}
