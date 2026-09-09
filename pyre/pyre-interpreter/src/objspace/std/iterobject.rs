//! `pypy/objspace/std/iterobject.py` sequence iterator declarations.

use indexmap::IndexMap;
use pyre_object::typedef::{TypeDef, TypeDefValue};

/// W_AbstractSeqIterObject.typedef's prebuilt declarations. The bodies still
/// live in baseobjspace until the existing iterator implementation is moved.
pub(crate) fn rawdict() -> IndexMap<String, TypeDefValue> {
    let mut entries = IndexMap::new();
    // [3.14-spec] PySeqIter_Type's public name/doc are "iterator"/None,
    // unlike W_AbstractSeqIterObject.typedef's "sequenceiterator"/iter doc.
    // Measured on the pinned free-threaded CPython v3.14.6 with a
    // __getitem__-only class (PySeqIter_Type in Objects/iterobject.c).
    entries.insert("__doc__".into(), TypeDefValue::None);
    for (name, func, arity, text_sig) in [
        (
            "__iter__",
            crate::baseobjspace::iter_self_method as crate::gateway::BuiltinCodeFn,
            1,
            "($self, /)",
        ),
        (
            "__next__",
            crate::baseobjspace::iter_next_method,
            1,
            "($self, /)",
        ),
        (
            "__reduce__",
            crate::baseobjspace::seq_iter_reduce_method,
            1,
            "($self, /)",
        ),
        (
            "__length_hint__",
            crate::baseobjspace::seq_iter_length_hint_method,
            1,
            "($self, /)",
        ),
        (
            "__setstate__",
            crate::baseobjspace::seq_iter_setstate_method,
            2,
            "($self, object, /)",
        ),
    ] {
        let code = crate::gateway::builtin_code_new_with_arity(name, func, arity);
        let gateway = crate::gateway::interp2app(code);
        unsafe {
            (*(gateway as *mut crate::gateway::interp2app))._explicit_text_sig = Some(text_sig);
            entries.insert(name.into(), TypeDefValue::root(gateway));
        }
    }
    entries
}

/// Identity of W_AbstractSeqIterObject.typedef, not a W_TypeObject cache.
pub(crate) fn typedef() -> *const TypeDef {
    static TYPEDEF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TYPEDEF.get_or_init(|| unsafe {
        TypeDef::from_rawdict(
            "iterator",
            vec![],
            rawdict(),
            &pyre_object::iterobject::SEQ_ITER_TYPE,
        ) as usize
    }) as *const TypeDef
}
