//! unicodedata implementation — PyPy: pypy/module/unicodedata/interp_ucd.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// unicodedata module stub — provides normalize() and category().
pub fn register_module(ns: &mut DictStorage) {
    // unicodedata.normalize(form, unistr) → unistr (stub: returns input unchanged)
    crate::dict_storage_store(
        ns,
        "normalize",
        crate::make_builtin_function_with_arity(
            "normalize",
            |args| {
                if args.len() >= 2 {
                    Ok(args[1])
                } else {
                    Ok(pyre_object::w_str_new(""))
                }
            },
            2,
        ),
    );
    // unicodedata.category(chr) → str (stub: returns "Cn" = unassigned)
    crate::dict_storage_store(
        ns,
        "category",
        crate::make_builtin_function_with_arity(
            "category",
            |_| Ok(pyre_object::w_str_new("Cn")),
            1,
        ),
    );
    // unicodedata.name(chr, default=None) → str
    crate::dict_storage_store(
        ns,
        "name",
        crate::make_builtin_function("name", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("no such name"))
            }
        }),
    );
    // unicodedata.lookup(name) → chr
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity(
            "lookup",
            |_| Err(crate::PyError::key_error("character not found")),
            1,
        ),
    );
    // unicodedata.decimal(chr, default=None) → int
    crate::dict_storage_store(
        ns,
        "decimal",
        crate::make_builtin_function("decimal", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a decimal"))
            }
        }),
    );
    // unicodedata.numeric(chr, default=None) → float
    crate::dict_storage_store(
        ns,
        "numeric",
        crate::make_builtin_function("numeric", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a numeric character"))
            }
        }),
    );
    // unicodedata.unidata_version
    crate::dict_storage_store(ns, "unidata_version", pyre_object::w_str_new("15.1.0"));
    // unicodedata.ucd_3_2_0 — alias for the module itself (used by IDNA)
    // We store a sentinel; os_helper only checks that the module imported.
}
