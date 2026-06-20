//! unicodedata module — PyPy: `pypy/module/unicodedata/`.
//!
//! Stub providing `normalize` / `category` / `name` / `lookup` /
//! `decimal` / `numeric` — enough to let `import unicodedata` succeed.
//! `category` returns `"Cn"` (unassigned) for every code point;
//! `normalize` is identity; `name` / `decimal` / `numeric` return the
//! caller-supplied default if any, else raise.

use pyre_object::*;

crate::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new("15.1.0"),
    },
    functions: {
        "normalize" / 2 = |args| Ok(args.get(1).copied().unwrap_or_else(|| w_str_new(""))),
        "category"  / 1 = |_| Ok(w_str_new("Cn")),
        "name"      / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("no such name")),
        "lookup"    / 1 = |_| Err(crate::PyError::key_error("character not found")),
        "decimal"   / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("not a decimal")),
        "numeric"   / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("not a numeric character")),
    },
    extra_init: |ns| {
        // `unicodedata.ucd_3_2_0` — a `UCD` instance pinned to the Unicode
        // 3.2 database (used by `stringprep`).  pyre carries no historical
        // tables, so it reuses the module's stub callables.  Functions live
        // in the instance __dict__, so attribute access returns them
        // unbound — `ucd_3_2_0.category(ch)` dispatches exactly like the
        // module-level `category(ch)`.
        let ucd_ty = crate::typedef::make_builtin_type("UCD", |_| {});
        unsafe { typeobject::w_type_set_hasdict(ucd_ty, true) };
        let ucd = w_instance_new(ucd_ty);
        let d = crate::baseobjspace::getdict(ucd);
        for name in ["normalize", "category", "name", "lookup", "decimal", "numeric"] {
            if let Some(f) = crate::runtime_ops::dict_storage_get(ns, name) {
                unsafe { w_dict_setitem_str(d, name, f) };
            }
        }
        unsafe { w_dict_setitem_str(d, "unidata_version", w_str_new("3.2.0")) };
        crate::dict_storage_store(ns, "ucd_3_2_0", ucd);
    },
}
