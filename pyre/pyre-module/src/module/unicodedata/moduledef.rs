//! `pypy/module/unicodedata/moduledef.py` — `class Module(MixedModule)`.
//!
//! Bodies live in `interp_ucd`.

use pyre_object::*;

use super::interp_ucd;

pyre_interpreter::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new(&rustpython_unicode::unicode_version()),
    },
    functions: {
        "category"         / 1 = interp_ucd::category,
        "bidirectional"    / 1 = interp_ucd::bidirectional,
        "east_asian_width" / 1 = interp_ucd::east_asian_width,
        "combining"        / 1 = interp_ucd::combining,
        "mirrored"         / 1 = interp_ucd::mirrored,
        "decomposition"    / 1 = interp_ucd::decomposition,
        "digit"            / * = interp_ucd::digit; pyre_interpreter::Signature::new(vec!["chr", "default"], None, None, 0, 2),
        "decimal"          / * = interp_ucd::decimal; pyre_interpreter::Signature::new(vec!["chr", "default"], None, None, 0, 2),
        "numeric"          / * = interp_ucd::numeric; pyre_interpreter::Signature::new(vec!["chr", "default"], None, None, 0, 2),
        "name"             / * = interp_ucd::name; pyre_interpreter::Signature::new(vec!["chr", "default"], None, None, 0, 2),
        "lookup"           / 1 = interp_ucd::lookup,
        "normalize"        / 2 = interp_ucd::normalize,
        "is_normalized"    / 2 = interp_ucd::is_normalized,
    },
    extra_init: |ns| {
        // `unicodedata.ucd_3_2_0` — a `UCD` instance pinned to the Unicode
        // 3.2.0 database (used by `stringprep`).  Version-sensitive queries
        // the typed UCD instance selects `Ucd::new(false)` while
        // lookup/normalize/is_normalized share version-independent
        // implementations with the module callables.
        // Install the TypeDef before allocation so the generated allocator
        // can stamp the canonical Python class in `w_class`.
        let ucd_type = interp_ucd::type_object();
        // `interp_ucd.py UCD.typedef` declares no `__new__`, so the two
        // database instances the module exports are the only ones that exist;
        // reaching generic allocation would hand back a `UCD` with no
        // database at all.
        unsafe { pyre_object::w_type_set_disallow_instantiation(ucd_type) };
        let ucd = interp_ucd::ucd_3_2_0();
        // Installing the module attribute can allocate; keep the freshly
        // allocated instance rooted until the namespace owns it.
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(ucd);
        let ucd = pyre_object::gc_roots::shadow_stack_get(
            pyre_object::gc_roots::shadow_stack_len() - 1,
        );
        pyre_interpreter::module_ns_store(ns, "ucd_3_2_0", ucd);
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `unicodedata.UCD`: an `allocate_stable` type with no inline object
    // payload, so the header `w_class` — which a Python subclass instance points
    // at a managed heap type — is the only edge its marker forwards.
    types.push(ModuleGcType {
        descriptor: <interp_ucd::W_UCD as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}
