//! mmap module — PyPy: pypy/module/mmap/
//!
//! `mmap.mmap(fileno, length, ...)` owns its native mapping and the descriptor
//! it duplicated on the corresponding typed object, matching PyPy's
//! `W_MMap`/`rmmap.MMap`.  It maps through `host_env::mmap`, so the module
//! works on POSIX and on Windows, where the constructor takes a `tagname`
//! instead of flags/prot.

pyre_interpreter::pyre_module_init!(interp_mmap);

#[cfg(any(unix, windows))]
pub use interp_mmap::{W_MMap, w_mmap_dealloc};

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // PyPy's W_MMap directly owns rmmap.MMap. The builtin layout has no Python
    // reference; a subclass is `typedef.py` `_getusercls`. The sweep
    // destructor closes the native mapping and the duplicated fd on both tids.
    let pyre_class = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <W_MMap as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_mmap_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: &interp_mmap::W_MMAP_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_mmap_dealloc)),
    });
}
