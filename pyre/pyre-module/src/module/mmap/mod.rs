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
    use pyre_interpreter::importing::{ModuleGcAnchor, ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // PyPy's W_MMap directly owns rmmap.MMap.  The typed wrapper carries the
    // subclass mapdict prefix (its mapping/fd payload holds no Python
    // reference) and a sweep destructor for the native mapping and duplicated fd.
    types.push(ModuleGcType {
        anchor: ModuleGcAnchor::AfterScandirIterator,
        descriptor: <W_MMap as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::CustomTrace(
            pyre_interpreter::objspace::std::mapdict::mapdict_storage_custom_trace,
        ),
        destructor: Some(gc_destructor!(w_mmap_dealloc)),
    });
}
