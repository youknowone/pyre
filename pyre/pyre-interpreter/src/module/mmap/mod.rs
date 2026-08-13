//! mmap module — PyPy: pypy/module/mmap/
//!
//! `mmap.mmap(fileno, length, ...)` owns its native mapping and the descriptor
//! it duplicated on the corresponding typed object, matching PyPy's
//! `W_MMap`/`rmmap.MMap`.  It maps through `host_env::mmap`, so the module
//! works on POSIX and on Windows, where the constructor takes a `tagname`
//! instead of flags/prot.

crate::pyre_module_init!(interp_mmap);

#[cfg(any(unix, windows))]
pub use interp_mmap::{W_MMap, w_mmap_dealloc};
