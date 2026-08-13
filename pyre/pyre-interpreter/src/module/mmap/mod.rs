//! mmap module — PyPy: pypy/module/mmap/
//!
//! `mmap.mmap(fileno, length, ...)` maps through `host_env::mmap`, so the
//! module works on POSIX and on Windows, where the constructor takes a
//! `tagname` instead of flags/prot.  Per-instance state lives in the instance
//! dict (`_ptr`/`_len`/`_pos`/`_access`); the pointer is invalidated on
//! close/`__exit__`, which unmaps.

crate::pyre_module_init!(interp_mmap);
