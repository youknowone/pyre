//! mmap module — PyPy: pypy/module/mmap/
//!
//! `mmap.mmap(fileno, length, ...)` wraps libc mmap(2) directly.  Per-instance
//! state lives in the instance dict (`_ptr`/`_len`/`_pos`/`_access`); the
//! pointer is invalidated on close/`__exit__` via munmap.

crate::pyre_module_init!(interp_mmap);
