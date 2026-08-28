//! `_cffi_backend` — PyPy: `pypy/module/_cffi_backend/`.
//!
//! cffi's native half.  PyPy implements it at interpreter level rather than as
//! a C extension, which is what lets its JIT see through a foreign call; pyre
//! ports that same interpreter-level module, so `interp_cffi_backend` holds the
//! module surface and [`parse_c_type`] binds the one piece that stays C — the
//! declaration parser whose opcode stream is the format a compiled cffi
//! extension embeds.

crate::pyre_module_init!(interp_cffi_backend);

pub mod parse_c_type;
