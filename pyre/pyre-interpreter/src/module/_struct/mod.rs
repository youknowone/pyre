//! _struct module — PyPy: pypy/module/struct/
//!
//! Stub implementing just enough of pack/unpack/calcsize/_clearcache and
//! the `error` type to let `struct.py` load.  Each packer handles the
//! format codes pyre actually uses during import (`<q`, `<d`, etc.).

pub mod interp_struct;
pub mod moduledef;
