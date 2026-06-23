//! _ast module — PyPy: pypy/module/_ast/
//!
//! Exposes the AST node type hierarchy as plain type stubs — enough to
//! satisfy `from _ast import *` in `ast.py`.  Real AST construction is
//! not supported (pyre uses RustPython's compiler).

crate::pyre_module_init!(moduledef);
