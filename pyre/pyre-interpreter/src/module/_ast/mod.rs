//! _ast module — PyPy: pypy/module/_ast/
//!
//! Exposes the AST node type hierarchy as plain type stubs — enough to
//! satisfy `from _ast import *` in `ast.py`.  Real AST construction is
//! not supported (pyre uses RustPython's compiler).

pub mod interp_ast;
pub use interp_ast::register_module as init;
