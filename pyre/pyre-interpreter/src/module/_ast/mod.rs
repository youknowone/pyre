//! _ast module — PyPy: pypy/module/_ast/
//!
//! Exposes the AST node hierarchy and converts RustPython/Ruff parser trees
//! into the interpreter-level objects consumed by `ast.py`.

crate::pyre_module_init!(moduledef);

pub mod convert;
