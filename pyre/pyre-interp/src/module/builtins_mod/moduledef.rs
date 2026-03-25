//! builtins module definition.
//!
//! PyPy equivalent: pypy/module/__builtin__/
//!
//! `import builtins` gives access to all builtin names.
//! This module re-uses the default builtins namespace.

use pyre_runtime::PyNamespace;

pub fn init(ns: &mut PyNamespace) {
    // Seed with all default builtins — same as the module-level namespace.
    // PyPy: __builtin__ module exposes the same names.
    pyre_runtime::install_default_builtins(ns);
}
