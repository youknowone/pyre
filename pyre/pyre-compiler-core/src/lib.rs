#![no_std]
#![recursion_limit = "256"]

extern crate alloc;

pub use rustpython_compiler_core_upstream::{
    LineIndex, Mode, OneIndexed, PositionEncoding, SourceFile, SourceFileBuilder, SourceLocation,
    bytecode, frozen, varint,
};

/// RustPython's pinned marshal implementation with pyre's runtime-container
/// construction hooks.  Everything else in compiler-core remains the exact
/// upstream type and implementation re-exported above.
pub mod marshal;
