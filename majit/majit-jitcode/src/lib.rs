//! The part of `rpython/jit/codewriter/` and `rpython/tool/algo/` that the
//! JIT runtime and the proc-macro front end read.
//!
//! `majit-translate` re-exports these modules at their original paths.
//! Keeping them in their own crate lets `majit-metainterp`, the backends and
//! `majit-macros` build without the translator.

pub mod codewriter {
    pub mod assembler;
    pub mod call;
    pub mod codewriter;
    pub mod flatten;
    pub mod insns;
    pub mod jitcode;
    pub mod jtransform;
    pub mod liveness;
}

pub use codewriter::{insns, jitcode, liveness};

pub mod artifacts;
pub mod parse;
pub mod rclass;
pub mod rffi;

pub mod tool {
    pub mod algo {
        pub mod color;
        pub mod unionfind;
    }
}
