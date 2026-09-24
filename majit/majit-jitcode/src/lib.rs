//! The part of `rpython/jit/codewriter/` and `rpython/tool/algo/` that the
//! JIT runtime and the proc-macro front end read.
//!
//! `majit-translate` re-exports these modules at their original paths.
//! Keeping them in their own crate lets `majit-metainterp`, the backends and
//! `majit-macros` build without the translator.

pub mod codewriter {
    pub mod insns;
}

pub mod tool {
    pub mod algo {
        pub mod color;
    }
}
