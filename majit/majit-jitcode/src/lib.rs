//! The part of `rpython/jit/codewriter/` and `rpython/tool/algo/` that the
//! JIT runtime and the proc-macro front end read.
//!
//! `majit-translate` re-exports these modules at their original paths.
//! Keeping them in their own crate lets `majit-metainterp`, the backends and
//! `majit-macros` build without the translator.
//!
//! The jitcode format proper is `codewriter::{jitcode, insns, liveness}` and
//! `artifacts`. The other modules are here because a runtime crate names
//! them:
//!
//! - `codewriter::assembler`, `codewriter::codewriter`: `pyjitpl.py`
//!   `finish_setup` reads the assembler's state (`majit-metainterp`).
//! - `codewriter::call`: the info-handle traits and the symbolic fnaddr
//!   scheme (`majit-metainterp`, `pyre-jit`, `pyre-jit-trace`).
//! - `codewriter::flatten`, `tool::algo::{color, unionfind}`: the runtime
//!   codewriter of `pyre-jit` and the regalloc of `majit-macros`.
//! - `codewriter::jtransform`: the array type ids `pyre-jit-trace` builds
//!   descrs from.
//! - `rclass`: `ImmutableRank` (`majit-metainterp`'s jitcode assembler).
//! - `rffi`: the errno flags `pyre-jit-trace` passes to residual calls.
//! - `parse`: `CallPath`, the key of `artifacts` and of symbolic fnaddrs.

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
