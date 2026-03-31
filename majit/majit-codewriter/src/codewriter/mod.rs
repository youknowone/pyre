//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-codewriter/src/codewriter/
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── codewriter.py          →     └── mod.rs (this file)
//! ```

pub mod codegen;
#[cfg(test)]
pub mod jtransform;

pub use codegen::{
    BinopMapping, CodegenValueKind, IoShim, JitDriverConfig, StorageConfig,
    VirtualizableCodegenConfig, generate_jitcode,
};
#[cfg(test)]
pub use jtransform::transform_all;
#[cfg(test)]
pub use jtransform::{LoweringRecipe, TransformConfig, transform_pattern};
