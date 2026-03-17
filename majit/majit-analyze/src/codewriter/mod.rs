//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-analyze/src/codewriter/
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── codewriter.py          →     └── mod.rs (this file)
//! ```

pub mod codegen;
pub mod jtransform;

pub use codegen::{generate_jitcode, JitDriverConfig, StackConfig, StorageConfig};
pub use jtransform::{transform_all, transform_pattern, LoweringRecipe, TransformConfig};
