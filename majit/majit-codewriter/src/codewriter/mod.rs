//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-codewriter/src/codewriter/
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── codewriter.py          →     └── mod.rs (this file)
//! ```

pub mod codegen;

pub use codegen::{
    BinopMapping, CodegenValueKind, IoShim, JitDriverConfig, StorageConfig,
    VirtualizableCodegenConfig, generate_jitcode,
};
