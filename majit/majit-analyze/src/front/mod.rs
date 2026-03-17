//! Front-end scaffolding for semantic graph construction.

pub mod ast;

pub use ast::{AstGraphOptions, SemanticFunction, SemanticProgram, build_semantic_program};
