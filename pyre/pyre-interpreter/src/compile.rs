//! Wrapper around RustPython's compiler to parse and compile Python source.

pub use rustpython_compiler::CompileError;
pub use rustpython_compiler::CompileOpts;
pub use rustpython_compiler::Mode;
pub use rustpython_compiler::ast;
pub use rustpython_compiler::codegen;
pub use rustpython_compiler::compile as rp_compile;
pub use rustpython_compiler::compile_with_syntax_warning_handler as rp_compile_with_syntax_warning_handler;
pub use rustpython_compiler::core::SourceLocation;
pub use rustpython_compiler::parser;
pub use rustpython_compiler_core::bytecode::{
    self, BinaryOperator, CodeFlags, CodeObject, ComparisonOperator, ConstantData, Instruction,
    MakeFunctionFlags, OpArg, OpArgState, SpecialMethod,
};

pub(crate) use crate::pyparser::pyparse::retry_named_escape_parse;
pub use crate::pyparser::pyparse::{
    compile_eval, compile_exec, compile_source, compile_source_named_by_bytes,
    compile_source_with_filename, compile_source_with_opts, decode_file_source_bytes,
    decode_source_bytes, universal_newline,
};
