//! Wrapper around RustPython's compiler to parse and compile Python source.

pub use rustpython_compiler::Mode;
pub use rustpython_compiler::compile as rp_compile;
pub use rustpython_compiler_core::bytecode::{
    self, BinaryOperator, CodeFlags, CodeObject, ComparisonOperator, ConstantData, Instruction,
    MakeFunctionFlags, OpArg, OpArgState,
};

/// Compile Python source code to a RustPython CodeObject.
pub fn compile_source(source: &str, mode: Mode) -> Result<CodeObject, String> {
    rp_compile(source, mode, "<pyre>".into(), Default::default())
        .map_err(|e| format!("compile error: {e}"))
}

/// Compile Python source code with a custom filename.
///
/// PyPy equivalent: `parse_source_module(space, pathname, source)` in importing.py
pub fn compile_source_with_filename(
    source: &str,
    mode: Mode,
    filename: &str,
) -> Result<CodeObject, String> {
    rp_compile(source, mode, filename.into(), Default::default())
        .map_err(|e| format!("compile error: {e}"))
}

/// Compile a Python expression.
pub fn compile_eval(source: &str) -> Result<CodeObject, String> {
    compile_source(source, Mode::Eval)
}

/// Compile a Python script (module).
pub fn compile_exec(source: &str) -> Result<CodeObject, String> {
    compile_source(source, Mode::Exec)
}
