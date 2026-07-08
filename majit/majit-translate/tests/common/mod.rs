//! Shared helpers for the majit-translate integration tests.

use majit_translate::flowspace::bytecode::ConstantData;
use rustpython_compiler::{Mode, compile as rp_compile};
use rustpython_compiler_core::bytecode::CodeObject;

/// Compile `src` as an exec-mode module and return the first nested code
/// object (the first function body). Panics if compilation fails or the
/// source contains no function body.
pub fn compile_first_code(src: &str) -> CodeObject {
    let module = rp_compile(src, Mode::Exec, "<pyre>".into(), Default::default())
        .expect("compile should succeed");
    module
        .constants
        .iter()
        .find_map(|c| match c {
            ConstantData::Code { code } => Some((**code).clone()),
            _ => None,
        })
        .expect("source should contain at least one function body")
}
