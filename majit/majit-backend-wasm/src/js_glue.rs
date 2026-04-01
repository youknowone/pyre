/// JavaScript interop for wasm JIT compilation and execution.
///
/// Uses wasm-bindgen to call WebAssembly.Module() + Instance() from
/// within the main wasm module.
use wasm_bindgen::prelude::*;

#[wasm_bindgen(raw_module = "./jit_glue.js")]
unsafe extern "C" {
    fn jit_compile_wasm(bytes_ptr: u32, bytes_len: u32) -> u32;
    fn jit_execute_wasm(func_id: u32, frame_ptr: u32) -> u32;
    fn jit_free_wasm(func_id: u32);
}

/// Compile a wasm module from bytes, returning a function handle ID.
pub fn compile_module(wasm_bytes: &[u8]) -> u32 {
    let ptr = wasm_bytes.as_ptr() as u32;
    let len = wasm_bytes.len() as u32;
    unsafe { jit_compile_wasm(ptr, len) }
}

/// Execute a compiled JIT function with the given frame pointer.
pub fn execute(func_id: u32, frame_ptr: u32) -> u32 {
    unsafe { jit_execute_wasm(func_id, frame_ptr) }
}

/// Free a compiled JIT function.
pub fn free(func_id: u32) {
    unsafe { jit_free_wasm(func_id) }
}
