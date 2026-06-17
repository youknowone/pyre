//! Glue for instantiating and executing JIT-emitted trace modules.
//!
//! An emitted trace module imports the host's linear memory (`env.memory`)
//! plus an optional `env.jit_call` trampoline, and exports a `trace`
//! function. Something on the host side must instantiate that module, wire
//! the shared memory and trampoline, and hand back a callable handle.
//!
//! Two host bindings are provided, selected by feature:
//!   * `web` — instantiate via the browser `WebAssembly` API through
//!     wasm-bindgen (`./jit_glue.js`).
//!   * `host-import` — call plain wasm imports that a native embedder
//!     (wasmi / wasmtime) supplies; no JavaScript runtime involved.
//!
//! Both expose the same `compile_module` / `execute` / `free` surface, so
//! the rest of the backend stays binding-agnostic.

#[cfg(feature = "web")]
mod imports {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(raw_module = "./jit_glue.js")]
    unsafe extern "C" {
        pub(super) fn jit_compile_wasm(bytes_ptr: u32, bytes_len: u32) -> u32;
        pub(super) fn jit_execute_wasm(func_id: u32, frame_ptr: u32) -> u32;
        pub(super) fn jit_free_wasm(func_id: u32);
    }
}

#[cfg(all(feature = "host-import", not(feature = "web")))]
mod imports {
    // Plain wasm imports from the `pyre_jit` module namespace. A native
    // embedder backs these with its own runtime, e.g. `wasmi::Module::new`
    // + `Instance::new` sharing this module's exported linear memory.
    #[link(wasm_import_module = "pyre_jit")]
    unsafe extern "C" {
        pub(super) fn jit_compile_wasm(bytes_ptr: u32, bytes_len: u32) -> u32;
        pub(super) fn jit_execute_wasm(func_id: u32, frame_ptr: u32) -> u32;
        pub(super) fn jit_free_wasm(func_id: u32);
    }
}

#[cfg(not(any(feature = "web", feature = "host-import")))]
mod imports {
    // No host binding selected — compiling on wasm32 without a glue
    // feature. Keep the surface defined so the backend still builds; any
    // actual JIT execution traps.
    const NO_BINDING: &str =
        "wasm backend: no JIT host binding (enable feature \"web\" or \"host-import\")";
    pub(super) unsafe fn jit_compile_wasm(_bytes_ptr: u32, _bytes_len: u32) -> u32 {
        panic!("{NO_BINDING}")
    }
    pub(super) unsafe fn jit_execute_wasm(_func_id: u32, _frame_ptr: u32) -> u32 {
        panic!("{NO_BINDING}")
    }
    pub(super) unsafe fn jit_free_wasm(_func_id: u32) {
        panic!("{NO_BINDING}")
    }
}

/// Compile a wasm module from bytes, returning a function handle ID.
pub fn compile_module(wasm_bytes: &[u8]) -> u32 {
    let ptr = wasm_bytes.as_ptr() as u32;
    let len = wasm_bytes.len() as u32;
    unsafe { imports::jit_compile_wasm(ptr, len) }
}

/// Execute a compiled JIT function with the given frame pointer.
pub fn execute(func_id: u32, frame_ptr: u32) -> u32 {
    unsafe { imports::jit_execute_wasm(func_id, frame_ptr) }
}

/// Free a compiled JIT function.
pub fn free(func_id: u32) {
    unsafe { imports::jit_free_wasm(func_id) }
}
