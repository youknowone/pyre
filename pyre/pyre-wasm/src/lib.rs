// `web` (wasm-bindgen) and `wasmi` (C-ABI) export conflicting `run_python`
// surfaces; exactly one host binding may be active at a time.
#[cfg(all(feature = "web", feature = "wasmi"))]
compile_error!("features `web` and `wasmi` are mutually exclusive");

// The wasmi C-ABI packs a result pointer and length into the high/low halves
// of a u64, which only round-trips with 32-bit pointers.
#[cfg(all(feature = "wasmi", not(target_arch = "wasm32")))]
compile_error!("feature `wasmi` requires target_arch = \"wasm32\"");

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

use pyre_interpreter::*;

use std::cell::RefCell;
use std::sync::Once;

static PANIC_HOOK: Once = Once::new();

fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let msg = format!("[pyre panic] {info}");
            OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&msg));
        }));
    });
}

thread_local! {
    static OUTPUT_BUF: RefCell<String> = RefCell::new(String::new());
}

fn install_wasm_print_hook() {
    pyre_interpreter::set_print_hook(|s| {
        OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(s));
    });
}

/// Run a Python source string and return the output as a string.
///
/// Host-agnostic core shared by the `web` (wasm-bindgen) and `wasmi`
/// (C-ABI) entry points below.
fn run_python_impl(source: &str) -> String {
    install_panic_hook();
    pyre_interpreter::importing::install_builtin_modules();
    install_wasm_print_hook();
    OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());

    let code = match compile_source(source, Mode::Exec) {
        Ok(code) => code,
        Err(e) => return format!("SyntaxError: {e}"),
    };

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    let mut frame =
        match pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context) {
            Ok(frame) => frame,
            Err(e) => return format!("Error: {e}"),
        };

    // catch_unwind to capture panics from JIT as error messages
    let eval_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pyre_jit::eval::eval_with_jit(&mut frame)
    })) {
        Ok(r) => r,
        Err(_) => {
            let panic_msg = OUTPUT_BUF.with(|buf| buf.borrow().clone());
            return if panic_msg.is_empty() {
                "[pyre] unknown panic".to_string()
            } else {
                panic_msg
            };
        }
    };

    let mut output = OUTPUT_BUF.with(|buf| buf.borrow().clone());

    match eval_result {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str(&format!("{}", PyDisplay(result)));
            }
        }
        Err(e) => {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&format!("Error: {e}"));
        }
    }

    output
}

/// Browser / JS entry point: marshalled by wasm-bindgen.
#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn run_python(source: &str) -> String {
    run_python_impl(source)
}

/// Native-host (wasmi / wasmtime) C-ABI surface.
///
/// wasm-bindgen is unavailable without a JS runtime, so the embedder talks
/// to the module through plain exports over linear memory:
///   1. `pyre_alloc(len)` → reserve `len` bytes, write the UTF-8 source there;
///   2. `pyre_run_python(ptr, len)` → run it, returns a packed `u64`
///      (`hi32` = result pointer, `lo32` = result byte length);
///   3. read the UTF-8 result, then `pyre_dealloc(ptr, len)` both buffers.
#[cfg(feature = "wasmi")]
mod host_abi {
    use super::run_python_impl;
    use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};

    // Buffers crossing the boundary are allocated and freed through the
    // global allocator with a `Layout::array::<u8>(len)` derived purely
    // from `len`, so the host only ever needs to remember the length to
    // free a buffer soundly.

    /// Reserve `len` bytes in linear memory and return a pointer the host
    /// can write into. Pair every call with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_alloc(len: usize) -> *mut u8 {
        if len == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        // Layout::array can only fail on overflow, impossible for a real
        // wasm linear-memory size.
        let layout = Layout::array::<u8>(len).expect("pyre_alloc: size overflow");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr
    }

    /// Release a buffer previously handed out by `pyre_alloc` or returned
    /// by `pyre_run_python`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_dealloc(ptr: *mut u8, len: usize) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let layout = Layout::array::<u8>(len).expect("pyre_dealloc: size overflow");
        unsafe { dealloc(ptr, layout) }
    }

    /// Run the UTF-8 Python source at `ptr[..len]`. Returns a packed
    /// `(result_ptr << 32) | result_len`; the result is a UTF-8 byte buffer
    /// the host must free with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_run_python(ptr: *const u8, len: usize) -> u64 {
        let source = if ptr.is_null() {
            String::new()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
            String::from_utf8_lossy(bytes).into_owned()
        };

        let out = run_python_impl(&source).into_bytes();
        let out_len = out.len();
        let out_ptr = pyre_alloc(out_len);
        if out_len != 0 {
            unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_ptr, out_len) };
        }
        ((out_ptr as u64) << 32) | (out_len as u64)
    }
}
