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

// Native-host (`wasmi`) builds target wasm32-unknown-unknown, which has no OS
// entropy. To avoid the wasm-bindgen-based `wasm_js` backend (whose imports a
// non-JS embedder cannot satisfy), getrandom is wired to its `custom` backend
// via `--cfg getrandom_backend="custom"`, which calls this hook. pyre seeds only
// non-cryptographic uses (string hash key, the `random` module) from it, and the
// values never affect check.py's oracle comparison, so a deterministic
// SplitMix64 stream is sufficient.
#[cfg(all(target_arch = "wasm32", feature = "wasmi"))]
mod custom_getrandom {
    use core::sync::atomic::{AtomicU64, Ordering};

    static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

    #[unsafe(no_mangle)]
    unsafe extern "Rust" fn __getrandom_v03_custom(
        dest: *mut u8,
        len: usize,
    ) -> Result<(), getrandom::Error> {
        let mut i = 0;
        while i < len {
            let mut z = STATE
                .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            let n = core::cmp::min(8, len - i);
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dest.add(i), n) };
            i += n;
        }
        Ok(())
    }
}

use pyre_interpreter::*;

use std::cell::RefCell;
use std::sync::Once;

// Residual-call host trampoline for the native-host (`wasmi`) build.
//
// wasm32 `call_indirect` type-checks every call, so the in-module metainterp
// cannot transmute a raw funcptr to a statically-guessed `extern "C" fn` and
// call it — a residual target whose real signature is not the uniform
// `(i64…) -> i64` traps. The compiled trace already round-trips such calls
// through the host (`env.jit_call`); this routes the recording / blackhole
// path through the symmetric `pyre_jit.jit_call_host` import, which reflects
// the callee's wasm signature and coerces each positional argument.
#[cfg(all(target_arch = "wasm32", feature = "wasmi"))]
mod residual_host {
    use core::cell::UnsafeCell;

    // Call-area layout shared with `majit-backend-wasm` codegen and the host
    // runner's `jit_call_trampoline`; offsets are relative to the frame-pointer
    // base passed to the import.
    const CALL_RESULT_OFS: usize = 2000;
    const CALL_FUNC_OFS: usize = 2008;
    const CALL_NARGS_OFS: usize = 2016;
    const CALL_ARGS_OFS: usize = 2024;
    const MAX_ARGS: usize = 16;
    const SCRATCH_LEN: usize = CALL_ARGS_OFS + MAX_ARGS * 8;

    #[link(wasm_import_module = "pyre_jit")]
    unsafe extern "C" {
        fn jit_call_host(frame_ptr: u32);
    }

    // A wasm32 module instance is single-threaded, so a shared scratch buffer
    // needs no synchronization. Residual calls nest synchronously: each level
    // writes its arguments, the host reads them before invoking the callee, and
    // each level reads its result immediately after the host returns — so an
    // inner call that reuses the buffer cannot clobber an outer call's
    // already-consumed arguments or not-yet-written result.
    struct Scratch(UnsafeCell<[u8; SCRATCH_LEN]>);
    unsafe impl Sync for Scratch {}
    static SCRATCH: Scratch = Scratch(UnsafeCell::new([0u8; SCRATCH_LEN]));

    fn residual_host_call(func_ptr: usize, args: &[i64]) -> i64 {
        assert!(
            args.len() <= MAX_ARGS,
            "residual_host_call: arity {} exceeds {MAX_ARGS}",
            args.len()
        );
        let base = SCRATCH.0.get() as *mut u8;
        unsafe {
            (base.add(CALL_FUNC_OFS) as *mut i64).write_unaligned(func_ptr as i64);
            (base.add(CALL_NARGS_OFS) as *mut i64).write_unaligned(args.len() as i64);
            for (i, &a) in args.iter().enumerate() {
                (base.add(CALL_ARGS_OFS + i * 8) as *mut i64).write_unaligned(a);
            }
            jit_call_host(base as u32);
            (base.add(CALL_RESULT_OFS) as *const i64).read_unaligned()
        }
    }

    /// Install the trampoline on the current thread. Idempotent.
    pub fn install() {
        majit_backend::call_stub::set_residual_host_call(Some(residual_host_call));
    }
}

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
    #[cfg(all(target_arch = "wasm32", feature = "wasmi"))]
    residual_host::install();
    pyre_interpreter::importing::install_builtin_modules();
    install_wasm_print_hook();
    OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());

    let code = match compile_source(source, Mode::Exec) {
        Ok(code) => code,
        Err(e) => return format!("SyntaxError: {e}"),
    };

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    // Seed the TLS execution-context slot (pyrex real_main does the same at
    // boot). `getexecutioncontext().gettopframe()` must be live so a residual
    // `bh_call_fn_impl` from a blackhole resume — e.g. a `print(...)` after a
    // JIT-compiled loop — can resolve its parent frame instead of tripping the
    // fail-fast topframe assert.
    pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
    let mut frame =
        match pyre_interpreter::pyframe::PyFrame::new_with_context(code, execution_context) {
            Ok(frame) => frame,
            Err(e) => return format!("Error: {e}"),
        };

    // Register the `__main__` module in sys.modules (pyrex real_main does the
    // same), reusing the canonical globals dict so `__main__.__dict__`,
    // `globals()`, and `function.__globals__` share one identity. Without this,
    // `sys.modules['__main__']` / `import __main__` raise KeyError.
    let canonical = frame.get_w_globals_obj();
    let main_module = pyre_object::moduleobject::w_module_new_aliasing_dict(
        "__main__",
        unsafe { pyre_object::w_dict_get_dict_storage_proxy(canonical) },
        canonical,
    );
    pyre_interpreter::importing::set_sys_module("__main__", main_module);

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
        let result = if ptr.is_null() || len == 0 {
            run_python_impl("")
        } else {
            // Reject a (ptr, len) that escapes linear memory before forming a
            // slice; the embedder supplies these raw, so an out-of-range pair
            // would otherwise be undefined behaviour.
            let mem_bytes = core::arch::wasm32::memory_size(0).saturating_mul(65536);
            match (ptr as usize).checked_add(len) {
                Some(end) if end <= mem_bytes => {
                    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
                    run_python_impl(&String::from_utf8_lossy(bytes))
                }
                _ => "Error: input buffer out of wasm memory bounds".to_string(),
            }
        };

        let out = result.into_bytes();
        let out_len = out.len();
        let out_ptr = pyre_alloc(out_len);
        if out_len != 0 {
            unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_ptr, out_len) };
        }
        ((out_ptr as u64) << 32) | (out_len as u64)
    }
}
