//! Benchmarks for the pyre interpreter.
//!
//! Each benchmark compiles and executes a small Python script through the
//! full interpreter + JIT pipeline, measuring end-to-end execution time.
//!
//! The per-iteration body mirrors the `pyrex` launcher's `run_source` shape
//! (`pyrex/src/lib.rs`): compile, build the `__main__` frame over a fresh
//! execution context, alias that frame's globals as the `__main__` module
//! dict, then hand the frame to `eval_with_jit`.  Process-global startup the
//! launcher performs once (`real_main`) is hoisted into [`startup`] so the
//! measured region contains only per-run work.

use std::rc::Rc;
use std::sync::Once;

use criterion::{Criterion, criterion_group, criterion_main};

use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

static STARTUP: Once = Once::new();

/// Process-global startup, performed once for the whole benchmark binary.
///
/// `pyrex::real_main` performs each of these before the first user statement;
/// they are process-global, so repeating them per iteration would measure
/// startup rather than execution.
fn startup() {
    STARTUP.call_once(|| {
        // Benchmark scripts print nothing, but a hook must be installed for
        // any that later do — writing to stdout from a benchmark would make
        // the timings depend on the terminal.
        pyre_interpreter::set_print_hook(|_| {});
        pyre_interpreter::stack_check::set_recursion_limit(5000)
            .expect("startup recursion limit must be applicable");
        init_jit_hooks();
        register_build_class();

        let cwd = std::env::current_dir().expect("benchmark cwd must be readable");
        importing::init_sys_path(&cwd, cwd.as_os_str());
        // Nothing here imports `site`, so perform the post-site `sys.path[0]`
        // insert directly.
        importing::add_sys_path_0();
        importing::set_sys_argv(&[std::ffi::OsString::from("pyre-bench")]);
    });
}

/// Execute a Python source string through the full interpreter + JIT pipeline.
fn run_python(source: &str, filename: &str) {
    let code = compile_source_with_filename(source, Mode::Exec, filename)
        .expect("benchmark script must compile");

    let execution_context = Rc::new(PyExecutionContext::default());
    // `threadlocals.py enter_thread` — the ExecutionContext slot belongs to
    // the OS-thread locals and every launcher installs it before running
    // anything.
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .expect("benchmark frame creation must succeed");

    // Reuse the canonical globals dict as the `__main__` module's dict so
    // `globals()` / `function.__globals__` share one identity (`run_source`
    // parity).
    let canonical = frame.get_w_globals();
    let main_module = pyre_object::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);

    eval_with_jit(&mut frame, None).expect("benchmark script must run without raising");
}

// ---------------------------------------------------------------------------
// Python benchmark scripts (with reduced iteration counts for CI)
// ---------------------------------------------------------------------------

const INT_LOOP: &str = r#"
def main():
    s = 0
    i = 0
    while i < 500000:
        s = s + i
        i = i + 1

main()
"#;

const FIB_LOOP: &str = r#"
def fib(n):
    a = 0
    b = 1
    i = 0
    while i < n:
        t = a + b
        a = b
        b = t
        i = i + 1
    return b

fib(5000)
"#;

const FIB_RECURSIVE: &str = r#"
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fib(20)
"#;

const INLINE_HELPER: &str = r#"
def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def square(x):
    return mul(x, x)

def compute(x):
    return add(square(x), x)

def main():
    s = 0
    i = 0
    while i < 200000:
        s = add(s, compute(i)) % 1000000007
        i = add(i, 1)

main()
"#;

const NESTED_LOOP: &str = r#"
def main():
    s = 0
    i = 0
    while i < 500:
        j = 0
        while j < 500:
            s = s + i * j
            j = j + 1
        i = i + 1

main()
"#;

const FLOAT_LOOP: &str = r#"
def main():
    s = 0.0
    i = 0
    while i < 500000:
        s = s + i * 0.1
        i = i + 1

main()
"#;

// ---------------------------------------------------------------------------
// Benchmark registration
// ---------------------------------------------------------------------------

fn bench_interpreter(c: &mut Criterion) {
    startup();

    for (name, source) in [
        ("int_loop", INT_LOOP),
        ("fib_loop", FIB_LOOP),
        ("fib_recursive", FIB_RECURSIVE),
        ("inline_helper", INLINE_HELPER),
        ("nested_loop", NESTED_LOOP),
        ("float_loop", FLOAT_LOOP),
    ] {
        let filename = format!("{name}.py");
        c.bench_function(name, |b| {
            b.iter(|| run_python(source, &filename));
        });
    }
}

criterion_group!(benches, bench_interpreter);
criterion_main!(benches);
