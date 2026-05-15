//! Benchmarks for the pyre interpreter.
//!
//! Each benchmark compiles and executes a small Python script through the
//! full interpreter + JIT pipeline, measuring end-to-end execution time.

use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::eval_with_jit;

/// Suppress print() output during benchmarks.
fn silence_print() {
    pyre_interpreter::set_print_hook(|_| {});
}

/// Execute a Python source string through the full interpreter + JIT pipeline.
fn run_python(source: &str, filename: &str) {
    let code = compile_source_with_filename(source, Mode::Exec, filename)
        .expect("benchmark script must compile");

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .expect("benchmark frame creation must succeed");

    let _ = eval_with_jit(&mut frame);
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
    silence_print();
    pyre_jit::eval::init_jit_hooks();
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .expect("startup recursion limit must be applicable");

    c.bench_function("int_loop", |b| {
        b.iter(|| run_python(INT_LOOP, "int_loop.py"));
    });

    c.bench_function("fib_loop", |b| {
        b.iter(|| run_python(FIB_LOOP, "fib_loop.py"));
    });

    c.bench_function("fib_recursive", |b| {
        b.iter(|| run_python(FIB_RECURSIVE, "fib_recursive.py"));
    });

    c.bench_function("inline_helper", |b| {
        b.iter(|| run_python(INLINE_HELPER, "inline_helper.py"));
    });

    c.bench_function("nested_loop", |b| {
        b.iter(|| run_python(NESTED_LOOP, "nested_loop.py"));
    });

    c.bench_function("float_loop", |b| {
        b.iter(|| run_python(FLOAT_LOOP, "float_loop.py"));
    });
}

criterion_group!(benches, bench_interpreter);
criterion_main!(benches);
