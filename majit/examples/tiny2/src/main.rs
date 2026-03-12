/// Rust port of rpython/jit/tl/tiny2_hotpath.py — word-based language.
///
/// Boxed values (IntBox/StrBox), integer-specialized JIT trace.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

/// Fibonacci(N): returns the N-th fibonacci number.
/// Program: "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
/// Args: [1, 1, N]
fn fibonacci_prog() -> Vec<&'static str> {
    "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
        .split_whitespace()
        .collect()
}

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let prog = fibonacci_prog();

    // Correctness check
    {
        let mut args = vec![
            interp::Box::Int(1),
            interp::Box::Int(1),
            interp::Box::Int(11),
        ];
        let result = interp::interpret(&prog, &mut args);
        assert_eq!(interp::repr_stack(&result), "89", "fib(11) should be 89");
    }

    // Benchmark: interpreter
    println!("--- fibonacci({n}) [interpreter] ---");
    {
        let start = Instant::now();
        let mut args = vec![
            interp::Box::Int(1),
            interp::Box::Int(1),
            interp::Box::Int(n),
        ];
        let result = interp::interpret(&prog, &mut args);
        let elapsed = start.elapsed();
        println!("result = {}", interp::repr_stack(&result));
        println!("time   = {elapsed:?}");
    }

    // Benchmark: JIT
    println!("\n--- fibonacci({n}) [JIT] ---");
    {
        let mut jit = jit_interp::JitTiny2Interp::new();
        let start = Instant::now();
        let mut args = vec![1i64, 1, n];
        let result = jit.run(&prog, &mut args);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }
}
