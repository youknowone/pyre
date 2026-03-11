/// Toy calculator interpreter — the first test target for the majit JIT.
///
/// Supports:
/// - Integer arithmetic (+, -, *, /, %)
/// - Variables (a-z, 26 registers)
/// - Comparison operators (<, <=, ==, !=, >, >=)
/// - While loops via conditional/unconditional jumps
/// - A simple bytecode format
pub mod bytecode;
pub mod interp;
pub mod jit_interp;

use bytecode::*;
use interp::CalcInterp;
use jit_interp::JitCalcInterp;
use std::time::Instant;

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    println!("=== majit calc interpreter (n={n}) ===\n");

    // Benchmark: sum(0..n) — interpreter only
    println!("--- sum(0..{n}) [interpreter] ---");
    let interp_result;
    {
        let prog = sum_program(n);
        let mut interp = CalcInterp::new();

        let start = Instant::now();
        interp_result = interp.run(&prog);
        let elapsed = start.elapsed();

        println!("result = {interp_result}");
        println!("time   = {elapsed:?}");
    }

    // Benchmark: sum(0..n) — JIT
    println!("\n--- sum(0..{n}) [JIT] ---");
    {
        let prog = sum_program(n);
        let mut jit = JitCalcInterp::new();

        let start = Instant::now();
        let result = jit.run(&prog);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, interp_result);
    }

    // JIT Demo: factorial(20)
    println!("\n--- JIT: factorial(20) ---");
    {
        let prog = factorial_program(20);
        let mut jit = JitCalcInterp::new();
        let result = jit.run(&prog);
        println!("factorial(20) = {result}");
        assert_eq!(result, 2_432_902_008_176_640_000);
    }
}
