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

use bytecode::*;
use interp::CalcInterp;
use std::time::Instant;

fn main() {
    println!("=== majit calc interpreter ===\n");

    // Demo 1: sum(0..1_000_000)
    {
        let prog = sum_program(1_000_000);
        let mut interp = CalcInterp::new();
        let result = interp.run(&prog);
        println!("sum(0..1_000_000) = {result}");
        assert_eq!(result, 499_999_500_000);
    }

    // Demo 2: factorial(20)
    {
        let prog = factorial_program(20);
        let mut interp = CalcInterp::new();
        let result = interp.run(&prog);
        println!("factorial(20) = {result}");
        assert_eq!(result, 2_432_902_008_176_640_000);
    }

    // Benchmark: sum(0..10_000_000)
    println!("\n--- Benchmark: sum(0..10_000_000) ---");
    {
        let prog = sum_program(10_000_000);
        let mut interp = CalcInterp::new();

        let start = Instant::now();
        let result = interp.run(&prog);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, 49_999_995_000_000);
    }
}
