/// TLR (Toy Language - Register) interpreter — port of rpython/jit/tl/tlr.py.
///
/// Demonstrates majit JIT on a register-based bytecode interpreter.
pub mod bytecode;
pub mod interp;
pub mod jit_interp;

use bytecode::*;
use interp::TlrInterp;
use jit_interp::JitTlrInterp;
use std::time::Instant;

fn main() {
    println!("=== majit TLR interpreter (port of rpython/jit/tl/tlr.py) ===\n");

    // Demo 1: square(100) — the original tlr.py example
    {
        let (prog, a) = square_program(100);
        let mut interp = TlrInterp::new();
        let result = interp.run(&prog, a);
        println!("square(100) = {result}");
        assert_eq!(result, 10_000);
    }

    // Demo 2: sum(0..1_000_000)
    {
        let (prog, a) = sum_program(1_000_000);
        let mut interp = TlrInterp::new();
        let result = interp.run(&prog, a);
        println!("sum(0..1_000_000) = {result}");
        assert_eq!(result, 499_999_500_000);
    }

    // Benchmark: sum(0..10_000_000) — interpreter
    println!("\n--- Benchmark: sum(0..10_000_000) [interpreter] ---");
    let interp_result;
    {
        let (prog, a) = sum_program(10_000_000);
        let mut interp = TlrInterp::new();

        let start = Instant::now();
        interp_result = interp.run(&prog, a);
        let elapsed = start.elapsed();

        println!("result = {interp_result}");
        println!("time   = {elapsed:?}");
        assert_eq!(interp_result, 49_999_995_000_000);
    }

    // Benchmark: sum(0..10_000_000) — JIT
    println!("\n--- Benchmark: sum(0..10_000_000) [JIT] ---");
    {
        let (prog, a) = sum_program(10_000_000);
        let mut jit = JitTlrInterp::new();

        let start = Instant::now();
        let result = jit.run(&prog, a);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, interp_result);
    }

    // Benchmark: square(10_000) — interpreter
    println!("\n--- Benchmark: square(10_000) [interpreter] ---");
    let square_result;
    {
        let (prog, a) = square_program(10_000);
        let mut interp = TlrInterp::new();

        let start = Instant::now();
        square_result = interp.run(&prog, a);
        let elapsed = start.elapsed();

        println!("result = {square_result}");
        println!("time   = {elapsed:?}");
        assert_eq!(square_result, 100_000_000);
    }

    // Benchmark: square(10_000) — JIT
    println!("\n--- Benchmark: square(10_000) [JIT] ---");
    {
        let (prog, a) = square_program(10_000);
        let mut jit = JitTlrInterp::new();

        let start = Instant::now();
        let result = jit.run(&prog, a);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, square_result);
    }
}
