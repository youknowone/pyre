/// TL (Toy Language) interpreter — port of rpython/jit/tl/tl.py.
///
/// Stack-based bytecode interpreter with JIT support.
pub mod bytecode;
pub mod interp;
pub mod jit_interp;

use bytecode::*;
use interp::TlInterp;
use jit_interp::JitTlInterp;
use std::time::Instant;

fn main() {
    println!("=== majit TL interpreter (port of rpython/jit/tl/tl.py) ===\n");

    // Demo 1: sum(0..100)
    {
        let (prog, arg) = sum_program(100);
        let mut interp = TlInterp::new();
        let result = interp.run(&prog, arg);
        println!("sum(0..100) = {result}");
        assert_eq!(result, 4950);
    }

    // Demo 2: factorial(20)
    {
        let (prog, arg) = factorial_program(20);
        let mut interp = TlInterp::new();
        let result = interp.run(&prog, arg);
        println!("factorial(20) = {result}");
        assert_eq!(result, 2_432_902_008_176_640_000);
    }

    // Demo 3: square(100)
    {
        let (prog, arg) = square_program(100);
        let mut interp = TlInterp::new();
        let result = interp.run(&prog, arg);
        println!("square(100) = {result}");
        assert_eq!(result, 10_000);
    }

    // Benchmark: sum(0..10_000_000) — interpreter
    println!("\n--- Benchmark: sum(0..10_000_000) [interpreter] ---");
    let interp_result;
    {
        let (prog, arg) = sum_program(10_000_000);
        let mut interp = TlInterp::new();

        let start = Instant::now();
        interp_result = interp.run(&prog, arg);
        let elapsed = start.elapsed();

        println!("result = {interp_result}");
        println!("time   = {elapsed:?}");
    }

    // Benchmark: sum(0..10_000_000) — JIT
    println!("\n--- Benchmark: sum(0..10_000_000) [JIT] ---");
    {
        let (prog, arg) = sum_program(10_000_000);
        let mut jit = JitTlInterp::new();

        let start = Instant::now();
        let result = jit.run(&prog, arg);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, interp_result);
    }

    // Benchmark: square(10_000) — interpreter
    println!("\n--- Benchmark: square(10_000) [interpreter] ---");
    let sq_result;
    {
        let (prog, arg) = square_program(10_000);
        let mut interp = TlInterp::new();

        let start = Instant::now();
        sq_result = interp.run(&prog, arg);
        let elapsed = start.elapsed();

        println!("result = {sq_result}");
        println!("time   = {elapsed:?}");
    }

    // Benchmark: square(10_000) — JIT
    println!("\n--- Benchmark: square(10_000) [JIT] ---");
    {
        let (prog, arg) = square_program(10_000);
        let mut jit = JitTlInterp::new();

        let start = Instant::now();
        let result = jit.run(&prog, arg);
        let elapsed = start.elapsed();

        println!("result = {result}");
        println!("time   = {elapsed:?}");
        assert_eq!(result, sq_result);
    }
}
