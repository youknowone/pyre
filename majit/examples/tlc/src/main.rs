/// Rust port of rpython/jit/tl/tlc.py — TLC (Toy Language with Cons Cells).
///
/// Stack-based interpreter with boxed values, cons cells, and OO features.
/// JIT traces integer-only loops (e.g., fibonacci with ROLL).
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    // ── Fibonacci benchmark ──
    let mut pool = interp::ConstantPool::new();
    let fibo_src = include_str!("../../../../rpython/jit/tl/fibo.tlc.src");
    let fibo_bc = interp::compile(fibo_src, &mut pool);

    // Correctness check
    {
        let mut check_pool = interp::ConstantPool::new();
        let check_bc = interp::compile(fibo_src, &mut check_pool);
        let result = interp::interp(&check_bc, 0, 7, &check_pool);
        assert_eq!(result, 13, "fibo(7) should be 13");
    }

    println!("--- fibo({n}) [interpreter] ---");
    {
        let start = Instant::now();
        let result = interp::interp(&fibo_bc, 0, n, &pool);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }

    println!("\n--- fibo({n}) [JIT] ---");
    {
        let mut jit = jit_interp::JitTlcInterp::new();
        let start = Instant::now();
        let result = jit.run(&fibo_bc, n, &pool);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }

    // ── Accumulator benchmark (uses SEND — JIT falls back to interp) ──
    println!("\n--- accumulator({n}) [interpreter] ---");
    {
        let mut acc_pool = interp::ConstantPool::new();
        let acc_src = include_str!("../../../../rpython/jit/tl/accumulator.tlc.src");
        let acc_bc = interp::compile(acc_src, &mut acc_pool);
        let start = Instant::now();
        let result = interp::interp(&acc_bc, 0, n, &acc_pool);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }
}
