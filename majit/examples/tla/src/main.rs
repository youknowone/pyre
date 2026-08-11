/// Rust port of rpython/jit/tl/tla/tla.py — TLA (Toy Language Advanced).
///
/// Object-oriented stack machine with wrapped values.
/// The JIT specializes for integer operations, eliminating object allocation.
pub mod interp;
pub mod jit_interp;

use interp::WObject;
use std::time::Instant;

const DUP: u8 = 5;
const CONST_INT: u8 = 0;
const SUB: u8 = 6;
const JUMP_IF: u8 = 4;
const POP: u8 = 1;
const RETURN: u8 = 3;

/// Countdown from N to 0. Returns N.
/// Tests loop tracing with object dispatch overhead.
fn countdown_bytecode() -> Vec<u8> {
    vec![
        DUP, // 0
        CONST_INT, 1,   // 1, 2
        SUB, // 3
        DUP, // 4
        JUMP_IF, 1,      // 5, 6 → back to CONST_INT
        POP,    // 7
        RETURN, // 8
    ]
}

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    let bytecode = countdown_bytecode();

    // Correctness check
    {
        let result = interp::run(&bytecode, WObject::Int(10));
        assert_eq!(result.int_value(), 10);
    }

    // Benchmark: interpreter (with object wrapping overhead)
    println!("--- countdown({n}) [interpreter, boxed] ---");
    {
        let start = Instant::now();
        let result = interp::run(&bytecode, WObject::Int(n));
        let elapsed = start.elapsed();
        println!("result = {}", result.int_value());
        println!("time   = {elapsed:?}");
    }

    // Benchmark: JIT (integer-specialized, no boxing)
    println!("\n--- countdown({n}) [JIT, unboxed] ---");
    {
        let mut jit = jit_interp::JitTlaInterp::new();
        let start = Instant::now();
        let result = jit.run(&bytecode, WObject::Int(n));
        let elapsed = start.elapsed();
        println!("result = {}", result.int_value());
        println!("time   = {elapsed:?}");
    }
}
