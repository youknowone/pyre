/// Rust port of rpython/jit/tl/tl.py — TL (Toy Language).
///
/// Stack-based interpreter with virtualizable stack.
/// Bytecode format identical to the RPython version.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

// Opcodes — identical values to tlopcode.py
const PUSH: u8 = 2;
const POP: u8 = 3;
const SWAP: u8 = 4;
const PICK: u8 = 6;
const ADD: u8 = 8;
const SUB: u8 = 9;
const BR_COND: u8 = 18;
const RETURN: u8 = 21;
const PUSHARG: u8 = 22;

/// sum(N) = 1 + 2 + ... + N by repeated addition in a loop.
fn sum_bytecode() -> Vec<u8> {
    vec![
        PUSH, 0,      // acc = 0
        PUSHARG,       // counter = N
        // loop (offset 3):
        PICK, 0,       // dup counter
        BR_COND, 2,    // if counter != 0, skip to body (offset 9)
        POP,           // pop counter
        RETURN,
        // body (offset 9):
        SWAP,          // [counter, acc]
        PICK, 1,       // [counter, acc, counter]
        ADD,           // [counter, acc+counter]
        SWAP,          // [acc+counter, counter]
        PUSH, 1,
        SUB,           // [acc, counter-1]
        PUSH, 1,
        BR_COND, 238,  // -18: jump to loop (offset 3)
    ]
}

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    let bytecode = sum_bytecode();

    // Correctness check
    {
        let result = interp::interpret(&bytecode, 10);
        assert_eq!(result, 55, "sum(10) should be 55");
    }

    // Benchmark: interpreter
    println!("--- sum({n}) [interpreter] ---");
    {
        let start = Instant::now();
        let result = interp::interpret(&bytecode, n);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }

    // Benchmark: JIT
    println!("\n--- sum({n}) [JIT] ---");
    {
        let mut jit = jit_interp::JitTlInterp::new();
        let start = Instant::now();
        let result = jit.run(&bytecode, n);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }
}
