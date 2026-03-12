/// Rust port of rpython/jit/tl/tlr.py — TLR (Toy Language Register).
///
/// Byte-level encoding identical to the RPython version:
/// bytecode is &[u8], opcodes and args are single bytes.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

// Opcodes — identical values to tlr.py
const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;

/// The SQUARE program from tlr.py — computes a*a by repeated addition.
fn square_bytecode() -> Vec<u8> {
    vec![
        ALLOCATE, 3, MOV_A_R, 0, // i = a
        MOV_A_R, 1, // copy of 'a'
        SET_A, 0, MOV_A_R, 2, // res = 0
        // 10:
        SET_A, 1, NEG_A, ADD_R_TO_A, 0, MOV_A_R, 0, // i--
        MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2, // res += a
        MOV_R_A, 0, JUMP_IF_A, 10, // if i!=0: goto 10
        MOV_R_A, 2, RETURN_A, // return res
    ]
}

fn main() {
    let a: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);

    let bytecode = square_bytecode();

    // Correctness check
    {
        let result = interp::interpret(&bytecode, 5);
        assert_eq!(result, 25, "5*5 should be 25");
    }

    // Benchmark: interpreter
    println!("--- square({a}) [interpreter] ---");
    {
        let start = Instant::now();
        let result = interp::interpret(&bytecode, a);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }

    // Benchmark: JIT
    println!("\n--- square({a}) [JIT] ---");
    {
        let mut jit = jit_interp::JitTlrInterp::new();
        let start = Instant::now();
        let result = jit.run(&bytecode, a);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }
}
