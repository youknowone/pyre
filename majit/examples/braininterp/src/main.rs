/// Rust port of rpython/jit/tl/braininterp.py — Brainfuck interpreter.
///
/// Tape-based interpreter with 30000 cells, byte-sized values.
/// Includes both a plain interpreter and a JIT-enabled interpreter.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Simple loop benchmark: add N to cell 0 by incrementing
    // This creates a BF program: set cell0=N, then loop cell0 times adding to cell1
    // Program: put N in cell0, [- > + <] (moves cell0 to cell1)
    let mut prog = String::new();
    for _ in 0..n {
        prog.push('+');
    }
    prog.push_str("[->+<]");

    println!("--- braininterp [interpreter] ---");
    {
        let start = Instant::now();
        let output = interp::interpret(prog.as_bytes());
        let elapsed = start.elapsed();
        println!("output len = {}", output.len());
        println!("time       = {elapsed:?}");
    }

    // Larger benchmark: nested loop
    // +++++++++[>+++++++++<-] sets cell1 = 81
    let big_prog = "+++++++++[>+++++++++<-]";
    println!("\n--- braininterp multiply [interpreter] ---");
    {
        let start = Instant::now();
        let output = interp::interpret(big_prog.as_bytes());
        let elapsed = start.elapsed();
        println!("output len = {}", output.len());
        println!("time       = {elapsed:?}");
    }

    println!("\n--- braininterp multiply [JIT] ---");
    {
        let mut jit = jit_interp::JitBrainInterp::new();
        let start = Instant::now();
        let output = jit.run(big_prog.as_bytes());
        let elapsed = start.elapsed();
        println!("output len = {}", output.len());
        println!("time       = {elapsed:?}");
    }
}
