/// Rust port of rpython/jit/tl/tinyframe/tinyframe.py — register-based VM.
///
/// Object system (Int, Func, CombinedFunc), frame introspection,
/// integer-specialized JIT trace on JUMP_IF_ABOVE back-edges.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    // loop.tf: count from 0 to N by adding 1 each iteration
    let code = interp::compile(
        "
    main:
    LOAD 0 => r1
    LOAD 1 => r2
    @add
    ADD r2 r1 => r1
    JUMP_IF_ABOVE r0 r1 @add
    RETURN r1
    ",
    );

    // Correctness check
    {
        let mut frame = interp::Frame::new(&code);
        frame.registers[0] = Some(interp::Object::Int(100));
        let result = frame.interpret(&code);
        assert_eq!(result.as_int(), 100, "count_to(100) should be 100");
    }

    // Benchmark: interpreter
    println!("--- count_to({n}) [interpreter] ---");
    {
        let start = Instant::now();
        let mut frame = interp::Frame::new(&code);
        frame.registers[0] = Some(interp::Object::Int(n));
        let result = frame.interpret(&code);
        let elapsed = start.elapsed();
        println!("result = {}", result.as_int());
        println!("time   = {elapsed:?}");
    }

    // Benchmark: JIT
    println!("\n--- count_to({n}) [JIT] ---");
    {
        let mut jit = jit_interp::JitTinyFrameInterp::new();
        let start = Instant::now();
        let result = jit.run(&code, &[(0, n)]);
        let elapsed = start.elapsed();
        println!("result = {result}");
        println!("time   = {elapsed:?}");
    }
}
