/// Rust port of rpython/jit/tl/tinyframe/tinyframe.py — register-based VM.
///
/// Object system (Int, Func, CombinedFunc), frame introspection,
/// integer-specialized JIT trace on JUMP_IF_ABOVE back-edges.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

/// Absolute trip-count gate on the JIT path.
///
/// `count_to` adds 1 to `r1` once per pass and stops when `r1` reaches `r0`,
/// so the returned value names the number of passes exactly: `n` passes answer
/// `n`, and one extra pass answers `n+1`. Agreement with `interp::Frame` alone
/// would not settle this — both run the same program, and a duplicated
/// iteration of the *compiled* loop is invisible to any check that does not
/// assert an absolute count.
///
/// Two lengths of different parity, because a peeled first iteration plus an
/// even/odd body count is exactly the shape an off-by-one hides in.
fn trip_count_gate(code: &interp::Code) {
    for n in [1001i64, 1002] {
        let mut jit = jit_interp::JitTinyFrameInterp::new();
        let got = jit.run(code, &[(0, n)]);
        assert_eq!(
            got, n,
            "count_to({n}) = {got}, so the loop ran {got} passes rather than {n} \
             — an off-by-one trip count is the signature of a terminal arm whose \
             exit the trace dropped"
        );
        println!("[trip-count] count_to({n}) = {got} — exactly {n} passes");
    }
}

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

    trip_count_gate(&code);

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
