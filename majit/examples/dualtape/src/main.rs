/// Two-tape Brainfuck-like interpreter exercising two `[int; virt]` state
/// arrays through the state-field JIT. See `jit_interp.rs` for the language.
pub mod interp;
pub mod jit_interp;

fn main() {
    // a[0]=10; loop 10x: a[0]-=1, a[1]+=1, b[0]+=1, b[1]+=1.
    let prog = b"++++++++++[->+<*}*{]";

    let interp_result = interp::interpret(prog);
    println!("interp = {interp_result}");

    let mut jit = jit_interp::JitDualInterp::new();
    let jit_result = jit.run(prog);
    println!("jit    = {jit_result}");

    assert_eq!(jit_result, interp_result, "JIT diverged from interpreter");
    println!("OK");
}
