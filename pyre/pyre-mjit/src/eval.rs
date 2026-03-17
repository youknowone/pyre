//! JIT-enabled evaluation entry point.
//!
//! This module is the sole entry point for JIT execution.
//! It orchestrates JIT function-entry checks, tracing, and
//! compiled-code execution, delegating pure interpretation
//! to pyre-interp's eval_frame_plain.

use pyre_interp::eval;
use pyre_interp::frame::PyFrame;
use pyre_runtime::PyResult;

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-mjit. It replaces
/// pyre-interp's eval_frame for JIT-enabled execution.
///
/// Flow:
/// 1. Install JIT call bridges (force/bridge callbacks)
/// 2. Try function-entry JIT (compiled code or start tracing)
/// 3. If tracing active → JIT-enabled eval loop
/// 4. Otherwise → plain interpreter loop
pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    // Register this function as the eval callback for recursive calls.
    // Like PyPy's __extend__(PyFrame).dispatch(), this replaces the
    // plain interpreter with JIT-aware evaluation for all function calls.
    pyre_interp::call::register_eval_override(eval_with_jit);
    pyre_interp::call::install_jit_call_bridge();
    frame.fix_array_ptrs();

    // Try running compiled code or start tracing for this function
    if let Some(result) = eval::try_function_entry_jit(frame) {
        return result;
    }

    // If function-entry triggered tracing, use JIT-enabled loop
    let (driver, _) = eval::driver_pair();
    if driver.is_tracing() {
        return eval::eval_loop_jit(frame);
    }

    // No JIT activity → pure interpreter
    eval::eval_frame_plain(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_simple_addition() {
        let source = "x = 1 + 2";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_eval_while_loop() {
        let source = "\
i = 0
s = 0
while i < 100:
    s = s + i
    i = i + 1";
        let code = pyre_bytecode::compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_with_jit(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(pyre_object::intobject::w_int_get_value(s), 4950);
        }
    }
}
