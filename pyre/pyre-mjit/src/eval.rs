//! JIT-enabled evaluation entry point.
//!
//! This module wraps pyre-interp's eval_frame with JIT capabilities.
//! Currently delegates to pyre-interp's existing JIT-enabled eval_frame.
//! As JIT code migrates from pyre-interp to pyre-mjit, this becomes
//! the sole entry point for JIT execution.

use pyre_interp::frame::PyFrame;
use pyre_runtime::PyResult;

/// Evaluate a Python frame with JIT compilation.
///
/// This is the main entry point for pyre-mjit. It replaces
/// pyre-interp's eval_frame for JIT-enabled execution.
///
/// Currently delegates to pyre-interp's eval_frame (which still
/// has JIT code). As the migration progresses, JIT logic moves here.
pub fn eval_with_jit(frame: &mut PyFrame) -> PyResult {
    // Phase 1: delegate to existing eval_frame
    // Phase 2: replace with mjit-native tracing using auto-generated code
    pyre_interp::eval::eval_frame(frame)
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
