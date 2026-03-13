//! pyre — A Rust meta-tracing JIT Python interpreter.
//!
//! Usage:
//!   pyre <script.py>       Execute a Python script
//!   pyre -c <code>         Execute a Python expression/statement
//!   pyre                   Interactive REPL (Phase 2+)

use std::rc::Rc;

use pyre_bytecode::*;
use pyre_interp::eval::eval_frame;
use pyre_interp::frame::PyFrame;
use pyre_runtime::{PyDisplay, PyExecutionContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (source, mode) = if args.len() >= 3 && args[1] == "-c" {
        (args[2].clone(), Mode::Exec)
    } else if args.len() >= 2 {
        match std::fs::read_to_string(&args[1]) {
            Ok(content) => (content, Mode::Exec),
            Err(e) => {
                eprintln!("pyre: cannot open '{}': {}", args[1], e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Usage: pyre <script.py> | pyre -c <code>");
        std::process::exit(1);
    };

    let code = match compile_source(&source, mode) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let execution_context = Rc::new(PyExecutionContext::default());
    let mut frame = PyFrame::new_with_context(code, execution_context);
    match eval_frame(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
        }
        Err(e) => {
            eprintln!("Traceback (most recent call last):");
            eprintln!("  {e}");
            std::process::exit(1);
        }
    }
}
