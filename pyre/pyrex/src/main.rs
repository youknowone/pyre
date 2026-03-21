//! pyre — A Rust meta-tracing JIT Python interpreter.
//!
//! Usage:
//!   pyre <script.py>       Execute a Python script
//!   pyre -c <code>         Execute a Python expression/statement
//!   pyre                   Interactive REPL

use std::io::{self, BufRead, Write};
use std::rc::Rc;

use pyre_bytecode::*;
use pyre_interp::frame::PyFrame;
use pyre_jit::eval::eval_with_jit;
use pyre_runtime::{PyDisplay, PyExecutionContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 3 && args[1] == "-c" {
        run_source(&args[2], Mode::Exec);
    } else if args.len() >= 2 {
        match std::fs::read_to_string(&args[1]) {
            Ok(content) => run_source(&content, Mode::Exec),
            Err(e) => {
                eprintln!("pyre: cannot open '{}': {}", args[1], e);
                std::process::exit(1);
            }
        }
    } else {
        run_repl();
    }
}

fn run_source(source: &str, mode: Mode) {
    let code = match compile_source(source, mode) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let execution_context = Rc::new(PyExecutionContext::default());
    let mut frame = PyFrame::new_with_context(code, execution_context);
    match eval_with_jit(&mut frame) {
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

fn run_repl() {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let execution_context = Rc::new(PyExecutionContext::default());
    let ctx_ptr = Rc::into_raw(Rc::clone(&execution_context));

    // Shared namespace across all REPL statements
    let mut namespace = Box::new(execution_context.fresh_namespace());
    namespace.fix_ptr();
    let namespace = Box::into_raw(namespace);

    println!("pyre 0.0.1 (Rust meta-tracing JIT)");
    println!("Type \"exit()\" or Ctrl-D to exit.");

    let mut buffer = String::new();
    let mut continuation = false;

    loop {
        let prompt = if continuation { "... " } else { ">>> " };
        print!("{prompt}");
        if io::stdout().flush().is_err() {
            break;
        }

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }

        if !continuation && line.trim().is_empty() {
            continue;
        }

        buffer.push_str(&line);

        // In continuation mode, only try to compile when the user
        // enters a blank line (like CPython's interactive mode).
        if continuation && !line.trim().is_empty() {
            continue;
        }

        match try_compile_single(&buffer) {
            CompileResult::Complete(code) => {
                let code_ptr = Box::into_raw(Box::new(code));
                let mut frame =
                    PyFrame::new_with_namespace(code_ptr, ctx_ptr, namespace);
                match eval_with_jit(&mut frame) {
                    Ok(result) => {
                        if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                            println!("{}", PyDisplay(result));
                        }
                    }
                    Err(e) => {
                        eprintln!("Traceback (most recent call last):");
                        eprintln!("  {e}");
                    }
                }
                buffer.clear();
                continuation = false;
            }
            CompileResult::Incomplete => {
                continuation = true;
            }
            CompileResult::Error(e) => {
                eprintln!("{e}");
                buffer.clear();
                continuation = false;
            }
        }
    }
}

enum CompileResult {
    Complete(CodeObject),
    Incomplete,
    Error(String),
}

/// Detect errors that indicate the input is incomplete rather than invalid.
fn is_incomplete_error(err: &str) -> bool {
    err.contains("Expected an indented block")
        || err.contains("unexpected EOF while parsing")
        || err.contains("expected an indented block")
        || err.contains("unexpected end of input")
}

/// Mimics CPython's codeop._maybe_compile:
///   compile(source + "\n", "single")     → err1
///   compile(source + "\n\n", "single")   → err2
///   if err1 != err2 → incomplete (adding more input changes the error)
///   if both succeed → use the "\n" version
///   if both fail with same error → real syntax error
fn try_compile_single(source: &str) -> CompileResult {
    let trimmed = source.trim_end();
    if trimmed.is_empty() {
        return CompileResult::Incomplete;
    }

    let with_one = format!("{trimmed}\n");
    let with_two = format!("{trimmed}\n\n");

    match compile_source(&with_one, Mode::Single) {
        Ok(code) => CompileResult::Complete(code),
        Err(e1) => {
            // RustPython's parser doesn't support PyCF_DONT_IMPLY_DEDENT,
            // so detect incomplete input by error message patterns.
            if is_incomplete_error(&e1) {
                return CompileResult::Incomplete;
            }
            match compile_source(&with_two, Mode::Single) {
                Ok(_) => CompileResult::Incomplete,
                Err(e2) if e2 != e1 => CompileResult::Incomplete,
                Err(_) => CompileResult::Error(e1),
            }
        }
    }
}
