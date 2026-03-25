//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::rc::Rc;

use lexopt::Arg::*;
use lexopt::ValueExt;

use pyre_bytecode::*;
use pyre_interpreter::call;
use pyre_interpreter::frame::PyFrame;
use pyre_interpreter::importing;
use pyre_interpreter::{PyDisplay, PyExecutionContext};
use pyre_jit::eval::eval_with_jit;

enum RunMode {
    Script(String),
    Command(String),
    Repl,
}

const USAGE: &str = "\
usage: pyre [option] ... [-c cmd | file | -] [arg] ...
Options:
-c cmd : program passed in as string (terminates option list)
-h     : print this help message and exit (also --help)
-i     : inspect interactively after running script
-O     : optimize (no-op, reserved for compatibility)
-q     : don't print version on interactive startup
-V     : print the Python version number and exit (also --version)
file   : program read from script file
-      : program read from stdin (default; interactive mode if a tty)
arg ...: arguments passed to program in sys.argv[1:]
";

fn parse_args() -> Result<(RunMode, bool, bool), lexopt::Error> {
    let mut parser = lexopt::Parser::from_env();
    let mut inspect = false;
    let mut quiet = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') => {
                let cmd = parser.value()?.string()?;
                return Ok((RunMode::Command(cmd), inspect, quiet));
            }
            Short('h') | Long("help") => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            Short('i') => inspect = true,
            Short('O') => {} // no-op
            Short('q') => quiet = true,
            Short('V') | Long("version") => {
                println!("pyre 0.0.1");
                std::process::exit(0);
            }
            Value(script) => {
                let script = script.string()?;
                let mode = if script == "-" {
                    RunMode::Repl
                } else {
                    RunMode::Script(script)
                };
                return Ok((mode, inspect, quiet));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok((RunMode::Repl, inspect, quiet))
}

fn main() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn main thread")
        .join()
        .unwrap();
}

fn real_main() {
    let (mode, inspect, quiet) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pyre: {e}");
            std::process::exit(2);
        }
    };

    match mode {
        RunMode::Command(cmd) => {
            // Initialize sys.path with CWD for -c mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            run_source(&cmd, Mode::Exec);
            if inspect {
                run_repl(true);
            }
        }
        RunMode::Script(path) => {
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("pyre: cannot open '{path}': {e}");
                    std::process::exit(1);
                }
            };
            // Initialize sys.path with the script's directory.
            let script_dir = Path::new(&path)
                .parent()
                .unwrap_or(Path::new("."))
                .canonicalize()
                .unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&script_dir);
            run_source(&source, Mode::Exec);
            if inspect {
                run_repl(true);
            }
        }
        RunMode::Repl => {
            // Initialize sys.path with CWD for REPL mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            run_repl(quiet);
        }
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

    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    call::register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    // Set execution context for __build_class__ to use
    call::set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
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

fn run_repl(quiet: bool) {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let execution_context = Rc::new(PyExecutionContext::default());
    let ctx_ptr = Rc::into_raw(Rc::clone(&execution_context));

    // Shared namespace across all REPL statements
    let mut namespace = Box::new(execution_context.fresh_namespace());
    namespace.fix_ptr();
    let namespace = Box::into_raw(namespace);

    if !quiet {
        println!("pyre 0.0.1 (Rust meta-tracing JIT)");
        println!("Type \"exit()\" or Ctrl-D to exit.");
    }

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
                let mut frame = PyFrame::new_with_namespace(code_ptr, ctx_ptr, namespace);
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
