//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::path::Path;
use std::rc::Rc;

use lexopt::Arg::*;
use lexopt::ValueExt;

use pyre_interpreter::call::{register_build_class, set_build_class_exec_ctx, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{
    Mode, PyDisplay, PyErrorKind, PyExecutionContext, compile_source_with_filename,
};
use pyre_jit::eval::eval_with_jit;

mod repl;
mod repl_readline;

enum RunMode {
    Script(String),
    Command(String),
    Repl,
}

fn usage(binary_name: &str) -> String {
    format!(
        "\
usage: {binary_name} [option] ... [-c cmd | file | -] [arg] ...
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
"
    )
}

fn parse_args(binary_name: &str) -> Result<(RunMode, bool, bool), lexopt::Error> {
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
                print!("{}", usage(binary_name));
                std::process::exit(0);
            }
            Short('i') => inspect = true,
            Short('O') => {} // no-op
            Short('q') => quiet = true,
            Short('V') | Long("version") => {
                println!("{binary_name} 0.0.1");
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

pub fn main_entry(binary_name: &'static str) {
    // Block async signals on this (the process's original) thread so the
    // kernel delivers process-directed signals to the interpreter thread
    // spawned below, where they can interrupt blocking syscalls.  The
    // interpreter thread inherits this mask and unblocks them at the top of
    // `real_main`.
    pyre_interpreter::module::_signal::signalstate::block_async_signals_on_origin_thread();
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| real_main(binary_name))
        .expect("spawn main thread")
        .join()
        .unwrap();
}

fn real_main(binary_name: &str) {
    // Receive process-directed async signals on this thread (see
    // `main_entry`) so blocking syscalls here are interrupted by Ctrl-C /
    // alarms.
    pyre_interpreter::module::_signal::signalstate::unblock_async_signals_on_interp_thread();
    // Suppress panic messages for the optimizer's silent control-flow panics
    // (InvalidLoop, SpeculativeError) — these are caught by catch_unwind in
    // the JIT compile paths but the default panic hook still prints to stderr,
    // making a graceful trace-abandon look like a crash.
    // RPython: both are silent exceptions, not errors
    // (`unroll.py:119-123 except SpeculativeError: raise InvalidLoop`). The
    // optimizer raises SpeculativeError to decline a speculative heap fold;
    // compile_loop / compile_bridge catch it and abandon the trace (correct
    // blackhole fallback), so its message is a false crash signal. A genuinely
    // uncaught one still aborts with a nonzero exit, so suppressing only the
    // message does not hide a real failure.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        let is_silent_control_flow = payload
            .downcast_ref::<majit_metainterp::optimize::InvalidLoop>()
            .is_some()
            || payload
                .downcast_ref::<majit_metainterp::optimize::SpeculativeError>()
                .is_some();
        if !is_silent_control_flow {
            default_hook(info);
        }
    }));
    let (mode, inspect, quiet) = match parse_args(binary_name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{binary_name}: {e}");
            std::process::exit(2);
        }
    };

    // pypy/interpreter/app_main.py:824-825 parity for the standalone
    // launcher: untranslated hosts need a higher startup recursion limit
    // than translated PyPy's default 1000 because each host-language
    // frame is larger. Pyre is likewise running on Rust frames here, so
    // raise the startup budget before executing user code.
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .expect("startup recursion limit must be applicable");

    // Eagerly install pyre-jit's hooks into pyre-interpreter so that
    // sys.settrace / set_jit_param routing is live from the very first
    // user statement, not only after the first JIT-traced bytecode.
    pyre_jit::eval::init_jit_hooks();

    match mode {
        RunMode::Command(cmd) => {
            // Initialize sys.path with CWD for -c mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            importing::set_sys_argv(&["-c".to_string()]);
            run_source(&cmd, Mode::Exec, "<string>");
            if inspect {
                repl::run_repl(true);
            }
        }
        RunMode::Script(path) => {
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{binary_name}: cannot open '{path}': {e}");
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
            // Collect remaining CLI args for sys.argv.
            let argv = vec![path.clone()];
            // lexopt consumed script name; remaining values go to sys.argv[1:]
            importing::set_sys_argv(&argv);
            run_source(&source, Mode::Exec, &path);
            if inspect {
                repl::run_repl(true);
            }
        }
        RunMode::Repl => {
            // Initialize sys.path with CWD for REPL mode.
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            repl::run_repl(quiet);
        }
    }
}

/// Print a one-line JIT statistics summary to stderr when `MAJIT_STATS` is
/// set. `internal_compile_panics > 0` means an internal JIT bug silently
/// disabled compilation for some traces (graceful degradation in release);
/// `check.py` asserts it stays 0. Must be called before any `process::exit`
/// since exits skip destructors.
fn maybe_print_jit_stats() {
    if std::env::var_os("MAJIT_STATS").is_none() {
        return;
    }
    let stats = pyre_jit::eval::driver_pair().0.get_stats();
    eprintln!(
        "[jit-stats] loops_compiled={} bridges_compiled={} loops_aborted={} \
         guard_failures={} internal_compile_panics={}",
        stats.loops_compiled,
        stats.bridges_compiled,
        stats.loops_aborted,
        stats.guard_failures,
        stats.internal_compile_panics,
    );
}

fn run_source(source: &str, mode: Mode, filename: &str) {
    let code = match compile_source_with_filename(source, mode, filename) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    // Set execution context for __build_class__ to use
    set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    // Eagerly seed the LAST_EXEC_CTX TLS slot so that
    // `space.getexecutioncontext()` (sys.settrace/setprofile/getframe)
    // returns the live ExecutionContext from the very first user
    // statement, not only after the first `eval_frame_plain` entry
    // updates the slot.  Mirrors PyPy's `space.threadlocals` always
    // holding the active EC for the current thread.
    set_last_exec_ctx(Rc::as_ptr(&execution_context));
    // app_main.py:926 — install SIGINT → default_int_handler so Ctrl-C
    // raises KeyboardInterrupt, and register the periodic signal-check
    // action on the execution context.
    unsafe {
        let ec_ptr = Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        pyre_interpreter::module::_signal::interp_signal::install_signal_handling(&mut *ec_ptr);
    }
    let mut frame = match PyFrame::new_with_context(code, execution_context) {
        Ok(frame) => frame,
        Err(e) => {
            pyre_interpreter::eprint_exception(&e, true);
            std::process::exit(1);
        }
    };

    // Register __main__ module in sys.modules — PyPy: app_main sets
    // sys.modules['__main__'] before executing user code so that
    // enum.global_enum and similar introspection works.
    //
    // Reuse the canonical W_DictObject already paired with the
    // frame's globals storage (`PyFrame::new_with_context` →
    // `dict_storage_to_dict` lazy mirror_target registration) so the
    // module's `w_dict` shares one identity with `globals()` /
    // `function.__globals__` (PyPy `module.py:77 Module.getdict()`
    // parity).
    let canonical = frame.get_w_globals_obj();
    let main_module = pyre_object::moduleobject::w_module_new_aliasing_dict(
        "__main__",
        unsafe { pyre_object::w_dict_get_dict_storage_proxy(canonical) },
        canonical,
    );
    importing::set_sys_module("__main__", main_module);

    match eval_with_jit(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
        }
        Err(e) => {
            if e.kind == PyErrorKind::SystemExit {
                maybe_print_jit_stats();
                std::process::exit(system_exit_code(&e));
            }
            maybe_print_jit_stats();
            pyre_interpreter::eprint_exception(&e, true);
            std::process::exit(1);
        }
    }
    maybe_print_jit_stats();
}

/// app_main.py:114-129 `handle_sys_exit` — `exitcode = e.code`; None
/// exits 0; otherwise `int(exitcode)` and a value `int()` rejects is
/// printed to stderr with exit status 1.  `e.code` itself is `args[0]`
/// for a 1-arg raise and the whole args tuple otherwise
/// (interp_exceptions.py:993-998 `W_SystemExit.descr_init`).
fn system_exit_code(e: &pyre_interpreter::PyError) -> i32 {
    let exc = e.exc_object;
    if exc.is_null() {
        // No object-backed SystemExit means no `code` attribute (the
        // class default None), i.e. a success exit.
        return 0;
    }
    let code = match pyre_interpreter::getattr(exc, pyre_object::w_str_new("code")) {
        Ok(c) => c,
        Err(_) => return 1,
    };
    if unsafe { pyre_object::is_none(code) } {
        return 0;
    }
    match pyre_interpreter::builtins::builtin_int(&[code]) {
        Ok(w_int) => unsafe { pyre_object::w_int_get_value(w_int) as i32 },
        Err(_) => {
            // app_main.py:124-125 `print(exitcode, file=sys.stderr)`.
            let text = unsafe { pyre_interpreter::display::py_str(code) }
                .unwrap_or_else(|_| "<unprintable>".to_string());
            eprintln!("{text}");
            1
        }
    }
}
