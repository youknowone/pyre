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
    Module(String),
    Repl,
}

fn usage(binary_name: &str) -> String {
    format!(
        "\
usage: {binary_name} [option] ... [-c cmd | file | -] [arg] ...
Options:
-c cmd : program passed in as string (terminates option list)
-m mod : run library module as a script (terminates option list)
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

/// Drain the parser's remaining raw arguments to become `sys.argv[1:]`.
/// `-c`, `-m`, and a script path each terminate option parsing, so anything
/// after them belongs to the program rather than the launcher.
fn drain_args(parser: &mut lexopt::Parser) -> Result<Vec<String>, lexopt::Error> {
    let mut rest = Vec::new();
    for raw in parser.raw_args()? {
        rest.push(raw.string()?);
    }
    Ok(rest)
}

fn parse_args(binary_name: &str) -> Result<(RunMode, bool, bool, Vec<String>), lexopt::Error> {
    let mut parser = lexopt::Parser::from_env();
    let mut inspect = false;
    let mut quiet = false;

    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') => {
                let cmd = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Command(cmd), inspect, quiet, rest));
            }
            Short('m') => {
                let module = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Module(module), inspect, quiet, rest));
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
                if script == "-" {
                    return Ok((RunMode::Repl, inspect, quiet, vec![]));
                }
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Script(script), inspect, quiet, rest));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok((RunMode::Repl, inspect, quiet, vec![]))
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
    let (mode, inspect, quiet, args) = match parse_args(binary_name) {
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
            let mut argv = vec!["-c".to_string()];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_source(&cmd, Mode::Exec, "<string>");
            if inspect {
                repl::run_repl(true);
            }
        }
        RunMode::Module(module) => {
            // `-m`: sys.path[0] is the cwd (runpy resets argv[0] to the
            // module's resolved origin via `_run_module_as_main`).
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            importing::init_sys_path(&cwd);
            let mut argv = vec![module.clone()];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_module(&module);
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
            // sys.argv[0] is the script path; remaining values go to argv[1:].
            let mut argv = vec![path.clone()];
            argv.extend(args);
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

/// Shared top-level launcher bootstrap for `run_source` and `run_module`:
/// register `__build_class__`, create the process `ExecutionContext`, seed
/// the build-class and `LAST_EXEC_CTX` TLS slots so
/// `space.getexecutioncontext()` (sys.settrace/getframe) resolves from the
/// first user statement, and install SIGINT handling (app_main.py:926).
fn setup_exec_context() -> Rc<PyExecutionContext> {
    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    register_build_class();
    let execution_context = Rc::new(PyExecutionContext::default());
    set_build_class_exec_ctx(Rc::as_ptr(&execution_context));
    set_last_exec_ctx(Rc::as_ptr(&execution_context));
    unsafe {
        let ec_ptr = Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        pyre_interpreter::module::_signal::interp_signal::install_signal_handling(&mut *ec_ptr);
    }
    execution_context
}

/// Run a library module as `__main__` via `runpy._run_module_as_main`,
/// the `-m` entry point. `vm.run_module` analog.
fn run_module(module: &str) {
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);

    // `_run_module_as_main` reads `sys.modules["__main__"].__dict__` and runs
    // the module's code in it, so a `__main__` module backed by a fresh dict
    // must exist before runpy is imported. Reuse the canonical W_DictObject
    // paired with the storage so `__main__.__dict__` and `globals()` share one
    // identity (module.py:77 Module.getdict()).
    let mut namespace = Box::new(execution_context.fresh_dict_storage());
    namespace.fix_ptr();
    pyre_interpreter::dict_storage_store(
        &mut namespace,
        "__name__",
        pyre_object::w_str_new("__main__"),
    );
    let namespace = Box::into_raw(namespace);
    let canonical = pyre_interpreter::baseobjspace::dict_storage_to_dict(namespace);
    let main_module = pyre_object::moduleobject::w_module_new_aliasing_dict(
        "__main__",
        namespace as *mut u8,
        canonical,
    );
    importing::set_sys_module("__main__", main_module);

    let result = (|| -> Result<(), pyre_interpreter::PyError> {
        let runpy = importing::importhook("runpy", canonical, pyre_object::PY_NULL, 0, ec_ptr)?;
        let func = pyre_interpreter::getattr(runpy, pyre_object::w_str_new("_run_module_as_main"))?;
        let res = pyre_interpreter::call_function(func, &[pyre_object::w_str_new(module)]);
        if res.is_null() {
            return Err(
                pyre_interpreter::call::take_call_error().unwrap_or_else(|| {
                    pyre_interpreter::PyError::new(
                        pyre_interpreter::PyErrorKind::RuntimeError,
                        "runpy._run_module_as_main returned NULL without an exception",
                    )
                }),
            );
        }
        Ok(())
    })();

    if let Err(e) = result {
        if e.kind == PyErrorKind::SystemExit {
            maybe_print_jit_stats();
            std::process::exit(system_exit_code(&e));
        }
        maybe_print_jit_stats();
        pyre_interpreter::eprint_exception(&e, true);
        std::process::exit(1);
    }
    maybe_print_jit_stats();
}

fn run_source(source: &str, mode: Mode, filename: &str) {
    let code = match compile_source_with_filename(source, mode, filename) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let execution_context = setup_exec_context();
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
