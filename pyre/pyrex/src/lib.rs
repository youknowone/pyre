//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::path::Path;
use std::rc::Rc;

use lexopt::Arg::*;
use lexopt::ValueExt;

use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
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
    /// `pyre interact <executable> [args…]` — run the trusted sandbox
    /// controller (`pypy/sandbox/pypy_interact.py`) over an untrusted child.
    Interact {
        exe: String,
        args: Vec<String>,
        tmpdir: Option<String>,
        lib_root: Option<String>,
        timeout: Option<f64>,
        allow_net: bool,
        log_file: Option<String>,
        verbose: bool,
    },
}

#[derive(Clone, Copy)]
struct LaunchFlags {
    inspect: bool,
    quiet: bool,
    no_site: bool,
    no_user_site: bool,
    ignore_environment: bool,
    isolated: bool,
    dev_mode: bool,
    utf8_mode: Option<i64>,
    safe_path: bool,
    // `-O` count on the command line; PYTHONOPTIMIZE folds in during finalize.
    optimize: i64,
    dont_write_bytecode: bool,
}

impl Default for LaunchFlags {
    fn default() -> Self {
        Self {
            inspect: false,
            quiet: false,
            no_site: false,
            no_user_site: false,
            ignore_environment: false,
            isolated: false,
            dev_mode: false,
            utf8_mode: None,
            safe_path: false,
            optimize: 0,
            dont_write_bytecode: false,
        }
    }
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
-E     : ignore PYTHON* environment variables
-I     : isolate Python from the user's environment
-B     : don't write .pyc files on import (also PYTHONDONTWRITEBYTECODE=x)
-O     : remove assert and __debug__-dependent statements (also PYTHONOPTIMIZE=x)
-OO    : do -O changes and also discard docstrings
-q     : don't print version on interactive startup
-s     : don't add user site directory to sys.path
-S     : don't imply 'import site' on initialization
-P     : don't prepend a potentially unsafe path to sys.path
-V     : print the Python version number and exit (also --version)
-X opt : set implementation-specific option
file   : program read from script file
-      : program read from stdin (default; interactive mode if a tty)
arg ...: arguments passed to program in sys.argv[1:]

Subcommands:
interact [--tmp DIR] [--lib DIR] [--timeout SECS] [--heapsize N] [--log FILE] [--allow-net] [--verbose] <exe> [arg ...]
       : run the trusted sandbox controller over an untrusted sandbox binary
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

/// Emit the `preconfig_init_utf8_mode` fatal error for an invalid PYTHONUTF8 /
/// `-X utf8` value and exit; the value is validated during pre-init config.
fn fatal_utf8_config_error(detail: &str) -> ! {
    eprintln!("Fatal Python error: preconfig_init_utf8_mode: {detail}");
    eprintln!("Python runtime state: preinitializing");
    eprintln!();
    std::process::exit(1);
}

fn locale_implies_utf8_mode() -> bool {
    // An empty variable is treated as unset (`setlocale` POSIX semantics) and
    // falls through to the next, so `LC_ALL= LC_CTYPE=en_US.UTF-8` resolves to
    // en_US.UTF-8, not C.
    let read = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    let locale = read("LC_ALL")
        .or_else(|| read("LC_CTYPE"))
        .or_else(|| read("LANG"));
    // Only the legacy C/POSIX locale — or a fully unset chain, which resolves to
    // C — coerces utf8_mode to 1; every named locale (en_US, C.UTF-8, …) leaves
    // it 0.
    matches!(locale.as_deref(), None | Some("C") | Some("POSIX"))
}

fn resolve_utf8_mode(flags: &LaunchFlags) -> i64 {
    if let Some(value) = flags.utf8_mode {
        return value;
    }
    if !flags.ignore_environment {
        if let Ok(value) = std::env::var("PYTHONUTF8") {
            if !value.is_empty() {
                return match value.as_str() {
                    "0" => 0,
                    "1" => 1,
                    _ => fatal_utf8_config_error("invalid PYTHONUTF8 environment variable value"),
                };
            }
        }
    }
    if locale_implies_utf8_mode() { 1 } else { 0 }
}

/// `_Py_get_env_flag`: an integer environment flag. An unset or empty value is
/// absent; a clean non-negative integer is used as-is; anything else (trailing
/// junk, negative, overflow) counts as 1.
fn env_int_flag(name: &str) -> Option<u32> {
    let raw = std::env::var(name).ok()?;
    if raw.is_empty() {
        return None;
    }
    let value = match raw.parse::<i64>() {
        Ok(v) if (0..=i64::from(u32::MAX)).contains(&v) => v as u32,
        _ => 1,
    };
    Some(value)
}

/// `-O` count folded with PYTHONOPTIMIZE (`config_init_optimization_level`):
/// the effective level is the larger of the two.  The level is kept as a wide
/// integer so `sys.flags.optimize` mirrors a large `PYTHONOPTIMIZE` verbatim;
/// the compiler clamps it into a byte at read time.
fn resolve_optimize(flags: &LaunchFlags) -> i64 {
    let mut level = flags.optimize;
    if !flags.ignore_environment {
        if let Some(v) = env_int_flag("PYTHONOPTIMIZE") {
            level = level.max(i64::from(v));
        }
    }
    level
}

/// `-B` folded with PYTHONDONTWRITEBYTECODE: either disables bytecode caches.
fn resolve_dont_write_bytecode(flags: &LaunchFlags) -> bool {
    if flags.dont_write_bytecode {
        return true;
    }
    if !flags.ignore_environment {
        if let Some(v) = env_int_flag("PYTHONDONTWRITEBYTECODE") {
            return v != 0;
        }
    }
    false
}

fn finalize_flags(mut flags: LaunchFlags) -> LaunchFlags {
    flags.utf8_mode = Some(resolve_utf8_mode(&flags));
    flags.optimize = resolve_optimize(&flags);
    flags.dont_write_bytecode = resolve_dont_write_bytecode(&flags);
    flags
}

fn parse_args(binary_name: &str) -> Result<(RunMode, LaunchFlags, Vec<String>), lexopt::Error> {
    let mut parser = lexopt::Parser::from_env();
    let mut flags = LaunchFlags::default();

    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') => {
                let cmd = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Command(cmd), finalize_flags(flags), rest));
            }
            Short('m') => {
                let module = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Module(module), finalize_flags(flags), rest));
            }
            Short('h') | Long("help") => {
                print!("{}", usage(binary_name));
                std::process::exit(0);
            }
            Short('i') => flags.inspect = true,
            // app_main.py `cmdline_options['E']` / `X_option`.
            Short('E') => flags.ignore_environment = true,
            // app_main.py `isolated_option`: -I implies -E, -s and -P.
            Short('I') => {
                flags.isolated = true;
                flags.ignore_environment = true;
                flags.no_user_site = true;
                flags.safe_path = true;
            }
            Short('X') => {
                let option = parser.value()?.string()?;
                match option.as_str() {
                    "dev" => flags.dev_mode = true,
                    "utf8" | "utf8=1" => flags.utf8_mode = Some(1),
                    "utf8=0" => flags.utf8_mode = Some(0),
                    _ if option.starts_with("utf8=") => {
                        fatal_utf8_config_error("invalid -X utf8 option value")
                    }
                    _ => {}
                }
            }
            // Accepted for CPython/PyPy command-line compatibility.  Warning
            // filter installation is handled by the warnings module; keeping
            // the option/value in launcher parsing lets stdlib subprocesses
            // reach their command or module.
            Short('W') => {
                let _filter = parser.value()?.string()?;
            }
            // `-O` / `-OO` raise the optimization level; each occurrence counts
            // (app_main.py `optimize`). PYTHONOPTIMIZE folds in during finalize.
            Short('O') => flags.optimize = flags.optimize.saturating_add(1),
            // `-B` disables writing bytecode caches (PYTHONDONTWRITEBYTECODE).
            Short('B') => flags.dont_write_bytecode = true,
            // Unbuffered stdio: pyre's stdout/stderr wrappers already write
            // through to the fd on every call, so the flag has nothing left
            // to disable; accepting it keeps `script_helper`-style spawns
            // (`sys.executable -E -u script`) working.
            Short('u') => {}
            Short('q') => flags.quiet = true,
            Short('s') => flags.no_user_site = true,
            Short('S') => flags.no_site = true,
            Short('P') => flags.safe_path = true,
            Short('V') | Long("version") => {
                println!("{binary_name} 0.0.1");
                std::process::exit(0);
            }
            Value(script) => {
                let script = script.string()?;
                if script == "interact" {
                    let mode = parse_interact(&mut parser)?;
                    return Ok((mode, finalize_flags(flags), vec![]));
                }
                if script == "-" {
                    return Ok((RunMode::Repl, finalize_flags(flags), vec![]));
                }
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Script(script), finalize_flags(flags), rest));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok((RunMode::Repl, finalize_flags(flags), vec![]))
}

/// Parse a `--heapsize` value (`pypy_interact.py:88-102`): a byte count with an
/// optional `k`/`m`/`g` suffix. pyre's GC has no runtime heap-limit knob, so the
/// value is validated and accepted for CLI-surface parity but not enforced
/// (accept-and-ignore); a non-positive or malformed value is a usage error.
fn parse_heapsize(value: &str) -> Result<u64, lexopt::Error> {
    let value = value.trim().to_ascii_lowercase();
    let (digits, mult) = match value.strip_suffix('k') {
        Some(d) => (d, 1024u64),
        None => match value.strip_suffix('m') {
            Some(d) => (d, 1024 * 1024),
            None => match value.strip_suffix('g') {
                Some(d) => (d, 1024 * 1024 * 1024),
                None => (value.as_str(), 1),
            },
        },
    };
    let n: u64 = digits.parse().map_err(|_| {
        lexopt::Error::Custom(format!("interact: invalid --heapsize value: {value}").into())
    })?;
    let bytes = n.checked_mul(mult).ok_or_else(|| {
        lexopt::Error::Custom(format!("interact: --heapsize overflow: {value}").into())
    })?;
    if bytes == 0 {
        return Err(lexopt::Error::Custom(
            "interact: --heapsize must be positive".into(),
        ));
    }
    Ok(bytes)
}

/// Parse the `interact` subcommand: `pyre interact [--tmp DIR] [--lib DIR]
/// [--timeout SECS] [--heapsize N] [--log FILE] [--allow-net] [--verbose]
/// <executable> [program args…]`. The first positional is the untrusted sandbox
/// binary; everything after it is passed through as that program's arguments.
fn parse_interact(parser: &mut lexopt::Parser) -> Result<RunMode, lexopt::Error> {
    let mut tmpdir = None;
    let mut lib_root = None;
    let mut timeout = None;
    let mut allow_net = false;
    let mut log_file = None;
    let mut verbose = false;
    while let Some(arg) = parser.next()? {
        match arg {
            Long("tmp") => tmpdir = Some(parser.value()?.string()?),
            Long("lib") => lib_root = Some(parser.value()?.string()?),
            Long("timeout") => {
                let secs: f64 = parser.value()?.parse()?;
                // `Duration::from_secs_f64` panics on NaN/inf/negative; reject
                // them here as a usage error instead.
                if !secs.is_finite() || secs < 0.0 {
                    return Err(lexopt::Error::Custom(
                        format!(
                            "interact: --timeout must be a non-negative finite number (got {secs})"
                        )
                        .into(),
                    ));
                }
                timeout = Some(secs);
            }
            // pypy_interact.py:88: validated and accepted for CLI parity, but
            // pyre has no runtime heap-limit knob, so the value is discarded.
            Long("heapsize") => {
                parse_heapsize(&parser.value()?.string()?)?;
            }
            // setlogfile (sandlib.py:334): append the guest's stdin to FILE.
            Long("log") => log_file = Some(parser.value()?.string()?),
            // VirtualizedSocketProc (sandlib.py:546): opt in to `tcp://host:port`
            // os.open mediation. Off by default so the sandbox stays
            // network-closed.
            Long("allow-net") => allow_net = true,
            Long("verbose") => verbose = true,
            Value(exe) => {
                let exe = exe.string()?;
                let args = drain_args(parser)?;
                return Ok(RunMode::Interact {
                    exe,
                    args,
                    tmpdir,
                    lib_root,
                    timeout,
                    allow_net,
                    log_file,
                    verbose,
                });
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Err(lexopt::Error::Custom(
        "interact: missing <executable> argument".into(),
    ))
}

/// The working directory to seed `sys.path` with. Under sandbox it is read from
/// the controller (the virtual `/tmp`) through the seam, so an untrusted child
/// never learns the trusted parent's real working directory; off sandbox it is
/// the process cwd.
fn sys_path_cwd() -> std::path::PathBuf {
    #[cfg(feature = "sandbox")]
    {
        use std::os::unix::ffi::OsStrExt;
        if let Ok(bytes) = pyre_interpreter::host_seam::ops::getcwd() {
            return std::path::PathBuf::from(std::ffi::OsStr::from_bytes(&bytes));
        }
        Path::new(".").to_path_buf()
    }
    #[cfg(not(feature = "sandbox"))]
    {
        std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
    }
}

pub fn main_entry(binary_name: &'static str) {
    // The sandboxed child runs single-threaded like pypy-c-sandbox: it neither
    // manages signals (the controller does) nor spawns a worker thread, so it
    // issues no sigprocmask/clone syscalls. Run `real_main` directly.
    //
    // Skipping the dedicated interpreter thread does not lose stack-overflow
    // protection: the recursion limit is a native byte budget (~3.75 MiB at the
    // default 5000), not a frame counter, and `stack_check` raises RecursionError
    // before the budget is exceeded — well under the main thread's default ~8 MiB
    // stack. A child that raises the limit far past that, or shrinks the stack
    // via ulimit, gets a SIGSEGV reaped by the trusted controller, not an escape.
    #[cfg(feature = "sandbox")]
    real_main(binary_name);

    // Block async signals on this (the process's original) thread so the
    // kernel delivers process-directed signals to the interpreter thread
    // spawned below, where they can interrupt blocking syscalls.  The
    // interpreter thread inherits this mask and unblocks them at the top of
    // `real_main`.
    #[cfg(not(feature = "sandbox"))]
    {
        pyre_interpreter::module::signal::signalstate::block_async_signals_on_origin_thread();
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(|| real_main(binary_name))
            .expect("spawn main thread")
            .join()
            .unwrap();
    }
}

fn real_main(binary_name: &str) {
    // Receive process-directed async signals on this thread (see
    // `main_entry`) so blocking syscalls here are interrupted by Ctrl-C /
    // alarms.  The sandboxed child does not touch the signal mask.
    #[cfg(not(feature = "sandbox"))]
    pyre_interpreter::module::signal::signalstate::unblock_async_signals_on_interp_thread();
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
    let (mode, flags, args) = match parse_args(binary_name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{binary_name}: {e}");
            std::process::exit(2);
        }
    };
    let LaunchFlags {
        inspect,
        quiet,
        no_site,
        no_user_site,
        ignore_environment,
        isolated,
        dev_mode,
        utf8_mode,
        safe_path,
        optimize,
        dont_write_bytecode,
    } = flags;

    // The `interact` controller does not run the embedded interpreter; it only
    // spawns and relays for an untrusted child, so it skips the interpreter
    // bootstrap (recursion budget, JIT hooks) entirely.
    let is_interact = matches!(mode, RunMode::Interact { .. });

    // The interactive REPL drives raw stdin/stdout directly, which under sandbox
    // are the controller's marshalling pipes — running it would corrupt the
    // protocol and bypass fd-0 mediation. Reject REPL / `-i` in sandbox builds.
    #[cfg(feature = "sandbox")]
    if !is_interact && (matches!(mode, RunMode::Repl) || inspect) {
        eprintln!("{binary_name}: interactive mode is unavailable in the sandbox");
        std::process::exit(2);
    }

    if !is_interact {
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
    }

    // Record `-S` before the first `import sys` so `sys.flags.no_site`
    // reflects whether site initialization was skipped.
    importing::set_no_site(no_site);
    importing::set_runtime_flags(
        no_user_site,
        ignore_environment,
        isolated,
        dev_mode,
        utf8_mode.unwrap_or(0),
        safe_path,
        optimize,
        dont_write_bytecode,
    );

    // OS-level hardening (default-on in any Linux `sandbox` build): lock the
    // sandboxed child to a host-neutral syscall allowlist so any un-mediated
    // syscall — a missed reroute, the linked host_env crate, or one reached by a
    // memory-safety exploit — is denied by the kernel (the analog of RPython's
    // os_level_sandboxing). The compile-out seam is the primary mechanism; this
    // kernel allowlist is the defense-in-depth backstop, always installed here so
    // it cannot be forgotten. The `interact` controller keeps full syscall access
    // (it spawns and relays for the child), so it is excluded. Set
    // PYRE_SANDBOX_NO_SECCOMP to bypass when debugging the allowlist.
    #[cfg(all(target_os = "linux", feature = "sandbox"))]
    if !is_interact && std::env::var_os("PYRE_SANDBOX_NO_SECCOMP").is_none() {
        // The GC is built lazily on the first allocation, and its Linux
        // construction reads `/proc/meminfo` (`get_total_memory`) — a direct
        // `openat` the filter below traps. Force the per-thread JIT driver, and
        // with it `MiniMarkGC::new`, to run now, while file opens are still
        // allowed; the post-lockdown allocations reuse the constructed GC and
        // issue no `openat`. Same warm-up-before-filter shape as the `gmtime_r`
        // tz-cache prime in `install_runtime_filter`.
        pyre_jit::eval::driver_pair();
        if let Err(e) = pyre_sandbox::seccomp::install_runtime_filter() {
            eprintln!("{binary_name}: failed to install sandbox seccomp filter: {e}");
            std::process::exit(1);
        }
    }

    match mode {
        RunMode::Command(cmd) => {
            // `_PyPathConfig_ComputeSysPath0` yields "" for a `-c` argv[0]. The
            // cwd is still the anchor the shadowing check compares module
            // origins against.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, "");
            let mut argv = vec!["-c".to_string()];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_source(&cmd, Mode::Exec, "<string>", no_site);
            if inspect {
                repl::run_repl(true, no_site);
            }
        }
        RunMode::Module(module) => {
            // `-m`: sys.path[0] is the cwd (runpy resets argv[0] to the
            // module's resolved origin via `_run_module_as_main`).
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, &cwd.to_string_lossy());
            let mut argv = vec![module.clone()];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_module(&module, no_site);
            if inspect {
                repl::run_repl(true, no_site);
            }
        }
        RunMode::Script(path) => {
            // Under sandbox the script is read through the seam so the controller
            // VFS mediates it, the same channel module imports use; off sandbox
            // it is a plain host read.
            #[cfg(feature = "sandbox")]
            let read = importing::read_source_to_string(Path::new(&path));
            #[cfg(not(feature = "sandbox"))]
            let read = std::fs::read_to_string(&path);
            let source = match read {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{binary_name}: cannot open '{path}': {e}");
                    std::process::exit(1);
                }
            };
            // Initialize sys.path with the script's directory. Under sandbox,
            // `canonicalize()` issues raw host-FS syscalls (realpath) past the
            // seccomp lockdown and resolves against the real filesystem, not the
            // controller VFS; use the virtual path as given instead.
            #[cfg(feature = "sandbox")]
            let script_dir = Path::new(&path)
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
            #[cfg(not(feature = "sandbox"))]
            let script_dir = {
                // `parent()` of a bare `x.py` is the empty path;
                // `_PyPathConfig_ComputeSysPath0` absolutizes the script's
                // directory, so resolve the empty case against the cwd rather
                // than leaving a relative `.` on `sys.path`.
                let parent = Path::new(&path).parent().unwrap_or(Path::new("."));
                let parent = if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                };
                parent.canonicalize().unwrap_or_else(|_| sys_path_cwd())
            };
            importing::init_sys_path(&script_dir, &script_dir.to_string_lossy());
            // sys.argv[0] is the script path; remaining values go to argv[1:].
            let mut argv = vec![path.clone()];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            run_source(&source, Mode::Exec, &path, no_site);
            if inspect {
                repl::run_repl(true, no_site);
            }
        }
        RunMode::Repl => {
            // The REPL and piped stdin both take "" for `sys.path[0]`
            // (`_PyPathConfig_ComputeSysPath0` on an argv[0] of "" / "-"); the
            // cwd stays the shadowing-check anchor.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, "");
            repl::run_repl(quiet, no_site);
        }
        RunMode::Interact {
            exe,
            args,
            tmpdir,
            lib_root,
            timeout,
            allow_net,
            log_file,
            verbose,
        } => {
            #[cfg(unix)]
            run_interact(
                binary_name,
                exe,
                args,
                tmpdir,
                lib_root,
                timeout,
                allow_net,
                log_file,
                verbose,
            );
            #[cfg(not(unix))]
            {
                let _ = (
                    exe, args, tmpdir, lib_root, timeout, allow_net, log_file, verbose,
                );
                eprintln!(
                    "{binary_name}: 'interact' (sandbox controller) is only supported on Unix"
                );
                std::process::exit(1);
            }
        }
    }
}

/// `pyre interact` — drive the trusted sandbox controller over the untrusted
/// `exe`, then exit with the child's return code. Port of
/// `pypy/sandbox/pypy_interact.py`'s `main`. Unix-only: the controller relies
/// on a fork/fd-based child, so non-unix hosts reject `interact` at dispatch.
#[cfg(unix)]
fn run_interact(
    binary_name: &str,
    exe: String,
    args: Vec<String>,
    tmpdir: Option<String>,
    lib_root: Option<String>,
    timeout: Option<f64>,
    allow_net: bool,
    log_file: Option<String>,
    _verbose: bool,
) {
    use pyre_sandbox::controller::PyPySandboxedProc;

    let tmpdir = tmpdir.map(std::path::PathBuf::from);
    let lib_root = lib_root.map(std::path::PathBuf::from);
    let timeout = timeout.map(std::time::Duration::from_secs_f64);
    let log_file = log_file.map(std::path::PathBuf::from);

    let mut proc =
        match PyPySandboxedProc::new(&exe, &args, tmpdir, lib_root, timeout, allow_net, log_file) {
            Ok(proc) => proc,
            Err(e) => {
                eprintln!("{binary_name}: cannot start sandbox '{exe}': {e}");
                std::process::exit(1);
            }
        };
    match proc.interact() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{binary_name}: sandbox controller error: {e}");
            std::process::exit(1);
        }
    }
}

/// Print a one-line JIT statistics summary to stderr when `MAJIT_STATS` is
/// set. `internal_compile_panics > 0` means an internal JIT bug silently
/// disabled compilation for some traces (graceful degradation in release);
/// `check.py` asserts it stays 0. Must be called before any `process::exit`
/// since exits skip destructors.
fn maybe_print_jit_stats() {
    // warmspot.py finish helper: `if profiler.initialized:
    // profiler.finish()` — emits the PYPYLOG `jit-summary` section on
    // stderr, visible under `MAJIT_LOG` (the `[jit-stats]` line below
    // stays the `MAJIT_STATS` machine-readable summary).
    if majit_metainterp::majit_log_enabled() {
        let profiler = &pyre_jit::eval::driver_pair()
            .0
            .meta_interp()
            .staticdata
            .profiler;
        if profiler
            .initialized
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            profiler.finish();
        }
    }
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
/// the `LAST_EXEC_CTX` TLS slot so
/// `space.getexecutioncontext()` (sys.settrace/getframe) resolves from the
/// first user statement, and install SIGINT handling (app_main.py:926).
fn setup_exec_context() -> Rc<PyExecutionContext> {
    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    register_build_class();
    let execution_context = Rc::new(PyExecutionContext::default());
    set_last_exec_ctx(Rc::as_ptr(&execution_context));
    unsafe {
        let ec_ptr = Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        (*ec_ptr).install_user_del_action();
        pyre_interpreter::module::signal::interp_signal::install_signal_handling(&mut *ec_ptr);
    }
    execution_context
}

/// app_main.py — unless `-S` (`no_site`) was given, `import site` once
/// `__main__` is registered so the standard `site` initialization runs before
/// user code (sys.path finalization, the `quit`/`exit`/`help` builtins). The
/// import failing is non-fatal (the bare `except`): print "'import site'
/// failed" to stderr and continue.
///
/// `pymain_run_python` then prepends the startup `sys.path[0]`, so that step
/// happens here too — after `site`, whose `removeduppaths()` would otherwise
/// rewrite the `-c` / REPL empty entry into the absolute cwd.
/// `add_main_module` / `set_main_loader` — seed `__main__.__loader__`.
///
/// BuiltinImporter is the initial setting for every `__main__`; a script run by
/// path replaces it with a `SourceFileLoader` bound to that file. `-m` is not
/// covered here: `runpy._run_module_as_main` installs the module's own loader.
/// Must run after the importlib bootstrap, which is what supplies both classes.
pub(crate) fn seed_main_loader(
    w_main_globals: pyre_object::PyObjectRef,
    script_file: Option<&str>,
    ec_ptr: *const pyre_interpreter::PyExecutionContext,
) {
    use pyre_interpreter::baseobjspace::getattr_str;

    let load = |module: &str, attr: &str| -> Option<pyre_object::PyObjectRef> {
        importing::importhook(module, w_main_globals, pyre_object::PY_NULL, 0, ec_ptr).ok()?;
        let w_mod = importing::get_sys_module(module)?;
        getattr_str(w_mod, attr).ok()
    };

    let loader = match script_file {
        Some(path) => load("importlib._bootstrap_external", "SourceFileLoader").and_then(|ty| {
            pyre_interpreter::call::call_function_impl_result(
                ty,
                &[
                    pyre_object::w_str_new("__main__"),
                    pyre_object::w_str_new(path),
                ],
            )
            .ok()
        }),
        None => load("importlib._bootstrap", "BuiltinImporter"),
    };
    let Some(loader) = loader else {
        return;
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(w_main_globals, "__loader__", loader);
    }
}

pub(crate) fn import_site(
    no_site: bool,
    w_main_globals: pyre_object::PyObjectRef,
    ec_ptr: *const pyre_interpreter::PyExecutionContext,
) {
    if !no_site
        && importing::importhook("site", w_main_globals, pyre_object::PY_NULL, 0, ec_ptr).is_err()
    {
        eprintln!("'import site' failed");
    }
    importing::add_sys_path_0();
}

/// Run the `init_importlib` / `init_importlib_external` sequence
/// (pylifecycle.c) so `sys.meta_path` / `sys.path_hooks` are populated and
/// `importlib.util.find_spec` works — which `runpy._get_module_details`
/// (the `-m` entry) requires. pyre's native importer does not consult
/// `sys.meta_path`, so before this `importlib._bootstrap` has neither `sys`
/// nor `_imp` injected and `meta_path` is empty.
///
/// `_bootstrap._setup` (importlib/_bootstrap.py) reads the bootstrap builtins
/// `_thread`/`_warnings`/`_weakref` from `sys.modules`, so import them first to
/// seed `sys.modules`; a name already present skips `_builtin_from_name` and
/// keeps the natively-registered module object authoritative.
pub(crate) fn init_importlib_bootstrap(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const pyre_interpreter::PyExecutionContext,
) -> Result<(), pyre_interpreter::PyError> {
    let import =
        |name: &str| importing::importhook(name, canonical, pyre_object::PY_NULL, 0, ec_ptr);

    for name in ["_thread", "_warnings", "_weakref"] {
        import(name)?;
    }
    import("sys")?;
    import("_imp")?;
    // Importing the bootstrap module fires `install_importlib_bootstrap`
    // (the native load hook) as its body finishes: `_install(sys, _imp)`,
    // `_install_external_importers()` — which imports and links
    // `_frozen_importlib_external` — and the `_frozen_importlib` alias.
    // A cached module skips the hook, so running this again (`-i` reaches
    // the REPL after `run_source`) does not re-append the importers.
    import("importlib._bootstrap")?;
    Ok(())
}

/// PyPy object-space finalization / `threading._shutdown`: run the app-level
/// callbacks before module teardown, then join native non-daemon handles.
fn run_threading_shutdown() {
    if let Some(threading) = importing::get_sys_module("threading")
        && let Ok(shutdown) =
            pyre_interpreter::getattr(threading, pyre_object::w_str_new("_shutdown"))
    {
        let result = pyre_interpreter::call_function(shutdown, &[]);
        if result.is_null()
            && let Some(err) = pyre_interpreter::call::take_call_error()
        {
            pyre_interpreter::eprint_exception(&err, true);
        }
    }
    // PyPy's `threading._shutdown` first waits for non-daemon threads. Those
    // threads may legally start further non-daemon threads while the shutdown
    // drain is in progress. Only reject new thread creation once that app-level
    // drain has returned and module/finalizer teardown is about to begin.
    pyre_interpreter::module::thread::set_finalizing();
}

fn collect_and_run_finalizers(ec_ptr: *const PyExecutionContext) {
    pyre_object::gc_hook::try_gc_collect();
    if !ec_ptr.is_null() {
        unsafe { (&mut *(ec_ptr as *mut PyExecutionContext))._run_finalizers_now() };
    }
}

/// PyPy `ObjSpace.finish()` / module teardown ordering: join non-daemon
/// threads, collect already-unreachable cycles, then release `__main__`
/// globals from newest to oldest while the older globals their `__del__`
/// methods may reference are still present.
fn finalize_runtime(canonical: pyre_object::PyObjectRef, ec_ptr: *const PyExecutionContext) {
    run_threading_shutdown();
    collect_and_run_finalizers(ec_ptr);
    let mut entries = unsafe { pyre_object::w_dict_str_entries(canonical) };
    entries.reverse();
    for (name, _) in entries {
        if name == "__builtins__" {
            continue;
        }
        unsafe {
            pyre_object::w_dict_delitem_str(canonical, &name);
        }
        collect_and_run_finalizers(ec_ptr);
    }
}

/// Keep the object references carried by a pending SystemExit live and
/// relocatable while interpreter finalization performs collections.  PyPy's
/// OperationError is GC-visible RPython state; pyre's Rust `PyError` is not, so
/// its three raw PyObjectRef fields need the same explicit shadow-root
/// treatment as other host-stack temporaries.
fn finalize_system_exit(
    mut error: pyre_interpreter::PyError,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for value in [
        error.exc_object,
        error.w_name_context,
        error.w_obj_context,
    ] {
        pyre_object::gc_roots::pin_root(value);
    }
    finalize_runtime(canonical, ec_ptr);
    error.exc_object = pyre_object::gc_roots::shadow_stack_get(base);
    error.w_name_context = pyre_object::gc_roots::shadow_stack_get(base + 1);
    error.w_obj_context = pyre_object::gc_roots::shadow_stack_get(base + 2);
    let code = system_exit_code(&error);
    drop(roots);
    maybe_print_jit_stats();
    std::process::exit(code);
}

/// Run a library module as `__main__` via `runpy._run_module_as_main`,
/// the `-m` entry point. `vm.run_module` analog.
fn run_module(module: &str, no_site: bool) {
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);

    // `_run_module_as_main` reads `sys.modules["__main__"].__dict__` and runs
    // the module's code in it, so a `__main__` module backed by a fresh dict
    // must exist before runpy is imported. Use the canonical W_DictObject so
    // `__main__.__dict__` and `globals()` share one identity (module.py:77
    // Module.getdict()).
    let w_globals = execution_context.fresh_module_globals();
    let _root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_globals);
    unsafe {
        pyre_object::w_dict_setitem_str(w_globals, "__name__", pyre_object::w_str_new("__main__"))
    };
    let canonical = w_globals;
    let main_module = pyre_object::module::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);

    let result = (|| -> Result<(), pyre_interpreter::PyError> {
        init_importlib_bootstrap(canonical, ec_ptr)?;
        import_site(no_site, canonical, ec_ptr);
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
            finalize_system_exit(e, canonical, ec_ptr);
        }
        maybe_print_jit_stats();
        pyre_interpreter::eprint_exception(&e, true);
        std::process::exit(1);
    }
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
}

fn run_source(source: &str, mode: Mode, filename: &str, no_site: bool) {
    let code = match compile_source_with_filename(source, mode, filename) {
        Ok(code) => code,
        Err(e) => {
            // Render the `File "…", line N` + source + caret + `SyntaxError:`
            // banner for the malformed source, matching an interactive run.
            let err = pyre_interpreter::compile_err_to_syntax_error(e, source);
            pyre_interpreter::eprint_syntax_error(&err);
            std::process::exit(1);
        }
    };

    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);
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
    // Reuse the frame's canonical globals dict so the module's `w_dict`
    // shares one identity with `globals()` /
    // `function.__globals__` (PyPy `module.py:77 Module.getdict()`
    // parity).
    let canonical = frame.get_w_globals();
    let main_module = pyre_object::module::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);

    // Seed the module-identity attributes every `__main__` namespace carries
    // (pythonrun.c seeds these; `runpy` does the `-m` case). Without them a
    // bare `__spec__` / `__package__` / `__doc__` reference falls through to
    // the `builtins` module and returns its values, so `if __spec__ is None`
    // main-detection (multiprocessing spawn, runpy, pytest) inverts.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(
            canonical,
            "__spec__",
            pyre_object::w_none(),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            canonical,
            "__package__",
            pyre_object::w_none(),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str(
            canonical,
            "__doc__",
            pyre_object::w_none(),
        );
    }

    // A script run by path gets `__file__` / `__cached__` in `__main__`
    // (pythonrun.c `_PyRun_SimpleFileObject`); the `-c "<string>"` command
    // path does not. `__file__` is absolutized (os.path.abspath, 3.9+) while
    // `sys.argv[0]` keeps the literal command-line path.
    let script_file = if filename != "<string>" {
        // Resolve a relative path against `sys_path_cwd()` — the seam-provided
        // virtual cwd under sandbox — before normalizing, so `std::path::absolute`
        // never consults (and leaks) the trusted process cwd. Off sandbox this is
        // identical to `absolute(filename)`.
        let path = Path::new(filename);
        let to_absolutize = if path.is_absolute() {
            path.to_path_buf()
        } else {
            sys_path_cwd().join(path)
        };
        let abs_file = match std::path::absolute(&to_absolutize) {
            Ok(p) => p.to_string_lossy().into_owned(),
            Err(_) => filename.to_string(),
        };
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__file__",
            pyre_object::w_str_new(&abs_file),
        );
        let _ = pyre_interpreter::baseobjspace::setattr_str(
            main_module,
            "__cached__",
            pyre_object::w_none(),
        );
        Some(abs_file)
    } else {
        None
    };

    // Import `sys` up front so its creation flushes the native search-path seed
    // into `sys.path` before `site` and user code read it.
    let _ = importing::importhook("sys", canonical, pyre_object::PY_NULL, 0, ec_ptr);

    // pylifecycle.c init_importlib before site: install the importlib
    // bootstrap so `builtins.__import__` routes imports through
    // `sys.meta_path` / `sys.path_hooks` from the first user statement.
    // A failure (no reachable stdlib) is non-fatal — the native importer
    // keeps serving imports, the minimal-importer role.
    if let Err(e) = init_importlib_bootstrap(canonical, ec_ptr) {
        eprintln!("pyre: importlib bootstrap failed: {}", e.message_text());
    }

    seed_main_loader(canonical, script_file.as_deref(), ec_ptr);

    import_site(no_site, canonical, ec_ptr);

    match eval_with_jit(&mut frame) {
        Ok(result) => {
            if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                println!("{}", PyDisplay(result));
            }
        }
        Err(e) => {
            if e.kind == PyErrorKind::SystemExit {
                finalize_system_exit(e, canonical, ec_ptr);
            }
            maybe_print_jit_stats();
            pyre_interpreter::eprint_exception(&e, true);
            std::process::exit(1);
        }
    }
    finalize_runtime(canonical, ec_ptr);
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

#[cfg(test)]
mod tests {
    use super::parse_heapsize;

    #[test]
    fn heapsize_suffixes_and_validation() {
        // pypy_interact.py:88-102: k/m/g multipliers, decimal fallback.
        assert_eq!(parse_heapsize("512").unwrap(), 512);
        assert_eq!(parse_heapsize("64k").unwrap(), 64 * 1024);
        assert_eq!(parse_heapsize("128M").unwrap(), 128 * 1024 * 1024);
        assert_eq!(parse_heapsize("2g").unwrap(), 2 * 1024 * 1024 * 1024);
        // non-positive and malformed values are usage errors.
        assert!(parse_heapsize("0").is_err());
        assert!(parse_heapsize("abc").is_err());
        assert!(parse_heapsize("").is_err());
    }
}
